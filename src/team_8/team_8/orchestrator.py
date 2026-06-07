"""Pipeline orchestrator: task_commands → segmentation → GraspGen → CuRobo → execution.

Pick execution follows yeina's 3-phase approach:
  1. Send approach-and-grasp trajectory (arm moves to grasp contact)
  2. Close gripper
  3. Send lift trajectory (arm lifts with object)

All execution calls are blocking (_send_and_wait); the node is spun with
MultiThreadedExecutor so spin_until_future_complete works inside callbacks.
"""

import json
import threading

try:  # pragma: no cover - runtime dependency
    import rclpy
    from action_msgs.msg import GoalStatus
    from rclpy.action import ActionClient
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.node import Node
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Bool, String
    from control_msgs.action import FollowJointTrajectory
    from geometry_msgs.msg import Pose
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from builtin_interfaces.msg import Duration as RosDuration
except ImportError:  # pragma: no cover - import-only test fallback
    rclpy = None
    ActionClient = None
    GoalStatus = None
    JointState = None
    FollowJointTrajectory = None
    Pose = None
    JointTrajectory = None
    JointTrajectoryPoint = None
    RosDuration = None
    MultiThreadedExecutor = None
    ReentrantCallbackGroup = None
    Bool = None

    class Node:  # type: ignore[override]
        pass

    class String:  # type: ignore[override]
        pass

try:  # pragma: no cover - runtime dependency
    from riro_srvs.srv import StringString
    from riro_srvs.srv import PlanTrajectory
    from riro_srvs.srv import BoolNone
except ImportError:  # pragma: no cover - runtime dependency
    StringString = None
    PlanTrajectory = None
    BoolNone = None

from team_8.pipeline_utils import as_bool as _as_bool
from team_8.pipeline_utils import env_float as _env_float
from team_8.pipeline_utils import pose_from_grasp_row as _pose_from_grasp_row_util


# Robotiq 2F-85 gripper constants (from challenge_constants.py / yeina).
_GRIPPER_JOINT = 'robotiq_85_left_knuckle_joint'
_GRIPPER_OPEN = 0.0
_GRIPPER_CLOSED = 0.8


class PipelineOrchestrator(Node):
    """ROS2 orchestrator for segmentation, GraspGen, and motion execution."""

    TASK_COMMANDS_TOPIC = '/task_commands'
    JOINT_STATES_TOPIC = '/joint_states'

    def __init__(self):
        if rclpy is None:
            raise ImportError('rclpy is required for team_8 runtime.')
        if StringString is None:
            raise ImportError('riro_srvs is required for team_8.')

        super().__init__('team_8')

        self.declare_parameter('segmentation_service_name',
                               '/segmentation/segment_prompt')
        self.declare_parameter('graspgen_service_name', '/graspgen/infer')
        self.declare_parameter('auto_run_on_task_command', True)
        self.declare_parameter('segmentation_service_wait_sec', 10.0)
        self.declare_parameter('graspgen_service_wait_sec', 10.0)
        self.declare_parameter('curobo_service_name', '/curobo/plan_trajectory')
        self.declare_parameter('curobo_service_wait_sec', 30.0)
        self.declare_parameter('curobo_ready_timeout_sec', 180.0)
        self.declare_parameter('enable_motion_execution', False)
        self.declare_parameter('arm_action_name',
                               '/ur5_controller/follow_joint_trajectory')
        self.declare_parameter('gripper_action_name',
                               '/gripper_controller/follow_joint_trajectory')
        self.declare_parameter('place_goal', 'storage_1')
        self.declare_parameter('auto_loop', True)

        self._segmentation_service_name = str(
            self.get_parameter('segmentation_service_name').value)
        self._graspgen_service_name = str(
            self.get_parameter('graspgen_service_name').value)
        self._auto_run_on_task_command = _as_bool(
            self.get_parameter('auto_run_on_task_command').value)
        self._segmentation_service_wait_sec = float(
            self.get_parameter('segmentation_service_wait_sec').value)
        self._graspgen_service_wait_sec = float(
            self.get_parameter('graspgen_service_wait_sec').value)
        self._curobo_service_name = str(
            self.get_parameter('curobo_service_name').value)
        self._curobo_service_wait_sec = float(
            self.get_parameter('curobo_service_wait_sec').value)
        self._curobo_ready_timeout_sec = float(
            self.get_parameter('curobo_ready_timeout_sec').value)
        self._enable_motion_execution = _as_bool(
            self.get_parameter('enable_motion_execution').value)
        self._place_goal = str(self.get_parameter('place_goal').value)
        self._auto_loop = _as_bool(self.get_parameter('auto_loop').value)

        # Reentrant group: pipeline callbacks invoke spin_until_future_complete
        # (via _wait_for_future) which would deadlock under a mutually exclusive group.
        self._reentrant = (ReentrantCallbackGroup()
                           if ReentrantCallbackGroup is not None else None)

        self._task_sub = self.create_subscription(
            String, self.TASK_COMMANDS_TOPIC, self.task_command_callback, 10,
            callback_group=self._reentrant)
        self._joint_sub = self.create_subscription(
            JointState, self.JOINT_STATES_TOPIC, self._cache_joints, 10)

        # Subscribe to /curobo/ready (TRANSIENT_LOCAL) so we catch it even if
        # curobo_service published before we started.
        self._curobo_ready_event = threading.Event()
        _latched_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        ) if QoSDurabilityPolicy is not None else 10
        self.create_subscription(
            Bool, '/curobo/ready', self._on_curobo_ready, _latched_qos)
        self._segmentation_client = self.create_client(
            StringString, self._segmentation_service_name)
        self._graspgen_client = self.create_client(
            StringString, self._graspgen_service_name)

        # Multi-view scan pose list.
        # /save_scan_pose  — append current joint state to the list
        # /clear_scan_poses — reset the list
        # On segmentation failure the orchestrator automatically moves to the
        # next saved pose and retries, handling both Gemini bbox failure and
        # insufficient point cloud coverage.
        if BoolNone is not None:
            self.create_service(BoolNone, '/save_scan_pose', self._save_scan_pose_callback)
            self.create_service(BoolNone, '/clear_scan_poses', self._clear_scan_poses_callback)

        self._pipeline_busy = False
        self._active_task = ''
        self._latest_segmentation = None
        self._latest_graspgen = None
        self._latest_joints = None
        self._scan_poses = []          # list of saved JointState scan positions
        self._scan_attempt_idx = 0     # which scan pose we're currently trying
        self._current_task = ''        # task string carried across retries
        # Minimum number of object points to accept segmentation result.
        self._min_object_points = 500

        self._curobo_client = None
        self._arm_client = None
        self._gripper_client = None
        if self._enable_motion_execution:
            if PlanTrajectory is None or ActionClient is None:
                raise ImportError(
                    'PlanTrajectory service and ROS2 actions are required '
                    'for motion execution.')
            self._curobo_client = self.create_client(
                PlanTrajectory, self._curobo_service_name)
            self._arm_client = ActionClient(
                self,
                FollowJointTrajectory,
                str(self.get_parameter('arm_action_name').value),
            )
            self._gripper_client = ActionClient(
                self,
                FollowJointTrajectory,
                str(self.get_parameter('gripper_action_name').value),
            )
        else:
            self.get_logger().info(
                'Motion execution disabled; stopping after GraspGen result.')

        self.get_logger().info(
            'team_8 ready '
            f'segmentation={self._segmentation_service_name} '
            f'graspgen={self._graspgen_service_name} '
            f'curobo={self._curobo_service_name} '
            f'motion_execution={self._enable_motion_execution}'
        )

    # ── ROS2 subscriptions ────────────────────────────────────────────────────

    def _on_curobo_ready(self, msg) -> None:
        if getattr(msg, 'data', False):
            self.get_logger().info('Received /curobo/ready — TSDF has enough frames.')
            self._curobo_ready_event.set()

    def task_command_callback(self, msg: String) -> None:
        self.get_logger().info(f'Received task command: {msg.data}')
        if self._auto_run_on_task_command:
            self._run_pipeline(msg.data)

    def _cache_joints(self, msg) -> None:
        self._latest_joints = msg

    def _save_scan_pose_callback(self, request, response):
        """Append current joint state to the scan pose list.

        Call from each desired camera viewpoint:
          ros2 service call /save_scan_pose riro_srvs/srv/BoolNone "{data: true}"
        Poses are tried in order on segmentation failure.
        """
        if self._latest_joints is None:
            self.get_logger().warn('save_scan_pose: no /joint_states received yet.')
            return response
        self._scan_poses.append(self._latest_joints)
        positions = [round(float(p), 4) for p in self._latest_joints.position]
        self.get_logger().info(
            f'Scan pose #{len(self._scan_poses)} saved: {positions}')
        return response

    def _clear_scan_poses_callback(self, request, response):
        """Clear all saved scan poses.

          ros2 service call /clear_scan_poses riro_srvs/srv/BoolNone "{data: true}"
        """
        n = len(self._scan_poses)
        self._scan_poses.clear()
        self._scan_attempt_idx = 0
        self.get_logger().info(f'Cleared {n} scan pose(s).')
        return response

    # ── pipeline entry ────────────────────────────────────────────────────────

    def _run_pipeline(self, task: str) -> None:
        task = task.strip()
        if not task:
            self.get_logger().warn('Ignoring empty task command.')
            return
        if self._pipeline_busy:
            self.get_logger().warn(
                f"Pipeline busy with '{self._active_task}'. "
                f"Ignoring new task '{task}'.")
            return
        if not self._segmentation_client.wait_for_service(
                timeout_sec=self._segmentation_service_wait_sec):
            self.get_logger().warn(
                f'Segmentation service unavailable: {self._segmentation_service_name} '
                f'(waited {self._segmentation_service_wait_sec:.1f}s)')
            return

        self._pipeline_busy = True
        self._active_task = task
        self._current_task = task
        self._scan_attempt_idx = 0  # reset retry counter for new task

        # Wait for TSDF to accumulate enough frames before doing anything.
        if self._enable_motion_execution:
            if not self._wait_for_curobo_ready():
                self.get_logger().error(
                    'CuRobo TSDF not ready within '
                    f'{self._curobo_ready_timeout_sec:.0f}s — aborting pipeline.')
                self._reset_pipeline_state()
                return

        # Ensure gripper is open before starting a new pick.
        if self._gripper_client is not None:
            self._send_gripper(closed=False)

        # Move to home pose so the wrist camera faces straight down for capture.
        if self._enable_motion_execution:
            self._home_before_capture()

        self._run_segmentation_attempt(task)

    def _run_segmentation_attempt(self, task: str) -> None:
        """Move to the current scan pose (if any) and call the segmentation service."""
        # Move to the scan pose for this attempt index.
        if self._scan_poses and self._arm_client is not None:
            idx = self._scan_attempt_idx % len(self._scan_poses)
            scan_traj = _joint_state_to_trajectory(self._scan_poses[idx])
            try:
                self._send_and_wait(self._arm_client, scan_traj, 'move_to_scan_pose')
                self.get_logger().info(
                    f'Moved to scan pose #{idx + 1}/{len(self._scan_poses)}.')
            except Exception as exc:
                self.get_logger().warn(f'move_to_scan_pose failed (non-fatal): {exc}')

        if not self._segmentation_client.wait_for_service(
                timeout_sec=self._segmentation_service_wait_sec):
            self.get_logger().warn('Segmentation service unavailable.')
            self._reset_pipeline_state()
            return

        request = StringString.Request()
        request.data = task
        future = self._segmentation_client.call_async(request)
        future.add_done_callback(self._on_segmentation_done)
        self.get_logger().info(
            f'Started segmentation for task: {task} '
            f'(attempt {self._scan_attempt_idx + 1})')

    # ── pipeline callbacks ────────────────────────────────────────────────────

    def _on_segmentation_done(self, future) -> None:
        try:
            result = future.result()
            payload = json.loads(result.data)
        except Exception as exc:
            self.get_logger().error(f'Segmentation service call failed: {exc}')
            self._reset_pipeline_state()
            return

        # Check both failure modes: Gemini bbox failure OR insufficient point cloud.
        point_count = int(payload.get('object_point_count', 0))
        seg_failed = not payload.get('success')
        too_few_points = payload.get('success') and point_count < self._min_object_points

        if seg_failed or too_few_points:
            reason = (
                f'Gemini/SAM2 failed: {payload.get("error", "unknown")}'
                if seg_failed
                else f'point cloud too sparse ({point_count} < {self._min_object_points})'
            )
            self.get_logger().warn(f'Segmentation attempt {self._scan_attempt_idx + 1} failed: {reason}')
            self._scan_attempt_idx += 1

            # Retry from the next scan pose if available.
            max_attempts = max(len(self._scan_poses), 1)
            if self._scan_attempt_idx < max_attempts:
                self.get_logger().info(
                    f'Retrying segmentation from scan pose '
                    f'#{self._scan_attempt_idx + 1}/{max_attempts}...')
                self._run_segmentation_attempt(self._current_task)
            else:
                self.get_logger().warn(
                    f'Segmentation failed after {self._scan_attempt_idx} attempt(s). Giving up.')
                self._reset_pipeline_state()
            return

        self._latest_segmentation = payload
        self.get_logger().info(
            f"Segmentation ready label={payload.get('label', 'target')} "
            f"points={point_count}; requesting GraspGen.")

        if not self._graspgen_client.wait_for_service(
                timeout_sec=self._graspgen_service_wait_sec):
            self.get_logger().warn(
                f'GraspGen service unavailable: {self._graspgen_service_name} '
                f'(waited {self._graspgen_service_wait_sec:.1f}s)')
            self._reset_pipeline_state()
            return

        request = StringString.Request()
        # Token: the segmented cloud's stamp, so GraspGen infers on this run's
        # cloud (empty string falls back to "use latest").
        request.data = str(self._latest_segmentation.get('cloud_stamp_ns', ''))
        future = self._graspgen_client.call_async(request)
        future.add_done_callback(self._on_graspgen_done)

    def _on_graspgen_done(self, future) -> None:
        try:
            result = future.result()
            payload = json.loads(result.data)
        except Exception as exc:
            self.get_logger().error(f'GraspGen service call failed: {exc}')
            self._reset_pipeline_state()
            return

        if not payload.get('success'):
            self.get_logger().warn(
                f"GraspGen failed: {payload.get('error', payload)}")
            self._reset_pipeline_state()
            return

        self._latest_graspgen = payload
        top_grasps = payload.get('top_grasps') or []
        if not top_grasps:
            self.get_logger().warn(
                'GraspGen returned success but no ranked grasps.')
            self._reset_pipeline_state()
            return

        self.get_logger().info(
            'Pipeline result '
            f"label={self._latest_segmentation.get('label', 'target')} "
            f"best_translation={top_grasps[0].get('translation')} "
            f"confidence={top_grasps[0].get('confidence')} "
            f"num_candidates={len(top_grasps)}"
        )

        if not self._enable_motion_execution:
            self._reset_pipeline_state()
            return

        # Build geometry_msgs/Pose for every ranked grasp candidate.
        grasp_poses = [
            _pose_from_grasp_row_util(row)
            for row in top_grasps
        ]
        grasp_poses = [p for p in grasp_poses if p is not None]
        if not grasp_poses:
            self.get_logger().warn(
                'Could not build any valid Pose from GraspGen rows; '
                'skipping motion execution.')
            self._reset_pipeline_state()
            return

        self._plan_and_execute_pick(grasp_poses)

    def _plan_and_execute_pick(self, grasp_poses: list) -> None:
        """Call CuRobo service with all candidates, then execute pick phases."""
        if self._latest_joints is None:
            self.get_logger().warn(
                'No /joint_states received; skipping motion execution.')
            self._reset_pipeline_state()
            return
        if self._curobo_client is None:
            self.get_logger().warn(
                'Motion execution enabled but CuRobo service client unavailable.')
            self._reset_pipeline_state()
            return

        self.get_logger().info(
            f'Waiting for CuRobo service {self._curobo_service_name} '
            f'(up to {self._curobo_service_wait_sec:.1f}s, '
            f'{len(grasp_poses)} candidates).')
        if not self._curobo_client.wait_for_service(
                timeout_sec=self._curobo_service_wait_sec):
            self.get_logger().warn(
                f'CuRobo service unavailable: {self._curobo_service_name}')
            self._reset_pipeline_state()
            return

        request = PlanTrajectory.Request()
        request.grasp_poses = grasp_poses
        request.joint_state = self._latest_joints
        future = self._curobo_client.call_async(request)
        future.add_done_callback(self._on_curobo_pick_done)
        self.get_logger().info(
            f'Requested CuRobo pick plan ({len(grasp_poses)} candidates).')

    def _on_curobo_pick_done(self, future) -> None:
        try:
            result = future.result()
        except Exception as exc:
            self.get_logger().error(f'CuRobo service call failed: {exc}')
            self._reset_pipeline_state()
            return

        if not result.success:
            self.get_logger().warn(f'CuRobo pick planning failed: {result.message}')
            self._reset_pipeline_state()
            return

        self.get_logger().info(result.message)

        # 4-phase pick execution: open gripper → approach+grasp → close → lift.
        # Opening first is essential: the planned grasp pose assumes open
        # fingers, so a gripper left closed from a prior cycle would collide
        # with the object during the approach instead of enclosing it.
        try:
            self._send_gripper(closed=False)
            self._send_and_wait(
                self._arm_client, result.trajectory, 'approach_and_grasp')
            self._send_gripper(closed=True)
            self._send_and_wait(
                self._arm_client, result.lift_trajectory, 'lift')
        except Exception as exc:
            self.get_logger().error(f'Pick execution failed: {exc}')
            try:
                self._send_gripper(closed=False)
            except Exception:
                pass
            self._reset_pipeline_state()
            return

        # Place: transit + release + return home.
        try:
            self._plan_and_execute_place(self._place_goal)
        except Exception as exc:
            self.get_logger().error(f'Place execution failed: {exc}')
            try:
                self._send_gripper(closed=False)
            except Exception:
                pass

        try:
            self._plan_and_execute_home()
        except Exception as exc:
            self.get_logger().warn(f'Return-home after place failed (non-fatal): {exc}')

        # Auto-loop: restart pipeline for the same task.
        if self._auto_loop:
            self.get_logger().info(
                f"Auto-loop enabled — restarting pipeline for '{self._current_task}'.")
            self._pipeline_busy = False
            self._active_task = ''
            self._run_pipeline(self._current_task)
        else:
            self._reset_pipeline_state()

    # ── execution helpers ─────────────────────────────────────────────────────

    def _send_and_wait(
        self,
        client: 'ActionClient',
        trajectory: 'JointTrajectory',
        label: str,
        server_timeout_sec: float = 5.0,
    ) -> None:
        """Send a JointTrajectory action goal and block until it completes."""
        if trajectory is None or not trajectory.points:
            self.get_logger().warn(
                f'_send_and_wait({label}): refusing empty trajectory.')
            return
        if client is None:
            self.get_logger().warn(
                f'_send_and_wait({label}): action client unavailable.')
            return
        if not client.wait_for_server(timeout_sec=server_timeout_sec):
            raise RuntimeError(
                f'{label} action server unavailable '
                f'(waited {server_timeout_sec}s)')
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory
        timeout_sec = _action_timeout_sec(trajectory)
        self.get_logger().info(
            f'Sending {label} trajectory ({_trajectory_summary(trajectory)}); '
            f'timeout={timeout_sec:.1f}s.')
        send_future = client.send_goal_async(goal)
        handle = self._wait_for_future(
            send_future, label, 'goal response', timeout_sec)
        if handle is None or not handle.accepted:
            raise RuntimeError(
                f'{label} goal rejected by action server '
                f'({_trajectory_summary(trajectory)})')
        result_future = handle.get_result_async()
        result_response = self._wait_for_future(
            result_future, label, 'result', timeout_sec)
        if result_response is None:
            raise RuntimeError(f'{label} action returned no result')
        if getattr(result_response, 'status', None) != _goal_status_succeeded():
            if _is_nonfatal_gripper_cancel(label, result_response.status, trajectory):
                self.get_logger().warn(
                    'Treating canceled gripper close action as non-fatal; '
                    'GazeboGraspFix can cancel after attaching the object.')
                self.get_logger().info(f'{label} trajectory executed.')
                return
            raise RuntimeError(
                f'{label} action failed with status {result_response.status}')
        action_result = getattr(result_response, 'result', None)
        if action_result is not None:
            success_code = getattr(action_result, 'SUCCESSFUL', 0)
            error_code = getattr(action_result, 'error_code', success_code)
            if error_code != success_code:
                detail = getattr(action_result, 'error_string', '')
                raise RuntimeError(
                    f'{label} trajectory failed with error_code '
                    f'{error_code}: {detail}')
        self.get_logger().info(f'{label} trajectory executed.')

    def _wait_for_future(self, future, label: str, phase: str, timeout_sec: float):
        if not hasattr(future, 'add_done_callback'):
            return future.result()
        done = threading.Event()
        future.add_done_callback(lambda _: done.set())
        if hasattr(future, 'done') and future.done():
            done.set()
        if not done.wait(max(timeout_sec, 0.1)):
            raise RuntimeError(f'Timed out waiting for {label} action {phase}')
        return future.result()

    def _send_gripper(self, closed: bool) -> None:
        """Send open / close command to the gripper controller and wait."""
        if self._gripper_client is None:
            self.get_logger().warn(
                '_send_gripper: gripper action client unavailable; skipping.')
            return
        jt = JointTrajectory()
        jt.joint_names = [_GRIPPER_JOINT]
        pt = JointTrajectoryPoint()
        pt.positions = [_GRIPPER_CLOSED if closed else _GRIPPER_OPEN]
        pt.time_from_start = RosDuration(sec=1, nanosec=0)
        jt.points.append(pt)
        label = 'gripper_close' if closed else 'gripper_open'
        try:
            self._send_and_wait(self._gripper_client, jt, label)
        except Exception as exc:
            self.get_logger().warn(f'{label} failed (non-fatal): {exc}')

    # ── curobo ready + home/place helpers ────────────────────────────────────

    def _wait_for_curobo_ready(self, timeout_sec: float | None = None) -> bool:
        """Block until /curobo/ready is received (or already received)."""
        t = timeout_sec if timeout_sec is not None else self._curobo_ready_timeout_sec
        if self._curobo_ready_event.is_set():
            return True
        self.get_logger().info(
            f'Waiting for /curobo/ready (up to {t:.0f}s)...')
        return self._curobo_ready_event.wait(timeout=t)

    def _call_curobo_goal_name(self, goal_name: str):
        """Call the CuRobo planning service with a named goal (home/place).

        Returns the service response or None on failure.
        """
        if self._curobo_client is None:
            self.get_logger().warn(
                f'_call_curobo_goal_name({goal_name!r}): no CuRobo client.')
            return None
        if not self._curobo_client.wait_for_service(
                timeout_sec=self._curobo_service_wait_sec):
            self.get_logger().warn(
                f'CuRobo service unavailable for goal_name={goal_name!r}.')
            return None
        request = PlanTrajectory.Request()
        request.goal_name = goal_name
        request.joint_state = self._latest_joints
        future = self._curobo_client.call_async(request)
        response = self._wait_for_future(
            future, f'curobo_{goal_name}', 'result',
            self._curobo_service_wait_sec)
        if response is None or not response.success:
            msg = getattr(response, 'message', 'no response') if response else 'no response'
            self.get_logger().warn(
                f'CuRobo planning failed for goal_name={goal_name!r}: {msg}')
            return None
        return response

    def _home_before_capture(self) -> None:
        """Move the arm to the home pose so the wrist camera faces straight down."""
        self.get_logger().info('Moving to home pose before capture...')
        response = self._call_curobo_goal_name('home')
        if response is None:
            self.get_logger().warn(
                'Home-before-capture planning failed; continuing without home move.')
            return
        try:
            self._send_and_wait(self._arm_client, response.trajectory, 'home')
        except Exception as exc:
            self.get_logger().warn(f'Home-before-capture execution failed (non-fatal): {exc}')

    def _plan_and_execute_place(self, goal_name: str) -> None:
        """Transit to place destination, release gripper, and handle bookshelf moves."""
        self.get_logger().info(f'Planning place trajectory to {goal_name!r}...')
        response = self._call_curobo_goal_name(goal_name)
        if response is None:
            raise RuntimeError(f'Place planning failed for goal_name={goal_name!r}')

        # Transit to the target (safe-Z waypoints).
        self._send_and_wait(self._arm_client, response.trajectory, f'place_transit_{goal_name}')

        # Bookshelf insert (push object onto shelf).
        insert_traj = getattr(response, 'insert_trajectory', None)
        if insert_traj and getattr(insert_traj, 'points', None):
            self._send_and_wait(self._arm_client, insert_traj, 'bookshelf_insert')

        # Release the gripper.
        self._send_gripper(closed=False)

        # Bookshelf retract (pull arm back from shelf).
        retract_traj = getattr(response, 'retract_trajectory', None)
        if retract_traj and getattr(retract_traj, 'points', None):
            self._send_and_wait(self._arm_client, retract_traj, 'bookshelf_retract')

        self.get_logger().info(f'Place to {goal_name!r} complete.')

    def _plan_and_execute_home(self) -> None:
        """Move the arm back to home pose after place."""
        self.get_logger().info('Returning to home pose...')
        response = self._call_curobo_goal_name('home')
        if response is None:
            raise RuntimeError('Return-home planning failed.')
        self._send_and_wait(self._arm_client, response.trajectory, 'return_home')
        self.get_logger().info('Returned to home pose.')

    def _reset_pipeline_state(self) -> None:
        self._pipeline_busy = False
        self._active_task = ''


def _goal_status_succeeded() -> int:
    if GoalStatus is None:
        return 4
    return int(getattr(GoalStatus, 'STATUS_SUCCEEDED', 4))


def _is_nonfatal_gripper_cancel(label: str, status, trajectory) -> bool:
    if GoalStatus is None:
        canceled = 5
    else:
        canceled = getattr(GoalStatus, 'STATUS_CANCELED', 5)
    return (
        str(label).startswith('gripper_close')
        and int(status) == int(canceled)
        and _is_gripper_close_trajectory(trajectory)
    )


def _is_gripper_close_trajectory(trajectory) -> bool:
    points = list(getattr(trajectory, 'points', []) or [])
    if not points:
        return False
    positions = list(getattr(points[-1], 'positions', []) or [])
    if not positions:
        return False
    return float(positions[0]) > 0.0


def _action_timeout_sec(trajectory) -> float:
    base = _env_float('PIPELINE_ACTION_TIMEOUT_SEC', 60.0)
    duration = _trajectory_duration_sec(trajectory)
    if duration is None:
        return base
    margin = _env_float('PIPELINE_ACTION_TIMEOUT_MARGIN_SEC', 15.0)
    return max(base, duration + margin)


def _trajectory_duration_sec(trajectory):
    points = list(getattr(trajectory, 'points', []) or [])
    if not points:
        return None
    stamp = getattr(points[-1], 'time_from_start', None)
    if stamp is None:
        return None
    return (
        float(getattr(stamp, 'sec', 0.0))
        + float(getattr(stamp, 'nanosec', 0.0)) / 1e9
    )


def _trajectory_summary(trajectory) -> str:
    joint_names = list(getattr(trajectory, 'joint_names', []) or [])
    points = list(getattr(trajectory, 'points', []) or [])
    if not points:
        return f'joints={joint_names} points=0'
    final = points[-1]
    positions = [round(float(x), 4) for x in getattr(final, 'positions', [])]
    duration = _trajectory_duration_sec(trajectory)
    duration_text = 'unknown' if duration is None else f'{duration:.2f}s'
    return (
        f'joints={joint_names} points={len(points)} '
        f'duration={duration_text} final_positions={positions}'
    )


def _joint_state_to_trajectory(joint_state) -> 'JointTrajectory':
    """Build a single-point JointTrajectory to move the arm to joint_state."""
    jt = JointTrajectory()
    jt.joint_names = list(joint_state.name)
    pt = JointTrajectoryPoint()
    pt.positions = list(joint_state.position)
    pt.velocities = [0.0] * len(joint_state.position)
    pt.time_from_start = RosDuration(sec=5, nanosec=0)
    jt.points.append(pt)
    return jt


def main(args=None) -> None:
    if rclpy is None:
        raise ImportError('rclpy is required for team_8 runtime.')
    rclpy.init(args=args)
    node = PipelineOrchestrator()
    # MultiThreadedExecutor: _send_and_wait calls spin_until_future_complete
    # from inside action-done callbacks, which deadlocks with a single-threaded
    # executor.
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
