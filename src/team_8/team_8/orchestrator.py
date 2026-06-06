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
    from rclpy.node import Node
    from rclpy.executors import MultiThreadedExecutor
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String
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

    class Node:  # type: ignore[override]
        pass

    class String:  # type: ignore[override]
        pass

try:  # pragma: no cover - runtime dependency
    from riro_srvs.srv import StringString
    from riro_srvs.srv import PlanTrajectory
except ImportError:  # pragma: no cover - runtime dependency
    StringString = None
    PlanTrajectory = None

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
        self.declare_parameter('enable_motion_execution', False)
        self.declare_parameter('arm_action_name',
                               '/ur5_controller/follow_joint_trajectory')
        self.declare_parameter('gripper_action_name',
                               '/gripper_controller/follow_joint_trajectory')
        self.declare_parameter('target_goal', 'storage_1')
        self._target_goal = str(self.get_parameter('target_goal').value)

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
        self._enable_motion_execution = _as_bool(
            self.get_parameter('enable_motion_execution').value)

        self._task_sub = self.create_subscription(
            String, self.TASK_COMMANDS_TOPIC, self.task_command_callback, 10)
        self._joint_sub = self.create_subscription(
            JointState, self.JOINT_STATES_TOPIC, self._cache_joints, 10)
        self._segmentation_client = self.create_client(
            StringString, self._segmentation_service_name)
        self._graspgen_client = self.create_client(
            StringString, self._graspgen_service_name)

        self._pipeline_busy = False
        self._active_task = ''
        self._latest_segmentation = None
        self._latest_graspgen = None
        self._latest_joints = None

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

    def task_command_callback(self, msg: String) -> None:
        self.get_logger().info(f'Received task command: {msg.data}')
        if self._auto_run_on_task_command:
            self._run_pipeline(msg.data)

    def _cache_joints(self, msg) -> None:
        self._latest_joints = msg

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

        # Move to home first so the wrist camera observes the workspace, then
        # gate segmentation on a frame captured after the arm settled there.
        try:
            min_stamp_ns = self._home_before_capture()
        except Exception as exc:
            self.get_logger().error(f'Initial home move failed: {exc}')
            self._reset_pipeline_state()
            return

        request = StringString.Request()
        request.data = json.dumps({'prompt': task, 'min_stamp_ns': min_stamp_ns})
        future = self._segmentation_client.call_async(request)
        future.add_done_callback(self._on_segmentation_done)
        self.get_logger().info(
            f'Started segmentation for task: {task} (min_stamp_ns={min_stamp_ns})')

    # ── pipeline callbacks ────────────────────────────────────────────────────

    def _on_segmentation_done(self, future) -> None:
        try:
            result = future.result()
            payload = json.loads(result.data)
        except Exception as exc:
            self.get_logger().error(f'Segmentation service call failed: {exc}')
            self._reset_pipeline_state()
            return

        if not payload.get('success'):
            self.get_logger().warn(f'Segmentation failed: {payload}')
            self._reset_pipeline_state()
            return

        self._latest_segmentation = payload
        self.get_logger().info(
            f"Segmentation ready label={payload.get('label', 'target')} "
            f"points={payload.get('object_point_count', 0)}; requesting GraspGen.")

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

        # pick (open → approach+grasp → close → lift) → place → release → home.
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
            self._plan_and_execute_place(self._target_goal)
            self._plan_and_execute_home()
        except Exception as exc:
            self.get_logger().error(f'Pick/place execution failed: {exc}')
            # Best-effort: try to open the gripper so we don't drop/drag things.
            try:
                self._send_gripper(closed=False)
            except Exception:
                pass
        finally:
            self._reset_pipeline_state()

    def _call_curobo_blocking(self, request, label):
        """Call the CuRobo plan service and block for the response.

        Safe from inside an action/service done-callback because the node is
        spun with a MultiThreadedExecutor (see main()).
        """
        if self._curobo_client is None:
            raise RuntimeError(f'{label}: CuRobo service client unavailable.')
        if not self._curobo_client.wait_for_service(
                timeout_sec=self._curobo_service_wait_sec):
            raise RuntimeError(
                f'{label}: CuRobo service unavailable '
                f'({self._curobo_service_name}).')
        timeout_sec = _env_float('PIPELINE_PLAN_SERVICE_TIMEOUT_SEC', 600.0)
        future = self._curobo_client.call_async(request)
        result = self._wait_for_future(future, label, 'plan response', timeout_sec)
        if result is None:
            raise RuntimeError(f'{label}: CuRobo service returned no result.')
        return result

    def _plan_and_execute_place(self, goal_name: str) -> None:
        """Plan (collision-off) + execute the transit to a place destination,
        release the object, and (bookshelf) retract."""
        request = PlanTrajectory.Request()
        request.goal_name = goal_name
        request.joint_state = self._latest_joints
        result = self._call_curobo_blocking(request, f'place:{goal_name}')
        if not result.success:
            raise RuntimeError(f'place planning failed: {result.message}')
        self._send_and_wait(self._arm_client, result.trajectory, 'place_transit')
        if result.insert_trajectory and result.insert_trajectory.points:
            self._send_and_wait(
                self._arm_client, result.insert_trajectory, 'bookshelf_insert')
        # Release the object onto the basket / shelf board.
        self._send_gripper(closed=False)
        if result.retract_trajectory and result.retract_trajectory.points:
            self._send_and_wait(
                self._arm_client, result.retract_trajectory, 'bookshelf_retract')

    def _home_before_capture(self) -> int:
        """Move the arm to home so the wrist camera observes the workspace, and
        return the ROS time (ns) at which it settled.

        Returns 0 when motion execution is disabled or joints/CuRobo are not yet
        available — meaning no home move and no freshness gate, so GraspGen-only
        and standalone modes keep working.
        """
        if not self._enable_motion_execution or self._curobo_client is None:
            return 0
        if self._latest_joints is None:
            self.get_logger().warn(
                'No /joint_states yet; skipping initial home move and '
                'frame freshness gate.')
            return 0
        # Open the gripper: nothing should be carried at the start of a cycle, and
        # the next grasp assumes open fingers.
        self._send_gripper(closed=False)
        self._plan_and_execute_home()
        return self.get_clock().now().nanoseconds

    def _plan_and_execute_home(self) -> None:
        """Plan (collision-aware) + execute the return to the home pose."""
        request = PlanTrajectory.Request()
        request.goal_name = 'home'
        request.joint_state = self._latest_joints
        result = self._call_curobo_blocking(request, 'home')
        if not result.success:
            raise RuntimeError(f'home planning failed: {result.message}')
        self._send_and_wait(self._arm_client, result.trajectory, 'home')

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
