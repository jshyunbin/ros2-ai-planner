"""Pipeline orchestrator: task_commands → segmentation → GraspGen → CuRobo → execution.

Pick execution follows yeina's 3-phase approach:
  1. Send approach-and-grasp trajectory (arm moves to grasp contact)
  2. Close gripper
  3. Send lift trajectory (arm lifts with object)

Natural-language task commands ("A를 B로 옮겨라") are parsed via GeminiAPI
into a structured task queue.  Each task specifies an object name and a
destination.  After a successful pick-and-place the result is verified by
sending a wrist-camera image back to Gemini.

All execution calls are blocking (_send_and_wait); the node is spun with
MultiThreadedExecutor so spin_until_future_complete works inside callbacks.
"""

import collections
import copy
import json
import os
import threading

try:  # pragma: no cover - runtime dependency
    import rclpy
    from action_msgs.msg import GoalStatus
    from rclpy.action import ActionClient
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.node import Node
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import JointState, Image
    from std_msgs.msg import Bool, Empty, String
    from control_msgs.action import FollowJointTrajectory
    from geometry_msgs.msg import Pose
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from builtin_interfaces.msg import Duration as RosDuration
except ImportError:  # pragma: no cover - import-only test fallback
    rclpy = None
    ActionClient = None
    GoalStatus = None
    JointState = None
    Image = None
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

try:
    from team_8.gemini_api import (
        GeminiAPI,
        GeminiAPIError,
        DESTINATIONS as _GEMINI_DESTINATIONS,
        build_segmentation_prompt as _build_seg_prompt,
    )
    _GEMINI_API_AVAILABLE = True
except ImportError:
    GeminiAPI = None  # type: ignore[misc,assignment]
    GeminiAPIError = RuntimeError  # type: ignore[misc,assignment]
    _GEMINI_API_AVAILABLE = False

    def _build_seg_prompt(object_name: str) -> str:  # type: ignore[misc]
        return object_name


# Robotiq 2F-85 gripper constants (from challenge_constants.py / yeina).
_GRIPPER_JOINT = 'robotiq_85_left_knuckle_joint'
_GRIPPER_OPEN = 0.0
# Do NOT set to 0.8 (fully closed): position-controlled gripper ignores contact
# force and embeds thin objects into the finger mesh.  0.6 rad grips firmly
# without interpenetration.  Override with PIPELINE_GRIPPER_CLOSE_POSITION.
_GRIPPER_CLOSED = _env_float('PIPELINE_GRIPPER_CLOSE_POSITION', 0.6)

# Destination → place_poses.yml goal_name mapping.
# Set PIPELINE_PLACE_GOAL_STORAGE_1 / _STORAGE_2 to override defaults.
_DESTINATION_TO_GOAL: dict[str, str] = {
    'storage_1':      os.environ.get('PIPELINE_PLACE_GOAL_STORAGE_1', 'storageB_1'),
    'storage_2':      os.environ.get('PIPELINE_PLACE_GOAL_STORAGE_2', 'storageA_1'),
    'bookshelf_floor1': os.environ.get('PIPELINE_PLACE_GOAL_BOOKSHELF1', 'bookshelf_floor1'),
    'bookshelf_floor2': os.environ.get('PIPELINE_PLACE_GOAL_BOOKSHELF2', 'bookshelf_floor2'),
    'unspecified':    '',  # falls back to self._place_goal parameter
}


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
        self.declare_parameter('curobo_service_wait_sec', 120.0)
        self.declare_parameter('curobo_ready_timeout_sec', 180.0)
        self.declare_parameter('enable_motion_execution', False)
        self.declare_parameter('arm_action_name',
                               '/ur5_controller/follow_joint_trajectory')
        self.declare_parameter('gripper_action_name',
                               '/gripper_controller/follow_joint_trajectory')
        self.declare_parameter('rgb_camera_topic', '/camera/color/image_raw')
        self.declare_parameter('place_goal', 'storageA_1')
        self.declare_parameter('auto_loop', True)
        self.declare_parameter('gemini_model', 'gemini-2.5-flash')
        self.declare_parameter('pick_verify_retries', 1)

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
        self._gemini_model = str(self.get_parameter('gemini_model').value)
        self._pick_verify_retries = int(
            self.get_parameter('pick_verify_retries').value)

        # Reentrant group: pipeline callbacks invoke spin_until_future_complete
        # (via _wait_for_future) which would deadlock under a mutually exclusive group.
        self._reentrant = (ReentrantCallbackGroup()
                           if ReentrantCallbackGroup is not None else None)

        self._task_sub = self.create_subscription(
            String, self.TASK_COMMANDS_TOPIC, self.task_command_callback, 10,
            callback_group=self._reentrant)
        self._joint_sub = self.create_subscription(
            JointState, self.JOINT_STATES_TOPIC, self._cache_joints, 10)

        # Subscribe to wrist camera for post-pick verification.
        _rgb_topic = str(self.get_parameter('rgb_camera_topic').value)
        self._latest_rgb_msg = None
        if Image is not None:
            self.create_subscription(
                Image, _rgb_topic, self._cache_rgb, 10)
            self.get_logger().info(f'Subscribed to RGB camera: {_rgb_topic}')

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
        if BoolNone is not None:
            self.create_service(
                BoolNone, '/save_scan_pose', self._save_scan_pose_callback)
            self.create_service(
                BoolNone, '/clear_scan_poses', self._clear_scan_poses_callback)

        # Publisher for TSDF reset — sent after each arm movement.
        self._reset_map_pub = self.create_publisher(Empty, '/curobo/reset_map', 10)

        # ── Pipeline state ────────────────────────────────────────────────────
        self._pipeline_busy = False
        self._latest_segmentation = None
        self._latest_graspgen = None
        self._latest_joints = None

        # Task queue: deque of dicts with 'object' and 'destination'.
        self._task_queue: collections.deque = collections.deque()
        # Currently executing task data.
        self._active_task_data: dict | None = None
        # How many pick-and-verify retries remain for the active task.
        self._active_task_retries_left: int = 0

        # Scan poses: arm moves to each before segmentation on retry.
        self._scan_poses = _load_default_scan_poses()
        self._scan_attempt_idx = 0
        self._current_task = ''  # object name carried across retries
        self._min_object_points = 500
        self._max_planning_retries = int(
            os.environ.get('PIPELINE_MAX_PLANNING_RETRIES', '2'))

        # ── GeminiAPI for command parsing + verification ───────────────────
        self._gemini_api: GeminiAPI | None = None
        if _GEMINI_API_AVAILABLE and GeminiAPI is not None:
            try:
                self._gemini_api = GeminiAPI(
                    model=self._gemini_model,
                    logger=self.get_logger(),
                )
                self.get_logger().info(
                    f'GeminiAPI ready (model={self._gemini_model}).')
            except Exception as exc:
                self.get_logger().warn(
                    f'GeminiAPI init failed — task parsing disabled: {exc}')

        # ── Motion clients ────────────────────────────────────────────────
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
            f'motion_execution={self._enable_motion_execution} '
            f'gemini_api={self._gemini_api is not None}'
        )

    # ── ROS2 subscriptions ────────────────────────────────────────────────────

    def _on_curobo_ready(self, msg) -> None:
        if getattr(msg, 'data', False):
            self.get_logger().info('Received /curobo/ready — TSDF has enough frames.')
            self._curobo_ready_event.set()

    def task_command_callback(self, msg: String) -> None:
        """Handle incoming task command string.

        If GeminiAPI is available, parse natural-language instruction into a
        structured task queue ("A를 B로 옮겨라").  Otherwise treat the raw
        string as a single object name with the default place destination.
        """
        instruction = str(msg.data).strip()
        self.get_logger().info(f'Received task command: {instruction!r}')

        if not self._auto_run_on_task_command:
            return

        tasks: list[dict] = []

        if self._gemini_api is not None:
            try:
                plan = self._gemini_api.parse_task_command(instruction)
                tasks = plan.get('tasks', [])
                self.get_logger().info(
                    f'Parsed task plan: {tasks}')
            except Exception as exc:
                self.get_logger().warn(
                    f'parse_task_command failed ({exc}); '
                    f'treating command as raw object name.')

        if not tasks:
            # Fallback: treat the whole command as a single object name.
            tasks = [{'object': instruction, 'destination': 'unspecified'}]

        # Enqueue all tasks and start if idle.
        for t in tasks:
            self._task_queue.append(t)
        self.get_logger().info(
            f'Task queue: {len(self._task_queue)} task(s) pending.')

        if not self._pipeline_busy:
            self._start_next_task()

    def _cache_joints(self, msg) -> None:
        self._latest_joints = msg

    def _cache_rgb(self, msg) -> None:
        self._latest_rgb_msg = msg

    def _save_scan_pose_callback(self, request, response):
        """Append current joint state to the scan pose list."""
        if self._latest_joints is None:
            self.get_logger().warn('save_scan_pose: no /joint_states received yet.')
            return response
        self._scan_poses.append(self._latest_joints)
        positions = [round(float(p), 4) for p in self._latest_joints.position]
        self.get_logger().info(
            f'Scan pose #{len(self._scan_poses)} saved: {positions}')
        return response

    def _clear_scan_poses_callback(self, request, response):
        """Clear all saved scan poses."""
        n = len(self._scan_poses)
        self._scan_poses.clear()
        self._scan_attempt_idx = 0
        self.get_logger().info(f'Cleared {n} scan pose(s).')
        return response

    # ── Task queue management ─────────────────────────────────────────────────

    def _start_next_task(self) -> None:
        """Pop the next task from the queue and start it, or go idle."""
        if not self._task_queue:
            self.get_logger().info('Task queue empty — pipeline idle.')
            return

        task_data = self._task_queue.popleft()
        self._active_task_data = task_data
        self._active_task_retries_left = self._pick_verify_retries
        object_name = task_data.get('object', '')
        destination = task_data.get('destination', 'unspecified')
        self.get_logger().info(
            f"Starting task: pick '{object_name}' → place '{destination}'.")
        self._start_task_attempt()

    def _start_task_attempt(self) -> None:
        """Begin a single pick attempt for the active task."""
        if self._active_task_data is None:
            return
        object_name = self._active_task_data.get('object', '')
        self._run_pipeline(object_name)

    def _restart_active_task(self) -> None:
        """Re-run the current task (pick verification failed, retries remain)."""
        if self._active_task_data is None:
            return
        object_name = self._active_task_data.get('object', '')
        self.get_logger().info(
            f"Restarting task for '{object_name}' "
            f'({self._active_task_retries_left} retr(ies) left).')
        self._start_task_attempt()

    def _active_destination(self) -> str:
        """Resolve the active task's destination to a place_poses.yml goal name."""
        if self._active_task_data is None:
            return self._place_goal
        dest = self._active_task_data.get('destination', 'unspecified')
        mapped = _DESTINATION_TO_GOAL.get(dest, '')
        return mapped if mapped else self._place_goal

    def _reset_pipeline_state(self, *, success: bool = False, reason: str = '') -> None:
        """Clear active task state and start the next task if any.

        When success=False and the gripper may still be holding an object
        (e.g. place failed), the remaining queue is cleared for safety.
        """
        if reason:
            level = 'info' if success else 'warn'
            getattr(self.get_logger(), level)(
                f'Task {"succeeded" if success else "failed"}: {reason}')

        self._pipeline_busy = False
        self._active_task_data = None
        self._scan_attempt_idx = 0
        self._current_task = ''
        self._latest_segmentation = None
        self._latest_graspgen = None

        # Auto-loop: re-queue a new attempt for the same object after success.
        # Only applies when the queue was empty (single-object continuous loop).
        if success and self._auto_loop and not self._task_queue:
            # Nothing else queued — idle; the user can send the next command.
            self.get_logger().info('Auto-loop idle — waiting for next command.')
            return

        self._start_next_task()

    # ── Pipeline entry ────────────────────────────────────────────────────────

    def _run_pipeline(self, object_name: str) -> None:
        """Begin the segmentation → GraspGen → CuRobo → execution pipeline."""
        object_name = object_name.strip()
        if not object_name:
            self.get_logger().warn('Ignoring empty object name.')
            self._reset_pipeline_state(success=False, reason='empty object name')
            return
        if self._pipeline_busy:
            self.get_logger().warn(
                f"Pipeline already busy with '{self._current_task}'. "
                f"Ignoring new request for '{object_name}'.")
            return
        if not self._segmentation_client.wait_for_service(
                timeout_sec=self._segmentation_service_wait_sec):
            self.get_logger().warn(
                f'Segmentation service unavailable: {self._segmentation_service_name}')
            self._reset_pipeline_state(
                success=False, reason='segmentation service unavailable')
            return

        self._pipeline_busy = True
        self._current_task = object_name
        self._scan_attempt_idx = 0

        # Wait for TSDF to accumulate enough frames before doing anything.
        if self._enable_motion_execution:
            if not self._wait_for_curobo_ready():
                self.get_logger().error(
                    'CuRobo TSDF not ready within '
                    f'{self._curobo_ready_timeout_sec:.0f}s — aborting pipeline.')
                self._reset_pipeline_state(
                    success=False, reason='curobo TSDF not ready')
                return

        # Ensure gripper is open before starting a new pick.
        if self._gripper_client is not None:
            self._send_gripper(closed=False)

        # Move to home pose so the wrist camera faces straight down for capture.
        if self._enable_motion_execution:
            self._home_before_capture()

        self._run_segmentation_attempt(object_name)

    def _run_segmentation_attempt(self, object_name: str) -> None:
        """Move to current scan pose (if any) and call segmentation service.

        Attempt 0 stays at home (wrist camera straight down).
        Retries (attempt ≥ 1) move to scan_poses[attempt-1] before segmenting.
        """
        scan_idx = self._scan_attempt_idx - 1
        if scan_idx >= 0 and self._scan_poses and self._arm_client is not None:
            pose = self._scan_poses[scan_idx % len(self._scan_poses)]
            scan_traj = _joint_state_to_trajectory(pose)
            try:
                self._send_and_wait(self._arm_client, scan_traj, 'move_to_scan_pose')
                self.get_logger().info(
                    f'Moved to scan pose #{scan_idx + 1}/{len(self._scan_poses)} '
                    f'positions={[round(float(p), 3) for p in pose.position]}.')
                self._reset_tsdf_after_move()
            except Exception as exc:
                self.get_logger().warn(f'move_to_scan_pose failed (non-fatal): {exc}')

        if not self._segmentation_client.wait_for_service(
                timeout_sec=self._segmentation_service_wait_sec):
            self.get_logger().warn('Segmentation service unavailable.')
            self._reset_pipeline_state(
                success=False, reason='segmentation service unavailable')
            return

        # Build a rich segmentation prompt including object descriptions.
        seg_prompt = _build_seg_prompt(object_name)

        request = StringString.Request()
        request.data = seg_prompt
        future = self._segmentation_client.call_async(request)
        future.add_done_callback(self._on_segmentation_done)
        self.get_logger().info(
            f'Started segmentation for "{object_name}" '
            f'(attempt {self._scan_attempt_idx + 1})')

    # ── Pipeline callbacks ────────────────────────────────────────────────────

    def _on_segmentation_done(self, future) -> None:
        try:
            result = future.result()
            payload = json.loads(result.data)
        except Exception as exc:
            self.get_logger().error(f'Segmentation service call failed: {exc}')
            self._reset_pipeline_state(
                success=False, reason=f'segmentation call error: {exc}')
            return

        point_count = int(payload.get('object_point_count', 0))
        seg_failed = not payload.get('success')
        too_few_points = (
            payload.get('success') and point_count < self._min_object_points)

        if seg_failed or too_few_points:
            # Special case: Gemini confirmed the object is absent → skip.
            if payload.get('object_not_found'):
                self.get_logger().warn(
                    f"Object '{self._current_task}' not found in workspace "
                    f"(Gemini returned no detections). Skipping task.")
                self._reset_pipeline_state(
                    success=False,
                    reason=f"'{self._current_task}' not found in workspace")
                return

            reason = (
                f'Gemini/SAM2 failed: {payload.get("error", "unknown")}'
                if seg_failed
                else f'point cloud too sparse ({point_count} < {self._min_object_points})'
            )
            self.get_logger().warn(
                f'Segmentation attempt {self._scan_attempt_idx + 1} failed: {reason}')
            self._scan_attempt_idx += 1

            max_attempts = 1 + len(self._scan_poses)
            if self._scan_attempt_idx < max_attempts:
                self.get_logger().info(
                    f'Retrying segmentation from scan pose '
                    f'#{self._scan_attempt_idx + 1}/{max_attempts}...')
                self._run_segmentation_attempt(self._current_task)
            else:
                self.get_logger().warn(
                    f'Segmentation failed after {self._scan_attempt_idx} attempt(s). '
                    f'Giving up.')
                self._reset_pipeline_state(success=False, reason=reason)
            return

        self._latest_segmentation = payload
        self.get_logger().info(
            f"Segmentation ready label={payload.get('label', 'target')} "
            f"points={point_count}; requesting GraspGen.")

        if not self._graspgen_client.wait_for_service(
                timeout_sec=self._graspgen_service_wait_sec):
            self.get_logger().warn(
                f'GraspGen service unavailable: {self._graspgen_service_name}')
            self._reset_pipeline_state(
                success=False, reason='GraspGen service unavailable')
            return

        request = StringString.Request()
        request.data = str(self._latest_segmentation.get('cloud_stamp_ns', ''))
        future = self._graspgen_client.call_async(request)
        future.add_done_callback(self._on_graspgen_done)

    def _on_graspgen_done(self, future) -> None:
        try:
            result = future.result()
            payload = json.loads(result.data)
        except Exception as exc:
            self.get_logger().error(f'GraspGen service call failed: {exc}')
            self._reset_pipeline_state(
                success=False, reason=f'GraspGen call error: {exc}')
            return

        if not payload.get('success'):
            self.get_logger().warn(
                f"GraspGen failed: {payload.get('error', payload)}")
            self._reset_pipeline_state(success=False, reason='GraspGen failed')
            return

        self._latest_graspgen = payload
        top_grasps = payload.get('top_grasps') or []
        if not top_grasps:
            self.get_logger().warn(
                'GraspGen returned success but no ranked grasps.')
            self._reset_pipeline_state(success=False, reason='no ranked grasps')
            return

        self.get_logger().info(
            'Pipeline result '
            f"label={self._latest_segmentation.get('label', 'target')} "
            f"best_translation={top_grasps[0].get('translation')} "
            f"confidence={top_grasps[0].get('confidence')} "
            f"num_candidates={len(top_grasps)}"
        )

        if not self._enable_motion_execution:
            self._reset_pipeline_state(
                success=True, reason='motion execution disabled')
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
            self._reset_pipeline_state(success=False, reason='no valid grasp poses')
            return

        # ── White-sphere gripper centering correction ─────────────────────────
        # segmentation_service detects the gripper white sphere and the object
        # centroid in the wrist camera image, back-projects both to world frame,
        # and returns the lateral offset needed to align the gripper with the
        # object center.  Apply it to all grasp candidates before planning.
        xy_corr = self._latest_segmentation.get('grasp_xy_correction')
        if xy_corr and len(xy_corr) == 2:
            dx, dy = float(xy_corr[0]), float(xy_corr[1])
            self.get_logger().info(
                f'Applying gripper-sphere centering: dx={dx:+.4f}m dy={dy:+.4f}m '
                f'to {len(grasp_poses)} candidates.')
            corrected = []
            for p in grasp_poses:
                pc = copy.deepcopy(p)
                pc.position.x += dx
                pc.position.y += dy
                corrected.append(pc)
            grasp_poses = corrected

        self._plan_and_execute_pick(grasp_poses)

    def _plan_and_execute_pick(self, grasp_poses: list) -> None:
        """Call CuRobo service with all candidates, then execute pick phases."""
        if self._latest_joints is None:
            self.get_logger().warn(
                'No /joint_states received; skipping motion execution.')
            self._reset_pipeline_state(success=False, reason='no joint states')
            return
        if self._curobo_client is None:
            self.get_logger().warn(
                'Motion execution enabled but CuRobo service client unavailable.')
            self._reset_pipeline_state(success=False, reason='no CuRobo client')
            return

        self.get_logger().info(
            f'Waiting for CuRobo service {self._curobo_service_name} '
            f'(up to {self._curobo_service_wait_sec:.1f}s, '
            f'{len(grasp_poses)} candidates).')
        if not self._curobo_client.wait_for_service(
                timeout_sec=self._curobo_service_wait_sec):
            self.get_logger().warn(
                f'CuRobo service unavailable: {self._curobo_service_name}')
            self._reset_pipeline_state(success=False, reason='CuRobo unavailable')
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
            self._reset_pipeline_state(
                success=False, reason=f'CuRobo call error: {exc}')
            return

        if not result.success:
            self.get_logger().warn(f'CuRobo pick planning failed: {result.message}')
            self._scan_attempt_idx += 1
            if self._scan_attempt_idx <= self._max_planning_retries:
                self.get_logger().info(
                    f'Retrying pick pipeline from scan pose '
                    f'#{self._scan_attempt_idx + 1} '
                    f'(attempt {self._scan_attempt_idx}/{self._max_planning_retries})...')
                self._run_segmentation_attempt(self._current_task)
            else:
                self.get_logger().warn(
                    f'Pick planning failed after {self._scan_attempt_idx} attempt(s). '
                    'Giving up.')
                self._reset_pipeline_state(
                    success=False, reason='pick planning failed')
            return

        self.get_logger().info(result.message)

        # 4-phase pick execution: open gripper → approach+grasp → close → lift.
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
            self._reset_pipeline_state(
                success=False, reason=f'pick execution error: {exc}')
            return

        # Place: transit → optional bookshelf insert → release → optional retract.
        dest_goal = self._active_destination()
        try:
            self._plan_and_execute_place(dest_goal)
        except Exception as exc:
            self.get_logger().error(f'Place execution failed: {exc}')
            try:
                self._send_gripper(closed=False)
            except Exception:
                pass
            # Don't reset here: try returning home first.

        # Return home; TSDF reset follows inside _plan_and_execute_home.
        try:
            self._plan_and_execute_home()
        except Exception as exc:
            self.get_logger().warn(
                f'Return-home after place failed (non-fatal): {exc}')

        # Post-pick verification via Gemini.
        if self._gemini_api is not None and self._enable_motion_execution:
            self._run_post_pick_verification()
        else:
            # No verification: succeed immediately and start next task.
            self._reset_pipeline_state(
                success=True, reason=f'pick-place complete → {dest_goal}')

    # ── Post-pick verification ────────────────────────────────────────────────

    def _run_post_pick_verification(self) -> None:
        """Capture current wrist-camera image and verify object was removed."""
        if self._active_task_data is None:
            self._reset_pipeline_state(success=True, reason='pick-place complete')
            return

        object_name = self._active_task_data.get('object', self._current_task)
        destination = self._active_task_data.get('destination', 'unspecified')
        dest_goal = self._active_destination()

        pil_image = self._capture_rgb_as_pil()
        if pil_image is None:
            self.get_logger().warn(
                'Post-pick verification skipped: no RGB image available.')
            self._reset_pipeline_state(
                success=True, reason=f'pick-place complete → {dest_goal} (no verify)')
            return

        try:
            result = self._gemini_api.verify_object_removed(
                pil_image,
                object_name=object_name,
                destination=destination,
            )
        except Exception as exc:
            self.get_logger().warn(
                f'verify_object_removed failed ({exc}); assuming success.')
            self._reset_pipeline_state(
                success=True, reason=f'pick-place complete → {dest_goal} (verify error)')
            return

        present = result.get('present_in_source_workspace', False)
        confidence = result.get('confidence', 0.0)
        reason_text = result.get('reason', '')

        self.get_logger().info(
            f'Post-pick verification: present={present} '
            f'confidence={confidence:.2f} reason="{reason_text}"')

        if not present:
            # Object is gone → success.
            self._reset_pipeline_state(
                success=True,
                reason=f'pick verified removed → {dest_goal}: {reason_text}')
        elif self._active_task_retries_left > 0:
            # Object still visible → retry pick.
            self._active_task_retries_left -= 1
            self.get_logger().warn(
                f"Object '{object_name}' still in workspace after pick "
                f'({reason_text}). Retrying '
                f'({self._active_task_retries_left} left)...')
            self._pipeline_busy = False
            self._scan_attempt_idx = 0
            self._restart_active_task()
        else:
            # Out of retries → fail.
            self.get_logger().warn(
                f"Object '{object_name}' still present after all retries. "
                'Skipping task.')
            self._reset_pipeline_state(
                success=False,
                reason=f'pick failed after verification retries: {reason_text}')

    def _capture_rgb_as_pil(self):
        """Convert the latest RGB sensor_msgs/Image to a PIL Image.

        Returns None if no image is available or PIL is not installed.
        """
        msg = self._latest_rgb_msg
        if msg is None:
            return None
        try:
            import numpy as np
            from PIL import Image as PILImage
            encoding = getattr(msg, 'encoding', 'rgb8')
            arr = (
                np.frombuffer(bytes(msg.data), dtype=np.uint8)
                .reshape(msg.height, msg.width, -1)
            )
            if arr.shape[2] == 3:
                if 'bgr' in encoding:
                    arr = arr[:, :, ::-1]  # BGR → RGB
            elif arr.shape[2] == 4:
                arr = arr[:, :, :3]
                if 'bgr' in encoding:
                    arr = arr[:, :, ::-1]
            return PILImage.fromarray(arr.copy())
        except Exception as exc:
            self.get_logger().warn(f'_capture_rgb_as_pil failed: {exc}')
            return None

    # ── Execution helpers ─────────────────────────────────────────────────────

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
        target_pos = _GRIPPER_CLOSED if closed else _GRIPPER_OPEN
        pt.positions = [target_pos]
        duration_sec = 2 if closed else 1
        pt.time_from_start = RosDuration(sec=duration_sec, nanosec=0)
        jt.points.append(pt)
        label = 'gripper_close' if closed else 'gripper_open'
        try:
            self._send_and_wait(self._gripper_client, jt, label)
        except Exception as exc:
            self.get_logger().warn(f'{label} failed (non-fatal): {exc}')
        if closed:
            self._log_gripper_contact(target_pos)

    def _log_gripper_contact(self, commanded_pos: float) -> None:
        """Log actual gripper position after closing to detect contact."""
        if self._latest_joints is None:
            return
        try:
            names = list(self._latest_joints.name)
            if _GRIPPER_JOINT not in names:
                return
            idx = names.index(_GRIPPER_JOINT)
            actual = float(self._latest_joints.position[idx])
            gap = commanded_pos - actual
            if gap > 0.05:
                self.get_logger().info(
                    f'Gripper contact detected: commanded={commanded_pos:.3f} '
                    f'actual={actual:.3f} rad (stopped {gap:.3f} rad early — '
                    f'object or floor contact).')
            else:
                self.get_logger().info(
                    f'Gripper closed fully: actual={actual:.3f} rad '
                    f'(no contact resistance).')
        except Exception:
            pass

    # ── CuRobo ready + home/place helpers ─────────────────────────────────────

    def _wait_for_curobo_ready(self, timeout_sec: float | None = None) -> bool:
        """Block until /curobo/ready is received (or already received)."""
        t = timeout_sec if timeout_sec is not None else self._curobo_ready_timeout_sec
        if self._curobo_ready_event.is_set():
            return True
        self.get_logger().info(
            f'Waiting for /curobo/ready (up to {t:.0f}s)...')
        return self._curobo_ready_event.wait(timeout=t)

    def _call_curobo_goal_name(self, goal_name: str):
        """Call the CuRobo planning service with a named goal."""
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
        try:
            response = self._wait_for_future(
                future, f'curobo_{goal_name}', 'result',
                self._curobo_service_wait_sec)
        except RuntimeError as exc:
            self.get_logger().warn(
                f'CuRobo service call timed out for goal_name={goal_name!r}: {exc}')
            return None
        if response is None or not response.success:
            msg = getattr(response, 'message', 'no response') if response else 'no response'
            self.get_logger().warn(
                f'CuRobo planning failed for goal_name={goal_name!r}: {msg}')
            return None
        return response

    def _reset_tsdf_after_move(self) -> None:
        """Publish /curobo/reset_map so ghost voxels from arm movement are cleared."""
        self._reset_map_pub.publish(Empty())
        self.get_logger().info(
            'Published /curobo/reset_map — TSDF cleared after arm movement.')

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
            self._reset_tsdf_after_move()
        except Exception as exc:
            self.get_logger().warn(
                f'Home-before-capture execution failed (non-fatal): {exc}')

    def _plan_and_execute_place(self, goal_name: str) -> None:
        """Transit to place destination, release gripper, handle bookshelf moves."""
        self.get_logger().info(f'Planning place trajectory to {goal_name!r}...')
        response = self._call_curobo_goal_name(goal_name)
        if response is None:
            raise RuntimeError(f'Place planning failed for goal_name={goal_name!r}')

        self._send_and_wait(
            self._arm_client, response.trajectory, f'place_transit_{goal_name}')

        insert_traj = getattr(response, 'insert_trajectory', None)
        if insert_traj and getattr(insert_traj, 'points', None):
            self._send_and_wait(self._arm_client, insert_traj, 'bookshelf_insert')

        self._send_gripper(closed=False)

        retract_traj = getattr(response, 'retract_trajectory', None)
        if retract_traj and getattr(retract_traj, 'points', None):
            self._send_and_wait(self._arm_client, retract_traj, 'bookshelf_retract')

        self.get_logger().info(f'Place to {goal_name!r} complete.')
        self._reset_tsdf_after_move()

    def _plan_and_execute_home(self) -> None:
        """Move the arm back to home pose after place."""
        self.get_logger().info('Returning to home pose...')
        response = self._call_curobo_goal_name('home')
        if response is None:
            raise RuntimeError('Return-home planning failed.')
        self._send_and_wait(self._arm_client, response.trajectory, 'return_home')
        self.get_logger().info('Returned to home pose.')
        self._reset_tsdf_after_move()


# ── Module-level helpers ──────────────────────────────────────────────────────

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


_UR5_JOINT_NAMES = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]


class _ScanPose:
    """Lightweight stand-in for sensor_msgs/JointState."""

    __slots__ = ('name', 'position')

    def __init__(self, positions: list[float]) -> None:
        self.name = list(_UR5_JOINT_NAMES)
        self.position = list(positions)


def _load_default_scan_poses() -> list[_ScanPose]:
    """Parse PIPELINE_SCAN_POSES env var into a list of _ScanPose objects.

    Format: semicolon-separated poses, each pose is 6 comma-separated rads.
    Example:
      PIPELINE_SCAN_POSES=-1.332,-2.402,1.538,-0.863,-1.114,-1.292
    """
    raw = os.environ.get('PIPELINE_SCAN_POSES', '').strip()
    if not raw:
        return []
    poses = []
    for i, token in enumerate(raw.split(';')):
        token = token.strip()
        if not token:
            continue
        try:
            angles = [float(v.strip()) for v in token.split(',')]
        except ValueError as exc:
            raise ValueError(
                f'PIPELINE_SCAN_POSES pose #{i + 1} is not valid floats: {token!r}'
            ) from exc
        if len(angles) != 6:
            raise ValueError(
                f'PIPELINE_SCAN_POSES pose #{i + 1} must have 6 values, '
                f'got {len(angles)}: {token!r}'
            )
        poses.append(_ScanPose(angles))
    return poses


def main(args=None) -> None:
    if rclpy is None:
        raise ImportError('rclpy is required for team_8 runtime.')
    rclpy.init(args=args)
    node = PipelineOrchestrator()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
