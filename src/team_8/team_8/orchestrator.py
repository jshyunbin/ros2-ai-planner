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
import time
from collections import deque
from pathlib import Path

try:  # pragma: no cover - runtime dependency
    from PIL import Image as PILImage
except ImportError:  # pragma: no cover - runtime dependency
    PILImage = None

try:  # pragma: no cover - runtime dependency
    import rclpy
    from action_msgs.msg import GoalStatus
    from cv_bridge import CvBridge
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image, JointState
    from std_msgs.msg import Bool, String
    from rclpy.callback_groups import ReentrantCallbackGroup
    from control_msgs.action import FollowJointTrajectory
    from geometry_msgs.msg import Pose
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from builtin_interfaces.msg import Duration as RosDuration
except ImportError:  # pragma: no cover - import-only test fallback
    rclpy = None
    ActionClient = None
    CvBridge = None
    GoalStatus = None
    Image = None
    JointState = None
    QoSProfile = None
    QoSDurabilityPolicy = None
    ReliabilityPolicy = None
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


try:  # pragma: no cover - ROS runtime dependency
    from ament_index_python.packages import get_package_share_directory
except ImportError:  # pragma: no cover - import-only test fallback
    get_package_share_directory = None

try:  # pragma: no cover - runtime dependency
    from riro_srvs.srv import StringString
    from riro_srvs.srv import PlanTrajectory
except ImportError:  # pragma: no cover - runtime dependency
    StringString = None
    PlanTrajectory = None

from team_8.gemini_api import GeminiAPI, GeminiAPIError
from team_8.place_pose_utils import load_place_poses
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
    TASK_PLAN_TOPIC = '/gemini/task_plan'
    TASK_VERIFICATION_TOPIC = '/gemini/task_verification'
    JOINT_STATES_TOPIC = '/joint_states'
    CUROBO_READY_TOPIC = '/curobo/ready'

    def __init__(self):
        if rclpy is None:
            raise ImportError('rclpy is required for team_8 runtime.')
        if StringString is None:
            raise ImportError('riro_srvs is required for team_8.')

        super().__init__('team_8')

        if get_package_share_directory is None:
            raise ImportError('ament_index_python is required for team_8.')
        default_place_poses_path = str(
            Path(get_package_share_directory('team_8'))
            / 'config'
            / 'place_poses.yml'
        )

        self.declare_parameter('segmentation_service_name',
                               '/segmentation/segment_prompt')
        self.declare_parameter('graspgen_service_name', '/graspgen/infer')
        self.declare_parameter('auto_run_on_task_command', True)
        self.declare_parameter('segmentation_service_wait_sec', 10.0)
        self.declare_parameter('graspgen_service_wait_sec', 10.0)
        self.declare_parameter('curobo_service_name', '/curobo/plan_trajectory')
        self.declare_parameter('curobo_service_wait_sec', 30.0)
        # Generous: covers CuRobo's background init (~tens of seconds) plus a
        # moment for the TSDF to fill. Only ever waited on the first cycle.
        self.declare_parameter('curobo_ready_wait_sec', 180.0)
        self.declare_parameter('enable_motion_execution', False)
        self.declare_parameter('arm_action_name',
                               '/ur5_controller/follow_joint_trajectory')
        self.declare_parameter('gripper_action_name',
                               '/gripper_controller/follow_joint_trajectory')
        self.declare_parameter('gemini_model', 'gemini-2.5-flash')
        self.declare_parameter('task_plan_topic', self.TASK_PLAN_TOPIC)
        self.declare_parameter('place_poses_path', default_place_poses_path)
        self.declare_parameter('return_home_after_place', True)
        self.declare_parameter(
            'verification_rgb_topic',
            '/wrist_camera/wrist_camera/color/image_raw')
        self.declare_parameter('enable_post_task_verification', True)
        self.declare_parameter('verification_settle_sec', 1.0)
        self.declare_parameter('verification_frame_timeout_sec', 3.0)
        self.declare_parameter('max_task_attempts', 3)
        self.declare_parameter(
            'verification_result_topic', self.TASK_VERIFICATION_TOPIC)
        self.declare_parameter(
            'verification_debug_dir', '.')

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
        self._return_home_after_place = _as_bool(
            self.get_parameter('return_home_after_place').value)
        self._verification_rgb_topic = str(
            self.get_parameter('verification_rgb_topic').value)
        self._enable_post_task_verification = _as_bool(
            self.get_parameter('enable_post_task_verification').value)
        self._verification_settle_sec = max(
            0.0, float(self.get_parameter('verification_settle_sec').value))
        self._verification_frame_timeout_sec = max(
            0.1,
            float(self.get_parameter('verification_frame_timeout_sec').value),
        )
        self._max_task_attempts = max(
            1, int(self.get_parameter('max_task_attempts').value))
        self._verification_debug_dir = Path(
            str(self.get_parameter('verification_debug_dir').value))
        self._verification_debug_dir.mkdir(parents=True, exist_ok=True)
        self._place_poses_path = str(
            self.get_parameter('place_poses_path').value)
        self._place_poses = load_place_poses(self._place_poses_path)
        self._curobo_ready_wait_sec = float(
            self.get_parameter('curobo_ready_wait_sec').value)

        self._bridge = CvBridge()
        image_qos = QoSProfile(depth=10)
        image_qos.reliability = ReliabilityPolicy.BEST_EFFORT

        # The pipeline runs synchronously inside task_command_callback and blocks
        # on service/action futures (home plan, place plan, arm/gripper actions).
        # Those clients MUST live in a callback group the executor can service
        # while this callback is blocked — otherwise the response can never be
        # delivered (it's stuck behind the blocked callback in the same
        # MutuallyExclusiveCallbackGroup) and we deadlock: planning succeeds but
        # the robot never moves. A shared ReentrantCallbackGroup + the
        # MultiThreadedExecutor lets those responses resolve on other threads.
        # (_cache_joints stays in the default group so joints keep updating while
        # the pipeline blocks.)
        self._pipeline_cbg = ReentrantCallbackGroup()

        self._task_sub = self.create_subscription(
            String, self.TASK_COMMANDS_TOPIC, self.task_command_callback, 10,
            callback_group=self._pipeline_cbg)
        self._joint_sub = self.create_subscription(
            JointState, self.JOINT_STATES_TOPIC, self._cache_joints, 10)
        self._verification_rgb_sub = self.create_subscription(
            Image,
            self._verification_rgb_topic,
            self._cache_verification_rgb,
            image_qos,
        )

        # Latched readiness from curobo_service: set once the planner has
        # initialised AND the TSDF map has frames. The orchestrator waits for
        # this before the first home move so it doesn't park the curobo executor
        # before the map is built (see _home_before_capture). A reentrant group
        # lets this callback fire while a task callback is blocked waiting on it.
        self._curobo_ready_event = threading.Event()
        ready_qos = QoSProfile(
            depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self._curobo_ready_sub = self.create_subscription(
            Bool, self.CUROBO_READY_TOPIC, self._on_curobo_ready, ready_qos,
            callback_group=ReentrantCallbackGroup())


        self._task_plan_pub = self.create_publisher(
            String,
            str(self.get_parameter('task_plan_topic').value),
            10,
        )
        self._task_verification_pub = self.create_publisher(
            String,
            str(self.get_parameter('verification_result_topic').value),
            10,
        )
        self._gemini = GeminiAPI(
            model=str(self.get_parameter('gemini_model').value),
            logger=self.get_logger(),
        )
        self._segmentation_client = self.create_client(
            StringString, self._segmentation_service_name,
            callback_group=self._pipeline_cbg)
        self._graspgen_client = self.create_client(
            StringString, self._graspgen_service_name,
            callback_group=self._pipeline_cbg)

        self._pipeline_busy = False
        self._active_task = ''
        self._active_task_data = None
        self._task_queue = deque()
        self._latest_segmentation = None
        self._latest_graspgen = None
        self._latest_joints = None
        self._holding_object = False
        self._latest_verification_rgb = None
        self._latest_verification_rgb_stamp_ns = 0
        self._verification_reference_stamp_ns = 0
        self._verification_deadline_monotonic = 0.0
        self._verification_timer = None

        self._curobo_client = None
        self._arm_client = None
        self._gripper_client = None
        if self._enable_motion_execution:
            if PlanTrajectory is None or ActionClient is None:
                raise ImportError(
                    'PlanTrajectory service and ROS2 actions are required '
                    'for motion execution.')
            self._curobo_client = self.create_client(
                PlanTrajectory, self._curobo_service_name,
                callback_group=self._pipeline_cbg)
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
            f'verification={self._enable_post_task_verification} '
            f'max_attempts={self._max_task_attempts}'
        )

    # ── ROS2 subscriptions ────────────────────────────────────────────────────

    def task_command_callback(self, msg: String) -> None:
        instruction = str(msg.data).strip()
        if not instruction:
            self.get_logger().warn('Ignoring empty task command.')
            return

        self.get_logger().info(f'Received task command: {instruction}')

        try:
            plan = self._gemini.parse_task_command(instruction)
        except (GeminiAPIError, ValueError) as exc:
            self.get_logger().error(f'Gemini task parsing failed: {exc}')
            return
        except Exception as exc:
            self.get_logger().error(
                f'Unexpected task parsing failure: {type(exc).__name__}: {exc}')
            return

        plan_json = json.dumps(plan, ensure_ascii=False)
        output = String()
        output.data = plan_json
        self._task_plan_pub.publish(output)
        self.get_logger().info(f'Published task plan: {plan_json}')

        if not self._auto_run_on_task_command:
            return

        for task in plan['tasks']:
            queued_task = dict(task)
            queued_task['_attempt_count'] = 0
            self._task_queue.append(queued_task)

        self.get_logger().info(
            f"Queued {len(plan['tasks'])} task(s); "
            f"total_pending={len(self._task_queue)}")
        self._start_next_task()

    def _start_next_task(self) -> None:
        if self._pipeline_busy or not self._task_queue:
            return

        task = self._task_queue.popleft()
        self._start_task_attempt(task)

    def _start_task_attempt(self, task: dict) -> None:
        if self._pipeline_busy:
            self.get_logger().warn(
                'Cannot start a task attempt while the pipeline is busy.')
            return

        task['_attempt_count'] = int(task.get('_attempt_count', 0)) + 1
        self._active_task_data = task

        object_name = str(task['object'])
        destination = str(task['destination'])
        attempt_count = int(task['_attempt_count'])

        if self._enable_motion_execution and destination == 'unspecified':
            self.get_logger().error(
                f"Cannot execute object={object_name}: "
                "destination is unspecified.")
            self._reset_pipeline_state(
                success=False, reason='destination is unspecified')
            return

        segmentation_prompt = self._build_segmentation_prompt(task)

        self.get_logger().info(
            f"Starting task attempt={attempt_count}/{self._max_task_attempts} "
            f"object={object_name} destination={destination} "
            f"remaining={len(self._task_queue)}")
        self.get_logger().info(
            f"Segmentation request prompt: {segmentation_prompt}")
        self._run_pipeline(segmentation_prompt)

    @staticmethod
    def _build_segmentation_prompt(task: dict) -> str:
        object_name = str(task['object']).replace('_', ' ').strip()
        return (
            f"Locate exactly one instance of the {object_name} in the image. "
            "Use the complete visible object as the target. "
            "Ignore the robot gripper, table, storage baskets, bookshelf, "
            "and every other object. If multiple candidates are visible, "
            "select the clearest instance that best matches the object name."
        )

    def _cache_joints(self, msg) -> None:
        self._latest_joints = msg

    def _on_curobo_ready(self, msg) -> None:
        if getattr(msg, 'data', False):
            if not self._curobo_ready_event.is_set():
                self.get_logger().info('cuRobo reported ready.')
            self._curobo_ready_event.set()

    def _cache_verification_rgb(self, msg: 'Image') -> None:
        try:
            image = self._bridge.imgmsg_to_cv2(
                msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().warn(
                f'Failed to decode verification RGB frame: {exc}')
            return

        self._latest_verification_rgb = image.copy()
        self._latest_verification_rgb_stamp_ns = _stamp_to_ns(
            msg.header.stamp)

    # ── pipeline entry ────────────────────────────────────────────────────────

    def _run_pipeline(self, task: str) -> None:
        task = task.strip()
        if not task:
            self.get_logger().warn('Ignoring empty task command.')
            self._reset_pipeline_state()
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
            self._retry_or_skip('segmentation service unavailable')
            return

        self._pipeline_busy = True
        self._active_task = task

        # Move to home first so the wrist camera observes the workspace, then
        # gate segmentation on a frame captured after the arm settled there.
        try:
            min_stamp_ns = self._home_before_capture()
        except Exception as exc:
            # The arm may be left partway home with the gripper open; that's a
            # safe idle state. Retry the task (which re-homes) while the attempt
            # budget allows, then skip — the home plan failure is often transient.
            self.get_logger().error(f'Initial home move failed: {exc}')
            self._retry_or_skip(f'initial home move failed: {exc}')
            return

        # Count target-type instances in the source workspace now (arm at home,
        # camera over the table) so post-task verification can confirm the count
        # dropped. Re-captured every attempt so retries compare against the
        # current scene. None (count unavailable) is stored as-is.
        if self._active_task_data is not None:
            object_name = str(self._active_task_data.get('object', ''))
            self._active_task_data['_before_count'] = (
                self._count_target_in_workspace(object_name, min_stamp_ns))

        request = StringString.Request()
        # min_stamp_ns == 0 (motion disabled / no joints yet) tells the
        # segmentation service to skip the freshness gate and use the latest frame.
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
            self._retry_or_skip(f'segmentation service error: {exc}')
            return

        if not payload.get('success'):
            self.get_logger().warn(f'Segmentation failed: {payload}')
            self._retry_or_skip(
                f"segmentation failed: {payload.get('error', payload)}")
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
            self._retry_or_skip('graspgen service unavailable')
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
            self._retry_or_skip(f'graspgen service error: {exc}')
            return

        if not payload.get('success'):
            self.get_logger().warn(
                f"GraspGen failed: {payload.get('error', payload)}")
            self._retry_or_skip(
                f"graspgen failed: {payload.get('error', payload)}")
            return

        self._latest_graspgen = payload
        top_grasps = payload.get('top_grasps') or []
        if not top_grasps:
            self.get_logger().warn(
                'GraspGen returned success but no ranked grasps.')
            self._retry_or_skip('graspgen returned no ranked grasps')
            return

        self.get_logger().info(
            'Pipeline result '
            f"label={self._latest_segmentation.get('label', 'target')} "
            f"best_translation={top_grasps[0].get('translation')} "
            f"confidence={top_grasps[0].get('confidence')} "
            f"num_candidates={len(top_grasps)} "
            f"destination={self._active_destination()}"
        )

        if not self._enable_motion_execution:
            self._reset_pipeline_state(
                success=True, reason='GraspGen completed; motion disabled')
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
            self._retry_or_skip('no valid grasp poses from GraspGen rows')
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
            self._retry_or_skip(f'pick plan service error: {exc}')
            return

        if not result.success:
            self.get_logger().warn(f'CuRobo pick planning failed: {result.message}')
            self._retry_or_skip(f'pick planning failed: {result.message}')
            return

        self.get_logger().info(result.message)

        # pick (open → approach+grasp → close → lift) → place → release → home.
        # Opening first is essential: the planned grasp pose assumes open
        # fingers, so a gripper left closed from a prior cycle would collide
        # with the object during the approach instead of enclosing it. The place
        # destination comes from the Gemini-parsed task (_active_destination).
        try:
            self._send_gripper(closed=False)
            self._send_and_wait(
                self._arm_client, result.trajectory, 'approach_and_grasp')
            self._send_gripper(closed=True)
            self._send_and_wait(
                self._arm_client, result.lift_trajectory, 'lift')
            self._plan_and_execute_place(self._active_destination())
            self._plan_and_execute_home()
        except Exception as exc:
            self.get_logger().error(f'Pick/place execution failed: {exc}')
            # Best-effort: try to open the gripper so we don't drop/drag things.
            try:
                self._send_gripper(closed=False)
            except Exception:
                pass
            self._reset_pipeline_state(
                success=False, reason=f'pick/place execution failed: {exc}')
            return

        # After place + home, optionally run Gemini post-task verification
        # (confirm the object actually left the pickup workspace, retrying the
        # task if not). When verification is disabled the task is marked done.
        if getattr(self, '_enable_post_task_verification', False):
            self._schedule_post_task_verification()
        else:
            self._reset_pipeline_state(
                success=True, reason='pick and place completed')

    def _call_curobo_blocking(self, request, label):
        """Call the CuRobo plan service and block for the response.

        Safe to block from inside the pipeline callback because the node is spun
        with a MultiThreadedExecutor AND the curobo client lives in a
        ReentrantCallbackGroup (self._pipeline_cbg) separate from the default
        group — so the executor can deliver the response on another thread while
        this callback is blocked. MultiThreadedExecutor alone is NOT enough: if
        the client shared the blocked callback's MutuallyExclusiveCallbackGroup,
        the response could never be delivered and this would deadlock.
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
        # Wait for cuRobo to report ready (init done + TSDF populated) before
        # issuing any motion. The curobo executor is single-threaded and builds
        # the TSDF on that same thread, so sending a plan before it's ready would
        # park the thread and starve the map (planning home cold against an empty
        # world). This only blocks on the first cycle; the signal is latched, so
        # later cycles return immediately. It also keeps the gripper/arm goals
        # below from being issued before the controllers are accepting them.
        if not self._curobo_ready_event.is_set():
            self.get_logger().info(
                'Waiting for cuRobo to report ready before the first home move '
                f'(up to {self._curobo_ready_wait_sec:.0f}s)...')
            if not self._curobo_ready_event.wait(self._curobo_ready_wait_sec):
                raise RuntimeError(
                    f'cuRobo did not report ready on {self.CUROBO_READY_TOPIC} '
                    f'within {self._curobo_ready_wait_sec:.0f}s.')
        # Plan + execute the home move, then open the gripper (controllers are
        # live by now). _plan_and_execute_home runs the collision-aware return.
        self._plan_and_execute_home()
        # Open the gripper for the upcoming grasp. In every normal/failure path the
        # gripper is already open by here, so this is idempotent insurance.
        self._send_gripper(closed=False)
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

    # ── post-task Gemini verification ────────────────────────────────────────

    def _schedule_post_task_verification(self) -> None:
        if self._active_task_data is None:
            self._reset_pipeline_state(
                success=False, reason='verification requested without active task')
            return

        self._verification_reference_stamp_ns = (
            self._latest_verification_rgb_stamp_ns)
        self._verification_deadline_monotonic = (
            time.monotonic()
            + self._verification_settle_sec
            + self._verification_frame_timeout_sec
        )

        self.get_logger().info(
            'Robot returned home; waiting for a fresh verification image ' 
            f'from {self._verification_rgb_topic}.')
        self._schedule_verification_timer(self._verification_settle_sec)

    def _schedule_verification_timer(self, delay_sec: float) -> None:
        self._cancel_verification_timer()

        def callback() -> None:
            timer = self._verification_timer
            self._verification_timer = None
            if timer is not None:
                timer.cancel()
            self._run_post_task_verification()

        # Must share the reentrant pipeline group: this callback runs the full
        # retry pipeline, whose _send_and_wait calls block on action
        # goal-response futures. In the default mutually-exclusive group (same
        # as the arm ActionClient) those callbacks can never be delivered while
        # this one blocks, deadlocking the home move (see _pipeline_cbg).
        self._verification_timer = self.create_timer(
            max(0.05, float(delay_sec)), callback,
            callback_group=self._pipeline_cbg)

    def _cancel_verification_timer(self) -> None:
        timer = self._verification_timer
        self._verification_timer = None
        if timer is not None:
            try:
                timer.cancel()
            except Exception:
                pass

    def _count_target_in_workspace(
        self, object_name: str, min_stamp_ns: int
    ) -> 'int | None':
        """Count target-type instances in the source workspace.

        Bounded-wait (``_verification_frame_timeout_sec``) for a verification RGB
        frame stamped after ``min_stamp_ns`` (``min_stamp_ns <= 0`` uses the
        latest frame), then ask Gemini to count. Returns the integer count, or
        ``None`` on timeout / decode / Gemini failure (logged, never raises) so
        callers can fall back to a no-retry success.
        """
        if PILImage is None:
            self.get_logger().warn(
                'Pillow unavailable; cannot count workspace objects.')
            return None

        deadline = time.monotonic() + self._verification_frame_timeout_sec
        frame = None
        while True:
            stamp_ns = self._latest_verification_rgb_stamp_ns
            candidate = self._latest_verification_rgb
            fresh = candidate is not None and (
                min_stamp_ns <= 0 or stamp_ns > min_stamp_ns)
            if fresh:
                frame = candidate
                break
            if time.monotonic() >= deadline:
                self.get_logger().warn(
                    'No fresh verification frame for object count '
                    f'(object={object_name}, min_stamp_ns={min_stamp_ns}).')
                return None
            time.sleep(0.05)

        image_rgb = frame[:, :, ::-1].copy()
        try:
            result = self._gemini.count_objects(
                PILImage.fromarray(image_rgb), object_name=object_name)
            count = int(result['count'])
        except Exception as exc:
            self.get_logger().warn(
                f'Object count failed (object={object_name}): {exc}')
            return None

        self.get_logger().info(
            f'Workspace count object={object_name} count={count} '
            f"reason={result.get('reason', '')}")
        return count

    def _run_post_task_verification(self) -> None:
        task = self._active_task_data
        if task is None:
            return

        has_image = self._latest_verification_rgb is not None
        has_fresh_image = (
            has_image
            and self._latest_verification_rgb_stamp_ns
            > self._verification_reference_stamp_ns
        )

        if not has_fresh_image:
            if time.monotonic() < self._verification_deadline_monotonic:
                self._schedule_verification_timer(0.2)
                return

            self.get_logger().error(
                'No fresh RGB frame arrived after the robot returned home; ' 
                'skipping verification and continuing to the next JSON task.')
            self._reset_pipeline_state(
                success=False,
                reason='verification image unavailable; skipped task',
            )
            return

        if PILImage is None:
            self._reset_pipeline_state(
                success=False,
                reason='Pillow unavailable for verification image',
            )
            return

        image_bgr = self._latest_verification_rgb.copy()
        image_rgb = image_bgr[:, :, ::-1].copy()
        pil_image = PILImage.fromarray(image_rgb)

        object_name = str(task['object'])
        destination = str(task['destination'])
        attempt_count = int(task.get('_attempt_count', 1))
        before_count = task.get('_before_count')

        try:
            count_result = self._gemini.count_objects(
                pil_image, object_name=object_name)
            after_count = int(count_result['count'])
            reason = str(count_result.get('reason', ''))
        except Exception as exc:
            after_count = None
            reason = f'count failed: {exc}'
            self.get_logger().warn(
                f'Post-task object count failed (object={object_name}): {exc}')

        removed = (
            before_count is not None
            and after_count is not None
            and after_count < before_count)

        verification_payload = {
            'count_available': after_count is not None,
            'object': object_name,
            'destination': destination,
            'attempt': attempt_count,
            'max_attempts': self._max_task_attempts,
            'before_count': before_count,
            'after_count': after_count,
            'removed': removed,
            'reason': reason,
        }
        self._publish_verification_result(
            task=task, result=verification_payload)
        self._save_verification_artifacts(
            image_rgb=image_rgb,
            task=task,
            result=verification_payload,
        )

        self.get_logger().info(
            'Post-task count result '
            f'object={object_name} attempt={attempt_count}/'
            f'{self._max_task_attempts} before={before_count} '
            f'after={after_count} removed={removed} reason={reason}')

        # Count unavailable (Gemini error / no fresh frame on either side): we
        # cannot tell if the pick worked. Succeed without retry — re-picking when
        # duplicates exist risks removing a second instance, which is worse than
        # a missed verification.
        if before_count is None or after_count is None:
            self.get_logger().warn(
                'Object count unavailable; marking task done without retry '
                f'(object={object_name}).')
            self._reset_pipeline_state(
                success=True,
                reason='object count unavailable; assumed removed',
            )
            return

        if removed:
            self._reset_pipeline_state(
                success=True,
                reason=(
                    f'count dropped {before_count}->{after_count}; removed'),
            )
            return

        if attempt_count < self._max_task_attempts:
            self.get_logger().warn(
                f"Count for '{object_name}' did not drop "
                f"({before_count}->{after_count}); retrying the same task "
                f"({attempt_count + 1}/{self._max_task_attempts}).")
            self._restart_active_task()
            return

        self.get_logger().warn(
            f"Count for '{object_name}' did not drop after "
            f"{attempt_count} attempts; skipping it and continuing to the "
            "next task.")
        self._reset_pipeline_state(
            success=False,
            reason=(
                f'count did not drop after {attempt_count} attempts; skipped'),
        )

    def _restart_active_task(self) -> None:
        task = self._active_task_data
        if task is None:
            self._reset_pipeline_state(
                success=False, reason='retry requested without active task')
            return

        self._cancel_verification_timer()
        self._pipeline_busy = False
        self._active_task = ''
        self._latest_segmentation = None
        self._latest_graspgen = None
        self._holding_object = False

        self._start_task_attempt(task)

    def _retry_or_skip(self, reason: str) -> None:
        """Handle a *pre-grasp* pipeline stage failure (segmentation, GraspGen,
        or pick planning).

        Re-attempt the same task in place while the attempt budget allows,
        otherwise log a clear skip and let the queue advance. Without this a
        transient failure (e.g. a Gemini 503 during segmentation) would drop the
        object permanently: every failure path used to call
        ``_reset_pipeline_state``, which pops the *next* task and never re-queues
        the failed one. The retry budget only ever covered post-execution
        verification, so a single transient hiccup silently skipped the object.

        Only safe before the gripper has closed on the object: it routes through
        ``_restart_active_task`` (which clears ``_holding_object``). Failures past
        the grasp must keep using ``_reset_pipeline_state`` so the held-object
        queue stop still fires.
        """
        task = self._active_task_data
        if task is not None:
            attempt_count = int(task.get('_attempt_count', 1))
            if attempt_count < self._max_task_attempts:
                self.get_logger().warn(
                    f"Task object={task.get('object')} "
                    f"destination={task.get('destination')} failed ({reason}); "
                    f"retrying ({attempt_count + 1}/{self._max_task_attempts}).")
                self._restart_active_task()
                return
            self.get_logger().error(
                f"Task object={task.get('object')} "
                f"destination={task.get('destination')} failed ({reason}) after "
                f"{attempt_count} attempt(s); skipping it and continuing to the "
                "next task.")
        self._reset_pipeline_state(success=False, reason=reason)

    def _publish_verification_result(
        self, *, task: dict, result: dict
    ) -> None:
        payload = {
            'object': str(task.get('object', '')),
            'destination': str(task.get('destination', '')),
            'attempt': int(task.get('_attempt_count', 0)),
            **result,
        }
        message = String()
        message.data = json.dumps(payload, ensure_ascii=False)
        self._task_verification_pub.publish(message)

    def _save_verification_artifacts(
        self, *, image_rgb, task: dict, result: dict
    ) -> None:
        try:
            stamp = time.strftime('%Y%m%d-%H%M%S')
            suffix = f"{time.time_ns() % 1_000_000_000:09d}"
            object_name = str(task.get('object', 'object'))
            attempt = int(task.get('_attempt_count', 0))
            output_dir = (
                self._verification_debug_dir
                / f'{stamp}-{suffix}-{object_name}-attempt{attempt}'
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            PILImage.fromarray(image_rgb).save(output_dir / 'verification_rgb.png')
            with open(
                output_dir / 'result.json', 'w', encoding='utf-8'
            ) as handle:
                json.dump(
                    {
                        'task': {
                            'object': task.get('object'),
                            'destination': task.get('destination'),
                            'attempt': attempt,
                        },
                        'result': result,
                    },
                    handle,
                    indent=2,
                    ensure_ascii=False,
                )
            self.get_logger().info(
                f'Saved verification artifacts: {output_dir}')
        except Exception as exc:
            self.get_logger().warn(
                f'Failed to save verification artifacts: {exc}')

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
            raise RuntimeError(f'{label}: empty trajectory')
        if client is None:
            raise RuntimeError(f'{label}: action client unavailable')
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

    def _send_gripper(self, closed: bool, *, strict: bool = False) -> None:
        """Send open/close and track whether the robot is holding an object."""
        if self._gripper_client is None:
            message = '_send_gripper: gripper action client unavailable.'
            if strict:
                raise RuntimeError(message)
            self.get_logger().warn(message)
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
            self._holding_object = bool(closed)
        except Exception as exc:
            if strict:
                raise
            self.get_logger().warn(f'{label} failed (non-fatal): {exc}')

    def _active_destination(self) -> str:
        if not self._active_task_data:
            return 'unspecified'
        return str(self._active_task_data.get('destination', 'unspecified'))

    def _reset_pipeline_state(
        self, *, success: bool = False, reason: str = ''
    ) -> None:
        finished = self._active_task_data
        self._cancel_verification_timer()
        self._verification_reference_stamp_ns = 0
        self._verification_deadline_monotonic = 0.0
        self._pipeline_busy = False
        self._active_task = ''
        self._active_task_data = None

        if finished is not None:
            # rclpy caches log severity per caller location, so info and error
            # must live on separate physical lines: aliasing them through one
            # call site raises ValueError('Logger severity cannot be changed
            # between calls.') the moment the queue mixes a failed task with a
            # successful one, killing the node.
            message = (
                f"Task {'completed' if success else 'failed'} "
                f"object={finished.get('object')} "
                f"destination={finished.get('destination')} "
                f"reason={reason or 'none'}")
            if success:
                self.get_logger().info(message)
            else:
                self.get_logger().error(message)

        if not success and self._holding_object:
            pending = len(self._task_queue)
            self._task_queue.clear()
            self.get_logger().error(
                'Stopping the queue because the gripper may still hold an object; '
                f'cleared {pending} pending task(s).')
            return

        if self._auto_run_on_task_command:
            self._start_next_task()



def _stamp_to_ns(stamp) -> int:
    return (
        int(getattr(stamp, 'sec', 0)) * 1_000_000_000
        + int(getattr(stamp, 'nanosec', 0))
    )

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