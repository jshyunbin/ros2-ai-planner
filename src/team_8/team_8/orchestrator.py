"""Pipeline orchestrator: task_commands → segmentation → GraspGen → CuRobo → execution.

Pick execution follows yeina's 3-phase approach:
  1. Send approach-and-grasp trajectory (arm moves to grasp contact)
  2. Close gripper
  3. Send lift trajectory (arm lifts with object)

All execution calls are blocking (_send_and_wait); the node is spun with
MultiThreadedExecutor so spin_until_future_complete works inside callbacks.
"""

import copy
import json
import math
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
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image, JointState
    from std_msgs.msg import String
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
    ReliabilityPolicy = None
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
from team_8.place_pose_utils import (
    BOOKSHELF_TARGETS,
    load_place_poses,
    resolve_target_pose,
)
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
        self.declare_parameter('max_task_attempts', 2)
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

        self._bridge = CvBridge()
        image_qos = QoSProfile(depth=10)
        image_qos.reliability = ReliabilityPolicy.BEST_EFFORT

        self._task_sub = self.create_subscription(
            String, self.TASK_COMMANDS_TOPIC, self.task_command_callback, 10)
        self._joint_sub = self.create_subscription(
            JointState, self.JOINT_STATES_TOPIC, self._cache_joints, 10)
        self._verification_rgb_sub = self.create_subscription(
            Image,
            self._verification_rgb_topic,
            self._cache_verification_rgb,
            image_qos,
        )
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
            StringString, self._segmentation_service_name)
        self._graspgen_client = self.create_client(
            StringString, self._graspgen_service_name)

        self._pipeline_busy = False
        self._active_task = ''
        self._active_task_data = None
        self._task_queue = deque()
        self._motion_steps = deque()
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
            self._reset_pipeline_state()
            return

        self._pipeline_busy = True
        self._active_task = task

        request = StringString.Request()
        request.data = task
        future = self._segmentation_client.call_async(request)
        future.add_done_callback(self._on_segmentation_done)
        self.get_logger().info(f'Started segmentation for task: {task}')

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
            self._reset_pipeline_state(success=False, reason=str(exc))
            return

        if not result.success:
            self.get_logger().warn(f'CuRobo pick planning failed: {result.message}')
            self._reset_pipeline_state(
                success=False, reason=f'pick planning failed: {result.message}')
            return

        self.get_logger().info(result.message)

        try:
            self._send_gripper(closed=False, strict=True)
            self._execute_arm_trajectory(
                result.trajectory, 'approach_and_grasp')
            self._send_gripper(closed=True, strict=True)
            self._execute_arm_trajectory(
                result.lift_trajectory, 'lift')
        except Exception as exc:
            self.get_logger().error(f'Pick execution failed: {exc}')
            try:
                self._send_gripper(closed=False, strict=False)
            except Exception:
                pass
            self._reset_pipeline_state(
                success=False, reason=f'pick execution failed: {exc}')
            return

        self._start_place_sequence()


    # ── place sequence ─────────────────────────────────────────────────────────

    def _start_place_sequence(self) -> None:
        destination = self._active_destination()

        try:
            destination_pose = resolve_target_pose(
                self._place_poses, destination)
            home_pose = resolve_target_pose(self._place_poses, 'home')
        except Exception as exc:
            self.get_logger().error(
                f"Cannot resolve destination '{destination}': {exc}")
            self._reset_pipeline_state(
                success=False, reason=f'invalid destination: {exc}')
            return

        if destination_pose is None or home_pose is None:
            self._reset_pipeline_state(
                success=False, reason='failed to construct place/home Pose')
            return

        self._motion_steps.clear()

        transit_pose = copy.deepcopy(destination_pose)
        transit_pose.position.z = float(self._place_poses['transit_z'])
        self._motion_steps.append({
            'label': f'transit_to_{destination}',
            'pose': transit_pose,
            'release_after': False,
        })

        if destination in BOOKSHELF_TARGETS:
            shelf = self._place_poses[destination]
            pre_insert_pose = destination_pose
            inserted_pose = _offset_pose_along_local_z(
                pre_insert_pose, float(shelf['insert_depth_m']))
            retract_pose = _offset_pose_along_local_z(
                inserted_pose, -float(shelf['retract_depth_m']))

            self._motion_steps.append({
                'label': f'{destination}_pre_insert',
                'pose': pre_insert_pose,
                'release_after': False,
            })
            self._motion_steps.append({
                'label': f'{destination}_insert',
                'pose': inserted_pose,
                'release_after': True,
            })
            self._motion_steps.append({
                'label': f'{destination}_retract',
                'pose': retract_pose,
                'release_after': False,
            })
        else:
            self._motion_steps.append({
                'label': f'place_{destination}',
                'pose': destination_pose,
                'release_after': True,
            })

        if (
            self._return_home_after_place
            or self._enable_post_task_verification
        ):
            self._motion_steps.append({
                'label': 'return_home',
                'pose': home_pose,
                'release_after': False,
            })

        self.get_logger().info(
            f"Built place sequence destination={destination} "
            f"steps={[step['label'] for step in self._motion_steps]}")
        self._request_next_motion_step()

    def _request_next_motion_step(self) -> None:
        if not self._motion_steps:
            if self._enable_post_task_verification:
                self._schedule_post_task_verification()
            else:
                self._reset_pipeline_state(
                    success=True, reason='pick and place completed')
            return

        if self._latest_joints is None:
            self._reset_pipeline_state(
                success=False, reason='joint state unavailable before place')
            return

        if self._curobo_client is None:
            self._reset_pipeline_state(
                success=False, reason='CuRobo client unavailable before place')
            return

        step = self._motion_steps.popleft()
        request = PlanTrajectory.Request()
        request.grasp_poses = []
        request.grasp_pose = step['pose']
        request.joint_state = self._latest_joints

        future = self._curobo_client.call_async(request)
        future.add_done_callback(
            lambda done_future, current_step=step:
            self._on_motion_step_planned(done_future, current_step)
        )
        self.get_logger().info(
            f"Requested CuRobo single-pose plan: {step['label']}")

    def _on_motion_step_planned(self, future, step: dict) -> None:
        label = str(step['label'])
        try:
            result = future.result()
        except Exception as exc:
            self.get_logger().error(
                f"CuRobo place service call failed at {label}: {exc}")
            self._reset_pipeline_state(
                success=False, reason=f'{label} service failure: {exc}')
            return

        if not result.success:
            self.get_logger().error(
                f"CuRobo place planning failed at {label}: {result.message}")
            self._reset_pipeline_state(
                success=False, reason=f'{label} planning failed')
            return

        try:
            self._execute_arm_trajectory(result.trajectory, label)
            if bool(step.get('release_after')):
                self.get_logger().info(
                    f"Reached release pose for {self._active_destination()}; "
                    "opening gripper.")
                self._send_gripper(closed=False, strict=True)
        except Exception as exc:
            self.get_logger().error(
                f"Motion step execution failed at {label}: {exc}")
            self._reset_pipeline_state(
                success=False, reason=f'{label} execution failed: {exc}')
            return

        self._request_next_motion_step()

    def _execute_arm_trajectory(
        self, trajectory: 'JointTrajectory', label: str
    ) -> None:
        self._send_and_wait(self._arm_client, trajectory, label)
        self._update_joint_state_from_trajectory(trajectory)

    def _update_joint_state_from_trajectory(
        self, trajectory: 'JointTrajectory'
    ) -> None:
        points = list(getattr(trajectory, 'points', []) or [])
        joint_names = list(getattr(trajectory, 'joint_names', []) or [])
        if not points or not joint_names:
            return

        final_positions = list(getattr(points[-1], 'positions', []) or [])
        if len(final_positions) != len(joint_names):
            return

        state = JointState()
        state.header.stamp = self.get_clock().now().to_msg()
        state.name = joint_names
        state.position = [float(value) for value in final_positions]
        self._latest_joints = state

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

        self._verification_timer = self.create_timer(
            max(0.05, float(delay_sec)), callback)

    def _cancel_verification_timer(self) -> None:
        timer = self._verification_timer
        self._verification_timer = None
        if timer is not None:
            try:
                timer.cancel()
            except Exception:
                pass

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

        try:
            result = self._gemini.verify_object_removed(
                pil_image,
                object_name=object_name,
                destination=destination,
            )
        except Exception as exc:
            self.get_logger().error(
                f'Gemini post-task verification failed: {exc}; ' 
                'continuing to the next JSON task.')
            self._publish_verification_result(
                task=task,
                result={
                    'verification_success': False,
                    'error': str(exc),
                },
            )
            self._save_verification_artifacts(
                image_rgb=image_rgb,
                task=task,
                result={'error': str(exc)},
            )
            self._reset_pipeline_state(
                success=False,
                reason='Gemini verification failed; skipped task',
            )
            return

        present = bool(result['present_in_source_workspace'])
        confidence = float(result.get('confidence', 0.0))
        reason = str(result.get('reason', ''))

        verification_payload = {
            'verification_success': True,
            'object': object_name,
            'destination': destination,
            'attempt': attempt_count,
            'max_attempts': self._max_task_attempts,
            **result,
        }
        self._publish_verification_result(
            task=task, result=verification_payload)
        self._save_verification_artifacts(
            image_rgb=image_rgb,
            task=task,
            result=verification_payload,
        )

        self.get_logger().info(
            'Gemini verification result ' 
            f'object={object_name} attempt={attempt_count}/'
            f'{self._max_task_attempts} ' 
            f'present_in_source_workspace={present} ' 
            f'confidence={confidence:.3f} reason={reason}')

        if not present:
            self._reset_pipeline_state(
                success=True,
                reason='verified removed from source workspace',
            )
            return

        if attempt_count < self._max_task_attempts:
            self.get_logger().warn(
                f"Object '{object_name}' is still visible in the source "
                f"workspace; retrying the same task "
                f"({attempt_count + 1}/{self._max_task_attempts}).")
            self._restart_active_task()
            return

        self.get_logger().warn(
            f"Object '{object_name}' is still visible after "
            f"{attempt_count} attempts; skipping it and continuing to the "
            "next JSON task.")
        self._reset_pipeline_state(
            success=False,
            reason=(
                f'object still present after {attempt_count} attempts; skipped'
            ),
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
        self._motion_steps.clear()
        self._latest_segmentation = None
        self._latest_graspgen = None
        self._holding_object = False

        self._start_task_attempt(task)

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
        self._motion_steps.clear()

        if finished is not None:
            level = self.get_logger().info if success else self.get_logger().error
            level(
                f"Task {'completed' if success else 'failed'} "
                f"object={finished.get('object')} "
                f"destination={finished.get('destination')} "
                f"reason={reason or 'none'}")

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

def _offset_pose_along_local_z(pose: 'Pose', distance_m: float) -> 'Pose':
    """Copy *pose* and translate it along the pose's local +Z axis."""
    out = copy.deepcopy(pose)

    x = float(pose.orientation.x)
    y = float(pose.orientation.y)
    z = float(pose.orientation.z)
    w = float(pose.orientation.w)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-9:
        raise ValueError('Cannot offset pose with a zero quaternion.')
    x, y, z, w = x / norm, y / norm, z / norm, w / norm

    # Third column of the quaternion rotation matrix: local +Z in world frame.
    axis_x = 2.0 * (x * z + y * w)
    axis_y = 2.0 * (y * z - x * w)
    axis_z = 1.0 - 2.0 * (x * x + y * y)

    out.position.x += float(distance_m) * axis_x
    out.position.y += float(distance_m) * axis_y
    out.position.z += float(distance_m) * axis_z
    return out

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