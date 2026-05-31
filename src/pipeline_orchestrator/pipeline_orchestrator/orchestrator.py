import json
import math

try:  # pragma: no cover - runtime dependency
    import rclpy
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String
    from std_srvs.srv import Trigger
    from control_msgs.action import FollowJointTrajectory
    from geometry_msgs.msg import Pose
except ImportError:  # pragma: no cover - import-only test fallback
    rclpy = None
    ActionClient = None
    JointState = None
    FollowJointTrajectory = None
    Pose = None

    class Node:  # type: ignore[override]
        pass

    class String:  # type: ignore[override]
        pass

    class Trigger:  # type: ignore[override]
        class Request:
            pass

try:  # pragma: no cover - runtime dependency
    from riro_srvs.srv import StringString
    from riro_srvs.srv import PlanTrajectory
except ImportError:  # pragma: no cover - runtime dependency
    StringString = None
    PlanTrajectory = None


class PipelineOrchestrator(Node):
    """ROS2 orchestrator for segmentation, GraspGen, and optional motion execution."""

    TASK_COMMANDS_TOPIC = "/task_commands"
    JOINT_STATES_TOPIC = "/joint_states"

    def __init__(self):
        if rclpy is None:
            raise ImportError("rclpy is required for pipeline_orchestrator runtime.")
        if StringString is None:
            raise ImportError("riro_srvs is required for pipeline_orchestrator.")

        super().__init__("pipeline_orchestrator")

        self.declare_parameter("segmentation_service_name", "/segmentation/segment_prompt")
        self.declare_parameter("graspgen_service_name", "/graspgen/infer")
        self.declare_parameter("auto_run_on_task_command", True)
        self.declare_parameter("segmentation_service_wait_sec", 10.0)
        self.declare_parameter("graspgen_service_wait_sec", 10.0)
        self.declare_parameter("curobo_service_name", "/curobo/plan_trajectory")
        self.declare_parameter("curobo_service_wait_sec", 30.0)
        self.declare_parameter("enable_motion_execution", False)
        self.declare_parameter("arm_action_name", "/ur5_controller/follow_joint_trajectory")

        self._segmentation_service_name = str(
            self.get_parameter("segmentation_service_name").value
        )
        self._graspgen_service_name = str(self.get_parameter("graspgen_service_name").value)
        self._auto_run_on_task_command = bool(
            self.get_parameter("auto_run_on_task_command").value
        )
        self._segmentation_service_wait_sec = float(
            self.get_parameter("segmentation_service_wait_sec").value
        )
        self._graspgen_service_wait_sec = float(
            self.get_parameter("graspgen_service_wait_sec").value
        )
        self._curobo_service_name = str(self.get_parameter("curobo_service_name").value)
        self._curobo_service_wait_sec = float(
            self.get_parameter("curobo_service_wait_sec").value
        )
        self._enable_motion_execution = bool(
            self.get_parameter("enable_motion_execution").value
        )

        self._task_sub = self.create_subscription(
            String, self.TASK_COMMANDS_TOPIC, self.task_command_callback, 10
        )
        self._joint_sub = self.create_subscription(
            JointState, self.JOINT_STATES_TOPIC, self._cache_joints, 10
        )
        self._segmentation_client = self.create_client(
            StringString, self._segmentation_service_name
        )
        self._graspgen_client = self.create_client(Trigger, self._graspgen_service_name)

        self._pipeline_busy = False
        self._active_task = ""
        self._latest_segmentation = None
        self._latest_graspgen = None
        self._latest_joints = None

        self._curobo_client = None
        self._arm_client = None
        if self._enable_motion_execution:
            if PlanTrajectory is None or ActionClient is None:
                raise ImportError("PlanTrajectory service and ROS2 actions are required for motion execution.")
            self._curobo_client = self.create_client(
                PlanTrajectory,
                self._curobo_service_name,
            )
            self._arm_client = ActionClient(
                self,
                FollowJointTrajectory,
                str(self.get_parameter("arm_action_name").value),
            )

        self.get_logger().info(
            "pipeline_orchestrator ready "
            f"segmentation={self._segmentation_service_name} "
            f"graspgen={self._graspgen_service_name} "
            f"curobo={self._curobo_service_name} "
            f"motion_execution={self._enable_motion_execution}"
        )

    def task_command_callback(self, msg: String) -> None:
        self.get_logger().info(f"Received task command: {msg.data}")
        if self._auto_run_on_task_command:
            self._run_pipeline(msg.data)

    def _cache_joints(self, msg) -> None:
        self._latest_joints = msg

    def _run_pipeline(self, task: str) -> None:
        task = task.strip()
        if not task:
            self.get_logger().warn("Ignoring empty task command.")
            return
        if self._pipeline_busy:
            self.get_logger().warn(
                f"Pipeline busy with '{self._active_task}'. Ignoring new task '{task}'."
            )
            return
        if not self._segmentation_client.wait_for_service(
            timeout_sec=self._segmentation_service_wait_sec
        ):
            self.get_logger().warn(
                "Segmentation service unavailable: "
                f"{self._segmentation_service_name} "
                f"(waited {self._segmentation_service_wait_sec:.1f}s)"
            )
            return

        self._pipeline_busy = True
        self._active_task = task

        request = StringString.Request()
        request.data = task
        future = self._segmentation_client.call_async(request)
        future.add_done_callback(self._on_segmentation_done)
        self.get_logger().info(f"Started segmentation for task: {task}")

    def _on_segmentation_done(self, future) -> None:
        try:
            result = future.result()
            payload = json.loads(result.data)
        except Exception as exc:
            self.get_logger().error(f"Segmentation service call failed: {exc}")
            self._reset_pipeline_state()
            return

        if not payload.get("success"):
            self.get_logger().warn(f"Segmentation failed: {payload}")
            self._reset_pipeline_state()
            return

        self._latest_segmentation = payload
        label = payload.get("label", "target")
        point_count = payload.get("object_point_count", 0)
        self.get_logger().info(
            f"Segmentation ready label={label} points={point_count}; requesting GraspGen."
        )

        if not self._graspgen_client.wait_for_service(timeout_sec=self._graspgen_service_wait_sec):
            self.get_logger().warn(
                "GraspGen service unavailable: "
                f"{self._graspgen_service_name} "
                f"(waited {self._graspgen_service_wait_sec:.1f}s)"
            )
            self._reset_pipeline_state()
            return

        future = self._graspgen_client.call_async(Trigger.Request())
        future.add_done_callback(self._on_graspgen_done)

    def _on_graspgen_done(self, future) -> None:
        try:
            result = future.result()
        except Exception as exc:
            self.get_logger().error(f"GraspGen service call failed: {exc}")
            self._reset_pipeline_state()
            return

        if not result.success:
            self.get_logger().warn(f"GraspGen failed: {result.message}")
            self._reset_pipeline_state()
            return

        try:
            payload = json.loads(result.message)
        except json.JSONDecodeError:
            self.get_logger().warn(f"GraspGen returned non-JSON payload: {result.message}")
            self._reset_pipeline_state()
            return

        self._latest_graspgen = payload
        top_grasps = payload.get("top_grasps") or []
        if not top_grasps:
            self.get_logger().warn("GraspGen returned success but no ranked grasps.")
            self._reset_pipeline_state()
            return

        top = top_grasps[0]
        self.get_logger().info(
            "Pipeline result "
            f"label={self._latest_segmentation.get('label', 'target')} "
            f"centroid={self._latest_segmentation.get('centroid')} "
            f"best_translation={top.get('translation')} "
            f"confidence={top.get('confidence')}"
        )

        if self._enable_motion_execution:
            self._plan_and_execute_best_grasp(top)
            return
        self._reset_pipeline_state()

    def _plan_and_execute_best_grasp(self, top_grasp: dict) -> None:
        if self._latest_joints is None:
            self.get_logger().warn("No /joint_states received; skipping motion execution.")
            self._reset_pipeline_state()
            return
        if self._curobo_client is None:
            self.get_logger().warn("Motion execution enabled but CuRobo service client is unavailable.")
            self._reset_pipeline_state()
            return

        grasp_pose = self._pose_from_grasp_row(top_grasp)
        if grasp_pose is None:
            self.get_logger().warn(f"Cannot build pose from GraspGen row: {top_grasp}")
            self._reset_pipeline_state()
            return

        if not self._curobo_client.wait_for_service(timeout_sec=self._curobo_service_wait_sec):
            self.get_logger().warn(
                "CuRobo service unavailable: "
                f"{self._curobo_service_name} "
                f"(waited {self._curobo_service_wait_sec:.1f}s)"
            )
            self._reset_pipeline_state()
            return

        request = PlanTrajectory.Request()
        request.grasp_pose = grasp_pose
        request.joint_state = self._latest_joints
        future = self._curobo_client.call_async(request)
        future.add_done_callback(self._on_curobo_done)
        self.get_logger().info("Requested CuRobo trajectory plan.")

    def _on_curobo_done(self, future) -> None:
        try:
            result = future.result()
        except Exception as exc:
            self.get_logger().error(f"CuRobo service call failed: {exc}")
            self._reset_pipeline_state()
            return

        if not result.success:
            self.get_logger().warn(f"CuRobo planning failed: {result.message}")
            self._reset_pipeline_state()
            return

        self.get_logger().info(result.message)
        self._execute_trajectory(result.trajectory)
        self._reset_pipeline_state()

    def _execute_trajectory(self, trajectory, timeout_sec: float = 2.0):
        """Deploy a planned JointTrajectory to the UR5 arm controller."""
        if trajectory is None or not trajectory.points:
            self.get_logger().warning("refusing to execute empty trajectory.")
            return None
        if self._arm_client is None:
            self.get_logger().warning("arm action client is unavailable.")
            return None
        if not self._arm_client.wait_for_server(timeout_sec=timeout_sec):
            self.get_logger().error(
                "arm action server /ur5_controller/follow_joint_trajectory unavailable."
            )
            return None
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory
        self.get_logger().info(f"deploying {len(trajectory.points)}-point trajectory.")
        return self._arm_client.send_goal_async(goal)

    @staticmethod
    def _pose_from_grasp_row(row: dict):
        if Pose is None:
            return None
        translation = row.get("translation")
        rotation = row.get("rotation_matrix")
        if translation is None or rotation is None:
            return None
        if len(translation) != 3 or len(rotation) != 3:
            return None

        quat = PipelineOrchestrator._quat_from_rotation_matrix(rotation)
        pose = Pose()
        pose.position.x = float(translation[0])
        pose.position.y = float(translation[1])
        pose.position.z = float(translation[2])
        pose.orientation.w = quat[0]
        pose.orientation.x = quat[1]
        pose.orientation.y = quat[2]
        pose.orientation.z = quat[3]
        return pose

    @staticmethod
    def _quat_from_rotation_matrix(rotation) -> tuple[float, float, float, float]:
        r00, r01, r02 = [float(v) for v in rotation[0]]
        r10, r11, r12 = [float(v) for v in rotation[1]]
        r20, r21, r22 = [float(v) for v in rotation[2]]
        trace = r00 + r11 + r22
        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            return (0.25 * s, (r21 - r12) / s, (r02 - r20) / s, (r10 - r01) / s)
        if r00 > r11 and r00 > r22:
            s = math.sqrt(1.0 + r00 - r11 - r22) * 2.0
            return ((r21 - r12) / s, 0.25 * s, (r01 + r10) / s, (r02 + r20) / s)
        if r11 > r22:
            s = math.sqrt(1.0 + r11 - r00 - r22) * 2.0
            return ((r02 - r20) / s, (r01 + r10) / s, 0.25 * s, (r12 + r21) / s)
        s = math.sqrt(1.0 + r22 - r00 - r11) * 2.0
        return ((r10 - r01) / s, (r02 + r20) / s, (r12 + r21) / s, 0.25 * s)

    def _reset_pipeline_state(self) -> None:
        self._pipeline_busy = False
        self._active_task = ""


def main(args=None) -> None:
    if rclpy is None:
        raise ImportError("rclpy is required for pipeline_orchestrator runtime.")
    rclpy.init(args=args)
    node = PipelineOrchestrator()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
