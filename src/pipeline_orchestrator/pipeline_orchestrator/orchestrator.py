import json

try:  # pragma: no cover - runtime dependency
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    from std_srvs.srv import Trigger
except ImportError:  # pragma: no cover - import-only test fallback
    rclpy = None

    class Node:  # type: ignore[override]
        pass

    class String:  # type: ignore[override]
        pass

    class Trigger:  # type: ignore[override]
        class Request:
            pass

try:  # pragma: no cover - runtime dependency
    from riro_srvs.srv import StringString
except ImportError:  # pragma: no cover - runtime dependency
    StringString = None


class PipelineOrchestrator(Node):
    """ROS2 orchestrator that chains segmentation and GraspGen service calls."""

    TASK_COMMANDS_TOPIC = "/task_commands"

    def __init__(self):
        if rclpy is None:
            raise ImportError("rclpy is required for pipeline_orchestrator runtime.")
        if StringString is None:
            raise ImportError("riro_srvs is required for pipeline_orchestrator.")

        super().__init__("pipeline_orchestrator")

        self.declare_parameter("segmentation_service_name", "/segmentation/segment_prompt")
        self.declare_parameter("graspgen_service_name", "/graspgen/infer")
        self.declare_parameter("auto_run_on_task_command", True)

        self._segmentation_service_name = str(
            self.get_parameter("segmentation_service_name").value
        )
        self._graspgen_service_name = str(self.get_parameter("graspgen_service_name").value)
        self._auto_run_on_task_command = bool(
            self.get_parameter("auto_run_on_task_command").value
        )

        self._task_sub = self.create_subscription(
            String, self.TASK_COMMANDS_TOPIC, self.task_command_callback, 10
        )
        self._segmentation_client = self.create_client(
            StringString, self._segmentation_service_name
        )
        self._graspgen_client = self.create_client(Trigger, self._graspgen_service_name)

        self._pipeline_busy = False
        self._active_task = ""
        self._latest_segmentation = None
        self._latest_graspgen = None

        self.get_logger().info(
            "pipeline_orchestrator ready "
            f"segmentation={self._segmentation_service_name} "
            f"graspgen={self._graspgen_service_name}"
        )

    def task_command_callback(self, msg: String) -> None:
        self.get_logger().info(f"Received task command: {msg.data}")
        if self._auto_run_on_task_command:
            self._run_pipeline(msg.data)

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
        if not self._segmentation_client.wait_for_service(timeout_sec=0.5):
            self.get_logger().warn(
                f"Segmentation service unavailable: {self._segmentation_service_name}"
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

        if not self._graspgen_client.wait_for_service(timeout_sec=0.5):
            self.get_logger().warn(f"GraspGen service unavailable: {self._graspgen_service_name}")
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
        if top_grasps:
            top = top_grasps[0]
            self.get_logger().info(
                "Pipeline result "
                f"label={self._latest_segmentation.get('label', 'target')} "
                f"centroid={self._latest_segmentation.get('centroid')} "
                f"best_translation={top.get('translation')} "
                f"confidence={top.get('confidence')}"
            )
        else:
            self.get_logger().warn("GraspGen returned success but no ranked grasps.")

        # Motion execution stays separate until cuRobo / MoveIt2 integration is ready.
        self._reset_pipeline_state()

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
