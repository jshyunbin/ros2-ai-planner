import traceback
import threading

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from pipeline_orchestrator.curobo import CuRobo
from riro_srvs.srv import PlanTrajectory


class CuRoboService(Node):
    """ROS2 service wrapper around the long-lived CuRobo planner instance."""

    JOINT_STATES_TOPIC = "/joint_states"

    def __init__(self) -> None:
        super().__init__("curobo_service")

        self.declare_parameter("service_name", "/curobo/plan_trajectory")
        self.declare_parameter("enable_viz", False)

        self._latest_joints = None
        self._curobo = None
        self._init_error = ""
        self._init_lock = threading.Lock()
        self.create_subscription(
            JointState,
            self.JOINT_STATES_TOPIC,
            self._cache_joints,
            10,
        )

        service_name = str(self.get_parameter("service_name").value)
        self.create_service(PlanTrajectory, service_name, self._handle_plan)
        self.get_logger().info(
            f"curobo_service advertised service={service_name}; initializing CuRobo in background."
        )
        self._init_thread = threading.Thread(
            target=self._initialize_curobo,
            name="curobo_initializer",
            daemon=True,
        )
        self._init_thread.start()

    @staticmethod
    def _as_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)

    def _initialize_curobo(self) -> None:
        self.get_logger().info("CuRobo initialization started.")
        try:
            curobo = CuRobo(
                self,
                enable_viz=self._as_bool(self.get_parameter("enable_viz").value),
            )
        except Exception as exc:
            with self._init_lock:
                self._init_error = f"{type(exc).__name__}: {exc}"
            self.get_logger().error(f"CuRobo initialization failed: {self._init_error}")
            self.get_logger().debug(traceback.format_exc())
            return

        with self._init_lock:
            self._curobo = curobo
            latest_joints = self._latest_joints
        if latest_joints is not None:
            self._curobo.update_joint_state(latest_joints)
        self.get_logger().info("CuRobo initialization complete; planner service is ready.")

    def _cache_joints(self, msg: JointState) -> None:
        self._latest_joints = msg
        with self._init_lock:
            curobo = self._curobo
        if curobo is not None:
            curobo.update_joint_state(msg)

    def _handle_plan(
        self,
        request: PlanTrajectory.Request,
        response: PlanTrajectory.Response,
    ) -> PlanTrajectory.Response:
        with self._init_lock:
            curobo = self._curobo
            init_error = self._init_error
        if init_error:
            response.success = False
            response.message = f"CuRobo initialization failed: {init_error}"
            return response
        if curobo is None:
            response.success = False
            response.message = "CuRobo is still initializing."
            return response

        joint_state = request.joint_state
        if not joint_state.name:
            joint_state = self._latest_joints
        if joint_state is None or not joint_state.name:
            response.success = False
            response.message = "No joint state supplied and no /joint_states message has been received."
            return response

        try:
            curobo.update_joint_state(joint_state)
            trajectory = curobo.plan_trajectory(request.grasp_pose, joint_state)
        except Exception as exc:
            response.success = False
            response.message = f"CuRobo planning exception: {type(exc).__name__}: {exc}"
            self.get_logger().error(response.message)
            self.get_logger().debug(traceback.format_exc())
            return response

        if trajectory is None or not trajectory.points:
            response.success = False
            response.message = "CuRobo did not produce an executable trajectory."
            return response

        response.success = True
        response.message = f"CuRobo planned {len(trajectory.points)} trajectory points."
        response.trajectory = trajectory
        return response


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CuRoboService()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
