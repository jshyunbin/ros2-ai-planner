import traceback

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
        self._curobo = CuRobo(
            self,
            enable_viz=bool(self.get_parameter("enable_viz").value),
        )
        self.create_subscription(
            JointState,
            self.JOINT_STATES_TOPIC,
            self._cache_joints,
            10,
        )

        service_name = str(self.get_parameter("service_name").value)
        self.create_service(PlanTrajectory, service_name, self._handle_plan)
        self.get_logger().info(f"curobo_service ready service={service_name}")

    def _cache_joints(self, msg: JointState) -> None:
        self._latest_joints = msg
        self._curobo.update_joint_state(msg)

    def _handle_plan(
        self,
        request: PlanTrajectory.Request,
        response: PlanTrajectory.Response,
    ) -> PlanTrajectory.Response:
        joint_state = request.joint_state
        if not joint_state.name:
            joint_state = self._latest_joints
        if joint_state is None or not joint_state.name:
            response.success = False
            response.message = "No joint state supplied and no /joint_states message has been received."
            return response

        try:
            self._curobo.update_joint_state(joint_state)
            trajectory = self._curobo.plan_trajectory(request.grasp_pose, joint_state)
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
