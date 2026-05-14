import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger


class CuRoboNode(Node):
    """Stub node for cuRobo motion planning.

    Exposes service ~/plan_trajectory.
    TODO: Replace Trigger with a custom service type that accepts
          (geometry_msgs/PoseStamped grasp_pose, sensor_msgs/JointState joint_states)
          and returns trajectory_msgs/JointTrajectory.
    """

    def __init__(self):
        super().__init__('curobo_node')
        self.srv = self.create_service(
            Trigger,
            '~/plan_trajectory',
            self.plan_trajectory_callback,
        )
        self.get_logger().info('curobo_node ready (stub).')

    def plan_trajectory_callback(self, request, response):
        # TODO: run cuRobo motion planner
        # Inputs:  request.grasp_pose   (geometry_msgs/PoseStamped)
        #          request.joint_states (sensor_msgs/JointState)
        # Outputs: response.trajectory  (trajectory_msgs/JointTrajectory)
        self.get_logger().warn('cuRobo not yet implemented.')
        response.success = False
        response.message = 'cuRobo not yet implemented'
        return response


def main(args=None):
    rclpy.init(args=args)
    node = CuRoboNode()
    rclpy.spin(node)
    rclpy.shutdown()
