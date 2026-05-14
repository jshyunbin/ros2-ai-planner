import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger


class GraspGenNode(Node):
    """Stub node for GraspGen grasp pose generation.

    Exposes service ~/generate_grasp.
    TODO: Replace Trigger with a custom service type that accepts
          (segmentation masks, sensor_msgs/Image depth) and returns
          geometry_msgs/PoseStamped grasp pose.
    """

    def __init__(self):
        super().__init__('graspgen_node')
        self.srv = self.create_service(
            Trigger,
            '~/generate_grasp',
            self.generate_grasp_callback,
        )
        self.get_logger().info('graspgen_node ready (stub).')

    def generate_grasp_callback(self, request, response):
        # TODO: run GraspGen diffusion model on masks + depth image
        # Inputs:  request.masks (segmentation masks from sam2_node)
        #          request.depth (sensor_msgs/Image)
        # Outputs: response.grasp_pose (geometry_msgs/PoseStamped)
        self.get_logger().warn('GraspGen not yet implemented.')
        response.success = False
        response.message = 'GraspGen not yet implemented'
        return response


def main(args=None):
    rclpy.init(args=args)
    node = GraspGenNode()
    rclpy.spin(node)
    rclpy.shutdown()
