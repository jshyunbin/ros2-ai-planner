import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger


class Sam2Node(Node):
    """Stub node for SAM2 segmentation.

    Exposes service ~/segment.
    TODO: Replace Trigger with a custom service type that accepts
          (sensor_msgs/Image rgb, string prompt) and returns segmentation masks.
    """

    def __init__(self):
        super().__init__('sam2_node')
        self.srv = self.create_service(
            Trigger,
            '~/segment',
            self.segment_callback,
        )
        self.get_logger().info('sam2_node ready (stub).')

    def segment_callback(self, request, response):
        # TODO: load SAM2 model and run inference on RGB image + text prompt
        # Inputs:  request.rgb   (sensor_msgs/Image)
        #          request.prompt (string)
        # Outputs: response.masks (segmentation masks)
        self.get_logger().warn('SAM2 not yet implemented.')
        response.success = False
        response.message = 'SAM2 not yet implemented'
        return response


def main(args=None):
    rclpy.init(args=args)
    node = Sam2Node()
    rclpy.spin(node)
    rclpy.shutdown()
