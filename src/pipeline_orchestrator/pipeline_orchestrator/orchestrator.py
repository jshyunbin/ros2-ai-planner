import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image, JointState
from std_srvs.srv import Trigger


class PipelineOrchestrator(Node):
    """Orchestrates the full SAM2 → GraspGen → cuRobo pipeline.

    External subscriptions (from manip_challenge / Gazebo):
      /task_commands              std_msgs/String
      /wrist_camera/.../color/image_raw  sensor_msgs/Image
      /wrist_camera/.../depth/image_raw  sensor_msgs/Image
      /joint_states               sensor_msgs/JointState

    Internal service clients:
      sam2_node/segment           std_srvs/Trigger  (TODO: custom type)
      graspgen_node/generate_grasp std_srvs/Trigger (TODO: custom type)
      curobo_node/plan_trajectory  std_srvs/Trigger (TODO: custom type)

    Sends planned trajectory to:
      /ur5_controller/follow_joint_trajectory  (action, control_msgs)
    """

    CAMERA_RGB_TOPIC = '/wrist_camera/wrist_camera/color/image_raw'
    CAMERA_DEPTH_TOPIC = '/wrist_camera/wrist_camera/depth/image_raw'
    JOINT_STATES_TOPIC = '/joint_states'
    TASK_COMMANDS_TOPIC = '/task_commands'

    def __init__(self):
        super().__init__('pipeline_orchestrator')

        # External subscribers
        self.task_sub = self.create_subscription(
            String, self.TASK_COMMANDS_TOPIC, self.task_command_callback, 10)
        self.rgb_sub = self.create_subscription(
            Image, self.CAMERA_RGB_TOPIC, self._cache_rgb, 10)
        self.depth_sub = self.create_subscription(
            Image, self.CAMERA_DEPTH_TOPIC, self._cache_depth, 10)
        self.joint_sub = self.create_subscription(
            JointState, self.JOINT_STATES_TOPIC, self._cache_joints, 10)

        # Internal service clients
        self.sam2_client = self.create_client(Trigger, '/sam2_node/segment')
        self.graspgen_client = self.create_client(
            Trigger, '/graspgen_node/generate_grasp')
        self.curobo_client = self.create_client(
            Trigger, '/curobo_node/plan_trajectory')

        # Cached sensor data
        self._latest_rgb = None
        self._latest_depth = None
        self._latest_joints = None

        self.get_logger().info('pipeline_orchestrator ready (stub).')

    def _cache_rgb(self, msg):
        self._latest_rgb = msg

    def _cache_depth(self, msg):
        self._latest_depth = msg

    def _cache_joints(self, msg):
        self._latest_joints = msg

    def task_command_callback(self, msg):
        self.get_logger().info(f'Received task command: {msg.data}')
        # TODO: parse task command, then run pipeline
        self.call_sam2()

    def call_sam2(self):
        # TODO: send cached RGB + parsed prompt to sam2_node/segment service
        # On response: call self.call_graspgen(masks)
        self.get_logger().warn('call_sam2 not yet implemented.')

    def call_graspgen(self, masks=None):
        # TODO: send masks + cached depth to graspgen_node/generate_grasp
        # On response: call self.call_curobo(grasp_pose)
        self.get_logger().warn('call_graspgen not yet implemented.')

    def call_curobo(self, grasp_pose=None):
        # TODO: send grasp_pose + cached joint states to
        # curobo_node/plan_trajectory
        # On response: send trajectory to /ur5_controller/follow_joint_trajectory
        self.get_logger().warn('call_curobo not yet implemented.')


def main(args=None):
    rclpy.init(args=args)
    node = PipelineOrchestrator()
    rclpy.spin(node)
    rclpy.shutdown()
