import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image, JointState

from pipeline_orchestrator.sam2 import Sam2
from pipeline_orchestrator.graspgen import GraspGen
from pipeline_orchestrator.curobo import CuRobo


class PipelineOrchestrator(Node):
    """Single ROS2 node running the full SAM2 → GraspGen → cuRobo pipeline.

    Subscribes (from manip_challenge / Gazebo):
      /task_commands                           std_msgs/String
      /wrist_camera/.../color/image_raw        sensor_msgs/Image
      /wrist_camera/.../depth/image_raw        sensor_msgs/Image
      /joint_states                            sensor_msgs/JointState

    Sends planned trajectory to:
      /ur5_controller/follow_joint_trajectory  (action, control_msgs)
    """

    CAMERA_RGB_TOPIC = '/wrist_camera/wrist_camera/color/image_raw'
    CAMERA_DEPTH_TOPIC = '/wrist_camera/wrist_camera/depth/image_raw'
    JOINT_STATES_TOPIC = '/joint_states'
    TASK_COMMANDS_TOPIC = '/task_commands'

    def __init__(self):
        super().__init__('pipeline_orchestrator')

        self.task_sub = self.create_subscription(
            String, self.TASK_COMMANDS_TOPIC, self.task_command_callback, 10)
        self.rgb_sub = self.create_subscription(
            Image, self.CAMERA_RGB_TOPIC, self._cache_rgb, 10)
        self.depth_sub = self.create_subscription(
            Image, self.CAMERA_DEPTH_TOPIC, self._cache_depth, 10)
        self.joint_sub = self.create_subscription(
            JointState, self.JOINT_STATES_TOPIC, self._cache_joints, 10)

        self._latest_rgb = None
        self._latest_depth = None
        self._latest_joints = None

        self._sam2 = Sam2(self.get_logger())
        self._graspgen = GraspGen(self.get_logger())
        self._curobo = CuRobo(self.get_logger())

        self.get_logger().info('pipeline_orchestrator ready (stub).')

    def _cache_rgb(self, msg):
        self._latest_rgb = msg

    def _cache_depth(self, msg):
        self._latest_depth = msg

    def _cache_joints(self, msg):
        self._latest_joints = msg

    def task_command_callback(self, msg):
        self.get_logger().info(f'Received task command: {msg.data}')
        self._run_pipeline(msg.data)

    def _run_pipeline(self, task: str):
        # TODO: parse task string into a text prompt for SAM2
        masks = self._sam2.segment(self._latest_rgb, prompt=task)
        if masks is None:
            return

        grasp_pose = self._graspgen.generate_grasp(masks, self._latest_depth)
        if grasp_pose is None:
            return

        trajectory = self._curobo.plan_trajectory(grasp_pose, self._latest_joints)
        if trajectory is None:
            return

        # TODO: send trajectory to /ur5_controller/follow_joint_trajectory


def main(args=None):
    rclpy.init(args=args)
    node = PipelineOrchestrator()
    rclpy.spin(node)
    rclpy.shutdown()
