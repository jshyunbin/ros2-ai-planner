import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image, JointState
from control_msgs.action import FollowJointTrajectory

from pipeline_orchestrator.sam2 import Sam2
from pipeline_orchestrator.graspgen import GraspGen
from pipeline_orchestrator.curobo import CuRobo
from pipeline_orchestrator.moveit2 import MoveIt2


class PipelineOrchestrator(Node):
    """Single ROS2 node running the full SAM2 → GraspGen → cuRobo pipeline.

    Subscribes (from manip_challenge / Gazebo):
      /task_commands                                std_msgs/String
      /camera/camera/color/image_raw               sensor_msgs/Image  (overhead)
      /camera/camera/depth/color/image_raw         sensor_msgs/Image  (overhead)
      /wrist_camera/wrist_camera/color/image_raw   sensor_msgs/Image  (wrist)
      /wrist_camera/wrist_camera/depth/color/image_raw sensor_msgs/Image (wrist)
      /joint_states                                sensor_msgs/JointState

    Action clients:
      /ur5_controller/follow_joint_trajectory      control_msgs/FollowJointTrajectory
      /gripper_controller/follow_joint_trajectory  control_msgs/FollowJointTrajectory
    """

    # Overhead (3rd-view) camera
    OVERHEAD_RGB_TOPIC = '/camera/camera/color/image_raw'
    OVERHEAD_DEPTH_TOPIC = '/camera/camera/depth/color/image_raw'

    # Wrist camera
    WRIST_RGB_TOPIC = '/wrist_camera/wrist_camera/color/image_raw'
    WRIST_DEPTH_TOPIC = '/wrist_camera/wrist_camera/depth/color/image_raw'

    JOINT_STATES_TOPIC = '/joint_states'
    TASK_COMMANDS_TOPIC = '/task_commands'

    def __init__(self):
        super().__init__('pipeline_orchestrator')

        self.task_sub = self.create_subscription(
            String, self.TASK_COMMANDS_TOPIC, self.task_command_callback, 10)
        self.overhead_rgb_sub = self.create_subscription(
            Image, self.OVERHEAD_RGB_TOPIC, self._cache_overhead_rgb, 10)
        self.overhead_depth_sub = self.create_subscription(
            Image, self.OVERHEAD_DEPTH_TOPIC, self._cache_overhead_depth, 10)
        self.wrist_rgb_sub = self.create_subscription(
            Image, self.WRIST_RGB_TOPIC, self._cache_wrist_rgb, 10)
        self.wrist_depth_sub = self.create_subscription(
            Image, self.WRIST_DEPTH_TOPIC, self._cache_wrist_depth, 10)
        self.joint_sub = self.create_subscription(
            JointState, self.JOINT_STATES_TOPIC, self._cache_joints, 10)

        self._latest_overhead_rgb = None
        self._latest_overhead_depth = None
        self._latest_wrist_rgb = None
        self._latest_wrist_depth = None
        self._latest_joints = None

        self._arm_client = ActionClient(
            self, FollowJointTrajectory, '/ur5_controller/follow_joint_trajectory')
        self._gripper_client = ActionClient(
            self, FollowJointTrajectory, '/gripper_controller/follow_joint_trajectory')

        self._sam2 = Sam2(self.get_logger())
        self._graspgen = GraspGen(self.get_logger())
        self._curobo = CuRobo(self.get_logger())
        self._moveit2 = MoveIt2(self)

        self.get_logger().info('pipeline_orchestrator ready (stub).')

    def _cache_overhead_rgb(self, msg):
        self._latest_overhead_rgb = msg

    def _cache_overhead_depth(self, msg):
        self._latest_overhead_depth = msg

    def _cache_wrist_rgb(self, msg):
        self._latest_wrist_rgb = msg

    def _cache_wrist_depth(self, msg):
        self._latest_wrist_depth = msg

    def _cache_joints(self, msg):
        self._latest_joints = msg

    def task_command_callback(self, msg):
        self.get_logger().info(f'Received task command: {msg.data}')
        self._run_pipeline(msg.data)

    def _run_pipeline(self, task: str):
        # TODO: parse task string into a text prompt for SAM2
        # Overhead camera used for initial scene understanding; wrist camera for close-up
        masks = self._sam2.segment(self._latest_overhead_rgb, prompt=task)
        if masks is None:
            return

        grasp_pose = self._graspgen.generate_grasp(masks, self._latest_overhead_depth)
        if grasp_pose is None:
            return

        trajectory = self._curobo.plan_trajectory(grasp_pose, self._latest_joints)
        if trajectory is None:
            self.get_logger().warn('cuRobo failed, falling back to MoveIt2.')
            trajectory = self._moveit2.plan_trajectory(grasp_pose, self._latest_joints)
        if trajectory is None:
            return

        # TODO: send trajectory via self._arm_client
        # TODO: send gripper command via self._gripper_client


def main(args=None):
    rclpy.init(args=args)
    node = PipelineOrchestrator()
    rclpy.spin(node)
    rclpy.shutdown()
