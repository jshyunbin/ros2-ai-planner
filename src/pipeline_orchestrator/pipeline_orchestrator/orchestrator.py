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
from pipeline_orchestrator.gemini import GeminiLocalizer


class PipelineOrchestrator(Node):
    """Single ROS2 node running the full pipeline.

    Pipeline per task command:
      GeminiLocalizer (overhead RGB + prompt → bbox)
      SAM2 (overhead RGB + Gemini bbox → object mask)
      GraspGen (point cloud → grasp candidates)
      CuRobo (grasp candidate + live ESDF → collision-free trajectory)
      [fallback] MoveIt2 if CuRobo fails

    CuRobo subscribes to depth streams internally (no separate NvBlox node).

    Subscribes (from manip_challenge / Gazebo):
      /task_commands                                    std_msgs/String
      /camera/camera/color/image_raw                   sensor_msgs/Image  (overhead)
      /wrist_camera/wrist_camera/color/image_raw       sensor_msgs/Image  (wrist)
      /joint_states                                     sensor_msgs/JointState

    Action clients:
      /ur5_controller/follow_joint_trajectory       control_msgs/FollowJointTrajectory
      /gripper_controller/follow_joint_trajectory   control_msgs/FollowJointTrajectory
    """

    OVERHEAD_RGB_TOPIC  = '/camera/camera/color/image_raw'
    WRIST_RGB_TOPIC     = '/wrist_camera/wrist_camera/color/image_raw'
    JOINT_STATES_TOPIC  = '/joint_states'
    TASK_COMMANDS_TOPIC = '/task_commands'

    def __init__(self):
        super().__init__('pipeline_orchestrator')

        self.task_sub = self.create_subscription(
            String, self.TASK_COMMANDS_TOPIC, self.task_command_callback, 10)
        self.overhead_rgb_sub = self.create_subscription(
            Image, self.OVERHEAD_RGB_TOPIC, self._cache_overhead_rgb, 10)
        self.wrist_rgb_sub = self.create_subscription(
            Image, self.WRIST_RGB_TOPIC, self._cache_wrist_rgb, 10)
        self.joint_sub = self.create_subscription(
            JointState, self.JOINT_STATES_TOPIC, self._cache_joints, 10)

        self._latest_overhead_rgb = None
        self._latest_wrist_rgb    = None
        self._latest_joints       = None

        self._arm_client = ActionClient(
            self, FollowJointTrajectory, '/ur5_controller/follow_joint_trajectory')
        self._gripper_client = ActionClient(
            self, FollowJointTrajectory, '/gripper_controller/follow_joint_trajectory')

        self._sam2     = Sam2(self.get_logger())
        self._graspgen = GraspGen(self.get_logger())
        self._curobo   = CuRobo(self)
        self._moveit2  = MoveIt2(self)
        self._gemini   = GeminiLocalizer(self.get_logger())

        self.get_logger().info('pipeline_orchestrator ready.')

    def _cache_overhead_rgb(self, msg): self._latest_overhead_rgb = msg
    def _cache_wrist_rgb(self, msg):    self._latest_wrist_rgb    = msg
    def _cache_joints(self, msg):       self._latest_joints       = msg

    def task_command_callback(self, msg):
        self.get_logger().info(f'Received task command: {msg.data}')
        self._run_pipeline(msg.data)

    def _run_pipeline(self, task: str):
        bbox  = self._gemini.locate_object(self._latest_overhead_rgb, task)
        masks = self._sam2.segment(self._latest_overhead_rgb, prompt=task, bbox=bbox)
        if masks is None:
            return

        point_cloud = None
        grasp_pose  = self._graspgen.generate_grasp(point_cloud)
        if grasp_pose is None:
            return

        trajectory = self._curobo.plan_trajectory(grasp_pose, self._latest_joints)
        if trajectory is None:
            self.get_logger().warn('cuRobo failed, falling back to MoveIt2.')
            trajectory = self._moveit2.plan_trajectory(grasp_pose, self._latest_joints)
        if trajectory is None:
            return


def main(args=None):
    rclpy.init(args=args)
    node = PipelineOrchestrator()
    rclpy.spin(node)
    rclpy.shutdown()
