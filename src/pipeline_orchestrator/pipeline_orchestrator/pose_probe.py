"""pose_probe: validate one place/home pose against the live cuRobo planner.

Debug utility (not part of the runtime pipeline). Loads a single named target
from config/place_poses.yml, plans to it via /curobo/plan_trajectory in
single-pose mode (grasp_pose set, grasp_poses empty), optionally executes the
trajectory on the arm, and prints the achieved tool0 pose so the YAML can be
tuned in the live sim. One-shot run() mirrors graspgen_service_caller.

Run (inside the container, workspace sourced):

  ros2 run pipeline_orchestrator pose_probe --ros-args \
    -p target:=home -p execute:=false
"""

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState
from riro_srvs.srv import PlanTrajectory
import tf2_ros

from pipeline_orchestrator.place_pose_utils import (
    load_place_poses,
    resolve_target_pose,
)


class PoseProbe(Node):
    """One-shot validator for a single place/home pose."""

    def __init__(self) -> None:
        super().__init__("pose_probe")

        self.declare_parameter(
            "poses_file",
            "/ros2_ws/src/pipeline_orchestrator/config/place_poses.yml",
        )
        self.declare_parameter("target", "home")
        self.declare_parameter("execute", True)
        self.declare_parameter("service_name", "/curobo/plan_trajectory")
        self.declare_parameter(
            "arm_action_name", "/ur5_controller/follow_joint_trajectory")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("tool_frame", "tool0")
        self.declare_parameter("service_timeout_sec", 120.0)

        self._poses_file = str(self.get_parameter("poses_file").value)
        self._target = str(self.get_parameter("target").value)
        self._execute = bool(self.get_parameter("execute").value)
        self._service_name = str(self.get_parameter("service_name").value)
        self._service_timeout = float(
            self.get_parameter("service_timeout_sec").value)

        self._latest_joints = None
        self.create_subscription(
            JointState, "/joint_states", self._cache_joints, 10)
        self._client = self.create_client(PlanTrajectory, self._service_name)
        self._arm = ActionClient(
            self,
            FollowJointTrajectory,
            str(self.get_parameter("arm_action_name").value),
        )
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

    def _cache_joints(self, msg: JointState) -> None:
        self._latest_joints = msg

    def run(self) -> int:
        try:
            data = load_place_poses(self._poses_file)
            pose = resolve_target_pose(data, self._target)
        except (OSError, ValueError, KeyError) as exc:
            self.get_logger().error(f"Could not load target: {exc}")
            return 1
        if pose is None:
            self.get_logger().error("geometry_msgs unavailable; cannot build pose.")
            return 1
        self.get_logger().info(
            f"target={self._target} "
            f"xyz=({pose.position.x:.3f},{pose.position.y:.3f},"
            f"{pose.position.z:.3f}) "
            f"quat_xyzw=({pose.orientation.x:.3f},{pose.orientation.y:.3f},"
            f"{pose.orientation.z:.3f},{pose.orientation.w:.3f})")

        for _ in range(50):
            if self._latest_joints is not None:
                break
            rclpy.spin_once(self, timeout_sec=0.1)
        if self._latest_joints is None:
            self.get_logger().error("No /joint_states received.")
            return 1

        if not self._client.wait_for_service(timeout_sec=30.0):
            self.get_logger().error(f"Service unavailable: {self._service_name}")
            return 1

        request = PlanTrajectory.Request()
        request.grasp_pose = pose          # single-pose mode (grasp_poses empty)
        request.joint_state = self._latest_joints
        future = self._client.call_async(request)
        rclpy.spin_until_future_complete(
            self, future, timeout_sec=self._service_timeout)
        if not future.done() or future.result() is None:
            self.get_logger().error("plan_trajectory call failed or timed out.")
            return 1
        result = future.result()
        if not result.success:
            self.get_logger().error(f"Planning failed: {result.message}")
            return 1
        self.get_logger().info(
            f"Planned OK: {result.message} "
            f"({len(result.trajectory.points)} points)")

        if not self._execute:
            self.get_logger().info("execute=false; not moving the arm.")
            return 0
        if not self._send_trajectory(result.trajectory):
            return 1
        self._report_tool_pose()
        return 0

    def _send_trajectory(self, trajectory) -> bool:
        if not self._arm.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Arm action server unavailable.")
            return False
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory
        send_future = self._arm.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=10.0)
        handle = send_future.result()
        if handle is None or not handle.accepted:
            self.get_logger().error("Arm goal rejected by action server.")
            return False
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=60.0)
        if result_future.result() is None:
            self.get_logger().error("Arm action returned no result.")
            return False
        self.get_logger().info("Trajectory executed.")
        return True

    def _report_tool_pose(self) -> None:
        base = str(self.get_parameter("base_frame").value)
        tool = str(self.get_parameter("tool_frame").value)
        for _ in range(10):              # let TF accumulate a couple of frames
            rclpy.spin_once(self, timeout_sec=0.1)
        try:
            tf = self._tf_buffer.lookup_transform(base, tool, rclpy.time.Time())
        except Exception as exc:         # tf2 raises several exception types
            self.get_logger().warning(f"TF {base}<-{tool} lookup failed: {exc}")
            return
        t = tf.transform.translation
        q = tf.transform.rotation
        self.get_logger().info(
            f"ACHIEVED {tool} in {base}: "
            f"xyz=({t.x:.3f},{t.y:.3f},{t.z:.3f}) "
            f"quat_xyzw=({q.x:.3f},{q.y:.3f},{q.z:.3f},{q.w:.3f})")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PoseProbe()
    try:
        raise SystemExit(node.run())
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
