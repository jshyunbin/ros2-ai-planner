"""home_config_tuner: interactively try arm joint configs and capture the
wrist-camera view, to pick a good ``home_joint_config`` for place_poses.yml.

Debug utility (not part of the runtime pipeline). A REPL reads six joint values
(``pan lift elbow wrist_1 wrist_2 wrist_3``), commands the arm there via
``/ur5_controller/follow_joint_trajectory``, waits for arrival, grabs the next
wrist-camera frame stamped *after* the move settled, and saves it as a PNG under
``/artifacts/home_tuner/`` (mounted to ``./artifacts/home_tuner`` on the host).
It also prints the achieved tool0 pose (xyz + quat_xyzw from TF) so a good config
can be copied straight into place_poses.yml.

Run this with Gazebo/manip_challenge up but the orchestrator NOT running (so the
two don't fight over the arm action):

  ros2 run team_8 home_config_tuner

At the prompt, type six numbers (space- or comma-separated), or ``q`` to quit:

  config> 0 -2.2 1.9 -1.383 -1.57 0
"""

import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import tf2_ros

# Local copy (curobo.py pulls in torch/cuRobo; the tuner stays dependency-light).
JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


class HomeConfigTuner(Node):
    """REPL that drives the arm to a joint config and saves the wrist view."""

    def __init__(self) -> None:
        super().__init__("home_config_tuner")

        self.declare_parameter(
            "arm_action_name", "/ur5_controller/follow_joint_trajectory")
        self.declare_parameter(
            "wrist_image_topic",
            "/wrist_camera/wrist_camera/color/image_raw")
        self.declare_parameter("output_dir", "/artifacts/home_tuner")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("tool_frame", "tool0")
        self.declare_parameter("move_duration_sec", 4.0)
        self.declare_parameter("execute_timeout_sec", 30.0)
        self.declare_parameter("fresh_frame_timeout_sec", 5.0)

        self._arm_action_name = str(self.get_parameter("arm_action_name").value)
        self._output_dir = Path(str(self.get_parameter("output_dir").value))
        self._base_frame = str(self.get_parameter("base_frame").value)
        self._tool_frame = str(self.get_parameter("tool_frame").value)
        self._move_duration = float(self.get_parameter("move_duration_sec").value)
        self._execute_timeout = float(
            self.get_parameter("execute_timeout_sec").value)
        self._fresh_timeout = float(
            self.get_parameter("fresh_frame_timeout_sec").value)

        self._bridge = CvBridge()
        self._latest_image = None
        self._latest_image_stamp_ns = 0
        self.create_subscription(
            Image,
            str(self.get_parameter("wrist_image_topic").value),
            self._cache_image,
            10,
        )
        self._arm = ActionClient(
            self, FollowJointTrajectory, self._arm_action_name)
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

    def _cache_image(self, msg: Image) -> None:
        self._latest_image = msg
        self._latest_image_stamp_ns = (
            msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec)

    # --- REPL -------------------------------------------------------------

    def run(self) -> int:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        if not self._arm.wait_for_server(timeout_sec=10.0):
            self.get_logger().error(
                f"Arm action server unavailable: {self._arm_action_name}. "
                "Is manip_challenge running (and the orchestrator stopped)?")
            return 1
        self.get_logger().info(
            f"Saving wrist views to {self._output_dir}. Joint order: "
            f"{' '.join(n.replace('_joint', '') for n in JOINT_NAMES)}.")

        while rclpy.ok():
            try:
                raw = input("\nconfig> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if raw.lower() in ("q", "quit", "exit"):
                break
            config = self._parse_config(raw)
            if config is None:
                continue
            self._try_config(config)
        return 0

    def _parse_config(self, raw: str):
        if not raw:
            return None
        tokens = raw.replace(",", " ").split()
        if len(tokens) != 6:
            self.get_logger().warning(
                f"Expected 6 joint values, got {len(tokens)}. Try again.")
            return None
        try:
            return [float(t) for t in tokens]
        except ValueError:
            self.get_logger().warning(
                "Could not parse all six values as numbers. Try again.")
            return None

    def _try_config(self, config) -> None:
        if not self._send_trajectory(config):
            return
        image = self._capture_fresh_image()
        if image is None:
            self.get_logger().warning(
                "Arm moved but no fresh wrist frame arrived; skipping save.")
        else:
            path = self._save_image(image, config)
            self.get_logger().info(f"Saved wrist view: {path}")
        self._report_tool_pose()

    def _send_trajectory(self, config) -> bool:
        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in config]
        secs = int(self._move_duration)
        point.time_from_start = Duration(
            sec=secs, nanosec=int((self._move_duration - secs) * 1e9))
        traj = JointTrajectory()
        traj.joint_names = list(JOINT_NAMES)
        traj.points = [point]

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        send_future = self._arm.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=10.0)
        if not send_future.done() or send_future.result() is None:
            self.get_logger().error("Arm goal-send timed out.")
            return False
        handle = send_future.result()
        if not handle.accepted:
            self.get_logger().error("Arm goal rejected by action server.")
            return False
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(
            self, result_future, timeout_sec=self._execute_timeout)
        if not result_future.done() or result_future.result() is None:
            self.get_logger().error("Arm action returned no result (timeout?).")
            return False
        return True

    def _capture_fresh_image(self):
        """Spin until a wrist frame stamped after now arrives (freshness gate)."""
        reference_ns = self.get_clock().now().nanoseconds
        deadline = time.monotonic() + self._fresh_timeout
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if (self._latest_image is not None
                    and self._latest_image_stamp_ns > reference_ns):
                return self._latest_image
        return None

    def _save_image(self, msg: Image, config) -> Path:
        bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        tag = "_".join(f"{v:+.3f}" for v in config)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = self._output_dir / f"cfg_{tag}_{stamp}.png"
        from PIL import Image as PILImage
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        PILImage.fromarray(rgb).save(path)
        return path

    def _report_tool_pose(self) -> None:
        for _ in range(10):              # let TF accumulate a couple of frames
            rclpy.spin_once(self, timeout_sec=0.1)
        try:
            tf = self._tf_buffer.lookup_transform(
                self._base_frame, self._tool_frame, rclpy.time.Time())
        except Exception as exc:         # tf2 raises several exception types
            self.get_logger().warning(
                f"TF {self._base_frame}<-{self._tool_frame} lookup failed: {exc}")
            return
        t = tf.transform.translation
        q = tf.transform.rotation
        self.get_logger().info(
            f"ACHIEVED {self._tool_frame} in {self._base_frame}: "
            f"xyz=[{t.x:.6f}, {t.y:.6f}, {t.z:.6f}] "
            f"quat_xyzw=[{q.x:.6f}, {q.y:.6f}, {q.z:.6f}, {q.w:.6f}]")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = HomeConfigTuner()
    try:
        raise SystemExit(node.run())
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


# Allows running without a colcon rebuild via `python3 -m team_8.home_config_tuner`
# (the entrypoint puts src on PYTHONPATH), convenient for the live-sim loop.
if __name__ == "__main__":
    main()
