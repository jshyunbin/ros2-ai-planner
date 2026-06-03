#!/usr/bin/env python3
import math
import os
import select
import sys
import termios
import time
import tty
from pathlib import Path

import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import TransformStamped
from pykdl_utils.kdl_kinematics import create_kdl_kin
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from tf2_ros import Buffer, TransformException, TransformListener
import xacro

from manip_challenge import get_joint, move_joint


TARGET_POINT = np.array([0.55, 0.0, 0.0], dtype=float)
WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=float)
CAPTURE_DIR = Path("/home/user/JW/iir/captures/teleop")


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-9:
        raise ValueError("zero-length vector")
    return vec / norm


def _rot_z(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def _transform_to_matrix(msg: TransformStamped) -> np.ndarray:
    t = msg.transform.translation
    q = msg.transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    r = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )
    out = np.eye(4, dtype=float)
    out[:3, :3] = r
    out[:3, 3] = [t.x, t.y, t.z]
    return out


def _make_pose_matrix(position: np.ndarray, target: np.ndarray, roll: float) -> np.ndarray:
    z_axis = _normalize(target - position)
    up = WORLD_UP.copy()
    if abs(np.dot(z_axis, up)) > 0.95:
        up = np.array([0.0, 1.0, 0.0], dtype=float)
    x_axis = _normalize(np.cross(up, z_axis))
    y_axis = _normalize(np.cross(z_axis, x_axis))
    r0 = np.column_stack([x_axis, y_axis, z_axis])
    r = r0 @ _rot_z(roll)

    pose = np.eye(4, dtype=float)
    pose[:3, :3] = r
    pose[:3, 3] = position
    return pose


class WorkspaceWristTeleop(Node):
    def __init__(self, step: float, rot_step_deg: float, motion_duration: float) -> None:
        super().__init__("workspace_wrist_teleop")
        self.step = step
        self.rot_step = math.radians(rot_step_deg)
        self.motion_duration = motion_duration
        self.target = TARGET_POINT.copy()
        self.roll = 0.0
        self.bridge = CvBridge()
        self.latest_image = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self._arm_kdl = self._create_arm_kdl()
        self._camera_from_ee = None
        self._desired_camera_pos = None
        self._last_q = None

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.create_subscription(
            Image,
            "/wrist_camera/wrist_camera/color/image_raw",
            self._image_cb,
            qos,
        )

    def _create_arm_kdl(self):
        ur5_description_path = os.path.join(get_package_share_directory("ur5_ros2_gazebo"))
        xacro_file = os.path.join(ur5_description_path, "urdf", "ur5.urdf.xacro")
        with open(xacro_file, "r", encoding="utf-8") as fh:
            doc = xacro.parse(fh)
        xacro.process_doc(
            doc,
            mappings={
                "cell_layout_1": "true",
                "cell_layout_2": "false",
                "hardware_interface": "PositionJointInterface",
                "camera_enabled": "true",
            },
        )
        return create_kdl_kin("base_link", "robotiq_85_base_link", urdf_xml=doc.toxml())

    def _image_cb(self, msg: Image) -> None:
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def _lookup_matrix(self, target_frame: str, source_frame: str) -> np.ndarray:
        tf = self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time())
        return _transform_to_matrix(tf)

    def initialize(self) -> None:
        last_log = 0.0
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.2)
            try:
                base_from_ee = self._lookup_matrix("base_link", "robotiq_85_base_link")
                base_from_cam = self._lookup_matrix("base_link", "wrist_camera_color_optical_frame")
                self._camera_from_ee = np.linalg.inv(base_from_ee) @ base_from_cam
                self._desired_camera_pos = base_from_cam[:3, 3].copy()
                self._last_q = np.array(get_joint.get_joint_angles(self), dtype=float)
                return
            except (TransformException, RuntimeError, ValueError):
                now = time.time()
                if now - last_log > 2.0:
                    self.get_logger().info("Waiting for TF frames and /joint_states...")
                    last_log = now
                continue
        raise RuntimeError("ROS shutdown before TF/joint state became available.")

    def _solve_ik_for_camera(self, camera_position: np.ndarray, roll: float, target: np.ndarray | None = None):
        aim_target = self.target if target is None else target
        base_from_cam = _make_pose_matrix(camera_position, aim_target, roll)
        base_from_ee = base_from_cam @ np.linalg.inv(self._camera_from_ee)
        q_sol = self._arm_kdl.inverse(base_from_ee, q_guess=self._last_q)
        if q_sol is None:
            return None
        return np.array(q_sol, dtype=float)

    def move_camera(self, dx: float, dy: float, dz: float, droll: float) -> bool:
        candidate_pos = self._desired_camera_pos + np.array([dx, dy, dz], dtype=float)
        candidate_roll = self.roll + droll
        q_sol = self._solve_ik_for_camera(candidate_pos, candidate_roll)
        if q_sol is None:
            self.get_logger().warn("IK failed for requested camera step.")
            return False

        move_joint.move_joint(self, q_sol.tolist(), duration=self.motion_duration)
        self._desired_camera_pos = candidate_pos
        self.roll = candidate_roll
        self._last_q = q_sol
        self.get_logger().info(
            f"camera=({candidate_pos[0]:.3f}, {candidate_pos[1]:.3f}, {candidate_pos[2]:.3f}) "
            f"target=({self.target[0]:.3f}, {self.target[1]:.3f}, {self.target[2]:.3f}) "
            f"roll={math.degrees(candidate_roll):.1f} deg"
        )
        return True

    def reset_from_current(self) -> None:
        base_from_cam = self._lookup_matrix("base_link", "wrist_camera_color_optical_frame")
        self._desired_camera_pos = base_from_cam[:3, 3].copy()
        self._last_q = np.array(get_joint.get_joint_angles(self), dtype=float)
        self.roll = 0.0
        self.get_logger().info("Reset desired camera state from current TF.")

    def nudge_target_z(self, dz: float) -> bool:
        candidate_target = self.target.copy()
        candidate_target[2] += dz
        q_sol = self._solve_ik_for_camera(self._desired_camera_pos, self.roll, target=candidate_target)
        if q_sol is None:
            self.get_logger().warn("IK failed for requested target adjustment.")
            return False
        move_joint.move_joint(self, q_sol.tolist(), duration=self.motion_duration)
        self.target = candidate_target
        self._last_q = q_sol
        self.get_logger().info(
            f"camera=({self._desired_camera_pos[0]:.3f}, {self._desired_camera_pos[1]:.3f}, {self._desired_camera_pos[2]:.3f}) "
            f"target=({self.target[0]:.3f}, {self.target[1]:.3f}, {self.target[2]:.3f}) "
            f"roll={math.degrees(self.roll):.1f} deg"
        )
        return True

    def save_snapshot(self) -> Path:
        if self.latest_image is None:
            raise RuntimeError("No wrist image available.")
        CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = CAPTURE_DIR / f"wrist_{stamp}.png"
        cv2.imwrite(str(path), self.latest_image)
        return path


HELP = """\
Controls
  w/s : +/- x in world
  a/d : +/- y in world
  q/e : +/- z in world
  z/c : rotate around viewing axis
  t/g : raise/lower look target
  p   : save wrist PNG
  r   : reset teleop state from current TF
  x   : exit
"""


def get_key(timeout: float = 0.1) -> str:
    dr, _, _ = select.select([sys.stdin], [], [], timeout)
    if not dr:
        return ""
    return sys.stdin.read(1)


def main() -> int:
    step = 0.03
    rot_step_deg = 8.0
    motion_duration = 0.8

    rclpy.init()
    node = WorkspaceWristTeleop(step=step, rot_step_deg=rot_step_deg, motion_duration=motion_duration)
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        node.initialize()
        print(HELP, flush=True)
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)
            key = get_key()
            if not key:
                continue
            if key == "w":
                node.move_camera(node.step, 0.0, 0.0, 0.0)
            elif key == "s":
                node.move_camera(-node.step, 0.0, 0.0, 0.0)
            elif key == "a":
                node.move_camera(0.0, node.step, 0.0, 0.0)
            elif key == "d":
                node.move_camera(0.0, -node.step, 0.0, 0.0)
            elif key == "q":
                node.move_camera(0.0, 0.0, node.step, 0.0)
            elif key == "e":
                node.move_camera(0.0, 0.0, -node.step, 0.0)
            elif key == "z":
                node.move_camera(0.0, 0.0, 0.0, node.rot_step)
            elif key == "c":
                node.move_camera(0.0, 0.0, 0.0, -node.rot_step)
            elif key == "t":
                node.nudge_target_z(node.step)
            elif key == "g":
                node.nudge_target_z(-node.step)
            elif key == "r":
                node.reset_from_current()
            elif key == "p":
                path = node.save_snapshot()
                print(path, flush=True)
            elif key == "x":
                break
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
