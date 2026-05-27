#!/usr/bin/env python3
"""
Live perception + robot visualization over viser.

Subscribes to both D435 depth cameras and /joint_states.  Back-projects
depth frames to world-frame point clouds (cyan = overhead, orange = wrist)
and animates the UR5 model from the live joint state stream.

Run inside the container while Gazebo is running on the host:
  docker compose run --rm -p 8080:8080 ai_planner \
    python3 /ros2_ws/src/pipeline_orchestrator/scripts/live_perception_viz.py \
    --target 0.5 0.0 0.3

Open http://localhost:8080.  SSH users forward the port first:
  ssh -L 8080:localhost:8080 user@host
"""

import argparse
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
import rclpy.duration
import rclpy.time
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from sensor_msgs.msg import JointState as JointStateMsg
from tf2_ros import Buffer, TransformListener
from cv_bridge import CvBridge

from rclpy.qos import (QoSProfile, QoSReliabilityPolicy,
                       QoSHistoryPolicy, QoSDurabilityPolicy)  # noqa: F401

import viser
from viser.extras import ViserUrdf

# ── constants ─────────────────────────────────────────────────────────────────

JOINT_NAMES = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
]
URDF_PATH   = '/ur5.urdf'
WORLD_FRAME = 'world'

OVERHEAD_DEPTH = '/camera/camera/depth/color/image_raw'
OVERHEAD_INFO  = '/camera/camera/depth/camera_info'
OVERHEAD_FRAME = 'camera_color_optical_frame'
WRIST_DEPTH    = '/wrist_camera/wrist_camera/depth/color/image_raw'
WRIST_INFO     = '/wrist_camera/wrist_camera/depth/camera_info'
WRIST_FRAME    = 'wrist_camera_color_optical_frame'

MIN_DEPTH     = 0.15   # m — discard closer returns (noise / arm occlusion)
MAX_DEPTH     = 2.0    # m — discard far background
POINT_STRIDE  = 4      # sample every Nth pixel; keeps point count manageable
VIZ_HZ        = 10     # viser refresh rate

CAM_COLOR = {
    'overhead': np.array([0,   180, 255], dtype=np.uint8),  # cyan
    'wrist':    np.array([255, 140, 0],   dtype=np.uint8),  # orange
}


# ── helpers ───────────────────────────────────────────────────────────────────

def resolve_urdf(urdf_path: str) -> Path:
    """Return a temp URDF with package:// mesh URIs replaced by absolute paths."""
    pkg_root = '/opt/ros/humble/share'
    with open(urdf_path) as f:
        content = f.read()
    content = content.replace('package://ur_description', f'{pkg_root}/ur_description')
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False)
    tmp.write(content)
    tmp.flush()
    return Path(tmp.name)


def quat_to_rot(x, y, z, w):
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),   2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),   2*(y*z - x*w)],
        [    2*(x*z - y*w),   2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def tf_to_matrix(tf):
    T = np.eye(4)
    t, r = tf.translation, tf.rotation
    T[:3, 3]  = [t.x, t.y, t.z]
    T[:3, :3] = quat_to_rot(r.x, r.y, r.z, r.w)
    return T


def depth_to_world_points(depth_m: np.ndarray, K: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Back-project a float32 depth image (metres) into world-frame XYZ."""
    H, W = depth_m.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    s  = POINT_STRIDE
    us = np.arange(0, W, s, dtype=np.float32)
    vs = np.arange(0, H, s, dtype=np.float32)
    uu, vv = np.meshgrid(us, vs)
    d = depth_m[::s, ::s]

    valid = (d > MIN_DEPTH) & (d < MAX_DEPTH)
    d, uu, vv = d[valid], uu[valid], vv[valid]
    if d.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    x = (uu - cx) * d / fx
    y = (vv - cy) * d / fy
    pts_cam = np.stack([x, y, d, np.ones_like(d)], axis=1)   # (N, 4)
    return (pose @ pts_cam.T).T[:, :3].astype(np.float32)     # (N, 3)


# ── ROS2 node ─────────────────────────────────────────────────────────────────

class PerceptionNode(Node):
    def __init__(self):
        super().__init__('live_perception_viz')
        self._lock   = threading.Lock()
        self._bridge = CvBridge()
        self._K      = {}   # cam_id → (3,3) float32
        self._points = {}   # cam_id → (N,3) float32
        self._joints = {}   # joint_name → float64

        self._tf_buf = Buffer()
        self._tf_lis = TransformListener(self._tf_buf, self)

        # Camera plugins publish BEST_EFFORT/VOLATILE — must match or Gazebo
        # sees no subscribers, stops the sensor, and unregisters the topic.
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(CameraInfo, OVERHEAD_INFO,
            lambda m: self._on_info(m, 'overhead'), sensor_qos)
        self.create_subscription(Image, OVERHEAD_DEPTH,
            lambda m: self._on_depth(m, 'overhead', OVERHEAD_FRAME), sensor_qos)
        self.create_subscription(CameraInfo, WRIST_INFO,
            lambda m: self._on_info(m, 'wrist'), sensor_qos)
        self.create_subscription(Image, WRIST_DEPTH,
            lambda m: self._on_depth(m, 'wrist', WRIST_FRAME), sensor_qos)
        self.create_subscription(JointStateMsg, '/joint_states',
            self._on_joints, 10)

    # ── callbacks ──────────────────────────────────────────────────────────

    def _on_info(self, msg, cam_id: str):
        K = np.array([[msg.k[0], 0,        msg.k[2]],
                      [0,        msg.k[4],  msg.k[5]],
                      [0,        0,         1       ]], dtype=np.float32)
        with self._lock:
            self._K[cam_id] = K

    def _on_depth(self, msg, cam_id: str, frame: str):
        with self._lock:
            if cam_id not in self._K:
                return
            K = self._K[cam_id]

        stamp = rclpy.time.Time(seconds=msg.header.stamp.sec,
                                nanoseconds=msg.header.stamp.nanosec)
        try:
            tf = self._tf_buf.lookup_transform(
                WORLD_FRAME, frame, stamp,
                timeout=rclpy.duration.Duration(seconds=0.05))
        except Exception:
            try:
                # Fall back to latest available transform
                tf = self._tf_buf.lookup_transform(
                    WORLD_FRAME, frame, rclpy.time.Time())
            except Exception as e:
                print(f'  [TF MISS] {WORLD_FRAME} → {frame}: {e}')
                return

        pose   = tf_to_matrix(tf.transform)
        cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        depth  = np.nan_to_num(cv_img.astype(np.float32) / 1000.0)
        pts    = depth_to_world_points(depth, K, pose)

        with self._lock:
            self._points[cam_id] = pts

    def _on_joints(self, msg):
        with self._lock:
            first = len(self._joints) == 0
            for name, pos in zip(msg.name, msg.position):
                self._joints[name] = np.float64(pos)
            if first:
                print(f'  [JOINTS] First message — names: {list(msg.name)}')

    def snapshot(self):
        with self._lock:
            return dict(self._points), dict(self._joints)


# ── viser loop ────────────────────────────────────────────────────────────────

def run_viser(node: PerceptionNode, target_xyz):
    server = viser.ViserServer(port=8080, verbose=False)
    print('\nViser running — open http://localhost:8080')

    server.scene.add_frame('/world', axes_length=0.3, axes_radius=0.01)

    if target_xyz is not None:
        tx, ty, tz = target_xyz
        server.scene.add_icosphere('/target', radius=0.04,
                                   color=(255, 80, 80),
                                   position=(tx, ty, tz))
        print(f'  Target marker at ({tx:.2f}, {ty:.2f}, {tz:.2f})')

    try:
        robot = ViserUrdf(server, urdf_or_path=resolve_urdf(URDF_PATH),
                          root_node_name='/ur5')
        print('  Robot model loaded.')
    except Exception as e:
        robot = None
        print(f'  Robot model unavailable ({e})')

    print('  Waiting for camera data...')
    interval = 1.0 / VIZ_HZ
    seen_cameras: set = set()
    robot_online = False
    robot_warned = False

    while True:
        t0 = time.monotonic()
        points, joints = node.snapshot()

        # ── point clouds ────────────────────────────────────────────────────
        for cam_id, pts in points.items():
            if pts.shape[0] == 0:
                continue
            if cam_id not in seen_cameras:
                print(f'  Camera online: {cam_id} ({pts.shape[0]} pts/frame)')
                seen_cameras.add(cam_id)
            colors = np.tile(CAM_COLOR[cam_id], (pts.shape[0], 1))
            server.scene.add_point_cloud(
                f'/map/{cam_id}',
                points=pts,
                colors=colors,
                point_size=0.005,
            )

        # ── robot ────────────────────────────────────────────────────────────
        if robot is not None and joints:
            cfg = {j: joints[j] for j in JOINT_NAMES if j in joints}
            missing = [j for j in JOINT_NAMES if j not in joints]
            if missing and not robot_warned:
                print(f'  [JOINTS] Available: {sorted(joints.keys())}')
                print(f'  [JOINTS] Missing from /joint_states: {missing}')
                robot_warned = True
            elif len(cfg) == len(JOINT_NAMES):
                try:
                    robot.update_cfg(cfg)
                    if not robot_online:
                        print('  Robot joints online.')
                        robot_online = True
                except Exception as e:
                    if not robot_warned:
                        print(f'  [ROBOT] update_cfg failed: {e}')
                        print(f'  [ROBOT] cfg = {cfg}')
                        robot_warned = True

        elapsed = time.monotonic() - t0
        time.sleep(max(0.0, interval - elapsed))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Live depth-camera map + UR5 joint state visualizer')
    parser.add_argument('--target', nargs=3, type=float,
                        metavar=('X', 'Y', 'Z'),
                        help='World-frame XYZ of the goal to mark (e.g. 0.5 0.0 0.3)')
    args = parser.parse_args()

    rclpy.init()
    node = PerceptionNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    print('ROS2 node started — subscribing to cameras and /joint_states')

    try:
        run_viser(node, args.target)
    except KeyboardInterrupt:
        print('\nStopped.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
