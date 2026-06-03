"""Debug visualization node.

Hosts a single viser server and renders the live pipeline state by subscribing
to lightweight ROS topics:

  /graspgen/segmented_object, /graspgen/background  (sensor_msgs/PointCloud2)
  /graspgen/grasp_poses                             (geometry_msgs/PoseArray)
  /curobo/tsdf_voxels                               (sensor_msgs/PointCloud2)

Visualization is fully decoupled: the heavy nodes publish; this node only reads.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
import viser
from geometry_msgs.msg import PoseArray
from sensor_msgs.msg import PointCloud2


def colors_by_rank(n: int) -> list[tuple[int, int, int]]:
    """Green (best rank) → red (worst rank) RGB tuples for n grasps."""
    if n <= 0:
        return []
    colors = []
    for i in range(n):
        t = i / max(n - 1, 1)
        colors.append((int(round(255 * t)), int(round(255 * (1 - t))), 0))
    return colors


def pose_to_position_wxyz(pose):
    """Extract ((x, y, z), (w, x, y, z)) from a geometry_msgs/Pose."""
    position = (float(pose.position.x), float(pose.position.y), float(pose.position.z))
    wxyz = (
        float(pose.orientation.w),
        float(pose.orientation.x),
        float(pose.orientation.y),
        float(pose.orientation.z),
    )
    return position, wxyz


def _cloud_to_xyz(msg: PointCloud2) -> np.ndarray:
    """Decode a PointCloud2 into an (N, 3) float32 array.

    Assumes X, Y, Z are the first three float32 fields (byte offsets 0/4/8);
    any trailing fields in point_step are ignored. All current producers
    (make_xyz_cloud, segmentation clouds, TSDF voxels) satisfy this.
    """
    if msg.width * msg.height == 0:
        return np.empty((0, 3), dtype=np.float32)
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    raw = raw.reshape(msg.height * msg.width, msg.point_step)
    # .copy(): the column slice is non-contiguous; make it contiguous before .view().
    xyz = raw[:, 0:12].copy().view(np.float32).reshape(-1, 3)
    return np.nan_to_num(xyz, nan=0.0)


class DebugVizNode(Node):
    """Subscribes to pipeline viz topics and renders them in viser."""

    def __init__(self) -> None:
        super().__init__('debug_viz')
        self.declare_parameter('viser_port', 8080)
        self.declare_parameter('segmented_point_cloud_topic', '/graspgen/segmented_object')
        self.declare_parameter('background_point_cloud_topic', '/graspgen/background')
        self.declare_parameter('grasp_poses_topic', '/graspgen/grasp_poses')
        self.declare_parameter('tsdf_voxels_topic', '/curobo/tsdf_voxels')
        self.declare_parameter('grasp_frame_axes_length', 0.05)

        self._server = viser.ViserServer(port=int(self.get_parameter('viser_port').value))
        self.get_logger().info(
            f'debug_viz viser server on http://0.0.0.0:'
            f'{int(self.get_parameter("viser_port").value)}'
        )
        self._grasp_frames = []

        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.RELIABLE

        self.create_subscription(
            PointCloud2,
            str(self.get_parameter('segmented_point_cloud_topic').value),
            lambda m: self._on_cloud(m, '/segmented', (50, 220, 50)),
            qos,
        )
        self.create_subscription(
            PointCloud2,
            str(self.get_parameter('background_point_cloud_topic').value),
            lambda m: self._on_cloud(m, '/background', (140, 140, 140)),
            qos,
        )
        self.create_subscription(
            PointCloud2,
            str(self.get_parameter('tsdf_voxels_topic').value),
            lambda m: self._on_cloud(m, '/tsdf', (60, 120, 255)),
            qos,
        )
        self.create_subscription(
            PoseArray,
            str(self.get_parameter('grasp_poses_topic').value),
            self._on_grasp_poses,
            qos,
        )

    def _on_cloud(self, msg: PointCloud2, name: str, color: tuple) -> None:
        xyz = _cloud_to_xyz(msg)
        if len(xyz) == 0:
            return
        colors = np.tile(np.asarray(color, dtype=np.uint8), (len(xyz), 1))
        self._server.scene.add_point_cloud(name, points=xyz, colors=colors, point_size=0.004)

    def _on_grasp_poses(self, msg: PoseArray) -> None:
        # Remove the previous render's frames, then add the current ranked set.
        for handle in self._grasp_frames:
            handle.remove()
        self._grasp_frames = []
        axes_length = float(self.get_parameter('grasp_frame_axes_length').value)
        colors = colors_by_rank(len(msg.poses))
        for i, pose in enumerate(msg.poses):
            position, wxyz = pose_to_position_wxyz(pose)
            handle = self._server.scene.add_frame(
                f'/grasps/{i:03d}',
                wxyz=wxyz,
                position=position,
                axes_length=axes_length,
                axes_radius=axes_length * 0.1,
                origin_radius=axes_length * 0.2,
                origin_color=colors[i],
            )
            self._grasp_frames.append(handle)
        self.get_logger().info(
            f'debug_viz rendered {len(msg.poses)} grasp frames'
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DebugVizNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
