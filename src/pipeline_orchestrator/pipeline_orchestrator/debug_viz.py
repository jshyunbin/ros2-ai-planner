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


def height_colormap(z: np.ndarray) -> np.ndarray:
    """Map z heights to (N, 3) uint8 RGB via a blue→cyan→green→yellow→red ramp.

    Normalized over the 2nd–98th percentile so a few outlier voxels don't
    flatten the colour range.
    """
    z = np.asarray(z, dtype=np.float32)
    lo, hi = np.percentile(z, 2.0), np.percentile(z, 98.0)
    if hi - lo < 1e-6:
        t = np.zeros_like(z)
    else:
        t = np.clip((z - lo) / (hi - lo), 0.0, 1.0)
    stops = np.array(
        [
            [0.0, 0.0, 255.0],    # blue
            [0.0, 255.0, 255.0],  # cyan
            [0.0, 255.0, 0.0],    # green
            [255.0, 255.0, 0.0],  # yellow
            [255.0, 0.0, 0.0],    # red
        ],
        dtype=np.float32,
    )
    pos = np.linspace(0.0, 1.0, len(stops))
    rgb = np.stack([np.interp(t, pos, stops[:, c]) for c in range(3)], axis=-1)
    return rgb.astype(np.uint8)


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
        self.declare_parameter('overhead_cloud_topic', '/curobo/overhead_cloud')
        self.declare_parameter('grasp_poses_topic', '/graspgen/grasp_poses')
        self.declare_parameter('tsdf_voxels_topic', '/curobo/tsdf_voxels')
        self.declare_parameter('grasp_frame_axes_length', 0.05)
        self.declare_parameter('point_size', 0.003)
        self.declare_parameter('tsdf_voxel_size', 0.02)

        self._server = viser.ViserServer(port=int(self.get_parameter('viser_port').value))
        self.get_logger().info(
            f'debug_viz viser server on http://0.0.0.0:'
            f'{int(self.get_parameter("viser_port").value)}'
        )
        self._point_size = float(self.get_parameter('point_size').value)
        self._tsdf_voxel_size = float(self.get_parameter('tsdf_voxel_size').value)

        # Per-layer visibility, current scene handle(s), and toggle checkboxes.
        # 'grasps' tracks a list of frame handles; the rest a single cloud handle.
        self._layer_visible = {
            'segmented': True,
            'background': True,
            'overhead': True,
            'tsdf': True,
            'grasps': True,
        }
        self._cloud_handles: dict = {}
        self._grasp_frames = []
        self._build_layer_toggles()

        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.RELIABLE

        self.create_subscription(
            PointCloud2,
            str(self.get_parameter('segmented_point_cloud_topic').value),
            lambda m: self._on_cloud(m, 'segmented', (50, 220, 50)),
            qos,
        )
        self.create_subscription(
            PointCloud2,
            str(self.get_parameter('background_point_cloud_topic').value),
            lambda m: self._on_cloud(m, 'background', (140, 140, 140)),
            qos,
        )
        self.create_subscription(
            PointCloud2,
            str(self.get_parameter('overhead_cloud_topic').value),
            lambda m: self._on_cloud(m, 'overhead', (255, 160, 40)),
            qos,
        )
        self.create_subscription(
            PointCloud2,
            str(self.get_parameter('tsdf_voxels_topic').value),
            self._on_tsdf,
            qos,
        )
        self.create_subscription(
            PoseArray,
            str(self.get_parameter('grasp_poses_topic').value),
            self._on_grasp_poses,
            qos,
        )

    def _build_layer_toggles(self) -> None:
        """Add a 'Layers' GUI folder with a visibility checkbox per layer."""
        labels = {
            'segmented': 'Segmented',
            'background': 'Background',
            'overhead': 'Overhead',
            'tsdf': 'TSDF',
            'grasps': 'Grasps',
        }
        with self._server.gui.add_folder('Layers'):
            for layer, label in labels.items():
                checkbox = self._server.gui.add_checkbox(label, True)
                checkbox.on_update(self._make_toggle(layer, checkbox))

    def _make_toggle(self, layer: str, checkbox):
        def _on_update(_event) -> None:
            visible = bool(checkbox.value)
            self._layer_visible[layer] = visible
            if layer == 'grasps':
                for handle in self._grasp_frames:
                    handle.visible = visible
            else:
                handle = self._cloud_handles.get(layer)
                if handle is not None:
                    handle.visible = visible
        return _on_update

    def _on_cloud(self, msg: PointCloud2, layer: str, color: tuple) -> None:
        xyz = _cloud_to_xyz(msg)
        if len(xyz) == 0:
            return
        colors = np.tile(np.asarray(color, dtype=np.uint8), (len(xyz), 1))
        handle = self._server.scene.add_point_cloud(
            f'/{layer}',
            points=xyz,
            colors=colors,
            point_size=self._point_size,
            point_shape='square',
        )
        handle.visible = self._layer_visible[layer]
        self._cloud_handles[layer] = handle

    def _on_tsdf(self, msg: PointCloud2) -> None:
        # Square points sized to the voxel resolution give the occupied TSDF a
        # solid, blocky look; colour each voxel by height.
        xyz = _cloud_to_xyz(msg)
        if len(xyz) == 0:
            return
        colors = height_colormap(xyz[:, 2])
        handle = self._server.scene.add_point_cloud(
            '/tsdf',
            points=xyz,
            colors=colors,
            point_size=self._tsdf_voxel_size,
            point_shape='square',
        )
        handle.visible = self._layer_visible['tsdf']
        self._cloud_handles['tsdf'] = handle

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
            handle.visible = self._layer_visible['grasps']
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
