import threading
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2


class NvBlox:
    """Wraps the Isaac ROS nvblox ROS2 node interface.

    The nvblox node runs as a separate process inside the container and fuses
    overhead + wrist RGBD streams into a persistent TSDF/ESDF. This class
    subscribes to the published ESDF topic and exposes two interfaces:
      - get_esdf()              → cuRobo WorldNvbloxCollision
      - extract_object_cloud()  → GraspGen (N, 3) point cloud in robot frame
    """

    ESDF_TOPIC = '/nvblox_node/static_esdf_pointcloud'

    def __init__(self, node: Node):
        self._node = node
        self._logger = node.get_logger()
        self._esdf_msg = None
        self._lock = threading.Lock()

        self._esdf_sub = node.create_subscription(
            PointCloud2,
            self.ESDF_TOPIC,
            self._on_esdf,
            10,
        )
        self._logger.info('NvBlox: subscribing to %s' % self.ESDF_TOPIC)

    def _on_esdf(self, msg: PointCloud2):
        with self._lock:
            self._esdf_msg = msg

    def get_esdf(self) -> PointCloud2 | None:
        """Return latest ESDF PointCloud2 from nvblox.

        Returns None until the nvblox node publishes its first map.
        Subsequent calls return the cached, auto-updating message.
        """
        with self._lock:
            return self._esdf_msg

    def extract_object_cloud(
        self,
        mask_2d: np.ndarray,
        camera: str = 'overhead',
    ) -> np.ndarray | None:
        """Extract object point cloud from nvblox mesh using a 2D segmentation mask.

        Args:
            mask_2d: Boolean (H, W) mask from SAM2 in the specified camera's image space.
            camera: Which camera the mask belongs to ('overhead' or 'wrist').

        Returns (N, 3) float32 point cloud in robot base frame, or None on failure.
        """
        # TODO: subscribe to /nvblox_node/mesh (nvblox_msgs/Mesh)
        # TODO: project mask_2d pixel rays into nvblox mesh via raycasting
        # TODO: transform resulting 3D points to robot base frame using TF2
        self._logger.warn('NvBlox.extract_object_cloud not yet implemented.')
        return None
