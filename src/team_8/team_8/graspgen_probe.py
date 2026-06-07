import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

from team_8.graspgen_client import GraspGenClient


class GraspGenProbe(Node):
    """Subscribe to wrist point cloud and send periodic inference requests to GraspGen."""

    DEFAULT_TOPIC = "/wrist_camera/wrist_camera/depth/color/points"

    def __init__(self):
        super().__init__("graspgen_probe")

        self.declare_parameter("point_cloud_topic", self.DEFAULT_TOPIC)
        self.declare_parameter("server_host", "127.0.0.1")
        self.declare_parameter("server_port", 5556)
        self.declare_parameter("request_period_sec", 3.0)
        self.declare_parameter("max_points", 4096)
        self.declare_parameter("workspace_min", [-1.0, -1.0, 0.05])
        self.declare_parameter("workspace_max", [1.5, 1.0, 2.0])
        self.declare_parameter("num_grasps", 200)
        self.declare_parameter("topk_num_grasps", 20)
        self.declare_parameter("gripper_name", "robotiq_2f_85")

        topic = self.get_parameter("point_cloud_topic").value
        host = self.get_parameter("server_host").value
        port = int(self.get_parameter("server_port").value)
        period = float(self.get_parameter("request_period_sec").value)

        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT

        self._latest_cloud = None
        self._latest_frame_id = ""
        self._request_in_flight = False
        self._gripper_name = str(self.get_parameter("gripper_name").value)

        self._client = GraspGenClient(host=host, port=port)
        self.get_logger().info(
            f"Connected to GraspGen server at {host}:{port} metadata={self._client.server_metadata}"
        )

        self.create_subscription(PointCloud2, topic, self._point_cloud_callback, qos)
        self.create_timer(period, self._timer_callback)

        self.get_logger().info(f"Listening for point clouds on {topic}")

    def _point_cloud_callback(self, msg: PointCloud2) -> None:
        points = point_cloud2.read_points_numpy(msg, field_names=("x", "y", "z"))
        if points.size == 0:
            return

        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            return

        finite_mask = np.isfinite(points).all(axis=1)
        points = points[finite_mask]
        if len(points) == 0:
            return

        workspace_min = np.asarray(self.get_parameter("workspace_min").value, dtype=np.float32)
        workspace_max = np.asarray(self.get_parameter("workspace_max").value, dtype=np.float32)
        in_bounds = np.logical_and(points >= workspace_min, points <= workspace_max).all(axis=1)
        points = points[in_bounds]
        if len(points) == 0:
            return

        max_points = int(self.get_parameter("max_points").value)
        if len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            points = points[indices]

        self._latest_cloud = points
        self._latest_frame_id = msg.header.frame_id

    def _timer_callback(self) -> None:
        if self._latest_cloud is None or self._request_in_flight:
            return

        cloud = self._latest_cloud
        self._request_in_flight = True
        try:
            grasps, confidences = self._client.infer(
                cloud,
                gripper_name=self._gripper_name,
                num_grasps=int(self.get_parameter("num_grasps").value),
                topk_num_grasps=int(self.get_parameter("topk_num_grasps").value),
            )
        except Exception as exc:
            self.get_logger().error(f"GraspGen inference failed: {exc}")
            return
        finally:
            self._request_in_flight = False

        if len(grasps) == 0:
            self.get_logger().warn(
                f"No grasps returned for cloud with {len(cloud)} points in frame {self._latest_frame_id}"
            )
            return

        top_conf = float(confidences.max())
        top_idx = int(confidences.argmax())
        top_grasp = grasps[top_idx]
        translation = top_grasp[:3, 3]
        self.get_logger().info(
            "Received "
            f"{len(grasps)} grasps from frame {self._latest_frame_id}; "
            f"best confidence={top_conf:.3f} "
            f"translation=[{translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f}]"
        )

    def destroy_node(self):
        self._client.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = GraspGenProbe()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
