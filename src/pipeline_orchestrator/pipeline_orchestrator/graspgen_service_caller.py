import json
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from std_srvs.srv import Trigger

from pipeline_orchestrator.pipeline_utils import make_xyz_cloud


class GraspGenServiceCaller(Node):
    def __init__(self):
        super().__init__("graspgen_service_caller")

        self.declare_parameter(
            "segmented_object_file",
            "/ros2_ws/segmented_objects/segmented_object_banana.npy",
        )
        self.declare_parameter("background_object_file", "")
        self.declare_parameter("segmented_point_cloud_topic", "/graspgen/segmented_object")
        self.declare_parameter("background_point_cloud_topic", "/graspgen/background")
        self.declare_parameter("service_name", "/graspgen/infer")
        self.declare_parameter("frame_id", "base_link")

        segmented_topic = str(self.get_parameter("segmented_point_cloud_topic").value)
        background_topic = str(self.get_parameter("background_point_cloud_topic").value)
        self._service_name = str(self.get_parameter("service_name").value)
        self._frame_id = str(self.get_parameter("frame_id").value)

        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT

        self._segmented_pub = self.create_publisher(PointCloud2, segmented_topic, qos)
        self._background_pub = self.create_publisher(PointCloud2, background_topic, qos)
        self._client = self.create_client(Trigger, self._service_name)

    def run(self) -> int:
        segmented_path = Path(str(self.get_parameter("segmented_object_file").value))
        background_raw = str(self.get_parameter("background_object_file").value)
        background_path = Path(background_raw) if background_raw else None

        if not segmented_path.exists():
            self.get_logger().error(f"Segmented object file not found: {segmented_path}")
            return 1
        if background_path is not None and not background_path.exists():
            self.get_logger().error(f"Background object file not found: {background_path}")
            return 1

        segmented_points = np.load(segmented_path)
        segmented_msg = make_xyz_cloud(segmented_points, self._frame_id)

        if background_path is not None:
            background_points = np.load(background_path)
            background_msg = make_xyz_cloud(background_points, self._frame_id)
        else:
            background_msg = None

        for _ in range(10):
            self._segmented_pub.publish(segmented_msg)
            if background_msg is not None:
                self._background_pub.publish(background_msg)
            rclpy.spin_once(self, timeout_sec=0.2)

        self.get_logger().info(
            f"Published segmented cloud {segmented_path.name} with {len(segmented_points)} points"
        )
        if background_msg is not None:
            self.get_logger().info(
                f"Published background cloud {background_path.name} with {len(background_points)} points"
            )
        if not self._client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error(f"Service not available: {self._service_name}")
            return 1

        future = self._client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=120.0)
        if not future.done() or future.result() is None:
            self.get_logger().error("Service call failed or timed out.")
            return 1

        result = future.result()
        self.get_logger().info(f"service_success={result.success}")
        try:
            parsed = json.loads(result.message)
            print(json.dumps(parsed, indent=2))
        except json.JSONDecodeError:
            print(result.message)
        return 0 if result.success else 1


def main(args=None):
    rclpy.init(args=args)
    node = GraspGenServiceCaller()
    try:
        raise SystemExit(node.run())
    finally:
        node.destroy_node()
        rclpy.shutdown()
