import json

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_srvs.srv import Trigger

from pipeline_orchestrator.graspgen_client import GraspGenClient

try:
    from grasp_gen.robot import get_gripper_info
    from grasp_gen.utils.point_cloud_utils import filter_colliding_grasps
except ImportError:  # pragma: no cover - import is environment-dependent
    get_gripper_info = None
    filter_colliding_grasps = None


class GraspGenService(Node):
    """Serve filtered GraspGen results for the latest segmented point cloud."""

    def __init__(self):
        super().__init__("graspgen_service")

        self.declare_parameter("segmented_point_cloud_topic", "/graspgen/segmented_object")
        self.declare_parameter("background_point_cloud_topic", "/graspgen/background")
        self.declare_parameter("service_name", "/graspgen/infer")
        self.declare_parameter("server_host", "127.0.0.1")
        self.declare_parameter("server_port", 5557)
        self.declare_parameter("num_grasps", 200)
        self.declare_parameter("topk_num_grasps", 100)
        self.declare_parameter("min_grasps", 20)
        self.declare_parameter("max_tries", 4)
        self.declare_parameter("remove_outliers", True)
        self.declare_parameter("rank_mode", "horizontal_grasp")
        self.declare_parameter("target_approach_dir", [0.0, 0.0, 1.0])
        self.declare_parameter("max_returned_grasps", 5)
        self.declare_parameter("enable_collision_check", False)
        self.declare_parameter("collision_threshold", 0.002)
        self.declare_parameter("collision_samples", 2000)

        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT

        self._latest_segmented_cloud = None
        self._latest_background_cloud = None
        self._latest_segmented_frame = ""
        self._latest_background_frame = ""

        host = str(self.get_parameter("server_host").value)
        port = int(self.get_parameter("server_port").value)
        self._client = GraspGenClient(host=host, port=port)
        self.get_logger().info(
            f"Connected to GraspGen server at {host}:{port} metadata={self._client.server_metadata}"
        )

        segmented_topic = str(self.get_parameter("segmented_point_cloud_topic").value)
        background_topic = str(self.get_parameter("background_point_cloud_topic").value)
        service_name = str(self.get_parameter("service_name").value)

        self.create_subscription(
            PointCloud2, segmented_topic, self._segmented_cloud_callback, qos
        )
        self.create_subscription(
            PointCloud2, background_topic, self._background_cloud_callback, qos
        )
        self.create_service(Trigger, service_name, self._infer_callback)

        self.get_logger().info(
            f"Listening for segmented clouds on {segmented_topic}, background clouds on {background_topic}, "
            f"service={service_name}"
        )

    def _segmented_cloud_callback(self, msg: PointCloud2) -> None:
        cloud = self._pointcloud2_to_xyz(msg)
        if len(cloud) == 0:
            return
        self._latest_segmented_cloud = cloud
        self._latest_segmented_frame = msg.header.frame_id

    def _background_cloud_callback(self, msg: PointCloud2) -> None:
        cloud = self._pointcloud2_to_xyz(msg)
        if len(cloud) == 0:
            return
        self._latest_background_cloud = cloud
        self._latest_background_frame = msg.header.frame_id

    def _pointcloud2_to_xyz(self, msg: PointCloud2) -> np.ndarray:
        points = point_cloud2.read_points_numpy(msg, field_names=("x", "y", "z"))
        if points.size == 0:
            return np.empty((0, 3), dtype=np.float32)

        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            return np.empty((0, 3), dtype=np.float32)

        finite_mask = np.isfinite(points).all(axis=1)
        return points[finite_mask]

    def _infer_callback(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request
        if self._latest_segmented_cloud is None:
            response.success = False
            response.message = "No segmented point cloud received yet."
            return response

        segmented_cloud = self._latest_segmented_cloud
        try:
            grasps, confidences = self._client.infer(
                segmented_cloud,
                num_grasps=int(self.get_parameter("num_grasps").value),
                topk_num_grasps=int(self.get_parameter("topk_num_grasps").value),
                min_grasps=int(self.get_parameter("min_grasps").value),
                max_tries=int(self.get_parameter("max_tries").value),
                remove_outliers=bool(self.get_parameter("remove_outliers").value),
            )
        except Exception as exc:
            response.success = False
            response.message = f"GraspGen inference failed: {exc}"
            return response

        if len(grasps) == 0:
            response.success = False
            response.message = "No grasps returned."
            return response

        collision_free_mask = self._compute_collision_free_mask(grasps)
        filtered_rows = self._rank_grasps(grasps, confidences, collision_free_mask)
        max_returned = int(self.get_parameter("max_returned_grasps").value)
        top_rows = filtered_rows[:max_returned]

        payload = {
            "frame_id": self._latest_segmented_frame,
            "num_input_points": int(len(segmented_cloud)),
            "num_grasps": int(len(grasps)),
            "num_collision_free_grasps": int(np.sum(collision_free_mask))
            if collision_free_mask is not None
            else None,
            "rank_mode": str(self.get_parameter("rank_mode").value),
            "top_grasps": top_rows,
        }
        response.success = len(top_rows) > 0
        response.message = json.dumps(payload)
        self.get_logger().info(
            f"Returned {len(top_rows)} filtered grasps from {len(grasps)} raw grasps "
            f"for frame {self._latest_segmented_frame}"
        )
        return response

    def _compute_collision_free_mask(self, grasps: np.ndarray):
        if not bool(self.get_parameter("enable_collision_check").value):
            return None
        if self._latest_background_cloud is None:
            self.get_logger().warn("Collision filtering enabled but no background cloud received yet.")
            return None
        if get_gripper_info is None or filter_colliding_grasps is None:
            self.get_logger().warn("Collision filtering requested but GraspGen collision utilities are unavailable.")
            return None

        gripper_name = self._client.server_metadata.get("gripper_name", "robotiq_2f_140")
        gripper_info = get_gripper_info(gripper_name)
        collision_threshold = float(self.get_parameter("collision_threshold").value)
        collision_samples = int(self.get_parameter("collision_samples").value)
        return filter_colliding_grasps(
            self._latest_background_cloud.astype(np.float32),
            np.asarray(grasps, dtype=np.float32),
            gripper_info.collision_mesh,
            collision_threshold=collision_threshold,
            num_collision_samples=collision_samples,
        )

    def _rank_grasps(self, grasps: np.ndarray, confidences: np.ndarray, collision_free_mask):
        rank_mode = str(self.get_parameter("rank_mode").value)
        target_dir = np.asarray(
            self.get_parameter("target_approach_dir").value, dtype=np.float32
        )
        target_dir = target_dir / max(np.linalg.norm(target_dir), 1e-6)

        rows = []
        for grasp, confidence in zip(grasps, confidences):
            rows.append(self._build_rank_row(grasp, float(confidence), rank_mode, target_dir))

        if collision_free_mask is not None:
            rows = [row for row, keep in zip(rows, collision_free_mask) if keep]

        if rank_mode == "approach_alignment":
            rows.sort(
                key=lambda row: (row["alignment"], row["confidence"]),
                reverse=True,
            )
        else:
            rows.sort(
                key=lambda row: (
                    row["horizontal_score"],
                    row["approach_flatness"],
                    row["finger_flatness"],
                    row["confidence"],
                ),
                reverse=True,
            )
        return rows

    def _build_rank_row(
        self,
        grasp: np.ndarray,
        confidence: float,
        rank_mode: str,
        target_dir: np.ndarray,
    ) -> dict:
        approach = np.asarray(grasp[:3, 2], dtype=np.float32)
        finger = np.asarray(grasp[:3, 0], dtype=np.float32)
        approach = approach / max(np.linalg.norm(approach), 1e-6)
        finger = finger / max(np.linalg.norm(finger), 1e-6)

        row = {
            "confidence": round(confidence, 4),
            "translation": [round(float(x), 4) for x in grasp[:3, 3]],
            "rotation_matrix": [[round(float(v), 4) for v in r] for r in grasp[:3, :3]],
        }

        if rank_mode == "approach_alignment":
            row["alignment"] = round(float(np.dot(approach, target_dir)), 4)
            return row

        approach_flatness = 1.0 - abs(float(approach[2]))
        finger_flatness = 1.0 - abs(float(finger[2]))
        row["approach_flatness"] = round(approach_flatness, 4)
        row["finger_flatness"] = round(finger_flatness, 4)
        row["horizontal_score"] = round(approach_flatness * finger_flatness, 4)
        return row

    def destroy_node(self):
        self._client.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = GraspGenService()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
