import json
import os
from pathlib import Path
import threading
import time

import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseArray
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

from team_8.graspgen_client import GraspGenClient
from team_8.pipeline_utils import pose_from_grasp_row

try:  # pragma: no cover - runtime dependency
    from riro_srvs.srv import StringString
except ImportError:  # pragma: no cover - runtime dependency
    StringString = None

try:
    # resolve_gripper_info reads gripper_descriptions assets (x_grippers/<name>),
    # the SAME source the GraspGenX server uses — unlike graspgenx.robot.get_gripper_info,
    # which globs the empty graspgenx/config/grippers and raises in serving.
    from graspgenx.x_grippers import resolve_gripper_info
    from graspgenx.utils.collision_filter import filter_colliding_grasps
except ImportError:  # pragma: no cover - import is environment-dependent
    resolve_gripper_info = None
    filter_colliding_grasps = None

# Single source of truth for the gripper assets + TCP z-offset, shared with
# curobo.py so grasp generation and motion planning stay aligned. See gripper_tcp.py.
from team_8.gripper_tcp import (
    DEFAULT_GRIPPER_NAME as _DEFAULT_GRIPPER_NAME,
    gripper_assets_dir as _gripper_assets_dir,
    resolve_gripper_tcp_z_offset as _resolve_gripper_tcp_z_offset,
)

# TCP is _GRIPPER_TCP_Z_OFFSET ahead of tool0 along the grasp +Z axis.
_GRIPPER_TCP_Z_OFFSET, _GRIPPER_TCP_Z_OFFSET_SOURCE = _resolve_gripper_tcp_z_offset()
_MAX_REACH = 0.82                 # m — UR5 kinematic reach limit
_MIN_TOOL_Z = 0.08                # m — minimum tool0 height above table


class GraspGenService(Node):
    """Serve filtered GraspGen results for the latest segmented point cloud."""

    def __init__(self):
        if StringString is None:
            raise ImportError("riro_srvs is required for graspgen_service.")

        super().__init__("graspgen_service")

        self.get_logger().info(
            f"Gripper TCP z-offset = {_GRIPPER_TCP_Z_OFFSET:.4f} m "
            f"(source: {_GRIPPER_TCP_Z_OFFSET_SOURCE})"
        )

        self.declare_parameter("segmented_point_cloud_topic", "/graspgen/segmented_object")
        self.declare_parameter("background_point_cloud_topic", "/graspgen/background")
        self.declare_parameter("service_name", "/graspgen/infer")
        self.declare_parameter("server_host", "127.0.0.1")
        self.declare_parameter("server_port", 5556)
        self.declare_parameter("num_grasps", 200)
        self.declare_parameter("topk_num_grasps", 100)
        # NOTE: min_grasps / max_tries / remove_outliers were removed — GraspGenX's
        # infer() no longer supports them (the old outlier-retry loop is gone).
        self.declare_parameter("rank_mode", "approach_alignment")
        self.declare_parameter("target_approach_dir", [0.0, 0.0, -1.0])
        self.declare_parameter("max_returned_grasps", 5)
        self.declare_parameter("expected_frame", "base_link")
        self.declare_parameter("enable_collision_check", False)
        # NOTE: GraspGenX's filter_colliding_grasps default is 0.02 m; this 0.002 m
        # (2 mm) gate is intentionally tighter for this tabletop scene. Only used
        # when enable_collision_check is True (off by default).
        self.declare_parameter("collision_threshold", 0.002)
        self.declare_parameter("collision_samples", 2000)
        self.declare_parameter("debug_dir", "/artifacts/graspgen_service")
        self.declare_parameter("cloud_wait_sec", 5.0)
        self.declare_parameter("publish_grasp_poses", False)
        self.declare_parameter("grasp_poses_topic", "/graspgen/grasp_poses")

        # RELIABLE to match the segmentation publishers; the cloud is bulk
        # request/response data, not a high-rate stream.
        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.RELIABLE

        self._cloud_wait_sec = float(self.get_parameter("cloud_wait_sec").value)
        # _cloud_cv guards the latest-cloud state and is notified whenever a new
        # segmented cloud arrives, so the service handler can wait for the cloud
        # that matches its request token.
        self._cloud_cv = threading.Condition()
        self._latest_segmented_cloud = None
        self._latest_background_cloud = None
        self._latest_segmented_frame = ""
        self._latest_background_frame = ""
        self._latest_segmented_stamp_ns = 0
        self._debug_dir = Path(str(self.get_parameter("debug_dir").value))
        self._debug_dir.mkdir(parents=True, exist_ok=True)

        host = str(self.get_parameter("server_host").value)
        port = int(self.get_parameter("server_port").value)
        self._client = GraspGenClient(host=host, port=port)
        self.get_logger().info(
            f"Connected to GraspGen server at {host}:{port} metadata={self._client.server_metadata}"
        )

        segmented_topic = str(self.get_parameter("segmented_point_cloud_topic").value)
        background_topic = str(self.get_parameter("background_point_cloud_topic").value)
        service_name = str(self.get_parameter("service_name").value)

        # Cloud subscriptions live in their own reentrant group so they can be
        # serviced (under the MultiThreadedExecutor in main()) while the service
        # handler is blocked waiting for the cloud that matches its token.
        cloud_group = ReentrantCallbackGroup()
        self.create_subscription(
            PointCloud2, segmented_topic, self._segmented_cloud_callback, qos,
            callback_group=cloud_group,
        )
        self.create_subscription(
            PointCloud2, background_topic, self._background_cloud_callback, qos,
            callback_group=cloud_group,
        )
        self.create_service(StringString, service_name, self._infer_callback)

        self._grasp_poses_pub = None
        if bool(self.get_parameter("publish_grasp_poses").value):
            self._grasp_poses_pub = self.create_publisher(
                PoseArray,
                str(self.get_parameter("grasp_poses_topic").value),
                qos,
            )

        self.get_logger().info(
            f"Listening for segmented clouds on {segmented_topic}, background clouds on {background_topic}, "
            f"service={service_name}"
        )

    def _segmented_cloud_callback(self, msg: PointCloud2) -> None:
        cloud = self._pointcloud2_to_xyz(msg)
        if len(cloud) == 0:
            return
        with self._cloud_cv:
            self._latest_segmented_cloud = cloud
            self._latest_segmented_frame = msg.header.frame_id
            self._latest_segmented_stamp_ns = self._stamp_to_ns(msg.header.stamp)
            self._cloud_cv.notify_all()

    def _background_cloud_callback(self, msg: PointCloud2) -> None:
        cloud = self._pointcloud2_to_xyz(msg)
        if len(cloud) == 0:
            return
        with self._cloud_cv:
            self._latest_background_cloud = cloud
            self._latest_background_frame = msg.header.frame_id

    @staticmethod
    def _stamp_to_ns(stamp) -> int:
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    @staticmethod
    def _parse_token(data: str):
        """Parse the request token into a nanosecond stamp, or None for 'latest'."""
        text = (data or "").strip()
        if not text:
            return None
        return int(text)

    def _wait_for_cloud(self, requested_ns: int) -> bool:
        """Block until a segmented cloud at >= requested_ns is cached, or timeout."""
        deadline = time.monotonic() + self._cloud_wait_sec
        with self._cloud_cv:
            while self._latest_segmented_stamp_ns < requested_ns:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                self._cloud_cv.wait(timeout=remaining)
            return True

    def _pointcloud2_to_xyz(self, msg: PointCloud2) -> np.ndarray:
        points = point_cloud2.read_points_numpy(msg, field_names=("x", "y", "z"))
        if points.size == 0:
            return np.empty((0, 3), dtype=np.float32)

        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            return np.empty((0, 3), dtype=np.float32)

        finite_mask = np.isfinite(points).all(axis=1)
        return points[finite_mask]

    def _infer_callback(self, request, response):
        """StringString service: request.data is the cloud token, response.data is JSON.

        An empty token means "use the latest cloud"; a nanosecond stamp means
        "wait for the segmented cloud with that stamp, then infer".
        """
        try:
            requested_ns = self._parse_token(request.data)
        except (TypeError, ValueError):
            response.data = json.dumps({
                "success": False,
                "error": f"Invalid request token (expected nanosecond stamp): {request.data!r}",
            })
            return response

        if requested_ns is not None and not self._wait_for_cloud(requested_ns):
            response.data = json.dumps({
                "success": False,
                "error": (
                    f"Timed out waiting for segmented cloud stamp {requested_ns} "
                    f"(waited {self._cloud_wait_sec:.1f}s)."
                ),
            })
            return response

        response.data = json.dumps(self._run_inference())
        return response

    def _run_inference(self) -> dict:
        with self._cloud_cv:
            segmented_cloud = self._latest_segmented_cloud
            segmented_frame = self._latest_segmented_frame
            background_cloud = self._latest_background_cloud
            background_frame = self._latest_background_frame

        if segmented_cloud is None:
            return {"success": False, "error": "No segmented point cloud received yet."}

        expected_frame = str(self.get_parameter("expected_frame").value)
        if expected_frame and segmented_frame != expected_frame:
            return {
                "success": False,
                "error": (
                    "Segmented point cloud frame mismatch: "
                    f"got {segmented_frame!r}, expected {expected_frame!r}."
                ),
            }
        if (
            background_cloud is not None
            and expected_frame
            and background_frame
            and background_frame != expected_frame
        ):
            return {
                "success": False,
                "error": (
                    "Background point cloud frame mismatch: "
                    f"got {background_frame!r}, expected {expected_frame!r}."
                ),
            }

        debug_id = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time_ns() % 1_000_000_000):09d}"
        debug_path = self._debug_dir / debug_id
        debug_path.mkdir(parents=True, exist_ok=True)
        try:
            # gripper_name omitted -> the GraspGenX server uses its --default_gripper
            # (robotiq_2f_85). num_grasps / topk_num_grasps are the only tunables
            # GraspGenX's infer() still accepts (plus grasp_threshold, left default).
            grasps, confidences = self._client.infer(
                segmented_cloud,
                num_grasps=int(self.get_parameter("num_grasps").value),
                topk_num_grasps=int(self.get_parameter("topk_num_grasps").value),
            )
        except Exception as exc:
            error = f"GraspGen inference failed: {exc}"
            self._save_debug_artifacts(
                debug_path,
                segmented_cloud=segmented_cloud,
                background_cloud=background_cloud,
                response_payload={"success": False, "error": error},
                grasps=None,
                confidences=None,
            )
            return {"success": False, "error": error, "debug_dir": str(debug_path)}

        if len(grasps) == 0:
            error = "No grasps returned."
            self._save_debug_artifacts(
                debug_path,
                segmented_cloud=segmented_cloud,
                background_cloud=background_cloud,
                response_payload={"success": False, "error": error},
                grasps=np.asarray(grasps, dtype=np.float32),
                confidences=np.asarray(confidences, dtype=np.float32),
            )
            return {"success": False, "error": error, "debug_dir": str(debug_path)}

        grasps, confidences = self._kinematic_filter(grasps, confidences)
        if len(grasps) == 0:
            error = "All grasps filtered by kinematic reachability (reach/table gate)."
            self._save_debug_artifacts(
                debug_path,
                segmented_cloud=segmented_cloud,
                background_cloud=background_cloud,
                response_payload={"success": False, "error": error},
                grasps=np.empty((0, 4, 4), dtype=np.float32),
                confidences=np.empty((0,), dtype=np.float32),
            )
            return {"success": False, "error": error, "debug_dir": str(debug_path)}

        collision_free_mask = self._compute_collision_free_mask(grasps)
        filtered_rows = self._rank_grasps(grasps, confidences, collision_free_mask)
        max_returned = int(self.get_parameter("max_returned_grasps").value)
        top_rows = filtered_rows[:max_returned]

        payload = {
            "success": len(top_rows) > 0,
            "frame_id": segmented_frame,
            "num_input_points": int(len(segmented_cloud)),
            "num_grasps": int(len(grasps)),
            "num_collision_free_grasps": int(np.sum(collision_free_mask))
            if collision_free_mask is not None
            else None,
            "rank_mode": str(self.get_parameter("rank_mode").value),
            "top_grasps": top_rows,
            "debug_dir": str(debug_path),
        }
        if not payload["success"]:
            payload["error"] = "No grasps survived ranking."
        self._save_debug_artifacts(
            debug_path,
            segmented_cloud=segmented_cloud,
            background_cloud=background_cloud,
            response_payload=payload,
            grasps=self._rows_to_grasps(top_rows),
            confidences=self._rows_to_confidences(top_rows),
        )
        self.get_logger().info(
            f"Returned {len(top_rows)} filtered grasps from {len(grasps)} raw grasps "
            f"for frame {segmented_frame}"
        )
        self._maybe_publish_grasp_poses(top_rows, segmented_frame)
        return payload

    def _maybe_publish_grasp_poses(self, rows: list[dict], frame_id: str) -> None:
        """Publish ranked grasps as a PoseArray when debug viz is enabled."""
        if self._grasp_poses_pub is None:
            return
        msg = PoseArray()
        msg.header.frame_id = frame_id
        msg.header.stamp = self.get_clock().now().to_msg()
        for row in rows:
            pose = pose_from_grasp_row(row)
            if pose is not None:
                msg.poses.append(pose)
        self._grasp_poses_pub.publish(msg)

    def _compute_collision_free_mask(self, grasps: np.ndarray):
        if not bool(self.get_parameter("enable_collision_check").value):
            return None
        if self._latest_background_cloud is None:
            self.get_logger().warn("Collision filtering enabled but no background cloud received yet.")
            return None
        if resolve_gripper_info is None or filter_colliding_grasps is None:
            self.get_logger().warn("Collision filtering requested but GraspGenX collision utilities are unavailable.")
            return None

        # GraspGenX's metadata action returns "default_gripper" (the server's
        # pre-loaded gripper); "gripper_name" only appears in the infer response.
        gripper_name = (
            (self._client.server_metadata or {}).get("default_gripper")
            or _DEFAULT_GRIPPER_NAME
        )
        # Load from the gripper_descriptions assets (x_grippers/<name>) — the same
        # source the GraspGenX server uses; gives a real .collision_mesh.
        gripper_info = resolve_gripper_info(gripper_name, str(_gripper_assets_dir()))
        collision_threshold = float(self.get_parameter("collision_threshold").value)
        collision_samples = int(self.get_parameter("collision_samples").value)
        return filter_colliding_grasps(
            self._latest_background_cloud.astype(np.float32),
            np.asarray(grasps, dtype=np.float32),
            gripper_info.collision_mesh,
            collision_threshold=collision_threshold,
            num_collision_samples=collision_samples,
        )

    def _kinematic_filter(
        self, grasps: np.ndarray, confidences: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Remove grasps whose tool0 position violates UR5 reach or table clearance.

        Ported from yeina graspgen.py AC2 filter.  tool0 is approximated by
        backing off from the GraspGen TCP along the grasp +Z axis by
        _GRIPPER_TCP_Z_OFFSET.
        """
        tool_pos = np.stack(
            [g[:3, 3] - g[:3, 2] * _GRIPPER_TCP_Z_OFFSET for g in grasps]
        )
        radii = np.linalg.norm(tool_pos, axis=1)
        min_tool_z = max(
            float(os.environ.get('PIPELINE_GRASPGEN_MIN_TOOL_Z', _MIN_TOOL_Z)),
            _MIN_TOOL_Z,
        )
        keep = (radii < _MAX_REACH) & (tool_pos[:, 2] > min_tool_z)
        self.get_logger().info(
            f'GraspGen kinematic filter: kept {int(keep.sum())}/{len(grasps)} grasps '
            f'(reach<{_MAX_REACH:.2f}m, tool_z>{min_tool_z:.2f}m)'
        )
        return grasps[keep], confidences[keep]

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

    @staticmethod
    def _rows_to_grasps(rows: list[dict]) -> np.ndarray:
        if not rows:
            return np.empty((0, 4, 4), dtype=np.float32)

        grasps = []
        for row in rows:
            grasp = np.eye(4, dtype=np.float32)
            grasp[:3, :3] = np.asarray(row["rotation_matrix"], dtype=np.float32)
            grasp[:3, 3] = np.asarray(row["translation"], dtype=np.float32)
            grasps.append(grasp)
        return np.asarray(grasps, dtype=np.float32)

    @staticmethod
    def _rows_to_confidences(rows: list[dict]) -> np.ndarray:
        if not rows:
            return np.empty((0,), dtype=np.float32)
        return np.asarray([row["confidence"] for row in rows], dtype=np.float32)

    def _save_debug_artifacts(
        self,
        debug_path: Path,
        *,
        segmented_cloud: np.ndarray,
        background_cloud: np.ndarray | None,
        response_payload: dict,
        grasps: np.ndarray | None,
        confidences: np.ndarray | None,
    ) -> None:
        np.save(debug_path / "segmented_cloud.npy", np.asarray(segmented_cloud, dtype=np.float32))
        if background_cloud is not None:
            np.save(debug_path / "background_cloud.npy", np.asarray(background_cloud, dtype=np.float32))
        if grasps is not None:
            np.save(debug_path / "grasps.npy", np.asarray(grasps, dtype=np.float32))
        if confidences is not None:
            np.save(debug_path / "confidences.npy", np.asarray(confidences, dtype=np.float32))
        with open(debug_path / "result.json", "w", encoding="utf-8") as handle:
            json.dump(response_payload, handle, indent=2)

    def destroy_node(self):
        self._client.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = GraspGenService()
    # MultiThreadedExecutor so the service handler can block in _wait_for_cloud
    # while the cloud subscription (separate callback group) keeps delivering.
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
