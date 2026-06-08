import json
import os
from pathlib import Path
import threading
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from PIL import Image as PILImage
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from tf2_ros import Buffer, TransformException, TransformListener

from team_8.pipeline_utils import env_float, make_xyz_cloud
from team_8.segmentation_utils import (
    build_overlay_image,
    compute_centroid,
    compute_mask_roi,
    depth_to_masked_points,
    resize_for_api,
    stable_downsample,
)

try:  # pragma: no cover - runtime dependency
    from google import genai
    from google.genai import types
except ImportError:  # pragma: no cover - runtime dependency
    genai = None
    types = None

try:  # pragma: no cover - runtime dependency
    from ultralytics import SAM
except ImportError:  # pragma: no cover - runtime dependency
    SAM = None

try:  # pragma: no cover - runtime dependency
    from riro_srvs.srv import StringString
except ImportError:  # pragma: no cover - runtime dependency
    StringString = None


PROMPT_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "box_2d": {
                "type": "ARRAY",
                "items": {"type": "INTEGER"},
                "description": "Bounding box [ymin, xmin, ymax, xmax] scaled strictly from 0 to 1000.",
            },
            "label": {
                "type": "STRING",
                "description": "Descriptive text label of the detected item.",
            },
        },
        "required": ["box_2d", "label"],
    },
}


def transform_to_matrix(msg) -> np.ndarray:
    t = msg.transform.translation
    q = msg.transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    r = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = r
    out[:3, 3] = [t.x, t.y, t.z]
    return out


class SegmentationService(Node):
    """Prompted 2D segmentation service that publishes masked point clouds for GraspGen."""

    def __init__(self) -> None:
        if StringString is None:
            raise ImportError("riro_srvs is required for segmentation_service.")

        super().__init__("segmentation_service")

        self.declare_parameter("service_name", "/segmentation/segment_prompt")
        self.declare_parameter("rgb_topic", "/wrist_camera/wrist_camera/color/image_raw")
        self.declare_parameter("depth_topic", "/wrist_camera/wrist_camera/depth/color/image_raw")
        self.declare_parameter(
            "camera_info_topic",
            "/wrist_camera/wrist_camera/depth/color/camera_info",
        )
        self.declare_parameter("segmented_point_cloud_topic", "/graspgen/segmented_object")
        self.declare_parameter("background_point_cloud_topic", "/graspgen/background")
        self.declare_parameter("overlay_topic", "/segmentation/overlay")
        self.declare_parameter("mask_topic", "/segmentation/mask")
        self.declare_parameter("gemini_model", "gemini-2.5-flash")
        self.declare_parameter("sam2_model", "/opt/models/sam2/sam2_t.pt")
        self.declare_parameter("output_frame", "base_link")
        self.declare_parameter("max_api_image_dim", 1024)
        self.declare_parameter("min_depth_m", 0.05)
        self.declare_parameter("max_depth_m", 2.5)
        self.declare_parameter("surface_band_m", 0.02)
        self.declare_parameter("max_object_points", 4096)
        self.declare_parameter("max_background_points", 12000)
        self.declare_parameter("camera_fx", 615.0)
        self.declare_parameter("camera_fy", 615.0)
        self.declare_parameter("camera_cx", 320.0)
        self.declare_parameter("camera_cy", 240.0)
        self.declare_parameter("depth_unit_scale", 0.001)
        self.declare_parameter("debug_dir", "/artifacts/segmentation_service")
        self._bridge = CvBridge()
        # _frame_cv guards the latest RGB/depth stamps and is notified whenever a
        # new frame arrives, so a service handler blocked in _wait_for_fresh_frames
        # wakes as soon as a post-home frame lands.
        self._frame_cv = threading.Condition()
        self._fresh_frame_timeout_sec = env_float(
            "PIPELINE_SEG_FRESH_FRAME_TIMEOUT_SEC", 5.0)
        self._latest_rgb = None
        self._latest_rgb_stamp_ns = 0
        self._latest_depth = None
        self._latest_depth_stamp = None
        self._latest_depth_stamp_ns = 0
        self._latest_frame_id = ""
        self._latest_camera_info = None
        self._logged_first_rgb = False
        self._logged_first_depth = False
        self._logged_first_camera_info = False
        self._debug_dir = Path(str(self.get_parameter("debug_dir").value))
        self._debug_dir.mkdir(parents=True, exist_ok=True)

        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not set.")
        if genai is None or types is None:
            raise ImportError("google-genai is required for segmentation_service.")
        if SAM is None:
            raise ImportError("ultralytics is required for segmentation_service.")

        self._gemini = genai.Client(api_key=gemini_api_key)
        self._gemini_model = str(self.get_parameter("gemini_model").value)
        self._sam2_model_name = str(self.get_parameter("sam2_model").value)
        self._sam2 = SAM(self._sam2_model_name)
        self._output_frame = str(self.get_parameter("output_frame").value)
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT

        # Cloud topics are request/response-style bulk data (not a high-rate
        # stream); RELIABLE prevents the one cloud GraspGen needs from being
        # silently dropped. Must match GraspGen's subscription QoS.
        cloud_qos = QoSProfile(depth=10)
        cloud_qos.reliability = ReliabilityPolicy.RELIABLE

        # Camera callbacks share a reentrant group so they can keep updating the
        # latest frames while a service handler blocks in _wait_for_fresh_frames
        # (the node is spun with a MultiThreadedExecutor in main()).
        camera_group = ReentrantCallbackGroup()
        self.create_subscription(
            Image, str(self.get_parameter("rgb_topic").value), self._rgb_callback, qos,
            callback_group=camera_group,
        )
        self.create_subscription(
            Image,
            str(self.get_parameter("depth_topic").value),
            self._depth_callback,
            qos,
            callback_group=camera_group,
        )
        self.create_subscription(
            CameraInfo,
            str(self.get_parameter("camera_info_topic").value),
            self._camera_info_callback,
            qos,
            callback_group=camera_group,
        )

        self._segmented_pub = self.create_publisher(
            PointCloud2, str(self.get_parameter("segmented_point_cloud_topic").value), cloud_qos
        )
        self._background_pub = self.create_publisher(
            PointCloud2, str(self.get_parameter("background_point_cloud_topic").value), cloud_qos
        )
        self._overlay_pub = self.create_publisher(
            Image, str(self.get_parameter("overlay_topic").value), qos
        )
        self._mask_pub = self.create_publisher(Image, str(self.get_parameter("mask_topic").value), qos)

        self.create_service(
            StringString, str(self.get_parameter("service_name").value), self._handle_request
        )

        self.get_logger().info(
            "segmentation_service ready "
            f"rgb={self.get_parameter('rgb_topic').value} "
            f"depth={self.get_parameter('depth_topic').value} "
            f"service={self.get_parameter('service_name').value}"
        )

    def _rgb_callback(self, msg: Image) -> None:
        rgb = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        stamp_ns = self._stamp_to_ns(msg.header.stamp)
        with self._frame_cv:
            self._latest_rgb = rgb
            self._latest_rgb_stamp_ns = stamp_ns
            self._frame_cv.notify_all()
        if not self._logged_first_rgb:
            self.get_logger().info(
                f"Received first RGB frame {msg.width}x{msg.height} on "
                f"{self.get_parameter('rgb_topic').value}"
            )
            self._logged_first_rgb = True

    def _depth_callback(self, msg: Image) -> None:
        if msg.encoding == "16UC1":
            depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1").astype(np.float32)
            depth *= float(self.get_parameter("depth_unit_scale").value)
        else:
            depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1").astype(np.float32)

        stamp = msg.header.stamp
        stamp_ns = self._stamp_to_ns(stamp)
        frame_id = msg.header.frame_id
        with self._frame_cv:
            self._latest_depth = depth
            self._latest_depth_stamp = stamp
            self._latest_depth_stamp_ns = stamp_ns
            self._latest_frame_id = frame_id
            self._frame_cv.notify_all()
        if not self._logged_first_depth:
            self.get_logger().info(
                f"Received first depth frame {msg.width}x{msg.height} "
                f"frame={frame_id} on {self.get_parameter('depth_topic').value}"
            )
            self._logged_first_depth = True

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        self._latest_camera_info = msg
        if not self._logged_first_camera_info:
            self.get_logger().info(
                f"Received first CameraInfo fx={msg.k[0]:.3f} fy={msg.k[4]:.3f} "
                f"cx={msg.k[2]:.3f} cy={msg.k[5]:.3f} frame={msg.header.frame_id}"
            )
            self._logged_first_camera_info = True

    def _camera_intrinsics(self) -> tuple[float, float, float, float]:
        # Read outside _frame_cv on purpose: intrinsics are quasi-static for a
        # fixed camera, so a one-message-stale CameraInfo is harmless (and the
        # reference grab is atomic under CPython). Not part of the frame snapshot.
        info = self._latest_camera_info
        if info is not None:
            return (
                float(info.k[0]),
                float(info.k[4]),
                float(info.k[2]),
                float(info.k[5]),
            )
        self.get_logger().warn(
            "No CameraInfo received yet; using configured camera intrinsics.")
        return (
            float(self.get_parameter("camera_fx").value),
            float(self.get_parameter("camera_fy").value),
            float(self.get_parameter("camera_cx").value),
            float(self.get_parameter("camera_cy").value),
        )

    def _handle_request(self, request: StringString.Request, response: StringString.Response):
        prompt, min_stamp_ns = self._parse_request(request.data)
        if not prompt:
            prompt = "Pick the requested object."

        # When the orchestrator passes a freshness gate (the home-arrival time),
        # wait for a wrist frame captured after the arm settled at home so we
        # never segment a stale mid-transit frame.
        if min_stamp_ns > 0 and not self._wait_for_fresh_frames(min_stamp_ns):
            response.data = json.dumps({
                "success": False,
                "error": (
                    f"Timed out waiting for wrist frames newer than {min_stamp_ns} ns "
                    f"(waited {self._fresh_frame_timeout_sec:.1f}s)."
                ),
            })
            return response

        # Snapshot the latest frame and its stamps atomically: the camera
        # callbacks run on another thread (MultiThreadedExecutor + reentrant
        # group), so reading these fields separately could mix a newer frame's
        # stamp with an older frame's pixels.
        with self._frame_cv:
            rgb_bgr = None if self._latest_rgb is None else self._latest_rgb.copy()
            depth_image = None if self._latest_depth is None else self._latest_depth.copy()
            source_frame = self._latest_frame_id
            cloud_stamp = self._latest_depth_stamp
            cloud_stamp_ns = int(self._latest_depth_stamp_ns)

        if rgb_bgr is None or depth_image is None:
            missing = []
            if rgb_bgr is None:
                missing.append("rgb")
            if depth_image is None:
                missing.append("depth")
            response.data = json.dumps(
                {
                    "success": False,
                    "error": "Waiting for RGB image and depth image.",
                    "missing": missing,
                }
            )
            return response

        start_time = time.perf_counter()
        debug_id = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time_ns() % 1_000_000_000):09d}"
        debug_path = self._debug_dir / debug_id
        debug_path.mkdir(parents=True, exist_ok=True)
        try:
            fx, fy, cx, cy = self._camera_intrinsics()

            api_image_bgr, scale_x, scale_y = resize_for_api(
                rgb_bgr, int(self.get_parameter("max_api_image_dim").value)
            )
            api_width = int(api_image_bgr.shape[1])
            api_height = int(api_image_bgr.shape[0])

            prompt_result = self._localize_prompt(api_image_bgr, prompt)
            prompt_bbox_api = self._sanitize_detection_box(
                prompt_result["box_2d"], api_width, api_height
            )
            prompt_bbox = self._scale_bbox(prompt_bbox_api, scale_x, scale_y, rgb_bgr.shape[1], rgb_bgr.shape[0])
            mask = self._segment_with_sam2(rgb_bgr, prompt_bbox)
            mask = self._clip_mask_to_bbox(mask, prompt_bbox)
            if not mask.any():
                raise RuntimeError("SAM2 returned an empty mask.")

            object_points, background_points = depth_to_masked_points(
                depth_image,
                mask,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                min_depth_m=float(self.get_parameter("min_depth_m").value),
                max_depth_m=float(self.get_parameter("max_depth_m").value),
            )
            if len(object_points) == 0:
                raise RuntimeError("Masked point cloud is empty after depth filtering.")

            output_frame = self._output_frame
            object_points = self._transform_points_to_output_frame(
                object_points,
                source_frame=source_frame,
                target_frame=output_frame,
                stamp=cloud_stamp,
            )
            background_points = self._transform_points_to_output_frame(
                background_points,
                source_frame=source_frame,
                target_frame=output_frame,
                stamp=cloud_stamp,
            )

            object_points = stable_downsample(
                object_points, int(self.get_parameter("max_object_points").value)
            )
            background_points = stable_downsample(
                background_points, int(self.get_parameter("max_background_points").value)
            )
            centroid = compute_centroid(
                object_points, float(self.get_parameter("surface_band_m").value)
            )
            if centroid is None:
                raise RuntimeError("Unable to compute centroid from segmented object cloud.")

            roi = compute_mask_roi(mask)
            assert roi is not None
            x_min, y_min, x_max, y_max = roi

            self._segmented_pub.publish(
                make_xyz_cloud(object_points, output_frame, stamp=cloud_stamp))
            if len(background_points) > 0:
                self._background_pub.publish(
                    make_xyz_cloud(background_points, output_frame, stamp=cloud_stamp))

            overlay = build_overlay_image(
                rgb_bgr,
                mask,
                [],
                prompt_result["label"],
            )
            self._overlay_pub.publish(self._bridge.cv2_to_imgmsg(overlay, encoding="bgr8"))
            self._mask_pub.publish(
                self._bridge.cv2_to_imgmsg((mask.astype(np.uint8) * 255), encoding="mono8")
            )

            elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            response_payload = {
                "success": True,
                "prompt": prompt,
                "label": prompt_result["label"],
                "frame_id": output_frame,
                "source_frame_id": source_frame,
                "cloud_stamp_ns": cloud_stamp_ns,
                "centroid": [round(float(v), 5) for v in centroid],
                "roi_xyxy": [x_min, y_min, x_max, y_max],
                "mask_pixel_count": int(mask.sum()),
                "object_point_count": int(len(object_points)),
                "background_point_count": int(len(background_points)),
                "mask_area_ratio": round(float(mask.mean()), 6),
                "processing_time_ms": elapsed_ms,
                "camera_intrinsics": {
                    "fx": round(float(fx), 5),
                    "fy": round(float(fy), 5),
                    "cx": round(float(cx), 5),
                    "cy": round(float(cy), 5),
                },
                "gemini_bbox_xyxy": list(prompt_bbox),
                "sam2_model": self._sam2_model_name,
                "debug_dir": str(debug_path),
            }
            response.data = json.dumps(response_payload)
            self._save_debug_artifacts(
                debug_path,
                prompt=prompt,
                rgb_bgr=rgb_bgr,
                depth_image=depth_image,
                api_image_bgr=api_image_bgr,
                mask=mask,
                overlay=overlay,
                prompt_result=prompt_result,
                prompt_bbox=prompt_bbox,
                object_points=object_points,
                background_points=background_points,
                response_payload=response_payload,
            )
            self.get_logger().info(
                f"Segmented '{prompt_result['label']}' "
                f"frame={output_frame} points={len(object_points)} centroid={centroid.tolist()} "
                f"time_ms={elapsed_ms}"
            )
            return response
        except Exception as exc:
            self.get_logger().error(f"Segmentation request failed: {exc}")
            failure_payload = {
                "success": False,
                "error": str(exc),
                "prompt": prompt,
                "debug_dir": str(debug_path),
            }
            self._save_failure_debug_artifacts(
                debug_path,
                prompt=prompt,
                rgb_bgr=rgb_bgr,
                depth_image=depth_image,
                failure_payload=failure_payload,
            )
            response.data = json.dumps(failure_payload)
            return response

    def _localize_prompt(self, image_bgr: np.ndarray, prompt: str) -> dict:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(rgb)
        width = int(image_bgr.shape[1])
        height = int(image_bgr.shape[0])

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            response_schema=PROMPT_SCHEMA,
            temperature=0.1,
        )

        full_prompt = (
            f"{prompt}\n"
            "Return a JSON list. For each object, return the label and the 'box_2d' "
            "as an array of exactly 4 integers: [ymin, xmin, ymax, xmax]."
        )
        response = self._gemini.models.generate_content(
            model=self._gemini_model,
            contents=[full_prompt, pil_image],
            config=config,
        )
        payload = json.loads(response.text)
        if not isinstance(payload, list) or not payload:
            raise RuntimeError("Gemini returned no detections.")

        detection = payload[0]
        label = str(detection.get("label", "")).strip() or "target"
        box_2d = detection.get("box_2d", [])
        return {"label": label, "box_2d": box_2d, "image_size": [width, height]}

    def _scale_bbox(
        self,
        bbox: tuple[int, int, int, int],
        scale_x: float,
        scale_y: float,
        width: int,
        height: int,
    ) -> tuple[int, int, int, int]:
        x_min, y_min, x_max, y_max = bbox
        scaled = (
            int(round(x_min * scale_x)),
            int(round(y_min * scale_y)),
            int(round(x_max * scale_x)),
            int(round(y_max * scale_y)),
        )
        return self._sanitize_pixel_box(scaled, width, height)

    def _sanitize_detection_box(
        self, raw_box: list[int], width: int, height: int
    ) -> tuple[int, int, int, int]:
        if not isinstance(raw_box, list) or len(raw_box) != 4:
            raise RuntimeError(f"Invalid Gemini box format: {raw_box}")

        try:
            ymin_norm, xmin_norm, ymax_norm, xmax_norm = [int(value) for value in raw_box]
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Non-integer Gemini box format: {raw_box}") from exc

        ymin_norm = int(np.clip(ymin_norm, 0, 1000))
        xmin_norm = int(np.clip(xmin_norm, 0, 1000))
        ymax_norm = int(np.clip(ymax_norm, 0, 1000))
        xmax_norm = int(np.clip(xmax_norm, 0, 1000))

        x_min = int(round(xmin_norm / 1000.0 * width))
        y_min = int(round(ymin_norm / 1000.0 * height))
        x_max = int(round(xmax_norm / 1000.0 * width))
        y_max = int(round(ymax_norm / 1000.0 * height))

        x_min = int(np.clip(x_min, 0, width - 1))
        y_min = int(np.clip(y_min, 0, height - 1))
        x_max = int(np.clip(x_max, 0, width - 1))
        y_max = int(np.clip(y_max, 0, height - 1))
        if x_max <= x_min or y_max <= y_min:
            raise RuntimeError(
                f"Degenerate Gemini box after scaling: raw={raw_box} scaled={[x_min, y_min, x_max, y_max]}"
            )
        return x_min, y_min, x_max, y_max

    def _sanitize_pixel_box(
        self,
        bbox: tuple[int, int, int, int],
        width: int,
        height: int,
    ) -> tuple[int, int, int, int]:
        x_min, y_min, x_max, y_max = bbox
        x_min = int(np.clip(x_min, 0, width - 1))
        y_min = int(np.clip(y_min, 0, height - 1))
        x_max = int(np.clip(x_max, 0, width - 1))
        y_max = int(np.clip(y_max, 0, height - 1))
        if x_max <= x_min or y_max <= y_min:
            raise RuntimeError(f"Degenerate pixel bbox: {[x_min, y_min, x_max, y_max]}")
        return x_min, y_min, x_max, y_max

    def _segment_with_sam2(
        self, image_bgr: np.ndarray, bbox: tuple[int, int, int, int]
    ) -> np.ndarray:
        x_min, y_min, x_max, y_max = bbox
        results = self._sam2(image_bgr, bboxes=[x_min, y_min, x_max, y_max], verbose=False)
        if not results:
            raise RuntimeError("SAM2 returned no results.")

        masks = results[0].masks
        if masks is None or masks.data is None or len(masks.data) == 0:
            raise RuntimeError("SAM2 returned no masks.")

        mask_data = masks.data.detach().cpu().numpy()
        best_index = int(np.argmax(mask_data.reshape(mask_data.shape[0], -1).sum(axis=1)))
        mask = mask_data[best_index] > 0
        if mask.shape != image_bgr.shape[:2]:
            raise RuntimeError(
                f"SAM2 mask shape {mask.shape} does not match image shape {image_bgr.shape[:2]}"
            )
        return mask.astype(bool)

    @staticmethod
    def _clip_mask_to_bbox(
        mask: np.ndarray, bbox: tuple[int, int, int, int]
    ) -> np.ndarray:
        x_min, y_min, x_max, y_max = bbox
        clipped = np.zeros_like(mask, dtype=bool)
        clipped[y_min : y_max + 1, x_min : x_max + 1] = mask[y_min : y_max + 1, x_min : x_max + 1]
        return clipped

    def _transform_points_to_output_frame(
        self,
        points: np.ndarray,
        *,
        source_frame: str,
        target_frame: str,
        stamp,
    ) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32)
        if len(points) == 0 or source_frame == target_frame:
            return points

        try:
            lookup_time = Time.from_msg(stamp) if stamp is not None else Time()
            tf = self._tf_buffer.lookup_transform(target_frame, source_frame, lookup_time)
        except TransformException:
            tf = self._tf_buffer.lookup_transform(target_frame, source_frame, Time())

        matrix = transform_to_matrix(tf)
        rotated = points @ matrix[:3, :3].T
        translated = rotated + matrix[:3, 3]
        return translated.astype(np.float32)

    def _save_debug_artifacts(
        self,
        debug_path: Path,
        *,
        prompt: str,
        rgb_bgr: np.ndarray,
        depth_image: np.ndarray,
        api_image_bgr: np.ndarray,
        mask: np.ndarray,
        overlay: np.ndarray,
        prompt_result: dict,
        prompt_bbox: tuple[int, int, int, int],
        object_points: np.ndarray,
        background_points: np.ndarray,
        response_payload: dict,
    ) -> None:
        cv2.imwrite(str(debug_path / "rgb.png"), rgb_bgr)
        cv2.imwrite(str(debug_path / "api_rgb.png"), api_image_bgr)
        cv2.imwrite(str(debug_path / "mask.png"), (mask.astype(np.uint8) * 255))
        cv2.imwrite(str(debug_path / "overlay.png"), overlay)
        np.save(debug_path / "depth_m.npy", depth_image.astype(np.float32))
        np.save(debug_path / "object_points.npy", object_points.astype(np.float32))
        np.save(debug_path / "background_points.npy", background_points.astype(np.float32))
        with open(debug_path / "result.json", "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "prompt": prompt,
                    "prompt_result": prompt_result,
                    "prompt_bbox_xyxy": list(prompt_bbox),
                    "response": response_payload,
                },
                handle,
                indent=2,
            )

    def _save_failure_debug_artifacts(
        self,
        debug_path: Path,
        *,
        prompt: str,
        rgb_bgr: np.ndarray | None,
        depth_image: np.ndarray | None,
        failure_payload: dict,
    ) -> None:
        if rgb_bgr is not None:
            cv2.imwrite(str(debug_path / "rgb.png"), rgb_bgr)
        if depth_image is not None:
            np.save(debug_path / "depth_m.npy", np.asarray(depth_image, dtype=np.float32))
        with open(debug_path / "failure.json", "w", encoding="utf-8") as handle:
            json.dump({"prompt": prompt, "response": failure_payload}, handle, indent=2)

    def _wait_for_fresh_frames(self, min_stamp_ns: int) -> bool:
        """Block until both the RGB and depth caches hold a frame stamped after
        *min_stamp_ns*, or until PIPELINE_SEG_FRESH_FRAME_TIMEOUT_SEC elapses.

        Used to guarantee segmentation runs on a frame captured *after* the arm
        settled at home, not a stale mid-transit frame.
        """
        deadline = time.monotonic() + self._fresh_frame_timeout_sec
        with self._frame_cv:
            # Strict ">" (loop while "<="): a frame stamped exactly at the gate
            # could have been captured mid-transit, so require one strictly newer.
            # (GraspGen's _wait_for_cloud uses ">=" because there the stamp is the
            # request token; the freshness semantics here are deliberately different.)
            while (self._latest_rgb_stamp_ns <= min_stamp_ns
                   or self._latest_depth_stamp_ns <= min_stamp_ns):
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                self._frame_cv.wait(timeout=remaining)
            return True

    @staticmethod
    def _parse_request(data: "str | None") -> tuple[str, int]:
        """Parse a segmentation request into (prompt, min_stamp_ns).

        The request `data` is normally a JSON object
        ``{"prompt": "...", "min_stamp_ns": <int>}``. For back-compat, a value
        that does not parse as a JSON *dict* (e.g. a bare prompt string sent by
        standalone callers) is treated as the prompt with no freshness gate.
        """
        text = (data or "").strip()
        if not text:
            return "", 0
        try:
            payload = json.loads(text)
        except (TypeError, ValueError):
            return text, 0
        if not isinstance(payload, dict):
            return text, 0
        prompt = str(payload.get("prompt", "")).strip()
        try:
            min_stamp_ns = int(payload.get("min_stamp_ns", 0) or 0)
        except (TypeError, ValueError):
            min_stamp_ns = 0
        return prompt, min_stamp_ns

    @staticmethod
    def _stamp_to_ns(stamp) -> int:
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SegmentationService()
    # MultiThreadedExecutor so the service handler can block in
    # _wait_for_fresh_frames while the camera callbacks (reentrant group) keep
    # delivering frames on another thread.
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
