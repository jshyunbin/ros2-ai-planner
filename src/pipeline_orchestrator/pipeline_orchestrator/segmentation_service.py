import hashlib
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from PIL import Image as PILImage
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header

from pipeline_orchestrator.segmentation_utils import (
    PromptPoint,
    build_overlay_image,
    compute_centroid,
    compute_mask_roi,
    depth_to_masked_points,
    encode_image_to_base64,
    parse_sam3_polygons,
    rasterize_polygons,
    resize_for_api,
    scale_polygons,
    stable_downsample,
)

try:  # pragma: no cover - runtime dependency
    from google import genai
except ImportError:  # pragma: no cover - runtime dependency
    genai = None

try:  # pragma: no cover - runtime dependency
    from riro_srvs.srv import StringString
except ImportError:  # pragma: no cover - runtime dependency
    StringString = None


PROMPT_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {
            "type": "string",
            "description": "Short target label such as banana or meat can.",
        },
        "points": {
            "type": "array",
            "description": "Up to four image points for SAM3 prompting.",
            "minItems": 1,
            "maxItems": 4,
            "items": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer", "minimum": 0},
                    "y": {"type": "integer", "minimum": 0},
                    "positive": {"type": "boolean"},
                },
                "required": ["x", "y", "positive"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["label", "points"],
    "additionalProperties": False,
}


def make_xyz_cloud(points: np.ndarray, frame_id: str) -> PointCloud2:
    xyz = np.asarray(points[:, :3], dtype=np.float32)
    msg = PointCloud2()
    msg.header = Header(frame_id=frame_id)
    msg.height = 1
    msg.width = len(xyz)
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = 12 * len(xyz)
    msg.data = xyz.tobytes()
    msg.is_dense = True
    return msg


class SegmentationService(Node):
    """Prompted 2D segmentation service that publishes masked point clouds for GraspGen."""

    def __init__(self) -> None:
        if StringString is None:
            raise ImportError("riro_srvs is required for segmentation_service.")

        super().__init__("segmentation_service")

        self.declare_parameter("service_name", "/segmentation/segment_prompt")
        self.declare_parameter("rgb_topic", "/wrist_camera/wrist_camera/color/image_raw")
        self.declare_parameter("depth_topic", "/wrist_camera/wrist_camera/depth/color/image_raw")
        self.declare_parameter("segmented_point_cloud_topic", "/graspgen/segmented_object")
        self.declare_parameter("background_point_cloud_topic", "/graspgen/background")
        self.declare_parameter("overlay_topic", "/segmentation/overlay")
        self.declare_parameter("mask_topic", "/segmentation/mask")
        self.declare_parameter("gemini_model", "gemini-2.5-flash")
        self.declare_parameter("sam3_endpoint", "https://sam3.ai/api/v1/pvs")
        self.declare_parameter("api_timeout_sec", 120.0)
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
        self._latest_rgb = None
        self._latest_rgb_stamp_ns = 0
        self._latest_depth = None
        self._latest_depth_stamp_ns = 0
        self._latest_frame_id = ""
        self._logged_first_rgb = False
        self._logged_first_depth = False
        self._debug_dir = Path(str(self.get_parameter("debug_dir").value))
        self._debug_dir.mkdir(parents=True, exist_ok=True)

        gemini_api_key = os.getenv("GEMINI_API_KEY")
        sam3_api_key = os.getenv("SAM3_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not set.")
        if not sam3_api_key:
            raise ValueError("SAM3_API_KEY is not set.")
        if genai is None:
            raise ImportError("google-genai is required for segmentation_service.")

        self._gemini = genai.Client(api_key=gemini_api_key)
        self._gemini_model = str(self.get_parameter("gemini_model").value)
        self._sam3_api_key = sam3_api_key
        self._sam3_endpoint = str(self.get_parameter("sam3_endpoint").value)

        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT

        self.create_subscription(
            Image, str(self.get_parameter("rgb_topic").value), self._rgb_callback, qos
        )
        self.create_subscription(
            Image,
            str(self.get_parameter("depth_topic").value),
            self._depth_callback,
            qos,
        )

        self._segmented_pub = self.create_publisher(
            PointCloud2, str(self.get_parameter("segmented_point_cloud_topic").value), qos
        )
        self._background_pub = self.create_publisher(
            PointCloud2, str(self.get_parameter("background_point_cloud_topic").value), qos
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
        self._latest_rgb = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self._latest_rgb_stamp_ns = self._stamp_to_ns(msg.header.stamp)
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

        self._latest_depth = depth
        self._latest_depth_stamp_ns = self._stamp_to_ns(msg.header.stamp)
        self._latest_frame_id = msg.header.frame_id
        if not self._logged_first_depth:
            self.get_logger().info(
                f"Received first depth frame {msg.width}x{msg.height} "
                f"frame={msg.header.frame_id} on {self.get_parameter('depth_topic').value}"
            )
            self._logged_first_depth = True

    def _handle_request(self, request: StringString.Request, response: StringString.Response):
        prompt = request.data.strip()
        if not prompt:
            prompt = "Pick the requested object."

        if self._latest_rgb is None or self._latest_depth is None:
            missing = []
            if self._latest_rgb is None:
                missing.append("rgb")
            if self._latest_depth is None:
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
            rgb_bgr = self._latest_rgb.copy()
            depth_image = self._latest_depth.copy()

            api_image_bgr, scale_x, scale_y = resize_for_api(
                rgb_bgr, int(self.get_parameter("max_api_image_dim").value)
            )
            api_width = int(api_image_bgr.shape[1])
            api_height = int(api_image_bgr.shape[0])

            prompt_result = self._localize_prompt(api_image_bgr, prompt)
            prompt_points = self._sanitize_prompt_points(prompt_result["points"], api_width, api_height)
            polygons_api = self._segment_with_sam3(api_image_bgr, prompt_points)
            if not polygons_api:
                raise RuntimeError("SAM3 returned no polygons.")

            polygons_full = scale_polygons(polygons_api, scale_x, scale_y)
            full_height, full_width = rgb_bgr.shape[:2]
            mask = rasterize_polygons(polygons_full, full_width, full_height)
            if not mask.any():
                raise RuntimeError("Rasterized mask is empty.")

            object_points, background_points = depth_to_masked_points(
                depth_image,
                mask,
                fx=float(self.get_parameter("camera_fx").value),
                fy=float(self.get_parameter("camera_fy").value),
                cx=float(self.get_parameter("camera_cx").value),
                cy=float(self.get_parameter("camera_cy").value),
                min_depth_m=float(self.get_parameter("min_depth_m").value),
                max_depth_m=float(self.get_parameter("max_depth_m").value),
            )
            if len(object_points) == 0:
                raise RuntimeError("Masked point cloud is empty after depth filtering.")

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

            self._segmented_pub.publish(make_xyz_cloud(object_points, self._latest_frame_id))
            if len(background_points) > 0:
                self._background_pub.publish(make_xyz_cloud(background_points, self._latest_frame_id))

            prompt_points_full = [
                PromptPoint(
                    x=int(round(point.x * scale_x)),
                    y=int(round(point.y * scale_y)),
                    positive=point.positive,
                )
                for point in prompt_points
            ]
            overlay = build_overlay_image(
                rgb_bgr,
                mask,
                prompt_points_full,
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
                "frame_id": self._latest_frame_id,
                "centroid": [round(float(v), 5) for v in centroid],
                "roi_xyxy": [x_min, y_min, x_max, y_max],
                "mask_pixel_count": int(mask.sum()),
                "object_point_count": int(len(object_points)),
                "background_point_count": int(len(background_points)),
                "mask_area_ratio": round(float(mask.mean()), 6),
                "processing_time_ms": elapsed_ms,
                "prompt_points": [
                    {"x": int(p.x), "y": int(p.y), "positive": bool(p.positive)}
                    for p in prompt_points_full
                ],
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
                prompt_points_full=prompt_points_full,
                polygons_full=polygons_full,
                object_points=object_points,
                background_points=background_points,
                response_payload=response_payload,
            )
            self.get_logger().info(
                f"Segmented '{prompt_result['label']}' "
                f"points={len(object_points)} centroid={centroid.tolist()} "
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
                rgb_bgr=self._latest_rgb,
                depth_image=self._latest_depth,
                failure_payload=failure_payload,
            )
            response.data = json.dumps(failure_payload)
            return response

    def _localize_prompt(self, image_bgr: np.ndarray, prompt: str) -> dict:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(rgb)
        width = int(image_bgr.shape[1])
        height = int(image_bgr.shape[0])

        prompt_text = (
            "You are selecting point prompts for point-based image segmentation.\n"
            f"Image width={width}, height={height}.\n"
            "Find only the single visible object that best matches the user request.\n"
            "Return 1-4 image points in pixel coordinates for the uploaded image.\n"
            "Use 2-3 positive points on the target surface when possible.\n"
            "Add at most one negative point only if it helps exclude a touching distractor.\n"
            "Do not use bounding boxes.\n"
            f"User request: {prompt}"
        )

        config = {
            "response_mime_type": "application/json",
            "response_json_schema": PROMPT_SCHEMA,
        }
        response = self._gemini.models.generate_content(
            model=self._gemini_model,
            contents=[pil_image, prompt_text],
            config=config,
        )
        payload = json.loads(response.text)
        label = str(payload.get("label", "")).strip() or "target"
        points = payload.get("points", [])
        if not isinstance(points, list) or not points:
            raise RuntimeError("Gemini returned no prompt points.")
        return {"label": label, "points": points}

    def _sanitize_prompt_points(self, raw_points: list[dict], width: int, height: int) -> list[PromptPoint]:
        sanitized = []
        for point in raw_points:
            try:
                x = int(point["x"])
                y = int(point["y"])
            except (KeyError, TypeError, ValueError):
                continue
            x = int(np.clip(x, 0, width - 1))
            y = int(np.clip(y, 0, height - 1))
            sanitized.append(PromptPoint(x=x, y=y, positive=bool(point.get("positive", True))))

        if not sanitized:
            raise RuntimeError("No valid Gemini prompt points remained after validation.")
        if not any(point.positive for point in sanitized):
            first = sanitized[0]
            sanitized[0] = PromptPoint(x=first.x, y=first.y, positive=True)
        return sanitized

    def _segment_with_sam3(self, image_bgr: np.ndarray, prompt_points: list[PromptPoint]) -> list[np.ndarray]:
        image_base64 = encode_image_to_base64(image_bgr)
        image_id = hashlib.sha1(image_base64[-512:].encode("ascii")).hexdigest()[:16]
        payload = {
            "image": image_base64,
            "imageId": image_id,
            "multimaskOutput": False,
            "points": [
                {"x": point.x, "y": point.y, "positive": point.positive}
                for point in prompt_points
            ],
        }
        timeout = float(self.get_parameter("api_timeout_sec").value)
        raw = self._http_post_json(
            self._sam3_endpoint,
            payload,
            headers={"Authorization": f"Bearer {self._sam3_api_key}"},
            timeout=timeout,
        )
        polygons = parse_sam3_polygons(raw)
        if not polygons:
            raise RuntimeError(f"SAM3 returned no valid polygons: {json.dumps(raw)[:400]}")
        return polygons

    def _http_post_json(
        self,
        url: str,
        payload: dict,
        *,
        headers: dict[str, str] | None = None,
        timeout: float,
    ) -> dict:
        request_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "ros2-ai-planner/0.1 (+https://sam3.ai/docs/api-reference)",
        }
        if headers:
            request_headers.update(headers)

        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=request_headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} from {url}: {body[:400]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Failed to reach {url}: {exc.reason}") from exc

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
        prompt_points_full: list[PromptPoint],
        polygons_full: list[np.ndarray],
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
        with open(debug_path / "polygons.json", "w", encoding="utf-8") as handle:
            json.dump([polygon.tolist() for polygon in polygons_full], handle)
        with open(debug_path / "result.json", "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "prompt": prompt,
                    "prompt_result": prompt_result,
                    "prompt_points_full": [
                        {"x": p.x, "y": p.y, "positive": p.positive} for p in prompt_points_full
                    ],
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

    @staticmethod
    def _stamp_to_ns(stamp) -> int:
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SegmentationService()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
