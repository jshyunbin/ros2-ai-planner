import base64
from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class PromptPoint:
    x: int
    y: int
    positive: bool = True


def prompt_points_from_box(
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
    width: int,
    height: int,
) -> list[PromptPoint]:
    if width <= 0 or height <= 0:
        raise ValueError("Image width and height must be positive.")

    x_min = int(np.clip(x_min, 0, width - 1))
    y_min = int(np.clip(y_min, 0, height - 1))
    x_max = int(np.clip(x_max, 0, width - 1))
    y_max = int(np.clip(y_max, 0, height - 1))
    if x_max < x_min:
        x_min, x_max = x_max, x_min
    if y_max < y_min:
        y_min, y_max = y_max, y_min

    raw_points = [
        (int(round((x_min + x_max) / 2.0)), int(round((y_min + y_max) / 2.0))),
        (int(round(x_min + (x_max - x_min) * 0.35)), int(round(y_min + (y_max - y_min) * 0.35))),
        (int(round(x_min + (x_max - x_min) * 0.65)), int(round(y_min + (y_max - y_min) * 0.65))),
    ]

    prompt_points = []
    seen = set()
    for x, y in raw_points:
        point = (int(np.clip(x, 0, width - 1)), int(np.clip(y, 0, height - 1)))
        if point in seen:
            continue
        seen.add(point)
        prompt_points.append(PromptPoint(x=point[0], y=point[1], positive=True))
    return prompt_points


def resize_for_api(image_bgr: np.ndarray, max_dimension: int) -> tuple[np.ndarray, float, float]:
    height, width = image_bgr.shape[:2]
    largest_dim = max(height, width)
    if largest_dim <= max_dimension:
        return image_bgr.copy(), 1.0, 1.0

    scale = float(max_dimension) / float(largest_dim)
    resized = cv2.resize(
        image_bgr,
        (int(round(width * scale)), int(round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    scale_x = float(width) / float(resized.shape[1])
    scale_y = float(height) / float(resized.shape[0])
    return resized, scale_x, scale_y


def encode_image_to_base64(image_bgr: np.ndarray, max_bytes: int = 2_000_000) -> str:
    for quality in (95, 90, 85, 80, 75, 70, 65, 60):
        ok, encoded = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            continue
        if encoded.nbytes <= max_bytes:
            payload = base64.b64encode(encoded.tobytes()).decode("ascii")
            return payload
    raise ValueError("Unable to encode image under SAM3 size limit.")


def scale_polygons(polygons: Iterable[np.ndarray], scale_x: float, scale_y: float) -> list[np.ndarray]:
    scaled = []
    for polygon in polygons:
        polygon = np.asarray(polygon, dtype=np.float32)
        if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
            continue
        polygon = polygon.copy()
        polygon[:, 0] *= scale_x
        polygon[:, 1] *= scale_y
        scaled.append(polygon)
    return scaled


def rasterize_polygons(polygons: Iterable[np.ndarray], width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    cv_polygons = []
    for polygon in polygons:
        polygon = np.asarray(polygon, dtype=np.float32)
        if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
            continue
        polygon = np.round(polygon).astype(np.int32)
        polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
        cv_polygons.append(polygon.reshape(-1, 1, 2))

    if cv_polygons:
        cv2.fillPoly(mask, cv_polygons, 255)
    return mask.astype(bool)


def compute_mask_roi(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())
    return x_min, y_min, x_max, y_max


def select_masked_points(
    xyz_image: np.ndarray,
    mask: np.ndarray,
    *,
    min_depth_m: float,
    max_depth_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(xyz_image).all(axis=2)
    depth = xyz_image[:, :, 2]
    valid &= depth >= min_depth_m
    valid &= depth <= max_depth_m

    object_points = xyz_image[np.logical_and(mask, valid)]
    background_points = xyz_image[np.logical_and(~mask, valid)]
    return object_points.astype(np.float32), background_points.astype(np.float32)


def depth_to_masked_points(
    depth_image: np.ndarray,
    mask: np.ndarray,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    min_depth_m: float,
    max_depth_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(depth_image, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"depth_image must be HxW, got shape {depth.shape}")

    if depth.shape != mask.shape:
        raise ValueError(
            f"mask shape {mask.shape} does not match depth image shape {depth.shape}"
        )

    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("Camera intrinsics fx and fy must be positive.")

    ys, xs = np.indices(depth.shape, dtype=np.float32)
    valid = np.isfinite(depth)
    valid &= depth >= min_depth_m
    valid &= depth <= max_depth_m

    z = depth[valid]
    x = (xs[valid] - float(cx)) * z / float(fx)
    y = (ys[valid] - float(cy)) * z / float(fy)
    xyz = np.stack([x, y, z], axis=1).astype(np.float32)

    object_points = xyz[mask[valid]]
    background_points = xyz[~mask[valid]]
    return object_points, background_points


def stable_downsample(points: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or len(points) <= max_points:
        return np.asarray(points, dtype=np.float32)
    indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
    return np.asarray(points[indices], dtype=np.float32)


def compute_centroid(points: np.ndarray, surface_band_m: float) -> np.ndarray | None:
    if len(points) == 0:
        return None
    z_min = float(points[:, 2].min())
    surface_mask = points[:, 2] <= (z_min + surface_band_m)
    surface_points = points[surface_mask]
    if len(surface_points) == 0:
        surface_points = points
    return surface_points.mean(axis=0).astype(np.float32)


def build_overlay_image(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    prompt_points: Iterable[PromptPoint],
    label: str,
) -> np.ndarray:
    overlay = image_bgr.copy()
    if mask.any():
        tint = np.zeros_like(overlay)
        tint[:, :] = (0, 200, 0)
        overlay[mask] = cv2.addWeighted(overlay, 0.55, tint, 0.45, 0.0)[mask]

        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)

        roi = compute_mask_roi(mask)
        if roi is not None:
            x_min, y_min, x_max, y_max = roi
            cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (255, 255, 0), 2)
            cv2.putText(
                overlay,
                label,
                (x_min, max(18, y_min - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )

    for point in prompt_points:
        color = (0, 255, 0) if point.positive else (0, 0, 255)
        cv2.circle(overlay, (point.x, point.y), 6, color, -1)
        cv2.circle(overlay, (point.x, point.y), 10, color, 2)
    return overlay


def parse_sam3_polygons(payload: dict) -> list[np.ndarray]:
    candidates: list[tuple[float, float, np.ndarray]] = []

    for prediction in _iter_prediction_dicts(payload):
        confidence = _coerce_float(prediction.get("confidence", prediction.get("score", 0.0)))
        for polygon in _extract_prediction_polygons(prediction):
            try:
                area = abs(float(cv2.contourArea(polygon.astype(np.float32))))
            except (TypeError, ValueError):
                continue
            if area > 0.0:
                candidates.append((area, confidence, polygon))

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [polygon for _, _, polygon in candidates]


def _iter_prediction_dicts(payload: object) -> Iterable[dict]:
    if isinstance(payload, dict):
        if _looks_like_prediction(payload):
            yield payload
        for value in payload.values():
            yield from _iter_prediction_dicts(value)
    elif isinstance(payload, list):
        for item in payload:
            yield from _iter_prediction_dicts(item)


def _looks_like_prediction(value: dict) -> bool:
    return any(key in value for key in ("masks", "polygon", "polygons", "segmentation"))


def _extract_prediction_polygons(prediction: dict) -> Iterable[np.ndarray]:
    keys = ("masks", "polygon", "polygons", "segmentation")
    for key in keys:
        if key not in prediction:
            continue
        yield from _coerce_polygon_collection(prediction[key])


def _coerce_polygon_collection(value: object) -> Iterable[np.ndarray]:
    polygon = _coerce_polygon(value)
    if polygon is not None:
        yield polygon
        return

    if isinstance(value, dict):
        for nested in value.values():
            yield from _coerce_polygon_collection(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _coerce_polygon_collection(nested)


def _coerce_polygon(value: object) -> np.ndarray | None:
    if not isinstance(value, list) or len(value) < 3:
        return None

    points = []
    for item in value:
        if isinstance(item, dict):
            if "x" in item and "y" in item:
                x = _coerce_float(item["x"])
                y = _coerce_float(item["y"])
                points.append((x, y))
            elif "X" in item and "Y" in item:
                x = _coerce_float(item["X"])
                y = _coerce_float(item["Y"])
                points.append((x, y))
            else:
                return None
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            x = _coerce_float(item[0])
            y = _coerce_float(item[1])
            points.append((x, y))
        else:
            return None

    polygon = np.asarray(points, dtype=np.float32)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
        return None
    return polygon


def _coerce_float(value: object) -> float:
    while isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("Empty sequence cannot be converted to float.")
        value = value[0]
    return float(value)
