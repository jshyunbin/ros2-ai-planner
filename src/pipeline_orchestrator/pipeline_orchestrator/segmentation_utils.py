from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class PromptPoint:
    x: int
    y: int
    positive: bool = True


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


def compute_mask_roi(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())
    return x_min, y_min, x_max, y_max


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
