"""Pure helper functions shared by live visualization scripts.

All functions here are stateless and have no ROS2 or cuRobo imports,
so they can be unit-tested directly without a ROS2 environment.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch


def depth_to_xyz(depth_m: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Unproject a depth image into a point cloud in camera frame.

    Args:
        depth_m: (H, W) float32 tensor, metres; zero values are invalid.
        K:       (3, 3) float32 tensor, camera intrinsics matrix.
                 Works on any device — output is on the same device as depth_m.

    Returns:
        (N, 3) float32 tensor of valid (x, y, z) points in camera frame.
        Returns shape (0, 3) when there are no valid pixels.
    """
    H, W = depth_m.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    v_coords, u_coords = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=depth_m.device),
        torch.arange(W, dtype=torch.float32, device=depth_m.device),
        indexing='ij',
    )

    valid = depth_m > 0

    # Handle case where there are no valid pixels
    if not valid.any():
        return torch.zeros(0, 3, dtype=torch.float32, device=depth_m.device)

    z = depth_m[valid]
    x = (u_coords[valid] - cx) / fx * z
    y = (v_coords[valid] - cy) / fy * z

    return torch.stack([x, y, z], dim=-1)


def esdf_to_points(voxel_grid: object) -> np.ndarray:
    """Extract occupied voxel centres from a cuRobo ESDF VoxelGrid.

    cuRobo's nvblox ESDF stores **inverted** signed distance:
      - positive  → inside an obstacle (penetration depth)
      - zero      → on the surface
      - negative  → free space (distance to nearest obstacle)

    A voxel is occupied when ``feature_tensor > -0.5 * voxel_size`` —
    matching cuRobo's internal collision threshold (see VoxelGrid
    .get_occupied_voxels in curobo._src.geom.types).

    Args:
        voxel_grid: cuRobo VoxelGrid with ``feature_tensor`` (nx, ny, nz)
                    CUDA tensor, ``pose`` (7,) list (xyz + wxyz at centre),
                    ``dims`` (3,) list (metres), and ``voxel_size`` scalar.

    Returns:
        (M, 3) float32 numpy array of world-frame XYZ voxel centres.
        Returns shape (0, 3) when the grid is entirely free or unreadable.
    """
    try:
        feat: torch.Tensor = voxel_grid.feature_tensor   # (nx, ny, nz)
        if feat is None:
            return np.zeros((0, 3), dtype=np.float32)
        vsize = float(voxel_grid.voxel_size)
        threshold = -0.5 * vsize
        occupied = feat > threshold
        if not occupied.any():
            return np.zeros((0, 3), dtype=np.float32)

        # Pose is at grid centre; origin = bottom-left-front corner.
        pose_xyz = np.asarray(voxel_grid.pose[:3], dtype=np.float32)
        dims     = np.asarray(voxel_grid.dims,     dtype=np.float32)
        origin   = pose_xyz - dims / 2.0

        indices = torch.argwhere(occupied).float().cpu().numpy()  # (M, 3)
        centres = origin + (indices + 0.5) * vsize
        return centres.astype(np.float32)
    except Exception:
        return np.zeros((0, 3), dtype=np.float32)


def _rewrite_package_uris(content: str) -> str:
    """Rewrite URDF mesh URIs to plain absolute paths.

    Handles two schemes seen in published URDFs:
      - ``package://<pkg>/...`` → ``/opt/ros/humble/share/<pkg>/...``
      - ``file:///...``         → ``/...``  (strip scheme; yourdfpy chokes on file://)
    """
    import re
    content = re.sub(r'package://([^/"]+)', r'/opt/ros/humble/share/\1', content)
    content = content.replace('file://', '')
    return content


def resolve_urdf(urdf_path: str) -> Path:
    """Rewrite ``package://`` URIs to absolute paths; return a temp file path.

    Args:
        urdf_path: Absolute path to the URDF file (may contain package:// URIs).

    Returns:
        Path to a temp URDF file with all URIs resolved.
    """
    with open(urdf_path) as f:
        content = f.read()
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False)
    tmp.write(_rewrite_package_uris(content))
    tmp.close()
    return Path(tmp.name)


def resolve_urdf_string(content: str) -> Path:
    """Rewrite ``package://`` URIs in a URDF string; write to a temp file.

    Designed for use with the ``/robot_description`` ROS2 topic which
    publishes the full robot URDF as a string.

    Args:
        content: Raw URDF XML string (may contain package:// URIs).

    Returns:
        Path to a temp URDF file with all URIs resolved.
    """
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False)
    tmp.write(_rewrite_package_uris(content))
    tmp.close()
    return Path(tmp.name)
