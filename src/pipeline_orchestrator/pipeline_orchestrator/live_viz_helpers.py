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

    A voxel is "occupied" when its ESDF value is < 0 (strictly inside
    an obstacle; the zero surface is treated as free to avoid noise).

    Args:
        voxel_grid: cuRobo VoxelGrid, or any duck-typed object with
                    ``esdf_tensor`` (X, Y, Z) CUDA tensor,
                    ``origin`` (3,) tensor, and ``voxel_size`` scalar.

    Returns:
        (M, 3) float32 numpy array of world-frame XYZ voxel centres.
        Returns shape (0, 3) on failure or when the grid is entirely free.
    """
    try:
        esdf: torch.Tensor = voxel_grid.esdf_tensor   # (X, Y, Z)
        occupied = esdf < 0.0
        if not occupied.any():
            return np.zeros((0, 3), dtype=np.float32)

        origin  = voxel_grid.origin.cpu().numpy()     # (3,)
        vsize   = float(voxel_grid.voxel_size)
        indices = torch.argwhere(occupied).float().cpu().numpy()  # (M, 3)
        centres = origin + indices * vsize + vsize / 2.0
        return centres.astype(np.float32)
    except Exception:
        return np.zeros((0, 3), dtype=np.float32)


def resolve_urdf(urdf_path: str) -> Path:
    """Rewrite ``package://`` URIs to absolute paths; return a temp file path.

    viser's URDF loader does not handle ROS2 package URIs. This rewrites
    ``package://ur_description`` to the absolute ROS2 share directory and
    writes the result to a temp file that persists for the session.

    Args:
        urdf_path: Absolute path to the URDF file (may contain package:// URIs).

    Returns:
        Path to a temp URDF file with all URIs resolved.
    """
    pkg_root = '/opt/ros/humble/share'
    with open(urdf_path) as f:
        content = f.read()
    content = content.replace(
        'package://ur_description', f'{pkg_root}/ur_description')
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False)
    tmp.write(content)
    tmp.close()
    return Path(tmp.name)
