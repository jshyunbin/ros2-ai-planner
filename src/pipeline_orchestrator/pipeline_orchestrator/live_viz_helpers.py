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
