import torch
import numpy as np
import types


def test_depth_to_xyz_single_pixel():
    """Valid pixel at (u=0, v=0) with depth=1.0 m unprojected correctly."""
    from pipeline_orchestrator.live_viz_helpers import depth_to_xyz

    K = torch.tensor([
        [500.0,   0.0, 320.0],
        [  0.0, 500.0, 240.0],
        [  0.0,   0.0,   1.0],
    ], dtype=torch.float32)
    depth = torch.zeros(2, 2, dtype=torch.float32)
    depth[0, 0] = 1.0   # valid pixel at (v=0, u=0)

    xyz = depth_to_xyz(depth, K)

    assert xyz.shape == (1, 3)
    # x = (u - cx)/fx * z = (0 - 320)/500 * 1 = -0.64
    assert abs(xyz[0, 0].item() - (-0.64)) < 1e-5
    # y = (v - cy)/fy * z = (0 - 240)/500 * 1 = -0.48
    assert abs(xyz[0, 1].item() - (-0.48)) < 1e-5
    # z = depth = 1.0
    assert abs(xyz[0, 2].item() - 1.0) < 1e-5


def test_depth_to_xyz_zero_pixels_excluded():
    """Pixels with depth == 0 must not appear in output."""
    from pipeline_orchestrator.live_viz_helpers import depth_to_xyz

    K = torch.tensor([
        [500.0,   0.0, 320.0],
        [  0.0, 500.0, 240.0],
        [  0.0,   0.0,   1.0],
    ], dtype=torch.float32)
    depth = torch.zeros(4, 4, dtype=torch.float32)   # all zero → all invalid

    xyz = depth_to_xyz(depth, K)

    assert xyz.shape[0] == 0


def test_depth_to_xyz_centre_pixel():
    """Pixel at the principal point (cx, cy) should have x=y=0."""
    from pipeline_orchestrator.live_viz_helpers import depth_to_xyz

    fx, fy, cx, cy = 600.0, 600.0, 80.0, 60.0
    H, W = 120, 160
    K = torch.tensor([
        [fx,  0.0, cx],
        [0.0, fy,  cy],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32)
    depth = torch.zeros(H, W, dtype=torch.float32)
    depth[int(cy), int(cx)] = 2.0   # at principal point, depth = 2 m

    xyz = depth_to_xyz(depth, K)

    assert xyz.shape == (1, 3)
    assert abs(xyz[0, 0].item()) < 1e-4    # x ≈ 0
    assert abs(xyz[0, 1].item()) < 1e-4    # y ≈ 0
    assert abs(xyz[0, 2].item() - 2.0) < 1e-5


def test_esdf_to_points_extracts_occupied():
    """Voxels with ESDF ≤ 0 are occupied; their centres should be returned."""
    from pipeline_orchestrator.live_viz_helpers import esdf_to_points

    esdf = torch.zeros(2, 2, 2, dtype=torch.float32)
    esdf[0, 0, 0] = -0.1   # occupied
    esdf[1, 0, 0] =  0.1   # free

    vg = types.SimpleNamespace(
        esdf_tensor=esdf,
        origin=torch.tensor([0.0, 0.0, 0.0]),
        voxel_size=0.1,
    )

    pts = esdf_to_points(vg)

    assert pts.shape == (1, 3)
    # centre of voxel index (0,0,0): origin + 0*size + size/2 = 0.05
    np.testing.assert_allclose(pts[0], [0.05, 0.05, 0.05], atol=1e-6)


def test_esdf_to_points_all_free():
    """Grid with all positive ESDF values should return empty (0, 3) array."""
    from pipeline_orchestrator.live_viz_helpers import esdf_to_points

    vg = types.SimpleNamespace(
        esdf_tensor=torch.ones(2, 2, 2, dtype=torch.float32),
        origin=torch.tensor([0.0, 0.0, 0.0]),
        voxel_size=0.1,
    )

    pts = esdf_to_points(vg)

    assert pts.shape == (0, 3)
    assert pts.dtype == np.float32


def test_esdf_to_points_malformed_object():
    """Missing attributes on voxel_grid must return empty array, not raise."""
    from pipeline_orchestrator.live_viz_helpers import esdf_to_points

    vg = types.SimpleNamespace()   # no attributes at all

    pts = esdf_to_points(vg)

    assert pts.shape == (0, 3)
