import torch
import numpy as np
import types


def test_depth_to_xyz_single_pixel():
    """Valid pixel at (u=0, v=0) with depth=1.0 m unprojected correctly."""
    from team_8.live_viz_helpers import depth_to_xyz

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
    from team_8.live_viz_helpers import depth_to_xyz

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
    from team_8.live_viz_helpers import depth_to_xyz

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


def test_live_viz_script_constants(monkeypatch):
    """test_live_viz.py must be importable and expose required constants."""
    from pathlib import Path
    import importlib.util
    import sys
    import unittest.mock as mock

    # Stub every heavy dep so we don't need a live ROS2/CUDA context
    stubs = [
        'rclpy', 'rclpy.node', 'rclpy.duration', 'rclpy.time', 'rclpy.qos',
        'sensor_msgs', 'sensor_msgs.msg',
        'std_msgs', 'std_msgs.msg',
        'tf2_ros', 'cv_bridge', 'viser', 'viser.extras', 'warp',
        'curobo', 'curobo.perception', 'curobo.motion_planner',
        'curobo.types', 'curobo._src', 'curobo._src.geom',
        'curobo._src.geom.types',
        'curobo._src.robot', 'curobo._src.robot.kinematics',
        'curobo._src.robot.kinematics.kinematics',
        'curobo._src.types', 'curobo._src.types.robot',
        'curobo._src.util_file',
    ]
    for name in stubs:
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, mock.MagicMock())

    script_path = Path(__file__).parent.parent / 'scripts' / 'test_live_viz.py'
    spec = importlib.util.spec_from_file_location('test_live_viz', script_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert mod.MIN_FRAMES   == 5
    assert mod.REPLAN_EVERY == 10
    assert mod.VIZ_HZ       == 10
    assert mod.GOAL_XYZ     == (0.3, 0.0, 0.4)
    assert mod.GOAL_QUAT    == (1.0, 0.0, 0.0, 0.0)
    assert mod.WORLD_FRAME  == 'base_link'
