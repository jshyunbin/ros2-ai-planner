"""Tests for the CuRobo viz hooks + action deployment, and the viz script.

These cover the three stages the test_curobo_pipeline_viz.py script drives
through the *real* CuRobo orchestrator class:
  - perception   → get_point_clouds() / get_tsdf_centers() / frame_count
  - motion plan  → plan_trajectory()  (already covered in test_orchestrator)
  - action deploy→ execute_trajectory()

Heavy deps (torch, cuRobo, viser) are mocked so the suite runs without CUDA.
"""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch


def make_curobo(enable_viz=False):
    """Return a (CuRobo, node) pair with all heavy deps mocked."""
    from pipeline_orchestrator.curobo import CuRobo
    node = MagicMock()
    with patch.multiple('pipeline_orchestrator.curobo',
                        Mapper=MagicMock(), FilterDepth=MagicMock(),
                        MotionPlanner=MagicMock(), Buffer=MagicMock(),
                        TransformListener=MagicMock(),
                        ActionClient=MagicMock()):
        return CuRobo(node, enable_viz=enable_viz), node


# --- viz hooks (perception output for viser) ---

def test_curobo_viz_disabled_by_default():
    curobo, _ = make_curobo()
    assert curobo._enable_viz is False


def test_curobo_accepts_enable_viz_flag():
    curobo, _ = make_curobo(enable_viz=True)
    assert curobo._enable_viz is True


def test_curobo_get_point_clouds_empty_initially():
    curobo, _ = make_curobo(enable_viz=True)
    assert curobo.get_point_clouds() == {}


def test_curobo_get_tsdf_centers_none_initially():
    curobo, _ = make_curobo(enable_viz=True)
    assert curobo.get_tsdf_centers() is None


def test_curobo_frame_count_property_reflects_internal_counter():
    curobo, _ = make_curobo(enable_viz=True)
    assert curobo.frame_count == 0
    curobo._frame_count = 7
    assert curobo.frame_count == 7


# --- forward kinematics (for the "return home" leg) ---

def test_curobo_tool_pose_returns_none_on_fk_failure():
    from sensor_msgs.msg import JointState
    curobo, _ = make_curobo()
    curobo._planner.compute_kinematics.side_effect = RuntimeError('boom')
    js = JointState()
    js.name = ['shoulder_pan_joint']
    js.position = [0.0]
    assert curobo.tool_pose(js) is None


# --- action deployment ---

def test_curobo_execute_trajectory_sends_goal_to_arm_server():
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    curobo, _ = make_curobo()
    curobo._arm_client.wait_for_server.return_value = True

    traj = JointTrajectory()
    traj.joint_names = ['shoulder_pan_joint']
    traj.points = [JointTrajectoryPoint()]

    curobo.execute_trajectory(traj)

    curobo._arm_client.send_goal_async.assert_called_once()
    sent_goal = curobo._arm_client.send_goal_async.call_args.args[0]
    assert list(sent_goal.trajectory.joint_names) == ['shoulder_pan_joint']


def test_curobo_execute_trajectory_returns_none_when_server_unavailable():
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    curobo, _ = make_curobo()
    curobo._arm_client.wait_for_server.return_value = False

    traj = JointTrajectory()
    traj.points = [JointTrajectoryPoint()]
    result = curobo.execute_trajectory(traj)

    assert result is None
    curobo._arm_client.send_goal_async.assert_not_called()


# --- viz script importability + constants ---

def test_curobo_pipeline_viz_script_constants(monkeypatch):
    """scripts/test_curobo_pipeline_viz.py imports and exposes constants."""
    from pathlib import Path
    import importlib.util
    import sys
    import unittest.mock as mock

    stubs = [
        'rclpy', 'rclpy.node', 'rclpy.duration', 'rclpy.time', 'rclpy.qos',
        'rclpy.action',
        'sensor_msgs', 'sensor_msgs.msg',
        'std_msgs', 'std_msgs.msg',
        'trajectory_msgs', 'trajectory_msgs.msg',
        'control_msgs', 'control_msgs.action',
        'builtin_interfaces', 'builtin_interfaces.msg',
        'tf2_ros', 'cv_bridge', 'viser', 'viser.extras', 'warp',
        'torch',
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

    script_path = (Path(__file__).parent.parent / 'scripts'
                   / 'test_curobo_pipeline_viz.py')
    spec = importlib.util.spec_from_file_location(
        'test_curobo_pipeline_viz', script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert mod.MIN_FRAMES == 5
    assert mod.VIZ_HZ == 10
    assert mod.GOAL_XYZ == (0.3, 0.0, 0.4)
    assert mod.GOAL_QUAT == (1.0, 0.0, 0.0, 0.0)
    assert mod.WORLD_FRAME == 'world'
    assert callable(mod.main)
    assert callable(mod.update_loop)
