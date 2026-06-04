import math

import pytest

from team_8.pipeline_utils import quat_from_rotation_matrix


def test_quat_from_identity_is_unit_quaternion():
    # (w, x, y, z) ordering, matching the orchestrator's original convention.
    w, x, y, z = quat_from_rotation_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert w == pytest.approx(1.0)
    assert (x, y, z) == pytest.approx((0.0, 0.0, 0.0))


def test_quat_from_180_deg_z_rotation():
    # 180° about Z: w=0, z=±1.
    w, x, y, z = quat_from_rotation_matrix([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    assert w == pytest.approx(0.0, abs=1e-6)
    assert abs(z) == pytest.approx(1.0, abs=1e-6)
    assert (x, y) == pytest.approx((0.0, 0.0), abs=1e-6)


def test_quat_is_normalized():
    rot = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]  # 90° about Z
    q = quat_from_rotation_matrix(rot)
    norm = math.sqrt(sum(c * c for c in q))
    assert norm == pytest.approx(1.0, abs=1e-6)


def test_quat_from_180_deg_x_rotation():
    # 180° about X hits the r00-dominant branch: (w, x, y, z) = (0, 1, 0, 0).
    w, x, y, z = quat_from_rotation_matrix([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    assert w == pytest.approx(0.0, abs=1e-6)
    assert abs(x) == pytest.approx(1.0, abs=1e-6)
    assert (y, z) == pytest.approx((0.0, 0.0), abs=1e-6)


def test_quat_from_180_deg_y_rotation():
    # 180° about Y hits the r11-dominant branch: (w, x, y, z) = (0, 0, 1, 0).
    w, x, y, z = quat_from_rotation_matrix([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
    assert w == pytest.approx(0.0, abs=1e-6)
    assert abs(y) == pytest.approx(1.0, abs=1e-6)
    assert (x, z) == pytest.approx((0.0, 0.0), abs=1e-6)
