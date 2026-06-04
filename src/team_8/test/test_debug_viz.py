from types import SimpleNamespace

import pytest

try:
    from team_8.debug_viz import (
        colors_by_rank,
        pose_to_position_wxyz,
        _cloud_to_xyz,
    )
    _IMPORT_ERROR = None
except Exception as exc:  # noqa: BLE001 - missing runtime dep should skip
    colors_by_rank = None
    pose_to_position_wxyz = None
    _cloud_to_xyz = None
    _IMPORT_ERROR = exc

pytestmark = pytest.mark.skipif(
    colors_by_rank is None,
    reason=f"debug_viz import unavailable: {_IMPORT_ERROR}",
)


def test_colors_by_rank_empty():
    assert colors_by_rank(0) == []


def test_colors_by_rank_single_is_green():
    assert colors_by_rank(1) == [(0, 255, 0)]


def test_colors_by_rank_first_green_last_red():
    colors = colors_by_rank(4)
    assert len(colors) == 4
    assert colors[0] == (0, 255, 0)      # best rank → green
    assert colors[-1] == (255, 0, 0)     # worst rank → red


def test_pose_to_position_wxyz_extracts_fields():
    pose = SimpleNamespace(
        position=SimpleNamespace(x=1.0, y=2.0, z=3.0),
        orientation=SimpleNamespace(w=1.0, x=0.0, y=0.0, z=0.0),
    )
    position, wxyz = pose_to_position_wxyz(pose)
    assert position == (1.0, 2.0, 3.0)
    assert wxyz == (1.0, 0.0, 0.0, 0.0)


def test_cloud_to_xyz_roundtrip_and_empty():
    import numpy as np
    from team_8.pipeline_utils import make_xyz_cloud

    pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    cloud = make_xyz_cloud(pts, "base_link")
    out = _cloud_to_xyz(cloud)
    assert out.shape == (2, 3)
    assert np.allclose(out, pts)

    empty = make_xyz_cloud(np.empty((0, 3), dtype=np.float32), "base_link")
    assert _cloud_to_xyz(empty).shape == (0, 3)
