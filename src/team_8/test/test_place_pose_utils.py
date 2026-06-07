import textwrap

import pytest

from team_8.place_pose_utils import (
    load_place_poses,
    pose_from_xyzquat,
    resolve_target_pose,
)

_YAML = textwrap.dedent("""
    transit_z: 0.80
    home:      {xyz: [0.55, 0.07, 0.90], quat_xyzw: [1.0, 0.0, 0.0, 0.0]}
    storage_1: {xyz: [0.0,  0.55, 0.85], quat_xyzw: [1.0, 0.0, 0.0, 0.0]}
    storage_2: {xyz: [0.0, -0.55, 0.85], quat_xyzw: [1.0, 0.0, 0.0, 0.0]}
    bookshelf_floor1:
      pre_insert:      {xyz: [0.60, -0.30, 0.55], quat_xyzw: [0.5, 0.5, 0.5, 0.5]}
      insert_depth_m:  0.22
      retract_depth_m: 0.22
    bookshelf_floor2:
      pre_insert:      {xyz: [0.60, -0.30, 0.76], quat_xyzw: [0.5, 0.5, 0.5, 0.5]}
      insert_depth_m:  0.22
      retract_depth_m: 0.22
""")


def _write(tmp_path, text):
    path = tmp_path / "place_poses.yml"
    path.write_text(text)
    return path


def test_load_parses_transit_and_simple_target(tmp_path):
    data = load_place_poses(_write(tmp_path, _YAML))
    assert data["transit_z"] == pytest.approx(0.80)
    assert data["storage_1"]["xyz"] == [0.0, 0.55, 0.85]
    assert data["storage_1"]["quat_xyzw"] == [1.0, 0.0, 0.0, 0.0]


def test_load_parses_bookshelf(tmp_path):
    data = load_place_poses(_write(tmp_path, _YAML))
    shelf = data["bookshelf_floor2"]
    assert shelf["pre_insert"]["xyz"] == [0.60, -0.30, 0.76]
    assert shelf["insert_depth_m"] == pytest.approx(0.22)


def test_load_rejects_missing_transit_z(tmp_path):
    bad = _YAML.replace("transit_z: 0.80\n", "")
    with pytest.raises(ValueError):
        load_place_poses(_write(tmp_path, bad))


def test_load_rejects_bad_xyz_length(tmp_path):
    bad = _YAML.replace("xyz: [0.55, 0.07, 0.90]", "xyz: [0.55, 0.07]")
    with pytest.raises(ValueError):
        load_place_poses(_write(tmp_path, bad))


def test_pose_from_xyzquat_sets_fields():
    pose = pose_from_xyzquat([0.1, 0.2, 0.3], [0.0, 0.0, 0.0, 1.0])
    assert (pose.position.x, pose.position.y, pose.position.z) == pytest.approx(
        (0.1, 0.2, 0.3))
    assert (pose.orientation.x, pose.orientation.y,
            pose.orientation.z, pose.orientation.w) == pytest.approx(
        (0.0, 0.0, 0.0, 1.0))


def test_resolve_simple_target_returns_pose(tmp_path):
    data = load_place_poses(_write(tmp_path, _YAML))
    pose = resolve_target_pose(data, "storage_1")
    assert pose.position.y == pytest.approx(0.55)


def test_resolve_bookshelf_returns_pre_insert(tmp_path):
    data = load_place_poses(_write(tmp_path, _YAML))
    pose = resolve_target_pose(data, "bookshelf_floor1")
    assert pose.position.z == pytest.approx(0.55)


def test_resolve_unknown_target_raises_keyerror(tmp_path):
    data = load_place_poses(_write(tmp_path, _YAML))
    with pytest.raises(KeyError):
        resolve_target_pose(data, "nonexistent")
