"""Loader, validator, and Pose builder for the hardcoded place/home poses.

Reads config/place_poses.yml. Kept dependency-light (yaml + a guarded
geometry_msgs import) so it can be unit-tested without a live ROS graph,
mirroring pipeline_utils.py.
"""

from os import PathLike
from pathlib import Path

import yaml

try:  # pragma: no cover - geometry_msgs only present in the ROS runtime
    from geometry_msgs.msg import Pose
except ImportError:  # pragma: no cover - import-only test fallback
    Pose = None


SIMPLE_TARGETS = ("home", "storage_1", "storage_2")
BOOKSHELF_TARGETS = ("bookshelf_floor1", "bookshelf_floor2")


def load_place_poses(path: "str | PathLike") -> dict:
    """Parse place_poses.yml into a plain dict, validating its structure.

    Raises ValueError if a required key is missing or malformed.
    """
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"place_poses file is not a mapping: {path}")
    if not isinstance(data.get("transit_z"), (int, float)):
        raise ValueError("place_poses: 'transit_z' must be a number")
    for name in SIMPLE_TARGETS:
        _validate_xyzquat(data.get(name), name)
    for name in BOOKSHELF_TARGETS:
        shelf = data.get(name)
        if not isinstance(shelf, dict):
            raise ValueError(f"place_poses: missing/invalid '{name}'")
        _validate_xyzquat(shelf.get("pre_insert"), f"{name}.pre_insert")
        for key in ("insert_depth_m", "retract_depth_m"):
            if not isinstance(shelf.get(key), (int, float)):
                raise ValueError(f"place_poses: '{name}.{key}' must be a number")
    return data


def _validate_xyzquat(entry, label) -> None:
    """Raise ValueError if *entry* lacks a valid xyz (3) / quat_xyzw (4) pair."""
    if not isinstance(entry, dict):
        raise ValueError(f"place_poses: '{label}' must be a mapping")
    xyz = entry.get("xyz")
    quat = entry.get("quat_xyzw")
    if not (isinstance(xyz, list) and len(xyz) == 3):
        raise ValueError(f"place_poses: '{label}.xyz' must be a 3-list")
    if not (isinstance(quat, list) and len(quat) == 4):
        raise ValueError(f"place_poses: '{label}.quat_xyzw' must be a 4-list")


def pose_from_xyzquat(xyz, quat_xyzw):
    """Build a geometry_msgs/Pose from xyz + (x, y, z, w) quaternion.

    Returns None when geometry_msgs is unavailable (import-only test fallback).
    """
    if Pose is None:
        return None
    pose = Pose()
    pose.position.x = float(xyz[0])
    pose.position.y = float(xyz[1])
    pose.position.z = float(xyz[2])
    pose.orientation.x = float(quat_xyzw[0])
    pose.orientation.y = float(quat_xyzw[1])
    pose.orientation.z = float(quat_xyzw[2])
    pose.orientation.w = float(quat_xyzw[3])
    return pose


def resolve_target_pose(data, target):
    """Return the geometry_msgs/Pose for a target name.

    Bookshelf targets resolve to their pre_insert pose. Raises KeyError for an
    unknown target name.
    """
    if target in BOOKSHELF_TARGETS:
        entry = data[target]["pre_insert"]
    elif target in SIMPLE_TARGETS:
        entry = data[target]
    else:
        raise KeyError(f"unknown target '{target}'")
    return pose_from_xyzquat(entry["xyz"], entry["quat_xyzw"])
