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


def load_place_poses(path: "str | PathLike") -> dict:
    """Parse place_poses.yml into a plain dict, validating its structure.

    Data-driven: requires a numeric ``transit_z`` and a simple ``home`` target.
    Every other top-level entry is validated as either a simple xyz/quat target
    or a bookshelf-style entry (``pre_insert`` + numeric insert/retract depths).
    Raises ValueError on any malformed entry.
    """
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"place_poses file is not a mapping: {path}")
    if not isinstance(data.get("transit_z"), (int, float)):
        raise ValueError("place_poses: 'transit_z' must be a number")
    if "transit_floor_z" in data and not isinstance(
            data["transit_floor_z"], (int, float)):
        raise ValueError("place_poses: 'transit_floor_z' must be a number")
    if "home" not in data:
        raise ValueError("place_poses: missing 'home' target")
    # Optional scalar (non-pose) config keys skipped by the per-target validation.
    scalar_keys = {"transit_z", "transit_floor_z"}
    for name, entry in data.items():
        if name in scalar_keys:
            continue
        if _looks_like_bookshelf(entry):
            _validate_xyzquat(entry.get("pre_insert"), f"{name}.pre_insert")
            for key in ("insert_depth_m", "retract_depth_m"):
                if not isinstance(entry.get(key), (int, float)):
                    raise ValueError(
                        f"place_poses: '{name}.{key}' must be a number")
        else:
            _validate_xyzquat(entry, name)
    return data


def _looks_like_bookshelf(entry) -> bool:
    return isinstance(entry, dict) and "pre_insert" in entry


def is_bookshelf_target(data, key) -> bool:
    """True when *key* resolves to a bookshelf-style entry (insert/retract)."""
    entry = data.get(key)
    return _looks_like_bookshelf(entry) and "insert_depth_m" in entry


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

    Bookshelf targets resolve to their ``pre_insert`` pose; simple targets to
    their ``xyz``/``quat_xyzw``. Raises KeyError for an unknown target name.
    """
    if target not in data:
        raise KeyError(f"unknown target '{target}'")
    entry = data[target]
    if _looks_like_bookshelf(entry):
        entry = entry["pre_insert"]
    return pose_from_xyzquat(entry["xyz"], entry["quat_xyzw"])


def build_transit_waypoints(current_xyz, current_quat_xyzw,
                            place_xyz, place_quat_xyzw, transit_z):
    """Three tool0 targets for the rule-based safe-z transit.

    1. lift straight up to ``transit_z`` (keep current xy + orientation)
    2. traverse to above the place xy at ``transit_z`` (place orientation)
    3. descend to the full place pose

    The lift height is ``max(current_z, transit_z)`` so the carried object is
    never driven *down* when the post-pick pose already sits above ``transit_z``.

    Returns a list of ``(xyz, quat_xyzw)`` tuples (plain Python lists).
    """
    lift_z = max(float(current_xyz[2]), float(transit_z))
    lift = ([float(current_xyz[0]), float(current_xyz[1]), lift_z],
            [float(q) for q in current_quat_xyzw])
    traverse = ([float(place_xyz[0]), float(place_xyz[1]), lift_z],
                [float(q) for q in place_quat_xyzw])
    descend = ([float(v) for v in place_xyz],
               [float(q) for q in place_quat_xyzw])
    return [lift, traverse, descend]


def translate_pose_x(xyz, dx):
    """Return *xyz* translated by *dx* along base_link +x (y, z unchanged)."""
    return [float(xyz[0]) + float(dx), float(xyz[1]), float(xyz[2])]
