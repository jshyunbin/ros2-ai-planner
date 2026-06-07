"""Place and home pose management for the pipeline orchestrator.

Loads target poses from place_poses.yml and provides helpers for building
safe-Z transit waypoints used by plan_place().

YAML schema (config/place_poses.yml):
  transit_z: 0.35          # safe transit height in metres (base_link Z)
  home:
    xyz: [x, y, z]
    quat_xyzw: [qx, qy, qz, qw]
  storage_1:               # simple place target
    xyz: [x, y, z]
    quat_xyzw: [qx, qy, qz, qw]
  bookshelf_a:             # bookshelf-style target with insert/retract
    pre_insert:
      xyz: [x, y, z]
      quat_xyzw: [qx, qy, qz, qw]
    insert_depth_m: 0.08
    retract_depth_m: 0.06
"""

import os
from pathlib import Path

import numpy as np
import yaml

try:
    from ament_index_python.packages import get_package_share_directory
    _PKG = 'pipeline_orchestrator'
except ImportError:
    get_package_share_directory = None
    _PKG = None

try:
    from geometry_msgs.msg import Pose
except ImportError:
    Pose = None


def _default_yaml_path() -> Path:
    if get_package_share_directory is not None and _PKG is not None:
        try:
            share = get_package_share_directory(_PKG)
            return Path(share) / 'config' / 'place_poses.yml'
        except Exception:
            pass
    # Fallback for testing without ROS install
    return Path(__file__).parent.parent / 'config' / 'place_poses.yml'


def load_place_poses(path: str | None = None) -> dict:
    """Parse place_poses.yml and validate required fields.

    Returns the raw dict so callers can inspect transit_z and target entries.
    """
    yaml_path = Path(path) if path else _default_yaml_path()
    if not yaml_path.exists():
        raise FileNotFoundError(
            f'place_poses.yml not found at {yaml_path}. '
            'Copy config/place_poses.yml.example and fill in your robot poses.')

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError('place_poses.yml must be a YAML mapping.')
    if 'transit_z' not in data or not isinstance(data['transit_z'], (int, float)):
        raise ValueError('place_poses.yml must have a numeric transit_z field.')
    if 'home' not in data:
        raise ValueError('place_poses.yml must have a home target.')

    return data


def is_bookshelf_target(cfg: dict, goal_name: str) -> bool:
    """Return True if goal_name is a bookshelf-style entry (has pre_insert)."""
    entry = cfg.get(goal_name, {})
    return isinstance(entry, dict) and 'pre_insert' in entry


def pose_from_xyzquat(xyz, quat_xyzw) -> 'Pose | None':
    """Build a geometry_msgs/Pose from xyz + quaternion (x,y,z,w)."""
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


def resolve_target_pose(cfg: dict, goal_name: str) -> 'Pose | None':
    """Return the geometry_msgs/Pose for goal_name.

    For bookshelf targets, returns the pre_insert pose (arm approaches here,
    then insert_trajectory pushes to final depth).
    """
    entry = cfg.get(goal_name)
    if entry is None:
        raise KeyError(f'goal_name {goal_name!r} not found in place_poses.yml')

    if is_bookshelf_target(cfg, goal_name):
        sub = entry['pre_insert']
        return pose_from_xyzquat(sub['xyz'], sub['quat_xyzw'])

    return pose_from_xyzquat(entry['xyz'], entry['quat_xyzw'])


def build_transit_waypoints(
    current_xyz: list | tuple,
    target_pose: 'Pose',
    transit_z: float,
) -> list:
    """Build three waypoint Poses for a safe-Z transit.

    Sequence:
      1. Lift   — same XY as current, Z = max(current_z, transit_z)
      2. Transit — target XY at transit_z height, target orientation
      3. Descend — target pose

    The lift step uses target orientation to avoid reorienting while carrying
    the object.  All waypoints are geometry_msgs/Pose.
    """
    if Pose is None:
        raise ImportError('geometry_msgs is required for build_transit_waypoints')

    cur_x, cur_y, cur_z = float(current_xyz[0]), float(current_xyz[1]), float(current_xyz[2])
    tgt_x = target_pose.position.x
    tgt_y = target_pose.position.y
    tgt_z = target_pose.position.z
    lift_z = max(cur_z, float(transit_z))

    orient = target_pose.orientation  # keep target orientation throughout transit

    def _pose(x, y, z):
        p = Pose()
        p.position.x = float(x)
        p.position.y = float(y)
        p.position.z = float(z)
        p.orientation = orient
        return p

    return [
        _pose(cur_x, cur_y, lift_z),   # 1. lift to safe height
        _pose(tgt_x, tgt_y, transit_z),  # 2. traverse horizontally
        _pose(tgt_x, tgt_y, tgt_z),    # 3. descend to target
    ]


def translate_pose_along_x(pose: 'Pose', delta_x: float) -> 'Pose':
    """Return a copy of pose shifted delta_x metres along the base_link X-axis."""
    if Pose is None:
        raise ImportError('geometry_msgs is required')
    p = Pose()
    p.position.x = pose.position.x + float(delta_x)
    p.position.y = pose.position.y
    p.position.z = pose.position.z
    p.orientation = pose.orientation
    return p
