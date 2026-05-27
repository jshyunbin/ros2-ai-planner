#!/usr/bin/env python3
"""
Live pipeline visualization test: real ROS2 depth → cuRobo Mapper → MotionPlanner → viser

Run inside the container:
  docker compose run --rm -p 8080:8080 ai_planner \\
    python3 /ros2_ws/src/pipeline_orchestrator/scripts/test_live_viz.py

Then open http://localhost:8080 in your browser.
SSH users — forward the port first:
  ssh -L 8080:localhost:8080 user@host
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener
import rclpy.duration
from rclpy.time import Time
from cv_bridge import CvBridge
import viser

from curobo.perception import FilterDepth, Mapper, MapperCfg
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import CameraObservation, Pose
from curobo.types import JointState as CuRoboJointState, GoalToolPose
from curobo._src.geom.types import SceneCfg

from pipeline_orchestrator.live_viz_helpers import (
    depth_to_xyz,
    esdf_to_points,
    resolve_urdf,
)

# ── constants ─────────────────────────────────────────────────────────────────
UR5_CONFIG   = '/ros2_ws/src/pipeline_orchestrator/config/ur5_curobo.yml'
URDF_PATH    = '/ur5.urdf'
JOINT_NAMES  = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
]
HOME_CFG     = [0.0, -2.2, 1.9, -1.383, -1.57, 0.0]
GOAL_XYZ     = (0.3, 0.0, 0.4)
GOAL_QUAT    = (1.0, 0.0, 0.0, 0.0)    # w x y z
MIN_FRAMES   = 5
REPLAN_EVERY = 10
VIZ_HZ       = 10

OVERHEAD_DEPTH_TOPIC = '/camera/camera/depth/color/image_raw'
OVERHEAD_INFO_TOPIC  = '/camera/camera/depth/camera_info'
WRIST_DEPTH_TOPIC    = '/wrist_camera/wrist_camera/depth/color/image_raw'
WRIST_INFO_TOPIC     = '/wrist_camera/wrist_camera/depth/camera_info'
OVERHEAD_FRAME       = 'camera_color_optical_frame'
WRIST_FRAME          = 'wrist_camera_color_optical_frame'
WORLD_FRAME          = 'world'


# ── shared state ──────────────────────────────────────────────────────────────
@dataclass
class SharedState:
    """Thread-safe container for data shared between ROS2 callbacks and viser."""
    lock:          threading.Lock                    = field(default_factory=threading.Lock)
    point_clouds:  Dict[str, Optional[torch.Tensor]] = field(default_factory=dict)
    voxel_grid:    Optional[object]                  = None   # cuRobo VoxelGrid
    traj:          Optional[np.ndarray]              = None   # (T, J) float32
    frame_count:   int                               = 0
    latest_joints: Optional[object]                  = None   # sensor_msgs/JointState
