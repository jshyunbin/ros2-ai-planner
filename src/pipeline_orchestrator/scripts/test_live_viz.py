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


# ── ROS2 node ─────────────────────────────────────────────────────────────────
class LiveVizNode(Node):
    """ROS2 node: integrates live depth into cuRobo Mapper and re-plans."""

    def __init__(self, state: SharedState, planner: MotionPlanner) -> None:
        super().__init__('live_viz_node')
        self._state   = state
        self._planner = planner
        self._bridge  = CvBridge()

        # Per-camera data — only touched in ROS2 callbacks; no lock needed
        self._cam_intrinsics: Dict[str, torch.Tensor] = {}
        self._cam_depth:      Dict[str, torch.Tensor] = {}
        self._cam_pose:       Dict[str, Pose]         = {}

        self._tf_buffer   = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._mapper = Mapper(MapperCfg(
            extent_meters_xyz=(2.0, 2.0, 1.5),
            voxel_size=0.02,
            esdf_voxel_size=0.05,
            truncation_distance=0.1,
            depth_minimum_distance=0.15,
            depth_maximum_distance=2.0,
            decay_factor=1.0,
            frustum_decay_factor=1.0,
            enable_static=False,
            num_cameras=2,
        ))
        self._depth_filter = FilterDepth(
            image_shape=(480, 640),
            depth_minimum_distance=0.15,
            depth_maximum_distance=2.0,
            flying_pixel_threshold=0.5,
            bilateral_kernel_size=3,
        )

        self.create_subscription(
            CameraInfo, OVERHEAD_INFO_TOPIC,
            lambda m: self._on_info(m, 'overhead'), 1)
        self.create_subscription(
            Image, OVERHEAD_DEPTH_TOPIC,
            lambda m: self._on_depth(m, 'overhead', OVERHEAD_FRAME), 10)
        self.create_subscription(
            CameraInfo, WRIST_INFO_TOPIC,
            lambda m: self._on_info(m, 'wrist'), 1)
        self.create_subscription(
            Image, WRIST_DEPTH_TOPIC,
            lambda m: self._on_depth(m, 'wrist', WRIST_FRAME), 10)
        self.create_subscription(
            JointState, '/joint_states', self._on_joints, 10)

        self.get_logger().info('LiveVizNode ready — waiting for depth frames.')

    # ── callbacks ─────────────────────────────────────────────────────────────

    def _on_joints(self, msg: JointState) -> None:
        with self._state.lock:
            self._state.latest_joints = msg

    def _on_info(self, msg: CameraInfo, cam_id: str) -> None:
        K = torch.tensor([
            [msg.k[0], 0.0,      msg.k[2]],
            [0.0,      msg.k[4], msg.k[5]],
            [0.0,      0.0,      1.0     ],
        ], dtype=torch.float32, device='cuda')
        self._cam_intrinsics[cam_id] = K

    def _on_depth(self, msg: Image, cam_id: str, frame: str) -> None:
        if cam_id not in self._cam_intrinsics:
            return   # wait for CameraInfo first

        K = self._cam_intrinsics[cam_id]

        # TF lookup: world ← camera_optical_frame at message timestamp
        try:
            tf_time   = Time(seconds=msg.header.stamp.sec,
                             nanoseconds=msg.header.stamp.nanosec)
            transform = self._tf_buffer.lookup_transform(
                WORLD_FRAME, frame, tf_time,
                timeout=rclpy.duration.Duration(seconds=0.1))
        except Exception as exc:
            self.get_logger().warning(
                f'TF lookup failed for {frame}: {exc}',
                throttle_duration_sec=2.0)
            return

        # Decode depth: uint16 mm → float32 m
        cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        depth  = torch.from_numpy(cv_img.astype(np.float32) / 1000.0).cuda()
        depth  = torch.nan_to_num(depth, nan=0.0)
        filtered, _ = self._depth_filter(depth.unsqueeze(0))
        depth  = filtered[0]

        # Camera pose in world frame (w, x, y, z quaternion convention)
        t = transform.transform.translation
        r = transform.transform.rotation
        pose = Pose.from_numpy(
            np.array([t.x, t.y, t.z], dtype=np.float32),
            np.array([r.w, r.x, r.y, r.z], dtype=np.float32),
        )

        # Unproject depth to XYZ (camera frame) for viser display
        xyz = depth_to_xyz(depth, K)

        # Cache per-camera data
        self._cam_depth[cam_id]      = depth
        self._cam_pose[cam_id]       = pose
        self._cam_intrinsics[cam_id] = K

        with self._state.lock:
            self._state.point_clouds[cam_id] = xyz

        # Integrate when both cameras have data
        if 'overhead' not in self._cam_depth or 'wrist' not in self._cam_depth:
            return

        batched = CameraObservation(
            depth_image=torch.stack([
                self._cam_depth['overhead'],
                self._cam_depth['wrist'],
            ]),
            intrinsics=torch.stack([
                self._cam_intrinsics['overhead'],
                self._cam_intrinsics['wrist'],
            ]),
            pose=Pose(
                position=torch.cat([
                    self._cam_pose['overhead'].position,
                    self._cam_pose['wrist'].position,
                ]),
                quaternion=torch.cat([
                    self._cam_pose['overhead'].quaternion,
                    self._cam_pose['wrist'].quaternion,
                ]),
            ),
        )
        self._mapper.integrate(batched)

        with self._state.lock:
            self._state.frame_count += 1
            count = self._state.frame_count

        if count >= MIN_FRAMES and count % REPLAN_EVERY == 0:
            self._replan()

    # ── planning ──────────────────────────────────────────────────────────────

    def _replan(self) -> None:
        """Compute a fresh ESDF and plan a new trajectory to GOAL_XYZ."""
        voxel_grid = self._mapper.compute_esdf()
        self._planner.update_world(SceneCfg(voxel=[voxel_grid]))

        with self._state.lock:
            self._state.voxel_grid = voxel_grid
            js = self._state.latest_joints

        # Start configuration: live joints or home pose fallback
        if js is not None:
            start = CuRoboJointState.from_position(
                torch.tensor(
                    [list(js.position)], dtype=torch.float32, device='cuda'),
                joint_names=list(js.name))
        else:
            start = CuRoboJointState.from_position(
                torch.tensor([HOME_CFG], dtype=torch.float32, device='cuda'),
                joint_names=JOINT_NAMES)

        goal = GoalToolPose(
            tool_frames=self._planner.tool_frames,
            position=torch.tensor(
                [[[[[GOAL_XYZ[0], GOAL_XYZ[1], GOAL_XYZ[2]]]]]], device='cuda',
                dtype=torch.float32),
            quaternion=torch.tensor(
                [[[[[GOAL_QUAT[0], GOAL_QUAT[1], GOAL_QUAT[2], GOAL_QUAT[3]]]]]], device='cuda',
                dtype=torch.float32),
        )

        result = self._planner.plan_pose(goal, start)
        if result is None or not result.success.any():
            self.get_logger().warning(
                'CuRobo planning failed — keeping previous trajectory.')
            return

        pos = result.get_interpolated_plan().position[0]
        while pos.dim() > 2:
            pos = pos[0]
        traj = pos.cpu().numpy()   # (T, J) float32
        self.get_logger().info(f'Planned {len(traj)}-waypoint trajectory.')

        with self._state.lock:
            self._state.traj = traj
