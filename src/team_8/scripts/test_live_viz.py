#!/usr/bin/env python3
"""
Live pipeline visualization test: real ROS2 depth → cuRobo Mapper → MotionPlanner → viser

Run inside the container:
  docker compose run --rm -p 8080:8080 ai_planner \\
    python3 /ros2_ws/src/team_8/scripts/test_live_viz.py

Then open http://localhost:8080 in your browser.
SSH users — forward the port first:
  ssh -L 8080:localhost:8080 user@host
"""
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
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener
import rclpy.duration
from rclpy.qos import (
    qos_profile_sensor_data, QoSProfile,
    QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy,
)
from cv_bridge import CvBridge
from pathlib import Path
import viser

from curobo.perception import FilterDepth, Mapper, MapperCfg, RobotSegmenter
# RobotSegmenter.from_robot_file does not forward ops_dtype to __init__, so we
# need to build the underlying Kinematics ourselves to override the default
# (bfloat16) which mismatches the float32 robot_spheres tensor at runtime.
from curobo._src.robot.kinematics.kinematics import Kinematics
from curobo._src.types.robot import RobotCfg
from curobo._src.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import CameraObservation, Pose
from curobo.types import JointState as CuRoboJointState, GoalToolPose
from curobo._src.geom.types import SceneCfg

from team_8.live_viz_helpers import (
    depth_to_xyz,
    resolve_urdf,
    resolve_urdf_string,
)

# ── constants ─────────────────────────────────────────────────────────────────
UR5_CONFIG   = '/ros2_ws/src/team_8/config/ur5_curobo.yml'
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
OVERHEAD_INFO_TOPIC  = '/camera/camera/depth/color/camera_info'
WRIST_DEPTH_TOPIC    = '/wrist_camera/wrist_camera/depth/color/image_raw'
WRIST_INFO_TOPIC     = '/wrist_camera/wrist_camera/depth/color/camera_info'
OVERHEAD_FRAME       = 'camera_color_optical_frame'
WRIST_FRAME          = 'wrist_camera_color_optical_frame'
WORLD_FRAME          = 'base_link'


# ── shared state ──────────────────────────────────────────────────────────────
@dataclass
class SharedState:
    """Thread-safe container for data shared between ROS2 callbacks and viser.

    All tensor data is converted to numpy by the producer (ROS callback /
    replan thread) before being placed here, so the viser update_loop on
    the main thread never has to call .cpu() — keeping CUDA ops confined
    to a single thread, which is required because Mapper.compute_esdf uses
    CUDA graph capture and a concurrent .cpu() in another thread would
    invalidate the capture (cudaErrorStreamCaptureUnsupported).
    """
    lock:            threading.Lock                    = field(default_factory=threading.Lock)
    point_clouds:    Dict[str, Optional[np.ndarray]]   = field(default_factory=dict)
    tsdf_centers:    Optional[np.ndarray]              = None   # (M, 3) reconstructed surface voxel centres
    traj:            Optional[np.ndarray]              = None   # (T, J) float32
    frame_count:     int                               = 0
    latest_joints:   Optional[object]                  = None   # sensor_msgs/JointState
    robot_desc_path: Optional[Path]                    = None   # resolved full URDF (UR5+gripper)


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
            # decay < 1 makes the TSDF weight on existing voxels fade each
            # frame, so stale single-view sensor noise doesn't accumulate
            # forever. 0.95 ≈ half-life of ~14 frames; tune lower if bands
            # still grow, higher if real obstacles flicker.
            decay_factor=0.95,
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
        # cuRobo's TSDF integrator unconditionally calls rgb_image.reshape(),
        # so CameraObservation needs an rgb_image even for depth-only mapping.
        self._dummy_rgb = torch.zeros(
            (2, 480, 640, 3), dtype=torch.uint8, device='cuda')

        # Mask the robot's own body out of depth before integrating, so the
        # ESDF never marks the arm itself as an obstacle. The segmenter
        # projects the robot's collision spheres into the camera using the
        # current joint state. Build manually so we can force
        # ops_dtype=float32 — the from_robot_file factory leaves it at the
        # bfloat16 default, which mismatches the float32 robot_spheres
        # tensor and crashes the segmenter at runtime.
        robot_yaml = load_yaml(join_path(get_robot_configs_path(), UR5_CONFIG))
        robot_cfg = RobotCfg.create(robot_yaml)
        self._segmenter = RobotSegmenter(
            Kinematics(robot_cfg.kinematics),
            distance_threshold=0.05,
            use_cuda_graph=False,
            ops_dtype=torch.float32,
        )

        self._plan_thread: Optional[threading.Thread] = None

        # Gazebo realsense plugin publishes camera streams with BEST_EFFORT
        # reliability; subscribers must match or no data arrives.
        self.create_subscription(
            CameraInfo, OVERHEAD_INFO_TOPIC,
            lambda m: self._on_info(m, 'overhead'), qos_profile_sensor_data)
        self.create_subscription(
            Image, OVERHEAD_DEPTH_TOPIC,
            lambda m: self._on_depth(m, 'overhead', OVERHEAD_FRAME),
            qos_profile_sensor_data)
        self.create_subscription(
            CameraInfo, WRIST_INFO_TOPIC,
            lambda m: self._on_info(m, 'wrist'), qos_profile_sensor_data)
        self.create_subscription(
            Image, WRIST_DEPTH_TOPIC,
            lambda m: self._on_depth(m, 'wrist', WRIST_FRAME),
            qos_profile_sensor_data)
        self.create_subscription(
            JointState, '/joint_states', self._on_joints, 10)

        # Subscribe to /robot_description to get the full URDF (UR5 + Robotiq
        # gripper). TRANSIENT_LOCAL so the latched message arrives even though
        # it was published before we subscribed.
        _robot_desc_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
        )
        self.create_subscription(
            String, '/robot_description', self._on_robot_description,
            _robot_desc_qos)

        self.get_logger().info('LiveVizNode ready — waiting for depth frames.')

    # ── callbacks ─────────────────────────────────────────────────────────────

    def _on_joints(self, msg: JointState) -> None:
        with self._state.lock:
            self._state.latest_joints = msg

    def _on_robot_description(self, msg: String) -> None:
        try:
            path = resolve_urdf_string(msg.data)
            with self._state.lock:
                self._state.robot_desc_path = path
            self.get_logger().info(f'robot_description resolved → {path}')
        except Exception as exc:
            self.get_logger().warning(
                f'Failed to resolve /robot_description: {exc}')

    def _on_info(self, msg: CameraInfo, cam_id: str) -> None:
        K = torch.tensor([
            [msg.k[0], 0.0,      msg.k[2]],
            [0.0,      msg.k[4], msg.k[5]],
            [0.0,      0.0,      1.0     ],
        ], dtype=torch.float32, device='cuda')
        self._cam_intrinsics[cam_id] = K

    def _on_depth(self, msg: Image, cam_id: str, frame: str) -> None:
        self.get_logger().info(
            f'[depth] {cam_id} arrived (intrinsics={cam_id in self._cam_intrinsics})',
            throttle_duration_sec=2.0)
        if cam_id not in self._cam_intrinsics:
            return   # wait for CameraInfo first

        K = self._cam_intrinsics[cam_id]

        # TF lookup: base_link ← camera_optical_frame (latest available transform)
        try:
            transform = self._tf_buffer.lookup_transform(
                WORLD_FRAME, frame, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1))
        except Exception as exc:
            self.get_logger().warning(
                f'TF lookup failed for {frame}: {exc}',
                throttle_duration_sec=2.0)
            return

        # Decode depth — handle both uint16 (mm) and float32 (m) encodings
        cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        if msg.encoding in ('32FC1', '32FC3'):
            depth = torch.from_numpy(cv_img.astype(np.float32)).cuda()
        else:   # default: 16UC1 (uint16, millimetres)
            depth = torch.from_numpy(cv_img.astype(np.float32) / 1000.0).cuda()
        depth  = torch.nan_to_num(depth, nan=0.0)
        filtered, _ = self._depth_filter(depth.unsqueeze(0))
        depth  = filtered[0]
        # Snapshot the pre-segmenter depth so the viser point cloud can show
        # the camera POV including the robot/gripper. The downstream `depth`
        # variable gets self-masked for ESDF integration; without this copy
        # the live cloud would also lose those pixels.
        depth_for_viz = depth.clone()

        # Camera pose in base_link frame (w, x, y, z quaternion convention)
        t = transform.transform.translation
        r = transform.transform.rotation
        pose = Pose.from_numpy(
            np.array([t.x, t.y, t.z], dtype=np.float32),
            np.array([r.w, r.x, r.y, r.z], dtype=np.float32),
        )

        # Mask out pixels that hit the robot itself (otherwise the ESDF
        # marks the arm as an obstacle and the planner refuses every
        # config). Skip if /joint_states hasn't arrived yet.
        with self._state.lock:
            js = self._state.latest_joints
        if js is not None:
            by_name = dict(zip(js.name, js.position))
            ordered = [by_name[n] for n in JOINT_NAMES if n in by_name]
            if len(ordered) == len(JOINT_NAMES):
                cam_obs_single = CameraObservation(
                    rgb_image=self._dummy_rgb[:1],   # (1, H, W, 3)
                    depth_image=depth.unsqueeze(0),  # (1, H, W)
                    intrinsics=K.unsqueeze(0),       # (1, 3, 3)
                    pose=pose,
                    # depth is already in metres; override the mm-default.
                    depth_to_meter=1.0,
                )
                seg_js = CuRoboJointState.from_position(
                    torch.tensor([ordered], dtype=torch.float32, device='cuda'),
                    joint_names=JOINT_NAMES)
                try:
                    _, depth_masked = self._segmenter.get_robot_mask_from_active_js(
                        cam_obs_single, seg_js)
                    depth = depth_masked[0]
                    # Flush segmenter ops before downstream Mapper kernels;
                    # otherwise their async work can poison a later CUDA
                    # graph capture in compute_esdf.
                    torch.cuda.synchronize()
                except Exception as exc:
                    self.get_logger().warning(
                        f'RobotSegmenter failed for {cam_id}: '
                        f'{type(exc).__name__}: {exc}',
                        throttle_duration_sec=5.0)

        # Unproject the *unmasked* depth to XYZ in camera frame, then transform
        # to base_link frame for viser display, so the camera POV shows the robot
        # too. (Mapper does its own transform internally using the pose we
        # pass, so its ESDF — built from the masked `depth` — is independent.)
        xyz_cam = depth_to_xyz(depth_for_viz, K)
        qw, qx, qy, qz = float(r.w), float(r.x), float(r.y), float(r.z)
        R = torch.tensor([
            [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
            [2*(qx*qy + qw*qz),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx),     1 - 2*(qx*qx + qy*qy)],
        ], dtype=torch.float32, device=xyz_cam.device)
        t_vec = torch.tensor(
            [t.x, t.y, t.z], dtype=torch.float32, device=xyz_cam.device)
        xyz_world = xyz_cam @ R.T + t_vec

        # Cache per-camera data
        self._cam_depth[cam_id] = depth
        self._cam_pose[cam_id]  = pose

        # Hand off to viser as numpy — keep CUDA ops out of update_loop.
        xyz_np = xyz_world.cpu().numpy()
        with self._state.lock:
            self._state.point_clouds[cam_id] = xyz_np

        # Integrate when both cameras have data
        if 'overhead' not in self._cam_depth or 'wrist' not in self._cam_depth:
            return

        batched = CameraObservation(
            rgb_image=self._dummy_rgb,
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
            # depth is already in metres; override the mm-default so the TSDF
            # integrator doesn't scale every depth by 0.001.
            depth_to_meter=1.0,
        )
        try:
            self._mapper.integrate(batched)
        except Exception as exc:
            self.get_logger().error(
                f'Mapper.integrate failed: {type(exc).__name__}: {exc}',
                throttle_duration_sec=2.0)
            return

        with self._state.lock:
            self._state.frame_count += 1
            count = self._state.frame_count

        if count >= MIN_FRAMES and count % REPLAN_EVERY == 0:
            self._replan()

    # ── planning ──────────────────────────────────────────────────────────────

    def _replan(self) -> None:
        """Compute a fresh ESDF synchronously, then plan in a background thread."""
        # ESDF update is fast — run on the callback thread so the mapper
        # doesn't race with integrate() from the same thread. Sync first so
        # any pending Warp/segmenter ops complete before compute_esdf opens
        # its CUDA graph capture; otherwise an error queued on a different
        # stream invalidates the capture (Warp CUDA error 901).
        torch.cuda.synchronize()
        try:
            voxel_grid = self._mapper.compute_esdf()
            self._planner.update_world(SceneCfg(voxel=[voxel_grid]))
        except Exception as exc:
            self.get_logger().error(
                f'compute_esdf/update_world failed: {type(exc).__name__}: {exc}',
                throttle_duration_sec=5.0)
            return

        # Extract the reconstructed TSDF surface voxels for viser display.
        # This is the cuRobo-demo-style "coloured voxel cubes" view of the
        # scene; we ignore the returned colors because we pass dummy RGB to
        # the integrator, and shade by height in the update_loop instead.
        try:
            centers, _ = self._mapper.integrator.extract_occupied_voxels(
                surface_only=True)
            tsdf_np = centers.cpu().numpy() if centers is not None else None
        except Exception as exc:
            self.get_logger().warning(
                f'extract_occupied_voxels failed: {type(exc).__name__}: {exc}',
                throttle_duration_sec=5.0)
            tsdf_np = None

        with self._state.lock:
            self._state.tsdf_centers = tsdf_np
            js = self._state.latest_joints

        # Slow GPU planning runs in a daemon thread so this callback returns fast.
        if self._plan_thread is not None and self._plan_thread.is_alive():
            return   # previous plan still running — skip this trigger

        self._plan_thread = threading.Thread(
            target=self._run_plan, args=(js,), daemon=True)
        self._plan_thread.start()

    def _run_plan(self, js) -> None:
        """Background thread: plan trajectory and store result in SharedState."""
        if js is not None:
            # /joint_states includes the gripper (7 joints); the planner
            # only knows the 6 UR5 joints. Extract those in the canonical
            # order; fall back to HOME_CFG for any missing.
            by_name = dict(zip(js.name, js.position))
            ordered = [by_name.get(n, HOME_CFG[i])
                       for i, n in enumerate(JOINT_NAMES)]
            start = CuRoboJointState.from_position(
                torch.tensor([ordered], dtype=torch.float32, device='cuda'),
                joint_names=JOINT_NAMES)
        else:
            start = CuRoboJointState.from_position(
                torch.tensor([HOME_CFG], dtype=torch.float32, device='cuda'),
                joint_names=JOINT_NAMES)

        # GoalToolPose expects a 5-D shape (B, n_goals, n_grasps, n_envs, dim).
        goal = GoalToolPose(
            tool_frames=self._planner.tool_frames,
            position=torch.tensor(
                GOAL_XYZ, device='cuda', dtype=torch.float32
            ).view(1, 1, 1, 1, 3),
            quaternion=torch.tensor(
                GOAL_QUAT, device='cuda', dtype=torch.float32
            ).view(1, 1, 1, 1, 4),
        )

        result = self._planner.plan_pose(goal, start)
        if result is None or not result.success.any():
            status = getattr(result, 'status', None)
            ik_succ = getattr(getattr(result, 'ik_result', None), 'success', None)
            self.get_logger().warning(
                f'CuRobo planning failed — status={status} '
                f'ik_success={ik_succ.any().item() if ik_succ is not None else None} '
                f'start_q={start.position.tolist() if hasattr(start, "position") else "?"}',
                throttle_duration_sec=5.0)
            return

        pos = result.get_interpolated_plan().position[0]
        while pos.dim() > 2:
            pos = pos[0]
        traj = pos.cpu().numpy()   # (T, J) float32
        self.get_logger().info(f'Planned {len(traj)}-waypoint trajectory.')

        with self._state.lock:
            self._state.traj = traj


# ── viser update loop ─────────────────────────────────────────────────────────

def update_loop(
    server: viser.ViserServer,
    state: SharedState,
    robot,   # ViserUrdf instance or None
) -> None:
    """Main-thread loop: refresh the viser scene at VIZ_HZ from shared state.

    Reads lock-protected SharedState and updates:
      - /depth/overhead, /depth/wrist   — live point clouds
      - /tsdf/voxels                    — reconstructed TSDF surface voxels
      - /robot joints                   — animated trajectory + live mirror
      - /status label                   — frame count and status text

    Loops forever; raise KeyboardInterrupt to exit.
    """
    traj_idx = 0
    period   = 1.0 / VIZ_HZ

    while True:
        t0 = time.time()

        with state.lock:
            clouds      = dict(state.point_clouds)   # shallow copy of dict
            tsdf_pts    = state.tsdf_centers
            traj        = state.traj                  # None if no plan yet
            n_frames    = state.frame_count
            latest_js   = state.latest_joints

        # ── status label ──────────────────────────────────────────────────────
        traj_len = 0 if traj is None else len(traj)
        if n_frames < MIN_FRAMES:
            status_text = f'Waiting for depth frames ({n_frames}/{MIN_FRAMES})…'
        else:
            status_text = (f'Frames: {n_frames}  |  '
                           f'Traj waypoints: {traj_len}  |  '
                           f'Mode: {"plan" if traj_len > 0 else "live"}')
        server.scene.add_label('/status', status_text, position=(0.0, 0.0, 1.6))

        # ── point clouds ──────────────────────────────────────────────────────
        for cam_id, pts in clouds.items():
            if pts is None or len(pts) == 0:
                continue
            color = (200, 200, 200) if cam_id == 'overhead' else (100, 150, 255)
            server.scene.add_point_cloud(
                f'/depth/{cam_id}',
                points=pts,
                colors=np.tile(color, (len(pts), 1)).astype(np.uint8),
                point_size=0.005,
            )

        # ── TSDF surface voxels ───────────────────────────────────────────────
        # cuRobo's reconstructed scene geometry. Shade by height for a viridis-
        # ish gradient since we don't integrate real RGB into the TSDF.
        if tsdf_pts is not None and len(tsdf_pts) > 0:
            z = tsdf_pts[:, 2]
            z_norm = np.clip((z - z.min()) / max(z.max() - z.min(), 1e-6), 0, 1)
            colors = np.stack([
                (255 * (1 - z_norm)).astype(np.uint8),     # red fades with height
                (255 * z_norm).astype(np.uint8),           # green grows with height
                np.full_like(z_norm, 120, dtype=np.uint8), # cyan tint
            ], axis=1)
            server.scene.add_point_cloud(
                '/tsdf/voxels',
                points=tsdf_pts,
                colors=colors,
                point_size=0.02,
            )

        # ── robot pose ────────────────────────────────────────────────────────
        # With a successful plan: animate through traj waypoints.
        # Otherwise: mirror the live /joint_states so the URDF reflects reality.
        if robot is not None:
            if traj is not None and len(traj) > 0:
                waypoint = traj[traj_idx % len(traj)]
                cfg = dict(zip(JOINT_NAMES, waypoint.tolist()))
                # Also mirror live gripper joints (not in the plan) so fingers
                # render at their actual position rather than the URDF default.
                if latest_js is not None:
                    by_name = dict(zip(latest_js.name, latest_js.position))
                    cfg.update({n: v for n, v in by_name.items()
                                if n not in cfg})
                robot.update_cfg(cfg)
                traj_idx += 1
                if traj_idx >= len(traj):
                    traj_idx = 0
                    time.sleep(1.0)   # brief pause before replaying
            elif latest_js is not None:
                # Mirror all live joints — ViserUrdf silently ignores any
                # joints that don't exist in the loaded URDF.
                by_name = dict(zip(latest_js.name, latest_js.position))
                if by_name:
                    robot.update_cfg(by_name)

        elapsed = time.time() - t0
        time.sleep(max(0.0, period - elapsed))


# ── motion planner setup ──────────────────────────────────────────────────────

def build_planner() -> MotionPlanner:
    """Construct and warm up the cuRobo MotionPlanner (~30 s on first run)."""
    print('  Loading MotionPlanner (warmup ~30 s)…')
    # Pre-allocate a voxel cache so update_world() can accept the ESDF
    # produced by Mapper.compute_esdf(). Observed empirically: cuRobo
    # allocates a 128**3 = 2,097,152-voxel tensor for the ESDF regardless
    # of MapperCfg.extent_meters_xyz, so the cache buffer must be at least
    # that many slots. Size to 140**3 = 2,744,000 for headroom.
    collision_cache = {
        'voxel': {
            'layers': 1,
            'dims': [7.0, 7.0, 7.0],
            'voxel_size': 0.05,
        }
    }
    planner = MotionPlanner(MotionPlannerCfg.create(
        robot=UR5_CONFIG,
        scene_model='collision_test.yml',
        collision_cache=collision_cache,
    ))
    planner.warmup(enable_graph=True, num_warmup_iterations=3)
    print('  MotionPlanner ready.')
    return planner


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print('=== Live Pipeline Visualization Test ===\n')

    print('[1/4] Initializing CUDA / Warp…')
    import warp as wp
    wp.init()
    print('  Warp OK.\n')

    print('[2/4] Setting up motion planner…')
    planner = build_planner()

    print('[3/4] Starting ROS2 node…')
    rclpy.init()
    state = SharedState()
    node  = LiveVizNode(state, planner)
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    print('  ROS2 node spinning in background.\n')

    print('[4/4] Starting viser…')
    server = viser.ViserServer(port=8080, verbose=False)
    print('  Viser running — open http://localhost:8080\n')

    server.scene.add_frame('/base_link', axes_length=0.3, axes_radius=0.01)
    server.scene.add_icosphere(
        '/target', radius=0.03, color=(255, 80, 80), position=GOAL_XYZ)

    # Wait up to 5 s for /robot_description (TRANSIENT_LOCAL — should arrive
    # within the first spin_once). Fall back to the baked UR5-only URDF.
    print('  Waiting for /robot_description (up to 5 s)…')
    for _ in range(50):
        with state.lock:
            robot_desc_path = state.robot_desc_path
        if robot_desc_path is not None:
            break
        time.sleep(0.1)
    if robot_desc_path is not None:
        print('  Full robot URDF received (UR5 + gripper).')
        urdf_source = robot_desc_path
    else:
        print('  /robot_description not received; falling back to UR5-only URDF.')
        urdf_source = resolve_urdf(URDF_PATH)

    try:
        from viser.extras import ViserUrdf
        robot = ViserUrdf(
            server,
            urdf_or_path=urdf_source,
            root_node_name='/robot',
        )
        print('  Robot model loaded.')
    except Exception as exc:
        print(f'  Robot model unavailable ({exc}), skipping URDF.')
        robot = None

    print('Entering update loop. Ctrl+C to stop.')
    try:
        update_loop(server, state, robot)
    except KeyboardInterrupt:
        print('\nStopped.')
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
