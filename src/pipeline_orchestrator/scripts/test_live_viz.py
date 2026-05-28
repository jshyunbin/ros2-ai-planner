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
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
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
OVERHEAD_INFO_TOPIC  = '/camera/camera/depth/color/camera_info'
WRIST_DEPTH_TOPIC    = '/wrist_camera/wrist_camera/depth/color/image_raw'
WRIST_INFO_TOPIC     = '/wrist_camera/wrist_camera/depth/color/camera_info'
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
        self.get_logger().info(
            f'[depth] {cam_id} arrived (intrinsics={cam_id in self._cam_intrinsics})',
            throttle_duration_sec=2.0)
        if cam_id not in self._cam_intrinsics:
            return   # wait for CameraInfo first

        K = self._cam_intrinsics[cam_id]

        # TF lookup: world ← camera_optical_frame (latest available transform)
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

        # Camera pose in world frame (w, x, y, z quaternion convention)
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

        # Unproject depth to XYZ in camera frame, then transform to world
        # frame for viser display. (Mapper does its own transform internally
        # using the pose we pass, so its ESDF is independent of this.)
        xyz_cam = depth_to_xyz(depth, K)
        qw, qx, qy, qz = float(r.w), float(r.x), float(r.y), float(r.z)
        R = torch.tensor([
            [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
            [2*(qx*qy + qw*qz),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx),     1 - 2*(qx*qx + qy*qy)],
        ], dtype=torch.float32, device=xyz_cam.device)
        t_vec = torch.tensor(
            [t.x, t.y, t.z], dtype=torch.float32, device=xyz_cam.device)
        xyz = xyz_cam @ R.T + t_vec

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

        with self._state.lock:
            self._state.voxel_grid = voxel_grid
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
      - /esdf/voxels                    — occupied ESDF voxels
      - /ur5 robot joints               — animated trajectory
      - /status label                   — frame count and status text

    Loops forever; raise KeyboardInterrupt to exit.
    """
    traj_idx = 0
    period   = 1.0 / VIZ_HZ

    while True:
        t0 = time.time()

        with state.lock:
            clouds      = dict(state.point_clouds)   # shallow copy of dict
            vg          = state.voxel_grid
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
        for cam_id, xyz in clouds.items():
            if xyz is None or len(xyz) == 0:
                continue
            pts   = xyz.cpu().numpy()
            color = (200, 200, 200) if cam_id == 'overhead' else (100, 150, 255)
            server.scene.add_point_cloud(
                f'/depth/{cam_id}',
                points=pts,
                colors=np.tile(color, (len(pts), 1)).astype(np.uint8),
                point_size=0.005,
            )

        # ── ESDF voxels ───────────────────────────────────────────────────────
        if vg is not None:
            occ_pts = esdf_to_points(vg)
            if len(occ_pts) > 0:
                server.scene.add_point_cloud(
                    '/esdf/voxels',
                    points=occ_pts,
                    colors=np.tile((220, 60, 60), (len(occ_pts), 1)).astype(np.uint8),
                    point_size=0.02,
                )

        # ── robot pose ────────────────────────────────────────────────────────
        # With a successful plan: animate through traj waypoints.
        # Otherwise: mirror the live /joint_states so the URDF reflects reality.
        if robot is not None:
            if traj is not None and len(traj) > 0:
                waypoint = traj[traj_idx % len(traj)]
                robot.update_cfg(dict(zip(JOINT_NAMES, waypoint.tolist())))
                traj_idx += 1
                if traj_idx >= len(traj):
                    traj_idx = 0
                    time.sleep(1.0)   # brief pause before replaying
            elif latest_js is not None:
                # Index /joint_states by name; only update joints we know.
                by_name = dict(zip(latest_js.name, latest_js.position))
                cfg = {n: by_name[n] for n in JOINT_NAMES if n in by_name}
                if cfg:
                    robot.update_cfg(cfg)

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

    server.scene.add_frame('/world', axes_length=0.3, axes_radius=0.01)
    server.scene.add_icosphere(
        '/target', radius=0.03, color=(255, 80, 80), position=GOAL_XYZ)

    try:
        from viser.extras import ViserUrdf
        robot = ViserUrdf(
            server,
            urdf_or_path=resolve_urdf(URDF_PATH),
            root_node_name='/ur5',
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
