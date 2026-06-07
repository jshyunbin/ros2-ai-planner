"""cuRobo motion planning with live dual-RGBD TSDF occupancy.

This keeps the original planner's Mapper/RobotSegmenter occupancy path, but
fixes its frame contract: all camera observations, voxel worlds, GraspGen TCP
poses, and tool goals are expressed in ``base_link``.  The trajectory planner
uses Yeina's polished pick logic: per-candidate / per-approach retry with
``plan_grasp`` phases (approach, grasp, lift).
"""

import os
import threading
import time

import numpy as np
import rclpy.duration
import torch
from builtin_interfaces.msg import Duration as RosDuration
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from curobo._src.geom.types import SceneCfg
from curobo._src.robot.kinematics.kinematics import Kinematics
from curobo._src.types.robot import RobotCfg
from curobo._src.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.perception import FilterDepth, Mapper, MapperCfg, RobotSegmenter
from curobo.types import CameraObservation, GoalToolPose
from curobo.types import JointState as CuRoboJointState
from curobo.types import Pose as CuRoboPose

from pipeline_orchestrator.pipeline_utils import (
    ROBOTIQ_2F_85_TCP_Z_OFFSET,
    env_bool as _env_bool,
    env_float as _env_float,
    env_int as _env_int,
)

OVERHEAD_DEPTH_TOPIC = '/camera/camera/depth/color/image_raw'
OVERHEAD_INFO_TOPIC = '/camera/camera/depth/color/camera_info'
WRIST_DEPTH_TOPIC = '/wrist_camera/wrist_camera/depth/color/image_raw'
WRIST_INFO_TOPIC = '/wrist_camera/wrist_camera/depth/color/camera_info'
OVERHEAD_FRAME = 'camera_color_optical_frame'
WRIST_FRAME = 'wrist_camera_color_optical_frame'
BASE_FRAME = 'base_link'
MIN_FRAMES = 5

def _ur5_config_path() -> str:
    """Resolve ur5_curobo.yml via ament_index, falling back to the Docker path."""
    try:
        from ament_index_python.packages import get_package_share_directory
        share = get_package_share_directory('pipeline_orchestrator')
        return os.path.join(share, 'config', 'ur5_curobo.yml')
    except Exception:
        return '/ros2_ws/src/pipeline_orchestrator/config/ur5_curobo.yml'

UR5_CONFIG = _ur5_config_path()
JOINT_NAMES = (
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
)

TOPK_GRASPS = 10
INTERP_DT = 0.02
GRIPPER_TCP_Z_OFFSET = ROBOTIQ_2F_85_TCP_Z_OFFSET  # imported from pipeline_utils

# Robotiq 2F-85: distance from tool0 origin to fingertip contact at full open.
_FINGERTIP_LEN = 0.136   # m
# Minimum clearance between fingertip and floor surface before a grasp is rejected.
# Keep small — cuRobo's floor cuboid handles motion-level avoidance.
_FLOOR_CLEARANCE_M = 0.010  # m — 1 cm: rejects only sub-table grasps


class CuRobo:
    """Dual-RGBD TSDF fusion plus Yeina-style UR5 trajectory planning."""

    def __init__(self, node: Node, enable_viz: bool = False):
        self._node = node
        self._logger = node.get_logger()
        self._lock = threading.Lock()
        self._cuda_lock = threading.RLock()
        self._frame_count = 0
        self._last_world_update_frame = -1
        self._mapping_paused_until = 0.0

        self._enable_viz = enable_viz
        self._point_clouds: dict = {}
        self._tsdf_centers = None

        self._cam_depth: dict = {}
        self._cam_intrinsics: dict = {}
        self._cam_pose: dict = {}
        self._latest_joints = None

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, node)
        self._bridge = CvBridge()

        self._mapper = Mapper(MapperCfg(
            extent_meters_xyz=(2.0, 2.0, 1.5),
            voxel_size=0.02,
            esdf_voxel_size=0.05,
            truncation_distance=0.1,
            depth_minimum_distance=0.15,
            depth_maximum_distance=2.0,
            decay_factor=0.95,
            frustum_decay_factor=1.0,
            enable_static=False,
            num_cameras=2,
        ))
        self._configure_mapper_cuda_graphs()
        self._depth_filter = FilterDepth(
            image_shape=(480, 640),
            depth_minimum_distance=0.15,
            depth_maximum_distance=2.0,
            flying_pixel_threshold=0.5,
            bilateral_kernel_size=3,
        )
        self._dummy_rgb = torch.zeros(
            (2, 480, 640, 3), dtype=torch.uint8, device='cuda')

        robot_yaml = load_yaml(join_path(get_robot_configs_path(), UR5_CONFIG))
        robot_cfg = RobotCfg.create(robot_yaml)
        self._segmenter = RobotSegmenter(
            Kinematics(robot_cfg.kinematics),
            distance_threshold=0.05,
            use_cuda_graph=False,
            ops_dtype=torch.float32,
        )

        self._planner = self._build_planner()
        self._static_cuboids = self._build_static_cuboids()
        self._subscriptions = []
        self._subscribe_depth_streams(node)
        self._logger.info(
            f'CuRobo: ready (dual-camera TSDF frame={BASE_FRAME}).')

    def _subscribe_depth_streams(self, node: Node) -> None:
        self._subscriptions.extend([
            node.create_subscription(
                Image,
                OVERHEAD_DEPTH_TOPIC,
                lambda msg: self._on_depth(msg, 'overhead', OVERHEAD_FRAME),
                qos_profile_sensor_data,
            ),
            node.create_subscription(
                CameraInfo,
                OVERHEAD_INFO_TOPIC,
                lambda msg: self._on_info(msg, 'overhead'),
                qos_profile_sensor_data,
            ),
            node.create_subscription(
                Image,
                WRIST_DEPTH_TOPIC,
                lambda msg: self._on_depth(msg, 'wrist', WRIST_FRAME),
                qos_profile_sensor_data,
            ),
            node.create_subscription(
                CameraInfo,
                WRIST_INFO_TOPIC,
                lambda msg: self._on_info(msg, 'wrist'),
                qos_profile_sensor_data,
            ),
        ])

    def update_joint_state(self, msg):
        with self._lock:
            self._latest_joints = msg

    def pause_mapping(self, seconds: float) -> None:
        if seconds <= 0.0:
            return
        until = time.monotonic() + float(seconds)
        with self._lock:
            self._mapping_paused_until = max(self._mapping_paused_until, until)

    def reset_mapping(self) -> None:
        with self._cuda_lock:
            self._mapper.reset()
            torch.cuda.synchronize()
        with self._lock:
            self._frame_count = 0
            self._last_world_update_frame = -1
            self._cam_depth.clear()
            self._cam_pose.clear()
            self._tsdf_centers = None

    @property
    def frame_count(self) -> int:
        with self._lock:
            return self._frame_count

    def get_point_clouds(self) -> dict:
        with self._lock:
            return dict(self._point_clouds)

    def get_tsdf_centers(self):
        with self._lock:
            return self._tsdf_centers

    def get_latest_joints(self):
        with self._lock:
            return self._latest_joints

    def _cache_viz_cloud(self, cam_id, depth, t, r, K):
        from pipeline_orchestrator.live_viz_helpers import depth_to_xyz
        try:
            xyz_cam = depth_to_xyz(depth, K)
            qw, qx, qy, qz = float(r.w), float(r.x), float(r.y), float(r.z)
            rot = torch.tensor([
                [1 - 2 * (qy * qy + qz * qz),
                 2 * (qx * qy - qw * qz),
                 2 * (qx * qz + qw * qy)],
                [2 * (qx * qy + qw * qz),
                 1 - 2 * (qx * qx + qz * qz),
                 2 * (qy * qz - qw * qx)],
                [2 * (qx * qz - qw * qy),
                 2 * (qy * qz + qw * qx),
                 1 - 2 * (qx * qx + qy * qy)],
            ], dtype=torch.float32, device=xyz_cam.device)
            t_vec = torch.tensor(
                [t.x, t.y, t.z], dtype=torch.float32, device=xyz_cam.device)
            xyz_base = (xyz_cam @ rot.T + t_vec).cpu().numpy()
        except Exception as exc:
            self._logger.warning(
                f'CuRobo: viz cloud failed for {cam_id}: '
                f'{type(exc).__name__}: {exc}')
            return
        with self._lock:
            self._point_clouds[cam_id] = xyz_base

    def _cache_viz_tsdf(self):
        try:
            centers, _ = self._mapper.integrator.extract_occupied_voxels(
                surface_only=True)
            tsdf_np = centers.cpu().numpy() if centers is not None else None
        except Exception as exc:
            self._logger.warning(
                f'CuRobo: extract_occupied_voxels failed: '
                f'{type(exc).__name__}: {exc}')
            return
        with self._lock:
            self._tsdf_centers = tsdf_np

    def _on_info(self, msg, cam_id: str):
        with self._cuda_lock:
            K = torch.tensor([
                [msg.k[0], 0.0, msg.k[2]],
                [0.0, msg.k[4], msg.k[5]],
                [0.0, 0.0, 1.0],
            ], dtype=torch.float32, device='cuda')
        with self._lock:
            self._cam_intrinsics[cam_id] = K

    def _on_depth(self, msg, cam_id: str, frame: str):
        with self._lock:
            if time.monotonic() < self._mapping_paused_until:
                return
            if cam_id not in self._cam_intrinsics:
                return
            K = self._cam_intrinsics[cam_id]

        try:
            transform = self._tf_buffer.lookup_transform(
                BASE_FRAME,
                frame,
                Time(),
                timeout=rclpy.duration.Duration(seconds=0.1),
            )
        except Exception as exc:
            self._logger.warning(
                f'CuRobo: TF lookup failed {BASE_FRAME} <- {frame}: {exc}',
                throttle_duration_sec=2.0)
            return

        try:
            with self._cuda_lock:
                cv_img = self._bridge.imgmsg_to_cv2(
                    msg, desired_encoding='passthrough')
                if msg.encoding.lower() in ('16uc1', 'mono16'):
                    depth_np = cv_img.astype(np.float32) / 1000.0
                else:
                    depth_np = cv_img.astype(np.float32)
                depth = torch.from_numpy(depth_np).cuda()
                depth = torch.nan_to_num(depth, nan=0.0)
                filtered, _ = self._depth_filter(depth.unsqueeze(0))
                depth = filtered[0]

                t = transform.transform.translation
                r = transform.transform.rotation
                pose = CuRoboPose.from_numpy(
                    np.array([t.x, t.y, t.z], dtype=np.float32),
                    np.array([r.w, r.x, r.y, r.z], dtype=np.float32),
                )

                if self._enable_viz:
                    self._cache_viz_cloud(cam_id, depth, t, r, K)

                depth = self._mask_robot_from_depth(depth, K, pose, cam_id)

                ready = False
                batched = None
                with self._lock:
                    self._cam_depth[cam_id] = depth
                    self._cam_pose[cam_id] = pose
                    self._cam_intrinsics[cam_id] = K
                    ready = (
                        'overhead' in self._cam_depth
                        and 'wrist' in self._cam_depth
                    )
                    if ready:
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
                            pose=CuRoboPose(
                                position=torch.cat([
                                    self._cam_pose['overhead'].position,
                                    self._cam_pose['wrist'].position,
                                ]),
                                quaternion=torch.cat([
                                    self._cam_pose['overhead'].quaternion,
                                    self._cam_pose['wrist'].quaternion,
                                ]),
                            ),
                            depth_to_meter=1.0,
                        )

                if ready and batched is not None:
                    self._mapper.integrate(batched)
                    torch.cuda.synchronize()
                    with self._lock:
                        self._frame_count += 1
        except Exception as exc:
            self._logger.error(
                f'CuRobo: depth integration failed for {cam_id}: '
                f'{type(exc).__name__}: {exc}')

    def _mask_robot_from_depth(self, depth, K, pose, cam_id: str):
        with self._lock:
            js = self._latest_joints
        if js is None:
            return depth

        by_name = dict(zip(js.name, js.position))
        ordered = [by_name[n] for n in JOINT_NAMES if n in by_name]
        if len(ordered) != len(JOINT_NAMES):
            return depth

        cam_obs_single = CameraObservation(
            rgb_image=self._dummy_rgb[:1],
            depth_image=depth.unsqueeze(0),
            intrinsics=K.unsqueeze(0),
            pose=pose,
            depth_to_meter=1.0,
        )
        seg_js = CuRoboJointState.from_position(
            torch.tensor([ordered], dtype=torch.float32, device='cuda'),
            joint_names=list(JOINT_NAMES),
        )
        try:
            _, depth_masked = self._segmenter.get_robot_mask_from_active_js(
                cam_obs_single, seg_js)
            torch.cuda.synchronize()
            return depth_masked[0]
        except Exception as exc:
            self._logger.warning(
                f'CuRobo: RobotSegmenter failed for {cam_id}: '
                f'{type(exc).__name__}: {exc}')
            return depth

    def plan_pick(self, grasp_candidates, joint_states, object_cloud=None):
        """Plan approach -> grasp -> lift for ranked GraspGen TCP poses.

        Args:
            grasp_candidates: Ranked grasp pose dicts from GraspGenX.
            joint_states: Current /joint_states message.
            object_cloud: (N, 3) float32 ndarray of the segmented object in
                base_link frame. When provided, the object's voxels are carved
                out of the TSDF collision world before planning so cuRobo does
                not treat the target object itself as an obstacle.
        """
        with self._cuda_lock:
            return self._plan_pick_locked(grasp_candidates, joint_states, object_cloud)

    def _plan_pick_locked(self, grasp_candidates, joint_states, object_cloud=None):
        """Plan pick using plan_trajectory for each candidate (approach + descent + lift).

        Strategy per candidate:
          1. plan_trajectory → approach pose (approach_offset above grasp in tool-z)
          2. plan_trajectory → grasp pose (short descent; floor cuboid prevents going too low)
          3. plan_trajectory → lift pose (lift_offset above grasp in world-z)

        Retries over multiple approach offsets and falls back to relaxed collision world.
        Results are wrapped in _TrajWrapper so curobo_service._handle_pick's
        _unwrap_traj → get_ros_traj path is used (bypasses interp_traj_to_ros).
        """
        import types

        try:
            self.update_joint_state(joint_states)
            self._update_world_from_tsdf(object_cloud=object_cloud)
            current = self._ros_js_to_curobo(joint_states)

            # Floor-clearance pre-filter: reject candidates where the fingertip
            # would descend into the table. cuRobo collision spheres leave the
            # fingertip region uncovered intentionally; this guard is the safety net.
            floor_z = _env_float('PIPELINE_FLOOR_Z', -0.07)
            min_tool0_z = floor_z + _FINGERTIP_LEN + _FLOOR_CLEARANCE_M
            candidates = []
            for i, candidate in enumerate(grasp_candidates):
                grasp_4x4 = _candidate_pose_4x4(candidate)
                t_tool_grasp = np.eye(4, dtype=np.float32)
                t_tool_grasp[2, 3] = _effective_gripper_tcp_z_offset()
                tool_4x4 = grasp_4x4 @ np.linalg.inv(t_tool_grasp)
                tool0_z = float(tool_4x4[2, 3])
                if tool0_z < min_tool0_z:
                    self._logger.warn(
                        f'plan_pick: skipping candidate {i} — '
                        f'tool0_z={tool0_z:.3f} < floor_min {min_tool0_z:.3f}')
                else:
                    candidates.append((i, candidate, tool_4x4))

            if not candidates:
                self._logger.warn('plan_pick: no candidates passed floor clearance filter')
                return None

            lift_offset = _env_float('PIPELINE_CUROBO_PICK_LIFT_OFFSET', 0.12)
            approach_offsets = _pick_approach_offsets()

            for world_mode in self._pick_world_modes():
                if world_mode == 'relaxed':
                    self._clear_collision_world()
                    self._logger.warn('plan_pick: retrying with relaxed collision world')

                for approach_offset in approach_offsets:
                    for orig_i, candidate, tool_4x4 in candidates:
                        result = self._plan_approach_descent_lift(
                            orig_i, tool_4x4, current, approach_offset,
                            lift_offset, world_mode,
                        )
                        if result is not None:
                            return result

            self._logger.warn('plan_pick: all attempts exhausted')
            return None
        except Exception as exc:
            self._logger.error(f'CuRobo.plan_pick error: {exc}')
            return None

    def _plan_approach_descent_lift(
        self, candidate_index, tool_4x4, current, approach_offset,
        lift_offset, world_mode,
    ):
        """Plan approach → descent → lift using three plan_trajectory calls.

        Returns a SimpleNamespace with _TrajWrapper fields matching the interface
        curobo_service._handle_pick expects:
          .approach_interpolated_trajectory  — home → approach pose
          .grasp_interpolated_trajectory     — approach → grasp pose (descent)
          .lift_interpolated_trajectory      — grasp → lift pose
          .success
        """
        import types

        z_hat = tool_4x4[:3, 2]  # tool0 z-axis in world frame (points toward object)

        # Approach pose: back off along tool-z by |approach_offset|.
        # approach_offset is negative (e.g. -0.10); z_hat points down for top grasps,
        # so approach = grasp - |offset| * z_hat_down = grasp + offset_above.
        approach_4x4 = tool_4x4.copy()
        approach_4x4[:3, 3] = tool_4x4[:3, 3] + approach_offset * z_hat
        approach_pose = self._pose_from_4x4(approach_4x4)

        _reset_planner_seed(self._planner)
        approach_traj = self._plan_trajectory_locked(approach_pose, None, current=current)
        if approach_traj is None or not approach_traj.points:
            self._logger.warn(
                f'plan_pick: approach failed '
                f'world={world_mode} candidate={candidate_index} '
                f'approach={approach_offset:.3f}m')
            return None

        # Descent: from approach end to grasp pose, with floor clearance clamp.
        # Clamp tool0_z so the fingertip never physically contacts the table.
        # cuRobo collision spheres intentionally don't cover the fingertip region
        # (z=0.095-0.136 from tool0), so this explicit clamp is the safety net.
        floor_z = _env_float('PIPELINE_FLOOR_Z', -0.07)
        descent_margin = _env_float('PIPELINE_CUROBO_DESCENT_MARGIN', 0.025)
        min_descent_z = floor_z + _FINGERTIP_LEN + descent_margin
        descent_4x4 = tool_4x4.copy()
        original_z = float(tool_4x4[2, 3])
        clamped_z = max(original_z, min_descent_z)
        if clamped_z > original_z:
            self._logger.info(
                f'plan_pick: descent z clamped {original_z:.3f} → {clamped_z:.3f}m '
                f'(floor={floor_z:.3f} + tip={_FINGERTIP_LEN:.3f} + '
                f'margin={descent_margin:.3f})')
        descent_4x4[2, 3] = clamped_z
        grasp_pose = self._pose_from_4x4(descent_4x4)
        approach_end = self._ros_js_to_curobo(self._traj_end_joint_state(approach_traj))
        _reset_planner_seed(self._planner)
        descent_traj = self._plan_trajectory_locked(grasp_pose, None, current=approach_end)
        if descent_traj is None or not descent_traj.points:
            self._logger.warn(
                f'plan_pick: descent failed — closing at approach pose '
                f'world={world_mode} candidate={candidate_index}')
            # Fallback: close gripper at approach position (no descent)
            descent_traj = approach_traj

        # Lift pose: grasp xyz + lift_offset in world-z, same orientation.
        lift_xyz = tool_4x4[:3, 3].copy()
        lift_xyz[2] += lift_offset
        quat_xyzw = R.from_matrix(tool_4x4[:3, :3]).as_quat()
        lift_pose = [
            float(lift_xyz[0]), float(lift_xyz[1]), float(lift_xyz[2]),
            float(quat_xyzw[3]), float(quat_xyzw[0]),
            float(quat_xyzw[1]), float(quat_xyzw[2]),
        ]
        grasp_end = self._ros_js_to_curobo(self._traj_end_joint_state(descent_traj))
        _reset_planner_seed(self._planner)
        lift_traj = self._plan_trajectory_locked(lift_pose, None, current=grasp_end)
        if lift_traj is None or not lift_traj.points:
            self._logger.warn(
                f'plan_pick: lift failed '
                f'world={world_mode} candidate={candidate_index} '
                f'approach={approach_offset:.3f}m lift={lift_offset:.3f}m')
            return None

        torch.cuda.synchronize()
        self._logger.info(
            f'plan_pick: succeeded '
            f'world={world_mode} candidate={candidate_index} '
            f'approach={approach_offset:.3f}m lift={lift_offset:.3f}m')

        result = types.SimpleNamespace()
        result.success = True
        result.approach_interpolated_trajectory = _TrajWrapper(approach_traj)
        result.approach_interpolated_last_tstep = None
        result.grasp_interpolated_trajectory = _TrajWrapper(descent_traj)
        result.grasp_interpolated_last_tstep = None
        result.lift_interpolated_trajectory = _TrajWrapper(lift_traj)
        result.lift_interpolated_last_tstep = None
        return result

    def _pose_from_4x4(self, mat: np.ndarray):
        """Convert 4×4 pose matrix to (x,y,z,qw,qx,qy,qz) tuple."""
        xyz = mat[:3, 3].tolist()
        quat_xyzw = R.from_matrix(mat[:3, :3]).as_quat()
        return [xyz[0], xyz[1], xyz[2],
                float(quat_xyzw[3]), float(quat_xyzw[0]),
                float(quat_xyzw[1]), float(quat_xyzw[2])]

    def _traj_end_joint_state(self, traj: 'JointTrajectory'):
        """Build a minimal JointState from the last point of a JointTrajectory."""
        from sensor_msgs.msg import JointState as RosJointState
        js = RosJointState()
        js.name = list(traj.joint_names)
        js.position = list(traj.points[-1].positions)
        return js

    def plan_trajectory(self, goal_pose, joint_states):
        """Plan a single tool0 trajectory for place/home style targets."""
        try:
            with self._cuda_lock:
                return self._plan_trajectory_locked(goal_pose, joint_states)
        except Exception as exc:
            self._logger.error(f'CuRobo.plan_trajectory error: {exc}')
            return None

    def plan_place(self, waypoints: list, joint_states) -> 'JointTrajectory | None':
        """Collision-off safe-Z transit through waypoints to a place destination.

        The carried object is invisible to the TSDF collision world, so we
        disable collision checking entirely and follow the waypoints with a
        rule-based lift → traverse → descend path.

        Args:
            waypoints: List of geometry_msgs/Pose targets (lift, transit, descend).
            joint_states: Starting /joint_states for the transit.

        Returns:
            A single concatenated JointTrajectory through all waypoints, or None.
        """
        try:
            with self._cuda_lock:
                return self._plan_place_locked(waypoints, joint_states)
        except Exception as exc:
            self._logger.error(f'CuRobo.plan_place error: {exc}')
            return None

    def _plan_place_locked(self, waypoints: list, joint_states) -> 'JointTrajectory | None':
        # Clear collision world: carried object would be treated as obstacle.
        self._clear_collision_world()
        self._logger.info(
            f'CuRobo.plan_place: planning {len(waypoints)}-waypoint transit '
            '(collision world cleared for carried object).')

        current = self._ros_js_to_curobo(joint_states)
        trajectories = []

        for i, pose in enumerate(waypoints):
            _reset_planner_seed(self._planner)
            traj = self._plan_trajectory_locked(pose, None, current=current)
            if traj is None or not traj.points:
                self._logger.warn(
                    f'CuRobo.plan_place: waypoint {i} planning failed.')
                return None
            trajectories.append(traj)
            current = self._ros_js_to_curobo(self._traj_end_joint_state(traj))

        if not trajectories:
            return None

        result = trajectories[0]
        for traj in trajectories[1:]:
            result = concat_trajectories(result, traj)

        self._logger.info(
            f'CuRobo.plan_place succeeded: '
            f'{len(result.points)} total points across {len(waypoints)} waypoints.')
        return result

    def _plan_trajectory_locked(self, goal_pose, joint_states, current=None):
        if joint_states is not None:
            self.update_joint_state(joint_states)
            self._update_world_from_tsdf()
        if current is None:
            current = self._ros_js_to_curobo(joint_states)
        goal = self._single_goal(goal_pose)
        last_status = 'unknown'
        for world_mode in self._trajectory_world_modes():
            if world_mode == 'relaxed':
                self._clear_collision_world()
                self._logger.warn(
                    'CuRobo.plan_trajectory retrying with relaxed collision '
                    'world; TSDF blocked the pose trajectory.')
            _reset_planner_seed(self._planner)
            result = self._planner.plan_pose(goal, current)
            if _result_success(result):
                if world_mode == 'relaxed':
                    self._logger.info(
                        'CuRobo.plan_trajectory succeeded: world=relaxed')
                torch.cuda.synchronize()
                return interp_traj_to_ros(
                    result.get_interpolated_plan(),
                    last_tstep=getattr(result, 'interpolated_last_tstep', None),
                )
            last_status = getattr(result, 'status', 'unknown')
            self._logger.warn(
                'CuRobo.plan_trajectory failed: '
                f'world={world_mode} status={last_status}')
            self._logger.warn(
                'CuRobo.plan_trajectory diagnostics: '
                + _trajectory_failure_diagnostics(
                    result, goal_pose=goal_pose, world_mode=world_mode))
        self._logger.warn(
            f'CuRobo.plan_trajectory: all attempts exhausted ({last_status})')
        return None

    def tool_pose(self, joint_states):
        try:
            with self._cuda_lock:
                current = self._ros_js_to_curobo(joint_states)
                tp = self._planner.compute_kinematics(current).tool_poses
                torch.cuda.synchronize()
            pos = tp.position.reshape(-1)[:3].tolist()
            quat = tp.quaternion.reshape(-1)[:4].tolist()
            return tuple(pos), tuple(quat)
        except Exception as exc:
            self._logger.warning(
                f'CuRobo: FK failed: {type(exc).__name__}: {exc}')
            return None

    def _update_world_from_tsdf(self, object_cloud: np.ndarray | None = None) -> bool:
        with self._lock:
            frame_count = self._frame_count
            last_update = self._last_world_update_frame

        required_frames = _min_tsdf_frames()
        if frame_count < required_frames:
            self._logger.warning(
                f'CuRobo: map not ready ({frame_count}/{required_frames} '
                'dual-camera frames); planning in current/free collision model.')
            return False
        if last_update == frame_count and object_cloud is None:
            return True

        torch.cuda.synchronize()
        voxel_grid = self._mapper.compute_esdf()
        if object_cloud is not None and len(object_cloud) > 0:
            voxel_grid = self._carve_object_from_voxel_grid(voxel_grid, object_cloud)
        try:
            self._planner.clear_scene_cache()
        except Exception:
            pass
        # Always include static cuboids (floor + baskets) so they survive TSDF updates.
        self._planner.update_world(SceneCfg(cuboid=self._static_cuboids, voxel=[voxel_grid]))
        torch.cuda.synchronize()
        with self._lock:
            self._last_world_update_frame = frame_count
        if self._enable_viz:
            self._cache_viz_tsdf()
        carved = f' (carved {len(object_cloud)} object pts)' if object_cloud is not None else ''
        self._logger.info(
            f'CuRobo: updated TSDF collision world from {frame_count} frames{carved}.')
        return True

    def _carve_object_from_voxel_grid(self, voxel_grid, object_cloud_np: np.ndarray):
        """Set voxels near the target object AND its approach corridor to free space.

        Two regions are carved:
        1. Object voxels (±1 dilation) — prevents false self-collision with target.
        2. Approach corridor — a vertical column above the object bounding box,
           clearing floating noise points in the TSDF that would block top-down
           grasp approach paths.

        Falls back silently so planning always has a relaxed-world retry.
        """
        try:
            feature = voxel_grid.feature_tensor
            if feature is None:
                return voxel_grid

            # Normalise shape: drop leading batch dimensions until 3D or 4D.
            while feature.dim() > 4:
                feature = feature.squeeze(0)

            # Support both (nx, ny, nz) and (nx, ny, nz, C) layouts.
            is_4d = feature.dim() == 4
            nx, ny, nz = feature.shape[0], feature.shape[1], feature.shape[2]
            voxel_size = float(voxel_grid.voxel_size)

            # Grid centre from pose [x, y, z, qw, qx, qy, qz].
            pose = voxel_grid.pose
            center = torch.tensor(
                [pose[0], pose[1], pose[2]],
                dtype=torch.float32, device=feature.device)
            dims = torch.tensor(
                list(voxel_grid.dims),
                dtype=torch.float32, device=feature.device)
            origin = center - dims / 2.0

            # ── region 1: object points (±1 dilation) ───────────────────────
            pts = torch.tensor(
                object_cloud_np, dtype=torch.float32, device=feature.device)
            idx = ((pts - origin) / voxel_size).long()

            offsets = torch.tensor(
                [[di, dj, dk]
                 for di in range(-1, 2)
                 for dj in range(-1, 2)
                 for dk in range(-1, 2)],
                dtype=torch.long, device=feature.device,
            )
            obj_idx = (idx.unsqueeze(1) + offsets.unsqueeze(0)).reshape(-1, 3)

            # ── region 2: cylindrical approach corridor above object ─────────
            # Removes floating TSDF noise directly above the target object so
            # top-down approach paths are not blocked.
            #
            # Uses a CYLINDER (not a rectangle) so nearby objects (e.g. a cola
            # can 10-15 cm away) are NOT accidentally erased from the collision
            # world.  Rectangle + padding was removing real obstacles that fell
            # inside the padded bbox of elongated objects like hammers.
            corridor_radius = _env_float(
                'PIPELINE_CUROBO_CORRIDOR_RADIUS', 0.06)   # 6 cm radius cylinder
            corridor_z_above = _env_float(
                'PIPELINE_CUROBO_CORRIDOR_Z_ABOVE', 0.35)  # 35 cm above object top

            obj_np = np.asarray(object_cloud_np, dtype=np.float32)
            cx = float(obj_np[:, 0].mean())
            cy = float(obj_np[:, 1].mean())
            z_obj_top = float(obj_np[:, 2].max())
            z_cor_top = z_obj_top + corridor_z_above

            # Sample a grid inside the bounding square, then mask to circle.
            xs = np.arange(cx - corridor_radius, cx + corridor_radius + voxel_size, voxel_size)
            ys = np.arange(cy - corridor_radius, cy + corridor_radius + voxel_size, voxel_size)
            zs = np.arange(z_obj_top, z_cor_top + voxel_size, voxel_size)
            if xs.size > 0 and ys.size > 0 and zs.size > 0:
                gx, gy, gz = np.meshgrid(xs, ys, zs, indexing='ij')
                in_circle = (gx - cx) ** 2 + (gy - cy) ** 2 <= corridor_radius ** 2
                corridor_pts_np = np.stack(
                    [gx[in_circle].ravel(),
                     gy[in_circle].ravel(),
                     gz[in_circle].ravel()], axis=1
                ).astype(np.float32)
                cor_pts = torch.tensor(
                    corridor_pts_np, dtype=torch.float32, device=feature.device)
                cor_idx = ((cor_pts - origin) / voxel_size).long()
            else:
                cor_idx = torch.zeros((0, 3), dtype=torch.long, device=feature.device)

            # ── merge and clamp to grid bounds ────────────────────────────────
            all_idx = torch.cat([obj_idx, cor_idx], dim=0)
            valid = (
                (all_idx[:, 0] >= 0) & (all_idx[:, 0] < nx) &
                (all_idx[:, 1] >= 0) & (all_idx[:, 1] < ny) &
                (all_idx[:, 2] >= 0) & (all_idx[:, 2] < nz)
            )
            all_idx = all_idx[valid]

            if all_idx.shape[0] == 0:
                return voxel_grid

            # Positive ESDF value = free space (distance to nearest surface).
            if is_4d:
                feature[all_idx[:, 0], all_idx[:, 1], all_idx[:, 2], 0] = voxel_size
            else:
                feature[all_idx[:, 0], all_idx[:, 1], all_idx[:, 2]] = voxel_size

            n_obj = obj_idx.shape[0]
            n_cor = all_idx.shape[0] - n_obj
            self._logger.info(
                f'CuRobo: carved {n_obj} object voxels + {n_cor} corridor voxels '
                f'({len(object_cloud_np)} pts, voxel_size={voxel_size:.3f}m, '
                f'corridor r={corridor_radius:.2f}m z_above={corridor_z_above:.2f}m).')
            return voxel_grid
        except Exception as exc:
            self._logger.warn(
                f'CuRobo: _carve_object_from_voxel_grid failed (non-fatal): '
                f'{type(exc).__name__}: {exc}')
            return voxel_grid

    def _build_static_cuboids(self) -> list:
        """Build permanent collision obstacles: floor/table and storage baskets.

        These are always included in every update_world call so cuRobo never
        plans a path that goes through the floor or the basket structures,
        even when TSDF voxels are cleared (e.g. during place motion).

        Coordinate convention (base_link):
          - Table surface: z ≈ -0.07 m  → floor cuboid top at -0.07 m
          - Storage baskets sit on the side platforms at y ≈ ±0.5 m.
            Basket walls extend from table surface up to ~0.28 m.
            The place poses (tool0 z = 0.37 m) are above the basket tops,
            so the arm descends from transit height (0.50 m) straight down
            through the open basket top — basket wall cuboids don't block this.

        All cuboid poses: [x, y, z, qw, qx, qy, qz]
        """
        from curobo._src.geom.types import Cuboid

        floor_top_z = float(os.environ.get('PIPELINE_FLOOR_Z', -0.07))
        floor_thickness = 0.12
        floor = Cuboid(
            name='floor',
            pose=[0.0, 0.0, floor_top_z - floor_thickness / 2, 1.0, 0.0, 0.0, 0.0],
            dims=[2.5, 2.5, floor_thickness],
        )

        cuboids = [floor]

        # Storage basket A (positive Y side).
        # X: [-0.021, 0.099] → center 0.039, span 0.12 + 0.08 margin = 0.20
        # Y: [ 0.439, 0.649] → center 0.544, span 0.21 + 0.09 margin = 0.30
        # Basket wall top must be below the gripper collision sphere when at place pose.
        # place pose tool0 z=0.370 → lowest sphere bottom = 0.370-0.055-0.040 = 0.275 m
        # With activation_dist=0.05 m: basket_wall_top < 0.275-0.05 = 0.225 m
        # Use 0.20 m (5 cm margin below the limit). Tune with PIPELINE_BASKET_WALL_TOP.
        basket_wall_top = float(os.environ.get('PIPELINE_BASKET_WALL_TOP', 0.20))
        basket_height = basket_wall_top - floor_top_z
        basket_center_z = floor_top_z + basket_height / 2
        if _env_bool('PIPELINE_STATIC_BASKETS', True):
            cuboids.append(Cuboid(
                name='basket_a',
                pose=[0.039, 0.544, basket_center_z, 1.0, 0.0, 0.0, 0.0],
                dims=[0.20, 0.30, basket_height],
            ))
            # Storage basket B (negative Y side).
            # X: [-0.051, 0.069] → center 0.009, span 0.12 + 0.08 = 0.20
            # Y: [-0.431,-0.611] → center -0.521, span 0.18 + 0.12 = 0.30
            cuboids.append(Cuboid(
                name='basket_b',
                pose=[0.009, -0.521, basket_center_z, 1.0, 0.0, 0.0, 0.0],
                dims=[0.20, 0.30, basket_height],
            ))

        self._logger.info(
            f'CuRobo: static collision world — floor top z={floor_top_z:.3f} m, '
            f'{"baskets A+B included" if len(cuboids) > 1 else "baskets disabled"}.')
        return cuboids

    def _clear_collision_world(self) -> None:
        """Reset world to static obstacles only (floor + baskets), clearing TSDF voxels."""
        try:
            self._planner.clear_scene_cache()
        except Exception:
            pass
        self._planner.update_world(SceneCfg(cuboid=self._static_cuboids))
        torch.cuda.synchronize()
        with self._lock:
            self._last_world_update_frame = -1

    def _pick_world_modes(self) -> tuple:
        # Try TSDF first for collision-aware planning, fall back to relaxed.
        # TSDF was previously disabled due to wrist sphere / IK failures at
        # low grasp heights. Tuned ur5_curobo.yml (smaller wrist spheres,
        # extended self_collision_ignore) should fix this.
        # Disable TSDF with: PIPELINE_CUROBO_PICK_USE_TSDF=false
        if _env_bool('PIPELINE_CUROBO_PICK_USE_TSDF', True):
            return ('tsdf', 'relaxed')
        return ('relaxed',)

    def _trajectory_world_modes(self) -> tuple:
        if _env_bool('PIPELINE_CUROBO_TRAJ_RELAXED_RETRY', True):
            return ('tsdf', 'relaxed')
        return ('tsdf',)

    def _ros_js_to_curobo(self, joint_states) -> CuRoboJointState:
        name_to_pos = dict(zip(joint_states.name, joint_states.position))
        missing = [j for j in JOINT_NAMES if j not in name_to_pos]
        if missing:
            raise ValueError(f'CuRobo: /joint_states missing joints: {missing}')
        pos = torch.tensor(
            [[name_to_pos[j] for j in JOINT_NAMES]],
            device='cuda', dtype=torch.float32,
        )
        return CuRoboJointState.from_position(
            pos, joint_names=list(JOINT_NAMES))

    def _grasps_to_goalset(self, grasp_candidates) -> GoalToolPose:
        mats = np.stack([_candidate_pose_4x4(g) for g in grasp_candidates])
        t_tool_grasp = np.eye(4, dtype=np.float32)
        t_tool_grasp[2, 3] = _effective_gripper_tcp_z_offset()
        tool = np.stack([m @ np.linalg.inv(t_tool_grasp) for m in mats])
        pos = tool[:, :3, 3]
        quat_xyzw = R.from_matrix(tool[:, :3, :3]).as_quat()
        quat_wxyz = np.concatenate(
            [quat_xyzw[:, 3:4], quat_xyzw[:, :3]], axis=1)
        n = pos.shape[0]
        return GoalToolPose(
            tool_frames=self._planner.tool_frames,
            position=torch.tensor(
                pos, device='cuda', dtype=torch.float32).view(1, 1, 1, n, 3),
            quaternion=torch.tensor(
                quat_wxyz, device='cuda', dtype=torch.float32).view(1, 1, 1, n, 4),
        )

    def _single_goal(self, goal_pose) -> GoalToolPose:
        if hasattr(goal_pose, 'position') and hasattr(goal_pose, 'orientation'):
            p, o = goal_pose.position, goal_pose.orientation
            xyz = [p.x, p.y, p.z]
            quat = [o.w, o.x, o.y, o.z]
        else:
            values = list(goal_pose)
            if len(values) != 7:
                raise ValueError(
                    'goal_pose must be geometry_msgs/Pose or '
                    '(x,y,z,qw,qx,qy,qz)')
            xyz = values[:3]
            quat = values[3:]
        return GoalToolPose(
            tool_frames=self._planner.tool_frames,
            position=torch.tensor(
                xyz, device='cuda', dtype=torch.float32).view(1, 1, 1, 1, 3),
            quaternion=torch.tensor(
                quat, device='cuda', dtype=torch.float32).view(1, 1, 1, 1, 4),
        )

    def _build_planner(self):
        collision_cache = {
            'primitive': 10,  # pre-allocate slots for static cuboids (floor, basket_a, basket_b + margin)
            'voxel': {
                'layers': 1,
                'dims': [7.0, 7.0, 7.0],
                'voxel_size': 0.05,
            }
        }
        config = MotionPlannerCfg.create(
            robot=UR5_CONFIG,
            scene_model='collision_test.yml',
            collision_cache=collision_cache,
            max_goalset=TOPK_GRASPS,
            use_cuda_graph=_env_bool('PIPELINE_CUROBO_USE_CUDA_GRAPH', False),
            optimizer_collision_activation_distance=_env_float(
                'PIPELINE_CUROBO_COLLISION_ACTIVATION_DIST', 0.05),
        )
        planner = MotionPlanner(config)
        use_graph = _env_bool('PIPELINE_CUROBO_USE_CUDA_GRAPH', False)
        planner.warmup(enable_graph=use_graph, num_warmup_iterations=3)
        return planner

    def _configure_mapper_cuda_graphs(self) -> None:
        if _env_bool('PIPELINE_CUROBO_MAPPER_USE_CUDA_GRAPH', False):
            return
        try:
            integrator = self._mapper.integrator
            integrator.use_cuda_graph = False
            integrator._integrate_graph = None
            integrator._compute_esdf_graph = None
            self._logger.info('CuRobo: Mapper CUDA graphs disabled.')
        except Exception as exc:
            self._logger.warning(
                f'CuRobo: could not disable Mapper CUDA graphs: {exc}')


def interp_traj_to_ros(
    interp_traj,
    dt: float = INTERP_DT,
    time_offset: float = 0.0,
    last_tstep=None,
) -> JointTrajectory:
    traj = interp_traj.squeeze(0) if hasattr(interp_traj, 'squeeze') else interp_traj
    positions, velocities, accelerations = _extract_traj_arrays(traj)
    last_tstep = _last_tstep_to_int(last_tstep)
    if last_tstep is not None and last_tstep > 0:
        last_tstep = min(last_tstep, positions.shape[0])
        positions = positions[:last_tstep]
        if velocities is not None:
            velocities = velocities[:last_tstep]
        if accelerations is not None:
            accelerations = accelerations[:last_tstep]
    time_scale = _trajectory_time_scale()

    jt = JointTrajectory()
    jt.joint_names = list(JOINT_NAMES)
    for i, pos in enumerate(positions):
        pt = JointTrajectoryPoint()
        pt.positions = [float(x) for x in pos]
        if velocities is not None:
            pt.velocities = [float(x) / time_scale for x in velocities[i]]
        if accelerations is not None:
            pt.accelerations = [
                float(x) / (time_scale * time_scale)
                for x in accelerations[i]
            ]
        t_sec = time_offset + (i + 1) * dt * time_scale
        pt.time_from_start = RosDuration(
            sec=int(t_sec),
            nanosec=int((t_sec % 1.0) * 1_000_000_000),
        )
        jt.points.append(pt)
    return jt


def concat_trajectories(
    traj_a: JointTrajectory, traj_b: JointTrajectory
) -> JointTrajectory:
    combined = JointTrajectory()
    combined.joint_names = list(traj_a.joint_names)
    combined.points = list(traj_a.points)
    if traj_a.points:
        last = traj_a.points[-1].time_from_start
        offset = last.sec + last.nanosec * 1e-9
    else:
        offset = 0.0
    for pt in traj_b.points:
        t = pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9 + offset
        new_pt = JointTrajectoryPoint()
        new_pt.positions = list(pt.positions)
        new_pt.velocities = list(pt.velocities)
        new_pt.accelerations = list(pt.accelerations)
        new_pt.time_from_start = RosDuration(
            sec=int(t),
            nanosec=int((t % 1.0) * 1_000_000_000),
        )
        combined.points.append(new_pt)
    return combined


def _extract_traj_arrays(traj):
    pos_t = traj.position
    while pos_t.dim() > 2:
        pos_t = pos_t[0]
    positions = pos_t.cpu().numpy()

    velocities = None
    vel_t = getattr(traj, 'velocity', None)
    if vel_t is not None:
        while vel_t.dim() > 2:
            vel_t = vel_t[0]
        velocities = vel_t.cpu().numpy()

    accelerations = None
    acc_t = getattr(traj, 'acceleration', None)
    if acc_t is not None:
        while acc_t.dim() > 2:
            acc_t = acc_t[0]
        accelerations = acc_t.cpu().numpy()
    return positions, velocities, accelerations


def _last_tstep_to_int(last_tstep):
    if last_tstep is None:
        return None
    try:
        if hasattr(last_tstep, 'detach'):
            last_tstep = last_tstep.detach().cpu().numpy()
        return int(np.asarray(last_tstep).reshape(-1)[0])
    except Exception as exc:
        raise ValueError(
            f'Invalid cuRobo interpolated last_tstep: {last_tstep!r}'
        ) from exc


def _candidate_pose_4x4(candidate) -> np.ndarray:
    if isinstance(candidate, dict):
        pose = candidate.get('pose_4x4')
    else:
        pose = getattr(candidate, 'pose_4x4', None)
    if pose is None:
        raise ValueError('grasp candidate is missing pose_4x4')
    pose = np.asarray(pose, dtype=np.float32)
    if pose.shape != (4, 4):
        raise ValueError(f'grasp candidate pose_4x4 has shape {pose.shape}')
    return pose


def _effective_gripper_tcp_z_offset() -> float:
    # graspgenX outputs tool0 frame directly, so base offset is 0.0.
    # PIPELINE_CUROBO_GRASP_CLOSE_EXTRA_M can add extra depth if grasps are
    # consistently too shallow (positive = push deeper toward object).
    close_extra = _env_float('PIPELINE_CUROBO_GRASP_CLOSE_EXTRA_M', 0.0)
    if close_extra < 0.0:
        raise ValueError('PIPELINE_CUROBO_GRASP_CLOSE_EXTRA_M must be non-negative')
    return GRIPPER_TCP_Z_OFFSET + close_extra




def _pick_failure_diagnostics(
    result,
    grasp_candidates,
    *,
    world_mode: str,
    candidate_index: int,
    approach_offset: float,
    lift_offset: float,
    disabled_collision_links,
) -> str:
    goal_index = _goalset_index(result)
    candidate = _candidate_at(grasp_candidates, goal_index)
    parts = [
        f'world={world_mode}',
        f'candidate_index={candidate_index}',
        f'goalset_index={goal_index}',
        f'approach_offset={approach_offset:.3f}',
        f'lift_offset={lift_offset:.3f}',
        f'disabled_contact_links={list(disabled_collision_links)}',
        _stage_result_text('goalset', getattr(result, 'goalset_result', None)),
        _stage_result_text('approach', getattr(result, 'approach_result', None)),
        _stage_result_text('grasp', getattr(result, 'grasp_result', None)),
        _stage_result_text('lift', getattr(result, 'lift_result', None)),
    ]
    if candidate is not None:
        try:
            tcp = _candidate_pose_4x4(candidate)
            tool = _tool_pose_from_grasp_tcp(tcp)
            approach = tool @ _translation_matrix(0.0, 0.0, approach_offset)
            lift = tool.copy()
            lift[:3, 3] += np.array([0.0, 0.0, lift_offset], dtype=np.float32)
            parts.extend([
                f'grasp_tcp_xyz={_xyz_text(tcp[:3, 3])}',
                f'tool0_grasp_xyz={_xyz_text(tool[:3, 3])}',
                f'tool0_approach_xyz={_xyz_text(approach[:3, 3])}',
                f'tool0_lift_xyz={_xyz_text(lift[:3, 3])}',
                f'tool0_z_axis={_xyz_text(tool[:3, 2])}',
            ])
        except Exception as exc:
            parts.append(f'pose_diagnostics_error={type(exc).__name__}: {exc}')
    return '; '.join(parts)


def _trajectory_failure_diagnostics(result, *, goal_pose, world_mode: str) -> str:
    return '; '.join([
        f'world={world_mode}',
        f'goal_pose={_pose_text(_pose_values(goal_pose))}',
        _stage_result_text('pose', result),
    ])


def _tool_pose_from_grasp_tcp(grasp_tcp) -> np.ndarray:
    t_tool_grasp = np.eye(4, dtype=np.float32)
    t_tool_grasp[2, 3] = _effective_gripper_tcp_z_offset()
    return np.asarray(grasp_tcp, dtype=np.float32) @ np.linalg.inv(t_tool_grasp)


def _translation_matrix(x, y, z) -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, 3] = [x, y, z]
    return transform


def _goalset_index(result):
    value = getattr(result, 'goalset_index', None)
    if value is None:
        return None
    try:
        if hasattr(value, 'detach'):
            value = value.detach().cpu().numpy()
        return int(np.asarray(value).reshape(-1)[0])
    except Exception:
        return None


def _candidate_at(candidates, index):
    if index is None:
        index = 0
    try:
        return list(candidates)[int(index)]
    except Exception:
        return None


def _stage_result_text(name, result) -> str:
    if result is None:
        return f'{name}=None'
    return (
        f'{name}('
        f'success={_success_text(result)}, '
        f'position_error={_numeric_text(getattr(result, "position_error", None))}, '
        f'rotation_error={_numeric_text(getattr(result, "rotation_error", None))}, '
        f'cspace_error={_numeric_text(getattr(result, "cspace_error", None))}, '
        f'feasible={_numeric_text(getattr(result, "feasible", None))}, '
        f'debug={getattr(result, "debug_info", None)!r})'
    )


def _success_text(result) -> str:
    success = getattr(result, 'success', None)
    if success is None:
        return 'None'
    try:
        if hasattr(success, 'detach'):
            success = success.detach().cpu().numpy()
        if (
            not isinstance(success, (bool, int, float, list, tuple, np.ndarray))
            and hasattr(success, 'any')
        ):
            return str([bool(success.any())])
        return str(np.asarray(success).astype(bool).reshape(-1).tolist())
    except Exception:
        try:
            return str([bool(success.any())])
        except Exception:
            return repr(success)


def _numeric_text(value) -> str:
    if value is None:
        return 'None'
    try:
        if hasattr(value, 'detach'):
            value = value.detach().cpu().numpy()
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return '[]'
        return '[' + ','.join(f'{float(item):.4g}' for item in arr[:6]) + ']'
    except Exception:
        return repr(value)


def _xyz_text(values) -> str:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    return '(' + ','.join(f'{float(value):.3f}' for value in arr[:3]) + ')'


def _pose_values(goal_pose) -> list:
    if hasattr(goal_pose, 'position') and hasattr(goal_pose, 'orientation'):
        p, o = goal_pose.position, goal_pose.orientation
        return [p.x, p.y, p.z, o.w, o.x, o.y, o.z]
    return list(goal_pose)


def _pose_text(values) -> str:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    return '(' + ','.join(f'{float(value):.3f}' for value in arr[:7]) + ')'


def _result_success(result) -> bool:
    if result is None:
        return False
    success = getattr(result, 'success', None)
    if success is None:
        return False
    try:
        return bool(success.any())
    except AttributeError:
        return bool(success)


def _trajectory_time_scale() -> float:
    scale = _env_float('PIPELINE_CUROBO_TRAJ_TIME_SCALE', 3.0)
    if scale <= 0.0:
        raise ValueError('PIPELINE_CUROBO_TRAJ_TIME_SCALE must be positive')
    return scale


def _min_tsdf_frames() -> int:
    return max(_env_int('PIPELINE_CUROBO_MIN_PLANNING_FRAMES', 5), MIN_FRAMES)



def _make_hold_trajectory(traj: 'JointTrajectory') -> 'JointTrajectory':
    """Return a single-point JointTrajectory that holds the last position
    of *traj* for a short duration.  Used as the 'grasp' segment when
    plan_pick uses plan_trajectory internally so that
    concat_trajectories(approach, grasp) doesn't replay the full approach."""
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from builtin_interfaces.msg import Duration as RosDuration
    hold = JointTrajectory()
    hold.joint_names = list(traj.joint_names)
    last = traj.points[-1]
    pt = JointTrajectoryPoint()
    pt.positions = list(last.positions)
    pt.velocities = [0.0] * len(last.positions)
    pt.accelerations = [0.0] * len(last.positions)
    # Short hold duration — just long enough for the gripper to close.
    pt.time_from_start = RosDuration(sec=1, nanosec=0)
    hold.points.append(pt)
    return hold


class _TrajWrapper:
    """Wraps a ROS JointTrajectory so interp_traj_to_ros can handle it.

    curobo_service._handle_pick calls interp_traj_to_ros on the
    *_interpolated_trajectory fields. When plan_pick uses plan_trajectory
    instead of plan_grasp, the trajectory is already a JointTrajectory,
    so we just pass it through.
    """

    def __init__(self, ros_traj):
        self._traj = ros_traj

    # interp_traj_to_ros squeezes leading batch dims from .position tensor.
    # We expose a fake tensor interface that returns the pre-built JointTrajectory.
    def get_ros_traj(self):
        return self._traj


def _pick_approach_offsets() -> tuple:
    """Approach offsets to try per candidate (negative = back off along tool-z)."""
    raw = os.environ.get('PIPELINE_CUROBO_GRASP_APPROACH_OFFSETS', '-0.05,-0.10,-0.15')
    offsets = []
    for item in raw.split(','):
        item = item.strip()
        if item:
            offsets.append(float(item))
    return tuple(offsets) if offsets else (-0.05, -0.10, -0.15)


def _reset_planner_seed(planner) -> None:
    try:
        planner.reset_seed()
    except Exception:
        pass
