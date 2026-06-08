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

from curobo._src.geom.types import Cuboid, SceneCfg
from curobo._src.robot.kinematics.kinematics import Kinematics
from curobo._src.types.robot import RobotCfg
from curobo._src.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.perception import FilterDepth, Mapper, MapperCfg, RobotSegmenter
from curobo.types import CameraObservation, GoalToolPose
from curobo.types import JointState as CuRoboJointState
from curobo.types import Pose as CuRoboPose

from team_8.pipeline_utils import (
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
UR5_CONFIG = '/ros2_ws/src/team_8/config/ur5_curobo.yml'
JOINT_NAMES = (
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
)

TOPK_GRASPS = 30
INTERP_DT = 0.02
GRIPPER_TCP_Z_OFFSET = 0.1034

# Distance from tool0 origin to the outermost finger tip (Robotiq 2F-85 fully
# open).  The planning collision spheres use 0.136 m, but the actual Gazebo
# mesh extends ~0.155 m.  Using the larger value ensures the z-clamp prevents
# any part of the physical mesh from reaching the floor.
_FINGERTIP_LEN = 0.155


class PickPlan:
    """Three deployable ROS trajectories for a pick: approach (to the
    pre-grasp pose, planned against the TSDF), grasp (the collision-off descent
    onto the object), and lift (the collision-off retreat after the gripper
    closes). ``goalset_index`` is the chosen grasp candidate.
    """

    __slots__ = ('approach', 'grasp', 'lift', 'goalset_index')

    def __init__(self, approach, grasp, lift, goalset_index):
        self.approach = approach
        self.grasp = grasp
        self.lift = lift
        self.goalset_index = goalset_index


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

        # Map is centered on base_link in xy, but offset up in z so it spans
        # base_link z in [-0.1, 0.75]: the table surface sits at z~=0, so the
        # -0.1 floor keeps the tabletop plane as an obstacle while dropping the
        # ~0.65m of empty grid that used to extend below the table (centered
        # extent would put the floor at -0.75). Cuts wasted voxels from both the
        # collision world and the debug viz.
        self._mapper = Mapper(MapperCfg(
            extent_meters_xyz=(2.0, 2.0, 0.85),
            grid_center=torch.tensor(
                [0.0, 0.0, 0.325], dtype=torch.float32, device='cuda'),
            voxel_size=0.015,
            esdf_voxel_size=0.015,
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
        from team_8.live_viz_helpers import depth_to_xyz
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
                        frame_count = self._frame_count
                    if self._enable_viz and frame_count % 10 == 0:
                        self._cache_viz_tsdf()
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
        """Plan approach -> grasp -> lift for ranked GraspGen TCP poses."""
        with self._cuda_lock:
            return self._plan_pick_locked(
                grasp_candidates, joint_states, object_cloud=object_cloud)

    def _plan_pick_locked(self, grasp_candidates, joint_states, object_cloud=None):
        """Plan directly to each GraspGenX grasp pose (no pre-grasp + descent).

        Strategy
        --------
        1. Convert all candidates to tool0 poses and apply a floor z-clamp.
        2. Carve the TSDF around the object bbox **and** around every grasp
           tool0 position (sphere of radius PIPELINE_CUROBO_GRASP_SPHERE_CARVE_RADIUS,
           default 0.15 m).  This removes TSDF noise right at the grasp site so
           that cuRobo's collision checker does not reject start/end states that
           are actually valid.
        3. Plan to the full goalset (all candidates simultaneously) with fingertip
           collision links disabled.  cuRobo picks the kinematically easiest
           candidate automatically.
        4. If the carved-TSDF plan fails, retry with a fully cleared world
           (neighbor collision risk, but a complete pick failure is worse).
        5. Lift with a cleared world (arm at grasp pose is adjacent to TSDF
           voxels; keeping them causes "start-state-in-collision").
        """
        try:
            t0 = time.perf_counter()
            self.update_joint_state(joint_states)

            candidates = list(grasp_candidates)[:TOPK_GRASPS]
            if not candidates:
                self._logger.warn('CuRobo.plan_pick: no grasp candidates given.')
                return None

            # ── Build tool0 poses for all candidates, apply floor z-clamp ──────
            # Two-component clamp:
            #
            # 1. Tilt-scaled component: fingertip only descends approach_z × _FINGERTIP_LEN
            #    in world-Z for a tilted approach, so the minimum is lower than vertical.
            #    approach_z = |mat[2,2]| = cos(tilt from vertical)
            #      vertical  → 1.0 → full _FINGERTIP_LEN applied
            #      45° tilt  → 0.71 → ~71 % applied
            #      60° tilt  → 0.50 → 50 % applied
            #
            # 2. Absolute minimum: gripper BODY parts (not just fingertip) can protrude
            #    below tool0 for tilted grasps. A hard lower bound on tool0-z prevents
            #    the palm / link geometry from colliding with the table regardless of tilt.
            #
            # The effective clamp is the maximum of both components.
            floor_z = _env_float('PIPELINE_FLOOR_Z', -0.07)
            descent_margin = _env_float('PIPELINE_CUROBO_DESCENT_MARGIN', 0.030)
            # Absolute clearance: tool0 always at least this far above floor_z.
            # 0.06 m ensures gripper body clears the table for any tilt angle.
            gripper_body_clearance = _env_float(
                'PIPELINE_CUROBO_GRIPPER_BODY_CLEARANCE', 0.06)
            min_tool_z_abs = floor_z + gripper_body_clearance

            grasp_tool_mats = []   # (4,4) tool0 pose in base_link for each candidate
            for c in candidates:
                mat = _tool_pose_from_grasp_tcp(_candidate_pose_4x4(c))
                # |mat[2,2]| = world-Z component of tool local-Z axis
                approach_z = abs(float(mat[2, 2]))
                effective_fingertip_drop = approach_z * _FINGERTIP_LEN
                # Scale margin by approach_z too: for a horizontal grasp the
                # fingers barely move in world-Z during closure, so the safety
                # margin is also proportionally smaller.
                effective_margin = descent_margin * approach_z
                min_tool_z_tilt = floor_z + effective_fingertip_drop + effective_margin
                # Take the stricter of the two bounds.
                min_tool_z = max(min_tool_z_tilt, min_tool_z_abs)
                original_z = float(mat[2, 3])
                clamped_z = max(original_z, min_tool_z)
                if clamped_z > original_z + 1e-4:
                    mat = mat.copy()
                    mat[2, 3] = clamped_z
                    self._logger.info(
                        f'CuRobo.plan_pick: candidate z clamped '
                        f'{original_z:.3f} → {clamped_z:.3f}m '
                        f'(floor={floor_z:.3f} approach_z={approach_z:.2f} '
                        f'tilt_min={min_tool_z_tilt:.3f} '
                        f'abs_min={min_tool_z_abs:.3f})')
                grasp_tool_mats.append(mat)

            # Positions used for TSDF sphere carving (one per candidate)
            grasp_positions = [m[:3, 3] for m in grasp_tool_mats]

            # ── Update TSDF world with aggressive carving around grasp sites ────
            self._update_world_from_tsdf(
                object_cloud=object_cloud, grasp_positions=grasp_positions)
            t_world = time.perf_counter()

            current = self._ros_js_to_curobo(joint_states)
            collision_links = _pick_disable_collision_links(self._planner)

            # ── Tiered planning: most-vertical candidates first ──────────────────
            # cuRobo minimises joint-space travel cost over the entire goalset; it
            # can therefore pick a tilted grasp simply because it requires fewer
            # wrist rotations.  We solve this by trying a top-down-only tier first
            # and only falling back to all candidates when that fails.
            #
            # Tier 0: approach_z >= TIER0_MIN_APPROACH_Z (default 0.85, ≈32° tilt)
            # Tier 1: all candidates (fallback)
            tier0_min_az = _env_float('PIPELINE_CUROBO_TIER0_MIN_APPROACH_Z', 0.85)
            tier0_mats = [m for m in grasp_tool_mats
                          if abs(float(m[2, 2])) >= tier0_min_az]

            # Build (mat_list, label) pairs in priority order.
            plan_tiers: list[tuple[list, str]] = []
            if tier0_mats and len(tier0_mats) < len(grasp_tool_mats):
                plan_tiers.append((
                    tier0_mats,
                    f'tier0({len(tier0_mats)} top-down, '
                    f'approach_z≥{tier0_min_az:.2f})',
                ))
            plan_tiers.append((
                grasp_tool_mats,
                f'all-candidates({len(grasp_tool_mats)})',
            ))

            grasp_result = None
            winning_mats = grasp_tool_mats  # mats for the successful tier (idx lookup)
            for tier_mats, tier_label in plan_tiers:
                _reset_planner_seed(self._planner)
                self._planner.disable_link_collision(collision_links)
                try:
                    r = self._planner.plan_pose(
                        self._mats_to_goalset(tier_mats), current)
                finally:
                    self._planner.enable_link_collision(collision_links)
                torch.cuda.synchronize()
                if _result_success(r):
                    grasp_result = r
                    winning_mats = tier_mats
                    self._logger.info(
                        f'CuRobo.plan_pick: succeeded with {tier_label}')
                    break
                self._logger.info(
                    f'CuRobo.plan_pick: {tier_label} failed, trying next tier.')

            if not _result_success(grasp_result):
                # Final fallback: wipe TSDF world entirely, try all candidates.
                self._logger.warn(
                    'CuRobo.plan_pick: all TSDF tiers failed; '
                    'retrying with cleared world.')
                self._clear_collision_world()
                _reset_planner_seed(self._planner)
                self._planner.disable_link_collision(collision_links)
                try:
                    grasp_result = self._planner.plan_pose(
                        self._mats_to_goalset(grasp_tool_mats), current)
                finally:
                    self._planner.enable_link_collision(collision_links)
                torch.cuda.synchronize()
                winning_mats = grasp_tool_mats

            if not _result_success(grasp_result):
                status = getattr(grasp_result, 'status', 'unknown')
                self._logger.warn(
                    'CuRobo.plan_pick: all planning attempts failed '
                    f'(status={status}).')
                return None

            t_grasp = time.perf_counter()
            idx = _goalset_index(grasp_result)
            if idx is None:
                idx = 0

            grasp_jt = interp_traj_to_ros(
                grasp_result.get_interpolated_plan(),
                last_tstep=getattr(grasp_result, 'interpolated_last_tstep', None),
            )

            # ── Lift: clear world then plan straight up ──────────────────────────
            lift_offset = _pick_lift_offset()
            chosen_mat = winning_mats[idx]  # pose from the tier that succeeded
            lift_tool = chosen_mat.copy()
            lift_tool[:3, 3] += np.array([0.0, 0.0, lift_offset], dtype=np.float32)

            self._clear_collision_world()
            lift_jt = self._plan_pose_segment(
                lift_tool, self._final_joint_state(grasp_jt), 'lift')
            if lift_jt is None:
                return None
            t_lift = time.perf_counter()

            chosen_approach_z = abs(float(chosen_mat[2, 2]))
            self._logger.info(
                'CuRobo.plan_pick succeeded (direct-to-grasp): '
                f'total_candidates={len(candidates)} '
                f'winning_tier_size={len(winning_mats)} '
                f'chosen_idx={idx} '
                f'chosen_approach_z={chosen_approach_z:.2f} '
                f'chosen_tool0_z={float(chosen_mat[2, 3]):.3f}m '
                f'lift={lift_offset:.3f}m '
                f'(grasp={len(grasp_jt.points)}pts lift={len(lift_jt.points)}pts) '
                f'timing[s]: world={t_world - t0:.2f} '
                f'grasp={t_grasp - t_world:.2f} '
                f'lift={t_lift - t_grasp:.2f} total={t_lift - t0:.2f}')
            # grasp=None signals to curobo_service that approach already ends at
            # the grasp pose (no separate descent phase to concatenate).
            return PickPlan(
                approach=grasp_jt, grasp=None, lift=lift_jt,
                goalset_index=idx)
        except Exception as exc:
            self._logger.error(f'CuRobo.plan_pick error: {exc}')
            return None

    def _plan_pose_segment(self, goal_mat, current_state, name):
        """``plan_pose`` to a single tool0 pose (4x4 in base_link).

        The caller controls the collision world state before invoking this.
        Returns a ROS ``JointTrajectory`` or ``None`` on failure.
        """
        _reset_planner_seed(self._planner)
        result = self._planner.plan_pose(
            self._tool_goal_from_matrix(goal_mat), current_state)
        torch.cuda.synchronize()
        if not _result_success(result):
            status = getattr(result, 'status', 'unknown')
            self._logger.warn(
                f'CuRobo.plan_pick: {name} planning failed (status={status}).')
            return None
        return interp_traj_to_ros(
            result.get_interpolated_plan(),
            last_tstep=getattr(result, 'interpolated_last_tstep', None),
        )

    def _tool_goal_from_matrix(self, mat) -> GoalToolPose:
        """Single-goal GoalToolPose from a 4x4 tool0 pose in base_link."""
        mat = np.asarray(mat, dtype=np.float32)
        quat_xyzw = R.from_matrix(mat[:3, :3]).as_quat()
        quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
        return GoalToolPose(
            tool_frames=self._planner.tool_frames,
            position=torch.tensor(
                mat[:3, 3], device='cuda', dtype=torch.float32).view(1, 1, 1, 1, 3),
            quaternion=torch.tensor(
                quat_wxyz, device='cuda', dtype=torch.float32).view(1, 1, 1, 1, 4),
        )

    def _mats_to_goalset(self, mats) -> GoalToolPose:
        """Build a GoalToolPose goalset from a list of (4,4) tool0 pose matrices."""
        mats = [np.asarray(m, dtype=np.float32) for m in mats]
        pos = np.stack([m[:3, 3] for m in mats])
        quat_xyzw = R.from_matrix(np.stack([m[:3, :3] for m in mats])).as_quat()
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

    def _final_joint_state(self, jt) -> CuRoboJointState:
        """CuRoboJointState from the last point of a ROS JointTrajectory."""
        pos = torch.tensor(
            [list(jt.points[-1].positions)], device='cuda', dtype=torch.float32)
        return CuRoboJointState.from_position(pos, joint_names=list(JOINT_NAMES))

    def plan_trajectory(self, goal_pose, joint_states):
        """Plan a single tool0 trajectory for place/home style targets."""
        try:
            with self._cuda_lock:
                return self._plan_trajectory_locked(goal_pose, joint_states)
        except Exception as exc:
            self._logger.error(f'CuRobo.plan_trajectory error: {exc}')
            return None

    def _plan_trajectory_locked(self, goal_pose, joint_states):
        self.update_joint_state(joint_states)
        self._update_world_from_tsdf()
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

    def _update_world_from_tsdf(
        self, object_cloud=None, grasp_positions=None
    ) -> bool:
        with self._lock:
            frame_count = self._frame_count
            last_update = self._last_world_update_frame

        required_frames = _min_tsdf_frames()
        if frame_count < required_frames:
            self._logger.warning(
                f'CuRobo: map not ready ({frame_count}/{required_frames} '
                'dual-camera frames); planning in current/free collision model.')
            return False
        if last_update == frame_count and object_cloud is None and grasp_positions is None:
            return True

        torch.cuda.synchronize()
        voxel_grid = self._mapper.compute_esdf()

        if object_cloud is not None:
            voxel_grid = self._carve_object_from_voxel_grid(
                voxel_grid, object_cloud, grasp_positions=grasp_positions)
        elif grasp_positions is not None:
            voxel_grid = self._carve_object_from_voxel_grid(
                voxel_grid, None, grasp_positions=grasp_positions)

        try:
            self._planner.clear_scene_cache()
        except Exception:
            pass
        self._planner.update_world(
            SceneCfg(voxel=[voxel_grid],
                     cuboid=self._build_static_cuboids()))
        torch.cuda.synchronize()
        with self._lock:
            self._last_world_update_frame = frame_count
        if self._enable_viz:
            self._cache_viz_tsdf()
        self._logger.info(
            f'CuRobo: updated TSDF collision world from {frame_count} frames.')
        return True

    def _build_static_cuboids(self) -> list:
        """Floor cuboid for approach-phase collision world.

        Only the floor is added as a static obstacle. Baskets are already
        captured in the TSDF. The floor cuboid lets cuRobo route around it
        during approach planning even when it is not well-captured by the
        depth cameras (flat surface parallel to camera).
        """
        floor_z = _env_float('PIPELINE_FLOOR_Z', -0.07)
        floor_thickness = 0.02
        return [
            Cuboid(
                name='floor',
                pose=[0.0, 0.0, floor_z - floor_thickness / 2,
                      1.0, 0.0, 0.0, 0.0],
                dims=[4.0, 4.0, floor_thickness],
            )
        ]

    def _carve_object_from_voxel_grid(
        self, voxel_grid, object_cloud, grasp_positions=None
    ):
        """Remove object voxels, corridor, and grasp-site noise spheres from TSDF.

        Parameters
        ----------
        object_cloud : array-like or None
            Nx3 point cloud of the target object (world frame).
        grasp_positions : list of array-like or None
            Each entry is a (3,) xyz position of a grasp tool0 frame (world).
            A sphere of ``_GRASP_SPHERE_CARVE_RADIUS`` is carved around each
            position to remove TSDF noise that would block a direct-to-grasp plan.
        """
        try:
            grid = voxel_grid
            if grid.xyzr_tensor is None:
                return voxel_grid   # ESDF empty — nothing to carve
            centers = grid.xyzr_tensor.cpu().numpy()   # (N,4) x,y,z,radius
            xyz = centers[:, :3]

            carve_mask = np.zeros(xyz.shape[0], dtype=bool)

            if object_cloud is not None:
                obj_np = np.asarray(object_cloud, dtype=np.float32)
                if obj_np.ndim == 1:
                    obj_np = obj_np.reshape(-1, 3)
                if obj_np.shape[0] > 0:
                    # Carve object voxels (±2cm bounding-box dilation)
                    obj_min = obj_np.min(axis=0) - 0.02
                    obj_max = obj_np.max(axis=0) + 0.02
                    in_obj = np.all((xyz >= obj_min) & (xyz <= obj_max), axis=1)

                    # Carve cylindrical corridor above object centroid
                    corridor_radius = _env_float(
                        'PIPELINE_CUROBO_CORRIDOR_RADIUS', 0.06)
                    corridor_z_above = _env_float(
                        'PIPELINE_CUROBO_CORRIDOR_Z_ABOVE', 0.35)
                    cx = float(obj_np[:, 0].mean())
                    cy = float(obj_np[:, 1].mean())
                    z_obj_top = float(obj_np[:, 2].max())
                    in_corridor = (
                        ((xyz[:, 0] - cx) ** 2 + (xyz[:, 1] - cy) ** 2)
                        <= corridor_radius ** 2
                    ) & (xyz[:, 2] > z_obj_top) & (
                        xyz[:, 2] < z_obj_top + corridor_z_above)

                    carve_mask |= in_obj | in_corridor
                    self._logger.info(
                        f'CuRobo: bbox+corridor carve: '
                        f'obj={int(in_obj.sum())} corridor={int(in_corridor.sum())}')

            # Carve spheres around each grasp tool0 position to remove
            # local TSDF noise that would block a direct-to-grasp plan.
            # Use vectorized distance computation (one broadcast op, not a loop)
            # to avoid O(N_voxels × N_candidates) Python overhead.
            sphere_radius = _env_float(
                'PIPELINE_CUROBO_GRASP_SPHERE_CARVE_RADIUS', 0.15)
            # Limit to top-K to keep memory bounded: (N_voxels × K × 3 × 4 bytes)
            max_sphere_poses = int(os.environ.get(
                'PIPELINE_CUROBO_GRASP_SPHERE_MAX_POSES', '10'))
            if grasp_positions is not None and sphere_radius > 0:
                positions_arr = np.stack(
                    [np.asarray(p, dtype=np.float32).reshape(3)
                     for p in grasp_positions[:max_sphere_poses]]
                )  # (K, 3)
                r2 = sphere_radius ** 2
                # xyz: (N,3)  positions_arr: (K,3)
                # diff: (N,K,3) → dist2: (N,K) → any below r2: (N,)
                diff = xyz[:, np.newaxis, :] - positions_arr[np.newaxis, :, :]
                dist2_all = np.einsum('nkd,nkd->nk', diff, diff)
                in_any_sphere = np.any(dist2_all <= r2, axis=1)
                n_sphere_carved = int((in_any_sphere & ~carve_mask).sum())
                carve_mask |= in_any_sphere
                self._logger.info(
                    f'CuRobo: grasp-sphere carve (vectorized): '
                    f'radius={sphere_radius:.3f}m '
                    f'n_poses={len(positions_arr)} '
                    f'additional_voxels={n_sphere_carved}')

            n_carved = int(carve_mask.sum())
            if n_carved > 0:
                keep = ~carve_mask
                new_xyzr = torch.tensor(
                    centers[keep], dtype=torch.float32, device='cuda')
                grid.xyzr_tensor = new_xyzr
                self._logger.info(
                    f'CuRobo: total carved {n_carved} voxels from TSDF.')
        except Exception as exc:
            self._logger.warning(
                f'CuRobo: voxel carving failed: {type(exc).__name__}: {exc}')
        return voxel_grid

    def _clear_collision_world(self) -> None:
        """Clear TSDF voxels for descent/lift planning.

        Descent and lift use a world with no TSDF so the gripper can
        reach the object. Floor protection relies on the descent z-clamp
        (floor_z + fingertip_len + margin) rather than collision spheres.
        """
        try:
            self._planner.clear_scene_cache()
        except Exception:
            pass
        self._planner.update_world(SceneCfg())
        torch.cuda.synchronize()
        with self._lock:
            self._last_world_update_frame = -1

    def _pick_world_modes(self) -> tuple:
        if _env_bool('PIPELINE_CUROBO_PICK_RELAXED_RETRY', True):
            return ('tsdf', 'relaxed')
        return ('tsdf',)

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

    def _grasps_to_goalset(
        self, grasp_candidates, tool_z_offset: float = 0.0
    ) -> GoalToolPose:
        mats = np.stack([_candidate_pose_4x4(g) for g in grasp_candidates])
        t_tool_grasp = np.eye(4, dtype=np.float32)
        t_tool_grasp[2, 3] = _effective_gripper_tcp_z_offset()
        tool = np.stack([m @ np.linalg.inv(t_tool_grasp) for m in mats])
        # Back the goal off along each tool's own +z (toward the object) by
        # ``tool_z_offset`` — negative values yield the pre-grasp stand-off.
        if abs(tool_z_offset) > 1e-9:
            t_off = _translation_matrix(0.0, 0.0, tool_z_offset)
            tool = np.stack([t @ t_off for t in tool])
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
            'voxel': {
                'layers': 1,
                'dims': [3.0, 3.0, 3.0],
                'voxel_size': 0.015,
            },
            # Pre-allocate cuboid slots for floor + basket_a + basket_b + margin.
            # Default is 2 which is too small; 'primitive' is the cuRobo key for cuboids.
            'primitive': 10,
        }
        config = MotionPlannerCfg.create(
            robot=UR5_CONFIG,
            scene_model='collision_test.yml',
            collision_cache=collision_cache,
            max_goalset=TOPK_GRASPS,
            use_cuda_graph=_env_bool('PIPELINE_CUROBO_USE_CUDA_GRAPH', False),
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
    # close_extra: how far tool0 travels past its standard offset toward the object.
    # Standard: GRIPPER_TCP_Z_OFFSET = 0.1034 m (robotiq_2f_140 model depth).
    # Actual 2F-85 fingertip contact is ~0.082–0.090 m from tool0.
    #
    # Setting close_extra < GRIPPER_TCP_Z_OFFSET keeps tool0 back from TCP
    # (safe, fingers won't over-extend into the floor).
    # Setting close_extra > GRIPPER_TCP_Z_OFFSET drives tool0 past TCP
    # (deeper into the object — risks floor penetration for low grasps).
    #
    # Default 0.100 m → offset = 0.1034 - 0.100 = +0.0034 m: tool0 stops
    # 3.4 mm before TCP (very slightly conservative, no floor penetration risk).
    close_extra = _env_float('PIPELINE_CUROBO_GRASP_CLOSE_EXTRA_M', 0.100)
    if close_extra < 0.0:
        raise ValueError('PIPELINE_CUROBO_GRASP_CLOSE_EXTRA_M must be non-negative')
    offset = GRIPPER_TCP_Z_OFFSET - close_extra
    # Catch gross mis-configuration that would ram the wrist far past the target.
    if offset < -0.05:
        raise ValueError(
            'PIPELINE_CUROBO_GRASP_CLOSE_EXTRA_M too large: tool0 would be '
            'driven more than 5cm past the grasp point'
        )
    return offset


def _pick_pregrasp_standoff() -> float:
    """Metres to back the pre-grasp goal off the grasp pose along tool +z."""
    value = _env_float('PIPELINE_CUROBO_GRASP_PREGRASP_STANDOFF_M', 0.05)
    if value <= 0.0:
        raise ValueError(
            'PIPELINE_CUROBO_GRASP_PREGRASP_STANDOFF_M must be positive')
    return value


def _pick_approach_offsets() -> tuple:
    raw = os.environ.get(
        'PIPELINE_CUROBO_GRASP_APPROACH_OFFSETS', '-0.035,-0.06,-0.10')
    return _parse_nonzero_float_list(
        raw, 'PIPELINE_CUROBO_GRASP_APPROACH_OFFSETS')


def _pick_lift_offset() -> float:
    value = _env_float('PIPELINE_CUROBO_GRASP_LIFT_OFFSET', 0.10)
    if abs(value) < 1e-6:
        raise ValueError('PIPELINE_CUROBO_GRASP_LIFT_OFFSET must be nonzero')
    return value


def _parse_nonzero_float_list(raw: str, name: str) -> tuple:
    offsets = []
    for item in raw.split(','):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if abs(value) < 1e-6:
            raise ValueError(f'{name} values must be nonzero')
        offsets.append(value)
    if not offsets:
        raise ValueError(f'{name} must not be empty')
    return tuple(offsets)


def _pick_disable_collision_links(planner) -> list:
    raw = os.environ.get('PIPELINE_CUROBO_GRASP_DISABLE_COLLISION_LINKS')
    if raw is not None:
        return [item.strip() for item in raw.split(',') if item.strip()]
    try:
        links = (
            planner.kinematics.config.kinematics_config.grasp_contact_link_names
        )
    except Exception:
        links = None
    return list(links) if links else ['tool0']


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


def _reset_planner_seed(planner) -> None:
    # Only the cheap RNG-seed reset. We deliberately do NOT call reset_shape():
    # it dropped the solver's cached batch tensors, forcing a full (~200s)
    # re-setup on the next solve. It existed only to clear the warm-start state
    # plan_grasp corrupted (P1); since the pick path no longer calls plan_grasp,
    # there is nothing to clear and the re-setup is pure overhead.
    try:
        planner.reset_seed()
    except Exception:
        pass
