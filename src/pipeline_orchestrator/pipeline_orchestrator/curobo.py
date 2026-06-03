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
UR5_CONFIG = '/ros2_ws/src/pipeline_orchestrator/config/ur5_curobo.yml'
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
GRIPPER_TCP_Z_OFFSET = 0.1034


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

    def plan_pick(self, grasp_candidates, joint_states):
        """Plan approach -> grasp -> lift for ranked GraspGen TCP poses."""
        with self._cuda_lock:
            return self._plan_pick_locked(grasp_candidates, joint_states)

    def _plan_pick_locked(self, grasp_candidates, joint_states):
        try:
            t0 = time.perf_counter()
            self.update_joint_state(joint_states)
            self._update_world_from_tsdf()
            t_world = time.perf_counter()
            current = self._ros_js_to_curobo(joint_states)
            candidates = list(grasp_candidates)[:TOPK_GRASPS]
            if not candidates:
                self._logger.warn('CuRobo.plan_pick: no grasp candidates given.')
                return None

            # Stage 1 — plan to the PRE-GRASP goal set against the live TSDF.
            # This is exactly the phase plan_grasp solves reliably (plan_pose on
            # a goalset, picking goalset_index), but we stop here: we never call
            # plan_grasp's linear grasp phase, which fails in free space AND
            # leaves the planner's tool-pose criteria stuck on linear_motion —
            # the P1 corruption that poisoned every later goalset solve.
            standoff = _pick_pregrasp_standoff()
            pregrasp = self._grasps_to_goalset(candidates, tool_z_offset=-standoff)
            collision_links = _pick_disable_collision_links(self._planner)
            _reset_planner_seed(self._planner)
            self._planner.disable_link_collision(collision_links)
            try:
                approach = self._planner.plan_pose(pregrasp, current)
            finally:
                self._planner.enable_link_collision(collision_links)
            torch.cuda.synchronize()
            if not _result_success(approach):
                status = getattr(approach, 'status', 'unknown')
                self._logger.warn(
                    'CuRobo.plan_pick: pre-grasp goalset planning failed '
                    f'(candidates={len(candidates)} standoff={standoff:.3f}m '
                    f'status={status}).')
                return None

            t_pre = time.perf_counter()
            idx = _goalset_index(approach)
            if idx is None:
                idx = 0
            chosen = candidates[idx]
            approach_jt = interp_traj_to_ros(
                approach.get_interpolated_plan(),
                last_tstep=getattr(approach, 'interpolated_last_tstep', None),
            )

            # Stage 2 — descent (pre-grasp -> grasp) and Stage 3 — lift, planned
            # against a CLEARED world. The gripper is committing to / holding the
            # object, so its own fused voxels (which can't be removed from a
            # single TSDF voxel grid) must not block these intentional-contact
            # motions.
            grasp_tool = _tool_pose_from_grasp_tcp(_candidate_pose_4x4(chosen))
            lift_offset = _pick_lift_offset()
            lift_tool = grasp_tool.copy()
            lift_tool[:3, 3] += np.array([0.0, 0.0, lift_offset], dtype=np.float32)

            self._clear_collision_world()
            t_clear = time.perf_counter()
            grasp_jt = self._plan_pose_segment(
                grasp_tool, self._final_joint_state(approach_jt), 'grasp descent')
            if grasp_jt is None:
                return None
            t_grasp = time.perf_counter()
            lift_jt = self._plan_pose_segment(
                lift_tool, self._final_joint_state(grasp_jt), 'lift')
            if lift_jt is None:
                return None
            t_lift = time.perf_counter()

            self._logger.info(
                'CuRobo.plan_pick succeeded: '
                f'candidates={len(candidates)} chosen_goalset_index={idx} '
                f'standoff={standoff:.3f}m lift={lift_offset:.3f}m '
                f'(approach={len(approach_jt.points)}pts '
                f'grasp={len(grasp_jt.points)}pts lift={len(lift_jt.points)}pts) '
                f'timing[s]: world={t_world - t0:.2f} pregrasp={t_pre - t_world:.2f} '
                f'clear={t_clear - t_pre:.2f} descent={t_grasp - t_clear:.2f} '
                f'lift={t_lift - t_grasp:.2f} total={t_lift - t0:.2f}')
            return PickPlan(
                approach=approach_jt, grasp=grasp_jt, lift=lift_jt,
                goalset_index=idx)
        except Exception as exc:
            self._logger.error(f'CuRobo.plan_pick error: {exc}')
            return None

    def _plan_pose_segment(self, goal_mat, current_state, name):
        """Collision-off ``plan_pose`` to a single tool0 pose (4x4 in base_link).

        The caller is responsible for clearing the collision world first; this
        only resets the seed and plans. Returns a ROS ``JointTrajectory`` or
        ``None`` on failure.
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

    def _update_world_from_tsdf(self) -> bool:
        with self._lock:
            frame_count = self._frame_count
            last_update = self._last_world_update_frame

        required_frames = _min_tsdf_frames()
        if frame_count < required_frames:
            self._logger.warning(
                f'CuRobo: map not ready ({frame_count}/{required_frames} '
                'dual-camera frames); planning in current/free collision model.')
            return False
        if last_update == frame_count:
            return True

        torch.cuda.synchronize()
        voxel_grid = self._mapper.compute_esdf()
        try:
            self._planner.clear_scene_cache()
        except Exception:
            pass
        self._planner.update_world(SceneCfg(voxel=[voxel_grid]))
        torch.cuda.synchronize()
        with self._lock:
            self._last_world_update_frame = frame_count
        if self._enable_viz:
            self._cache_viz_tsdf()
        self._logger.info(
            f'CuRobo: updated TSDF collision world from {frame_count} frames.')
        return True

    def _clear_collision_world(self) -> None:
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
            }
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
    close_extra = _env_float('PIPELINE_CUROBO_GRASP_CLOSE_EXTRA_M', 0.110)
    if close_extra < 0.0:
        raise ValueError('PIPELINE_CUROBO_GRASP_CLOSE_EXTRA_M must be non-negative')
    offset = GRIPPER_TCP_Z_OFFSET - close_extra
    # A small negative offset is allowed: tool0 then sits just past the GraspGen
    # grasp point (driven deeper onto the object). Floor it to catch gross
    # misconfig that would ram the wrist well past the target.
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
