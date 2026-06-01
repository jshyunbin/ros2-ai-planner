"""cuRobo motion planning module.

Ported from yeina's proven integration (static-cuboid collision world +
plan_grasp 3-phase pick with per-candidate / approach-offset retry).

Key design decisions vs the previous TSDF-streaming approach:
  - Collision world: explicit static cuboids (table, baskets, bookshelf) only.
    The Mesh-from-pointcloud path requires warp.torch which is broken in this
    Docker, and yeina's own code already treats static-only as the reliable
    fallback.  Dynamic object collision is handled by GraspGen's inference.
  - plan_grasp() gives approach → grasp → lift as separate result phases, so
    the service can return them to the orchestrator for gripper interleaving.
  - Close-in bias: _effective_gripper_tcp_z_offset() offsets the cuRobo tool0
    goal slightly beyond the GraspGen TCP to compensate for 2F-85 vs 2F-140
    checkpoint geometry mismatch.
"""

import os

import numpy as np
import torch
from builtin_interfaces.msg import Duration as RosDuration
from scipy.spatial.transform import Rotation as R
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import GoalToolPose, JointState as CuRoboJointState
from curobo.scene import Cuboid
from curobo._src.geom.data.data_scene import SceneCfg

# Robot config: use the cuRobo built-in UR5 config (matches yeina).
# The custom ur5_curobo.yml has 8 tool0 gripper spheres extending to z=0.18m
# which cause false collision detections with scene cuboids during approach
# trajectory planning.  The built-in config has a minimal tool0 model.
UR5_CONFIG = 'ur5.yml'

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

# GraspGen TCP → UR5 tool0 distance along grasp +Z (robotiq_2f_140 checkpoint).
GRIPPER_TCP_Z_OFFSET = 0.1034

# Static scene obstacles in base_link (name, dims_xyz, pose_xyz_wxyz).
# Conservative approximations around the audited world-model poses.
_STATIC_CUBOIDS = (
    ('table_top',                   [1.40, 0.90, 0.04], [0.45,  0.0,  -0.02, 1.0, 0.0, 0.0, 0.0]),
    ('left_storage_basket_support', [0.34, 0.26, 0.12], [0.0,   0.55,  0.54, 1.0, 0.0, 0.0, 0.0]),
    ('right_storage_basket_support',[0.34, 0.26, 0.12], [0.0,  -0.55,  0.54, 1.0, 0.0, 0.0, 0.0]),
    ('workspace_basket_support',    [0.34, 0.30, 0.10], [0.55,  0.0,   0.46, 1.0, 0.0, 0.0, 0.0]),
    ('bookshelf_lower_body',        [0.20, 0.70, 0.55], [0.95, -0.30,  0.28, 1.0, 0.0, 0.0, 0.0]),
)


class CuRobo:
    """UR5 motion planner (static-cuboid world, yeina-style pick pipeline)."""

    def __init__(self, logger):
        self._logger = logger
        cfg = MotionPlannerCfg.create(
            robot=UR5_CONFIG,
            max_goalset=TOPK_GRASPS,
            collision_cache={'obb': 30},
        )
        self._planner = MotionPlanner(cfg)
        self._planner.warmup(enable_graph=True, num_warmup_iterations=5)
        self._planner.update_world(SceneCfg(cuboid=_static_cuboids()))
        self._logger.info('CuRobo: MotionPlanner ready (static-cuboid world).')

    # ── public API ────────────────────────────────────────────────────────────

    def plan_pick(self, grasp_candidates, joint_states):
        """3-phase pick plan: approach → grasp → lift.

        grasp_candidates: iterable of dicts with key 'pose_4x4' ((4,4) ndarray,
            GraspGen TCP pose in base_link, ordered best-first).
        Returns a cuRobo GraspPlanResult on success, None on failure.
        """
        try:
            current = self._ros_js_to_curobo(joint_states)
            candidates = list(grasp_candidates)
            collision_links = _pick_disable_collision_links(self._planner)
            last_status = 'unknown'
            for approach_offset in _pick_approach_offsets():
                for ci, candidate in enumerate(candidates):
                    goalset = self._grasps_to_goalset([candidate])
                    result = self._planner.plan_grasp(
                        grasp_poses=goalset,
                        current_state=current,
                        grasp_approach_offset=approach_offset,
                        grasp_approach_in_tool_frame=True,
                        grasp_lift_axis='z',
                        grasp_lift_offset=_env_float(
                            'PIPELINE_CUROBO_GRASP_LIFT_OFFSET', 0.10),
                        grasp_lift_in_tool_frame=False,
                        plan_approach_to_grasp=True,
                        plan_grasp_to_lift=True,
                        disable_collision_links=collision_links,
                    )
                    if _result_success(result):
                        self._logger.info(
                            f'CuRobo.plan_pick succeeded: '
                            f'candidate={ci} approach={approach_offset:.3f}m'
                        )
                        return result
                    last_status = getattr(result, 'status', 'unknown')
                    self._logger.warn(
                        f'CuRobo.plan_pick: failed ci={ci} '
                        f'approach={approach_offset:.3f}m status={last_status}'
                    )
            self._logger.warn(
                f'CuRobo.plan_pick: all attempts exhausted ({last_status})')
            return None
        except Exception as exc:
            self._logger.error(f'CuRobo.plan_pick error: {exc}')
            return None

    def plan_trajectory(self, goal_pose, joint_states):
        """Single-segment plan for place / home.

        goal_pose: geometry_msgs/Pose (tool0 target in base_link).
        Returns JointTrajectory or None.
        """
        try:
            current = self._ros_js_to_curobo(joint_states)
            p, o = goal_pose.position, goal_pose.orientation
            goal = GoalToolPose(
                tool_frames=self._planner.tool_frames,
                position=torch.tensor(
                    [p.x, p.y, p.z],
                    device='cuda', dtype=torch.float32).view(1, 1, 1, 1, 3),
                quaternion=torch.tensor(
                    [o.w, o.x, o.y, o.z],
                    device='cuda', dtype=torch.float32).view(1, 1, 1, 1, 4),
            )
            result = self._planner.plan_pose(goal, current)
            if not _result_success(result):
                self._logger.warn(
                    f'CuRobo.plan_trajectory failed: '
                    f'{getattr(result, "status", "unknown")}'
                )
                return None
            return interp_traj_to_ros(result.get_interpolated_plan())
        except Exception as exc:
            self._logger.error(f'CuRobo.plan_trajectory error: {exc}')
            return None

    # ── private helpers ───────────────────────────────────────────────────────

    def _ros_js_to_curobo(self, joint_states) -> CuRoboJointState:
        name_to_pos = dict(zip(joint_states.name, joint_states.position))
        missing = [j for j in JOINT_NAMES if j not in name_to_pos]
        if missing:
            raise ValueError(f'CuRobo: /joint_states missing joints: {missing}')
        pos = torch.tensor(
            [[name_to_pos[j] for j in JOINT_NAMES]],
            device='cuda', dtype=torch.float32,
        )
        return CuRoboJointState.from_position(pos, joint_names=list(JOINT_NAMES))

    def _grasps_to_goalset(self, grasp_candidates) -> GoalToolPose:
        """Top-K candidate dicts (pose_4x4) → GoalToolPose at tool0.

        Applies the close-in bias: drives tool0 goal close to the GraspGen TCP
        so the 2F-85 gripper (shorter than 2F-140) closes at the right depth.
        """
        mats = np.stack([g['pose_4x4'] for g in grasp_candidates])
        t_tool_grasp = np.eye(4)
        t_tool_grasp[2, 3] = _effective_gripper_tcp_z_offset()
        inv = np.linalg.inv(t_tool_grasp)
        tool = np.stack([m @ inv for m in mats])
        pos = tool[:, :3, 3]
        quat_xyzw = R.from_matrix(tool[:, :3, :3]).as_quat()   # scipy: xyzw
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


# ── module-level helpers (also used by curobo_service.py) ─────────────────────

def interp_traj_to_ros(interp_traj, dt: float = INTERP_DT,
                       time_offset: float = 0.0) -> JointTrajectory:
    """Convert a cuRobo interpolated trajectory to a ROS JointTrajectory.

    time_offset: seconds to add to every time_from_start (for concatenation).
    """
    traj = interp_traj
    # Squeeze the batch dimension cuRobo pads onto plan results.
    if hasattr(traj, 'squeeze'):
        traj = traj.squeeze(0)
    pos_t = traj.position
    while pos_t.dim() > 2:
        pos_t = pos_t[0]
    positions = pos_t.cpu().numpy()

    velocities = None
    vel_raw = getattr(traj, 'velocity', None)
    if vel_raw is not None:
        while vel_raw.dim() > 2:
            vel_raw = vel_raw[0]
        velocities = vel_raw.cpu().numpy()

    jt = JointTrajectory()
    jt.joint_names = list(JOINT_NAMES)
    for i, pos in enumerate(positions):
        pt = JointTrajectoryPoint()
        pt.positions = [float(x) for x in pos]
        if velocities is not None:
            pt.velocities = [float(x) for x in velocities[i]]
        t_sec = time_offset + (i + 1) * dt
        pt.time_from_start = RosDuration(
            sec=int(t_sec), nanosec=int((t_sec % 1.0) * 1_000_000_000))
        jt.points.append(pt)
    return jt


def concat_trajectories(
    traj_a: JointTrajectory, traj_b: JointTrajectory
) -> JointTrajectory:
    """Append traj_b after traj_a, adjusting traj_b timestamps to be monotonic."""
    combined = JointTrajectory()
    combined.joint_names = traj_a.joint_names
    combined.points = list(traj_a.points)
    if traj_a.points:
        last = traj_a.points[-1].time_from_start
        offset = last.sec + last.nanosec * 1e-9
    else:
        offset = 0.0
    for pt in traj_b.points:
        t = pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9 + offset
        new_pt = JointTrajectoryPoint()
        new_pt.positions = pt.positions
        new_pt.velocities = pt.velocities
        new_pt.accelerations = pt.accelerations
        new_pt.time_from_start = RosDuration(
            sec=int(t), nanosec=int((t % 1.0) * 1_000_000_000))
        combined.points.append(new_pt)
    return combined


# ── private module helpers ────────────────────────────────────────────────────

def _static_cuboids():
    return [Cuboid(name=n, dims=d, pose=p) for n, d, p in _STATIC_CUBOIDS]


def _effective_gripper_tcp_z_offset() -> float:
    """Close-in bias: reduce the TCP→tool0 back-off to compensate 2F-85 geometry.

    Default close_extra=0.1025 → effective_offset=0.0009 m, meaning cuRobo's
    tool0 goal is placed almost at the GraspGen TCP, driving the 2F-85 fingers
    ~10 cm deeper than the 2F-140 checkpoint originally intended.
    """
    close_extra = _env_float('PIPELINE_CUROBO_GRASP_CLOSE_EXTRA_M', 0.1025)
    if close_extra < 0.0:
        raise ValueError('PIPELINE_CUROBO_GRASP_CLOSE_EXTRA_M must be non-negative')
    offset = GRIPPER_TCP_Z_OFFSET - close_extra
    if offset <= 0.0:
        raise ValueError(
            'PIPELINE_CUROBO_GRASP_CLOSE_EXTRA_M must be smaller than '
            'GRIPPER_TCP_Z_OFFSET'
        )
    return offset


def _pick_approach_offsets() -> tuple:
    """Ordered list of approach pre-grasp distances to try (negative = along -Z)."""
    raw = os.environ.get('PIPELINE_CUROBO_GRASP_APPROACH_OFFSETS', '-0.035,-0.06,-0.10')
    offsets = []
    for item in raw.split(','):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if abs(value) < 1e-6:
            raise ValueError(
                'PIPELINE_CUROBO_GRASP_APPROACH_OFFSETS values must be nonzero')
        offsets.append(value)
    if not offsets:
        raise ValueError('PIPELINE_CUROBO_GRASP_APPROACH_OFFSETS must not be empty')
    return tuple(offsets)


def _pick_disable_collision_links(planner) -> list:
    raw = os.environ.get('PIPELINE_CUROBO_GRASP_DISABLE_COLLISION_LINKS')
    if raw is not None:
        return [item.strip() for item in raw.split(',') if item.strip()]
    try:
        links = (
            planner.kinematics.config.kinematics_config.grasp_contact_link_names
        )
    except Exception:  # noqa: BLE001
        links = None
    return list(links) if links else ['tool0']


def _result_success(result) -> bool:
    if result is None:
        return False
    s = getattr(result, 'success', None)
    if s is None:
        return False
    try:
        return bool(s.any())
    except AttributeError:
        return bool(s)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == '':
        return float(default)
    return float(raw)
