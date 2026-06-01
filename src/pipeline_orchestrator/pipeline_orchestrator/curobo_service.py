"""ROS2 service node wrapping the CuRobo motion planner.

Accepts two planning modes via PlanTrajectory.srv:

  pick mode  (grasp_poses[] non-empty):
    Calls CuRobo.plan_pick() with all top-K GraspGen TCP poses and the current
    joint state.  Returns:
      trajectory      — approach-to-grasp (approach + grasp phases concatenated)
      lift_trajectory — lift phase (execute after closing gripper)

  single-pose mode  (grasp_poses empty, grasp_pose set):
    Calls CuRobo.plan_trajectory() for place / home planning.  Returns:
      trajectory      — motion to the goal pose

CuRobo warmup runs in a background thread so the service is immediately
advertised while the planner initialises.
"""

import traceback
import threading
import os

import numpy as np
import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import JointState

from pipeline_orchestrator.curobo import (
    CuRobo,
    concat_trajectories,
    interp_traj_to_ros,
)
from riro_srvs.srv import PlanTrajectory


class CuRoboService(Node):
    """ROS2 service wrapper around the long-lived CuRobo planner instance."""

    JOINT_STATES_TOPIC = '/joint_states'

    def __init__(self) -> None:
        super().__init__('curobo_service')

        self.declare_parameter('service_name', '/curobo/plan_trajectory')
        self.declare_parameter('enable_viz', False)

        self._latest_joints = None
        self._curobo: CuRobo | None = None
        self._init_error = ''
        self._init_lock = threading.Lock()

        self.create_subscription(
            JointState,
            self.JOINT_STATES_TOPIC,
            self._cache_joints,
            10,
        )

        service_name = str(self.get_parameter('service_name').value)
        self.create_service(PlanTrajectory, service_name, self._handle_plan)
        self.get_logger().info(
            f'curobo_service advertised {service_name}; '
            'initialising CuRobo in background.'
        )

        self._init_thread = threading.Thread(
            target=self._init_curobo,
            name='curobo_initializer',
            daemon=True,
        )
        self._init_thread.start()

    # ── background init ───────────────────────────────────────────────────────

    def _init_curobo(self) -> None:
        self.get_logger().info('CuRobo initialisation started.')
        try:
            curobo = CuRobo(
                self,
                enable_viz=_as_bool(self.get_parameter('enable_viz').value),
            )
        except Exception as exc:
            with self._init_lock:
                self._init_error = f'{type(exc).__name__}: {exc}'
            self.get_logger().error(
                f'CuRobo initialisation failed: {self._init_error}')
            self.get_logger().debug(traceback.format_exc())
            return
        with self._init_lock:
            self._curobo = curobo
            latest_joints = self._latest_joints
        if latest_joints is not None:
            curobo.update_joint_state(latest_joints)
        self.get_logger().info('CuRobo initialisation complete; service is ready.')

    # ── joint state cache ─────────────────────────────────────────────────────

    def _cache_joints(self, msg: JointState) -> None:
        self._latest_joints = msg
        with self._init_lock:
            curobo = self._curobo
        if curobo is not None:
            curobo.update_joint_state(msg)

    # ── service handler ───────────────────────────────────────────────────────

    def _handle_plan(
        self,
        request: PlanTrajectory.Request,
        response: PlanTrajectory.Response,
    ) -> PlanTrajectory.Response:
        with self._init_lock:
            curobo = self._curobo
            init_error = self._init_error
        if init_error:
            response.success = False
            response.message = f'CuRobo init failed: {init_error}'
            return response
        if curobo is None:
            response.success = False
            response.message = 'CuRobo is still initializing.'
            return response

        joint_state = request.joint_state
        if not joint_state.name:
            joint_state = self._latest_joints
        if joint_state is None or not joint_state.name:
            response.success = False
            response.message = (
                'No joint state supplied and no /joint_states received yet.')
            return response

        try:
            if request.grasp_poses:
                return self._handle_pick(curobo, request, joint_state, response)
            return self._handle_single_pose(curobo, request, joint_state, response)
        except Exception as exc:
            response.success = False
            response.message = (
                f'CuRobo planning exception: {type(exc).__name__}: {exc}')
            self.get_logger().error(response.message)
            self.get_logger().debug(traceback.format_exc())
            return response

    def _handle_pick(self, curobo, request, joint_state, response):
        """Pick mode: plan_pick() → approach+grasp + lift."""
        candidates = _poses_to_candidates(request.grasp_poses)
        curobo.update_joint_state(joint_state)

        result = curobo.plan_pick(candidates, joint_state)
        if result is None:
            response.success = False
            response.message = 'CuRobo.plan_pick failed for all candidates.'
            return response

        approach_jt = interp_traj_to_ros(
            result.approach_interpolated_trajectory,
            last_tstep=getattr(result, 'approach_interpolated_last_tstep', None),
        )
        grasp_jt, n_preclose = _append_preclose_insertion_to_trajectory(
            interp_traj_to_ros(
                result.grasp_interpolated_trajectory,
                last_tstep=getattr(result, 'grasp_interpolated_last_tstep', None),
            )
        )
        lift_jt = interp_traj_to_ros(
            result.lift_interpolated_trajectory,
            last_tstep=getattr(result, 'lift_interpolated_last_tstep', None),
        )

        response.trajectory = concat_trajectories(approach_jt, grasp_jt)
        response.lift_trajectory = lift_jt
        curobo.pause_mapping(_pick_mapping_pause_sec(response.trajectory, lift_jt))
        response.success = True
        n_approach = len(approach_jt.points)
        n_grasp = len(grasp_jt.points)
        n_lift = len(lift_jt.points)
        response.message = (
            f'CuRobo pick planned: '
            f'approach={n_approach}pts grasp={n_grasp}pts '
            f'preclose_insert={n_preclose}pts lift={n_lift}pts'
        )
        return response

    def _handle_single_pose(self, curobo, request, joint_state, response):
        """Single-pose mode: plan_trajectory() for place / home."""
        curobo.update_joint_state(joint_state)
        trajectory = curobo.plan_trajectory(request.grasp_pose, joint_state)
        if trajectory is None or not trajectory.points:
            response.success = False
            response.message = 'CuRobo.plan_trajectory did not produce a trajectory.'
            return response
        response.success = True
        response.trajectory = trajectory
        response.message = (
            f'CuRobo trajectory planned: {len(trajectory.points)} points.')
        return response


# ── helpers ────────────────────────────────────────────────────────────────────

def _poses_to_candidates(ros_poses) -> list[dict]:
    """Convert geometry_msgs/Pose[] → list of {'pose_4x4': (4,4) ndarray}.

    Each pose represents the GraspGen TCP in base_link.  CuRobo.plan_pick
    applies the close-in bias internally when building the goal set.
    """
    candidates = []
    for pose in ros_poses:
        mat = np.eye(4, dtype=np.float32)
        mat[:3, :3] = R.from_quat([
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ]).as_matrix().astype(np.float32)
        mat[:3, 3] = [
            float(pose.position.x),
            float(pose.position.y),
            float(pose.position.z),
        ]
        candidates.append({'pose_4x4': mat})
    return candidates


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ('1', 'true', 'yes', 'on')
    return bool(value)


def _append_preclose_insertion_to_trajectory(trajectory):
    max_delta = _env_float('PIPELINE_GRASP_CLOSE_NUDGE_MAX_JOINT_DELTA_RAD', 0.04)
    if max_delta <= 0.0 or trajectory is None or len(trajectory.points) < 2:
        return trajectory, 0

    first = np.asarray(trajectory.points[0].positions, dtype=np.float32)
    prev = np.asarray(trajectory.points[-2].positions, dtype=np.float32)
    final = np.asarray(trajectory.points[-1].positions, dtype=np.float32)
    direction = final - prev
    peak = float(np.max(np.abs(direction))) if direction.size else 0.0
    if peak <= 1e-6:
        direction = final - first
        peak = float(np.max(np.abs(direction))) if direction.size else 0.0
    if peak <= 1e-6:
        return trajectory, 0

    duration = max(
        _env_float('PIPELINE_GRASP_CLOSE_NUDGE_DURATION_SEC', 0.60), 0.02)
    steps = max(int(_env_float('PIPELINE_GRASP_CLOSE_NUDGE_STEPS', 4)), 1)
    last_time = trajectory.points[-1].time_from_start
    start_sec = last_time.sec + last_time.nanosec * 1e-9
    full_delta = direction * (float(max_delta) / peak)

    from builtin_interfaces.msg import Duration as RosDuration
    from trajectory_msgs.msg import JointTrajectoryPoint

    for index in range(steps):
        alpha = float(index + 1) / float(steps)
        nudge = final + full_delta * alpha
        t_sec = start_sec + duration * alpha
        pt = JointTrajectoryPoint()
        pt.positions = [float(value) for value in nudge]
        pt.time_from_start = RosDuration(
            sec=int(t_sec),
            nanosec=int((t_sec % 1.0) * 1_000_000_000),
        )
        trajectory.points.append(pt)
    return trajectory, steps


def _pick_mapping_pause_sec(approach_and_grasp, lift) -> float:
    duration = _trajectory_duration_sec(approach_and_grasp)
    duration += _trajectory_duration_sec(lift)
    duration += _env_float('PIPELINE_GRASP_MAPPING_PAUSE_EXTRA_SEC', 2.0)
    return max(duration, 0.0)


def _trajectory_duration_sec(trajectory) -> float:
    if trajectory is None or not trajectory.points:
        return 0.0
    stamp = trajectory.points[-1].time_from_start
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == '':
        return float(default)
    return float(raw)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CuRoboService()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
