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

import numpy as np
import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import JointState

from pipeline_orchestrator.curobo import (
    BASE_FRAME,
    CuRobo,
    concat_trajectories,
    interp_traj_to_ros,
)
from sensor_msgs.msg import PointCloud2
from pipeline_orchestrator.pipeline_utils import as_bool as _as_bool
from pipeline_orchestrator.pipeline_utils import env_float as _env_float
from pipeline_orchestrator.pipeline_utils import make_xyz_cloud
from riro_srvs.srv import PlanTrajectory


class CuRoboService(Node):
    """ROS2 service wrapper around the long-lived CuRobo planner instance."""

    JOINT_STATES_TOPIC = '/joint_states'

    def __init__(self) -> None:
        super().__init__('curobo_service')

        self.declare_parameter('service_name', '/curobo/plan_trajectory')
        self.declare_parameter('enable_viz', False)
        self.declare_parameter('tsdf_voxels_topic', '/curobo/tsdf_voxels')
        self.declare_parameter('overhead_cloud_topic', '/curobo/overhead_cloud')
        self.declare_parameter('init_wait_sec', 120.0)

        self._latest_joints = None
        self._curobo: CuRobo | None = None
        self._init_error = ''
        self._init_done = False
        # Condition guards _curobo/_init_error/_init_done and lets a plan
        # request block until the background initialiser finishes, instead of
        # failing if it arrives before the planner is ready.
        self._init_cv = threading.Condition()
        self._init_wait_sec = float(self.get_parameter('init_wait_sec').value)

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

        self._tsdf_pub = None
        self._overhead_pub = None
        if _as_bool(self.get_parameter('enable_viz').value):
            self._tsdf_pub = self.create_publisher(
                PointCloud2,
                str(self.get_parameter('tsdf_voxels_topic').value),
                1,
            )
            self._overhead_pub = self.create_publisher(
                PointCloud2,
                str(self.get_parameter('overhead_cloud_topic').value),
                1,
            )
            self.create_timer(1.0, self._publish_tsdf_voxels)
            self.create_timer(1.0, self._publish_overhead_cloud)

        self._init_thread = threading.Thread(
            target=self._init_curobo,
            name='curobo_initializer',
            daemon=True,
        )
        self._init_thread.start()

    # ── background init ───────────────────────────────────────────────────────

    def _init_curobo(self) -> None:
        self.get_logger().info('CuRobo initialisation started.')
        curobo: CuRobo | None = None
        init_error = ''
        try:
            curobo = CuRobo(
                self,
                enable_viz=_as_bool(self.get_parameter('enable_viz').value),
            )
        except Exception as exc:
            init_error = f'{type(exc).__name__}: {exc}'
            self.get_logger().error(f'CuRobo initialisation failed: {init_error}')
            self.get_logger().debug(traceback.format_exc())

        # Publish the outcome and wake any plan request blocked in _wait_for_init.
        with self._init_cv:
            self._curobo = curobo
            self._init_error = init_error
            self._init_done = True
            latest_joints = self._latest_joints
            self._init_cv.notify_all()

        if curobo is None:
            return
        if latest_joints is not None:
            curobo.update_joint_state(latest_joints)
        self.get_logger().info('CuRobo initialisation complete; service is ready.')

    # ── joint state cache ─────────────────────────────────────────────────────

    def _cache_joints(self, msg: JointState) -> None:
        self._latest_joints = msg
        with self._init_cv:
            curobo = self._curobo
        if curobo is not None:
            curobo.update_joint_state(msg)

    # ── TSDF voxel publisher ──────────────────────────────────────────────────

    def _publish_tsdf_voxels(self) -> None:
        """Publish occupied TSDF voxel centers as a PointCloud2 (debug viz)."""
        if self._tsdf_pub is None:
            return
        with self._init_cv:
            curobo = self._curobo
        if curobo is None:
            return
        centers = curobo.get_tsdf_centers()
        if centers is None or len(centers) == 0:
            return
        cloud = make_xyz_cloud(
            centers,
            BASE_FRAME,
            self.get_clock().now().to_msg(),
        )
        self._tsdf_pub.publish(cloud)

    def _publish_overhead_cloud(self) -> None:
        """Publish the overhead camera's back-projected cloud (debug viz)."""
        if self._overhead_pub is None:
            return
        with self._init_cv:
            curobo = self._curobo
        if curobo is None:
            return
        points = curobo.get_point_clouds().get('overhead')
        if points is None or len(points) == 0:
            return
        cloud = make_xyz_cloud(
            points,
            BASE_FRAME,
            self.get_clock().now().to_msg(),
        )
        self._overhead_pub.publish(cloud)

    # ── init gating ───────────────────────────────────────────────────────────

    def _wait_for_init(self, timeout_sec: float):
        """Block until background CuRobo init finishes (or fails), or timeout.

        Returns ``(curobo, init_error)``.  A plan request that arrives before
        the planner is ready waits here instead of failing immediately, so the
        orchestrator can fire as soon as the service is advertised.  Init runs
        on its own thread and does not depend on the executor spinning, so
        parking the (single-threaded) executor here is safe; planning itself
        still runs on the executor thread once this returns.
        """
        with self._init_cv:
            if not self._init_done:
                self.get_logger().info(
                    'Plan request received before CuRobo finished initialising; '
                    f'waiting up to {timeout_sec:.1f}s.')
                self._init_cv.wait_for(
                    lambda: self._init_done, timeout=max(timeout_sec, 0.0))
            return self._curobo, self._init_error

    # ── service handler ───────────────────────────────────────────────────────

    def _handle_plan(
        self,
        request: PlanTrajectory.Request,
        response: PlanTrajectory.Response,
    ) -> PlanTrajectory.Response:
        curobo, init_error = self._wait_for_init(self._init_wait_sec)
        if init_error:
            response.success = False
            response.message = f'CuRobo init failed: {init_error}'
            return response
        if curobo is None:
            response.success = False
            response.message = (
                f'CuRobo still initializing after waiting '
                f'{self._init_wait_sec:.1f}s.')
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


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CuRoboService()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
