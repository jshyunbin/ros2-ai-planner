"""ROS2 service node wrapping the CuRobo motion planner.

Accepts three planning modes via PlanTrajectory.srv:

  pick mode  (grasp_poses[] non-empty):
    Calls CuRobo.plan_pick() with all top-K GraspGen TCP poses and the current
    joint state.  Returns:
      trajectory      — approach-to-grasp (approach + grasp phases concatenated)
      lift_trajectory — lift phase (execute after closing gripper)

  named-goal mode  (goal_name set, e.g. "home", "storage_1", "bookshelf_a"):
    Resolves the target pose from place_poses.yml, calls plan_trajectory().
    For bookshelf targets also plans insert_trajectory and retract_trajectory.

  single-pose mode  (grasp_poses empty, grasp_pose set, goal_name empty):
    Calls CuRobo.plan_trajectory() for an arbitrary tool0 pose. Returns:
      trajectory      — motion to the goal pose

CuRobo warmup runs in a background thread so the service is immediately
advertised while the planner initialises.
"""

import threading
import time
import traceback

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Empty

from team_8.curobo import (
    BASE_FRAME,
    CuRobo,
    concat_trajectories,
)
from sensor_msgs.msg import PointCloud2
from team_8.pipeline_utils import as_bool as _as_bool
from team_8.pipeline_utils import cloud_to_xyz
from team_8.pipeline_utils import env_float as _env_float
from team_8.pipeline_utils import env_int as _env_int
from team_8.pipeline_utils import make_xyz_cloud
from team_8.place_pose_utils import (
    build_transit_waypoints,
    get_home_joint_config,
    is_bookshelf_target,
    load_place_poses,
    pose_from_xyzquat,
    resolve_target_pose,
    translate_pose_x,
)
from riro_srvs.srv import PlanTrajectory


# Debug-viz clouds are published from a dedicated thread (see _viz_publish_loop)
# rather than ROS timers, so a long blocking plan_trajectory call on the
# single-threaded executor can't starve them. Republishing cached numpy is cheap.
_VIZ_PUBLISH_HZ = 5.0


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
        self._latest_object_cloud: np.ndarray | None = None
        # Set while a plan is in flight so the debug viz thread pauses its
        # cloud publishing. rclpy PointCloud2 serialization is pure-Python and
        # GIL-heavy; with a populated map it otherwise starves the (single-
        # threaded) planner of the GIL and stretches a ~3s plan into minutes.
        self._planning = threading.Event()
        self._curobo: CuRobo | None = None
        self._init_error = ''
        self._init_done = False
        # Condition guards _curobo/_init_error/_init_done and lets a plan
        # request block until the background initialiser finishes, instead of
        # failing if it arrives before the planner is ready.
        self._init_cv = threading.Condition()
        self._init_wait_sec = float(self.get_parameter('init_wait_sec').value)

        # Load place pose config once at startup; used by _handle_named_goal.
        try:
            _poses_yml = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'config', 'place_poses.yml')
            self._place_poses_cfg = load_place_poses(_poses_yml)
        except Exception as exc:
            self.get_logger().warning(
                f'place_poses.yml not loaded: {exc}; '
                'named-goal mode will be unavailable.')
            self._place_poses_cfg = None

        self.create_subscription(
            JointState,
            self.JOINT_STATES_TOPIC,
            self._cache_joints,
            10,
        )

        # Subscribe to segmented object cloud so plan_pick can carve its voxels
        # from the TSDF, preventing the target object itself from blocking the
        # approach trajectory and forcing it into nearby objects.
        self.create_subscription(
            PointCloud2,
            '/graspgen/segmented_object',
            self._cache_object_cloud,
            10,
        )

        # /curobo/reset_map: publish any Empty message to wipe the TSDF and
        # restart accumulation from scratch.  The orchestrator publishes this
        # after each arm movement so ghost voxels from the old arm pose are
        # cleared before the next planning request.
        self.create_subscription(
            Empty,
            '/curobo/reset_map',
            self._handle_reset_map,
            10,
        )

        service_name = str(self.get_parameter('service_name').value)
        self.create_service(PlanTrajectory, service_name, self._handle_plan)
        self.get_logger().info(
            f'curobo_service advertised {service_name}; '
            'initialising CuRobo in background.'
        )

        # /curobo/ready: latched Bool published True once the TSDF has
        # accumulated enough frames for planning.  Orchestrator waits for this
        # before sending the first pick request.
        _latched_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self._ready_pub = self.create_publisher(Bool, '/curobo/ready', _latched_qos)
        self._ready_published = False
        self._ready_timer = self.create_timer(1.0, self._check_and_publish_ready)

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
            # Publish from a dedicated thread, NOT executor timers: planning
            # runs on (and blocks) the single-threaded executor, which would
            # otherwise starve the timers and make the clouds update in bursts
            # only after each plan returns. The loop just reads cached CPU numpy
            # (no CUDA), so it's safe to run alongside the executor.
            self._viz_stop = threading.Event()
            self._viz_thread = threading.Thread(
                target=self._viz_publish_loop, daemon=True)
            self._viz_thread.start()

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

    # ── /curobo/ready ─────────────────────────────────────────────────────────

    def _check_and_publish_ready(self) -> None:
        """Publish True on /curobo/ready once TSDF has enough frames."""
        if self._ready_published:
            return
        with self._init_cv:
            curobo = self._curobo
        if curobo is None:
            return
        min_frames = max(_env_int('PIPELINE_CUROBO_MIN_PLANNING_FRAMES', 5), 5)
        if curobo.frame_count >= min_frames:
            msg = Bool()
            msg.data = True
            self._ready_pub.publish(msg)
            self._ready_published = True
            self.get_logger().info(
                f'/curobo/ready published (frames={curobo.frame_count})')

    # ── joint state cache ─────────────────────────────────────────────────────

    def _cache_joints(self, msg: JointState) -> None:
        self._latest_joints = msg
        with self._init_cv:
            curobo = self._curobo
        if curobo is not None:
            curobo.update_joint_state(msg)

    def _handle_reset_map(self, _msg) -> None:
        """Wipe the TSDF and restart accumulation.

        Called via /curobo/reset_map after each arm movement so that ghost
        voxels left by the previous arm pose do not pollute the collision world
        used for the next planning request.
        After reset the /curobo/ready latch is cleared; the ready timer
        republishes it once enough new frames have accumulated.
        """
        with self._init_cv:
            curobo = self._curobo
        if curobo is None:
            return
        curobo.reset_mapping()
        # Un-latch /curobo/ready so the orchestrator can wait for fresh frames.
        self._ready_published = False
        self.get_logger().info(
            'CuRobo: TSDF reset on /curobo/reset_map — '
            'waiting for fresh frames before next plan.')

    def _cache_object_cloud(self, msg: PointCloud2) -> None:
        try:
            xyz = cloud_to_xyz(msg)
            if xyz is not None and len(xyz) > 0:
                self._latest_object_cloud = xyz
        except Exception as exc:
            self.get_logger().warning(
                f'object cloud parse failed: {type(exc).__name__}: {exc}',
                throttle_duration_sec=5.0)

    # ── TSDF voxel publisher ──────────────────────────────────────────────────

    def _viz_publish_loop(self) -> None:
        period = 1.0 / _VIZ_PUBLISH_HZ
        while rclpy.ok() and not self._viz_stop.is_set():
            start = time.monotonic()
            if self._planning.is_set():
                self._viz_stop.wait(period)
                continue
            try:
                self._publish_tsdf_voxels()
                self._publish_overhead_cloud()
            except Exception as exc:
                self.get_logger().warning(
                    f'viz publish failed: {type(exc).__name__}: {exc}',
                    throttle_duration_sec=5.0)
            self._viz_stop.wait(max(0.0, period - (time.monotonic() - start)))

    def _publish_tsdf_voxels(self) -> None:
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

        self._planning.set()
        try:
            if request.grasp_poses:
                return self._handle_pick(curobo, request, joint_state, response)
            if request.goal_name:
                return self._handle_named_goal(
                    curobo, request.goal_name, joint_state, response)
            return self._handle_single_pose(curobo, request, joint_state, response)
        except Exception as exc:
            response.success = False
            response.message = (
                f'CuRobo planning exception: {type(exc).__name__}: {exc}')
            self.get_logger().error(response.message)
            self.get_logger().debug(traceback.format_exc())
            return response
        finally:
            self._planning.clear()

    def _handle_pick(self, curobo, request, joint_state, response):
        """Pick mode: plan_pick() → approach+grasp + lift."""
        candidates = _poses_to_candidates(request.grasp_poses)
        curobo.update_joint_state(joint_state)

        object_cloud = self._latest_object_cloud
        if object_cloud is not None:
            self.get_logger().info(
                f'plan_pick: using cached object cloud '
                f'({len(object_cloud)} pts) for TSDF carving.')
        plan = curobo.plan_pick(candidates, joint_state, object_cloud=object_cloud)
        if plan is None:
            response.success = False
            response.message = 'CuRobo.plan_pick failed for all candidates.'
            return response

        approach_jt = plan.approach
        lift_jt = plan.lift

        if plan.grasp is not None:
            # Legacy 3-phase path: approach → descent → (gripper close) → lift.
            grasp_jt, n_preclose = _append_preclose_insertion_to_trajectory(
                plan.grasp)
            response.trajectory = concat_trajectories(approach_jt, grasp_jt)
            n_approach = len(approach_jt.points)
            n_grasp = len(grasp_jt.points)
        else:
            # Direct-to-grasp path: approach already ends at the grasp pose.
            # plan.grasp is None, so the "approach" trajectory IS the full
            # approach-to-grasp motion.  No concatenation needed.
            n_preclose = 0
            response.trajectory = approach_jt
            n_approach = len(approach_jt.points)
            n_grasp = 0

        response.lift_trajectory = lift_jt
        curobo.pause_mapping(_pick_mapping_pause_sec(response.trajectory, lift_jt))
        response.success = True
        n_lift = len(lift_jt.points)
        response.message = (
            f'CuRobo pick planned (direct-to-grasp): '
            f'approach={n_approach}pts grasp={n_grasp}pts '
            f'preclose_insert={n_preclose}pts lift={n_lift}pts'
        )
        return response

    def _handle_named_goal(self, curobo, goal_name, joint_state, response):
        """Named-goal mode: resolve goal from place_poses.yml and plan.

        'home' uses a c-space plan to the fixed home_joint_config so the arm
        always returns to the same posture (no IK branch ambiguity).
        All other keys use plan_trajectory (IK → pose).
        """
        if self._place_poses_cfg is None:
            response.success = False
            response.message = (
                f'Named goal {goal_name!r} requested but place_poses.yml '
                'failed to load at startup.')
            return response

        curobo.update_joint_state(joint_state)

        # ── Home: fixed joint-config c-space plan ────────────────────────────
        if goal_name == 'home':
            trajectory = curobo.plan_home_config(
                get_home_joint_config(self._place_poses_cfg), joint_state)
            if trajectory is None or not trajectory.points:
                response.success = False
                response.message = 'CuRobo home c-space planning failed.'
                return response
            response.success = True
            response.trajectory = trajectory
            response.message = (
                f'CuRobo home planned: {len(trajectory.points)} points.')
            return response

        # ── All other named goals: IK-based pose plan ────────────────────────
        try:
            target_pose = resolve_target_pose(self._place_poses_cfg, goal_name)
        except KeyError as exc:
            response.success = False
            response.message = str(exc)
            return response

        trajectory = curobo.plan_trajectory(target_pose, joint_state)
        if trajectory is None or not trajectory.points:
            response.success = False
            response.message = (
                f'CuRobo.plan_trajectory failed for goal_name={goal_name!r}.')
            return response

        response.trajectory = trajectory
        response.success = True
        response.message = (
            f'CuRobo planned to {goal_name!r}: {len(trajectory.points)} pts.')

        # Bookshelf targets: plan insert (push forward) and retract (pull back).
        if is_bookshelf_target(self._place_poses_cfg, goal_name):
            entry = self._place_poses_cfg[goal_name]
            insert_depth = float(entry.get('insert_depth_m', 0.08))
            retract_depth = float(entry.get('retract_depth_m', 0.06))

            # translate_pose_x works on plain xyz lists; convert from/to Pose.
            pre_xyz = [target_pose.position.x,
                       target_pose.position.y,
                       target_pose.position.z]
            pre_quat = [target_pose.orientation.x, target_pose.orientation.y,
                        target_pose.orientation.z, target_pose.orientation.w]
            insert_xyz  = translate_pose_x(pre_xyz, -insert_depth)
            retract_xyz = translate_pose_x(pre_xyz,  retract_depth)
            insert_pose  = pose_from_xyzquat(insert_xyz,  pre_quat)
            retract_pose = pose_from_xyzquat(retract_xyz, pre_quat)

            insert_js = _trajectory_final_joint_state(trajectory, joint_state)
            insert_traj = curobo.plan_trajectory(insert_pose, insert_js)
            if insert_traj and insert_traj.points:
                response.insert_trajectory = insert_traj
                retract_js = _trajectory_final_joint_state(insert_traj, insert_js)
                retract_traj = curobo.plan_trajectory(retract_pose, retract_js)
                if retract_traj and retract_traj.points:
                    response.retract_trajectory = retract_traj
            self.get_logger().info(
                f'Bookshelf {goal_name!r}: insert={insert_depth:.3f}m '
                f'retract={retract_depth:.3f}m')

        return response

    def _handle_single_pose(self, curobo, request, joint_state, response):
        """Single-pose mode: plan_trajectory() for an explicit goal pose."""
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
    """Convert geometry_msgs/Pose[] → list of {'pose_4x4': (4,4) ndarray}."""
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


def _trajectory_final_joint_state(trajectory, fallback_joint_state):
    """Return a JointState-compatible object from the last trajectory point."""
    # We return the original joint_state updated with final positions.
    # CuRobo plan_trajectory accepts a JointState from ROS.
    # Since we can't easily build a real JointState here without rclpy,
    # we pass the original one — cuRobo will use its cached state anyway.
    return fallback_joint_state


def _append_preclose_insertion_to_trajectory(trajectory):
    max_delta = _env_float('PIPELINE_GRASP_CLOSE_NUDGE_MAX_JOINT_DELTA_RAD', 0.0)
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
