import threading

# Light-weight ROS2 message types (always available in ROS2 environment)
try:
    from sensor_msgs.msg import Image, CameraInfo
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from builtin_interfaces.msg import Duration as RosDuration
    from rclpy.qos import qos_profile_sensor_data
except ImportError:
    Image = CameraInfo = JointTrajectory = JointTrajectoryPoint = RosDuration = None
    qos_profile_sensor_data = None

# Heavy deps — imported lazily so tests can mock them via patch.multiple
try:
    import numpy as np
    import torch
    import rclpy.duration
    from rclpy.node import Node
    from rclpy.time import Time
    from tf2_ros import Buffer, TransformListener, LookupException, ExtrapolationException
    from cv_bridge import CvBridge
    from curobo.perception import Mapper, MapperCfg, FilterDepth, RobotSegmenter
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.types import CameraObservation, Pose, JointState as CuRoboJointState, GoalToolPose
    from curobo._src.geom.types import SceneCfg, VoxelGrid
    # RobotSegmenter.from_robot_file does not forward ops_dtype to __init__,
    # so build Kinematics ourselves to override the default (bfloat16) which
    # mismatches the float32 robot_spheres tensor at runtime.
    from curobo._src.robot.kinematics.kinematics import Kinematics
    from curobo._src.types.robot import RobotCfg
    from curobo._src.util_file import get_robot_configs_path, join_path, load_yaml
    _HEAVY_DEPS_AVAILABLE = True
except ImportError:
    _HEAVY_DEPS_AVAILABLE = False
    # Provide stub names so that patch.multiple targets exist at module level.
    # Constructors (Mapper, MotionPlanner, Buffer, TransformListener, FilterDepth)
    # must be MagicMock instances so they are callable and return MagicMocks.
    # Config classes (MapperCfg, MotionPlannerCfg) are also instances so that
    # attribute access like MotionPlannerCfg.create(...) works via MagicMock.
    import unittest.mock as _mock

    def _make_mock_class():
        """Return a MagicMock that is callable (acts like a class)."""
        m = _mock.MagicMock()
        return m

    np = _mock.MagicMock()
    torch = _mock.MagicMock()
    Buffer = _make_mock_class()
    TransformListener = _make_mock_class()
    Mapper = _make_mock_class()
    MapperCfg = _make_mock_class()
    FilterDepth = _make_mock_class()
    RobotSegmenter = _make_mock_class()
    Kinematics = _make_mock_class()
    RobotCfg = _make_mock_class()
    get_robot_configs_path = _mock.MagicMock(return_value='')
    join_path = _mock.MagicMock(return_value='')
    load_yaml = _mock.MagicMock(return_value={})
    MotionPlanner = _make_mock_class()
    MotionPlannerCfg = _make_mock_class()
    # Types used inside methods
    CameraObservation = _make_mock_class()
    Pose = _make_mock_class()
    CuRoboJointState = _make_mock_class()
    GoalToolPose = _make_mock_class()
    SceneCfg = _make_mock_class()
    VoxelGrid = _make_mock_class()
    Time = _make_mock_class()

    class _RclpyDurationStub:
        class Duration:
            def __init__(self, **kwargs):
                pass
    rclpy = _mock.MagicMock()
    rclpy.duration = _RclpyDurationStub()

    class _CvBridgeStub:
        def imgmsg_to_cv2(self, msg, desired_encoding='passthrough'):
            import numpy
            return numpy.zeros((480, 640), dtype=numpy.uint16)
    CvBridge = _CvBridgeStub

OVERHEAD_DEPTH_TOPIC = '/camera/camera/depth/color/image_raw'
OVERHEAD_INFO_TOPIC  = '/camera/camera/depth/color/camera_info'
WRIST_DEPTH_TOPIC    = '/wrist_camera/wrist_camera/depth/color/image_raw'
WRIST_INFO_TOPIC     = '/wrist_camera/wrist_camera/depth/color/camera_info'
OVERHEAD_FRAME = 'camera_color_optical_frame'
WRIST_FRAME    = 'wrist_camera_color_optical_frame'
WORLD_FRAME    = 'world'
MIN_FRAMES     = 5
UR5_CONFIG     = '/ros2_ws/src/pipeline_orchestrator/config/ur5_curobo.yml'
JOINT_NAMES    = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
]


class CuRobo:
    """Dual-RGBD TSDF fusion + collision-aware UR5 motion planning via cuRoboV2."""

    def __init__(self, node, enable_viz=False):
        self._node = node
        self._logger = node.get_logger()
        self._lock = threading.Lock()
        self._frame_count = 0

        # When enabled, _on_depth unprojects each frame into a world-frame
        # point cloud and plan_trajectory snapshots the reconstructed TSDF
        # surface voxels, so a viser front-end can render the perception
        # state. Off by default to keep the orchestrator's hot path lean.
        self._enable_viz = enable_viz
        self._point_clouds: dict = {}   # cam_id -> (N, 3) np world points
        self._tsdf_centers = None       # (M, 3) np surface voxel centres

        self._cam_depth: dict = {}
        self._cam_intrinsics: dict = {}
        self._cam_pose: dict = {}
        self._latest_joints = None   # sensor_msgs/JointState; needed by segmenter

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
            # 1.0 = cuRobo default, fine for offline static datasets, but for
            # live noisy depth (Gazebo RealSense) it pins every noisy pixel
            # in forever. 0.95 fades stale single-view noise over ~14 frames.
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
        # ESDF never marks the arm itself as an obstacle. Build Kinematics
        # manually to force ops_dtype=float32 -- the from_robot_file factory
        # leaves it at bfloat16 default which mismatches robot_spheres.
        robot_yaml = load_yaml(join_path(get_robot_configs_path(), UR5_CONFIG))
        robot_cfg = RobotCfg.create(robot_yaml)
        self._segmenter = RobotSegmenter(
            Kinematics(robot_cfg.kinematics),
            distance_threshold=0.05,
            use_cuda_graph=False,
            ops_dtype=torch.float32,
        )

        # Gazebo's realsense plugin publishes camera streams with BEST_EFFORT
        # reliability; subscribers must match (qos_profile_sensor_data) or no
        # data ever arrives.
        #
        # Note CuRobo only subscribes to the high-bandwidth perception streams
        # it must fuse continuously (depth/info); /joint_states is owned by the
        # orchestrator and pushed in via update_joint_state(), and trajectory
        # deployment is the orchestrator's job too.
        node.create_subscription(Image, OVERHEAD_DEPTH_TOPIC,
                                  lambda msg: self._on_depth(msg, 'overhead', OVERHEAD_FRAME),
                                  qos_profile_sensor_data)
        node.create_subscription(CameraInfo, OVERHEAD_INFO_TOPIC,
                                  lambda msg: self._on_info(msg, 'overhead'),
                                  qos_profile_sensor_data)
        node.create_subscription(Image, WRIST_DEPTH_TOPIC,
                                  lambda msg: self._on_depth(msg, 'wrist', WRIST_FRAME),
                                  qos_profile_sensor_data)
        node.create_subscription(CameraInfo, WRIST_INFO_TOPIC,
                                  lambda msg: self._on_info(msg, 'wrist'),
                                  qos_profile_sensor_data)

        self._planner = self._build_planner()
        self._logger.info('CuRobo: ready.')

    def update_joint_state(self, msg):
        """Feed the latest /joint_states (called by the orchestrator).

        The robot segmenter needs the live joints to mask the arm out of each
        depth frame; the orchestrator owns the subscription and pushes them here.
        """
        with self._lock:
            self._latest_joints = msg

    # ── viz accessors (thread-safe snapshots for a viser front-end) ──────────

    @property
    def frame_count(self) -> int:
        """Number of dual-camera frames integrated into the TSDF so far."""
        with self._lock:
            return self._frame_count

    def get_point_clouds(self) -> dict:
        """Latest per-camera world-frame point clouds: {cam_id: (N, 3) ndarray}.

        Empty until enable_viz is set and depth frames have arrived.
        """
        with self._lock:
            return dict(self._point_clouds)

    def get_tsdf_centers(self):
        """Reconstructed TSDF surface voxel centres as (M, 3) ndarray, or None.

        Populated by plan_trajectory once the ESDF has been computed.
        """
        with self._lock:
            return self._tsdf_centers

    def get_latest_joints(self):
        """Most recent sensor_msgs/JointState, or None."""
        with self._lock:
            return self._latest_joints

    def _cache_viz_cloud(self, cam_id, depth, t, r, K):
        """Unproject depth to a world-frame point cloud and cache it (numpy).

        Mirrors the transform the Mapper does internally so the viser cloud
        lines up with the reconstructed TSDF. Best-effort: failures never
        disrupt perception.
        """
        from pipeline_orchestrator.live_viz_helpers import depth_to_xyz
        try:
            xyz_cam = depth_to_xyz(depth, K)
            qw, qx, qy, qz = float(r.w), float(r.x), float(r.y), float(r.z)
            R = torch.tensor([
                [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
                [2*(qx*qy + qw*qz),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
                [2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx),     1 - 2*(qx*qx + qy*qy)],
            ], dtype=torch.float32, device=xyz_cam.device)
            t_vec = torch.tensor(
                [t.x, t.y, t.z], dtype=torch.float32, device=xyz_cam.device)
            xyz_world = (xyz_cam @ R.T + t_vec).cpu().numpy()
        except Exception as exc:
            self._logger.warning(
                f'CuRobo: viz cloud failed for {cam_id}: '
                f'{type(exc).__name__}: {exc}')
            return
        with self._lock:
            self._point_clouds[cam_id] = xyz_world

    def _cache_viz_tsdf(self):
        """Snapshot the reconstructed TSDF surface voxel centres (numpy)."""
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
        K = torch.tensor([
            [msg.k[0], 0.0,      msg.k[2]],
            [0.0,      msg.k[4], msg.k[5]],
            [0.0,      0.0,      1.0     ],
        ], dtype=torch.float32, device='cuda')
        with self._lock:
            self._cam_intrinsics[cam_id] = K

    def _on_depth(self, msg, cam_id: str, frame: str):
        with self._lock:
            if cam_id not in self._cam_intrinsics:
                return
            K = self._cam_intrinsics[cam_id]

        try:
            # Use the latest available transform (Time() == 0) rather than the
            # depth message's exact stamp: the wrist camera's TF (which moves
            # with the arm) lags the depth stream by a few hundred ms, so an
            # exact-stamp lookup throws "extrapolation into the future" and the
            # wrist frame never integrates — stalling the whole dual-cam map.
            transform = self._tf_buffer.lookup_transform(
                WORLD_FRAME, frame, Time(),
                timeout=rclpy.duration.Duration(seconds=0.1))
        except Exception as e:
            self._logger.warning(
                f'CuRobo: TF lookup failed for {frame}: {e}',
                throttle_duration_sec=2.0)
            return

        cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        depth = torch.from_numpy(cv_img.astype(np.float32) / 1000.0).cuda()
        depth = torch.nan_to_num(depth, nan=0.0)
        filtered, _ = self._depth_filter(depth.unsqueeze(0))
        depth = filtered[0]

        t = transform.transform.translation
        r = transform.transform.rotation
        pose = Pose.from_numpy(
            np.array([t.x, t.y, t.z], dtype=np.float32),
            np.array([r.w, r.x, r.y, r.z], dtype=np.float32),
        )

        # Snapshot the pre-segmenter cloud for viser (shows the robot/gripper
        # too, since masking happens below only for the ESDF path).
        if self._enable_viz:
            self._cache_viz_cloud(cam_id, depth, t, r, K)

        # Mask out pixels that hit the robot itself; otherwise the ESDF marks
        # the arm/gripper as obstacles and the planner refuses every config.
        # Skip silently if /joint_states hasn't arrived yet.
        with self._lock:
            js = self._latest_joints
        if js is not None:
            by_name = dict(zip(js.name, js.position))
            ordered = [by_name[n] for n in JOINT_NAMES if n in by_name]
            if len(ordered) == len(JOINT_NAMES):
                cam_obs_single = CameraObservation(
                    rgb_image=self._dummy_rgb[:1],
                    depth_image=depth.unsqueeze(0),
                    intrinsics=K.unsqueeze(0),
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
                    self._logger.warning(
                        f'CuRobo: RobotSegmenter failed for {cam_id}: '
                        f'{type(exc).__name__}: {exc}')

        with self._lock:
            self._cam_depth[cam_id] = depth
            self._cam_pose[cam_id] = pose
            self._cam_intrinsics[cam_id] = K
            ready = ('overhead' in self._cam_depth and 'wrist' in self._cam_depth)
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
                    # depth is already in metres; override the mm-default so
                    # the TSDF integrator doesn't scale every depth by 0.001.
                    depth_to_meter=1.0,
                )

        if ready:
            self._mapper.integrate(batched)
            with self._lock:
                self._frame_count += 1

    def plan_trajectory(self, grasp_pose, joint_states):
        with self._lock:
            frame_count = self._frame_count

        if frame_count >= MIN_FRAMES:
            # Sync first so any pending Warp/segmenter ops complete before
            # compute_esdf opens its CUDA graph capture; an error queued on a
            # different stream otherwise invalidates the capture (Warp 901).
            torch.cuda.synchronize()
            voxel_grid = self._mapper.compute_esdf()
            self._planner.update_world(SceneCfg(voxel=[voxel_grid]))
            if self._enable_viz:
                self._cache_viz_tsdf()
        else:
            self._logger.warning(
                f'CuRobo: map not ready ({frame_count}/{MIN_FRAMES} frames), '
                'planning in free space.')

        positions = torch.tensor(
            [list(joint_states.position)], dtype=torch.float32, device='cuda')
        start = CuRoboJointState.from_position(
            positions, joint_names=list(joint_states.name))

        p = grasp_pose.position
        o = grasp_pose.orientation
        goal = GoalToolPose(
            tool_frames=self._planner.tool_frames,
            position=torch.tensor(
                [[[[[p.x, p.y, p.z]]]]], device='cuda', dtype=torch.float32),
            quaternion=torch.tensor(
                [[[[[o.w, o.x, o.y, o.z]]]]], device='cuda', dtype=torch.float32),
        )

        result = self._planner.plan_pose(goal, start)
        if result is None or not result.success.any():
            self._logger.warning('CuRobo: planning failed.')
            return None

        return self._to_ros_trajectory(result)

    def tool_pose(self, joint_states):
        """Forward-kinematics tool pose for a joint state.

        Returns ((x, y, z), (w, x, y, z)) in the planner's base frame, or None
        on failure. Useful for capturing the robot's current end-effector pose
        (e.g. a "home" target to return to). Touches CUDA, so call it from the
        same thread that runs plan_trajectory (the ROS executor).
        """
        try:
            positions = torch.tensor(
                [list(joint_states.position)], dtype=torch.float32, device='cuda')
            cjs = CuRoboJointState.from_position(
                positions, joint_names=list(joint_states.name))
            tp = self._planner.compute_kinematics(cjs).tool_poses
            pos = tp.position.reshape(-1)[:3].tolist()
            quat = tp.quaternion.reshape(-1)[:4].tolist()
            return tuple(pos), tuple(quat)
        except Exception as exc:
            self._logger.warning(
                f'CuRobo: FK failed: {type(exc).__name__}: {exc}')
            return None

    def _to_ros_trajectory(self, result):
        traj_msg = JointTrajectory()
        traj_msg.joint_names = list(self._planner.joint_names)

        plan = result.get_interpolated_plan()
        # position may be (B, H, L, G, J) or (B, T, J); reduce to (T, J) so each
        # waypoint row is a flat float sequence (ROS2 rejects nested lists).
        pos_t = plan.position
        while pos_t.dim() > 2:
            pos_t = pos_t[0]
        positions = pos_t.cpu().numpy()
        if plan.velocity is not None:
            vel_t = plan.velocity
            while vel_t.dim() > 2:
                vel_t = vel_t[0]
            velocities = vel_t.cpu().numpy()
        else:
            velocities = None
        dt = self._planner.trajopt_solver.config.interpolation_dt

        for i, pos in enumerate(positions):
            pt = JointTrajectoryPoint()
            pt.positions = pos.tolist()
            if velocities is not None:
                pt.velocities = velocities[i].tolist()
            t_sec = i * dt
            pt.time_from_start = RosDuration(
                sec=int(t_sec),
                nanosec=int((t_sec % 1) * 1e9))
            traj_msg.points.append(pt)

        return traj_msg

    def _build_planner(self):
        # scene_model + a pre-allocated voxel collision cache are required so the
        # planner builds a voxel scene_collision_checker; without them
        # update_world(SceneCfg(voxel=...)) hits a None checker. cuRobo allocates
        # a 128**3 ESDF tensor regardless of MapperCfg.extent_meters_xyz, so the
        # 7 m / 0.05 m cache (140**3 slots) gives headroom over that.
        collision_cache = {
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
        )
        planner = MotionPlanner(config)
        planner.warmup(enable_graph=True, num_warmup_iterations=3)
        return planner
