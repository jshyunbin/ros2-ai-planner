import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


# --- CuRobo ---

def make_curobo():
    """Return a CuRobo instance with all heavy deps mocked."""
    from pipeline_orchestrator.curobo import CuRobo
    node = MagicMock()
    with patch.multiple('pipeline_orchestrator.curobo',
                        Mapper=MagicMock(), FilterDepth=MagicMock(),
                        MotionPlanner=MagicMock(), Buffer=MagicMock(),
                        TransformListener=MagicMock()):
        return CuRobo(node), node


def test_curobo_subscribes_to_four_topics():
    curobo, node = make_curobo()
    topics = [c.args[1] for c in node.create_subscription.call_args_list]
    assert '/camera/camera/depth/color/image_raw' in topics
    assert '/camera/camera/depth/color/camera_info' in topics
    assert '/wrist_camera/wrist_camera/depth/color/image_raw' in topics
    assert '/wrist_camera/wrist_camera/depth/color/camera_info' in topics


def test_curobo_does_not_subscribe_to_joint_states():
    # The service node owns /joint_states and feeds CuRobo via update_joint_state;
    # the CuRobo helper itself must not open a duplicate subscription.
    curobo, node = make_curobo()
    topics = [c.args[1] for c in node.create_subscription.call_args_list]
    assert '/joint_states' not in topics


def test_curobo_update_joint_state_is_retrievable():
    curobo, _ = make_curobo()
    msg = MagicMock()
    curobo.update_joint_state(msg)
    assert curobo.get_latest_joints() is msg


def test_curobo_skips_depth_without_camera_info():
    from pipeline_orchestrator.curobo import CuRobo
    node = MagicMock()
    mock_mapper = MagicMock()
    with patch.multiple('pipeline_orchestrator.curobo',
                        Mapper=MagicMock(return_value=mock_mapper),
                        FilterDepth=MagicMock(),
                        MotionPlanner=MagicMock(), Buffer=MagicMock(),
                        TransformListener=MagicMock()):
        curobo = CuRobo(node)
        curobo._on_depth(MagicMock(), 'overhead', 'camera_color_optical_frame')
        mock_mapper.integrate.assert_not_called()


def test_curobo_plan_trajectory_calls_update_world_after_min_frames(monkeypatch):
    from pipeline_orchestrator.curobo import CuRobo, MIN_FRAMES
    monkeypatch.setenv('PIPELINE_CUROBO_MIN_PLANNING_FRAMES', str(MIN_FRAMES))
    monkeypatch.setenv('PIPELINE_CUROBO_TRAJ_RELAXED_RETRY', '0')
    node = MagicMock()
    mock_mapper = MagicMock()
    mock_planner = MagicMock()
    mock_planner.plan_pose.return_value = None
    mock_planner.tool_frames = ["tool0"]
    grasp_pose = MagicMock()
    grasp_pose.position.x = 0.4
    grasp_pose.position.y = 0.0
    grasp_pose.position.z = 0.3
    grasp_pose.orientation.w = 1.0
    grasp_pose.orientation.x = 0.0
    grasp_pose.orientation.y = 0.0
    grasp_pose.orientation.z = 0.0
    joint_state = MagicMock()
    joint_state.name = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]
    joint_state.position = [0.0, -2.2, 1.9, -1.383, -1.57, 0.0]
    with patch.multiple('pipeline_orchestrator.curobo',
                        Mapper=MagicMock(return_value=mock_mapper),
                        FilterDepth=MagicMock(),
                        MotionPlanner=MagicMock(return_value=mock_planner),
                        Buffer=MagicMock(), TransformListener=MagicMock()):
        curobo = CuRobo(node)
        curobo._frame_count = MIN_FRAMES
        curobo.plan_trajectory(grasp_pose, joint_state)
        mock_mapper.compute_esdf.assert_called_once()
        mock_planner.update_world.assert_called_once()


def test_curobo_pick_retries_relaxed_world_after_tsdf_failures(monkeypatch):
    import numpy as np
    import pipeline_orchestrator.curobo as mod
    from pipeline_orchestrator.curobo import CuRobo

    class FakePlanner:
        tool_frames = ['tool0']

        def __init__(self):
            self.calls = []
            self.world_updates = []

        def clear_scene_cache(self):
            pass

        def update_world(self, scene):
            self.world_updates.append(scene)

        def reset_seed(self):
            pass

        def plan_grasp(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 5:
                return SimpleNamespace(
                    success=np.array([True]),
                    status='ok',
                    goalset_index=np.array([0]),
                )
            return SimpleNamespace(
                success=np.array([False]),
                status='Planning to grasp pose failed.',
                goalset_index=np.array([0]),
                goalset_result=SimpleNamespace(success=np.array([True])),
                approach_result=SimpleNamespace(success=np.array([True])),
                grasp_result=SimpleNamespace(success=np.array([False])),
                lift_result=None,
            )

    fake_planner = FakePlanner()
    curobo = CuRobo.__new__(CuRobo)
    curobo._planner = fake_planner
    curobo._mapper = MagicMock()
    curobo._mapper.compute_esdf.return_value = 'voxel-grid'
    curobo._logger = MagicMock()
    curobo._lock = threading.Lock()
    curobo._frame_count = mod.MIN_FRAMES
    curobo._last_world_update_frame = -1
    curobo._enable_viz = False
    curobo._latest_joints = None
    curobo.update_joint_state = MagicMock()
    curobo._ros_js_to_curobo = MagicMock(return_value='current')
    curobo._grasps_to_goalset = MagicMock(side_effect=lambda items: items[0]['id'])

    monkeypatch.setenv('PIPELINE_CUROBO_MIN_PLANNING_FRAMES', str(mod.MIN_FRAMES))
    monkeypatch.setenv('PIPELINE_CUROBO_GRASP_APPROACH_OFFSETS', '-0.035,-0.06')
    monkeypatch.setenv('PIPELINE_CUROBO_GRASP_LIFT_OFFSET', '0.10')
    monkeypatch.setattr(mod.torch.cuda, 'synchronize', lambda: None)

    candidates = [
        {'id': 'first', 'pose_4x4': np.eye(4, dtype=np.float32)},
        {'id': 'second', 'pose_4x4': np.eye(4, dtype=np.float32)},
    ]

    result = curobo._plan_pick_locked(candidates, MagicMock())

    assert result.success.any()
    assert [call['grasp_poses'] for call in fake_planner.calls] == [
        'first',
        'second',
        'first',
        'second',
        'first',
    ]
    assert [call['grasp_approach_offset'] for call in fake_planner.calls] == [
        -0.035,
        -0.035,
        -0.06,
        -0.06,
        -0.035,
    ]
    assert all(call['grasp_lift_offset'] == 0.10 for call in fake_planner.calls)
    assert len(fake_planner.world_updates) == 2
    assert any(
        'world=tsdf' in str(call.args[0])
        for call in curobo._logger.warn.call_args_list
    )
    assert any(
        'world=relaxed' in str(call.args[0])
        for call in curobo._logger.info.call_args_list
    )


def test_curobo_pick_diagnostics_include_world_and_tool_pose():
    import numpy as np
    from pipeline_orchestrator.curobo import _pick_failure_diagnostics

    result = SimpleNamespace(
        success=np.array([False]),
        goalset_index=np.array([0]),
        goalset_result=SimpleNamespace(success=np.array([True])),
        approach_result=SimpleNamespace(success=np.array([True])),
        grasp_result=SimpleNamespace(success=np.array([False])),
        lift_result=None,
    )
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = [0.5, 0.1, 0.2]

    text = _pick_failure_diagnostics(
        result,
        [{'pose_4x4': pose}],
        world_mode='tsdf',
        candidate_index=2,
        approach_offset=-0.035,
        lift_offset=0.10,
        disabled_collision_links=['tool0'],
    )

    assert 'world=tsdf' in text
    assert 'candidate_index=2' in text
    assert 'approach(success=[True]' in text
    assert 'grasp(success=[False]' in text
    assert 'tool0_grasp_xyz=' in text
    assert 'tool0_lift_xyz=' in text


def test_curobo_trajectory_retries_relaxed_world_after_tsdf_failure(monkeypatch):
    import numpy as np
    import pipeline_orchestrator.curobo as mod
    from pipeline_orchestrator.curobo import CuRobo

    class FakePlanner:
        def __init__(self):
            self.plan_calls = []
            self.world_updates = []

        def clear_scene_cache(self):
            pass

        def update_world(self, scene):
            self.world_updates.append(scene)

        def reset_seed(self):
            pass

        def plan_pose(self, goal, current):
            self.plan_calls.append((goal, current))
            if len(self.plan_calls) == 2:
                return SimpleNamespace(
                    success=np.array([True]),
                    get_interpolated_plan=lambda: 'interpolated',
                    interpolated_last_tstep=3,
                )
            return SimpleNamespace(
                success=np.array([False]),
                status='blocked',
                position_error=np.array([0.1]),
            )

    fake_planner = FakePlanner()
    curobo = CuRobo.__new__(CuRobo)
    curobo._planner = fake_planner
    curobo._mapper = MagicMock()
    curobo._mapper.compute_esdf.return_value = 'voxel-grid'
    curobo._logger = MagicMock()
    curobo._lock = threading.Lock()
    curobo._frame_count = mod.MIN_FRAMES
    curobo._last_world_update_frame = -1
    curobo._enable_viz = False
    curobo.update_joint_state = MagicMock()
    curobo._ros_js_to_curobo = MagicMock(return_value='current')
    curobo._single_goal = MagicMock(return_value='goal')

    monkeypatch.setenv('PIPELINE_CUROBO_MIN_PLANNING_FRAMES', str(mod.MIN_FRAMES))
    monkeypatch.setattr(mod.torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(mod, 'interp_traj_to_ros', lambda *_args, **_kwargs: 'ros-traj')

    trajectory = curobo._plan_trajectory_locked(
        (0.4, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0),
        MagicMock(),
    )

    assert trajectory == 'ros-traj'
    assert len(fake_planner.plan_calls) == 2
    assert len(fake_planner.world_updates) == 2
    assert any(
        'world=tsdf' in str(call.args[0])
        for call in curobo._logger.warn.call_args_list
    )
    assert any(
        'world=relaxed' in str(call.args[0])
        for call in curobo._logger.info.call_args_list
    )


def test_interp_traj_to_ros_scales_derivatives(monkeypatch):
    import torch
    from pipeline_orchestrator.curobo import INTERP_DT, JOINT_NAMES, interp_traj_to_ros

    class FakeInterpolatedTrajectory:
        position = torch.zeros((2, len(JOINT_NAMES)), dtype=torch.float32)
        velocity = torch.ones((2, len(JOINT_NAMES)), dtype=torch.float32)
        acceleration = torch.ones((2, len(JOINT_NAMES)), dtype=torch.float32)

        def squeeze(self, *_args):
            return self

    monkeypatch.setenv('PIPELINE_CUROBO_TRAJ_TIME_SCALE', '2.0')
    traj = interp_traj_to_ros(
        FakeInterpolatedTrajectory(),
        dt=INTERP_DT,
        last_tstep=torch.tensor([1]),
    )

    assert len(traj.points) == 1
    assert traj.points[0].time_from_start.sec == 0
    assert traj.points[0].time_from_start.nanosec == int(INTERP_DT * 2.0 * 1e9)
    assert list(traj.points[0].velocities) == [0.5] * len(JOINT_NAMES)
    assert list(traj.points[0].accelerations) == [0.25] * len(JOINT_NAMES)


# --- Orchestrator ---

def test_orchestrator_importable():
    from pipeline_orchestrator.orchestrator import PipelineOrchestrator
    assert PipelineOrchestrator is not None


def test_orchestrator_has_task_command_callback():
    from pipeline_orchestrator.orchestrator import PipelineOrchestrator
    assert callable(PipelineOrchestrator.task_command_callback)


def test_orchestrator_has_run_pipeline():
    from pipeline_orchestrator.orchestrator import PipelineOrchestrator
    assert callable(PipelineOrchestrator._run_pipeline)


def _orchestrator_skeleton():
    """A PipelineOrchestrator instance without running __init__.

    __init__ creates real ROS subscriptions/action clients and warms up the
    CuRobo planner, so build a bare instance via __new__ and inject only the
    collaborators each unit test needs.
    """
    from pipeline_orchestrator.orchestrator import PipelineOrchestrator
    orch = PipelineOrchestrator.__new__(PipelineOrchestrator)
    orch.get_logger = lambda: MagicMock()
    return orch


def test_orchestrator_execute_trajectory_sends_goal_to_arm_server():
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    orch = _orchestrator_skeleton()
    orch._arm_client = MagicMock()
    orch._arm_client.wait_for_server.return_value = True

    traj = JointTrajectory()
    traj.joint_names = ['shoulder_pan_joint']
    traj.points = [JointTrajectoryPoint()]

    orch._execute_trajectory(traj)

    orch._arm_client.send_goal_async.assert_called_once()
    sent_goal = orch._arm_client.send_goal_async.call_args.args[0]
    assert list(sent_goal.trajectory.joint_names) == ['shoulder_pan_joint']


def test_orchestrator_execute_trajectory_returns_none_when_server_unavailable():
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    orch = _orchestrator_skeleton()
    orch._arm_client = MagicMock()
    orch._arm_client.wait_for_server.return_value = False

    traj = JointTrajectory()
    traj.points = [JointTrajectoryPoint()]
    result = orch._execute_trajectory(traj)

    assert result is None
    orch._arm_client.send_goal_async.assert_not_called()


def test_orchestrator_execute_trajectory_refuses_empty():
    from trajectory_msgs.msg import JointTrajectory
    orch = _orchestrator_skeleton()
    orch._arm_client = MagicMock()

    result = orch._execute_trajectory(JointTrajectory())

    assert result is None
    orch._arm_client.wait_for_server.assert_not_called()
    orch._arm_client.send_goal_async.assert_not_called()


def test_orchestrator_caches_joints_without_touching_planner_runtime():
    orch = _orchestrator_skeleton()
    msg = MagicMock()

    orch._cache_joints(msg)

    assert orch._latest_joints is msg


def test_orchestrator_plan_execute_pick_calls_curobo_service():
    from geometry_msgs.msg import Pose
    from sensor_msgs.msg import JointState
    orch = _orchestrator_skeleton()
    orch._latest_joints = JointState()
    orch._latest_joints.name = ["shoulder_pan_joint"]
    orch._latest_joints.position = [0.0]
    orch._curobo_client = MagicMock()
    orch._curobo_client.wait_for_service.return_value = True
    orch._curobo_service_name = "/curobo/plan_trajectory"
    orch._curobo_service_wait_sec = 0.1

    grasp_pose = Pose()
    orch._plan_and_execute_pick([grasp_pose])

    orch._curobo_client.call_async.assert_called_once()
    request = orch._curobo_client.call_async.call_args.args[0]
    assert list(request.grasp_poses) == [grasp_pose]
    assert request.joint_state is orch._latest_joints


def test_orchestrator_curobo_pick_done_executes_phase_sequence():
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    orch = _orchestrator_skeleton()
    orch._send_and_wait = MagicMock()
    orch._send_gripper = MagicMock()
    orch._reset_pipeline_state = MagicMock()
    trajectory = JointTrajectory()
    trajectory.points = [JointTrajectoryPoint()]
    lift_trajectory = JointTrajectory()
    lift_trajectory.points = [JointTrajectoryPoint()]
    orch._arm_client = MagicMock()
    future = MagicMock()
    future.result.return_value = MagicMock(
        success=True,
        message="planned",
        trajectory=trajectory,
        lift_trajectory=lift_trajectory,
    )

    orch._on_curobo_pick_done(future)

    assert orch._send_and_wait.call_args_list[0].args == (
        orch._arm_client, trajectory, 'approach_and_grasp')
    orch._send_gripper.assert_called_once_with(closed=True)
    assert orch._send_and_wait.call_args_list[1].args == (
        orch._arm_client, lift_trajectory, 'lift')
    orch._reset_pipeline_state.assert_called_once()


def test_curobo_service_preclose_insertion_extends_grasp(monkeypatch):
    from builtin_interfaces.msg import Duration
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from pipeline_orchestrator.curobo_service import (
        _append_preclose_insertion_to_trajectory,
    )

    monkeypatch.setenv('PIPELINE_GRASP_CLOSE_NUDGE_MAX_JOINT_DELTA_RAD', '0.04')
    monkeypatch.setenv('PIPELINE_GRASP_CLOSE_NUDGE_DURATION_SEC', '0.60')
    monkeypatch.setenv('PIPELINE_GRASP_CLOSE_NUDGE_STEPS', '4')

    traj = JointTrajectory()
    for positions, time_sec in [([0.0, 0.0], 0.1), ([0.1, 0.05], 0.2)]:
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start = Duration(
            sec=int(time_sec),
            nanosec=int((time_sec % 1.0) * 1_000_000_000),
        )
        traj.points.append(point)

    nudged, n_added = _append_preclose_insertion_to_trajectory(traj)

    assert n_added == 4
    assert len(nudged.points) == 6
    final = list(nudged.points[-1].positions)
    assert abs(final[0] - 0.14) < 1e-6
    assert abs(final[1] - 0.07) < 1e-6
    assert nudged.points[-1].time_from_start.sec == 0
    assert nudged.points[-1].time_from_start.nanosec == 800_000_000


def test_curobo_service_pick_plans_without_waiting_for_tsdf_map(monkeypatch):
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    import pipeline_orchestrator.curobo_service as mod
    from pipeline_orchestrator.curobo_service import CuRoboService

    node = CuRoboService.__new__(CuRoboService)
    node.get_logger = lambda: MagicMock()

    traj = JointTrajectory()
    traj.points = [JointTrajectoryPoint(), JointTrajectoryPoint()]

    monkeypatch.setattr(mod, 'interp_traj_to_ros', lambda *_args, **_kwargs: traj)
    monkeypatch.setattr(
        mod,
        '_append_preclose_insertion_to_trajectory',
        lambda trajectory: (trajectory, 0),
    )
    monkeypatch.setattr(mod, 'concat_trajectories', lambda first, _second: first)

    curobo = MagicMock()
    curobo.plan_pick.return_value = SimpleNamespace(
        approach_interpolated_trajectory='approach',
        approach_interpolated_last_tstep=2,
        grasp_interpolated_trajectory='grasp',
        grasp_interpolated_last_tstep=2,
        lift_interpolated_trajectory='lift',
        lift_interpolated_last_tstep=2,
    )

    pose = SimpleNamespace(
        position=SimpleNamespace(x=0.5, y=0.0, z=0.1),
        orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    request = SimpleNamespace(grasp_poses=[pose])
    response = SimpleNamespace()
    joint_state = MagicMock()

    result = node._handle_pick(curobo, request, joint_state, response)

    curobo.update_joint_state.assert_called_once_with(joint_state)
    curobo.plan_pick.assert_called_once()
    curobo.wait_for_map_frames.assert_not_called()
    curobo.min_planning_frames.assert_not_called()
    curobo.reset_mapping.assert_not_called()
    assert result.success is True
    assert result.trajectory is traj
    assert result.lift_trajectory is traj


def test_orchestrator_send_and_wait_uses_future_callbacks(monkeypatch):
    from types import SimpleNamespace
    import pipeline_orchestrator.orchestrator as mod
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    class DoneFuture:
        def __init__(self, result):
            self._result = result

        def add_done_callback(self, callback):
            callback(self)

        def done(self):
            return True

        def result(self):
            return self._result

    class AcceptedHandle:
        accepted = True

        def get_result_async(self):
            action_result = SimpleNamespace(
                SUCCESSFUL=0,
                error_code=0,
                error_string='',
            )
            return DoneFuture(SimpleNamespace(
                status=mod._goal_status_succeeded(),
                result=action_result,
            ))

    orch = _orchestrator_skeleton()
    client = MagicMock()
    client.wait_for_server.return_value = True
    client.send_goal_async.return_value = DoneFuture(AcceptedHandle())
    spin = MagicMock()
    monkeypatch.setattr(mod.rclpy, 'spin_until_future_complete', spin)

    trajectory = JointTrajectory()
    trajectory.joint_names = ['shoulder_pan_joint']
    point = JointTrajectoryPoint()
    point.positions = [0.1]
    point.time_from_start.sec = 1
    trajectory.points = [point]

    orch._send_and_wait(client, trajectory, 'arm')

    client.send_goal_async.assert_called_once()
    spin.assert_not_called()


def test_curobo_service_plans_with_supplied_joint_state():
    from pipeline_orchestrator.curobo_service import CuRoboService

    node = CuRoboService.__new__(CuRoboService)
    node.get_logger = lambda: MagicMock()
    node._curobo = MagicMock()
    node._init_error = ""
    node._init_lock = threading.Lock()
    node._latest_joints = None
    request = MagicMock()
    request.grasp_poses = []
    request.joint_state.name = ["shoulder_pan_joint"]
    request.grasp_pose = MagicMock()
    response = MagicMock()
    trajectory = MagicMock(points=[MagicMock()])
    node._curobo.plan_trajectory.return_value = trajectory

    result = node._handle_plan(request, response)

    node._curobo.update_joint_state.assert_called_once_with(request.joint_state)
    node._curobo.plan_trajectory.assert_called_once_with(request.grasp_pose, request.joint_state)
    assert result.success is True
    assert result.trajectory is trajectory


def test_curobo_service_reports_initializing_before_planner_ready():
    from pipeline_orchestrator.curobo_service import CuRoboService

    node = CuRoboService.__new__(CuRoboService)
    node._curobo = None
    node._init_error = ""
    node._init_lock = threading.Lock()
    request = MagicMock()
    response = MagicMock()

    result = node._handle_plan(request, response)

    assert result.success is False
    assert result.message == "CuRobo is still initializing."


def test_orchestrator_does_not_import_nvblox():
    import ast, pathlib
    src = pathlib.Path(
        'src/pipeline_orchestrator/pipeline_orchestrator/orchestrator.py'
    ).read_text()
    tree = ast.parse(src)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.names:
            imports.extend([n.name for n in node.names])
    assert not any('nvblox' in i.lower() for i in imports), f"Found nvblox import: {imports}"
