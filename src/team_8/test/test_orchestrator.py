import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# --- CuRobo ---

def make_curobo():
    """Return a CuRobo instance with all heavy deps mocked."""
    from team_8.curobo import CuRobo
    node = MagicMock()
    with patch.multiple('team_8.curobo',
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
    from team_8.curobo import CuRobo
    node = MagicMock()
    mock_mapper = MagicMock()
    with patch.multiple('team_8.curobo',
                        Mapper=MagicMock(return_value=mock_mapper),
                        FilterDepth=MagicMock(),
                        MotionPlanner=MagicMock(), Buffer=MagicMock(),
                        TransformListener=MagicMock()):
        curobo = CuRobo(node)
        curobo._on_depth(MagicMock(), 'overhead', 'camera_color_optical_frame')
        mock_mapper.integrate.assert_not_called()


def test_curobo_plan_trajectory_calls_update_world_after_min_frames(monkeypatch):
    from team_8.curobo import CuRobo, MIN_FRAMES
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
    with patch.multiple('team_8.curobo',
                        Mapper=MagicMock(return_value=mock_mapper),
                        FilterDepth=MagicMock(),
                        MotionPlanner=MagicMock(return_value=mock_planner),
                        Buffer=MagicMock(), TransformListener=MagicMock()):
        curobo = CuRobo(node)
        curobo._frame_count = MIN_FRAMES
        curobo.plan_trajectory(grasp_pose, joint_state)
        mock_mapper.compute_esdf.assert_called_once()
        mock_planner.update_world.assert_called_once()


def test_curobo_pick_diagnostics_include_world_and_tool_pose():
    import numpy as np
    from team_8.curobo import _pick_failure_diagnostics

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
    import team_8.curobo as mod
    from team_8.curobo import CuRobo

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

        def plan_pose(self, goal, current, enable_graph_attempt=1):
            self.plan_calls.append((goal, current, enable_graph_attempt))
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
    # The single-goal trajectory plan must disable the PRM graph-seed fallback
    # (large enable_graph_attempt) so a TSDF-blocked attempt fails fast and falls
    # through to the relaxed retry, instead of stalling in a silent graph search.
    assert all(call[2] >= 1000 for call in fake_planner.plan_calls)
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
    from team_8.curobo import INTERP_DT, JOINT_NAMES, interp_traj_to_ros

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
    from team_8.orchestrator import PipelineOrchestrator
    assert PipelineOrchestrator is not None


def test_orchestrator_has_task_command_callback():
    from team_8.orchestrator import PipelineOrchestrator
    assert callable(PipelineOrchestrator.task_command_callback)


def test_orchestrator_has_run_pipeline():
    from team_8.orchestrator import PipelineOrchestrator
    assert callable(PipelineOrchestrator._run_pipeline)


def _orchestrator_skeleton():
    """A PipelineOrchestrator instance without running __init__.

    __init__ creates real ROS subscriptions/action clients and warms up the
    CuRobo planner, so build a bare instance via __new__ and inject only the
    collaborators each unit test needs.
    """
    from team_8.orchestrator import PipelineOrchestrator
    orch = PipelineOrchestrator.__new__(PipelineOrchestrator)
    orch.get_logger = lambda: MagicMock()
    return orch


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
    from unittest.mock import call
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    orch = _orchestrator_skeleton()
    orch._send_and_wait = MagicMock()
    orch._send_gripper = MagicMock()
    orch._reset_pipeline_state = MagicMock()
    orch._plan_and_execute_place = MagicMock()
    orch._plan_and_execute_home = MagicMock()
    orch._active_task_data = {'object': 'cup', 'destination': 'storage_1'}
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
    # Gripper opens before the grasp approach, then closes on the object.
    assert orch._send_gripper.call_args_list == [
        call(closed=False), call(closed=True)]
    assert orch._send_and_wait.call_args_list[1].args == (
        orch._arm_client, lift_trajectory, 'lift')
    # Place uses the Gemini-parsed destination from the active task.
    orch._plan_and_execute_place.assert_called_once_with('storage_1')
    orch._plan_and_execute_home.assert_called_once()
    orch._reset_pipeline_state.assert_called_once()


def test_reset_pipeline_state_logs_failure_then_success_without_crashing():
    """Regression: _reset_pipeline_state must not log both severities from one
    call site.

    rclpy's RcutilsLogger caches the severity per caller location (file, function,
    line). Aliasing ``info``/``error`` through a single variable and calling it on
    one line means that line logs at ERROR for a failed task and INFO for a
    successful one. The second severity raises
    ``ValueError('Logger severity cannot be changed between calls.')`` from inside
    the service-done callback, which propagates out of executor.spin() and kills
    the orchestrator process. A queue that mixes a failed task with a later
    successful one therefore crashes the node.
    """
    from collections import deque
    from rclpy.impl.rcutils_logger import RcutilsLogger

    orch = _orchestrator_skeleton()
    # Use a real rclpy logger; a MagicMock logger would not exercise the
    # per-call-site severity caching that triggers the crash.
    logger = RcutilsLogger(name='test_reset_pipeline_state')
    orch.get_logger = lambda: logger
    orch._auto_run_on_task_command = False
    orch._holding_object = False
    orch._task_queue = deque()
    # _reset_pipeline_state cancels any pending post-task verification timer.
    orch._verification_timer = None

    orch._active_task_data = {'object': 'strawberry', 'destination': 'storage_1'}
    orch._reset_pipeline_state(success=False, reason='pick planning failed')

    orch._active_task_data = {'object': 'strawberry', 'destination': 'storage_1'}
    # Before the fix this second call raised ValueError from the shared call site.
    orch._reset_pipeline_state(success=True, reason='pick and place completed')


def test_curobo_service_preclose_insertion_extends_grasp(monkeypatch):
    from builtin_interfaces.msg import Duration
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from team_8.curobo_service import (
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
    import team_8.curobo_service as mod
    from team_8.curobo_service import CuRoboService

    node = CuRoboService.__new__(CuRoboService)
    node.get_logger = lambda: MagicMock()

    approach = JointTrajectory()
    approach.points = [JointTrajectoryPoint()]
    grasp = JointTrajectory()
    grasp.points = [JointTrajectoryPoint()]
    lift = JointTrajectory()
    lift.points = [JointTrajectoryPoint(), JointTrajectoryPoint()]

    # _handle_pick splices a pre-close nudge into the grasp leg and concatenates
    # approach+grasp into the executed trajectory; stub both so the test stays
    # focused on the "plan without waiting for a TSDF map" contract.
    monkeypatch.setattr(
        mod,
        '_append_preclose_insertion_to_trajectory',
        lambda trajectory: (trajectory, 0),
    )
    monkeypatch.setattr(mod, 'concat_trajectories', lambda first, _second: first)

    curobo = MagicMock()
    curobo.plan_pick.return_value = SimpleNamespace(
        approach=approach, grasp=grasp, lift=lift, goalset_index=0)

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
    # concat_trajectories is stubbed to return its first arg (approach+grasp).
    assert result.trajectory is approach
    assert result.lift_trajectory is lift


def test_orchestrator_send_and_wait_uses_future_callbacks(monkeypatch):
    from types import SimpleNamespace
    import team_8.orchestrator as mod
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
    from team_8.curobo_service import CuRoboService

    node = CuRoboService.__new__(CuRoboService)
    node.get_logger = lambda: MagicMock()
    node._curobo = MagicMock()
    node._init_error = ""
    node._init_done = True
    node._init_cv = threading.Condition()
    node._init_wait_sec = 5.0
    node._latest_joints = None
    node._planning = threading.Event()
    request = MagicMock()
    request.goal_name = ''  # not a place/home request -> single-pose mode
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


def test_curobo_service_waits_then_reports_timeout_before_planner_ready():
    from team_8.curobo_service import CuRoboService

    node = CuRoboService.__new__(CuRoboService)
    node.get_logger = lambda: MagicMock()
    node._curobo = None
    node._init_error = ""
    node._init_done = False
    node._init_cv = threading.Condition()
    node._init_wait_sec = 0.1
    request = MagicMock()
    response = MagicMock()

    result = node._handle_plan(request, response)

    assert result.success is False
    assert "still initializing" in result.message


def test_orchestrator_does_not_import_nvblox():
    import ast, pathlib
    src = pathlib.Path(
        'src/team_8/team_8/orchestrator.py'
    ).read_text()
    tree = ast.parse(src)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.names:
            imports.extend([n.name for n in node.names])
    assert not any('nvblox' in i.lower() for i in imports), f"Found nvblox import: {imports}"


def test_curobo_service_publishes_tsdf_voxels_when_centers_present():
    import threading
    import numpy as np
    from unittest.mock import MagicMock
    from builtin_interfaces.msg import Time
    from team_8.curobo_service import CuRoboService

    svc = CuRoboService.__new__(CuRoboService)
    pub = MagicMock()
    svc._tsdf_pub = pub
    svc._init_cv = threading.Condition()
    svc._curobo = MagicMock(
        get_tsdf_centers=MagicMock(return_value=np.zeros((4, 3), dtype=np.float32))
    )
    svc.get_clock = MagicMock(
        return_value=MagicMock(now=lambda: MagicMock(to_msg=lambda: Time()))
    )

    svc._publish_tsdf_voxels()

    pub.publish.assert_called_once()
    cloud = pub.publish.call_args.args[0]
    assert cloud.width == 4
    assert cloud.header.frame_id == "base_link"


def test_curobo_service_tsdf_publish_noop_without_centers():
    import threading
    from unittest.mock import MagicMock
    from team_8.curobo_service import CuRoboService

    svc = CuRoboService.__new__(CuRoboService)
    pub = MagicMock()
    svc._tsdf_pub = pub
    svc._init_cv = threading.Condition()
    svc._curobo = MagicMock(get_tsdf_centers=MagicMock(return_value=None))
    svc._publish_tsdf_voxels()
    pub.publish.assert_not_called()


def test_curobo_service_tsdf_publish_noop_with_empty_centers():
    import threading
    import numpy as np
    from unittest.mock import MagicMock
    from team_8.curobo_service import CuRoboService

    svc = CuRoboService.__new__(CuRoboService)
    pub = MagicMock()
    svc._tsdf_pub = pub
    svc._init_cv = threading.Condition()
    svc._curobo = MagicMock(
        get_tsdf_centers=MagicMock(return_value=np.zeros((0, 3), dtype=np.float32))
    )
    svc._publish_tsdf_voxels()
    pub.publish.assert_not_called()


def test_mat_from_xyz_quat_xyzw_builds_pose_matrix():
    import numpy as np
    from team_8.curobo import _mat_from_xyz_quat_xyzw
    mat = _mat_from_xyz_quat_xyzw([0.1, 0.2, 0.3], [0.0, 0.0, 0.0, 1.0])
    assert mat.shape == (4, 4)
    np.testing.assert_allclose(mat[:3, 3], [0.1, 0.2, 0.3], atol=1e-6)
    np.testing.assert_allclose(mat[:3, :3], np.eye(3), atol=1e-6)


def test_plan_place_storage_chains_three_collision_off_legs(monkeypatch):
    import team_8.curobo as mod
    from geometry_msgs.msg import Pose
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    curobo = mod.CuRobo.__new__(mod.CuRobo)
    curobo._cuda_lock = threading.Lock()
    curobo._logger = MagicMock()
    curobo.update_joint_state = MagicMock()
    curobo._ros_js_to_curobo = MagicMock(return_value='current')
    curobo.tool_pose = MagicMock(return_value=([0.3, 0.1, 0.5], [1.0, 0.0, 0.0, 0.0]))
    floor_calls = []
    curobo._set_transit_floor_world = MagicMock(
        side_effect=lambda z: floor_calls.append(z))

    def _fake_segment(mat, state, name, in_branch=False):
        jt = JointTrajectory()
        jt.points = [JointTrajectoryPoint()]
        return jt
    curobo._plan_pose_segment = MagicMock(side_effect=_fake_segment)
    curobo._final_joint_state = MagicMock(return_value='next')
    monkeypatch.setattr(mod, 'concat_trajectories', lambda a, b: a)

    place_pose = Pose()
    place_pose.position.x, place_pose.position.y, place_pose.position.z = 0.0, 0.55, 0.70
    place_pose.orientation.x, place_pose.orientation.w = 1.0, 0.0

    plan = curobo.plan_place(place_pose, transit_z=0.80, bookshelf=False,
                             joint_states=MagicMock())

    assert floor_calls == [0.0]   # transit planned over a ground plane, not empty
    assert curobo._plan_pose_segment.call_count == 3
    assert plan.insert is None and plan.retract is None
    assert plan.move is not None


def test_plan_place_bookshelf_adds_insert_and_retract(monkeypatch):
    import team_8.curobo as mod
    from geometry_msgs.msg import Pose
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    curobo = mod.CuRobo.__new__(mod.CuRobo)
    curobo._cuda_lock = threading.Lock()
    curobo._logger = MagicMock()
    curobo.update_joint_state = MagicMock()
    curobo._ros_js_to_curobo = MagicMock(return_value='current')
    curobo.tool_pose = MagicMock(return_value=([0.3, 0.1, 0.5], [1.0, 0.0, 0.0, 0.0]))
    curobo._set_transit_floor_world = MagicMock()

    def _fake_segment(mat, state, name, in_branch=False):
        jt = JointTrajectory()
        jt.points = [JointTrajectoryPoint()]
        return jt
    curobo._plan_pose_segment = MagicMock(side_effect=_fake_segment)
    curobo._final_joint_state = MagicMock(return_value='next')
    monkeypatch.setattr(mod, 'concat_trajectories', lambda a, b: a)

    place_pose = Pose()
    place_pose.position.x, place_pose.position.y, place_pose.position.z = 0.60, -0.30, 0.76
    place_pose.orientation.x = 0.5

    plan = curobo.plan_place(place_pose, transit_z=0.80, bookshelf=True,
                             insert_depth=0.22, retract_depth=0.22,
                             joint_states=MagicMock())

    assert curobo._plan_pose_segment.call_count == 5
    assert plan.insert is not None and plan.retract is not None
    seg_names = [c.args[2] for c in curobo._plan_pose_segment.call_args_list]
    assert seg_names[-2:] == ['insert', 'retract']


def test_set_transit_floor_world_builds_ground_plane(monkeypatch):
    """The transit world is a single cuboid whose top face sits at floor_z."""
    import team_8.curobo as mod

    curobo = mod.CuRobo.__new__(mod.CuRobo)
    curobo._logger = MagicMock()
    curobo._lock = threading.Lock()
    curobo._last_world_update_frame = 99
    updates = []
    planner = MagicMock()
    planner.update_world = MagicMock(side_effect=lambda scene: updates.append(scene))
    curobo._planner = planner
    monkeypatch.setattr(mod.torch.cuda, 'synchronize', lambda: None)

    curobo._set_transit_floor_world(0.05)

    assert len(updates) == 1
    scene = updates[0]
    assert len(scene.cuboid) == 1            # exactly one ground plane, no voxels
    assert len(scene.voxel) == 0
    floor = scene.cuboid[0]
    top_face_z = floor.pose[2] + floor.dims[2] / 2.0
    assert abs(top_face_z - 0.05) < 1e-9     # top of the box is exactly at floor_z
    assert floor.dims[0] >= 1.0 and floor.dims[1] >= 1.0  # spans the workspace
    assert curobo._last_world_update_frame == -1


def test_plan_place_lift_leg_lowers_height_until_reachable(monkeypatch):
    """A far grasp's lift leg steps its target height down until IK is reachable.

    Reproduces the far-banana failure: lifting straight to transit_z at a large
    horizontal radius is unreachable, so the lift leg must lower its target until
    the arm can reach it instead of aborting the whole place.
    """
    import team_8.curobo as mod
    from geometry_msgs.msg import Pose
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    curobo = mod.CuRobo.__new__(mod.CuRobo)
    curobo._cuda_lock = threading.Lock()
    curobo._logger = MagicMock()
    curobo.update_joint_state = MagicMock()
    curobo._ros_js_to_curobo = MagicMock(return_value='current')
    # Current tool sits at a far radius (~0.73 m), low z (post-lift grasp pose).
    curobo.tool_pose = MagicMock(
        return_value=([0.717, -0.152, 0.31], [1.0, 0.0, 0.0, 0.0]))
    curobo._set_transit_floor_world = MagicMock()
    curobo._final_joint_state = MagicMock(return_value='next')
    monkeypatch.setattr(mod, 'concat_trajectories', lambda a, b: a)
    monkeypatch.setenv('PIPELINE_PLACE_LIFT_HEIGHT_STEP_M', '0.05')

    # IK only succeeds at z <= 0.45 at this far radius; higher lift fails.
    reachable_ceiling = 0.45
    seg_calls = []

    def _fake_segment(mat, state, name, in_branch=False):
        z = float(mat[2, 3])
        seg_calls.append((name, round(z, 3)))
        if name.startswith('lift@') and z > reachable_ceiling + 1e-9:
            return None
        jt = JointTrajectory()
        jt.points = [JointTrajectoryPoint()]
        return jt
    curobo._plan_pose_segment = MagicMock(side_effect=_fake_segment)

    place_pose = Pose()
    place_pose.position.x, place_pose.position.y, place_pose.position.z = \
        0.069, 0.649, 0.37
    place_pose.orientation.x, place_pose.orientation.w = 1.0, 0.0

    plan = curobo.plan_place(place_pose, transit_z=0.55, bookshelf=False,
                             joint_states=MagicMock())

    assert plan is not None and plan.move is not None
    lift_calls = [c for c in seg_calls if c[0].startswith('lift@')]
    assert lift_calls[0][1] == 0.55                       # tried full transit_z
    assert any(z > reachable_ceiling for _, z in lift_calls)  # stepped down
    assert lift_calls[-1][1] <= reachable_ceiling + 1e-9  # settled within reach
    # Exactly one successful lift, plus the traverse + descend legs.
    assert len([c for c in seg_calls if not c[0].startswith('lift@')]) == 2


def test_plan_place_lift_leg_falls_back_to_current_height(monkeypatch):
    """If no lifted height is reachable, the lift settles at the current height."""
    import team_8.curobo as mod
    from geometry_msgs.msg import Pose
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    curobo = mod.CuRobo.__new__(mod.CuRobo)
    curobo._cuda_lock = threading.Lock()
    curobo._logger = MagicMock()
    curobo.update_joint_state = MagicMock()
    curobo._ros_js_to_curobo = MagicMock(return_value='current')
    curobo.tool_pose = MagicMock(
        return_value=([0.75, 0.0, 0.34], [1.0, 0.0, 0.0, 0.0]))
    curobo._set_transit_floor_world = MagicMock()
    curobo._final_joint_state = MagicMock(return_value='next')
    monkeypatch.setattr(mod, 'concat_trajectories', lambda a, b: a)
    monkeypatch.setenv('PIPELINE_PLACE_LIFT_HEIGHT_STEP_M', '0.05')

    seg_calls = []

    def _fake_segment(mat, state, name, in_branch=False):
        z = float(mat[2, 3])
        seg_calls.append((name, round(z, 3)))
        # Every lifted height fails; only the floor (current height) is reachable.
        if name == 'lift@current' or not name.startswith('lift@'):
            jt = JointTrajectory()
            jt.points = [JointTrajectoryPoint()]
            return jt
        return None
    curobo._plan_pose_segment = MagicMock(side_effect=_fake_segment)

    place_pose = Pose()
    place_pose.position.x, place_pose.position.y, place_pose.position.z = \
        0.069, 0.649, 0.37
    place_pose.orientation.x, place_pose.orientation.w = 1.0, 0.0

    plan = curobo.plan_place(place_pose, transit_z=0.55, bookshelf=False,
                             joint_states=MagicMock())

    assert plan is not None and plan.move is not None
    names = [c[0] for c in seg_calls]
    assert 'lift@current' in names               # fell through to the floor
    assert names[names.index('lift@current')]    # floor produced the lift leg
    # Floor lift planned at the current height (0.34), not below it.
    floor_call = next(c for c in seg_calls if c[0] == 'lift@current')
    assert floor_call[1] == 0.34


def _inbranch_curobo(planner):
    import team_8.curobo as mod
    curobo = mod.CuRobo.__new__(mod.CuRobo)
    curobo._logger = MagicMock()
    curobo._tool_goal_from_matrix = MagicMock(return_value='goal')
    curobo._planner = planner
    return curobo


def test_plan_pose_inbranch_selects_nearest_ik_solution(monkeypatch):
    """In-branch planning picks the IK solution closest to the current config
    (not the lowest-cost/flipped one) and c-space-plans to that fixed goal."""
    import team_8.curobo as mod
    import torch

    cur = torch.tensor([[0.0, -1.5, 1.5, -1.5, -1.5, 0.0]])
    current_state = mod.CuRoboJointState.from_position(
        cur, joint_names=list(mod.JOINT_NAMES))

    # Seed 0 is a flipped branch (far in joint space); seed 1 is in-branch (near).
    flipped = [3.0, 1.5, -1.5, 1.5, 1.5, 3.0]
    near = [0.1, -1.4, 1.4, -1.4, -1.5, 0.05]
    ik_result = SimpleNamespace(
        solution=torch.tensor([[flipped, near]]),   # (1, 2, 6)
        success=torch.tensor([[True, True]]))

    planner = MagicMock()
    planner.ik_solver.config.num_seeds = 2
    planner.ik_solver.solve_pose = MagicMock(return_value=ik_result)
    cspace_goals = []

    def _fake_cspace(goal_state, current):
        cspace_goals.append(goal_state)
        return SimpleNamespace(success=torch.tensor([True]),
                               get_interpolated_plan=lambda: 'plan',
                               interpolated_last_tstep=None)
    planner.plan_cspace = MagicMock(side_effect=_fake_cspace)

    curobo = _inbranch_curobo(planner)
    monkeypatch.setattr(mod, '_reset_planner_seed', lambda p: None)
    monkeypatch.setattr(mod.torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(mod, 'interp_traj_to_ros',
                        lambda plan, last_tstep=None: 'ros_traj')

    out = curobo._plan_pose_segment('mat', current_state, 'descend',
                                    in_branch=True)

    assert out == 'ros_traj'
    # Solved IK across all seeds, then c-space-planned to the NEAR solution.
    planner.ik_solver.solve_pose.assert_called_once()
    assert planner.ik_solver.solve_pose.call_args.kwargs['return_seeds'] == 2
    assert len(cspace_goals) == 1
    chosen = cspace_goals[0].position.view(-1).tolist()
    assert chosen == pytest.approx(near, abs=1e-5)
    # plan_pose (the flip-prone free path) is never used in-branch.
    planner.plan_pose.assert_not_called()


def test_plan_pose_inbranch_fails_safely_when_ik_infeasible(monkeypatch):
    """No feasible IK solution -> return None (fail the leg, never c-space-plan)."""
    import team_8.curobo as mod
    import torch

    cur = torch.tensor([[0.0, -1.5, 1.5, -1.5, -1.5, 0.0]])
    current_state = mod.CuRoboJointState.from_position(
        cur, joint_names=list(mod.JOINT_NAMES))
    ik_result = SimpleNamespace(
        solution=torch.zeros((1, 2, 6)),
        success=torch.tensor([[False, False]]))

    planner = MagicMock()
    planner.ik_solver.config.num_seeds = 2
    planner.ik_solver.solve_pose = MagicMock(return_value=ik_result)

    curobo = _inbranch_curobo(planner)
    monkeypatch.setattr(mod, '_reset_planner_seed', lambda p: None)

    out = curobo._plan_pose_segment('mat', current_state, 'descend',
                                    in_branch=True)

    assert out is None
    planner.plan_cspace.assert_not_called()


# --- route_place_or_home ---

def _place_data():
    return {
        "transit_z": 0.80,
        "home_joint_config": [0.0, -2.2, 1.9, -1.383, -1.57, 0.0],
        "storage_1": {"xyz": [0.0, 0.55, 0.70], "quat_xyzw": [1.0, 0.0, 0.0, 0.0]},
        "bookshelf": {
            "pre_insert": {"xyz": [0.60, -0.30, 0.76],
                           "quat_xyzw": [0.5, 0.5, 0.5, 0.5]},
            "insert_depth_m": 0.22, "retract_depth_m": 0.22,
        },
    }


def test_route_home_uses_collision_aware_joint_config_plan():
    from riro_srvs.srv import PlanTrajectory
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from team_8.curobo_service import route_place_or_home
    curobo = MagicMock()
    traj = JointTrajectory(); traj.points = [JointTrajectoryPoint()]
    curobo.plan_home_config.return_value = traj
    resp = route_place_or_home(curobo, _place_data(), "home",
                               MagicMock(), PlanTrajectory.Response())
    curobo.plan_home_config.assert_called_once()
    # home plans to the fixed joint config, not an IK pose or a place transit.
    assert curobo.plan_home_config.call_args.args[0] == [
        0.0, -2.2, 1.9, -1.383, -1.57, 0.0]
    curobo.plan_place.assert_not_called()
    assert resp.success is True
    assert resp.trajectory is traj


def test_route_storage_calls_plan_place_simple():
    from riro_srvs.srv import PlanTrajectory
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from team_8.curobo_service import route_place_or_home
    from team_8.curobo import PlacePlan
    curobo = MagicMock()
    move = JointTrajectory(); move.points = [JointTrajectoryPoint()]
    curobo.plan_place.return_value = PlacePlan(move=move)
    resp = route_place_or_home(curobo, _place_data(), "storage_1",
                               MagicMock(), PlanTrajectory.Response())
    assert curobo.plan_place.call_args.kwargs["bookshelf"] is False
    assert resp.success is True
    assert resp.trajectory is move
    assert not resp.insert_trajectory.points
    assert not resp.retract_trajectory.points


def test_route_bookshelf_populates_insert_and_retract():
    from riro_srvs.srv import PlanTrajectory
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from team_8.curobo_service import route_place_or_home
    from team_8.curobo import PlacePlan
    curobo = MagicMock()
    move = JointTrajectory(); move.points = [JointTrajectoryPoint()]
    ins = JointTrajectory(); ins.points = [JointTrajectoryPoint()]
    ret = JointTrajectory(); ret.points = [JointTrajectoryPoint()]
    curobo.plan_place.return_value = PlacePlan(move=move, insert=ins, retract=ret)
    resp = route_place_or_home(curobo, _place_data(), "bookshelf",
                               MagicMock(), PlanTrajectory.Response())
    assert curobo.plan_place.call_args.kwargs["bookshelf"] is True
    assert curobo.plan_place.call_args.kwargs["insert_depth"] == 0.22
    assert resp.insert_trajectory is ins
    assert resp.retract_trajectory is ret


def test_route_unknown_goal_fails_cleanly():
    from riro_srvs.srv import PlanTrajectory
    from team_8.curobo_service import route_place_or_home
    curobo = MagicMock()
    resp = route_place_or_home(curobo, _place_data(), "nope",
                               MagicMock(), PlanTrajectory.Response())
    assert resp.success is False
    curobo.plan_place.assert_not_called()


def test_route_place_pauses_mapping_but_home_does_not():
    from riro_srvs.srv import PlanTrajectory
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from team_8.curobo_service import route_place_or_home
    from team_8.curobo import PlacePlan
    # Place: mapping is paused so the carried object doesn't fuse into the world.
    curobo = MagicMock()
    move = JointTrajectory(); move.points = [JointTrajectoryPoint()]
    curobo.plan_place.return_value = PlacePlan(move=move)
    route_place_or_home(curobo, _place_data(), "storage_1",
                        MagicMock(), PlanTrajectory.Response())
    curobo.pause_mapping.assert_called_once()
    # Home: mapping stays live for collision-aware planning.
    curobo_home = MagicMock()
    htraj = JointTrajectory(); htraj.points = [JointTrajectoryPoint()]
    curobo_home.plan_trajectory.return_value = htraj
    route_place_or_home(curobo_home, _place_data(), "home",
                        MagicMock(), PlanTrajectory.Response())
    curobo_home.pause_mapping.assert_not_called()


def _make_place_response(insert=False, retract=False, success=True):
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    def _t(points):
        jt = JointTrajectory()
        if points:
            jt.points = [JointTrajectoryPoint()]
        return jt
    return MagicMock(
        success=success, message='ok',
        trajectory=_t(True),
        insert_trajectory=_t(insert),
        retract_trajectory=_t(retract),
    )


def test_orchestrator_place_storage_sequence_releases_then_no_retract():
    from sensor_msgs.msg import JointState
    orch = _orchestrator_skeleton()
    orch._arm_client = MagicMock()
    orch._send_and_wait = MagicMock()
    orch._send_gripper = MagicMock()
    orch._latest_joints = JointState()
    orch._call_curobo_blocking = MagicMock(
        return_value=_make_place_response(insert=False, retract=False))

    orch._plan_and_execute_place('storage_1')

    labels = [c.args[2] for c in orch._send_and_wait.call_args_list]
    assert labels == ['place_transit']
    orch._send_gripper.assert_called_once_with(closed=False)


def test_orchestrator_place_bookshelf_sequence_insert_release_retract():
    from sensor_msgs.msg import JointState
    orch = _orchestrator_skeleton()
    orch._arm_client = MagicMock()
    calls = []
    orch._send_and_wait = MagicMock(
        side_effect=lambda c, t, label, **k: calls.append(('move', label)))
    orch._send_gripper = MagicMock(
        side_effect=lambda closed: calls.append(('grip', closed)))
    orch._latest_joints = JointState()
    orch._call_curobo_blocking = MagicMock(
        return_value=_make_place_response(insert=True, retract=True))

    orch._plan_and_execute_place('bookshelf')

    assert calls == [
        ('move', 'place_transit'),
        ('move', 'bookshelf_insert'),
        ('grip', False),
        ('move', 'bookshelf_retract'),
    ]


def test_orchestrator_home_executes_returned_trajectory():
    from sensor_msgs.msg import JointState
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    orch = _orchestrator_skeleton()
    orch._arm_client = MagicMock()
    orch._send_and_wait = MagicMock()
    orch._latest_joints = JointState()
    home_traj = JointTrajectory(); home_traj.points = [JointTrajectoryPoint()]
    orch._call_curobo_blocking = MagicMock(
        return_value=MagicMock(success=True, message='ok', trajectory=home_traj))

    orch._plan_and_execute_home()

    assert orch._send_and_wait.call_args.args == (
        orch._arm_client, home_traj, 'home')


def test_orchestrator_pick_done_runs_place_then_home():
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    orch = _orchestrator_skeleton()
    orch._send_and_wait = MagicMock()
    orch._send_gripper = MagicMock()
    orch._reset_pipeline_state = MagicMock()
    orch._plan_and_execute_place = MagicMock()
    orch._plan_and_execute_home = MagicMock()
    orch._active_task_data = {'object': 'book', 'destination': 'storage_2'}
    orch._arm_client = MagicMock()
    trajectory = JointTrajectory(); trajectory.points = [JointTrajectoryPoint()]
    lift_trajectory = JointTrajectory(); lift_trajectory.points = [JointTrajectoryPoint()]
    future = MagicMock()
    future.result.return_value = MagicMock(
        success=True, message='planned',
        trajectory=trajectory, lift_trajectory=lift_trajectory)

    orch._on_curobo_pick_done(future)

    orch._plan_and_execute_place.assert_called_once_with('storage_2')
    orch._plan_and_execute_home.assert_called_once_with()
    orch._reset_pipeline_state.assert_called_once()


def test_orchestrator_verification_timer_uses_pipeline_callback_group():
    # The verification timer callback runs the full retry pipeline, whose
    # _send_and_wait calls block on action goal-response futures. If the timer
    # lives in the default mutually-exclusive group (same as the arm
    # ActionClient), the goal-response callback can never be delivered and the
    # home move deadlocks (60s "Timed out waiting for home action goal
    # response"). It must share the reentrant pipeline group, like the task
    # subscription and service clients.
    orch = _orchestrator_skeleton()
    sentinel_group = object()
    orch._pipeline_cbg = sentinel_group
    orch._verification_timer = None
    orch.create_timer = MagicMock(return_value='timer-handle')

    orch._schedule_verification_timer(1.0)

    _, kwargs = orch.create_timer.call_args
    assert kwargs.get('callback_group') is sentinel_group


def test_orchestrator_pick_done_schedules_verification_when_enabled():
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    orch = _orchestrator_skeleton()
    orch._send_and_wait = MagicMock()
    orch._send_gripper = MagicMock()
    orch._reset_pipeline_state = MagicMock()
    orch._schedule_post_task_verification = MagicMock()
    orch._plan_and_execute_place = MagicMock()
    orch._plan_and_execute_home = MagicMock()
    orch._enable_post_task_verification = True
    orch._active_task_data = {'object': 'book', 'destination': 'storage_2'}
    orch._arm_client = MagicMock()
    trajectory = JointTrajectory(); trajectory.points = [JointTrajectoryPoint()]
    lift_trajectory = JointTrajectory(); lift_trajectory.points = [JointTrajectoryPoint()]
    future = MagicMock()
    future.result.return_value = MagicMock(
        success=True, message='planned',
        trajectory=trajectory, lift_trajectory=lift_trajectory)

    orch._on_curobo_pick_done(future)

    # With verification enabled, pick-done hands off to the verification path
    # (the teammate's intent) instead of marking the task complete directly.
    orch._schedule_post_task_verification.assert_called_once()
    orch._reset_pipeline_state.assert_not_called()


def test_orchestrator_home_before_capture_homes_before_opening_gripper():
    orch = _orchestrator_skeleton()
    orch._enable_motion_execution = True
    orch._curobo_client = MagicMock()
    orch._latest_joints = MagicMock()
    orch._curobo_ready_event = threading.Event()
    orch._curobo_ready_event.set()  # cuRobo already ready: no wait
    orch._curobo_ready_wait_sec = 1.0
    # Record call order: the home move (which blocks on the cuRobo service until
    # the planner has initialised) must run BEFORE the gripper command, so the
    # gripper action isn't issued while the controllers are still coming up and
    # not yet accepting goals (otherwise the goal response times out).
    calls = []
    orch._send_gripper = MagicMock(
        side_effect=lambda closed: calls.append(('grip', closed)))
    orch._plan_and_execute_home = MagicMock(
        side_effect=lambda: calls.append(('home', None)))
    orch.get_clock = MagicMock(
        return_value=MagicMock(now=lambda: MagicMock(nanoseconds=999)))

    stamp = orch._home_before_capture()

    assert calls == [('home', None), ('grip', False)]
    assert stamp == 999


def test_orchestrator_home_before_capture_raises_if_curobo_never_ready():
    orch = _orchestrator_skeleton()
    orch._enable_motion_execution = True
    orch._curobo_client = MagicMock()
    orch._latest_joints = MagicMock()
    orch._curobo_ready_event = threading.Event()  # never set
    orch._curobo_ready_wait_sec = 0.05
    orch._plan_and_execute_home = MagicMock()
    orch._send_gripper = MagicMock()

    with pytest.raises(RuntimeError, match='/curobo/ready'):
        orch._home_before_capture()

    # Never moves the arm or touches the gripper if cuRobo isn't ready.
    orch._plan_and_execute_home.assert_not_called()
    orch._send_gripper.assert_not_called()


def test_orchestrator_on_curobo_ready_sets_event():
    from types import SimpleNamespace
    orch = _orchestrator_skeleton()
    orch._curobo_ready_event = threading.Event()

    orch._on_curobo_ready(SimpleNamespace(data=False))
    assert not orch._curobo_ready_event.is_set()

    orch._on_curobo_ready(SimpleNamespace(data=True))
    assert orch._curobo_ready_event.is_set()


def test_orchestrator_home_before_capture_skips_when_motion_disabled():
    orch = _orchestrator_skeleton()
    orch._enable_motion_execution = False
    orch._curobo_client = None
    orch._plan_and_execute_home = MagicMock()

    assert orch._home_before_capture() == 0
    orch._plan_and_execute_home.assert_not_called()


def test_orchestrator_home_before_capture_skips_without_joints():
    orch = _orchestrator_skeleton()
    orch._enable_motion_execution = True
    orch._curobo_client = MagicMock()
    orch._latest_joints = None
    orch._plan_and_execute_home = MagicMock()

    assert orch._home_before_capture() == 0
    orch._plan_and_execute_home.assert_not_called()


def test_orchestrator_run_pipeline_homes_before_segmenting():
    import json
    import team_8.orchestrator as _orch_mod
    orch = _orchestrator_skeleton()
    orch._pipeline_busy = False
    orch._active_task = ''
    orch._active_task_data = None
    orch._segmentation_service_name = '/segmentation/segment_prompt'
    orch._segmentation_service_wait_sec = 0.1
    calls = []
    orch._segmentation_client = MagicMock()
    orch._segmentation_client.wait_for_service.return_value = True
    orch._segmentation_client.call_async.side_effect = (
        lambda req: calls.append(('segment', req)) or MagicMock())
    orch._home_before_capture = MagicMock(
        side_effect=lambda: calls.append(('home', None)) or 555)
    orch._on_segmentation_done = MagicMock()
    mock_stringstring = MagicMock()
    mock_stringstring.Request.return_value = MagicMock(data='')

    with patch.object(_orch_mod, 'StringString', mock_stringstring):
        orch._run_pipeline('pick the mug')

    # Home move happens before segmentation is requested.
    assert [c[0] for c in calls] == ['home', 'segment']
    request = calls[1][1]
    payload = json.loads(request.data)
    assert payload == {'prompt': 'pick the mug', 'min_stamp_ns': 555}


def test_orchestrator_run_pipeline_aborts_when_home_fails():
    # A failed initial home move must not request segmentation, and it must take
    # the retry-or-skip path (a transient home-plan failure should not silently
    # drop the object), not a bare drop to the next task.
    orch = _orchestrator_skeleton()
    orch._pipeline_busy = False
    orch._active_task = ''
    orch._segmentation_service_name = '/segmentation/segment_prompt'
    orch._segmentation_service_wait_sec = 0.1
    orch._segmentation_client = MagicMock()
    orch._segmentation_client.wait_for_service.return_value = True
    orch._home_before_capture = MagicMock(side_effect=RuntimeError('no plan'))
    orch._retry_or_skip = MagicMock()

    orch._run_pipeline('pick the mug')

    orch._segmentation_client.call_async.assert_not_called()
    orch._retry_or_skip.assert_called_once()


def test_orchestrator_segmentation_failure_retries_task_in_place():
    # A pre-grasp pipeline failure (e.g. Gemini 503 on segmentation) must
    # consume a retry attempt and re-run the SAME task, not silently drop it
    # and advance to the next queued object. The retry budget previously only
    # applied to post-execution verification, so a transient stage failure
    # skipped the object entirely (the "silent skip" bug).
    import json
    orch = _orchestrator_skeleton()
    orch._max_task_attempts = 2
    orch._active_task_data = {
        'object': 'banana', 'destination': 'storage_2', '_attempt_count': 1}
    orch._restart_active_task = MagicMock()
    orch._reset_pipeline_state = MagicMock()
    future = MagicMock()
    future.result.return_value = MagicMock(
        data=json.dumps({'success': False, 'error': '503 UNAVAILABLE'}))

    orch._on_segmentation_done(future)

    orch._restart_active_task.assert_called_once()
    orch._reset_pipeline_state.assert_not_called()


def test_orchestrator_segmentation_failure_skips_after_attempts_exhausted():
    # Once the attempt budget is spent the task is skipped (the queue advances
    # to the next object) rather than retried forever.
    import json
    orch = _orchestrator_skeleton()
    orch._max_task_attempts = 2
    orch._active_task_data = {
        'object': 'banana', 'destination': 'storage_2', '_attempt_count': 2}
    orch._restart_active_task = MagicMock()
    orch._reset_pipeline_state = MagicMock()
    future = MagicMock()
    future.result.return_value = MagicMock(
        data=json.dumps({'success': False, 'error': '503 UNAVAILABLE'}))

    orch._on_segmentation_done(future)

    orch._restart_active_task.assert_not_called()
    orch._reset_pipeline_state.assert_called_once()
    assert orch._reset_pipeline_state.call_args.kwargs.get('success') is False


def test_orchestrator_graspgen_failure_retries_task_in_place():
    # GraspGen failures are also pre-grasp, so they take the same retry path.
    import json
    orch = _orchestrator_skeleton()
    orch._max_task_attempts = 2
    orch._active_task_data = {
        'object': 'banana', 'destination': 'storage_2', '_attempt_count': 1}
    orch._restart_active_task = MagicMock()
    orch._reset_pipeline_state = MagicMock()
    future = MagicMock()
    future.result.return_value = MagicMock(
        data=json.dumps({'success': False, 'error': 'no grasps'}))

    orch._on_graspgen_done(future)

    orch._restart_active_task.assert_called_once()
    orch._reset_pipeline_state.assert_not_called()


# --- curobo_service /curobo/ready readiness signal ---

def _ready_service_skeleton(init_done, frame_count):
    """A CuRoboService with only the readiness state _check_ready touches."""
    from team_8.curobo_service import CuRoboService
    svc = CuRoboService.__new__(CuRoboService)
    svc.get_logger = lambda: MagicMock()
    svc._init_cv = threading.Condition()
    svc._init_done = init_done
    svc._curobo = None if frame_count is None else MagicMock(frame_count=frame_count)
    svc._ready_published = False
    svc._ready_pub = MagicMock()
    svc._ready_timer = MagicMock()
    return svc


def test_curobo_service_check_ready_publishes_when_init_done_and_map_ready():
    from team_8.curobo import MIN_FRAMES
    svc = _ready_service_skeleton(init_done=True, frame_count=MIN_FRAMES)

    svc._check_ready()

    svc._ready_pub.publish.assert_called_once()
    published = svc._ready_pub.publish.call_args.args[0]
    assert published.data is True
    assert svc._ready_published is True
    svc._ready_timer.cancel.assert_called_once()


def test_curobo_service_check_ready_waits_for_init():
    svc = _ready_service_skeleton(init_done=False, frame_count=None)
    svc._check_ready()
    svc._ready_pub.publish.assert_not_called()
    assert svc._ready_published is False


def test_curobo_service_check_ready_waits_for_map_frames():
    from team_8.curobo import MIN_FRAMES
    svc = _ready_service_skeleton(init_done=True, frame_count=MIN_FRAMES - 1)
    svc._check_ready()
    svc._ready_pub.publish.assert_not_called()
    assert svc._ready_published is False


def test_curobo_service_check_ready_is_one_shot():
    from team_8.curobo import MIN_FRAMES
    svc = _ready_service_skeleton(init_done=True, frame_count=MIN_FRAMES)
    svc._ready_published = True  # already fired
    svc._check_ready()
    svc._ready_pub.publish.assert_not_called()


def test_count_target_in_workspace_returns_count_when_frame_fresh():
    import numpy as np
    orch = _orchestrator_skeleton()
    orch._verification_frame_timeout_sec = 1.0
    orch._latest_verification_rgb = np.zeros((4, 4, 3), dtype='uint8')
    orch._latest_verification_rgb_stamp_ns = 200
    orch._gemini = MagicMock()
    orch._gemini.count_objects.return_value = {"count": 2, "reason": "two"}

    count = orch._count_target_in_workspace('coke_can', min_stamp_ns=100)

    assert count == 2
    assert orch._gemini.count_objects.call_args.kwargs['object_name'] == 'coke_can'


def test_count_target_in_workspace_returns_none_on_stale_frame():
    import numpy as np
    orch = _orchestrator_skeleton()
    orch._verification_frame_timeout_sec = 0.1
    orch._latest_verification_rgb = np.zeros((4, 4, 3), dtype='uint8')
    orch._latest_verification_rgb_stamp_ns = 50  # not newer than min_stamp_ns
    orch._gemini = MagicMock()

    count = orch._count_target_in_workspace('coke_can', min_stamp_ns=100)

    assert count is None
    orch._gemini.count_objects.assert_not_called()


def test_count_target_in_workspace_returns_none_on_gemini_error():
    import numpy as np
    orch = _orchestrator_skeleton()
    orch._verification_frame_timeout_sec = 1.0
    orch._latest_verification_rgb = np.zeros((4, 4, 3), dtype='uint8')
    orch._latest_verification_rgb_stamp_ns = 999
    orch._gemini = MagicMock()
    orch._gemini.count_objects.side_effect = RuntimeError('boom')

    # min_stamp_ns == 0 uses the latest frame regardless of stamp.
    assert orch._count_target_in_workspace('banana', min_stamp_ns=0) is None


def test_run_pipeline_captures_before_count_after_home():
    import json
    import team_8.orchestrator as _orch_mod
    orch = _orchestrator_skeleton()
    orch._pipeline_busy = False
    orch._active_task = ''
    orch._active_task_data = {'object': 'coke_can', 'destination': 'storage_1'}
    orch._segmentation_service_name = '/segmentation/segment_prompt'
    orch._segmentation_service_wait_sec = 0.1
    orch._segmentation_client = MagicMock()
    orch._segmentation_client.wait_for_service.return_value = True
    orch._segmentation_client.call_async.return_value = MagicMock()
    orch._home_before_capture = MagicMock(return_value=777)
    orch._count_target_in_workspace = MagicMock(return_value=2)
    orch._on_segmentation_done = MagicMock()
    mock_stringstring = MagicMock()
    mock_stringstring.Request.return_value = MagicMock(data='')

    with patch.object(_orch_mod, 'StringString', mock_stringstring):
        orch._run_pipeline('pick the coke can')

    # The before-count uses the target name and the home-arrival stamp, and is
    # stored on the active task for the post-task comparison.
    orch._count_target_in_workspace.assert_called_once_with('coke_can', 777)
    assert orch._active_task_data['_before_count'] == 2
