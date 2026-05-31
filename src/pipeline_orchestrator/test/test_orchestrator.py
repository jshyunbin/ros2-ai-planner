import numpy as np
import threading
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


def test_curobo_plan_trajectory_calls_update_world_after_min_frames():
    from pipeline_orchestrator.curobo import CuRobo, MIN_FRAMES
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


def test_orchestrator_plan_execute_calls_curobo_service():
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
    orch._pose_from_grasp_row = MagicMock(return_value=Pose())

    orch._plan_and_execute_best_grasp({"translation": [0, 0, 0], "rotation_matrix": np.eye(3).tolist()})

    orch._curobo_client.call_async.assert_called_once()
    request = orch._curobo_client.call_async.call_args.args[0]
    assert request.grasp_pose is orch._pose_from_grasp_row.return_value
    assert request.joint_state is orch._latest_joints


def test_orchestrator_curobo_done_executes_returned_trajectory():
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    orch = _orchestrator_skeleton()
    orch._execute_trajectory = MagicMock()
    orch._reset_pipeline_state = MagicMock()
    trajectory = JointTrajectory()
    trajectory.points = [JointTrajectoryPoint()]
    future = MagicMock()
    future.result.return_value = MagicMock(
        success=True,
        message="planned",
        trajectory=trajectory,
    )

    orch._on_curobo_done(future)

    orch._execute_trajectory.assert_called_once_with(trajectory)
    orch._reset_pipeline_state.assert_called_once()


def test_curobo_service_plans_with_supplied_joint_state():
    from pipeline_orchestrator.curobo_service import CuRoboService

    node = CuRoboService.__new__(CuRoboService)
    node.get_logger = lambda: MagicMock()
    node._curobo = MagicMock()
    node._init_error = ""
    node._init_lock = threading.Lock()
    node._latest_joints = None
    request = MagicMock()
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
