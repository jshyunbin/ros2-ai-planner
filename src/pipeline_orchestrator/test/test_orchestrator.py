import numpy as np
import pytest
from unittest.mock import MagicMock, patch


# --- GeminiLocalizer ---

def test_gemini_localizer_importable():
    from pipeline_orchestrator.gemini import GeminiLocalizer
    assert GeminiLocalizer is not None


def test_gemini_locate_object_returns_none_stub():
    from pipeline_orchestrator.gemini import GeminiLocalizer
    localizer = GeminiLocalizer(MagicMock())
    result = localizer.locate_object(MagicMock(), "pick up the red cube")
    assert result is None


# --- Sam2 ---

def test_sam2_segment_accepts_bbox():
    from pipeline_orchestrator.sam2 import Sam2
    sam = Sam2(MagicMock())
    result = sam.segment(MagicMock(), prompt="red cube", bbox=(10, 20, 100, 200))
    assert result is None  # stub


def test_sam2_segment_works_without_bbox():
    from pipeline_orchestrator.sam2 import Sam2
    sam = Sam2(MagicMock())
    result = sam.segment(MagicMock(), prompt="red cube")
    assert result is None  # stub


# --- GraspGen ---

def test_graspgen_accepts_point_cloud():
    from pipeline_orchestrator.graspgen import GraspGen
    graspgen = GraspGen(MagicMock())
    point_cloud = np.zeros((100, 3), dtype=np.float32)
    result = graspgen.generate_grasp(point_cloud)
    assert result is None  # stub


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
    # The orchestrator owns /joint_states and feeds CuRobo via update_joint_state;
    # CuRobo must not open its own duplicate subscription.
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


def test_orchestrator_caches_and_forwards_joints_to_curobo():
    orch = _orchestrator_skeleton()
    orch._curobo = MagicMock()
    msg = MagicMock()

    orch._cache_joints(msg)

    assert orch._latest_joints is msg
    orch._curobo.update_joint_state.assert_called_once_with(msg)


def test_orchestrator_caches_joints_without_curobo_when_motion_disabled():
    orch = _orchestrator_skeleton()
    orch._curobo = None
    msg = MagicMock()

    orch._cache_joints(msg)

    assert orch._latest_joints is msg


def test_orchestrator_plan_execute_uses_curobo_first():
    orch = _orchestrator_skeleton()
    orch._latest_joints = MagicMock()
    orch._curobo = MagicMock()
    orch._moveit2 = MagicMock()
    orch._execute_trajectory = MagicMock()
    orch._pose_from_grasp_row = MagicMock(return_value=MagicMock())
    orch._curobo.plan_trajectory.return_value = MagicMock(points=[MagicMock()])

    orch._plan_and_execute_best_grasp({"translation": [0, 0, 0], "rotation_matrix": np.eye(3).tolist()})

    orch._curobo.plan_trajectory.assert_called_once()
    orch._moveit2.plan_trajectory.assert_not_called()
    orch._execute_trajectory.assert_called_once()


def test_orchestrator_plan_execute_falls_back_to_moveit2():
    orch = _orchestrator_skeleton()
    orch._latest_joints = MagicMock()
    orch._curobo = MagicMock()
    orch._moveit2 = MagicMock()
    orch._execute_trajectory = MagicMock()
    orch._pose_from_grasp_row = MagicMock(return_value=MagicMock())
    orch._curobo.plan_trajectory.return_value = None
    orch._moveit2.plan_trajectory.return_value = MagicMock(points=[MagicMock()])

    orch._plan_and_execute_best_grasp({"translation": [0, 0, 0], "rotation_matrix": np.eye(3).tolist()})

    orch._curobo.plan_trajectory.assert_called_once()
    orch._moveit2.plan_trajectory.assert_called_once()
    orch._execute_trajectory.assert_called_once()


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
