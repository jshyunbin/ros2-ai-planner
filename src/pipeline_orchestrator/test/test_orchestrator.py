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
                        TransformListener=MagicMock(),
                        ActionClient=MagicMock()):
        return CuRobo(node), node


def test_curobo_subscribes_to_four_topics():
    curobo, node = make_curobo()
    topics = [c.args[1] for c in node.create_subscription.call_args_list]
    assert '/camera/camera/depth/color/image_raw' in topics
    assert '/camera/camera/depth/color/camera_info' in topics
    assert '/wrist_camera/wrist_camera/depth/color/image_raw' in topics
    assert '/wrist_camera/wrist_camera/depth/color/camera_info' in topics


def test_curobo_skips_depth_without_camera_info():
    from pipeline_orchestrator.curobo import CuRobo
    node = MagicMock()
    mock_mapper = MagicMock()
    with patch.multiple('pipeline_orchestrator.curobo',
                        Mapper=MagicMock(return_value=mock_mapper),
                        FilterDepth=MagicMock(),
                        MotionPlanner=MagicMock(), Buffer=MagicMock(),
                        TransformListener=MagicMock(),
                        ActionClient=MagicMock()):
        curobo = CuRobo(node)
        curobo._on_depth(MagicMock(), 'overhead', 'camera_color_optical_frame')
        mock_mapper.integrate.assert_not_called()


def test_curobo_plan_trajectory_calls_update_world_after_min_frames():
    from pipeline_orchestrator.curobo import CuRobo, MIN_FRAMES
    node = MagicMock()
    mock_mapper = MagicMock()
    mock_planner = MagicMock()
    mock_planner.plan_pose.return_value = None
    with patch.multiple('pipeline_orchestrator.curobo',
                        Mapper=MagicMock(return_value=mock_mapper),
                        FilterDepth=MagicMock(),
                        MotionPlanner=MagicMock(return_value=mock_planner),
                        Buffer=MagicMock(), TransformListener=MagicMock(),
                        ActionClient=MagicMock()):
        curobo = CuRobo(node)
        curobo._frame_count = MIN_FRAMES
        curobo.plan_trajectory(MagicMock(), MagicMock())
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
