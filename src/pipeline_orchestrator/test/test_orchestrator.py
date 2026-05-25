import numpy as np
import pytest
from unittest.mock import MagicMock


# --- GeminiLocalizer ---

def test_gemini_localizer_importable():
    from pipeline_orchestrator.gemini import GeminiLocalizer
    assert GeminiLocalizer is not None


def test_gemini_locate_object_returns_none_stub():
    from pipeline_orchestrator.gemini import GeminiLocalizer
    localizer = GeminiLocalizer(MagicMock())
    result = localizer.locate_object(MagicMock(), "pick up the red cube")
    assert result is None


# --- NvBlox ---

def test_nvblox_importable():
    from pipeline_orchestrator.nvblox import NvBlox
    assert NvBlox is not None


def test_nvblox_get_esdf_returns_none_before_map():
    from pipeline_orchestrator.nvblox import NvBlox
    node = MagicMock()
    nvblox = NvBlox(node)
    assert nvblox.get_esdf() is None


def test_nvblox_extract_object_cloud_returns_none_stub():
    from pipeline_orchestrator.nvblox import NvBlox
    node = MagicMock()
    nvblox = NvBlox(node)
    mask = np.zeros((480, 640), dtype=bool)
    result = nvblox.extract_object_cloud(mask)
    assert result is None


def test_nvblox_registers_esdf_subscription():
    from pipeline_orchestrator.nvblox import NvBlox
    node = MagicMock()
    NvBlox(node)
    assert node.create_subscription.called


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

def test_curobo_accepts_esdf():
    from pipeline_orchestrator.curobo import CuRobo
    curobo = CuRobo(MagicMock())
    result = curobo.plan_trajectory(MagicMock(), MagicMock(), esdf=None)
    assert result is None  # stub


def test_curobo_esdf_optional():
    from pipeline_orchestrator.curobo import CuRobo
    curobo = CuRobo(MagicMock())
    result = curobo.plan_trajectory(MagicMock(), MagicMock())
    assert result is None  # stub


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
