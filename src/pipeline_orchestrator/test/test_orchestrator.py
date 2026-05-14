import pytest


def test_orchestrator_importable():
    from pipeline_orchestrator.orchestrator import PipelineOrchestrator
    assert PipelineOrchestrator is not None


def test_orchestrator_has_task_command_callback():
    from pipeline_orchestrator.orchestrator import PipelineOrchestrator
    assert hasattr(PipelineOrchestrator, 'task_command_callback')
    assert callable(PipelineOrchestrator.task_command_callback)


def test_orchestrator_has_pipeline_methods():
    from pipeline_orchestrator.orchestrator import PipelineOrchestrator
    assert hasattr(PipelineOrchestrator, 'call_sam2')
    assert hasattr(PipelineOrchestrator, 'call_graspgen')
    assert hasattr(PipelineOrchestrator, 'call_curobo')
