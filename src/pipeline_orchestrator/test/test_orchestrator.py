import pytest


def test_orchestrator_importable():
    from pipeline_orchestrator.orchestrator import PipelineOrchestrator
    assert PipelineOrchestrator is not None


def test_orchestrator_has_task_command_callback():
    from pipeline_orchestrator.orchestrator import PipelineOrchestrator
    assert hasattr(PipelineOrchestrator, 'task_command_callback')
    assert callable(PipelineOrchestrator.task_command_callback)


def test_orchestrator_has_run_pipeline():
    from pipeline_orchestrator.orchestrator import PipelineOrchestrator
    assert hasattr(PipelineOrchestrator, '_run_pipeline')
    assert callable(PipelineOrchestrator._run_pipeline)
