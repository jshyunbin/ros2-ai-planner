import pytest


def test_graspgen_node_importable():
    from graspgen_node.graspgen_node import GraspGenNode
    assert GraspGenNode is not None


def test_graspgen_node_has_generate_grasp_callback():
    from graspgen_node.graspgen_node import GraspGenNode
    assert hasattr(GraspGenNode, 'generate_grasp_callback')
    assert callable(GraspGenNode.generate_grasp_callback)
