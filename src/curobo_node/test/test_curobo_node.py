import pytest


def test_curobo_node_importable():
    from curobo_node.curobo_node import CuRoboNode
    assert CuRoboNode is not None


def test_curobo_node_has_plan_trajectory_callback():
    from curobo_node.curobo_node import CuRoboNode
    assert hasattr(CuRoboNode, 'plan_trajectory_callback')
    assert callable(CuRoboNode.plan_trajectory_callback)
