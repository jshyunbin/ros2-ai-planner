import pytest


def test_sam2_node_importable():
    from sam2_node.sam2_node import Sam2Node
    assert Sam2Node is not None


def test_sam2_node_has_segment_service():
    from sam2_node.sam2_node import Sam2Node
    import inspect
    assert hasattr(Sam2Node, 'segment_callback')
    assert callable(Sam2Node.segment_callback)
