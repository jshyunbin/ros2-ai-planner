import threading
from types import SimpleNamespace

import pytest

# graspgen_service pulls in the ROS runtime + ZMQ/msgpack client at import time;
# skip the whole module (without breaking collection) where those are
# unavailable. These tests run inside the container.
try:
    from pipeline_orchestrator.graspgen_service import GraspGenService
    _IMPORT_ERROR = None
except Exception as exc:  # noqa: BLE001 - any missing runtime dep should skip
    GraspGenService = None
    _IMPORT_ERROR = exc

pytestmark = pytest.mark.skipif(
    GraspGenService is None,
    reason=f"graspgen_service import unavailable: {_IMPORT_ERROR}",
)


def _service_skeleton():
    """A GraspGenService with only the cloud-sync state the unit tests need."""
    svc = GraspGenService.__new__(GraspGenService)
    svc._cloud_cv = threading.Condition()
    svc._latest_segmented_cloud = None
    svc._latest_segmented_frame = ""
    svc._latest_background_cloud = None
    svc._latest_background_frame = ""
    svc._latest_segmented_stamp_ns = 0
    svc._cloud_wait_sec = 0.2
    return svc


def test_parse_token_empty_means_latest():
    assert GraspGenService._parse_token("") is None
    assert GraspGenService._parse_token("   ") is None
    assert GraspGenService._parse_token(None) is None
    assert GraspGenService._parse_token("123") == 123


def test_parse_token_rejects_non_integer():
    with pytest.raises(ValueError):
        GraspGenService._parse_token("not-a-stamp")


def test_stamp_to_ns_combines_sec_and_nanosec():
    stamp = SimpleNamespace(sec=2, nanosec=500)
    assert GraspGenService._stamp_to_ns(stamp) == 2_000_000_500


def test_wait_for_cloud_times_out_when_absent():
    svc = _service_skeleton()
    assert svc._wait_for_cloud(100) is False


def test_wait_for_cloud_returns_when_matching_stamp_arrives():
    svc = _service_skeleton()
    svc._cloud_wait_sec = 2.0

    def deliver():
        with svc._cloud_cv:
            svc._latest_segmented_stamp_ns = 100
            svc._cloud_cv.notify_all()

    timer = threading.Timer(0.05, deliver)
    timer.start()
    try:
        assert svc._wait_for_cloud(100) is True
    finally:
        timer.join()


def test_run_inference_without_cloud_fails_cleanly():
    svc = _service_skeleton()
    result = svc._run_inference()
    assert result["success"] is False
    assert "No segmented point cloud" in result["error"]


def test_maybe_publish_grasp_poses_builds_ranked_pose_array():
    from unittest.mock import MagicMock
    from builtin_interfaces.msg import Time

    svc = GraspGenService.__new__(GraspGenService)
    pub = MagicMock()
    svc._grasp_poses_pub = pub
    # Real Time() so PoseArray.header.stamp accepts the assignment.
    svc.get_clock = MagicMock(
        return_value=MagicMock(now=lambda: MagicMock(to_msg=lambda: Time()))
    )

    rows = [
        {"translation": [0.1, 0.2, 0.3],
         "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
         "confidence": 0.9},
        {"translation": [0.4, 0.5, 0.6],
         "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
         "confidence": 0.5},
    ]
    svc._maybe_publish_grasp_poses(rows, "base_link")

    pub.publish.assert_called_once()
    msg = pub.publish.call_args.args[0]
    assert msg.header.frame_id == "base_link"
    assert len(msg.poses) == 2
    # Rank order preserved: first row maps to first pose.
    assert msg.poses[0].position.x == pytest.approx(0.1)


def test_maybe_publish_grasp_poses_noop_without_publisher():
    svc = GraspGenService.__new__(GraspGenService)
    svc._grasp_poses_pub = None
    # Must not raise when publishing is disabled.
    svc._maybe_publish_grasp_poses([{"translation": [0, 0, 0],
                                     "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                     "confidence": 1.0}], "base_link")
