import threading
from types import SimpleNamespace

import pytest

# segmentation_service imports cv2 / cv_bridge / rclpy / genai at module load;
# skip the whole module (without breaking collection) where those are
# unavailable. These tests run inside the container.
try:
    from team_8.segmentation_service import SegmentationService
    _IMPORT_ERROR = None
except Exception as exc:  # noqa: BLE001 - any missing runtime dep should skip
    SegmentationService = None
    _IMPORT_ERROR = exc

pytestmark = pytest.mark.skipif(
    SegmentationService is None,
    reason=f"segmentation_service import unavailable: {_IMPORT_ERROR}",
)


def test_parse_request_json_extracts_prompt_and_stamp():
    prompt, min_stamp_ns = SegmentationService._parse_request(
        '{"prompt": "pick the mug", "min_stamp_ns": 123}')
    assert prompt == "pick the mug"
    assert min_stamp_ns == 123


def test_parse_request_plain_string_is_prompt_without_gate():
    # Back-compat: a non-JSON string is the whole prompt, no freshness gate.
    prompt, min_stamp_ns = SegmentationService._parse_request("pick the mug")
    assert prompt == "pick the mug"
    assert min_stamp_ns == 0


def test_parse_request_blank_returns_empty_no_gate():
    assert SegmentationService._parse_request("") == ("", 0)
    assert SegmentationService._parse_request("   ") == ("", 0)
    assert SegmentationService._parse_request(None) == ("", 0)


def test_parse_request_non_dict_json_is_prompt_without_gate():
    # "123" is valid JSON (an int) but not a request dict; treat as a prompt.
    prompt, min_stamp_ns = SegmentationService._parse_request("123")
    assert prompt == "123"
    assert min_stamp_ns == 0


def test_parse_request_invalid_stamp_falls_back_to_no_gate():
    # A corrupt min_stamp_ns must disable the gate (return 0), not crash.
    prompt, min_stamp_ns = SegmentationService._parse_request(
        '{"prompt": "pick the mug", "min_stamp_ns": "abc"}')
    assert prompt == "pick the mug"
    assert min_stamp_ns == 0


def _service_skeleton():
    """A SegmentationService with only the freshness-gate state the unit tests
    need (mirrors the GraspGen test skeleton; __init__ builds real ROS handles
    and warms up Gemini/SAM2, so build a bare instance via __new__)."""
    svc = SegmentationService.__new__(SegmentationService)
    svc._frame_cv = threading.Condition()
    svc._latest_rgb_stamp_ns = 0
    svc._latest_depth_stamp_ns = 0
    svc._fresh_frame_timeout_sec = 0.2
    return svc


def test_wait_for_fresh_frames_times_out_when_stale():
    svc = _service_skeleton()
    # Both stamps are at/under the gate -> never satisfied -> timeout.
    svc._latest_rgb_stamp_ns = 100
    svc._latest_depth_stamp_ns = 100
    assert svc._wait_for_fresh_frames(100) is False


def test_wait_for_fresh_frames_returns_when_both_fresh():
    svc = _service_skeleton()
    svc._latest_rgb_stamp_ns = 101
    svc._latest_depth_stamp_ns = 101
    assert svc._wait_for_fresh_frames(100) is True


def test_wait_for_fresh_frames_waits_for_lagging_depth():
    svc = _service_skeleton()
    svc._fresh_frame_timeout_sec = 2.0
    svc._latest_rgb_stamp_ns = 101  # rgb already fresh
    svc._latest_depth_stamp_ns = 50  # depth still stale

    def _deliver_depth():
        with svc._frame_cv:
            svc._latest_depth_stamp_ns = 101
            svc._frame_cv.notify_all()

    timer = threading.Timer(0.05, _deliver_depth)
    timer.start()
    try:
        assert svc._wait_for_fresh_frames(100) is True
    finally:
        timer.join()
