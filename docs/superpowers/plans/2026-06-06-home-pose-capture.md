# Home-Pose Wrist Capture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the arm to the `home` pose at the start of every `/task_commands` cycle and segment from a wrist RGB-D frame captured *after* the arm has settled there, so each pick-and-place sees the current scene rather than a stale, arbitrary-viewpoint frame.

**Architecture:** The orchestrator front-loads a collision-aware home move into `_run_pipeline`, records the home-arrival ROS time, and passes it as a freshness gate to the segmentation service inside a JSON request. The segmentation service (now under a `MultiThreadedExecutor` with the camera subscriptions in a `ReentrantCallbackGroup`) blocks the service handler until both RGB and depth frames are stamped after that time, or times out. This mirrors the existing GraspGen cloud-sync pattern (`_wait_for_cloud` + condition variable + reentrant group + multithreaded executor).

**Tech Stack:** ROS2 Humble (rclpy, `riro_srvs/StringString`), `threading.Condition`, `MultiThreadedExecutor`, `ReentrantCallbackGroup`, pytest. `use_sim_time=True` on both nodes (set in `pipeline_common.launch.py`), so the orchestrator clock and camera header stamps share the same time base.

---

## Test & Build Workflow

`riro_srvs`, `rclpy`, `geometry_msgs`, `cv_bridge`, `genai` are **not importable on the host**. **All `pytest` commands run inside the container**, where `./src` is live-mounted at `/ros2_ws/src`:

```bash
# from the repo root, open a container shell (debug override = live-mounted src)
docker compose -f docker-compose.yml -f docker-compose.debug.yml run --rm ai_planner bash
# then, inside the container:
cd /ros2_ws
source install/setup.bash
python3 -m pytest src/team_8/test/<file> -v
```

This change touches **only Python node code** (no `.srv`, no entry points, no launch files), so no `colcon build` is required — `source install/setup.bash` then run pytest directly. The new test file `test_segmentation_service.py` guards its imports and skips cleanly when the ROS runtime is absent (mirroring `test_graspgen_service.py`), so it can at least be *collected* on the host.

---

## File Structure

- **Modify** `src/team_8/team_8/segmentation_service.py` — add request parsing (`_parse_request`), a frame-freshness condition variable + wait (`_wait_for_fresh_frames`), notify in the RGB/depth callbacks, a reentrant callback group for the camera subscriptions, the gate check in `_handle_request`, and a `MultiThreadedExecutor` in `main()`.
- **Modify** `src/team_8/team_8/orchestrator.py` — add `_home_before_capture()` and front-load it into `_run_pipeline`, sending segmentation a JSON request `{prompt, min_stamp_ns}`.
- **Create** `src/team_8/test/test_segmentation_service.py` — unit tests for `_parse_request`, `_wait_for_fresh_frames`, and the handler gate-timeout branch.
- **Modify** `src/team_8/test/test_orchestrator.py` — add tests for `_home_before_capture` and the home-before-segment ordering in `_run_pipeline`.
- **Modify** `.claude/CLAUDE.md` — update the orchestrator + segmentation_service rows to describe home-before-capture and the freshness gate.

---

## Task 1: Segmentation request parsing (`_parse_request`)

**Files:**
- Create: `src/team_8/test/test_segmentation_service.py`
- Modify: `src/team_8/team_8/segmentation_service.py`

- [ ] **Step 1: Write the failing test**

Create `src/team_8/test/test_segmentation_service.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest src/team_8/test/test_segmentation_service.py -v`
Expected: FAIL with `AttributeError: ... has no attribute '_parse_request'`.

- [ ] **Step 3: Implement `_parse_request`**

In `src/team_8/team_8/segmentation_service.py`, add this staticmethod to the `SegmentationService` class (place it next to `_stamp_to_ns` near the end of the class):

```python
    @staticmethod
    def _parse_request(data) -> tuple[str, int]:
        """Parse a segmentation request into (prompt, min_stamp_ns).

        The request `data` is normally a JSON object
        ``{"prompt": "...", "min_stamp_ns": <int>}``. For back-compat, a value
        that does not parse as a JSON *dict* (e.g. a bare prompt string sent by
        standalone callers) is treated as the prompt with no freshness gate.
        """
        text = (data or "").strip()
        if not text:
            return "", 0
        try:
            payload = json.loads(text)
        except (TypeError, ValueError):
            return text, 0
        if not isinstance(payload, dict):
            return text, 0
        prompt = str(payload.get("prompt", "")).strip()
        try:
            min_stamp_ns = int(payload.get("min_stamp_ns", 0) or 0)
        except (TypeError, ValueError):
            min_stamp_ns = 0
        return prompt, min_stamp_ns
```

(`json` is already imported at the top of the file.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest src/team_8/test/test_segmentation_service.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/team_8/test/test_segmentation_service.py src/team_8/team_8/segmentation_service.py
git commit -m "Add segmentation request JSON parsing (prompt + freshness gate)

Written By: Claude Sonnet 4.6"
```

---

## Task 2: Frame-freshness condition variable + wait (`_wait_for_fresh_frames`)

**Files:**
- Modify: `src/team_8/team_8/segmentation_service.py`
- Test: `src/team_8/test/test_segmentation_service.py`

- [ ] **Step 1: Write the failing tests**

Append to `src/team_8/test/test_segmentation_service.py`:

```python
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

    threading.Timer(0.05, _deliver_depth).start()
    assert svc._wait_for_fresh_frames(100) is True
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest src/team_8/test/test_segmentation_service.py -k wait_for_fresh -v`
Expected: FAIL with `AttributeError: ... has no attribute '_wait_for_fresh_frames'`.

- [ ] **Step 3: Implement the condition variable, callback notifies, and the wait**

In `src/team_8/team_8/segmentation_service.py`:

(a) Extend the `pipeline_utils` import (currently `from team_8.pipeline_utils import make_xyz_cloud`) to also import `env_float`:

```python
from team_8.pipeline_utils import env_float, make_xyz_cloud
```

(b) Add `threading` to the imports at the top of the file (after `import time`):

```python
import threading
```

(c) In `__init__`, immediately after `self._bridge = CvBridge()` and before the `self._latest_rgb = None` block, create the condition variable and read the timeout env var. Then leave the existing `_latest_*` initializers as they are:

```python
        # _frame_cv guards the latest RGB/depth stamps and is notified whenever a
        # new frame arrives, so a service handler blocked in _wait_for_fresh_frames
        # wakes as soon as a post-home frame lands.
        self._frame_cv = threading.Condition()
        self._fresh_frame_timeout_sec = env_float(
            "PIPELINE_SEG_FRESH_FRAME_TIMEOUT_SEC", 5.0)
```

(d) Replace the RGB callback body so the cache update + stamp happen under the condition with a notify:

```python
    def _rgb_callback(self, msg: Image) -> None:
        rgb = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        stamp_ns = self._stamp_to_ns(msg.header.stamp)
        with self._frame_cv:
            self._latest_rgb = rgb
            self._latest_rgb_stamp_ns = stamp_ns
            self._frame_cv.notify_all()
        if not self._logged_first_rgb:
            self.get_logger().info(
                f"Received first RGB frame {msg.width}x{msg.height} on "
                f"{self.get_parameter('rgb_topic').value}"
            )
            self._logged_first_rgb = True
```

(e) Replace the depth callback body so its cache update + stamps happen under the condition with a notify (keep the encoding handling):

```python
    def _depth_callback(self, msg: Image) -> None:
        if msg.encoding == "16UC1":
            depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1").astype(np.float32)
            depth *= float(self.get_parameter("depth_unit_scale").value)
        else:
            depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1").astype(np.float32)

        stamp = msg.header.stamp
        stamp_ns = self._stamp_to_ns(stamp)
        frame_id = msg.header.frame_id
        with self._frame_cv:
            self._latest_depth = depth
            self._latest_depth_stamp = stamp
            self._latest_depth_stamp_ns = stamp_ns
            self._latest_frame_id = frame_id
            self._frame_cv.notify_all()
        if not self._logged_first_depth:
            self.get_logger().info(
                f"Received first depth frame {msg.width}x{msg.height} "
                f"frame={frame_id} on {self.get_parameter('depth_topic').value}"
            )
            self._logged_first_depth = True
```

(f) Add the wait method (place it next to `_parse_request`):

```python
    def _wait_for_fresh_frames(self, min_stamp_ns: int) -> bool:
        """Block until both the RGB and depth caches hold a frame stamped after
        *min_stamp_ns*, or until PIPELINE_SEG_FRESH_FRAME_TIMEOUT_SEC elapses.

        Used to guarantee segmentation runs on a frame captured *after* the arm
        settled at home, not a stale mid-transit frame.
        """
        deadline = time.monotonic() + self._fresh_frame_timeout_sec
        with self._frame_cv:
            while (self._latest_rgb_stamp_ns <= min_stamp_ns
                   or self._latest_depth_stamp_ns <= min_stamp_ns):
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                self._frame_cv.wait(timeout=remaining)
            return True
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest src/team_8/test/test_segmentation_service.py -v`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add src/team_8/team_8/segmentation_service.py src/team_8/test/test_segmentation_service.py
git commit -m "Segmentation: condition-guarded frame stamps + freshness wait

Written By: Claude Sonnet 4.6"
```

---

## Task 3: Wire the gate into the handler + concurrency (reentrant group + MultiThreadedExecutor)

**Files:**
- Modify: `src/team_8/team_8/segmentation_service.py`
- Test: `src/team_8/test/test_segmentation_service.py`

- [ ] **Step 1: Write the failing test (gate-timeout branch)**

Append to `src/team_8/test/test_segmentation_service.py`:

```python
def test_handle_request_returns_timeout_when_frame_never_fresh():
    import json
    svc = _service_skeleton()
    svc.get_logger = lambda: SimpleNamespace(
        info=lambda *a, **k: None, warn=lambda *a, **k: None,
        error=lambda *a, **k: None)
    # Frames exist, but the freshness gate is never satisfied.
    svc._latest_rgb = object()
    svc._latest_depth = object()
    svc._wait_for_fresh_frames = lambda min_stamp_ns: False

    request = SimpleNamespace(
        data='{"prompt": "pick the mug", "min_stamp_ns": 999}')
    response = SimpleNamespace(data=None)

    result = svc._handle_request(request, response)

    payload = json.loads(result.data)
    assert payload["success"] is False
    assert "999" in payload["error"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest src/team_8/test/test_segmentation_service.py -k timeout_when_frame_never_fresh -v`
Expected: FAIL — the current `_handle_request` parses `request.data.strip()` as the raw prompt and never consults a freshness gate, so it proceeds past the gate (and then errors on the mocked frames / missing Gemini), not returning the `"999"` timeout payload.

- [ ] **Step 3: Wire the gate into `_handle_request`**

In `src/team_8/team_8/segmentation_service.py`, replace the top of `_handle_request` — the current:

```python
    def _handle_request(self, request: StringString.Request, response: StringString.Response):
        prompt = request.data.strip()
        if not prompt:
            prompt = "Pick the requested object."

        if self._latest_rgb is None or self._latest_depth is None:
```

with:

```python
    def _handle_request(self, request: StringString.Request, response: StringString.Response):
        prompt, min_stamp_ns = self._parse_request(request.data)
        if not prompt:
            prompt = "Pick the requested object."

        # When the orchestrator passes a freshness gate (the home-arrival time),
        # wait for a wrist frame captured after the arm settled at home so we
        # never segment a stale mid-transit frame.
        if min_stamp_ns > 0 and not self._wait_for_fresh_frames(min_stamp_ns):
            response.data = json.dumps({
                "success": False,
                "error": (
                    f"Timed out waiting for wrist frames newer than {min_stamp_ns} ns "
                    f"(waited {self._fresh_frame_timeout_sec:.1f}s)."
                ),
            })
            return response

        if self._latest_rgb is None or self._latest_depth is None:
```

(The rest of `_handle_request` is unchanged.)

- [ ] **Step 4: Put the camera subscriptions in a reentrant group**

Add the import near the other rclpy imports at the top of the file:

```python
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
```

In `__init__`, replace the three `self.create_subscription(...)` calls (rgb, depth, camera_info) with versions that share a reentrant callback group, so the camera callbacks keep firing (under the MultiThreadedExecutor) while the service handler is blocked in `_wait_for_fresh_frames`:

```python
        # Camera callbacks share a reentrant group so they can keep updating the
        # latest frames while a service handler blocks in _wait_for_fresh_frames
        # (the node is spun with a MultiThreadedExecutor in main()).
        camera_group = ReentrantCallbackGroup()
        self.create_subscription(
            Image, str(self.get_parameter("rgb_topic").value), self._rgb_callback, qos,
            callback_group=camera_group,
        )
        self.create_subscription(
            Image,
            str(self.get_parameter("depth_topic").value),
            self._depth_callback,
            qos,
            callback_group=camera_group,
        )
        self.create_subscription(
            CameraInfo,
            str(self.get_parameter("camera_info_topic").value),
            self._camera_info_callback,
            qos,
            callback_group=camera_group,
        )
```

- [ ] **Step 5: Switch `main()` to a MultiThreadedExecutor**

Replace the `main()` function at the bottom of the file:

```python
def main(args=None) -> None:
    rclpy.init(args=args)
    node = SegmentationService()
    # MultiThreadedExecutor so the service handler can block in
    # _wait_for_fresh_frames while the camera callbacks (reentrant group) keep
    # delivering frames on another thread.
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
```

- [ ] **Step 6: Run the full segmentation test file**

Run: `python3 -m pytest src/team_8/test/test_segmentation_service.py -v`
Expected: PASS (8 passed).

- [ ] **Step 7: Commit**

```bash
git add src/team_8/team_8/segmentation_service.py src/team_8/test/test_segmentation_service.py
git commit -m "Segmentation: gate handler on fresh frame; reentrant cams + MT executor

Written By: Claude Sonnet 4.6"
```

---

## Task 4: Orchestrator `_home_before_capture()`

**Files:**
- Modify: `src/team_8/team_8/orchestrator.py`
- Test: `src/team_8/test/test_orchestrator.py`

- [ ] **Step 1: Write the failing tests**

Append to `src/team_8/test/test_orchestrator.py` (the file already imports `MagicMock` and defines `_orchestrator_skeleton`):

```python
def test_orchestrator_home_before_capture_moves_home_and_returns_clock():
    orch = _orchestrator_skeleton()
    orch._enable_motion_execution = True
    orch._curobo_client = MagicMock()
    orch._latest_joints = MagicMock()
    orch._send_gripper = MagicMock()
    orch._plan_and_execute_home = MagicMock()
    orch.get_clock = MagicMock(
        return_value=MagicMock(now=lambda: MagicMock(nanoseconds=999)))

    stamp = orch._home_before_capture()

    orch._send_gripper.assert_called_once_with(closed=False)
    orch._plan_and_execute_home.assert_called_once_with()
    assert stamp == 999


def test_orchestrator_home_before_capture_skips_when_motion_disabled():
    orch = _orchestrator_skeleton()
    orch._enable_motion_execution = False
    orch._curobo_client = None
    orch._plan_and_execute_home = MagicMock()

    assert orch._home_before_capture() == 0
    orch._plan_and_execute_home.assert_not_called()


def test_orchestrator_home_before_capture_skips_without_joints():
    orch = _orchestrator_skeleton()
    orch._enable_motion_execution = True
    orch._curobo_client = MagicMock()
    orch._latest_joints = None
    orch._plan_and_execute_home = MagicMock()

    assert orch._home_before_capture() == 0
    orch._plan_and_execute_home.assert_not_called()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest src/team_8/test/test_orchestrator.py -k home_before_capture -v`
Expected: FAIL with `AttributeError: ... has no attribute '_home_before_capture'`.

- [ ] **Step 3: Implement `_home_before_capture`**

In `src/team_8/team_8/orchestrator.py`, add this method just above `_plan_and_execute_home` (so the two home helpers sit together):

```python
    def _home_before_capture(self) -> int:
        """Move the arm to home so the wrist camera observes the workspace, and
        return the ROS time (ns) at which it settled.

        Returns 0 when motion execution is disabled or joints/CuRobo are not yet
        available — meaning no home move and no freshness gate, so GraspGen-only
        and standalone modes keep working.
        """
        if not self._enable_motion_execution or self._curobo_client is None:
            return 0
        if self._latest_joints is None:
            self.get_logger().warn(
                'No /joint_states yet; skipping initial home move and '
                'frame freshness gate.')
            return 0
        # Open the gripper: nothing should be carried at the start of a cycle, and
        # the next grasp assumes open fingers.
        self._send_gripper(closed=False)
        self._plan_and_execute_home()
        return self.get_clock().now().nanoseconds
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest src/team_8/test/test_orchestrator.py -k home_before_capture -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/team_8/team_8/orchestrator.py src/team_8/test/test_orchestrator.py
git commit -m "Orchestrator: add _home_before_capture (home move + arrival stamp)

Written By: Claude Sonnet 4.6"
```

---

## Task 5: Front-load home + send freshness-gated segmentation request

**Files:**
- Modify: `src/team_8/team_8/orchestrator.py`
- Test: `src/team_8/test/test_orchestrator.py`

- [ ] **Step 1: Write the failing test**

Append to `src/team_8/test/test_orchestrator.py`:

```python
def test_orchestrator_run_pipeline_homes_before_segmenting():
    import json
    orch = _orchestrator_skeleton()
    orch._pipeline_busy = False
    orch._active_task = ''
    orch._segmentation_service_name = '/segmentation/segment_prompt'
    orch._segmentation_service_wait_sec = 0.1
    calls = []
    orch._segmentation_client = MagicMock()
    orch._segmentation_client.wait_for_service.return_value = True
    orch._segmentation_client.call_async.side_effect = (
        lambda req: calls.append(('segment', req)) or MagicMock())
    orch._home_before_capture = MagicMock(
        side_effect=lambda: calls.append(('home', None)) or 555)
    orch._on_segmentation_done = MagicMock()

    orch._run_pipeline('pick the mug')

    # Home move happens before segmentation is requested.
    assert [c[0] for c in calls] == ['home', 'segment']
    request = calls[1][1]
    payload = json.loads(request.data)
    assert payload == {'prompt': 'pick the mug', 'min_stamp_ns': 555}


def test_orchestrator_run_pipeline_aborts_when_home_fails():
    orch = _orchestrator_skeleton()
    orch._pipeline_busy = False
    orch._active_task = ''
    orch._segmentation_service_name = '/segmentation/segment_prompt'
    orch._segmentation_service_wait_sec = 0.1
    orch._segmentation_client = MagicMock()
    orch._segmentation_client.wait_for_service.return_value = True
    orch._home_before_capture = MagicMock(side_effect=RuntimeError('no plan'))
    orch._reset_pipeline_state = MagicMock()

    orch._run_pipeline('pick the mug')

    orch._segmentation_client.call_async.assert_not_called()
    orch._reset_pipeline_state.assert_called_once()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest src/team_8/test/test_orchestrator.py -k run_pipeline_homes_before_segmenting -v`
Expected: FAIL — today `_run_pipeline` sends `request.data = task` (a raw string) and never calls `_home_before_capture`, so `calls` is `['segment']` and `json.loads(request.data)` raises.

- [ ] **Step 3: Restructure `_run_pipeline`**

In `src/team_8/team_8/orchestrator.py`, replace the tail of `_run_pipeline` — the current:

```python
        self._pipeline_busy = True
        self._active_task = task

        request = StringString.Request()
        request.data = task
        future = self._segmentation_client.call_async(request)
        future.add_done_callback(self._on_segmentation_done)
        self.get_logger().info(f'Started segmentation for task: {task}')
```

with:

```python
        self._pipeline_busy = True
        self._active_task = task

        # Move to home first so the wrist camera observes the workspace, then
        # gate segmentation on a frame captured after the arm settled there.
        try:
            min_stamp_ns = self._home_before_capture()
        except Exception as exc:
            self.get_logger().error(f'Initial home move failed: {exc}')
            self._reset_pipeline_state()
            return

        request = StringString.Request()
        request.data = json.dumps({'prompt': task, 'min_stamp_ns': min_stamp_ns})
        future = self._segmentation_client.call_async(request)
        future.add_done_callback(self._on_segmentation_done)
        self.get_logger().info(
            f'Started segmentation for task: {task} (min_stamp_ns={min_stamp_ns})')
```

(`json` is already imported at the top of `orchestrator.py`.)

- [ ] **Step 4: Run the new tests to verify they pass**

Run: `python3 -m pytest src/team_8/test/test_orchestrator.py -k "run_pipeline_homes_before_segmenting or run_pipeline_aborts_when_home_fails" -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Run the full orchestrator + segmentation test files (no regressions)**

Run: `python3 -m pytest src/team_8/test/test_orchestrator.py src/team_8/test/test_segmentation_service.py -v`
Expected: PASS (all green; the existing orchestrator tests still pass).

- [ ] **Step 6: Commit**

```bash
git add src/team_8/team_8/orchestrator.py src/team_8/test/test_orchestrator.py
git commit -m "Orchestrator: home before each capture; send freshness-gated request

Written By: Claude Sonnet 4.6"
```

---

## Task 6: Documentation

**Files:**
- Modify: `.claude/CLAUDE.md`

- [ ] **Step 1: Update the orchestrator row**

In `.claude/CLAUDE.md`, in the Nodes table, append to the `orchestrator.py` row description a sentence describing the new behavior:

> At the start of each `/task_commands` cycle it first opens the gripper and moves the arm to the `home` pose (collision-aware), then captures the home-arrival time and passes it to segmentation as a freshness gate so each pick-and-place segments a wrist frame taken *after* the arm settled at home.

- [ ] **Step 2: Update the segmentation_service row**

Append to the `segmentation_service.py` row description:

> The request is a JSON object `{prompt, min_stamp_ns}` (a bare prompt string still works); when `min_stamp_ns > 0` the service blocks until RGB+depth frames are stamped after that time (`PIPELINE_SEG_FRESH_FRAME_TIMEOUT_SEC`, default 5s) before segmenting. Runs under a `MultiThreadedExecutor` with the camera subscriptions in a reentrant group so frames keep arriving while the handler waits.

- [ ] **Step 3: Commit**

```bash
git add .claude/CLAUDE.md
git commit -m "Docs: home-before-capture + segmentation freshness gate

Written By: Claude Sonnet 4.6"
```

---

## Self-Review

**Spec coverage:**
- Spec §1 (orchestrator home before capture) → Tasks 4 + 5.
- Spec §2 (JSON request carrying the gate) → Task 1 (parse) + Task 5 (orchestrator sends it).
- Spec §3 (service waits for post-arrival frame; reentrant group + MultiThreadedExecutor concurrency fix) → Tasks 2 + 3.
- Spec §4 (config `PIPELINE_SEG_FRESH_FRAME_TIMEOUT_SEC`) → Task 2, Step 3(c).
- Spec error handling (startup home fail; fresh-frame timeout; gripper open non-fatal) → Task 5 (`_run_pipeline` try/except) + Task 3 (timeout payload) + reuse of existing `_send_gripper` swallow.
- Spec back-compat (standalone callers send a raw prompt) → Task 1 tests + `_parse_request` non-dict path.

**Placeholder scan:** No TBD/TODO; every code step shows complete code.

**Type/name consistency:** `_parse_request` returns `(prompt, min_stamp_ns)` and is consumed as such in Task 3. `_wait_for_fresh_frames(min_stamp_ns) -> bool` defined in Task 2, called in Task 3 and the handler test. `_home_before_capture() -> int` defined in Task 4, called in Task 5 and asserted with `min_stamp_ns`. `_frame_cv` / `_fresh_frame_timeout_sec` / `_latest_rgb_stamp_ns` / `_latest_depth_stamp_ns` names match across the callbacks, the wait, and the test skeleton. `self.get_clock().now().nanoseconds` matches the orchestrator test's mock shape.
```
