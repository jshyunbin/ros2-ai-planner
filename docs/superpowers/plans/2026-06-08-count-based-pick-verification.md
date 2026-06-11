# Count-based Pick Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify a pick succeeded by checking that the count of the target object type in the source workspace dropped, instead of asking whether the type is still present (which never clears when duplicates exist).

**Architecture:** Replace `GeminiAPI.verify_object_removed` (boolean presence) with `GeminiAPI.count_objects` (integer count). The orchestrator captures a per-attempt "before" count after the initial home move and an "after" count in post-task verification; success means `after < before`. An unobtainable count (Gemini error / no fresh frame) succeeds without retry.

**Tech Stack:** Python 3, ROS2 Humble (rclpy), Google GenAI SDK (Gemini), pytest, unittest.mock.

---

## Background for the implementer

- `src/team_8/team_8/gemini_api.py` wraps Gemini. Methods return plain dicts and validate the JSON via response schemas. Heavy client calls go through `_generate_json`. `GeminiAPI.__init__` requires a real API key + the `google-genai` package, so unit tests build a bare instance with `GeminiAPI.__new__(GeminiAPI)` and patch `_generate_json`.
- `src/team_8/team_8/orchestrator.py` is the pipeline driver. Post-task verification lives in `_run_post_task_verification` (around line 762). The orchestrator already caches the wrist-camera RGB in `self._latest_verification_rgb` / `self._latest_verification_rgb_stamp_ns` via `_cache_verification_rgb`.
- Tests build orchestrators with `_orchestrator_skeleton()` (`__new__`, no `__init__`) and inject only what each test needs. See `src/team_8/test/test_orchestrator.py`.
- `time`, `json`, and `PILImage` are already imported at the top of `orchestrator.py`. `re`, `time`, `json` are imported in `gemini_api.py`.
- The camera subscription is in the default callback group while the pipeline blocks in the reentrant `_pipeline_cbg` under a `MultiThreadedExecutor`, so a bounded blocking wait inside the pipeline still receives fresh frames.

**Run tests from the package dir:** `cd src/team_8 && python -m pytest test/<file> -v`
(If `team_8` import fails, prefix with `PYTHONPATH=.`.)

---

## Task 1: `count_objects` in the Gemini wrapper

**Files:**
- Modify: `src/team_8/team_8/gemini_api.py`
- Test: `src/team_8/test/test_gemini_api.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `src/team_8/test/test_gemini_api.py`:

```python
from unittest.mock import MagicMock

import pytest


def _bare_api():
    """A GeminiAPI without __init__ (no real client/key needed)."""
    from team_8.gemini_api import GeminiAPI
    api = GeminiAPI.__new__(GeminiAPI)
    api._logger = None
    return api


def test_count_objects_returns_count_and_reason():
    api = _bare_api()
    api._generate_json = MagicMock(
        return_value={"count": 2, "reason": "two coke cans on the table"})
    image = MagicMock()
    image.size = (640, 480)

    result = api.count_objects(image, object_name="coke_can")

    assert result == {"count": 2, "reason": "two coke cans on the table"}
    # The image and a prompt mentioning the display name are sent to Gemini.
    contents = api._generate_json.call_args.kwargs["contents"]
    assert image in contents
    assert any("coke can" in str(c) for c in contents)


def test_count_objects_clamps_negative_to_zero():
    api = _bare_api()
    api._generate_json = MagicMock(return_value={"count": -3, "reason": "x"})
    image = MagicMock(); image.size = (1, 1)

    assert api.count_objects(image, object_name="banana")["count"] == 0


def test_count_objects_rejects_non_integer_count():
    from team_8.gemini_api import GeminiAPIError
    api = _bare_api()
    api._generate_json = MagicMock(return_value={"count": "lots", "reason": "x"})
    image = MagicMock(); image.size = (1, 1)

    with pytest.raises(GeminiAPIError):
        api.count_objects(image, object_name="banana")


def test_count_objects_requires_pil_image():
    api = _bare_api()
    with pytest.raises(TypeError):
        api.count_objects(object(), object_name="banana")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd src/team_8 && PYTHONPATH=. python -m pytest test/test_gemini_api.py -v`
Expected: FAIL — `AttributeError: ... has no attribute 'count_objects'`.

- [ ] **Step 3: Add the schema and method, remove the old verifier**

In `src/team_8/team_8/gemini_api.py`, **replace** the `TASK_VERIFICATION_SCHEMA` block (the `TASK_VERIFICATION_SCHEMA = { ... }` assignment, currently lines ~60-84) with:

```python
OBJECT_COUNT_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "count": {
            "type": "INTEGER",
            "description": (
                "How many instances of the target object type are visible in "
                "the source pickup workspace. 0 if none remain."
            ),
        },
        "reason": {
            "type": "STRING",
            "description": "Brief visual reason for the count.",
        },
    },
    "required": ["count", "reason"],
}
```

**Delete** the entire `verify_object_removed` method (currently ~lines 171-253) and **add** this method in its place (same indentation, inside `class GeminiAPI`):

```python
    def count_objects(
        self,
        pil_image: Any,
        *,
        object_name: str,
    ) -> dict[str, Any]:
        """Count how many instances of object_name remain in the pickup area.

        The image must be captured with the arm at home so the wrist camera sees
        the source workspace. Objects already placed in the destination
        basket/storage/bookshelf are excluded, as are all non-target objects, so
        the count reflects only target-type instances still awaiting pickup.
        """
        if not hasattr(pil_image, "size"):
            raise TypeError(
                "count_objects expects a PIL image with a size attribute."
            )

        normalized_object = _normalize_object_name(object_name)
        if not normalized_object:
            raise ValueError("Count object name is empty.")

        display_name = normalized_object.replace("_", " ")

        prompt = f"""
You are counting objects after a robotic pick-and-place task.

Target object type: {display_name}

The image was captured with the robot arm at its home pose, looking down at the
source pickup workspace (the main table/work area where loose objects are
picked up).

Count how many instances of the target object type are STILL PRESENT IN THE
SOURCE PICKUP WORKSPACE.

Important rules:
1. Ignore the robot arm and gripper.
2. Do not count instances that are inside the destination basket, storage area,
   or bookshelf. Count only instances loose in the pickup workspace.
3. Count only the target object type. Ignore every other kind of object.
4. If none are visible, return 0.

Return strict JSON with:
- count: integer number of target instances in the pickup workspace (>= 0)
- reason: one short sentence
""".strip()

        payload = self._generate_json(
            contents=[prompt, pil_image],
            schema=OBJECT_COUNT_SCHEMA,
            temperature=0.0,
        )

        if not isinstance(payload, dict):
            raise GeminiAPIError("Object count response must be a JSON object.")

        raw_count = payload.get("count")
        if isinstance(raw_count, bool) or not isinstance(raw_count, int):
            raise GeminiAPIError(
                f"Count field must be an integer, got {raw_count!r}."
            )

        count = max(0, raw_count)
        reason = str(payload.get("reason", "")).strip()

        return {"count": count, "reason": reason}
```

(Note: signature returns a dict `{count, reason}` — the design's `-> int` is refined to carry `reason` for the published verification payload.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd src/team_8 && PYTHONPATH=. python -m pytest test/test_gemini_api.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/team_8/team_8/gemini_api.py src/team_8/test/test_gemini_api.py
git commit -m "feat: Gemini count_objects replaces verify_object_removed

Written By: Claude Opus 4.8"
```

---

## Task 2: orchestrator `_count_target_in_workspace` helper

**Files:**
- Modify: `src/team_8/team_8/orchestrator.py`
- Test: `src/team_8/test/test_orchestrator.py`

- [ ] **Step 1: Write the failing tests**

Append to `src/team_8/test/test_orchestrator.py`:

```python
def test_count_target_in_workspace_returns_count_when_frame_fresh():
    import numpy as np
    orch = _orchestrator_skeleton()
    orch._verification_frame_timeout_sec = 1.0
    orch._latest_verification_rgb = np.zeros((4, 4, 3), dtype='uint8')
    orch._latest_verification_rgb_stamp_ns = 200
    orch._gemini = MagicMock()
    orch._gemini.count_objects.return_value = {"count": 2, "reason": "two"}

    count = orch._count_target_in_workspace('coke_can', min_stamp_ns=100)

    assert count == 2
    assert orch._gemini.count_objects.call_args.kwargs['object_name'] == 'coke_can'


def test_count_target_in_workspace_returns_none_on_stale_frame():
    import numpy as np
    orch = _orchestrator_skeleton()
    orch._verification_frame_timeout_sec = 0.1
    orch._latest_verification_rgb = np.zeros((4, 4, 3), dtype='uint8')
    orch._latest_verification_rgb_stamp_ns = 50  # not newer than min_stamp_ns
    orch._gemini = MagicMock()

    count = orch._count_target_in_workspace('coke_can', min_stamp_ns=100)

    assert count is None
    orch._gemini.count_objects.assert_not_called()


def test_count_target_in_workspace_returns_none_on_gemini_error():
    import numpy as np
    orch = _orchestrator_skeleton()
    orch._verification_frame_timeout_sec = 1.0
    orch._latest_verification_rgb = np.zeros((4, 4, 3), dtype='uint8')
    orch._latest_verification_rgb_stamp_ns = 0
    orch._gemini = MagicMock()
    orch._gemini.count_objects.side_effect = RuntimeError('boom')

    # min_stamp_ns == 0 uses the latest frame regardless of stamp.
    assert orch._count_target_in_workspace('banana', min_stamp_ns=0) is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd src/team_8 && PYTHONPATH=. python -m pytest test/test_orchestrator.py -k count_target -v`
Expected: FAIL — `AttributeError: ... has no attribute '_count_target_in_workspace'`.

- [ ] **Step 3: Implement the helper**

In `src/team_8/team_8/orchestrator.py`, add this method immediately before `_run_post_task_verification` (around line 762):

```python
    def _count_target_in_workspace(
        self, object_name: str, min_stamp_ns: int
    ) -> 'int | None':
        """Count target-type instances in the source workspace.

        Bounded-wait (``_verification_frame_timeout_sec``) for a verification RGB
        frame stamped after ``min_stamp_ns`` (``min_stamp_ns <= 0`` uses the
        latest frame), then ask Gemini to count. Returns the integer count, or
        ``None`` on timeout / decode / Gemini failure (logged, never raises) so
        callers can fall back to a no-retry success.
        """
        if PILImage is None:
            self.get_logger().warn(
                'Pillow unavailable; cannot count workspace objects.')
            return None

        deadline = time.monotonic() + self._verification_frame_timeout_sec
        frame = None
        while True:
            candidate = self._latest_verification_rgb
            fresh = candidate is not None and (
                min_stamp_ns <= 0
                or self._latest_verification_rgb_stamp_ns > min_stamp_ns)
            if fresh:
                frame = candidate
                break
            if time.monotonic() >= deadline:
                self.get_logger().warn(
                    'No fresh verification frame for object count '
                    f'(object={object_name}, min_stamp_ns={min_stamp_ns}).')
                return None
            time.sleep(0.05)

        image_rgb = frame[:, :, ::-1].copy()
        try:
            result = self._gemini.count_objects(
                PILImage.fromarray(image_rgb), object_name=object_name)
            count = int(result['count'])
        except Exception as exc:
            self.get_logger().warn(
                f'Object count failed (object={object_name}): {exc}')
            return None

        self.get_logger().info(
            f'Workspace count object={object_name} count={count} '
            f"reason={result.get('reason', '')}")
        return count
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd src/team_8 && PYTHONPATH=. python -m pytest test/test_orchestrator.py -k count_target -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/team_8/team_8/orchestrator.py src/team_8/test/test_orchestrator.py
git commit -m "feat: orchestrator workspace object-count helper

Written By: Claude Opus 4.8"
```

---

## Task 3: capture the per-attempt "before" count

**Files:**
- Modify: `src/team_8/team_8/orchestrator.py` (`_run_pipeline`, ~line 408-449)
- Test: `src/team_8/test/test_orchestrator.py`

- [ ] **Step 1: Write the failing test**

Append to `src/team_8/test/test_orchestrator.py`:

```python
def test_run_pipeline_captures_before_count_after_home():
    import json
    orch = _orchestrator_skeleton()
    orch._pipeline_busy = False
    orch._active_task = ''
    orch._active_task_data = {'object': 'coke_can', 'destination': 'storage_1'}
    orch._segmentation_service_name = '/segmentation/segment_prompt'
    orch._segmentation_service_wait_sec = 0.1
    orch._segmentation_client = MagicMock()
    orch._segmentation_client.wait_for_service.return_value = True
    orch._segmentation_client.call_async.return_value = MagicMock()
    orch._home_before_capture = MagicMock(return_value=777)
    orch._count_target_in_workspace = MagicMock(return_value=2)
    orch._on_segmentation_done = MagicMock()

    orch._run_pipeline('pick the coke can')

    # The before-count uses the target name and the home-arrival stamp, and is
    # stored on the active task for the post-task comparison.
    orch._count_target_in_workspace.assert_called_once_with('coke_can', 777)
    assert orch._active_task_data['_before_count'] == 2
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd src/team_8 && PYTHONPATH=. python -m pytest test/test_orchestrator.py -k before_count -v`
Expected: FAIL — `AssertionError` (helper never called / `_before_count` missing).

- [ ] **Step 3: Implement the capture**

In `src/team_8/team_8/orchestrator.py`, inside `_run_pipeline`, locate the block right after the `min_stamp_ns = self._home_before_capture()` try/except (just before `request = StringString.Request()`, ~line 441) and insert:

```python
        # Count target-type instances in the source workspace now (arm at home,
        # camera over the table) so post-task verification can confirm the count
        # dropped. Re-captured every attempt so retries compare against the
        # current scene. None (count unavailable) is stored as-is.
        if self._active_task_data is not None:
            object_name = str(self._active_task_data.get('object', ''))
            self._active_task_data['_before_count'] = (
                self._count_target_in_workspace(object_name, min_stamp_ns))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd src/team_8 && PYTHONPATH=. python -m pytest test/test_orchestrator.py -k "before_count or run_pipeline" -v`
Expected: PASS (the new test plus the existing `_run_pipeline` tests still pass).

- [ ] **Step 5: Commit**

```bash
git add src/team_8/team_8/orchestrator.py src/team_8/test/test_orchestrator.py
git commit -m "feat: capture per-attempt before-count after home move

Written By: Claude Opus 4.8"
```

---

## Task 4: count-based decision in post-task verification

**Files:**
- Modify: `src/team_8/team_8/orchestrator.py` (`_run_post_task_verification`, ~line 762-882)
- Test: `src/team_8/test/test_orchestrator.py`

- [ ] **Step 1: Write the failing tests**

Append to `src/team_8/test/test_orchestrator.py`:

```python
def _verification_orch(before_count, after_count):
    """Orchestrator wired so _run_post_task_verification reaches the decision.

    after_count drives the mocked _gemini.count_objects; a None after_count
    raises to exercise the count-unavailable branch.
    """
    import numpy as np
    orch = _orchestrator_skeleton()
    orch._active_task_data = {
        'object': 'coke_can', 'destination': 'storage_1',
        '_attempt_count': 1, '_before_count': before_count}
    orch._max_task_attempts = 2
    orch._verification_frame_timeout_sec = 1.0
    orch._verification_reference_stamp_ns = 100
    orch._latest_verification_rgb = np.zeros((4, 4, 3), dtype='uint8')
    orch._latest_verification_rgb_stamp_ns = 200  # fresh
    orch._gemini = MagicMock()
    if after_count is None:
        orch._gemini.count_objects.side_effect = RuntimeError('boom')
    else:
        orch._gemini.count_objects.return_value = {
            'count': after_count, 'reason': 'r'}
    orch._publish_verification_result = MagicMock()
    orch._save_verification_artifacts = MagicMock()
    orch._reset_pipeline_state = MagicMock()
    orch._restart_active_task = MagicMock()
    return orch


def test_post_task_success_when_count_decreased():
    orch = _verification_orch(before_count=2, after_count=1)
    orch._run_post_task_verification()
    orch._reset_pipeline_state.assert_called_once()
    assert orch._reset_pipeline_state.call_args.kwargs['success'] is True
    orch._restart_active_task.assert_not_called()


def test_post_task_retries_when_count_not_decreased():
    orch = _verification_orch(before_count=2, after_count=2)  # attempt 1 of 2
    orch._run_post_task_verification()
    orch._restart_active_task.assert_called_once()
    orch._reset_pipeline_state.assert_not_called()


def test_post_task_skips_when_count_not_decreased_at_max_attempts():
    orch = _verification_orch(before_count=2, after_count=2)
    orch._active_task_data['_attempt_count'] = 2  # == _max_task_attempts
    orch._run_post_task_verification()
    orch._reset_pipeline_state.assert_called_once()
    assert orch._reset_pipeline_state.call_args.kwargs['success'] is False
    orch._restart_active_task.assert_not_called()


def test_post_task_success_no_retry_when_after_count_unavailable():
    orch = _verification_orch(before_count=2, after_count=None)
    orch._run_post_task_verification()
    orch._reset_pipeline_state.assert_called_once()
    assert orch._reset_pipeline_state.call_args.kwargs['success'] is True
    orch._restart_active_task.assert_not_called()


def test_post_task_success_no_retry_when_before_count_unavailable():
    orch = _verification_orch(before_count=None, after_count=1)
    orch._run_post_task_verification()
    orch._reset_pipeline_state.assert_called_once()
    assert orch._reset_pipeline_state.call_args.kwargs['success'] is True
    orch._restart_active_task.assert_not_called()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd src/team_8 && PYTHONPATH=. python -m pytest test/test_orchestrator.py -k post_task -v`
Expected: FAIL — the current code calls `verify_object_removed` / reads `present_in_source_workspace`, so the count branches don't exist.

- [ ] **Step 3: Rewrite the decision half of `_run_post_task_verification`**

In `src/team_8/team_8/orchestrator.py`, **replace** everything from the `object_name = str(task['object'])` line through the end of the method (the `try: result = self._gemini.verify_object_removed(...)` block and the `present = ...` decision logic, currently ~lines 799-882) with:

```python
        object_name = str(task['object'])
        destination = str(task['destination'])
        attempt_count = int(task.get('_attempt_count', 1))
        before_count = task.get('_before_count')

        try:
            count_result = self._gemini.count_objects(
                pil_image, object_name=object_name)
            after_count = int(count_result['count'])
            reason = str(count_result.get('reason', ''))
        except Exception as exc:
            after_count = None
            reason = f'count failed: {exc}'
            self.get_logger().warn(
                f'Post-task object count failed (object={object_name}): {exc}')

        removed = (
            before_count is not None
            and after_count is not None
            and after_count < before_count)

        verification_payload = {
            'verification_success': after_count is not None,
            'object': object_name,
            'destination': destination,
            'attempt': attempt_count,
            'max_attempts': self._max_task_attempts,
            'before_count': before_count,
            'after_count': after_count,
            'removed': removed,
            'reason': reason,
        }
        self._publish_verification_result(
            task=task, result=verification_payload)
        self._save_verification_artifacts(
            image_rgb=image_rgb,
            task=task,
            result=verification_payload,
        )

        self.get_logger().info(
            'Post-task count result '
            f'object={object_name} attempt={attempt_count}/'
            f'{self._max_task_attempts} before={before_count} '
            f'after={after_count} removed={removed} reason={reason}')

        # Count unavailable (Gemini error / no fresh frame on either side): we
        # cannot tell if the pick worked. Succeed without retry — re-picking when
        # duplicates exist risks removing a second instance, which is worse than
        # a missed verification.
        if before_count is None or after_count is None:
            self.get_logger().warn(
                'Object count unavailable; marking task done without retry '
                f'(object={object_name}).')
            self._reset_pipeline_state(
                success=True,
                reason='object count unavailable; assumed removed',
            )
            return

        if removed:
            self._reset_pipeline_state(
                success=True,
                reason=(
                    f'count dropped {before_count}->{after_count}; removed'),
            )
            return

        if attempt_count < self._max_task_attempts:
            self.get_logger().warn(
                f"Count for '{object_name}' did not drop "
                f"({before_count}->{after_count}); retrying the same task "
                f"({attempt_count + 1}/{self._max_task_attempts}).")
            self._restart_active_task()
            return

        self.get_logger().warn(
            f"Count for '{object_name}' did not drop after "
            f"{attempt_count} attempts; skipping it and continuing to the "
            "next task.")
        self._reset_pipeline_state(
            success=False,
            reason=(
                f'count did not drop after {attempt_count} attempts; skipped'),
        )
```

(Leave the earlier part of `_run_post_task_verification` — the fresh-frame wait and the `image_bgr`/`image_rgb`/`pil_image` construction, ~lines 762-797 — unchanged.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd src/team_8 && PYTHONPATH=. python -m pytest test/test_orchestrator.py -k post_task -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/team_8/team_8/orchestrator.py src/team_8/test/test_orchestrator.py
git commit -m "feat: count-based post-task pick verification

Written By: Claude Opus 4.8"
```

---

## Task 5: full-suite regression + stale-reference sweep

**Files:**
- Modify (only if a stale reference is found): `src/team_8/test/test_orchestrator.py`, `src/team_8/team_8/orchestrator.py`

- [ ] **Step 1: Grep for any leftover references to the removed API**

Run:
```bash
grep -rn "verify_object_removed\|present_in_source\|TASK_VERIFICATION_SCHEMA" src/
```
Expected: **no matches.** If any appear (e.g. an old test or log line), update it to the count-based payload (`before_count`/`after_count`/`removed`) or delete the dead reference, then re-run the relevant test file.

- [ ] **Step 2: Run the full orchestrator + gemini suites**

Run: `cd src/team_8 && PYTHONPATH=. python -m pytest test/test_orchestrator.py test/test_gemini_api.py -v`
Expected: PASS — all tests green, including the pre-existing `_run_pipeline`, verification-timer, and pick-done tests.

- [ ] **Step 3: Smoke-test orchestrator import/startup path**

Per the project's known test gap (unit tests use `__new__` and skip `__init__`), verify the module imports cleanly so a typo in the edited methods is caught:

Run: `cd src/team_8 && PYTHONPATH=. python -c "import team_8.orchestrator, team_8.gemini_api; print('import OK')"`
Expected: `import OK` (no NameError / SyntaxError).

- [ ] **Step 4: Commit any sweep fixes (skip if none)**

```bash
git add -A && git commit -m "test: sweep stale verify_object_removed references

Written By: Claude Opus 4.8"
```

---

## Self-review notes (already applied)

- **Spec coverage:** Gemini `count_objects` + removal of `verify_object_removed` (Task 1); `_count_target_in_workspace` helper (Task 2); per-attempt before-count after home (Task 3); after-count + `after < before` decision + count-unavailable→success-no-retry + payload shape change (Task 4); test/reference sweep (Task 5). All design sections mapped.
- **Signature refinement:** design said `count_objects -> int`; plan returns `{count, reason}` so the published payload keeps a `reason`. Documented in Task 1 Step 3.
- **Type consistency:** `count_objects` returns `{'count': int, 'reason': str}` everywhere; orchestrator reads `result['count']` / `result['reason']`; `_count_target_in_workspace` returns `int | None`; `_before_count` stored as `int | None` and read in Task 4. Consistent across tasks.
- **Commit-message style:** uses `Written By: Claude Opus 4.8` per repo convention.
