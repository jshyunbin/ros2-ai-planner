# Count-based pick verification

**Date:** 2026-06-08
**Branch:** pose_enhance
**Status:** Approved (design)

## Problem

Post-task verification (`orchestrator._run_post_task_verification` +
`GeminiAPI.verify_object_removed`) decides pick success by asking Gemini a
boolean: *is the target object still present in the source pickup workspace?*
If yes → retry; if no → success.

The scene can contain **multiple identical objects** (e.g. two coke cans).
After a successful pick of one, the *type* is still visible, so
`present_in_source_workspace` stays `true`. The task is judged a failure, burns
its entire retry budget re-picking, and is finally skipped — even though every
pick succeeded.

## Fix (high level)

Replace the boolean presence check with a **count comparison of the target
object type**:

1. Count instances of the target type in the source workspace **before** the
   attempt's pick.
2. Count again **after** the place → home sequence.
3. Success ⇔ **after < before** (strictly decreased).

"Decreased" (not "exactly one fewer") is the success rule: robust to Gemini
miscounting by one and to accidentally knocking two off.

## Components

### 1. Gemini layer — `src/team_8/team_8/gemini_api.py`

- **Add** `count_objects(pil_image, *, object_name) -> int`.
  - New `COUNT_SCHEMA`: `{ "count": INTEGER, "reason": STRING }`, both required.
  - Prompt: count how many instances of `<object>` are visible in the **source
    pickup workspace / main table area**. Ignore the robot arm and gripper.
    Ignore objects inside the destination baskets / storage / bookshelf. Ignore
    every object that is not the target type. Return `count` (>= 0) and a short
    `reason`.
  - Validate `count` is a non-negative int; clamp negatives to 0; raise
    `GeminiAPIError` on malformed payloads (same pattern as the existing
    methods).
- **Remove** `verify_object_removed` and `TASK_VERIFICATION_SCHEMA`
  (the orchestrator is their only caller).

### 2. Orchestrator — "before" count

- **New blocking helper**
  `_count_target_in_workspace(object_name: str, min_stamp_ns: int) -> int | None`:
  - Bounded-wait (reuse `_verification_frame_timeout_sec`) for a
    `_latest_verification_rgb` stamped `> min_stamp_ns` (when `min_stamp_ns == 0`,
    use the latest frame — mirrors the segmentation freshness-gate convention).
  - Frames keep arriving while we block: the camera subscription is in the
    default callback group, the pipeline blocks in the reentrant
    `_pipeline_cbg`, and the node runs under a `MultiThreadedExecutor`.
  - Build a PIL image (same BGR→RGB conversion as the post-task path) and call
    `self._gemini.count_objects`.
  - Return the int count, or `None` on timeout / decode / Gemini error (logged
    as a warning, never raises).
- **Capture point:** in `_run_pipeline`, immediately after
  `_home_before_capture()` returns `min_stamp_ns` (arm settled at home, wrist
  camera over the workspace), set
  `task['_before_count'] = self._count_target_in_workspace(object_name, min_stamp_ns)`.
  Re-captured on **every attempt** (per-attempt timing) so retries self-correct
  against the current scene.

### 3. Orchestrator — "after" count + decision — `_run_post_task_verification`

- Reuse the fresh post-home `pil_image` the method already builds; call
  `count_objects` → `after_count` (None on error, handled below).
- Decision table:

  | before | after | outcome |
  |---|---|---|
  | known | known, `after < before` | **success** (verified removed) |
  | known | known, `after >= before` | retry within budget, else skip (existing `_max_task_attempts` logic) |
  | `None` (either) | — | **success, no retry** (warning) |

  Rationale for the unavailable-count branch: with identical objects, blindly
  retrying risks picking a *second* instance, which is worse than a missed
  verification. Confirmed with user.

- Existing retry / skip plumbing (`_restart_active_task`,
  `_reset_pipeline_state`) is reused unchanged; only the success predicate
  changes from `not present` to `after < before`.

### 4. Published verification result

`/gemini/task_verification` payload shape changes:

- **Removed:** `present_in_source_workspace`.
- **Added:** `before_count`, `after_count`, `removed` (bool = `after < before`).
- **Kept:** `verification_success`, `object`, `destination`, `attempt`,
  `max_attempts`, `reason` (sourced from the after-count call).
- **Dropped:** `confidence` (the count path produces no confidence score).

Debug artifacts (`_save_verification_artifacts`) keep saving the RGB frame +
the new result dict.

## Testing (TDD)

`src/team_8/test/test_orchestrator.py`:

- Update existing verification tests for the new method names / payload.
- Add cases for `_run_post_task_verification`:
  - before=2, after=1 → success, no retry.
  - before=2, after=2, attempt < max → retry.
  - before=2, after=2, attempt == max → skip (failure).
  - before unavailable (`None`) → success, no retry.
  - after-count Gemini error → success, no retry.
- Add a `count_objects` unit test in the Gemini-facing tests if one exists for
  the parsing helpers; otherwise cover via orchestrator mocks.

## Out of scope

- Counting *all* objects regardless of type (rejected: confounded by unrelated
  scene changes; user chose target-type-only).
- Changing segmentation, GraspGen, or cuRobo stages.
- NL parsing of destinations.
