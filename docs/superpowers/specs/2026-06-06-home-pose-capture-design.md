# Capture wrist RGBD from the home pose, fresh each cycle

**Date:** 2026-06-06
**Branch:** move_item
**Status:** Design approved

## Problem

The wrist RGB-D camera is mounted on the UR5 arm, so what it sees depends on the
arm's pose. Today the orchestrator, on each `/task_commands` message, immediately
runs segmentation against whatever the wrist camera currently sees — wherever the
arm happens to be parked. The arm only visits the `home` pose at the *end* of a
pick-and-place cycle. Consequences:

- The first task command segments from an arbitrary (often unhelpful) viewpoint.
- There is no guarantee the frame used for Gemini + GraspGen reflects a known,
  workspace-observing vantage point.

We want: on every cycle the arm first moves to `home` (a tool0 pose that observes
the workspace), then a **fresh** wrist RGB-D frame captured *at* home drives
segmentation → GraspGen. Because the scene changes after each pick-and-place, the
next cycle must re-home and re-capture rather than reuse a stale frame.

## Decisions (from brainstorming)

- **Loop model: per-command.** Each `/task_commands` message = one pick-and-place.
  The fix is to move home + capture a fresh frame at the *start* of every command;
  no internal multi-object loop. This covers both "home before the first capture"
  and "recapture after each iteration" with no looping logic.
- **Freshness: timestamp-gated wait.** The orchestrator records the home-arrival
  ROS time and passes it to the segmentation service, which blocks until it has an
  RGB **and** depth frame stamped after that time (or times out).

Both the orchestrator and segmentation_service run with `use_sim_time=True`
(see `pipeline_common.launch.py`), so the orchestrator's `get_clock().now()` and
the camera header stamps share the same sim-time base — the gate is
clock-consistent.

## Components

### 1. Orchestrator — home before capture (per command)

In `PipelineOrchestrator._run_pipeline`, after validating the task and marking the
pipeline busy, but **before** starting segmentation:

1. Open the gripper (safety: nothing should be carried at cycle start).
2. Drive the arm to the `home` tool0 pose by reusing `_plan_and_execute_home()`
   (collision-aware cuRobo plan against the live TSDF). On failure: log, call
   `_reset_pipeline_state()`, and return.
3. After `_send_and_wait(...)` returns (arm settled at home), record
   `arrival_ns = self.get_clock().now().nanoseconds`.
4. Start segmentation, passing `arrival_ns` as the freshness gate (see Component 2).

The end-of-cycle `_plan_and_execute_home()` call in `_on_curobo_pick_done` stays
unchanged — it leaves the arm in a safe, workspace-observing pose ready for the
next command.

The startup home move is blocking inside the task callback. This is consistent
with the existing pattern: place/home already use blocking
`_call_curobo_blocking` / `_send_and_wait` under the `MultiThreadedExecutor`, so
blocking here does not deadlock.

**Implementation note:** the segmentation call is currently issued from
`_run_pipeline` as an async `call_async(...)` + done-callback chain. The home move
is inserted ahead of that async kickoff. `_run_pipeline` does the blocking home
move, then issues the segmentation request (now carrying `min_stamp_ns`).

### 2. Segmentation request — carry the freshness gate

The service keeps the `riro_srvs/StringString` interface (no `.srv` change, no
`riro_srvs` rebuild). The request `data` field becomes a small JSON object:

```json
{"prompt": "<task text>", "min_stamp_ns": <int>}
```

Back-compat: if `data` does not parse as a JSON **dict**, the whole string is
treated as the prompt with `min_stamp_ns = 0` (no gating). This mirrors the
JSON-over-StringString pattern already used for GraspGen responses.

The orchestrator builds this JSON in the segmentation request instead of sending
the raw task string.

### 3. Segmentation service — wait for a post-arrival frame

In `SegmentationService._handle_request`, parse `prompt` and `min_stamp_ns` from
the JSON request. If `min_stamp_ns > 0`, block until **both**:

- `_latest_rgb_stamp_ns > min_stamp_ns`, and
- `_latest_depth_stamp_ns > min_stamp_ns`

or until a timeout (`PIPELINE_SEG_FRESH_FRAME_TIMEOUT_SEC`, default `5.0` s). On
timeout, return `{"success": false, "error": "...", ...}`; the orchestrator treats
that as a failed cycle and resets.

**Concurrency fix (required).** The node currently runs on a
`SingleThreadedExecutor` (`rclpy.spin(node)`). If the service handler blocks
waiting for new frames while the RGB/depth callbacks sit in the same default
mutually-exclusive callback group, those callbacks can never run → deadlock. So:

- Run the node under a `MultiThreadedExecutor` (≥2 threads).
- Put the three camera subscriptions (rgb, depth, camera_info) in a dedicated
  `ReentrantCallbackGroup`, separate from the service callback, so frame callbacks
  keep firing on another thread while the handler waits.
- The handler waits on a `threading.Event` set by the rgb/depth callbacks when a
  frame newer than the gate arrives (with a timeout). A `threading.Lock` guards the
  shared `_latest_*` fields and the gate/event so the handler and callbacks don't
  race.

When `min_stamp_ns == 0`, behavior is unchanged (use the latest cached frame), so
standalone/debug callers and the back-compat path keep working.

### 4. Config

- New env knob `PIPELINE_SEG_FRESH_FRAME_TIMEOUT_SEC` (default `5.0`), read in the
  segmentation service. Reuse the existing `_env_float` helper pattern.

## Data flow (per `/task_commands`)

```
task_cmd ->
  orchestrator: open gripper
  orchestrator: _plan_and_execute_home()  (collision-aware)  [blocking]
  orchestrator: arrival_ns = clock.now()
  orchestrator: segment({prompt, min_stamp_ns=arrival_ns})   [async]
    segmentation_service: wait rgb.stamp & depth.stamp > arrival_ns (or timeout)
    segmentation_service: Gemini + SAM2 -> publish clouds -> respond
  orchestrator: graspgen -> curobo pick -> approach/grasp/lift
  orchestrator: place -> release
  orchestrator: _plan_and_execute_home()  (collision-aware)
  orchestrator: _reset_pipeline_state()
(next task_cmd repeats; arm already at home)
```

## Error handling

- **Startup home plan/exec fails:** log, `_reset_pipeline_state()`, abort the cycle.
- **Fresh-frame timeout:** segmentation returns `success:false`; existing
  `_on_segmentation_done` failure path resets the pipeline.
- **Gripper open at start fails:** non-fatal (existing `_send_gripper` swallows and
  warns), consistent with current behavior.

## Risks / notes

- The startup home move uses collision-aware planning against the live TSDF. The
  overhead camera feeds the TSDF from launch, so it should be populated by the time
  the first command arrives. If the arm starts somewhere it cannot plan home from,
  the cycle aborts cleanly (logged).
- Adds one extra home traversal per cycle (a few seconds).
- Standalone segmentation callers (`graspgen_service_caller`, probes) that send a
  raw prompt string still work via the back-compat (`min_stamp_ns = 0`) path.

## Out of scope

- Internal multi-object looping / autonomous scene clearing.
- Changing the `home` pose definition or `place_poses.yml`.
- Any change to GraspGen / cuRobo planning logic beyond being driven by the
  fresher cloud.
```
