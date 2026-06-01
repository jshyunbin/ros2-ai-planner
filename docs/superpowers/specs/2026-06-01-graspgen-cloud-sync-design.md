# Design: Synchronize segmentation → GraspGen cloud hand-off

Date: 2026-06-01
Status: Approved (approach), pending implementation

## Problem

`SegmentationService` publishes the segmented and background point clouds on
BEST_EFFORT topics and then returns its JSON response. The orchestrator, on
success, immediately triggers `GraspGenService` over `/graspgen/infer`
(`std_srvs/Trigger`). GraspGen infers on whatever cloud its topic subscription
last cached (`_latest_segmented_cloud`).

Nothing correlates the Trigger with a specific cloud:

- **First run:** GraspGen may not have received the cloud yet → "No segmented
  point cloud received yet" or inference on nothing.
- **Repeat runs:** GraspGen can silently infer on the *previous* run's cloud if
  the new one is still in flight.
- **BEST_EFFORT** can also drop the one cloud that matters.

It works most of the time due to timing slack, but there is no guarantee.

## Approach (chosen: "sync token, keep topics")

Keep point clouds flowing over topics (correct transport for bulk data), but
add a correlation token so GraspGen runs on exactly the cloud this run
produced. Close the drop window by making the cloud topics RELIABLE.

### Token

- The token is the segmented cloud's `header.stamp` expressed in **nanoseconds
  as a decimal string**.
- Stamp source: the depth frame the cloud was built from
  (`SegmentationService._latest_depth_stamp`).
- An **empty token means "use latest"** — preserves the `graspgen_service_caller`
  debug path and any non-orchestrated callers.

### Wait semantics

GraspGen waits until `latest_received_stamp_ns >= requested_ns` or a timeout
elapses, then infers. `>=` (not `==`) is robust: within a single pipeline run
segmentation publishes exactly one cloud, so the latest stamp equals the
requested one; `>=` simply guarantees "at least as fresh as this run's cloud,
never older." Timeout → clean failure response (better than silently grasping a
stale cloud).

## Interface changes

### `/graspgen/infer`: `std_srvs/Trigger` → `riro_srvs/StringString`

No new srv is introduced; reuse the existing `StringString`
(`string data` request / `string data` response).

- **Request** `data`: the token (nanosecond stamp string, or empty for latest).
- **Response** `data`: a JSON blob containing a `success` field plus the
  existing payload (`frame_id`, `num_grasps`, `top_grasps`, …). This matches the
  shape `SegmentationService` already returns, so after this change both
  upstream services share one "StringString in, JSON out" contract.

### Cloud topics → RELIABLE QoS

`/graspgen/segmented_object` and `/graspgen/background` switch from BEST_EFFORT
to RELIABLE on **both** the segmentation publishers and the GraspGen
subscriptions (QoS must match). These are request/response-style bulk messages
(≤4096 downsampled points, tens of KB), not a high-rate stream, so RELIABLE is
appropriate and removes the silent-drop failure mode.

## Concurrency

`GraspGenService` currently runs on `rclpy.spin()` (single-threaded). A blocking
wait inside the service callback would deadlock: the cloud subscription callback
cannot run on the same thread that is blocked waiting for it.

Fix: run `GraspGenService` under a `MultiThreadedExecutor`, with the cloud
subscriptions in a separate callback group from the service (a
`ReentrantCallbackGroup`, or distinct `MutuallyExclusiveCallbackGroup`s), so the
subscription can deliver while the service handler waits. This mirrors the
pattern `PipelineOrchestrator` already uses. The service handler waits on a
`threading.Condition` that the segmented-cloud callback notifies.

## Component-level changes

| File | Change |
|---|---|
| `segmentation_service.py` | Stamp published clouds with `_latest_depth_stamp`; add the token (`cloud_stamp_ns`) to the response JSON; switch the two cloud publishers to RELIABLE QoS. |
| `pipeline_utils.py` | `make_xyz_cloud` gains an optional `stamp` parameter (defaults to unset, preserving current behavior). |
| `graspgen_service.py` | Track received segmented-cloud stamp; switch service to `StringString`; parse the requested token; wait on a `Condition` (bounded by a `cloud_wait_sec` param) until the matching cloud is present; emit JSON with `success`; switch cloud subscriptions to RELIABLE; run under `MultiThreadedExecutor` with a separate callback group for subscriptions. |
| `orchestrator.py` | `/graspgen/infer` client `Trigger → StringString`; send the token from the segmentation payload; parse `result.data` JSON (with `success`) like `_on_segmentation_done`. |
| `graspgen_service_caller.py` | Call the `StringString` service with an empty token; parse `result.data` JSON. |
| `graspgen_probe.py` | No change (uses the ZMQ client, not the ROS service). |

## Parameters (new)

- `graspgen_service`: `cloud_wait_sec` (default ~5.0) — max time to wait for the
  requested cloud before failing.

## Error handling

- Empty token → infer on latest cloud immediately (no wait); current behavior
  for debug callers.
- Token given but no matching cloud within `cloud_wait_sec` → response JSON
  `{success: false, error: "timed out waiting for segmented cloud <stamp>"}`.
- Orchestrator treats `success: false` in the GraspGen JSON exactly as it
  treats segmentation failure today (warn + reset pipeline).

## Testing

- Pure-Python unit test for the stamp/token round-trip helper if one is
  extracted (nanosecond ↔ `builtin_interfaces/Time`).
- `graspgen_service` handler logic test (mock node) covering: empty token →
  immediate infer; matching stamp present → infer; timeout → failure JSON.
- Verify orchestrator `_on_graspgen_done` parses the new JSON contract.
- Full ROS integration verified inside the container (the ROS-dependent test
  suites and a live `planner_pipeline.launch.py` run).

## Out of scope

- No change to GraspGen ranking/filtering, the ZMQ inference path, or cuRobo.
- No change to the segmentation algorithm.
