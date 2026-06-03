# nvblox Integration Design

**Date:** 2026-05-25
**Branch:** nvblox
**Status:** Approved

## Goal

Add nvblox as a persistent scene map that serves two consumers:
1. **cuRobo** — ESDF for collision-aware trajectory planning across multiple pick-and-place cycles
2. **GraspGen** — object point cloud extracted from the nvblox mesh, replacing the raw depth image input

This replaces the previously considered standalone point cloud fusion pipeline, unifying scene representation in nvblox so that the map stays consistent as objects are removed across cycles.

## Architecture

```
Overhead RGB+D ──┐
                 ├──► nvblox (Isaac ROS node, persistent TSDF/ESDF)
Wrist RGB+D ─────┘         │
                            ├──► ESDF ──────────────────────► cuRobo (WorldNvbloxCollision)
                            └──► Mesh surface
                                     │
[parallel with nvblox init]          │
Overhead RGB + task string           │
    ──► GeminiLocalizer              │
        → bounding box (px coords)   │
            ──► SAM2 → 2D mask ──────┘
                    → project mask onto nvblox mesh
                    → object point cloud (robot base frame)
                    ──► GraspGen → ranked grasp candidates
                                       │
                               cuRobo batch plan
                               (WorldNvbloxCollision)
                                       │
                               execute best trajectory
                               → gripper close → lift
                               [nvblox map auto-updates for next cycle]
```

## Components

### New: `nvblox.py` — `NvBlox`

Wraps the Isaac ROS nvblox ROS2 node (runs as subprocess in same container). Subscribes to nvblox's published ESDF and mesh topics.

**Interface:**
- `get_esdf() -> NvbloxLayer` — returns latest ESDF handle for cuRobo; blocks until first map is ready, then returns cached value that auto-updates
- `extract_object_cloud(mask_2d, camera='overhead') -> np.ndarray` — projects SAM2 2D mask rays into nvblox mesh; returns `(N, 3)` point cloud in robot base frame

The nvblox node subscribes to both camera depth topics and `/tf`. Wrist camera pose is derived from FK via the TF tree (joint states already published).

### New: `gemini.py` — `GeminiLocalizer` (stub)

Placeholder for teammate implementation. Sends overhead RGB + task prompt string to Gemini Vision API, returns a bounding box in image pixel coordinates.

**Interface:**
- `locate_object(rgb_image: Image, task_prompt: str) -> tuple[int, int, int, int] | None` — returns `(x1, y1, x2, y2)` in pixel coordinates, or `None` on failure

### Modified: `orchestrator.py`

`_run_pipeline` updated to:
1. Fire Gemini API call and nvblox map readiness check **in parallel** via `concurrent.futures.ThreadPoolExecutor` (both I/O-bound on first call)
2. Pass Gemini bounding box to SAM2 as spatial prompt (replacing raw text prompt)
3. Call `nvblox.extract_object_cloud(mask)` → pass point cloud to GraspGen
4. Pass `nvblox.get_esdf()` to cuRobo as world collision argument

### Modified: `graspgen.py`

`generate_grasp` signature changes:
- **Before:** `generate_grasp(masks, depth: Image)`
- **After:** `generate_grasp(point_cloud: np.ndarray)`

### Modified: `curobo.py`

`plan_trajectory` signature changes:
- **Before:** `plan_trajectory(grasp_pose, joint_states)`
- **After:** `plan_trajectory(grasp_pose, joint_states, esdf)`

Uses `WorldNvbloxCollision` configured with the provided ESDF layer.

### Modified: `Dockerfile`

Add Isaac ROS nvblox apt packages (Isaac ROS common + nvblox). Requires adding the Isaac ROS apt repository.

### New: `requirements/nvblox.txt`

Python-side dependencies for nvblox integration (e.g. `nvblox-torch` if used for point cloud extraction, otherwise empty placeholder).

## Key Design Decisions

**Why nvblox for both collision and GraspGen input?**
Running two separate depth fusion systems (nvblox for cuRobo + teammate's fusion for GraspGen) would produce inconsistent scene representations. Using nvblox as the single source of truth means cuRobo and GraspGen always agree on where objects are, and the map is automatically correct after each pick-and-place cycle.

**Why Gemini in parallel with nvblox?**
Both are I/O-bound on the first call (~1–2s each). Running them concurrently keeps pipeline latency at `max(t_gemini, t_nvblox)` rather than their sum.

**Why not nvblox for GraspGen directly?**
GraspGen needs an object-centric point cloud, not the full scene ESDF. SAM2 provides the 2D segmentation mask to carve out just the target object from the nvblox mesh surface — the two work together.

**Wrist camera at planning time**
Both cameras feed nvblox continuously. The wrist depth improves map quality from multiple viewpoints, but the wrist camera pose must be resolved via TF (FK from joint states). This is already available since joint states are published.

## What Is Not Changing

- `sam2.py` — `segment(rgb, prompt: str, bbox: tuple[int,int,int,int] | None = None)` gains optional `bbox`; if provided, uses it as SAM2 spatial prompt and ignores `prompt`
- `moveit2.py` — fallback planner unchanged
- `docker-compose.yml` — host networking and NVIDIA runtime unchanged
- `setup.py` / `package.xml` — no new ROS2 packages needed

## Open Items

- Confirm Isaac ROS nvblox is compatible with CUDA 12.8 base image (Isaac ROS typically targets specific CUDA versions)
- Decide nvblox voxel resolution (trade-off: finer = more VRAM, coarser = less accurate collision geometry)
- Teammate to implement `GeminiLocalizer.locate_object`
