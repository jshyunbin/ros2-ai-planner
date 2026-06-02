# Design: Build & launch refactor (single Dockerfile + deploy/debug modes)

Date: 2026-06-02
Status: Approved (approach), pending implementation

## Problem

Standing up the pipeline today takes too many moving parts:

- **Two Dockerfiles.** `Dockerfile.base` (heavy: CUDA/ROS/PyTorch/SAM2/GraspGen/
  cuRobo + model downloads, ~20 min) and `Dockerfile` (light: `COPY src`,
  `colcon build`), glued by a `PLANNER_BASE_IMAGE` build arg and a
  `build_base_image.sh` → `build_image.sh` script chain.
- **Two compose files.** `docker-compose.yml` bakes `src` in;
  `docker-compose.dev.yml` overrides to live-mount it.
- **A long manual launch.** After `docker compose up` the user execs into bash
  and hand-types `ros2 launch pipeline_orchestrator planner_pipeline.launch.py`
  with a subset of ~25 launch args (`enable_motion_execution`,
  `start_graspgen_server`, `curobo_enable_viz`, …) to get the behavior they want.
- **No first-class debug visualization.** `curobo.py` can *cache* TSDF/cloud viz
  data behind `enable_viz`, but nothing serves it in the runtime pipeline; grasp
  poses are only visualized in standalone test scripts.

## Goals

1. One Dockerfile, heavy/stable layers first so `src` edits never rebuild them.
2. Source baked into the deploy image; live-mounted in debug. No build-arg chain.
3. Two modes — **deploy** and **debug** — each launchable with a single
   copy-paste command and zero required ROS flags.
4. Debug mode runs the same pipeline as deploy (including arm execution) **plus**
   a viser server visualizing segmented clouds, ranked grasp poses, and the live
   TSDF.

## Non-goals

- No change to the pipeline *sequence* (segmentation → graspgen → curobo →
  execute) or to the cloud-sync hand-off from the prior design.
- No visualization of the planned trajectory (explicitly out of scope).
- No new model versions or dependency changes.

## Section 1 — Single Dockerfile

Merge `Dockerfile.base` + `Dockerfile` into one `Dockerfile`, ordered so the
heaviest and most stable layers come first and the only frequently-changing
layer (`COPY src`) is last:

1. Base image (`nvcr.io/nvidia/pytorch:23.07-py3`) + locale + ROS apt source + git-lfs
2. ROS Humble + system/build packages, UR5 URDF generation
3. `pip` planner-runtime + SAM2 runtime (`--no-deps`)
4. Clone pinned GraspGen fork → its pip reqs → compile `pointnet2_ops` → install fork (`--no-deps -e`)
5. cuRobo from source (v0.8.0) + numpy re-pin
6. Model downloads — SAM2 checkpoint (curl) and GraspGen weights (selective LFS pull). Large but immutable.
7. `COPY src/ scripts/ misc/` + `colcon build --symlink-install`

Because step 7 is last, Docker's layer cache shields steps 1–6 from code edits —
the same "don't rebuild the world on a `src` change" property the two-file split
provided, without the indirection. The pinned GraspGen/model build ARGs
(`GRASPGEN_REPO_URL`, `GRASPGEN_COMMIT`, `GRASPGEN_MODELS_COMMIT`, …) move onto
this single Dockerfile so reproducibility is preserved.

Removed scripts: `build.sh`, `build_base_image.sh`, `build_image.sh`, `run.sh`.
Kept scripts (real logic): `entrypoint.sh`, `start_graspgen_server.sh`.

## Section 2 — Two compose files, documented one-liners

- `docker-compose.yml` — **deploy**: baked image, no `src` mount; container
  command runs `deploy.launch.py`.
- `docker-compose.debug.yml` — override that live-mounts `./src`, `./scripts`,
  `./config` over the baked image and runs `debug.launch.py`. (Replaces
  `docker-compose.dev.yml`.)

No wrapper shell scripts. `CLAUDE.md` documents the copy-paste one-liners:

```bash
# Build (heavy layers cached after first run)
docker compose build

# Deploy mode: full pipeline, executes on the UR5, no viz
docker compose up

# Debug mode: same pipeline + viser visualization, src live-mounted
docker compose -f docker-compose.yml -f docker-compose.debug.yml up
```

`GEMINI_API_KEY` continues to come from the environment / `.env`.

## Section 3 — Two launch files over a shared core

- `pipeline_common.launch.py` — holds the segmentation + graspgen + curobo +
  orchestrator nodes with all current params as **internal defaults**. Exposes a
  small set of overridable arguments (execution, viz, graspgen server) consumed
  by the two mode files. Power users can still override any param, but neither
  mode requires passing flags.
- `deploy.launch.py` — includes common with `enable_motion_execution=true`, viz
  publishing off, `start_graspgen_server=true` (auto-starts the GraspGen ZMQ
  server so it is not a separate manual step).
- `debug.launch.py` — includes common with the same execution settings **plus**
  the `debug_viz` node and viz publishing on (`enable_viz=true`,
  `publish_grasp_poses=true`). Debug = deploy + visualization; it also executes
  on the arm.

The legacy `planner_pipeline.launch.py` is superseded by these three files.

## Section 4 — Dedicated `debug_viz` node + viser

New module `debug_viz.py` (console_scripts entry point `debug_viz`): a single
rclpy node hosting one viser server, subscribing only to lightweight ROS topics
so visualization is fully decoupled from the heavy nodes.

| Source node | Publication | Type | Gating |
|---|---|---|---|
| segmentation_service | `/graspgen/segmented_object`, `/graspgen/background` (exist) | `sensor_msgs/PointCloud2` | always |
| graspgen_service | `/graspgen/grasp_poses` (**new**) | `geometry_msgs/PoseArray`, rank order | `publish_grasp_poses` param |
| curobo_service | `/curobo/tsdf_voxels` (**new**) — occupied voxel centers as a cloud | `sensor_msgs/PointCloud2` | `enable_viz` param |

`debug_viz` renders the clouds, grasp gripper frames (colored by rank), and the
TSDF voxel centers, reusing `live_viz_helpers`. The TSDF is serialized cheaply as
occupied voxel-center points rather than a full voxel grid or cube markers.

**Publisher changes:**

- `graspgen_service.py` — when `publish_grasp_poses` is set, publish the ranked
  grasps it already computes as a `PoseArray` (rank order) alongside the existing
  JSON service response.
- `curobo_service.py` / `curobo.py` — when `enable_viz` is set, publish occupied
  TSDF voxel centers as a `PointCloud2` (reusing the data already cached by the
  existing `enable_viz` path).

**Orchestrator refactor:** the orchestrator reads its execution/viz behavior from
params supplied by the launch file instead of relying on hand-passed flags; the
pipeline sequence itself is unchanged.

## Open follow-ups (deferred, not blocking)

- Coloring grasps by GraspGen *confidence* (not just rank) would need confidence
  carried alongside the `PoseArray` (parallel `Float32MultiArray` or a small
  custom msg). Deferred — rank-order coloring is sufficient for v1.
- TSDF as voxel-cube markers (blockier/clearer but heavier) — deferred in favor
  of voxel-center points.

## Testing

- Existing unit tests (`test_orchestrator.py`, `test_live_viz_helpers.py`,
  `test_graspgen_service.py`) must still pass.
- New: a unit test that `graspgen_service` emits a correctly-ordered `PoseArray`
  when `publish_grasp_poses` is set, and that `debug_viz` builds its viser scene
  from synthetic cloud/pose/voxel messages without a live robot.
- Manual: `docker compose build` caches steps 1–6 on a `src`-only change; both
  one-liners bring up their mode; the viser URL renders all three layers in debug.
