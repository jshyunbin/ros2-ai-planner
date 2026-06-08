# CLAUDE.md

This file provides guidance to Claude Code when working in this repository.

## Overview

`ros2-ai-planner` is a standalone Dockerized ROS2 system that acts as an external AI planning stack for the CS477 `manip_challenge` pick-and-place project. It runs on the same machine as the `manip_challenge` simulation (Gazebo + ROS2 Humble) and communicates over a shared host network.

**Pipeline:** task command → segmentation (Gemini bounding box + SAM2) → GraspGen (grasp poses from the segmented point cloud) → cuRobo (plan joint trajectory against a live dual-RGBD TSDF) → execute on the UR5 via `/ur5_controller/follow_joint_trajectory`, then move the grasped object to a goal destination (collision-off safe-`transit_z` transit; bookshelf gets a +x insert / −x retract), release it, and return the empty arm home (collision-aware).

All three AI stages are implemented and run as **separate ROS2 nodes**, coordinated by the orchestrator over ROS2 services. This is no longer a single stub node.

## Build & Run

```bash
# Build the image (heavy layers cached after first run; ~20 min first time)
docker compose build

# Deploy mode: full pipeline, executes on the UR5, no visualization
docker compose up

# Debug mode: same pipeline + viser visualization, src live-mounted
docker compose -f docker-compose.yml -f docker-compose.debug.yml up

# Interactive shell (debug override gives the live-mounted src)
docker compose -f docker-compose.yml -f docker-compose.debug.yml run --rm ai_planner bash
```

> In debug mode `./src` is live-mounted, so Python edits to node code take effect on the next launch. Changes to `setup.py`, entry points, or `*.launch.py` files still require a workspace rebuild inside the container: `colcon build --packages-select team_8` (from `/ros2_ws`), then relaunch.

The container uses `network_mode: host` — all ROS2 topics from the host are immediately visible inside.

## Package Architecture

Two ROS2 packages live under `src/`:

- `team_8` — all pipeline nodes (below).
- `utils/riro_srvs` — custom service definitions (`StringString`, `PlanTrajectory`, …).

Nodes (each is a `console_scripts` entry point in `setup.py`):

| Module | Node / entry point | Role |
|---|---|---|
| `orchestrator.py` | `orchestrator` | Drives the pipeline: subscribes `/task_commands`, calls the segmentation, GraspGen, and cuRobo services in turn, then executes the trajectory via FollowJointTrajectory actions. At the start of each `/task_commands` cycle it first waits for the latched `/curobo/ready` signal (so it never issues a plan before CuRobo has initialised and built the TSDF — only blocks on the first cycle), then moves the arm to the `home` joint configuration (collision-aware c-space plan), opens the gripper, and captures the home-arrival time, passing it to segmentation as a freshness gate so each pick-and-place segments a wrist frame taken *after* the arm settled at home. After the pick lift it forwards a destination key (`target_goal` param — placeholder until upstream NL parsing resolves it) to the cuRobo `goal_name` mode and executes the returned place → release → home sequence. |
| `segmentation_service.py` | `segmentation_service` | `StringString` service `/segmentation/segment_prompt`. Localizes the prompt with Gemini, refines with SAM2, back-projects depth, and **publishes** the segmented + background point clouds (in `base_link`). The request is a JSON object `{prompt, min_stamp_ns}` (a bare prompt string still works); when `min_stamp_ns > 0` the service blocks until RGB+depth frames are stamped after that time (`PIPELINE_SEG_FRESH_FRAME_TIMEOUT_SEC`, default 5s) before segmenting. Runs under a `MultiThreadedExecutor` with the camera subscriptions in a reentrant group so frames keep arriving while the handler waits. |
| `graspgen_service.py` | `graspgen_service` | `StringString` service `/graspgen/infer`. The request carries a cloud-stamp token (empty = latest); GraspGen waits for the matching segmented cloud, runs inference (via a ZMQ client to a separate inference server), applies kinematic/collision filtering and ranking, and returns ranked grasp poses as JSON (with a `success` field). |
| `curobo_service.py` | `curobo_service` | `PlanTrajectory` service `/curobo/plan_trajectory`. Wraps the long-lived `CuRobo` planner. Three modes: pick (`grasp_poses` → approach+grasp / lift), single-pose (`grasp_pose`), and **`goal_name`** — resolves a `place_poses.yml` key to a collision-off safe-`transit_z` place transit (`trajectory` + bookshelf `insert_trajectory`/`retract_trajectory`) or, for `home`, a collision-aware c-space plan to the fixed `home_joint_config` (not an IK pose, so the arm always returns to the same centered, camera-down posture). Also publishes a latched `/curobo/ready` (`std_msgs/Bool`) once background init is done **and** the TSDF has ≥`MIN_FRAMES`, so the orchestrator can hold the first plan until the single-threaded executor has built the map (issuing one earlier parks that thread and starves the depth callbacks). |
| `curobo.py` | — | `CuRobo` class: dual-RGBD TSDF occupancy mapping + cuRobo motion planning. Owns only depth/CameraInfo/TF; fed joints via `update_joint_state()`. |
| `graspgen_client.py` | — | Minimal ZMQ client to the standalone GraspGen inference server. |
| `segmentation_utils.py` | — | Pure helpers for segmentation (resize, depth back-projection, downsample, centroid, overlay). |
| `live_viz_helpers.py` | — | Visualization helpers (point-cloud / TSDF). |
| `debug_viz.py` | `debug_viz` | Hosts one viser server; subscribes to the segmented/background/overhead clouds, `/graspgen/grasp_poses`, and `/curobo/tsdf_voxels` and renders them (grasp frames colored by rank; TSDF as voxel-sized squares colored by height; a "Layers" GUI folder toggles each layer's visibility). Debug mode only. |
| `graspgen_probe.py`, `graspgen_service_caller.py` | `graspgen_probe`, `graspgen_service_caller` | Standalone debugging utilities (not part of the runtime pipeline). |
| `home_config_tuner.py` | `home_config_tuner` | Standalone REPL tool (not part of the runtime pipeline): commands the arm to a typed 6-joint config via `/ur5_controller/follow_joint_trajectory`, saves the resulting wrist-camera frame to `/artifacts/home_tuner/`, and prints the achieved tool0 pose — used to pick `home_joint_config` for `place_poses.yml`. Run with the orchestrator stopped. |

The orchestrator coordinates stages over **ROS2 services**, not in-process Python calls. Stages exchange point clouds over ROS2 topics; GraspGen talks to its heavy inference model over ZMQ in a separate process.

### Ownership boundary (important)

- The **orchestrator** owns `/joint_states` and all action deployment (arm + gripper).
- **`CuRobo`** owns only depth images, CameraInfo, and TF; it never subscribes to `/joint_states` — the orchestrator/curobo_service feed joints in via `update_joint_state()`.

## ROS2 Topics & Services

External (from manip_challenge / Gazebo on host):

| Topic / interface | Type | Used by |
|---|---|---|
| `/task_commands` | `std_msgs/String` | orchestrator |
| `/wrist_camera/wrist_camera/color/image_raw` | `sensor_msgs/Image` | segmentation_service (RGB) |
| `/wrist_camera/wrist_camera/depth/color/image_raw` (+ `camera_info`) | `sensor_msgs/Image`, `CameraInfo` | segmentation_service, curobo |
| `/camera/camera/depth/color/image_raw` (+ `camera_info`) | `sensor_msgs/Image`, `CameraInfo` | curobo (overhead camera) |
| `/joint_states` | `sensor_msgs/JointState` | orchestrator, curobo_service |
| `/ur5_controller/follow_joint_trajectory` | action | orchestrator (arm) |
| `/gripper_controller/follow_joint_trajectory` | action | orchestrator (gripper) |

Internal:

| Interface | Type | Provider → consumer |
|---|---|---|
| `/segmentation/segment_prompt` | `riro_srvs/StringString` | orchestrator → segmentation_service |
| `/graspgen/segmented_object`, `/graspgen/background` | `sensor_msgs/PointCloud2` | segmentation_service → graspgen_service |
| `/graspgen/infer` | `riro_srvs/StringString` | orchestrator → graspgen_service (request = cloud-stamp token, response = JSON) |
| `/curobo/plan_trajectory` | `riro_srvs/PlanTrajectory` | orchestrator → curobo_service |
| `/graspgen/grasp_poses` | `geometry_msgs/PoseArray` | graspgen_service → debug_viz (debug only) |
| `/curobo/ready` | `std_msgs/Bool` (latched) | curobo_service → orchestrator (init done + TSDF ready) |
| `/curobo/tsdf_voxels` | `sensor_msgs/PointCloud2` | curobo_service → debug_viz (debug only) |
| `/curobo/overhead_cloud` | `sensor_msgs/PointCloud2` | curobo_service → debug_viz (overhead depth back-projection, debug only) |

## Adding Dependencies

Pip dependencies live under `requirements/` (`sam2.txt`, `graspgen.txt`, `curobo.txt`, `nvblox.txt`, `planner-runtime.txt`) and are installed in the Dockerfile. cuRobo must be installed from source.

## Key Files

- `Dockerfile` — single layer-ordered image (CUDA 12.8 + ROS2 Humble + PyTorch + SAM2/GraspGen/cuRobo + baked models; `COPY src` last). No separate base image.
- `docker-compose.yml` — deploy mode (baked image, runs `deploy.launch.py`).
- `docker-compose.debug.yml` — override that live-mounts `./src`/`./scripts`/`./config` and runs `debug.launch.py`.
- `src/team_8/launch/{pipeline_common,deploy,debug}.launch.py` — shared node graph + the two mode entry points.
- `src/team_8/team_8/` — all pipeline nodes (see table above)
- `src/utils/riro_srvs/srv/` — custom service definitions
- `src/team_8/config/ur5_curobo.yml` — cuRobo robot config (keep ASCII-only: cuRobo's `load_yaml` opens it with the container's default ASCII codec, so non-ASCII bytes crash it)
