# CLAUDE.md

This file provides guidance to Claude Code when working in this repository.

## Overview

`ros2-ai-planner` is a standalone Dockerized ROS2 system that acts as an external AI planning stack for the CS477 `manip_challenge` pick-and-place project. It runs on the same machine as the `manip_challenge` simulation (Gazebo + ROS2 Humble) and communicates over a shared host network.

**Pipeline:** task command → segmentation (Gemini bounding box + SAM2) → GraspGen (grasp poses from the segmented point cloud) → cuRobo (plan joint trajectory against a live dual-RGBD TSDF) → execute on the UR5 via `/ur5_controller/follow_joint_trajectory`.

All three AI stages are implemented and run as **separate ROS2 nodes**, coordinated by the orchestrator over ROS2 services. This is no longer a single stub node.

## Build & Run

```bash
# Build image (first run ~20 min — downloads PyTorch CUDA 12.8 wheels)
docker compose build

# Start container
docker compose up

# Interactive shell inside container
docker compose run --rm ai_planner bash

# Rebuild ROS2 workspace inside container (after adding packages)
docker compose run --rm ai_planner bash /ros2_ws/scripts/build.sh

# Launch the full pipeline (segmentation + graspgen + orchestrator;
# add enable_motion_execution:=true to also start curobo_service + execution)
ros2 launch pipeline_orchestrator planner_pipeline.launch.py
```

The container uses `network_mode: host` — all ROS2 topics from the host are immediately visible inside.

## Package Architecture

Two ROS2 packages live under `src/`:

- `pipeline_orchestrator` — all pipeline nodes (below).
- `utils/riro_srvs` — custom service definitions (`StringString`, `PlanTrajectory`, …).

Nodes (each is a `console_scripts` entry point in `setup.py`):

| Module | Node / entry point | Role |
|---|---|---|
| `orchestrator.py` | `orchestrator` | Drives the pipeline: subscribes `/task_commands`, calls the segmentation, GraspGen, and cuRobo services in turn, then executes the trajectory via FollowJointTrajectory actions. |
| `segmentation_service.py` | `segmentation_service` | `StringString` service `/segmentation/segment_prompt`. Localizes the prompt with Gemini, refines with SAM2, back-projects depth, and **publishes** the segmented + background point clouds (in `base_link`). |
| `graspgen_service.py` | `graspgen_service` | `StringString` service `/graspgen/infer`. The request carries a cloud-stamp token (empty = latest); GraspGen waits for the matching segmented cloud, runs inference (via a ZMQ client to a separate inference server), applies kinematic/collision filtering and ranking, and returns ranked grasp poses as JSON (with a `success` field). |
| `curobo_service.py` | `curobo_service` | `PlanTrajectory` service `/curobo/plan_trajectory`. Wraps the long-lived `CuRobo` planner; plans pick (approach+grasp / lift) or single-pose (place/home). |
| `curobo.py` | — | `CuRobo` class: dual-RGBD TSDF occupancy mapping + cuRobo motion planning. Owns only depth/CameraInfo/TF; fed joints via `update_joint_state()`. |
| `graspgen_client.py` | — | Minimal ZMQ client to the standalone GraspGen inference server. |
| `segmentation_utils.py` | — | Pure helpers for segmentation (resize, depth back-projection, downsample, centroid, overlay). |
| `live_viz_helpers.py` | — | Visualization helpers (point-cloud / TSDF). |
| `graspgen_probe.py`, `graspgen_service_caller.py` | `graspgen_probe`, `graspgen_service_caller` | Standalone debugging utilities (not part of the runtime pipeline). |

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

## Adding Dependencies

Pip dependencies live under `requirements/` (`sam2.txt`, `graspgen.txt`, `curobo.txt`, `nvblox.txt`, `planner-runtime.txt`) and are installed in the Dockerfile. cuRobo must be installed from source.

## Key Files

- `Dockerfile` / `Dockerfile.base` — CUDA 12.8 + ROS2 Humble + PyTorch base image
- `docker-compose.yml` — host networking, NVIDIA runtime (service: `ai_planner`); mounts `./artifacts` and `./config:ro`. **`src` is baked into the image at build time, not mounted** — code edits require a rebuild unless you use the dev override below.
- `docker-compose.dev.yml` — override that live-mounts `./src` and `./scripts` into `/ros2_ws/` for iterative development. Must be passed explicitly: `docker compose -f docker-compose.yml -f docker-compose.dev.yml run --rm ai_planner …` (or set `COMPOSE_FILE=docker-compose.yml:docker-compose.dev.yml`).
- `src/pipeline_orchestrator/launch/planner_pipeline.launch.py` — brings up the full pipeline
- `src/pipeline_orchestrator/pipeline_orchestrator/` — all pipeline nodes (see table above)
- `src/utils/riro_srvs/srv/` — custom service definitions
- `src/pipeline_orchestrator/config/ur5_curobo.yml` — cuRobo robot config (keep ASCII-only: cuRobo's `load_yaml` opens it with the container's default ASCII codec, so non-ASCII bytes crash it)
