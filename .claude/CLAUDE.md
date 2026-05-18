# CLAUDE.md

This file provides guidance to Claude Code when working in this repository.

## Overview

`ros2-ai-planner` is a standalone Dockerized ROS2 node that acts as an external AI planning system for the CS477 `manip_challenge` pick-and-place project. It runs on the same machine as the `manip_challenge` simulation (Gazebo + ROS2 Humble) and communicates over a shared host network.

**Pipeline:** SAM2 (segment RGB image) → GraspGen (generate grasp pose from masks + depth) → cuRobo (plan joint trajectory) → send trajectory to UR5 via `/ur5_controller/follow_joint_trajectory`

All three AI modules are currently stubs. The goal is to implement SAM2, GraspGen, and cuRobo incrementally.

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
```

The container uses `network_mode: host` — all ROS2 topics from the host are immediately visible inside.

## Package Architecture

There is a single ROS2 package: `pipeline_orchestrator`.

| File | Role | Status |
|---|---|---|
| `orchestrator.py` | ROS2 node — subscribes to external topics, runs pipeline | stub |
| `sam2.py` | `Sam2` class — segmentation via SAM2 | stub |
| `graspgen.py` | `GraspGen` class — grasp pose via diffusion model | stub |
| `curobo.py` | `CuRobo` class — motion planning via cuRobo | stub |

The orchestrator instantiates all three modules and calls them as plain Python objects — no inter-process ROS2 services. This avoids serialization overhead when passing tensors between pipeline stages.

## ROS2 Topics (from manip_challenge / Gazebo on host)

| Topic | Type | Used by |
|---|---|---|
| `/task_commands` | `std_msgs/String` | orchestrator subscribes |
| `/wrist_camera/wrist_camera/color/image_raw` | `sensor_msgs/Image` | orchestrator → Sam2 |
| `/wrist_camera/wrist_camera/depth/image_raw` | `sensor_msgs/Image` | orchestrator → GraspGen |
| `/joint_states` | `sensor_msgs/JointState` | orchestrator → CuRobo |
| `/ur5_controller/follow_joint_trajectory` | action | orchestrator publishes |

## Adding Dependencies

When implementing a component, add its pip dependencies to the matching file and install in the Dockerfile:

- `requirements/sam2.txt` → Sam2 class
- `requirements/graspgen.txt` → GraspGen class
- `requirements/curobo.txt` → CuRobo class (note: cuRobo must be installed from source)

## Key Files

- `Dockerfile` — CUDA 12.8 + ROS2 Humble + PyTorch base image
- `docker-compose.yml` — host networking, NVIDIA runtime, live src volume mount
- `src/pipeline_orchestrator/pipeline_orchestrator/orchestrator.py` — ROS2 node entry point
- `src/pipeline_orchestrator/pipeline_orchestrator/sam2.py` — SAM2 module
- `src/pipeline_orchestrator/pipeline_orchestrator/graspgen.py` — GraspGen module
- `src/pipeline_orchestrator/pipeline_orchestrator/curobo.py` — cuRobo module
