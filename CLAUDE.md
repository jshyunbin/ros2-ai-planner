# CLAUDE.md

This file provides guidance to Claude Code when working in this repository.

## Overview

`ros2-ai-planner` is a standalone Dockerized ROS2 node that acts as an external AI planning system for the CS477 `manip_challenge` pick-and-place project. It runs on the same machine as the `manip_challenge` simulation (Gazebo + ROS2 Humble) and communicates over a shared host network.

**Pipeline:** SAM2 (segment RGB image) → GraspGen (generate grasp pose from masks + depth) → cuRobo (plan joint trajectory) → send trajectory to UR5 via `/ur5_controller/follow_joint_trajectory`

The three AI nodes are currently stubs. The goal is to implement SAM2, GraspGen, and cuRobo incrementally.

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

| Package | Location | Exposes | Status |
|---|---|---|---|
| `sam2_node` | `src/sam2_node/` | service `~/segment` | stub |
| `graspgen_node` | `src/graspgen_node/` | service `~/generate_grasp` | stub |
| `curobo_node` | `src/curobo_node/` | service `~/plan_trajectory` | stub |
| `pipeline_orchestrator` | `src/pipeline_orchestrator/` | subscribes to `/task_commands` | stub |

The `pipeline_orchestrator` is the only package that talks to the outside world. It subscribes to external topics, calls the three internal services in sequence, and sends the resulting trajectory to the UR5.

## ROS2 Topics (from manip_challenge / Gazebo on host)

| Topic | Type | Used by |
|---|---|---|
| `/task_commands` | `std_msgs/String` | orchestrator subscribes |
| `/wrist_camera/wrist_camera/color/image_raw` | `sensor_msgs/Image` | orchestrator → sam2_node |
| `/wrist_camera/wrist_camera/depth/image_raw` | `sensor_msgs/Image` | orchestrator → graspgen_node |
| `/joint_states` | `sensor_msgs/JointState` | orchestrator → curobo_node |
| `/ur5_controller/follow_joint_trajectory` | action | orchestrator publishes |

## Adding Dependencies

When implementing a component, add its pip dependencies to the matching file and install in the Dockerfile:

- `requirements/sam2.txt` → sam2_node
- `requirements/graspgen.txt` → graspgen_node
- `requirements/curobo.txt` → curobo_node (note: cuRobo must be installed from source)

## Key Files

- `Dockerfile` — CUDA 12.8 + ROS2 Humble + PyTorch base image
- `docker-compose.yml` — host networking, NVIDIA runtime, live src volume mount
- `src/pipeline_orchestrator/pipeline_orchestrator/orchestrator.py` — entry point for the full pipeline
- `docs/superpowers/specs/2026-05-14-ros2-ai-planner-design.md` — full design spec
- `docs/superpowers/plans/2026-05-14-ros2-ai-planner.md` — implementation plan
