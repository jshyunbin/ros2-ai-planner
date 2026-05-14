# ros2-ai-planner Design Spec

**Date:** 2026-05-14
**Status:** Approved

## Overview

A new standalone repository (`ros2-ai-planner`) that runs as a Dockerized external ROS2 node alongside the existing `manip_challenge` system. It implements a full perception-to-action pipeline: SAM2 segments the wrist camera feed, GraspGen generates grasp poses, and cuRobo plans the trajectory. The result is sent back to the UR5 controller directly via the `follow_joint_trajectory` action server.

The container runs on the same machine as `manip_challenge` and Gazebo. Host networking (`--network host`) means all ROS2 topics are shared transparently — no bridging required.

## Repo Structure

```
ros2-ai-planner/
├── Dockerfile                      # CUDA 12.8 + ROS2 Humble + PyTorch
├── docker-compose.yml              # host networking, nvidia runtime, volume mount
├── src/
│   ├── sam2_node/                  # perception: RGB image → segmentation masks
│   ├── graspgen_node/              # grasping: masks + depth → grasp pose
│   ├── curobo_node/                # planning: grasp pose + joint states → trajectory
│   └── pipeline_orchestrator/     # chains pipeline, owns all external ROS2 I/O
├── requirements/
│   ├── sam2.txt
│   ├── graspgen.txt
│   └── curobo.txt
└── scripts/
    ├── build.sh                    # colcon build --symlink-install inside container
    └── run.sh                      # docker compose up shortcut
```

## Dockerfile Strategy

- **Base image:** `nvidia/cuda:12.8.0-cudnn9-devel-ubuntu22.04`
- Install ROS2 Humble (ros-base + cv-bridge + vision-msgs) on top
- Install PyTorch with CUDA 12.8 wheels
- `colcon build --symlink-install` of all 4 packages at image build time
- Single container — all pipeline components share GPU memory

## ROS2 Package Architecture

The pipeline is sequential. The orchestrator drives execution by calling each AI node as a **ROS2 service** (request/response). The three AI nodes expose only internal services — they do not publish to or subscribe from the outside world directly.

### Data Flow

```
/task_commands ──────────────────────────────────────────► pipeline_orchestrator
/wrist_camera/.../color/image_raw ──► sam2_node (service)
/wrist_camera/.../depth/image_raw ──► graspgen_node (service)
/joint_states ───────────────────────► curobo_node (service)
                                                          │
                                                          ▼
                            /ur5_controller/follow_joint_trajectory (action)
```

### Package Interfaces

| Package | Exposes | Inputs | Output |
|---|---|---|---|
| `sam2_node` | service `~/segment` | RGB image + text prompt | segmentation masks |
| `graspgen_node` | service `~/generate_grasp` | masks + depth image | `geometry_msgs/PoseStamped` |
| `curobo_node` | service `~/plan_trajectory` | grasp pose + joint states | `trajectory_msgs/JointTrajectory` |
| `pipeline_orchestrator` | subscribes to `/task_commands` | calls all three services | sends trajectory to UR5 action server |

### External Topics (from manip_challenge / Gazebo)

| Topic | Type | Consumer |
|---|---|---|
| `/task_commands` | `std_msgs/String` | orchestrator (subscribe) |
| `/wrist_camera/wrist_camera/color/image_raw` | `sensor_msgs/Image` | orchestrator → sam2_node |
| `/wrist_camera/wrist_camera/depth/image_raw` | `sensor_msgs/Image` | orchestrator → graspgen_node |
| `/joint_states` | `sensor_msgs/JointState` | orchestrator → curobo_node |
| `/ur5_controller/follow_joint_trajectory` | action | orchestrator (publish) |

## docker-compose Configuration

```yaml
services:
  ai_planner:
    build: .
    network_mode: host
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=all
      - ROS_DOMAIN_ID=0
    volumes:
      - ./src:/ros2_ws/src
    ipc: host
    stdin_open: true
    tty: true
```

Key flags:
- `network_mode: host` — container shares host network stack; all ROS2 topics visible
- `runtime: nvidia` + `NVIDIA_VISIBLE_DEVICES=all` — full GPU access for SAM2, GraspGen, cuRobo
- `ipc: host` — enables ROS2 shared-memory transport
- `./src:/ros2_ws/src` volume mount — live Python edits without image rebuild (symlink install)

## Development Workflow

1. Edit Python files in `src/` on the host machine
2. Changes reflect immediately inside the container (volume mount + `--symlink-install`)
3. Rebuild the image only when adding system-level deps to `Dockerfile` or `requirements/*.txt`
4. `scripts/run.sh` → `docker compose up` to start
5. `scripts/build.sh` → `colcon build --symlink-install` inside the container

## Out of Scope (Initial Scaffold)

- Actual SAM2, GraspGen, cuRobo implementations — stub nodes only
- Custom ROS2 message types — use standard msgs for now
- CI/CD, devcontainer.json
- Multi-machine / multi-container networking
