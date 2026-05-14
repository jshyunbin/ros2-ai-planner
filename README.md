# ros2-ai-planner

Dockerized ROS2 AI planning node for the CS477 manipulation challenge. Runs alongside the existing `manip_challenge` system on the same machine and implements a full perception-to-action pipeline:

```
SAM2 (segment) → GraspGen (grasp pose) → cuRobo (trajectory) → UR5
```

## Requirements

- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- `manip_challenge` running on the host (Gazebo + ROS2 Humble)

## Usage

```bash
# Build the image (first run takes ~20 min — downloads PyTorch CUDA wheels)
docker compose build

# Start the container
docker compose up
# or
./scripts/run.sh
```

The container uses `network_mode: host`, so it automatically sees all ROS2 topics from the host (Gazebo, `/joint_states`, `/wrist_camera/...`, `/task_commands`).

## Package Structure

| Package | Service | Role |
|---|---|---|
| `sam2_node` | `~/segment` | Segments RGB image into object masks |
| `graspgen_node` | `~/generate_grasp` | Generates grasp pose from masks + depth |
| `curobo_node` | `~/plan_trajectory` | Plans joint trajectory to grasp pose |
| `pipeline_orchestrator` | — | Subscribes to `/task_commands`, chains the pipeline, sends trajectory to UR5 |

## Development

Source edits in `src/` take effect immediately inside the container — no rebuild needed (volume mount + `--symlink-install`).

To rebuild the ROS2 workspace inside the container:

```bash
docker compose run --rm ai_planner bash /ros2_ws/scripts/build.sh
```

To open an interactive shell:

```bash
docker compose run --rm ai_planner bash
```

## Adding Dependencies

Fill in the relevant file under `requirements/` and add a `pip3 install` step to the Dockerfile:

| File | For |
|---|---|
| `requirements/sam2.txt` | SAM2 |
| `requirements/graspgen.txt` | GraspGen |
| `requirements/curobo.txt` | cuRobo |
