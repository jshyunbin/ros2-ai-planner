# Operator Notes

Date: 2026-05-27

This file is the shortest practical guide for running `ros2-ai-planner`.

## Intent

Use one persistent Docker container, launch the planner stack once, and test the pipeline through `/task_commands`.

The final target is a single self-contained planner image:

- GraspGen code lives inside the image under `/opt/GraspGen`
- GraspGen checkpoints live inside the image under `/opt/GraspGenModels`
- host bind mounts are development-only and come from `docker-compose.dev.yml`
- runtime debug artifacts are written to host `./artifacts/`

Do not keep creating new `docker compose run ...` containers for each shell.

## Working Environment

Use these on the host when testing ROS2 communication:

```bash
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
```

The planner container receives the same values from `.env` through `docker compose`.

On the host, unpause Gazebo before testing:

```bash
ros2 service call /unpause_physics std_srvs/srv/Empty "{}"
```

## Docker Workflow

Build images from `ros2-ai-planner/`:

```bash
cp .env.example .env
# Fill in GEMINI_API_KEY and SAM3_API_KEY in .env if using segmentation.
./scripts/build_base_image.sh
./scripts/build_image.sh
```

Start one persistent planner container from the self-contained image:

```bash
docker compose run --name ai_planner_dev --service-ports ai_planner bash
```

For host bind mounts during development only:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml run --name ai_planner_dev --service-ports ai_planner bash
```

Open more shells into the same container:

```bash
docker exec -it ai_planner_dev bash
```

Remove the container when done:

```bash
docker rm -f ai_planner_dev
```

Verify the container sees the expected ROS2 and API environment:

```bash
docker compose run --rm ai_planner env | rg 'ROS_DOMAIN_ID|ROS_LOCALHOST_ONLY|RMW_IMPLEMENTATION|FASTDDS_BUILTIN_TRANSPORTS|GEMINI_API_KEY|SAM3_API_KEY'
```

Artifacts are saved on the host under:

```text
./artifacts/segmentation_service/
./artifacts/graspgen_service/
```

## Inside The Container

Source the environment in every shell:

```bash
source /opt/ros/humble/setup.bash
cd /ros2_ws
source install/setup.bash
```

Quick image sanity checks:

```bash
ls /opt/GraspGen
ls /opt/GraspGenModels/checkpoints
```

## Main Launcher

Recommended planner-side launch:

```bash
ros2 launch pipeline_orchestrator planner_pipeline.launch.py \
  start_graspgen_server:=true \
  auto_run_on_task_command:=true \
  use_sim_time:=true
```

This starts:

- embedded GraspGen server
- `segmentation_service`
- `graspgen_service`
- `orchestrator`

For debugging without automatic task execution:

```bash
ros2 launch pipeline_orchestrator planner_pipeline.launch.py \
  start_graspgen_server:=true \
  auto_run_on_task_command:=false \
  use_sim_time:=true
```

## Manual Rebuild

If mounted source changes inside the container with `docker-compose.dev.yml`:

```bash
source /opt/ros/humble/setup.bash
cd /ros2_ws
colcon build --symlink-install --packages-select pipeline_orchestrator
source install/setup.bash
```

## Essential Scripts

- `scripts/build_base_image.sh`
  - builds the reusable heavy Docker base image

- `scripts/build_image.sh`
  - builds the planner image

- `scripts/start_graspgen_server.sh`
  - starts embedded GraspGen manually if needed

## Essential Nodes

- `segmentation_service`
  - Gemini point prompts + SAM3 + pointcloud masking

- `graspgen_service`
  - subscribes to masked clouds and serves `/graspgen/infer`

- `orchestrator`
  - subscribes to `/task_commands` and runs the pipeline

## Minimal Live Test

Host:

```bash
source ~/.bashrc
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
ros2 service call /unpause_physics std_srvs/srv/Empty "{}"
ros2 topic pub --once /task_commands std_msgs/msg/String "{data: 'banana'}"
```

Container:

```bash
source /opt/ros/humble/setup.bash
cd /ros2_ws
source install/setup.bash
ros2 launch pipeline_orchestrator planner_pipeline.launch.py \
  start_graspgen_server:=true \
  auto_run_on_task_command:=true \
  use_sim_time:=true
```

## Minimal Debug Commands

Inside container:

```bash
ros2 node list
ros2 topic list
ros2 service list
ros2 topic echo /task_commands
ros2 service call /segmentation/segment_prompt riro_srvs/srv/StringString "{data: 'banana'}"
ros2 service call /graspgen/infer std_srvs/srv/Trigger "{}"
```

## Current Status

Working:

- single-image Docker build
- embedded GraspGen server startup
- host-to-container ROS transport
- RGB and organized pointcloud ingestion
- Gemini prompt generation
- SAM3 request path
- segmented pointcloud publication

Current blocker:

- external API reliability / authorization
  - Gemini may return `503 UNAVAILABLE`
  - SAM3 access may fail depending on account / edge policy
