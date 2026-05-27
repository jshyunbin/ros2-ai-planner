# Operator Notes

Date: 2026-05-27

This file is the shortest practical guide for running `ros2-ai-planner`.

## Intent

Use one persistent Docker container, launch the planner stack once, and test the pipeline through `/task_commands`.

Do not keep creating new `docker compose run ...` containers for each shell.

## Working Environment

Use these on both host and planner container when testing ROS2 communication:

```bash
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
```

On the host, unpause Gazebo before testing:

```bash
ros2 service call /unpause_physics std_srvs/srv/Empty "{}"
```

## Docker Workflow

Build images from `ros2-ai-planner/`:

```bash
./scripts/build_base_image.sh
./scripts/build_image.sh
```

Start one persistent planner container:

```bash
docker compose run --name ai_planner_dev --service-ports ai_planner bash
```

Open more shells into the same container:

```bash
docker exec -it ai_planner_dev bash
```

Remove the container when done:

```bash
docker rm -f ai_planner_dev
```

## Inside The Container

Source the environment in every shell:

```bash
source /opt/ros/humble/setup.bash
cd /ros2_ws
source install/setup.bash
```

For live segmentation:

```bash
export GEMINI_API_KEY=...
export SAM3_API_KEY=...
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

If mounted source changes inside the container:

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
ros2 topic pub /task_commands std_msgs/msg/String "{data: 'pick banana'}" -r 1
```

Container:

```bash
source /opt/ros/humble/setup.bash
cd /ros2_ws
source install/setup.bash
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
export GEMINI_API_KEY=...
export SAM3_API_KEY=...
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

- Docker build
- host-to-container ROS transport
- RGB and organized pointcloud ingestion
- Gemini prompt generation
- SAM3 request path
- segmented pointcloud publication
- embedded GraspGen startup

Current blocker:

- external API reliability / authorization
  - Gemini may return `503 UNAVAILABLE`
  - SAM3 access may fail depending on account / edge policy
