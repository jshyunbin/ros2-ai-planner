# Pipeline Debug Cheat Sheet

Quick copy-paste commands for running and debugging the planner pipeline.

## Host Env

```bash
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
```

```bash
ros2 service call /unpause_physics std_srvs/srv/Empty "{}"
```

## Build Images

```bash
cp .env.example .env
# Fill GEMINI_API_KEY in .env.
./scripts/build_base_image.sh
./scripts/build_image.sh
```

## Open Docker

Persistent container:

```bash
docker compose run --name ai_planner_dev --service-ports ai_planner bash
```

Persistent development container with source mounts:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml run --name ai_planner_dev --service-ports ai_planner bash
```

Extra shell:

```bash
docker exec -it ai_planner_dev bash
```

Cleanup:

```bash
docker rm -f ai_planner_dev
```

## Source Workspace

Run inside every container shell:

```bash
source /opt/ros/humble/setup.bash
cd /ros2_ws
source install/setup.bash
```

## Rebuild Mounted Source

Use this after source/interface edits in the development container:

```bash
source /opt/ros/humble/setup.bash
cd /ros2_ws
colcon build --symlink-install --packages-select riro_srvs pipeline_orchestrator
source install/setup.bash
```

## Launch Pipeline

Perception and grasp ranking only:

```bash
ros2 launch pipeline_orchestrator planner_pipeline.launch.py \
  start_graspgen_server:=true \
  auto_run_on_task_command:=true \
  enable_motion_execution:=false \
  use_sim_time:=true
```

Perception, grasp ranking, CuRobo service planning, and arm trajectory execution:

```bash
ros2 launch pipeline_orchestrator planner_pipeline.launch.py \
  start_graspgen_server:=true \
  auto_run_on_task_command:=true \
  enable_motion_execution:=true \
  use_sim_time:=true
```

Launch without auto-running task commands:

```bash
ros2 launch pipeline_orchestrator planner_pipeline.launch.py \
  start_graspgen_server:=true \
  auto_run_on_task_command:=false \
  enable_motion_execution:=false \
  use_sim_time:=true
```

## Publish Task

Run on host or inside a sourced container shell:

```bash
ros2 topic pub --once /task_commands std_msgs/msg/String "{data: 'banana'}"
```

## Manual Service Calls

```bash
ros2 service call /segmentation/segment_prompt riro_srvs/srv/StringString "{data: 'banana'}"
```

```bash
ros2 service call /graspgen/infer std_srvs/srv/Trigger "{}"
```

CuRobo planning service is normally called by `orchestrator` because it needs a grasp pose and joint state:

```bash
ros2 interface show riro_srvs/srv/PlanTrajectory
ros2 service type /curobo/plan_trajectory
```

## Inspect Runtime

```bash
ros2 node list
ros2 topic list
ros2 service list
ros2 action list
```

```bash
ros2 topic echo /task_commands
ros2 topic echo /joint_states --once
```

## Quick Checks

```bash
docker compose run --rm ai_planner env | grep -E 'ROS_DOMAIN_ID|ROS_LOCALHOST_ONLY|RMW_IMPLEMENTATION|FASTDDS_BUILTIN_TRANSPORTS|GEMINI_API_KEY'
```

```bash
ls /opt/GraspGen
ls /opt/GraspGenModels/checkpoints
ls /opt/models/sam2
```

## Artifacts

```text
./artifacts/segmentation_service/
./artifacts/graspgen_service/
```
