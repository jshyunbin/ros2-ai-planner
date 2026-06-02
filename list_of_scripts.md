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

## Build Image

```bash
cp .env.example .env
# Fill GEMINI_API_KEY in .env.
docker compose build
```

## Open Docker

Deploy mode (full pipeline, no visualization):

```bash
docker compose up
```

Debug mode (same pipeline + viser visualization, `./src` live-mounted):

```bash
docker compose -f docker-compose.yml -f docker-compose.debug.yml up
```

Interactive shell (debug override gives live-mounted src):

```bash
docker compose -f docker-compose.yml -f docker-compose.debug.yml run --rm ai_planner bash
```

Persistent interactive container:

```bash
docker compose -f docker-compose.yml -f docker-compose.debug.yml run --name ai_planner_dev --service-ports ai_planner bash
```

Extra shell into running container:

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

The compose commands invoke the launch files automatically — you do not normally type `ros2 launch` by hand.

Deploy mode (full pipeline, executes on the UR5, no visualization — runs `deploy.launch.py`):

```bash
docker compose up
```

Debug mode (same pipeline + viser visualization on port 8080, `./src` live-mounted — runs `debug.launch.py`):

```bash
docker compose -f docker-compose.yml -f docker-compose.debug.yml up
```

Both modes auto-start the embedded GraspGen server and run motion execution. Debug mode additionally enables grasp-pose and TSDF publishing and the `debug_viz` viser node.

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
