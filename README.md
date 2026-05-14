# ros2-ai-planner

Dockerized ROS2 AI planning node for the CS477 manipulation challenge. Runs alongside the existing `manip_challenge` system on the same machine and implements a full perception-to-action pipeline:

```
SAM2 (segment) → GraspGen (grasp pose) → cuRobo (trajectory) → UR5
                                             ↓ fallback
                                          MoveIt2 (trajectory)
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

The container uses `network_mode: host`, so it automatically sees all ROS2 topics from the host.

## Architecture

All pipeline logic lives in a single ROS2 package (`pipeline_orchestrator`). SAM2, GraspGen, and cuRobo are plain Python classes instantiated directly by the node — no inter-process ROS2 services. This avoids serialization overhead when passing tensors between pipeline stages.

MoveIt2 is the fallback motion planner if cuRobo fails. Because MoveIt2 is ROS2-native (it communicates with the `move_group` node via action/service clients), its module receives the full ROS2 node handle rather than just a logger.

```
src/pipeline_orchestrator/pipeline_orchestrator/
├── orchestrator.py   # ROS2 node — subscribes to sensors, runs pipeline, sends commands
├── sam2.py           # Sam2 class — segments RGB image into object masks
├── graspgen.py       # GraspGen class — generates grasp pose from masks + depth
├── curobo.py         # CuRobo class — plans joint trajectory to grasp pose
└── moveit2.py        # MoveIt2 class — fallback planner via move_group (ROS2-native)
```

### Topics subscribed

| Topic | Type | Source |
|---|---|---|
| `/task_commands` | `std_msgs/String` | manip_challenge |
| `/camera/camera/color/image_raw` | `sensor_msgs/Image` | overhead D435 |
| `/camera/camera/depth/color/image_raw` | `sensor_msgs/Image` | overhead D435 |
| `/wrist_camera/wrist_camera/color/image_raw` | `sensor_msgs/Image` | wrist D435 |
| `/wrist_camera/wrist_camera/depth/color/image_raw` | `sensor_msgs/Image` | wrist D435 |
| `/joint_states` | `sensor_msgs/JointState` | arm + gripper state |

### Action clients

| Action | Type | Target |
|---|---|---|
| `/ur5_controller/follow_joint_trajectory` | `control_msgs/FollowJointTrajectory` | UR5 arm |
| `/gripper_controller/follow_joint_trajectory` | `control_msgs/FollowJointTrajectory` | Robotiq 85 gripper |

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
