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

## Current Status

GraspGen work is currently split into two experiments:

1. Docker-only inference check
   Goal: confirm pretrained `GraspGen` inference works inside Docker on this machine at all.
2. Challenge-side feasibility check
   Goal: confirm pretrained `robotiq_2f_140` inference is usable when fed challenge point-cloud input.

At the moment, both are blocked by the local NVIDIA driver mismatch:

- loaded kernel module: `535.288.01`
- installed user-space NVIDIA libraries: `535.309.01`

The newer `535.309.01` DKMS module is already built for the running kernel, so a clean reboot is the most likely fix. After reboot, the first check should be:

```bash
nvidia-smi
```

Do not spend time on ROS2-side GraspGen testing until `nvidia-smi` is healthy.

## Experiment 1: Docker-Only Inference Check

This is the first real technical gate. It does not require ROS2 yet.

Target:
- build `GraspGen` Docker image
- load pretrained `robotiq_2f_140`
- run sample inference and verify grasps are returned

If this fails, stop there and debug Docker/GPU/model issues before touching the challenge pipeline.

### 1. Start the GraspGen server

From the `GraspGen` repo root:

```bash
bash docker/build.sh
MODELS_DIR=/absolute/path/to/GraspGenModels \
docker compose -f docker/compose.serve.yml up --build
```

The default server uses the pretrained `robotiq_2f_140` checkpoint and listens on `localhost:5556`.

You will also need a valid `GraspGenModels` directory containing the released checkpoints and sample data.

## Experiment 2: Challenge-Side Feasibility Check

This uses a split development setup:
- `manip_challenge` runs on the host
- `GraspGen` runs in its own GPU container
- `ros2-ai-planner` runs this lightweight ROS2 probe client

This is only worth running after Experiment 1 succeeds.

### 1. Start the challenge environment

Launch the normal `manip_challenge` Gazebo/ROS2 stack on the host. The probe expects the wrist point cloud topic:

```text
/wrist_camera/wrist_camera/depth/color/points
```

### 2. Start the planner container

From this repo root:

```bash
docker compose build
docker compose run --rm ai_planner bash
```

Inside the container, rebuild once if needed:

```bash
. /opt/ros/humble/setup.bash
cd /ros2_ws
colcon build --symlink-install
source install/setup.bash
```

### 3. Run the probe node

Inside the planner container:

```bash
ros2 run pipeline_orchestrator graspgen_probe
```

Useful overrides:

```bash
ros2 run pipeline_orchestrator graspgen_probe --ros-args \
  -p server_host:=127.0.0.1 \
  -p server_port:=5556 \
  -p point_cloud_topic:=/wrist_camera/wrist_camera/depth/color/points \
  -p max_points:=4096 \
  -p request_period_sec:=3.0
```

If the setup is healthy, the node should log:
- successful connection to the GraspGen server
- point cloud reception from the wrist camera
- number of returned grasps and the best grasp translation/confidence

This probe does not do segmentation, retargeting, collision filtering, or execution. It only verifies that ROS2 point cloud data can reach the GPU-side model and produce grasp candidates.

## Architecture

All pipeline logic lives in a single ROS2 package (`pipeline_orchestrator`). SAM2, GraspGen, and cuRobo are plain Python classes instantiated directly by the node — no inter-process ROS2 services. This avoids serialization overhead when passing tensors between pipeline stages.

MoveIt2 is the fallback motion planner if cuRobo fails. Because MoveIt2 is ROS2-native (it communicates with the `move_group` node via action/service clients), its module receives the full ROS2 node handle rather than just a logger.

```
src/pipeline_orchestrator/pipeline_orchestrator/
├── orchestrator.py   # ROS2 node — subscribes to sensors, runs pipeline, sends commands
├── graspgen_probe.py # ROS2 node — sends raw wrist point cloud to standalone GraspGen server
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
| `/wrist_camera/wrist_camera/depth/color/points` | `sensor_msgs/PointCloud2` | wrist D435 |
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
