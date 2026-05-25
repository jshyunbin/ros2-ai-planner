# ros2-ai-planner

Dockerized ROS2 AI planning workspace for the CS477 manipulation challenge.

Intended long-term pipeline:

```text
SAM2 (segment) -> pointcloud masking -> GraspGen (grasp pose) -> cuRobo (trajectory) -> UR5
                                                                  fallback
                                                                  MoveIt2
```

Current reality is narrower than that intended architecture. The repo is still mainly an integration scaffold plus a working raw-point-cloud GraspGen probe.

## Requirements

- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- host ROS2 Humble installation for `manip_challenge`
- `manip_challenge` running on the host (Gazebo + ROS2 Humble)

Important environment split:

- the host runs `manip_challenge`, Gazebo, and the base ROS2 graph
- the `ros2-ai-planner` container also includes ROS2, because it runs its own ROS2 nodes
- the two communicate over DDS using `network_mode: host`

## Usage

```bash
# Build the image
docker compose build

# Start the container
docker compose up
# or
./scripts/run.sh
```

The container uses `network_mode: host`, so it automatically sees all ROS2 topics from the host.

## Current Status

GraspGen work is now split into five practical layers:

1. Docker-only GraspGen inference check: done
   Result: local `GraspGen` Docker build works, the pretrained `robotiq_2f_140` checkpoint loads, and sample inference returns grasps.
2. Offline segmented-object feasibility check: done
   Result: the sample object-centric point clouds under `segmented_objects/` produce plausible grasp candidates through the standalone GraspGen server.
3. ROS2 raw-point-cloud probe: implemented
   Result: `graspgen_probe` exists and can send raw wrist point clouds to the standalone server.
4. In-container GraspGen integration: done at image level
   Result: the `ros2-ai-planner` Docker image now includes ROS2 plus the forked `pianojay/GraspGen` `jaeuk` branch and mounted `GraspGenModels`.
5. Full live object pipeline: not implemented
   Missing pieces: real 2D segmentation, mask-to-point-cloud conversion, calling GraspGen from planner code, and Gazebo grasp-success check.

The old NVIDIA driver mismatch was resolved by reboot. `nvidia-smi` is now healthy on driver `535.309.01`.

Important current reality:

- `ros2-ai-planner` now has two possible GraspGen development paths:
  - standalone GraspGen server in a separate container
  - embedded GraspGen inside the planner image
- the currently implemented ROS2-side node is still the lightweight remote client path via `graspgen_probe`
- the in-repo `SAM2` and `GraspGen` pipeline modules are still stubs
- `segmented_objects/` contains offline sample point clouds, not outputs of a live segmentation pipeline inside this repo

Known working offline samples:

- `segmented_object_banana.npy`
- `segmented_object_coke_can.npy`
- `segmented_object_hammer.npy`
- `segmented_object_meat_can.npy`
- `segmented_object_strawberry.npy`

Observed current single-object GraspGen numbers after model load:

- round-trip inference time: roughly `90-180 ms`
- GraspGen server process RAM: about `1.4 GiB`
- GPU memory: about `546 MiB`

These numbers are only for standalone GraspGen inference. They do not include segmentation, collision filtering, ROS2 transport overhead, or planning.

## Planner Image Status

The planner image is no longer just a lightweight ROS2 client image.

Current Docker image behavior:

- base image: local `graspgen:latest`
- adds ROS2 Humble runtime and Python tooling
- clones `https://github.com/pianojay/GraspGen.git` on branch `jaeuk` into `/opt/GraspGen`
- mounts local model assets from `../GraspGenModels` into `/opt/GraspGenModels`
- keeps the existing `pipeline_orchestrator` package and `graspgen_probe` entrypoint

Verified image-level checks:

- `docker compose build` succeeds
- `grasp_gen` imports successfully inside the planner container
- `/start_graspgen_server.sh` resolves the embedded repo and mounted model checkpoint
- model load succeeds inside the planner container

The only failed startup check was binding port `5556` when the standalone server was already using it. That is expected, not a model or dependency failure.

## Standalone GraspGen Baseline

This is the current known-good inference path.

From the local `GraspGen` repo root:

```bash
cd /home/user/JW/iir/GraspGen
bash docker/build.sh
MODELS_DIR=/home/user/JW/iir/GraspGenModels \
docker compose -f docker/compose.serve.yml up --build
```

The default server uses the pretrained `robotiq_2f_140` checkpoint and listens on `localhost:5556`.

The local `GraspGen` checkout required three fixes before this worked:

- `docker/graspgen_cuda121.dockerfile`
  - changed `pip install ./pointnet2_ops` to `pip install --no-build-isolation ./pointnet2_ops`
- `docker/serve.dockerfile`
  - corrected stale entrypoint path to `client-server/graspgen_server.py`
- `docker/run_server.sh`
  - corrected the same stale entrypoint path

## ROS2 Challenge-Side Probe

This uses the split development setup:

- `manip_challenge` runs on the host
- `GraspGen` runs in its own GPU container
- `ros2-ai-planner` runs the lightweight ROS2 probe client

Launch the normal `manip_challenge` Gazebo/ROS2 stack on the host. The probe expects:

```text
/wrist_camera/wrist_camera/depth/color/points
```

From this repo root:

```bash
docker compose build
docker compose run --rm ai_planner bash
```

Inside the container:

```bash
. /opt/ros/humble/setup.bash
cd /ros2_ws
colcon build --symlink-install
source install/setup.bash
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

If healthy, the node should log:

- successful connection to the GraspGen server
- point cloud reception from the wrist camera
- number of returned grasps and the best grasp translation/confidence

This probe still does not do segmentation, retargeting, collision filtering, or execution. It only verifies that ROS2 point cloud data can reach the GPU-side model and produce grasp candidates.

## Embedded GraspGen Path

The planner image can also launch GraspGen internally instead of depending on a separate GraspGen container.

From this repo root:

```bash
docker compose build
docker compose run --rm ai_planner bash
```

Inside the container:

```bash
/start_graspgen_server.sh
```

Defaults:

- repo path: `/opt/GraspGen`
- models path: `/opt/GraspGenModels`
- gripper config: `/opt/GraspGenModels/checkpoints/graspgen_robotiq_2f_140.yml`
- port: `5556`

Useful overrides:

```bash
GRASPGEN_PORT=5557 /start_graspgen_server.sh
GRIPPER_CONFIG=/opt/GraspGenModels/checkpoints/graspgen_franka_panda.yml /start_graspgen_server.sh
```

At the moment this only proves that the planner image can host GraspGen. The planner code itself is not yet calling the embedded model path.

## Architecture

Intended package layout:

```text
src/pipeline_orchestrator/pipeline_orchestrator/
├── orchestrator.py   # ROS2 node scaffold
├── graspgen_probe.py # ROS2 node: raw PointCloud2 -> standalone GraspGen server
├── sam2.py           # intended SAM2 segmentation module
├── graspgen.py       # intended GraspGen wrapper from masks + depth
├── curobo.py         # intended planner
└── moveit2.py        # intended fallback planner
```

Current implementation status:

- `orchestrator.py`: scaffold only
- `sam2.py`: stub, returns `None`
- `graspgen.py`: stub, returns `None`
- `curobo.py`: stub
- `moveit2.py`: stub
- `graspgen_probe.py`: implemented

This means there is not yet a proper runtime segmentation pipeline inside `ros2-ai-planner`.

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

Source edits in `src/` take effect immediately inside the container because of the volume mount plus `--symlink-install`.

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

## Immediate Next Step

The next milestone is:

1. choose the first runtime integration path inside planner code:
   - keep using the internal ZMQ server/client boundary inside one container, or
   - call GraspGen Python APIs directly
2. connect one segmented object sample to the embedded planner-side GraspGen path
3. implement and test the first real challenge-side vertical slice:
   - 2D segmentation
   - mask-to-point-cloud conversion
   - GraspGen inference
   - grasp success check in Gazebo

The final competition target still remains a single Docker image even though current debugging uses a split-container setup.
