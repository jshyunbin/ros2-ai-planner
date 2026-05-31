# ros2-ai-planner

Dockerized ROS2 AI planning workspace for the CS477 manipulation challenge.

Intended long-term pipeline:

```text
SAM2 (segment) -> pointcloud masking -> GraspGen (grasp pose) -> cuRobo (trajectory) -> UR5
                                                                  fallback
                                                                  MoveIt2
```

Current reality is still narrower than the final target architecture, but the prompted segmentation to GraspGen inference path is now working end-to-end inside the planner stack.

**CuRobo owns the full depth pipeline.** It subscribes to both D435 depth streams internally, fuses them into a block-sparse TSDF/ESDF using cuRoboV2's built-in Mapper (GPU, no external nvblox node), and uses that map for collision-aware motion planning on every call.

## Requirements

- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- host ROS2 Humble installation for `manip_challenge`
- NVIDIA GPU (Turing or newer, ≥ 4 GB VRAM)
- NVIDIA driver ≥ 580 (CUDA 12 support)
- `manip_challenge` running on the host (Gazebo + ROS2 Humble)

Important environment split:

- the host runs `manip_challenge`, Gazebo, and the base ROS2 graph
- the `ros2-ai-planner` container also includes ROS2, because it runs its own ROS2 nodes
- the two communicate over DDS using `network_mode: host`
- planner service nodes use the vendored `riro_srvs/StringString` package under `src/utils/riro_srvs`
- the competition image must be self-contained; host bind mounts are now treated as development-only

## Usage

```bash
# Optional: create a local env file for ROS2 transport and API keys
cp .env.example .env

# Build the reusable ROS2/GraspGen base image once
./scripts/build_base_image.sh

# Build the planner image on top of that base
./scripts/build_image.sh

# Start the self-contained container
docker compose up
```

The default `docker-compose.yml` is now the competition-oriented path: no source or model bind mounts.
For local hot-reload development with bind mounts, use:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up
```

Container-side runtime environment expected by the planner:

- `ROS_DOMAIN_ID=0`
- `ROS_LOCALHOST_ONLY=0`
- `RMW_IMPLEMENTATION=rmw_fastrtps_cpp`
- `FASTDDS_BUILTIN_TRANSPORTS=UDPv4`
- `GEMINI_API_KEY` when running `segmentation_service`

## Current Status

Current work is now split into seven practical layers:

1. Docker-only GraspGen inference check: done
   Result: local `GraspGen` Docker build works, the pretrained `robotiq_2f_140` checkpoint loads, and sample inference returns grasps.
2. Offline segmented-object feasibility check: done
   Result: the sample object-centric point clouds under `segmented_objects/` produce plausible grasp candidates through the standalone GraspGen server.
3. ROS2 raw-point-cloud probe: implemented
   Result: `graspgen_probe` exists and can send raw wrist point clouds to the standalone server.
4. In-container GraspGen integration: done at image level
   Result: the `ros2-ai-planner` Docker image now includes ROS2 plus the forked `pianojay/GraspGen` `jaeuk` branch and a pinned GraspGen checkpoint set downloaded during image build.
5. Prompted segmentation service path: implemented
   Result: `segmentation_service` now performs `prompt -> Gemini bbox -> local Ultralytics SAM2 mask -> world-frame point-cloud masking`, publishes segmented/background clouds for GraspGen, and returns centroid/status to a ROS2 service caller.
6. GraspGen service path: implemented
   Result: `graspgen_service` now consumes segmented/background clouds, runs GraspGen inference, and returns ranked grasp candidates plus debug artifacts.
7. Full live object pipeline: partially implemented
   Missing pieces: better best-grasp filtering, actual arm/gripper execution, Gazebo grasp-success check, and replacing the remaining planner stubs (`curobo.py`, `moveit2.py`).

The old NVIDIA driver mismatch was resolved by reboot. `nvidia-smi` is now healthy on driver `535.309.01`.

Important current reality:

- `ros2-ai-planner` now has two possible GraspGen development paths:
  - standalone GraspGen server in a separate container
  - embedded GraspGen inside the planner image
- the ROS2 side now has both:
  - `graspgen_probe` for raw point-cloud probing
  - `segmentation_service` for prompted segmentation and masked cloud publication
- `graspgen_service` for ranked grasp inference on segmented object clouds
- `orchestrator.py` now acts as a ROS2 service caller for segmentation and GraspGen
- the older in-repo `SAM2` and `GraspGen` wrapper modules are still stubs and are no longer the primary integration path
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

- reusable heavy base image: `ros2-ai-planner-base:latest`
- base image starts from public `nvcr.io/nvidia/pytorch:23.07-py3`
- base image adds ROS2 Humble runtime, OpenCV, GraspGen source, and Python tooling
- planner image then only copies `src/`, builds the ROS2 workspace, and installs entrypoint scripts
- required GraspGen model assets are downloaded from Hugging Face during image build and copied into the image

Verified image-level checks:

- `./scripts/build_base_image.sh` creates the reusable base image
- `./scripts/build_image.sh` builds the thin planner image on top of it
- `grasp_gen` imports successfully inside the planner container
- `/start_graspgen_server.sh` resolves the embedded repo and the checkpoint downloaded into the image at build time
- model load succeeds inside the planner container

Pinned external sources used by the image build:

- GraspGen code: `https://github.com/pianojay/GraspGen.git` branch `jaeuk` at commit `beddd216a62781670a9b0938e7624b1ea10925f6`
- GraspGen model repo: `https://huggingface.co/adithyamurali/GraspGenModels` at commit `ec1ccbb5eec0680db669246ac312a3636f16ee43`

This means directory mounts are no longer required for GraspGen assets, but internet access is required when building the base image unless you prebuild and distribute the image itself.

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
./scripts/build_image.sh
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
./scripts/build_image.sh
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

This embedded path is now used by the planner-side segmentation and grasp service flow.

## Single Launcher

The planner-side stack can now be started with one ROS2 launch command.

Inside the planner container:

```bash
source /opt/ros/humble/setup.bash
cd /ros2_ws
source install/setup.bash
export GEMINI_API_KEY=...
ros2 launch pipeline_orchestrator planner_pipeline.launch.py \
  start_graspgen_server:=true \
  auto_run_on_task_command:=true
```

## Ultralytics SAM2 Status

The current planner image can install and run Ultralytics SAM2 without changing the
existing torch/CUDA stack.

- install path used in this repo: `pip install --no-deps "ultralytics>=8.2.70"`
- rationale: the base image already contains a working runtime set, and allowing pip
  to resolve dependencies would risk replacing pinned GPU packages used elsewhere
- baked checkpoint path in the image: `/opt/models/sam2/sam2_t.pt`
- practical status: local testing in the container succeeded with Ultralytics SAM2
  checkpoint download and image inference

This is distinct from the official `facebookresearch/sam2` installation path, which
has different version expectations and is not the integration target for this repo.

What this starts:

- embedded GraspGen server when `start_graspgen_server:=true`
- `segmentation_service`
- `graspgen_service`
- `orchestrator`

Useful overrides:

```bash
ros2 launch pipeline_orchestrator planner_pipeline.launch.py \
  start_graspgen_server:=false \
  graspgen_host:=127.0.0.1 \
  graspgen_port:=5556 \
  use_sim_time:=true \
  auto_run_on_task_command:=false
```

Default topic wiring:

- RGB: `/camera/camera/color/image_raw`
- organized point cloud: `/camera/camera/depth/color/points`
- segmentation service: `/segmentation/segment_prompt`
- segmented object cloud: `/graspgen/segmented_object`
- background cloud: `/graspgen/background`
- grasp service: `/graspgen/infer`

## Architecture

Intended package layout:

```text
src/pipeline_orchestrator/pipeline_orchestrator/
├── orchestrator.py   # ROS2 node: task command -> segmentation service -> GraspGen service
├── segmentation_service.py # ROS2 node: Gemini bbox + local SAM2 + point-cloud masking
├── segmentation_utils.py   # helpers for mask parsing / rasterization / cloud extraction
├── graspgen_probe.py # ROS2 node: raw PointCloud2 -> standalone GraspGen server
├── curobo.py         # cuRoboV2 dual-RGBD Mapper + MotionPlanner
├── live_viz_helpers.py # helpers for live point-cloud / ESDF visualization
├── sam2.py           # intended SAM2 segmentation module
├── graspgen.py       # intended GraspGen wrapper from masks + depth
└── moveit2.py        # intended fallback planner
```

Current implementation status:

- `orchestrator.py`: implemented as a segmentation + GraspGen service caller with optional motion execution
- `segmentation_service.py`: implemented
- `segmentation_utils.py`: implemented
- `sam2.py`: stub, returns `None`
- `graspgen.py`: stub, returns `None`
- `curobo.py`: implemented from the main branch CuRoboV2 integration
- `moveit2.py`: stub
- `graspgen_probe.py`: implemented

This means `ros2-ai-planner` now has a runtime segmentation-to-GraspGen path, and CuRobo planning can be enabled separately with the `enable_motion_execution` orchestrator parameter.

### Topics subscribed

| Topic | Type | Source |
|---|---|---|
| `/task_commands` | `std_msgs/String` | manip_challenge |
| `/camera/camera/color/image_raw` | `sensor_msgs/Image` | overhead D435 |
| `/wrist_camera/wrist_camera/color/image_raw` | `sensor_msgs/Image` | wrist D435 |
| `/joint_states` | `sensor_msgs/JointState` | arm + gripper |

### Topics subscribed (CuRobo — depth, internal)

| Topic | Type | Source |
|---|---|---|
| `/camera/camera/depth/color/image_raw` | `sensor_msgs/Image` | overhead D435 |
| `/camera/camera/depth/color/camera_info` | `sensor_msgs/CameraInfo` | overhead D435 |
| `/wrist_camera/wrist_camera/depth/color/image_raw` | `sensor_msgs/Image` | wrist D435 |
| `/wrist_camera/wrist_camera/depth/color/camera_info` | `sensor_msgs/CameraInfo` | wrist D435 |

### Action clients

| Action | Type | Target |
|---|---|---|
| `/ur5_controller/follow_joint_trajectory` | `control_msgs/FollowJointTrajectory` | UR5 arm |
| `/gripper_controller/follow_joint_trajectory` | `control_msgs/FollowJointTrajectory` | Robotiq 85 gripper |

## Testing

With the development override compose file, source edits in `src/` take effect immediately inside the container because of the volume mount plus `--symlink-install`.

### Pipeline visualization (web-based, works over SSH)

```bash
# Forward the port if on SSH
ssh -L 8080:localhost:8080 user@host

# Run the test
docker compose run --rm -p 8080:8080 ai_planner \
  python3 /ros2_ws/src/pipeline_orchestrator/scripts/test_pipeline_viz.py
```

Open `http://localhost:8080`. The script runs synthetic depth frames through the full Mapper → MotionPlanner pipeline and animates the planned trajectory on the UR5 model.

### Unit tests

```bash
docker compose run --rm ai_planner bash -c "
  source /ros2_ws/install/setup.bash &&
  python3 -m pytest src/pipeline_orchestrator/test/test_orchestrator.py -v"
```

## Development

To rebuild the ROS2 workspace inside the container:

```bash
docker compose run --rm ai_planner bash /ros2_ws/scripts/build.sh
```

To open an interactive shell:

```bash
docker compose run --rm ai_planner bash
```

## Build Speed Mitigation

The slowest rebuild layer is installing ROS2 Humble and related apt packages. That layer is now split out into a reusable base image:

- `./scripts/build_base_image.sh`
  - rebuild only when ROS2/system Python/GraspGen base dependencies change
- `./scripts/build_image.sh`
  - reuses the base image and rebuilds only the planner workspace image

If the base image already exists locally, `build_image.sh` skips rebuilding it.

## Adding Dependencies

## Implementation Status

| Module | Status |
|---|---|
| `requirements/sam2.txt` | SAM2 |
| `requirements/graspgen.txt` | GraspGen |
| `requirements/curobo.txt` | cuRobo |
| `curobo.py` — dual-RGBD Mapper + MotionPlanner | implemented |
| `orchestrator.py` — service pipeline | implemented |
| `orchestrator.py` — motion execution | optional, gated by `enable_motion_execution` |

## Immediate Next Step

The next milestone is grasp selection and execution:

1. improve best-grasp filtering for vertical pickup in a world-frame interpretation
2. reject obvious bad grasps using scene-specific geometric filters such as basket height
3. connect the top-ranked grasp to actual arm and gripper execution
4. verify grasp success in Gazebo

The final competition target remains a single Docker image that can segment, choose a grasp, and execute the pickup without relying on split-container debugging paths.
