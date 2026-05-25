# ros2-ai-planner

Dockerized ROS2 AI planning node for the CS477 manipulation challenge. Runs alongside the existing `manip_challenge` system on the same machine and implements a full perception-to-action pipeline:

```
[parallel] Gemini Vision (task prompt → object bbox)
           nvblox (depth streams → persistent TSDF/ESDF)
                │
           SAM2 (overhead RGB + bbox → object mask)
                │
           nvblox.extract_object_cloud (mask → point cloud)
                │
           GraspGen (point cloud → grasp candidates)
                │
           cuRobo (grasp + ESDF → collision-free trajectory) → UR5
                │ fallback
           MoveIt2 (trajectory) → UR5
```

nvblox maintains a persistent scene map across pick-and-place cycles. After each grasp, the map automatically reflects the updated scene without a manual reset.

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

All pipeline logic lives in a single ROS2 package (`pipeline_orchestrator`). All modules are plain Python classes instantiated directly by the orchestrator node — no inter-process ROS2 services. This avoids serialization overhead when passing tensors between pipeline stages.

The Isaac ROS nvblox node runs as a separate process inside the same container, subscribing to both camera depth streams and publishing an ESDF over ROS2 topics. The `NvBlox` Python class wraps it.

```
src/pipeline_orchestrator/pipeline_orchestrator/
├── orchestrator.py   # ROS2 node — runs pipeline, dispatches all modules
├── nvblox.py         # NvBlox — subscribes to nvblox ESDF + extracts object point clouds
├── gemini.py         # GeminiLocalizer — Gemini Vision API → object bounding box (stub)
├── sam2.py           # Sam2 — segments overhead RGB using Gemini bbox
├── graspgen.py       # GraspGen — grasp pose from (N,3) point cloud
├── curobo.py         # CuRobo — collision-free trajectory via WorldNvbloxCollision
└── moveit2.py        # MoveIt2 — fallback planner via move_group (ROS2-native)
```

### Topics subscribed (orchestrator)

| Topic | Type | Source |
|---|---|---|
| `/task_commands` | `std_msgs/String` | manip_challenge |
| `/camera/camera/color/image_raw` | `sensor_msgs/Image` | overhead D435 |
| `/camera/camera/depth/color/image_raw` | `sensor_msgs/Image` | overhead D435 |
| `/wrist_camera/wrist_camera/color/image_raw` | `sensor_msgs/Image` | wrist D435 |
| `/wrist_camera/wrist_camera/depth/color/image_raw` | `sensor_msgs/Image` | wrist D435 |
| `/joint_states` | `sensor_msgs/JointState` | arm + gripper |

The Isaac ROS nvblox node (inside container) additionally subscribes to both depth topics and `/tf` to build the scene map.

### Topics subscribed (nvblox → orchestrator)

| Topic | Type | Published by |
|---|---|---|
| `/nvblox_node/static_esdf_pointcloud` | `sensor_msgs/PointCloud2` | Isaac ROS nvblox |

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
| `requirements/nvblox.txt` | nvblox Python bindings |

## Implementation Status

| Module | Status |
|---|---|
| `nvblox.py` — ESDF subscription | ✅ done |
| `nvblox.py` — `extract_object_cloud` | 🔧 stub |
| `gemini.py` — `locate_object` | 🔧 stub (teammate) |
| `sam2.py` — `segment` | 🔧 stub |
| `graspgen.py` — `generate_grasp` | 🔧 stub |
| `curobo.py` — `plan_trajectory` | 🔧 stub |
| `moveit2.py` — `plan_trajectory` | 🔧 stub |
