# ros2-ai-planner

Dockerized ROS2 AI planning node for the CS477 manipulation challenge. Runs alongside the existing `manip_challenge` system on the same machine and implements a full perception-to-action pipeline:

```
Gemini Vision (task prompt → object bbox)
      │
SAM2 (overhead RGB + bbox → object mask)
      │
GraspGen (point cloud → grasp candidates)
      │
CuRobo (grasp + live ESDF → collision-free trajectory) → UR5
      │ fallback
MoveIt2 (trajectory) → UR5
```

**CuRobo owns the full depth pipeline.** It subscribes to both D435 depth streams internally, fuses them into a block-sparse TSDF/ESDF using cuRoboV2's built-in Mapper (GPU, no external nvblox node), and uses that map for collision-aware motion planning on every call.

## Requirements

- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- NVIDIA GPU (Turing or newer, ≥ 4 GB VRAM)
- NVIDIA driver ≥ 580 (CUDA 12 support)
- `manip_challenge` running on the host (Gazebo + ROS2 Humble)

## Usage

```bash
# Build the image (~30 min first run — downloads PyTorch + cuRoboV2)
docker compose build

# Run the full pipeline (competition command)
ros2 launch pipeline_orchestrator contest_run.launch.py

# Or start the container directly
docker compose up
```

The container uses `network_mode: host`, so it automatically sees all ROS2 topics from the host.

## Architecture

All pipeline logic lives in a single ROS2 package (`pipeline_orchestrator`). Modules are plain Python classes instantiated directly by the orchestrator — no inter-process ROS2 services, avoiding serialization overhead when passing tensors between stages.

```
src/pipeline_orchestrator/pipeline_orchestrator/
├── orchestrator.py   # ROS2 node — subscribes to topics, runs pipeline
├── curobo.py         # CuRobo — dual-RGBD Mapper + MotionPlanner (cuRoboV2)
├── gemini.py         # GeminiLocalizer — Gemini Vision API → object bbox (stub)
├── sam2.py           # Sam2 — segments overhead RGB using Gemini bbox (stub)
├── graspgen.py       # GraspGen — grasp pose from point cloud (stub)
└── moveit2.py        # MoveIt2 — fallback planner via move_group (stub)

src/pipeline_orchestrator/config/
├── ur5_curobo.yml    # UR5 robot config for cuRoboV2 (collision spheres, joint limits)
└── curobo.yaml       # Mapper + MotionPlanner tuning parameters

src/pipeline_orchestrator/launch/
└── contest_run.launch.py   # Competition launch file
```

### Topics subscribed (orchestrator)

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
| `/camera/camera/depth/camera_info` | `sensor_msgs/CameraInfo` | overhead D435 |
| `/wrist_camera/wrist_camera/depth/color/image_raw` | `sensor_msgs/Image` | wrist D435 |
| `/wrist_camera/wrist_camera/depth/camera_info` | `sensor_msgs/CameraInfo` | wrist D435 |

### Action clients

| Action | Type | Target |
|---|---|---|
| `/ur5_controller/follow_joint_trajectory` | `control_msgs/FollowJointTrajectory` | UR5 arm |
| `/gripper_controller/follow_joint_trajectory` | `control_msgs/FollowJointTrajectory` | Robotiq 85 gripper |

## Testing

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

Note: `--symlink-install` is not used in the Docker image (uv upgrades setuptools past the version that supports it). Rebuild the image after editing `setup.py` or `package.xml`.

## Implementation Status

| Module | Status |
|---|---|
| `curobo.py` — dual-RGBD Mapper (TSDF/ESDF fusion) | ✅ done |
| `curobo.py` — MotionPlanner (collision-aware planning) | ✅ done |
| `gemini.py` — `locate_object` | 🔧 stub (teammate) |
| `sam2.py` — `segment` | 🔧 stub |
| `graspgen.py` — `generate_grasp` | 🔧 stub |
| `moveit2.py` — `plan_trajectory` | 🔧 stub (fallback) |
