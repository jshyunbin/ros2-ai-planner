# cuRoboV2 Depth-to-Planning Integration Design

**Date:** 2026-05-26 (revised)
**Branch:** nvblox
**Scope:** End-to-end RGBD → TSDF/ESDF → collision-aware MotionGen inside a single CuRobo class

---

## Overview

cuRoboV2 (v0.8.0) ships its own block-sparse TSDF/ESDF mapper (`curobo.perception.Mapper`) with
no external nvblox dependency. This replaces the original nvblox ROS node design entirely.

The `NvBlox` class and `nvblox.py` are **deleted**. The `CuRobo` class absorbs all
perception-to-planning responsibility: it subscribes to both RGBD streams, continuously fuses
them into an internal TSDF, computes the ESDF on demand, and runs collision-aware trajectory
optimization — all in one cohesive module.

---

## Architecture

```
[Overhead D435]  /camera/.../depth/color/image_raw     ──┐
                 /camera/.../depth/camera_info          ──┤
                                                          │   CuRobo._on_depth()
[Wrist D435]    /wrist_camera/.../depth/color/image_raw──┤   TF lookup → CameraObservation
                /wrist_camera/.../depth/camera_info     ──┘   FilterDepth → mapper.integrate()
                                                                      │
                          orchestrator._run_pipeline()                │ (continuous)
                                    │                                 ▼
                                    └──► CuRobo.plan_trajectory() ──► mapper.compute_esdf()
                                                                    → VoxelGrid
                                                                    → WorldVoxelCollision.update_voxel_data()
                                                                    → MotionGen.plan_single()
                                                                    → JointTrajectory
```

---

## What changes vs. current code

| Item | Before | After |
|------|--------|-------|
| `nvblox.py` | ESDF PointCloud2 subscriber stub | **Deleted** |
| `curobo.py` | Stub (takes `logger`) | Full Mapper + MotionGen (takes `node`) |
| `orchestrator.py` | Imports + instantiates NvBlox; passes esdf to CuRobo | Removes NvBlox entirely; calls `curobo.plan_trajectory(grasp_pose, joints)` |
| `Dockerfile` | `ros-humble-isaac-ros-nvblox` apt block | **Removed**; cuRoboV2 installed from source |
| `requirements/nvblox.txt` | nvblox placeholder comment | Cleared (no pip deps needed) |
| `requirements/curobo.txt` | Placeholder | cuRoboV2 source install + `warp-lang` |
| `launch/contest_run.launch.py` | (new) | Launches orchestrator only |
| `config/nvblox.yaml` | (new) | Mapper + CuRobo tuning params |

---

## Component Design

### 1. `curobo.py` — CuRobo class

**Responsibilities:** subscribe to both RGBD streams, fuse into TSDF, compute ESDF, run
MotionGen with WorldVoxelCollision.

**Constructor — `CuRobo(node: Node)`:**

```python
# Mapper (cuRoboV2 built-in, no nvblox)
self._mapper = Mapper(MapperCfg(
    voxel_size=0.05,
    extent_meters_xyz=(2.0, 2.0, 1.5),
    truncation_distance=0.15,
    depth_minimum_distance=0.15,
    depth_maximum_distance=2.0,
    decay_factor=1.0,
    frustum_decay_factor=1.0,
    enable_static=False,
    num_cameras=2,
    image_height=480,
    image_width=640,
))
self._depth_filter = FilterDepth(
    image_shape=(480, 640),
    depth_minimum_distance=0.15,
    depth_maximum_distance=2.0,
    flying_pixel_threshold=0.5,
    bilateral_kernel_size=3,
)

# Per-camera cache: latest (depth_tensor, intrinsics_tensor, Pose)
self._cam_cache: dict[str, tuple] = {}
self._frame_count = 0
self._tf_buffer = Buffer()
self._tf_listener = TransformListener(self._tf_buffer, node)

# ROS subscriptions
node.create_subscription(Image, OVERHEAD_DEPTH_TOPIC, self._on_overhead_depth, 10)
node.create_subscription(CameraInfo, OVERHEAD_INFO_TOPIC, self._on_overhead_info, 1)
node.create_subscription(Image, WRIST_DEPTH_TOPIC, self._on_wrist_depth, 10)
node.create_subscription(CameraInfo, WRIST_INFO_TOPIC, self._on_wrist_info, 1)

# MotionGen (WorldVoxelCollision) — initialized after warmup
self._motion_gen = self._build_motion_gen()
```

**Depth callback pattern** (same for both cameras, identified by `cam_id`):
1. Convert `Image` msg → float32 torch tensor (metres)
2. Look up TF: `world` ← `{cam}_color_optical_frame` at `msg.header.stamp`; skip on timeout
3. `FilterDepth` → clamp + remove flying pixels
4. Cache `(depth, intrinsics, pose)` for this `cam_id`
5. If both cameras cached: stack into batched `CameraObservation` → `mapper.integrate()` →
   `frame_count += 1`

**`plan_trajectory(grasp_pose, joint_states) → JointTrajectory | None`:**
1. If `frame_count < MIN_FRAMES` (=5): plan in free space (skip ESDF update, log warning)
2. `voxel_grid = self._mapper.compute_esdf()`
3. `self._motion_gen.world_collision.update_voxel_data(voxel_grid)`
4. Build cuRobo `JointState` from `joint_states`
5. Build cuRobo `Pose` from `grasp_pose`
6. `result = self._motion_gen.plan_single(start, goal, MotionGenPlanConfig(max_attempts=3))`
7. Return interpolated `JointTrajectory` or `None`

**TF frame constants (module-level):**
```python
OVERHEAD_DEPTH_TOPIC = '/camera/camera/depth/color/image_raw'
OVERHEAD_INFO_TOPIC  = '/camera/camera/depth/camera_info'
WRIST_DEPTH_TOPIC    = '/wrist_camera/wrist_camera/depth/color/image_raw'
WRIST_INFO_TOPIC     = '/wrist_camera/wrist_camera/depth/camera_info'
OVERHEAD_FRAME = 'camera_color_optical_frame'
WRIST_FRAME    = 'wrist_camera_color_optical_frame'
WORLD_FRAME    = 'world'
MIN_FRAMES     = 5
```

### 2. `orchestrator.py` — changes

Remove:
- `from pipeline_orchestrator.nvblox import NvBlox`
- `self._nvblox = NvBlox(self)`
- The `concurrent.futures` parallel ESDF fetch
- `esdf = future_esdf.result()` and `point_cloud = self._nvblox.extract_object_cloud(masks)`
  (point cloud comes from teammate's depth backprojection module — separate concern)

Change:
- `self._curobo = CuRobo(self.get_logger())` → `self._curobo = CuRobo(self)`
- `self._curobo.plan_trajectory(grasp_pose, joints, esdf=esdf)` →
  `self._curobo.plan_trajectory(grasp_pose, joints)`

The simplified `_run_pipeline`:
```python
def _run_pipeline(self, task: str):
    bbox = self._gemini.locate_object(self._latest_overhead_rgb, task)
    masks = self._sam2.segment(self._latest_overhead_rgb, prompt=task, bbox=bbox)
    if masks is None:
        return
    # point_cloud from teammate's module (separate PR)
    point_cloud = ...
    grasp_pose = self._graspgen.generate_grasp(point_cloud)
    if grasp_pose is None:
        return
    trajectory = self._curobo.plan_trajectory(grasp_pose, self._latest_joints)
    if trajectory is None:
        self.get_logger().warn('cuRobo failed, falling back to MoveIt2.')
        trajectory = self._moveit2.plan_trajectory(grasp_pose, self._latest_joints)
    if trajectory is None:
        return
    # TODO: execute trajectory
```

### 3. `nvblox.py` — deleted

The file is removed. The `extract_object_cloud` stub is left to the teammate's depth
backprojection work.

### 4. `Dockerfile` — changes

- Remove the `ros-humble-isaac-ros-nvblox` apt block (it was reverted in this PR; keep it gone)
- Add cuRoboV2 source install after PyTorch:

```dockerfile
# cuRoboV2 + Warp (GPU programming framework used internally by cuRobo)
RUN pip3 install --no-cache-dir warp-lang
RUN git clone --depth 1 --branch v0.8.0 https://github.com/NVlabs/curobo.git /tmp/curobo && \
    pip3 install --no-cache-dir /tmp/curobo && \
    rm -rf /tmp/curobo
```

### 5. `launch/contest_run.launch.py` (new)

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='pipeline_orchestrator',
            executable='orchestrator',
            name='pipeline_orchestrator',
            output='screen',
        ),
    ])
```

### 6. `config/nvblox.yaml` → `config/curobo.yaml` (new)

```yaml
curobo:
  mapper:
    voxel_size: 0.05
    extent_meters_xyz: [2.0, 2.0, 1.5]
    depth_minimum_distance: 0.15
    depth_maximum_distance: 2.0
    min_frames_before_esdf: 5
  tf:
    overhead_frame: camera_color_optical_frame
    wrist_frame: wrist_camera_color_optical_frame
    world_frame: world
  motion_gen:
    robot_config: ur5.yml   # verify filename in curobo/content/configs/robot/
    max_attempts: 3
```

---

## Error Handling

| Failure | Behaviour |
|---------|-----------|
| TF lookup timeout | Skip frame, log warning |
| camera_info not yet received | Skip depth frame (intrinsics not available) |
| `frame_count < MIN_FRAMES` at plan time | Plan in free space, log warning |
| `MotionGen.plan_single` fails | Return `None` → orchestrator falls back to MoveIt2 |
| MoveIt2 also fails | Return `None` → task skipped, log error |

---

## Open Questions (resolve at implementation start)

1. **TF frame names** — verify with `ros2 run tf2_tools view_frames` in the live simulation.
   Assumed: `camera_color_optical_frame` and `wrist_camera_color_optical_frame`.
2. **cuRoboV2 UR5 config filename** — after installing cuRoboV2, run:
   `find $(python3 -c "import curobo; print(curobo.__file__[:-12]}") -name "ur5*"` to find it.
3. **`WorldVoxelCollision` init shape** — must match Mapper's ESDF output shape. Derive from
   `extent_meters_xyz / voxel_size` at construction time.
4. **cuRoboV2 v0.8.0 tag** — confirm `v0.8.0` exists on the GitHub repo; if not, use `main`.
