# NvBlox + cuRoboV2 Integration Design

**Date:** 2026-05-26
**Branch:** nvblox
**Scope:** End-to-end RGBD → TSDF/ESDF → collision-aware MotionGen integration

---

## Overview

Replace the original nvblox ROS node + PointCloud2 approach with cuRoboV2's built-in `Mapper`
class, which fuses dual RGBD streams into a block-sparse TSDF/ESDF entirely in Python/CUDA with
no external nvblox dependency. The ESDF is passed as a `VoxelGrid` to `WorldVoxelCollision` for
collision-aware motion planning via cuRoboV2 `MotionGen`.

---

## Architecture

```
[Overhead D435]  depth + camera_info → TF lookup → CameraObservation(cam 0)
[Wrist D435]     depth + camera_info → TF lookup → CameraObservation(cam 1)
                                                        ↓
                                           NvBlox.Mapper.integrate()     (every frame pair)
                                                        ↓
                                           NvBlox.get_esdf_voxels()       (on demand)
                                           → mapper.compute_esdf() → VoxelGrid
                                                        ↓
                                   CuRobo.plan_trajectory(grasp_pose, joints, esdf_voxels)
                                   → WorldVoxelCollision.update_voxel_data(VoxelGrid)
                                   → MotionGen.plan_single(start, goal)
                                                        ↓
                                              JointTrajectory → UR5
```

### What changes vs. the current stubs

| Component | Before | After |
|-----------|--------|-------|
| `nvblox.py` | Subscribes to `/nvblox_node/static_esdf_pointcloud` | Holds `Mapper`, subscribes to depth + camera_info + TF |
| `curobo.py` | Stub | `MotionGen` + `WorldVoxelCollision` |
| `Dockerfile` | `ros-humble-isaac-ros-nvblox` apt block | Remove it; cuRoboV2 Mapper is self-contained |
| Launch file | (none) | `contest_run.launch.py` — orchestrator only, no nvblox node |

---

## Component Design

### 1. `nvblox.py` — NvBlox class

**Responsibilities:** fuse incoming depth frames from both cameras into a persistent TSDF and
expose the resulting ESDF as a `VoxelGrid`.

**Construction:**
```python
mapper = Mapper(MapperCfg(
    voxel_size=0.05,               # 5 cm — adequate for arm-scale collision
    extent_meters_xyz=(2.0, 2.0, 1.5),  # workspace bounding box in world frame
    truncation_distance=0.15,      # 3× voxel_size
    depth_minimum_distance=0.15,   # D435 min range
    depth_maximum_distance=2.0,    # table workspace max
    decay_factor=1.0,              # static scene — no time decay
    frustum_decay_factor=1.0,
    enable_static=False,           # no analytic primitives needed
    num_cameras=2,
    image_height=480,
    image_width=640,
))
depth_filter = FilterDepth(
    image_shape=(480, 640),
    depth_minimum_distance=0.15,
    depth_maximum_distance=2.0,
    flying_pixel_threshold=0.5,
    bilateral_kernel_size=3,
)
```

**Subscriptions added:**
- `/camera/camera/depth/color/image_raw` (`sensor_msgs/Image`)
- `/camera/camera/depth/camera_info` (`sensor_msgs/CameraInfo`)
- `/wrist_camera/wrist_camera/depth/color/image_raw` (`sensor_msgs/Image`)
- `/wrist_camera/wrist_camera/depth/camera_info` (`sensor_msgs/CameraInfo`)

**TF frames:**
- Overhead: `camera_color_optical_frame` → `world`
- Wrist: `wrist_camera_color_optical_frame` → `world`
- Looked up at each frame's timestamp via `tf2_ros.Buffer`.

**Key logic:**
- Each depth callback caches the latest depth + intrinsics + TF pose for that camera.
  On every callback from either camera, if both cameras have been seen at least once, the
  latest pair is stacked into a single batched `CameraObservation`
  (shape `(2, H, W)` for depth, `(2, 3, 3)` for intrinsics, `(2,)` for poses) and passed
  to `mapper.integrate()`. For a static scene, slight temporal mismatch between cameras is
  harmless — reinserting slightly stale data just reinforces existing TSDF weights.
- `FilterDepth` is applied before integration to remove flying pixels and clamp range.
- `get_esdf()` is removed. New public API: `get_esdf_voxels() → VoxelGrid | None`.
  Calls `mapper.compute_esdf()` on demand; returns `None` if fewer than
  `MIN_FRAMES_BEFORE_ESDF = 5` frames have been integrated (map not yet trustworthy).

### 2. `curobo.py` — CuRobo class

**Responsibilities:** initialize cuRoboV2 `MotionGen` with `WorldVoxelCollision`, update the
collision world from the latest ESDF, and run trajectory optimization.

**Construction (at node startup):**
```python
robot_cfg = RobotConfig.from_basic("ur5.yml", ...)
world_cfg = WorldCollisionConfig(
    world_model=WorldConfig(voxel=[VoxelGrid(...)]),  # pre-allocated to workspace bounds
)
motion_gen_cfg = MotionGenConfig.load_from_robot_config(
    robot_cfg, world_cfg, ...
)
motion_gen = MotionGen(motion_gen_cfg)
motion_gen.warmup()
```

**`plan_trajectory(grasp_pose, joint_states, esdf_voxels)`:**
1. If `esdf_voxels` is not `None`: call `motion_gen.world_collision.update_voxel_data(esdf_voxels)`
2. Build `JointState` start from `joint_states`
3. Build `Pose` goal from `grasp_pose`
4. Call `motion_gen.plan_single(start, goal, MotionGenPlanConfig(max_attempts=3))`
5. Return `Result.get_interpolated_plan()` as a `trajectory_msgs/JointTrajectory`, or `None` on failure

**Fallback:** if `esdf_voxels` is `None` (ESDF not ready yet), plan without updating the voxel
world. The pre-allocated grid is initialized to all-free so planning still works, just without
dynamic obstacle awareness.

### 3. `orchestrator.py` — changes

The `_run_pipeline` call site changes from:
```python
esdf = future_esdf.result()           # was PointCloud2 | None
point_cloud = self._nvblox.extract_object_cloud(masks)
...
trajectory = self._curobo.plan_trajectory(grasp_pose, joints, esdf=esdf)
```
to:
```python
esdf_voxels = self._nvblox.get_esdf_voxels()   # VoxelGrid | None
...
trajectory = self._curobo.plan_trajectory(grasp_pose, joints, esdf_voxels=esdf_voxels)
```

The `concurrent.futures` parallel block is simplified: Gemini still runs in parallel with
`get_esdf_voxels()` since the voxel extraction step (compute_esdf) is GPU-bound and can
overlap with the Gemini API call.

### 4. `Dockerfile` — changes

Remove the Isaac ROS apt block entirely (the one we just fixed). cuRoboV2's Mapper uses its own
Warp kernels and has no runtime dependency on `ros-humble-isaac-ros-nvblox`.

cuRoboV2 is installed from source per CLAUDE.md: `requirements/curobo.txt` pins the GitHub ref.

### 5. Launch file — `launch/contest_run.launch.py`

Starts one node: `pipeline_orchestrator`. No nvblox ROS node. The full competition invocation is:

```bash
ros2 launch pipeline_orchestrator contest_run.launch.py
```

The entrypoint already sources `/opt/ros/humble/setup.sh` and the colcon workspace.

---

## Data Flow — Frame Integration

```
overhead depth msg (Image)  →  _cache_overhead_depth()
wrist depth msg   (Image)   →  _cache_wrist_depth()
                                      ↓ (both updated)
               lookup TF: world←camera_color_optical_frame @ msg.header.stamp
               lookup TF: world←wrist_camera_color_optical_frame @ msg.header.stamp
                                      ↓
               FilterDepth × 2  (remove flying pixels, clamp range)
                                      ↓
               CameraObservation(
                   depth_image   = stack([overhead_depth, wrist_depth]),   # (2,H,W)
                   intrinsics    = stack([K_overhead, K_wrist]),           # (2,3,3)
                   pose          = Pose(positions, quaternions),           # (2,...)
               )
                                      ↓
               mapper.integrate(batched_obs)
               frame_count += 1
```

---

## Error Handling

| Failure | Behaviour |
|---------|-----------|
| TF lookup times out | Skip frame, log warning, continue |
| `get_esdf_voxels()` called before `MIN_FRAMES_BEFORE_ESDF` | Return `None`; cuRobo plans in free space |
| `MotionGen.plan_single` fails | Return `None`; orchestrator falls back to MoveIt2 |
| MoveIt2 also fails | Return `None`; orchestrator logs error, task skipped |
| Camera_info not yet received | Skip frame until intrinsics available |

---

## Configuration

All tunable parameters go in `config/nvblox.yaml`:

```yaml
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

curobo:
  robot_config: ur5.yml
  max_attempts: 3
```

---

## Testing

- **Unit:** mock `CameraObservation` with synthetic depth → verify `mapper.integrate()` increments
  frame count and `get_esdf_voxels()` returns a non-None `VoxelGrid` after `MIN_FRAMES_BEFORE_ESDF`
- **Integration:** publish synthetic depth images on the ROS topics, verify the orchestrator
  receives a non-None `esdf_voxels` before calling `plan_trajectory`
- **No hardware required** for either test since Mapper runs on any CUDA-capable GPU

---

## Open Questions (to resolve at implementation start)

1. **TF frame names** — overhead is assumed `camera_color_optical_frame`, wrist is assumed
   `wrist_camera_color_optical_frame`. Verify with `ros2 run tf2_tools view_frames` inside the
   running simulation before writing the TF lookup code.
2. **cuRoboV2 UR5 config** — check `python -c "import curobo; print(curobo.__file__)"` then
   look in `content/configs/robot/` for the UR5 YAML name (likely `ur5.yml` or `ur5e.yml`).
3. **`WorldVoxelCollision` pre-allocation** — grid dims are `(int(ex/vs), int(ey/vs), int(ez/vs))`
   from `extent_meters_xyz` and `voxel_size`; must match what `MapperCfg` allocates so the
   `VoxelGrid` from `compute_esdf()` can be passed to `update_voxel_data()` without shape errors.
