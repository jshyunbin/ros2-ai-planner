# Live Visualization Test Script — Design

**Date:** 2026-05-27
**Branch:** nvblox
**File to create:** `src/pipeline_orchestrator/scripts/test_live_viz.py`

---

## Goal

Replace the synthetic depth in `test_pipeline_viz.py` with real depth frames from live ROS2 topics. The script subscribes to both overhead and wrist depth cameras, builds a real ESDF via the cuRobo Mapper, plans a trajectory with cuRobo MotionPlanner, and visualizes the result (point clouds, voxel occupancy, animated robot) in viser.

Gemini and SAM2 are **not** involved — the goal pose is hardcoded to `(0.3, 0.0, 0.4)` in the robot base frame.

---

## Architecture

A standalone script run directly with `python3` inside the container (not a ROS2 package entry point):

```
main thread:   viser server  ──────────────────────────────────────► (loops at ~10 Hz)
                   ▲  reads shared state (lock-protected)
                   │
bg thread:     rclpy.spin(node)
                   │
               Node callbacks:
                 _on_info(cam_id)   → stores intrinsics K
                 _on_depth(cam_id)  → TF lookup → Mapper.integrate()
                                      every 10 frames: compute_esdf() → _replan()
                                      stores: point_cloud, voxel_grid, traj
```

---

## ROS2 Topics Subscribed

| Topic | Type | Purpose |
|---|---|---|
| `/camera/camera/depth/color/image_raw` | `sensor_msgs/Image` | Overhead depth frames |
| `/camera/camera/depth/camera_info` | `sensor_msgs/CameraInfo` | Overhead intrinsics K |
| `/wrist_camera/wrist_camera/depth/color/image_raw` | `sensor_msgs/Image` | Wrist depth frames |
| `/wrist_camera/wrist_camera/depth/camera_info` | `sensor_msgs/CameraInfo` | Wrist intrinsics K |
| `/joint_states` | `sensor_msgs/JointState` | Robot start config for planning |

TF frames used: `camera_color_optical_frame` and `wrist_camera_color_optical_frame` → `world`.

---

## Shared State (protected by `threading.Lock`)

| Field | Type | Description |
|---|---|---|
| `_point_clouds` | `dict[str, tensor (N,3)]` | Unprojected XYZ per camera, for viser display |
| `_voxel_grid` | cuRobo `VoxelGrid` or `None` | Latest ESDF from `Mapper.compute_esdf()` |
| `_traj` | `ndarray (T, J)` or `None` | Latest planned trajectory waypoints |
| `_frame_count` | `int` | Total depth frames integrated; gates planning |
| `_latest_joints` | `JointState` msg or `None` | Latest `/joint_states` message |

---

## Background Thread — ROS2 Node

**`_on_info(msg, cam_id)`**
Exactly as in `curobo.py`: build a `(3,3)` float32 CUDA tensor from `msg.k` and store in `_cam_intrinsics[cam_id]`.

**`_on_depth(msg, cam_id, frame)`**
1. Guard: skip if intrinsics not yet received for this camera.
2. TF lookup (`tf2_ros.Buffer.lookup_transform`) from `world` → `frame` at message stamp with 0.1 s timeout; log warning and skip on failure.
3. `CvBridge.imgmsg_to_cv2` → float32 metres tensor (÷ 1000).
4. `FilterDepth` (bilateral, min/max clamp).
5. Unproject depth → XYZ point cloud using K; store in `_point_clouds[cam_id]`.
6. `Mapper.integrate(CameraObservation(...))` when both cameras have data.
7. Increment `_frame_count`; every 10 frames call `_replan()`.

**`_replan()`**
1. `voxel_grid = mapper.compute_esdf()`; store in `_voxel_grid`.
2. Build `start` from `_latest_joints` if available, else fall back to `HOME_CFG`.
3. `planner.plan_pose(goal=(0.3, 0.0, 0.4), start)`.
4. On success: store trajectory in `_traj`. On failure: keep previous trajectory (or home pose if no prior success).

---

## Main Thread — Viser

**Scene objects (updated in `update_loop()` at ~10 Hz):**

| Scene path | Content |
|---|---|
| `/world` | Axes frame |
| `/target` | Red icosphere at goal `(0.3, 0.0, 0.4)` |
| `/ur5` | Robot URDF via `ViserUrdf`, animated through `_traj` waypoints |
| `/depth/overhead` | Point cloud from `_point_clouds['overhead']` (grey) |
| `/depth/wrist` | Point cloud from `_point_clouds['wrist']` (blue) |
| `/esdf/voxels` | Occupied voxel centres from `_voxel_grid` (semi-transparent red) |
| `/status` | Text label: "Waiting for frames…" until `MIN_FRAMES`, then "Planning…" / "OK" |

Each `add_point_cloud` / `add_mesh` call replaces the previous object of the same name (viser deduplication by path).

---

## Error Handling

| Situation | Behaviour |
|---|---|
| TF lookup fails | `logger.warning`; frame skipped |
| Camera info not yet received | Depth frame silently skipped |
| `< MIN_FRAMES` integrated | Viser shows "Waiting…" label; no plan attempted |
| Planning fails | Keep previous `_traj`; if no prior success, show `HOME_CFG` |
| `/joint_states` not received | Fall back to `HOME_CFG` for start config |
| ESDF voxel grid empty | Skip `/esdf/voxels` update; no crash |

---

## Constants (matching existing `test_pipeline_viz.py` and `curobo.py`)

```python
UR5_CONFIG   = '/ros2_ws/src/pipeline_orchestrator/config/ur5_curobo.yml'
URDF_PATH    = '/ur5.urdf'
JOINT_NAMES  = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
HOME_CFG     = [0.0, -2.2, 1.9, -1.383, -1.57, 0.0]
GOAL_XYZ     = (0.3, 0.0, 0.4)
GOAL_QUAT    = (1.0, 0.0, 0.0, 0.0)  # w x y z
MIN_FRAMES   = 5
REPLAN_EVERY = 10   # depth frames between re-plans
VIZ_HZ       = 10
OVERHEAD_DEPTH_TOPIC = '/camera/camera/depth/color/image_raw'
OVERHEAD_INFO_TOPIC  = '/camera/camera/depth/camera_info'
WRIST_DEPTH_TOPIC    = '/wrist_camera/wrist_camera/depth/color/image_raw'
WRIST_INFO_TOPIC     = '/wrist_camera/wrist_camera/depth/camera_info'
OVERHEAD_FRAME = 'camera_color_optical_frame'
WRIST_FRAME    = 'wrist_camera_color_optical_frame'
WORLD_FRAME    = 'world'
```

---

## Run Command

```bash
docker compose run --rm -p 8080:8080 ai_planner \
  python3 /ros2_ws/src/pipeline_orchestrator/scripts/test_live_viz.py
```

Then open `http://localhost:8080` in your browser.

---

## Out of Scope

- SAM2 / Gemini / GraspGen integration
- CLI-configurable goal pose
- Re-plan button in viser GUI
- Writing trajectory back to `/ur5_controller/follow_joint_trajectory`
