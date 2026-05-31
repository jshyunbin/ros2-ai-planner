# Live Visualization Test Script Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `scripts/test_live_viz.py` — a standalone script that subscribes to live ROS2 depth topics, fuses them via cuRobo Mapper into an ESDF, plans a UR5 trajectory with cuRobo MotionPlanner, and visualizes point clouds, voxel occupancy, and an animated robot in viser.

**Architecture:** A `rclpy` node spins on a daemon background thread, calling `Mapper.integrate()` on each stereo depth frame and re-planning every 10 frames. The main thread runs a viser server at 10 Hz, reading lock-protected `SharedState` to update point clouds, ESDF voxels, and robot joint angles. TF lookups use `tf2_ros` (same pattern as `curobo.py`).

**Tech Stack:** Python 3.10, ROS2 Humble, cuRoboV2 (`Mapper`, `FilterDepth`, `MotionPlanner`), `tf2_ros`, `cv_bridge`, `viser`, PyTorch CUDA, NumPy

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Create | `src/pipeline_orchestrator/pipeline_orchestrator/live_viz_helpers.py` | Pure helpers: `depth_to_xyz`, `esdf_to_points`, `resolve_urdf` |
| Create | `src/pipeline_orchestrator/test/test_live_viz_helpers.py` | Unit tests for the above |
| Create | `src/pipeline_orchestrator/scripts/test_live_viz.py` | Main script: constants, SharedState, LiveVizNode, update_loop, main |

---

### Task 1: Create live_viz_helpers.py — depth_to_xyz (TDD)

**Files:**
- Create: `src/pipeline_orchestrator/pipeline_orchestrator/live_viz_helpers.py`
- Create: `src/pipeline_orchestrator/test/test_live_viz_helpers.py`

- [ ] **Step 1: Write failing tests for depth_to_xyz**

Create `src/pipeline_orchestrator/test/test_live_viz_helpers.py`:

```python
import torch
import numpy as np
import pytest


def test_depth_to_xyz_single_pixel():
    """Valid pixel at (u=0, v=0) with depth=1.0 m unprojected correctly."""
    from pipeline_orchestrator.live_viz_helpers import depth_to_xyz

    K = torch.tensor([
        [500.0,   0.0, 320.0],
        [  0.0, 500.0, 240.0],
        [  0.0,   0.0,   1.0],
    ], dtype=torch.float32)
    depth = torch.zeros(2, 2, dtype=torch.float32)
    depth[0, 0] = 1.0   # valid pixel at (v=0, u=0)

    xyz = depth_to_xyz(depth, K)

    assert xyz.shape == (1, 3)
    # x = (u - cx)/fx * z = (0 - 320)/500 * 1 = -0.64
    assert abs(xyz[0, 0].item() - (-0.64)) < 1e-5
    # y = (v - cy)/fy * z = (0 - 240)/500 * 1 = -0.48
    assert abs(xyz[0, 1].item() - (-0.48)) < 1e-5
    # z = depth = 1.0
    assert abs(xyz[0, 2].item() - 1.0) < 1e-5


def test_depth_to_xyz_zero_pixels_excluded():
    """Pixels with depth == 0 must not appear in output."""
    from pipeline_orchestrator.live_viz_helpers import depth_to_xyz

    K = torch.tensor([
        [500.0,   0.0, 320.0],
        [  0.0, 500.0, 240.0],
        [  0.0,   0.0,   1.0],
    ], dtype=torch.float32)
    depth = torch.zeros(4, 4, dtype=torch.float32)   # all zero → all invalid

    xyz = depth_to_xyz(depth, K)

    assert xyz.shape[0] == 0


def test_depth_to_xyz_centre_pixel():
    """Pixel at the principal point (cx, cy) should have x=y=0."""
    from pipeline_orchestrator.live_viz_helpers import depth_to_xyz

    fx, fy, cx, cy = 600.0, 600.0, 80.0, 60.0
    H, W = 120, 160
    K = torch.tensor([
        [fx,  0.0, cx],
        [0.0, fy,  cy],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32)
    depth = torch.zeros(H, W, dtype=torch.float32)
    depth[int(cy), int(cx)] = 2.0   # at principal point, depth = 2 m

    xyz = depth_to_xyz(depth, K)

    assert xyz.shape == (1, 3)
    assert abs(xyz[0, 0].item()) < 1e-4    # x ≈ 0
    assert abs(xyz[0, 1].item()) < 1e-4    # y ≈ 0
    assert abs(xyz[0, 2].item() - 2.0) < 1e-5
```

- [ ] **Step 2: Run tests to confirm they fail (module missing)**

```bash
docker compose run --rm ai_planner \
  python3 -m pytest src/pipeline_orchestrator/test/test_live_viz_helpers.py -v 2>&1 | head -20
```
Expected: `ModuleNotFoundError: No module named 'pipeline_orchestrator.live_viz_helpers'`

- [ ] **Step 3: Create live_viz_helpers.py with depth_to_xyz**

Create `src/pipeline_orchestrator/pipeline_orchestrator/live_viz_helpers.py`:

```python
"""Pure helper functions shared by live visualization scripts.

All functions here are stateless and have no ROS2 or cuRobo imports,
so they can be unit-tested directly without a ROS2 environment.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch


def depth_to_xyz(depth_m: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Unproject a depth image into a point cloud in camera frame.

    Args:
        depth_m: (H, W) float32 tensor, metres; zero values are invalid.
        K:       (3, 3) float32 tensor, camera intrinsics matrix.
                 Works on any device — output is on the same device as depth_m.

    Returns:
        (N, 3) float32 tensor of valid (x, y, z) points in camera frame.
        Returns shape (0, 3) when there are no valid pixels.
    """
    H, W = depth_m.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    v_coords, u_coords = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=depth_m.device),
        torch.arange(W, dtype=torch.float32, device=depth_m.device),
        indexing='ij',
    )

    valid = depth_m > 0
    z = depth_m[valid]
    x = (u_coords[valid] - cx) / fx * z
    y = (v_coords[valid] - cy) / fy * z

    return torch.stack([x, y, z], dim=-1)
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
docker compose run --rm ai_planner \
  python3 -m pytest src/pipeline_orchestrator/test/test_live_viz_helpers.py -v
```
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/pipeline_orchestrator/pipeline_orchestrator/live_viz_helpers.py \
        src/pipeline_orchestrator/test/test_live_viz_helpers.py
git commit -m "feat: add depth_to_xyz helper with tests"
```

---

### Task 2: Add esdf_to_points and resolve_urdf to live_viz_helpers.py (TDD)

**Files:**
- Modify: `src/pipeline_orchestrator/pipeline_orchestrator/live_viz_helpers.py`
- Modify: `src/pipeline_orchestrator/test/test_live_viz_helpers.py`

- [ ] **Step 1: Append failing tests for esdf_to_points**

Append to `src/pipeline_orchestrator/test/test_live_viz_helpers.py`:

```python
import types


def test_esdf_to_points_extracts_occupied():
    """Voxels with ESDF ≤ 0 are occupied; their centres should be returned."""
    from pipeline_orchestrator.live_viz_helpers import esdf_to_points

    esdf = torch.zeros(2, 2, 2, dtype=torch.float32)
    esdf[0, 0, 0] = -0.1   # occupied
    esdf[1, 0, 0] =  0.1   # free

    vg = types.SimpleNamespace(
        esdf_tensor=esdf,
        origin=torch.tensor([0.0, 0.0, 0.0]),
        voxel_size=0.1,
    )

    pts = esdf_to_points(vg)

    assert pts.shape == (1, 3)
    # centre of voxel index (0,0,0): origin + 0*size + size/2 = 0.05
    np.testing.assert_allclose(pts[0], [0.05, 0.05, 0.05], atol=1e-6)


def test_esdf_to_points_all_free():
    """Grid with all positive ESDF values should return empty (0, 3) array."""
    from pipeline_orchestrator.live_viz_helpers import esdf_to_points

    vg = types.SimpleNamespace(
        esdf_tensor=torch.ones(2, 2, 2, dtype=torch.float32),
        origin=torch.tensor([0.0, 0.0, 0.0]),
        voxel_size=0.1,
    )

    pts = esdf_to_points(vg)

    assert pts.shape == (0, 3)
    assert pts.dtype == np.float32


def test_esdf_to_points_malformed_object():
    """Missing attributes on voxel_grid must return empty array, not raise."""
    from pipeline_orchestrator.live_viz_helpers import esdf_to_points

    vg = types.SimpleNamespace()   # no attributes at all

    pts = esdf_to_points(vg)

    assert pts.shape == (0, 3)
```

- [ ] **Step 2: Run tests to confirm they fail (function missing)**

```bash
docker compose run --rm ai_planner \
  python3 -m pytest src/pipeline_orchestrator/test/test_live_viz_helpers.py \
                    -k "esdf" -v 2>&1 | head -20
```
Expected: `ImportError: cannot import name 'esdf_to_points'`

- [ ] **Step 3: Append esdf_to_points and resolve_urdf to live_viz_helpers.py**

Append after `depth_to_xyz` in `src/pipeline_orchestrator/pipeline_orchestrator/live_viz_helpers.py`:

```python


def esdf_to_points(voxel_grid: object) -> np.ndarray:
    """Extract occupied voxel centres from a cuRobo ESDF VoxelGrid.

    A voxel is "occupied" when its ESDF value is ≤ 0 (inside or on the
    surface of an obstacle).

    Args:
        voxel_grid: cuRobo VoxelGrid, or any duck-typed object with
                    ``esdf_tensor`` (X, Y, Z) CUDA tensor,
                    ``origin`` (3,) tensor, and ``voxel_size`` scalar.

    Returns:
        (M, 3) float32 numpy array of world-frame XYZ voxel centres.
        Returns shape (0, 3) on failure or when the grid is entirely free.
    """
    try:
        esdf: torch.Tensor = voxel_grid.esdf_tensor   # (X, Y, Z)
        occupied = esdf <= 0.0
        if not occupied.any():
            return np.zeros((0, 3), dtype=np.float32)

        origin  = voxel_grid.origin.cpu().numpy()     # (3,)
        vsize   = float(voxel_grid.voxel_size)
        indices = torch.argwhere(occupied).float().cpu().numpy()  # (M, 3)
        centres = origin + indices * vsize + vsize / 2.0
        return centres.astype(np.float32)
    except Exception:
        return np.zeros((0, 3), dtype=np.float32)


def resolve_urdf(urdf_path: str) -> Path:
    """Rewrite ``package://`` URIs to absolute paths; return a temp file path.

    viser's URDF loader does not handle ROS2 package URIs. This rewrites
    ``package://ur_description`` to the absolute ROS2 share directory and
    writes the result to a temp file that persists for the session.

    Args:
        urdf_path: Absolute path to the URDF file (may contain package:// URIs).

    Returns:
        Path to a temp URDF file with all URIs resolved.
    """
    pkg_root = '/opt/ros/humble/share'
    with open(urdf_path) as f:
        content = f.read()
    content = content.replace(
        'package://ur_description', f'{pkg_root}/ur_description')
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False)
    tmp.write(content)
    tmp.flush()
    return Path(tmp.name)
```

- [ ] **Step 4: Run all tests in the file**

```bash
docker compose run --rm ai_planner \
  python3 -m pytest src/pipeline_orchestrator/test/test_live_viz_helpers.py -v
```
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add src/pipeline_orchestrator/pipeline_orchestrator/live_viz_helpers.py \
        src/pipeline_orchestrator/test/test_live_viz_helpers.py
git commit -m "feat: add esdf_to_points and resolve_urdf helpers with tests"
```

---

### Task 3: Scaffold test_live_viz.py with constants and SharedState

**Files:**
- Create: `src/pipeline_orchestrator/scripts/test_live_viz.py`
- Modify: `src/pipeline_orchestrator/test/test_live_viz_helpers.py`

- [ ] **Step 1: Write a failing import-level smoke test**

Append to `src/pipeline_orchestrator/test/test_live_viz_helpers.py`:

```python
def test_live_viz_script_constants(monkeypatch):
    """test_live_viz.py must be importable and expose required constants."""
    from pathlib import Path
    import importlib.util
    import sys
    import unittest.mock as mock

    # Stub every heavy dep so we don't need a live ROS2/CUDA context
    stubs = [
        'rclpy', 'rclpy.node', 'rclpy.duration', 'rclpy.time',
        'sensor_msgs', 'sensor_msgs.msg',
        'tf2_ros', 'cv_bridge', 'viser', 'viser.extras', 'warp',
        'curobo', 'curobo.perception', 'curobo.motion_planner',
        'curobo.types', 'curobo._src', 'curobo._src.geom',
        'curobo._src.geom.types',
    ]
    for name in stubs:
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, mock.MagicMock())

    script_path = Path(__file__).parent.parent / 'scripts' / 'test_live_viz.py'
    spec = importlib.util.spec_from_file_location('test_live_viz', script_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert mod.MIN_FRAMES   == 5
    assert mod.REPLAN_EVERY == 10
    assert mod.VIZ_HZ       == 10
    assert mod.GOAL_XYZ     == (0.3, 0.0, 0.4)
    assert mod.GOAL_QUAT    == (1.0, 0.0, 0.0, 0.0)
    assert mod.WORLD_FRAME  == 'world'
```

- [ ] **Step 2: Run smoke test to confirm it fails (script missing)**

```bash
docker compose run --rm ai_planner \
  python3 -m pytest \
    src/pipeline_orchestrator/test/test_live_viz_helpers.py::test_live_viz_script_constants \
    -v 2>&1 | head -20
```
Expected: `FileNotFoundError` — script doesn't exist yet.

- [ ] **Step 3: Create test_live_viz.py scaffold**

Create `src/pipeline_orchestrator/scripts/test_live_viz.py`:

```python
#!/usr/bin/env python3
"""
Live pipeline visualization test: real ROS2 depth → cuRobo Mapper → MotionPlanner → viser

Run inside the container:
  docker compose run --rm -p 8080:8080 ai_planner \\
    python3 /ros2_ws/src/pipeline_orchestrator/scripts/test_live_viz.py

Then open http://localhost:8080 in your browser.
SSH users — forward the port first:
  ssh -L 8080:localhost:8080 user@host
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener
import rclpy.duration
from rclpy.time import Time
from cv_bridge import CvBridge
import viser

from curobo.perception import FilterDepth, Mapper, MapperCfg
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import CameraObservation, Pose
from curobo.types import JointState as CuRoboJointState, GoalToolPose
from curobo._src.geom.types import SceneCfg

from pipeline_orchestrator.live_viz_helpers import (
    depth_to_xyz,
    esdf_to_points,
    resolve_urdf,
)

# ── constants ─────────────────────────────────────────────────────────────────
UR5_CONFIG   = '/ros2_ws/src/pipeline_orchestrator/config/ur5_curobo.yml'
URDF_PATH    = '/ur5.urdf'
JOINT_NAMES  = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
]
HOME_CFG     = [0.0, -2.2, 1.9, -1.383, -1.57, 0.0]
GOAL_XYZ     = (0.3, 0.0, 0.4)
GOAL_QUAT    = (1.0, 0.0, 0.0, 0.0)    # w x y z
MIN_FRAMES   = 5
REPLAN_EVERY = 10
VIZ_HZ       = 10

OVERHEAD_DEPTH_TOPIC = '/camera/camera/depth/color/image_raw'
OVERHEAD_INFO_TOPIC  = '/camera/camera/depth/camera_info'
WRIST_DEPTH_TOPIC    = '/wrist_camera/wrist_camera/depth/color/image_raw'
WRIST_INFO_TOPIC     = '/wrist_camera/wrist_camera/depth/camera_info'
OVERHEAD_FRAME       = 'camera_color_optical_frame'
WRIST_FRAME          = 'wrist_camera_color_optical_frame'
WORLD_FRAME          = 'world'


# ── shared state ──────────────────────────────────────────────────────────────
@dataclass
class SharedState:
    """Thread-safe container for data shared between ROS2 callbacks and viser."""
    lock:          threading.Lock                    = field(default_factory=threading.Lock)
    point_clouds:  Dict[str, Optional[torch.Tensor]] = field(default_factory=dict)
    voxel_grid:    Optional[object]                  = None   # cuRobo VoxelGrid
    traj:          Optional[np.ndarray]              = None   # (T, J) float32
    frame_count:   int                               = 0
    latest_joints: Optional[object]                  = None   # sensor_msgs/JointState
```

- [ ] **Step 4: Run smoke test**

```bash
docker compose run --rm ai_planner \
  python3 -m pytest \
    src/pipeline_orchestrator/test/test_live_viz_helpers.py::test_live_viz_script_constants \
    -v
```
Expected: `1 passed`

- [ ] **Step 5: Run full test suite**

```bash
docker compose run --rm ai_planner \
  python3 -m pytest src/pipeline_orchestrator/test/test_live_viz_helpers.py -v
```
Expected: `7 passed`

- [ ] **Step 6: Commit**

```bash
git add src/pipeline_orchestrator/scripts/test_live_viz.py \
        src/pipeline_orchestrator/test/test_live_viz_helpers.py
git commit -m "feat: scaffold test_live_viz.py with constants and SharedState"
```

---

### Task 4: Implement LiveVizNode in test_live_viz.py

**Files:**
- Modify: `src/pipeline_orchestrator/scripts/test_live_viz.py`

- [ ] **Step 1: Append LiveVizNode class to test_live_viz.py**

Add after the `SharedState` dataclass (before `if __name__ == '__main__':`):

```python

# ── ROS2 node ─────────────────────────────────────────────────────────────────
class LiveVizNode(Node):
    """ROS2 node: integrates live depth into cuRobo Mapper and re-plans."""

    def __init__(self, state: SharedState, planner: MotionPlanner) -> None:
        super().__init__('live_viz_node')
        self._state   = state
        self._planner = planner
        self._bridge  = CvBridge()

        # Per-camera data — only touched in ROS2 callbacks; no lock needed
        self._cam_intrinsics: Dict[str, torch.Tensor] = {}
        self._cam_depth:      Dict[str, torch.Tensor] = {}
        self._cam_pose:       Dict[str, Pose]         = {}

        self._tf_buffer   = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._mapper = Mapper(MapperCfg(
            extent_meters_xyz=(2.0, 2.0, 1.5),
            voxel_size=0.02,
            esdf_voxel_size=0.05,
            truncation_distance=0.1,
            depth_minimum_distance=0.15,
            depth_maximum_distance=2.0,
            decay_factor=1.0,
            frustum_decay_factor=1.0,
            enable_static=False,
            num_cameras=2,
        ))
        self._depth_filter = FilterDepth(
            image_shape=(480, 640),
            depth_minimum_distance=0.15,
            depth_maximum_distance=2.0,
            flying_pixel_threshold=0.5,
            bilateral_kernel_size=3,
        )

        self.create_subscription(
            CameraInfo, OVERHEAD_INFO_TOPIC,
            lambda m: self._on_info(m, 'overhead'), 1)
        self.create_subscription(
            Image, OVERHEAD_DEPTH_TOPIC,
            lambda m: self._on_depth(m, 'overhead', OVERHEAD_FRAME), 10)
        self.create_subscription(
            CameraInfo, WRIST_INFO_TOPIC,
            lambda m: self._on_info(m, 'wrist'), 1)
        self.create_subscription(
            Image, WRIST_DEPTH_TOPIC,
            lambda m: self._on_depth(m, 'wrist', WRIST_FRAME), 10)
        self.create_subscription(
            JointState, '/joint_states', self._on_joints, 10)

        self.get_logger().info('LiveVizNode ready — waiting for depth frames.')

    # ── callbacks ─────────────────────────────────────────────────────────────

    def _on_joints(self, msg: JointState) -> None:
        with self._state.lock:
            self._state.latest_joints = msg

    def _on_info(self, msg: CameraInfo, cam_id: str) -> None:
        K = torch.tensor([
            [msg.k[0], 0.0,      msg.k[2]],
            [0.0,      msg.k[4], msg.k[5]],
            [0.0,      0.0,      1.0     ],
        ], dtype=torch.float32, device='cuda')
        self._cam_intrinsics[cam_id] = K

    def _on_depth(self, msg: Image, cam_id: str, frame: str) -> None:
        if cam_id not in self._cam_intrinsics:
            return   # wait for CameraInfo first

        K = self._cam_intrinsics[cam_id]

        # TF lookup: world ← camera_optical_frame at message timestamp
        try:
            tf_time   = Time(seconds=msg.header.stamp.sec,
                             nanoseconds=msg.header.stamp.nanosec)
            transform = self._tf_buffer.lookup_transform(
                WORLD_FRAME, frame, tf_time,
                timeout=rclpy.duration.Duration(seconds=0.1))
        except Exception as exc:
            self.get_logger().warning(
                f'TF lookup failed for {frame}: {exc}',
                throttle_duration_sec=2.0)
            return

        # Decode depth: uint16 mm → float32 m
        cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        depth  = torch.from_numpy(cv_img.astype(np.float32) / 1000.0).cuda()
        depth  = torch.nan_to_num(depth, nan=0.0)
        filtered, _ = self._depth_filter(depth.unsqueeze(0))
        depth  = filtered[0]

        # Camera pose in world frame (w, x, y, z quaternion convention)
        t = transform.transform.translation
        r = transform.transform.rotation
        pose = Pose.from_numpy(
            np.array([t.x, t.y, t.z], dtype=np.float32),
            np.array([r.w, r.x, r.y, r.z], dtype=np.float32),
        )

        # Unproject depth to XYZ (camera frame) for viser display
        xyz = depth_to_xyz(depth, K)

        # Cache per-camera data
        self._cam_depth[cam_id]      = depth
        self._cam_pose[cam_id]       = pose
        self._cam_intrinsics[cam_id] = K

        with self._state.lock:
            self._state.point_clouds[cam_id] = xyz

        # Integrate when both cameras have data
        if 'overhead' not in self._cam_depth or 'wrist' not in self._cam_depth:
            return

        batched = CameraObservation(
            depth_image=torch.stack([
                self._cam_depth['overhead'],
                self._cam_depth['wrist'],
            ]),
            intrinsics=torch.stack([
                self._cam_intrinsics['overhead'],
                self._cam_intrinsics['wrist'],
            ]),
            pose=Pose(
                position=torch.cat([
                    self._cam_pose['overhead'].position,
                    self._cam_pose['wrist'].position,
                ]),
                quaternion=torch.cat([
                    self._cam_pose['overhead'].quaternion,
                    self._cam_pose['wrist'].quaternion,
                ]),
            ),
        )
        self._mapper.integrate(batched)

        with self._state.lock:
            self._state.frame_count += 1
            count = self._state.frame_count

        if count >= MIN_FRAMES and count % REPLAN_EVERY == 0:
            self._replan()

    # ── planning ──────────────────────────────────────────────────────────────

    def _replan(self) -> None:
        """Compute a fresh ESDF and plan a new trajectory to GOAL_XYZ."""
        voxel_grid = self._mapper.compute_esdf()
        self._planner.update_world(SceneCfg(voxel=[voxel_grid]))

        with self._state.lock:
            self._state.voxel_grid = voxel_grid
            js = self._state.latest_joints

        # Start configuration: live joints or home pose fallback
        if js is not None:
            start = CuRoboJointState.from_position(
                torch.tensor(
                    [list(js.position)], dtype=torch.float32, device='cuda'),
                joint_names=list(js.name))
        else:
            start = CuRoboJointState.from_position(
                torch.tensor([HOME_CFG], dtype=torch.float32, device='cuda'),
                joint_names=JOINT_NAMES)

        goal = GoalToolPose(
            tool_frames=self._planner.tool_frames,
            position=torch.tensor(
                [[[[[GOAL_XYZ[0], GOAL_XYZ[1], GOAL_XYZ[2]]]]]], device='cuda',
                dtype=torch.float32),
            quaternion=torch.tensor(
                [[[[[GOAL_QUAT[0], GOAL_QUAT[1], GOAL_QUAT[2], GOAL_QUAT[3]]]]]], device='cuda',
                dtype=torch.float32),
        )

        result = self._planner.plan_pose(goal, start)
        if result is None or not result.success.any():
            self.get_logger().warning(
                'CuRobo planning failed — keeping previous trajectory.')
            return

        pos = result.get_interpolated_plan().position[0]
        while pos.dim() > 2:
            pos = pos[0]
        traj = pos.cpu().numpy()   # (T, J) float32
        self.get_logger().info(f'Planned {len(traj)}-waypoint trajectory.')

        with self._state.lock:
            self._state.traj = traj
```

- [ ] **Step 2: Run smoke test to confirm import still works**

```bash
docker compose run --rm ai_planner \
  python3 -m pytest \
    src/pipeline_orchestrator/test/test_live_viz_helpers.py::test_live_viz_script_constants \
    -v
```
Expected: `1 passed`

- [ ] **Step 3: Commit**

```bash
git add src/pipeline_orchestrator/scripts/test_live_viz.py
git commit -m "feat: implement LiveVizNode with depth callbacks and _replan()"
```

---

### Task 5: Implement update_loop, build_planner, and main()

**Files:**
- Modify: `src/pipeline_orchestrator/scripts/test_live_viz.py`

- [ ] **Step 1: Append update_loop, build_planner, and main to test_live_viz.py**

Add after the `LiveVizNode` class:

```python

# ── viser update loop ─────────────────────────────────────────────────────────

def update_loop(
    server: viser.ViserServer,
    state: SharedState,
    robot,   # ViserUrdf instance or None
) -> None:
    """Main-thread loop: refresh the viser scene at VIZ_HZ from shared state.

    Reads lock-protected SharedState and updates:
      - /depth/overhead, /depth/wrist   — live point clouds
      - /esdf/voxels                    — occupied ESDF voxels
      - /ur5 robot joints               — animated trajectory
      - /status label                   — frame count and status text

    Loops forever; raise KeyboardInterrupt to exit.
    """
    home_array = np.array([HOME_CFG])
    traj_idx   = 0
    period     = 1.0 / VIZ_HZ

    while True:
        t0 = time.time()

        with state.lock:
            clouds   = dict(state.point_clouds)   # shallow copy of dict
            vg       = state.voxel_grid
            traj     = state.traj if state.traj is not None else home_array
            n_frames = state.frame_count

        # ── status label ──────────────────────────────────────────────────────
        if n_frames < MIN_FRAMES:
            status_text = f'Waiting for depth frames ({n_frames}/{MIN_FRAMES})…'
        else:
            status_text = f'Frames: {n_frames}  |  Traj waypoints: {len(traj)}'
        server.scene.add_label('/status', status_text, position=(0.0, 0.0, 1.6))

        # ── point clouds ──────────────────────────────────────────────────────
        for cam_id, xyz in clouds.items():
            if xyz is None or len(xyz) == 0:
                continue
            pts   = xyz.cpu().numpy()
            color = (200, 200, 200) if cam_id == 'overhead' else (100, 150, 255)
            server.scene.add_point_cloud(
                f'/depth/{cam_id}',
                points=pts,
                colors=np.tile(color, (len(pts), 1)).astype(np.uint8),
                point_size=0.005,
            )

        # ── ESDF voxels ───────────────────────────────────────────────────────
        if vg is not None:
            occ_pts = esdf_to_points(vg)
            if len(occ_pts) > 0:
                server.scene.add_point_cloud(
                    '/esdf/voxels',
                    points=occ_pts,
                    colors=np.tile((220, 60, 60), (len(occ_pts), 1)).astype(np.uint8),
                    point_size=0.02,
                )

        # ── robot animation ───────────────────────────────────────────────────
        if len(traj) > 0:
            waypoint = traj[traj_idx % len(traj)]
            if robot is not None:
                robot.update_cfg(dict(zip(JOINT_NAMES, waypoint.tolist())))
            traj_idx += 1
            if traj_idx >= len(traj):
                traj_idx = 0
                time.sleep(1.0)   # brief pause before replaying

        elapsed = time.time() - t0
        time.sleep(max(0.0, period - elapsed))


# ── motion planner setup ──────────────────────────────────────────────────────

def build_planner() -> MotionPlanner:
    """Construct and warm up the cuRobo MotionPlanner (~30 s on first run)."""
    print('  Loading MotionPlanner (warmup ~30 s)…')
    planner = MotionPlanner(MotionPlannerCfg.create(
        robot=UR5_CONFIG,
        scene_model='collision_test.yml',
    ))
    planner.warmup(enable_graph=True, num_warmup_iterations=3)
    print('  MotionPlanner ready.')
    return planner


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print('=== Live Pipeline Visualization Test ===\n')

    print('[1/4] Initializing CUDA / Warp…')
    import warp as wp
    wp.init()
    print('  Warp OK.\n')

    print('[2/4] Setting up motion planner…')
    planner = build_planner()

    print('[3/4] Starting ROS2 node…')
    rclpy.init()
    state = SharedState()
    node  = LiveVizNode(state, planner)
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    print('  ROS2 node spinning in background.\n')

    print('[4/4] Starting viser…')
    server = viser.ViserServer(port=8080, verbose=False)
    print('  Viser running — open http://localhost:8080\n')

    server.scene.add_frame('/world', axes_length=0.3, axes_radius=0.01)
    server.scene.add_icosphere(
        '/target', radius=0.03, color=(255, 80, 80), position=GOAL_XYZ)

    try:
        from viser.extras import ViserUrdf
        robot = ViserUrdf(
            server,
            urdf_or_path=resolve_urdf(URDF_PATH),
            root_node_name='/ur5',
        )
        print('  Robot model loaded.')
    except Exception as exc:
        print(f'  Robot model unavailable ({exc}), skipping URDF.')
        robot = None

    print('Entering update loop. Ctrl+C to stop.')
    try:
        update_loop(server, state, robot)
    except KeyboardInterrupt:
        print('\nStopped.')
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run all helper tests**

```bash
docker compose run --rm ai_planner \
  python3 -m pytest src/pipeline_orchestrator/test/test_live_viz_helpers.py -v
```
Expected: `7 passed`

- [ ] **Step 3: Verify syntax**

```bash
docker compose run --rm ai_planner python3 -c "
import ast
with open('/ros2_ws/src/pipeline_orchestrator/scripts/test_live_viz.py') as f:
    src = f.read()
ast.parse(src)
print('Syntax OK')
"
```
Expected: `Syntax OK`

- [ ] **Step 4: Final commit**

```bash
git add src/pipeline_orchestrator/scripts/test_live_viz.py
git commit -m "feat: add live viz test script with real ROS2 depth, ESDF, and viser"
```

---

## Running the Script

With the ROS2 simulation running on the host (Gazebo + manip_challenge):

```bash
docker compose run --rm -p 8080:8080 ai_planner \
  python3 /ros2_ws/src/pipeline_orchestrator/scripts/test_live_viz.py
```

Open `http://localhost:8080`. Expected progression:

| Phase | What you see |
|---|---|
| Startup (~30 s) | Warp init + MotionPlanner warmup printed to terminal |
| First 1–4 frames | Status: "Waiting for depth frames (N/5)…" |
| Frame 5+ | Point clouds appear; first plan triggered at frame 10 |
| Every 10 frames | Trajectory re-planned with latest ESDF |
| Continuously | Live grey (overhead) + blue (wrist) point clouds, red ESDF voxels, animated robot |
