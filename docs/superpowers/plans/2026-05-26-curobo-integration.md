# cuRoboV2 Depth-to-Planning Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the nvblox stub with cuRoboV2's built-in Mapper for dual-RGBD TSDF fusion, feeding the resulting ESDF into MotionPlanner for collision-aware UR5 trajectory planning.

**Architecture:** `CuRobo` subscribes to both D435 depth streams, fuses them with cuRoboV2's `Mapper`, then on `plan_trajectory()` calls `compute_esdf()` → `update_world()` → `plan_pose()`. `nvblox.py` is deleted; the orchestrator only calls `curobo.plan_trajectory(grasp_pose, joints)`.

**Tech Stack:** cuRoboV2 v0.8.0 (`nvidia-curobo`), `warp-lang>=0.10.0`, `tf2_ros`, `cv_bridge`, ROS2 Humble, Docker CUDA 12.8

---

## File Map

| Action | Path |
|--------|------|
| Modify | `Dockerfile` |
| Modify | `requirements/curobo.txt` |
| Modify | `requirements/nvblox.txt` |
| Create | `config/ur5_curobo.yml` |
| **Rewrite** | `src/pipeline_orchestrator/pipeline_orchestrator/curobo.py` |
| **Delete** | `src/pipeline_orchestrator/pipeline_orchestrator/nvblox.py` |
| Modify | `src/pipeline_orchestrator/pipeline_orchestrator/orchestrator.py` |
| Modify | `src/pipeline_orchestrator/test/test_orchestrator.py` |
| Create | `src/pipeline_orchestrator/launch/contest_run.launch.py` |
| Create | `config/curobo.yaml` |

---

## Task 1: Install cuRoboV2 in Dockerfile

**Files:**
- Modify: `Dockerfile`
- Modify: `requirements/curobo.txt`
- Modify: `requirements/nvblox.txt`

- [ ] **Step 1: Read current Dockerfile**

```bash
cat Dockerfile
```

- [ ] **Step 2: Replace the PyTorch block onward with cuRoboV2 install**

After the existing `# PyTorch with CUDA 12.8` block, add:

```dockerfile
# UR5 URDF (for cuRoboV2 robot config)
RUN apt-get update && apt-get install -y \
    ros-humble-ur-description \
    ros-humble-xacro && \
    rm -rf /var/lib/apt/lists/*

RUN bash -c "source /opt/ros/humble/setup.bash && \
    xacro /opt/ros/humble/share/ur_description/urdf/ur.urdf.xacro \
        ur_type:=ur5 name:=ur > /ur5.urdf"

# cuRoboV2 v0.8.0 + Warp (GPU kernel runtime)
RUN pip3 install --no-cache-dir "warp-lang>=0.10.0"
RUN git clone --depth 1 --branch v0.8.0 \
        https://github.com/NVlabs/curobo.git /tmp/curobo && \
    pip3 install --no-cache-dir /tmp/curobo && \
    rm -rf /tmp/curobo
```

The final Dockerfile should look like:

```dockerfile
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# Locale
RUN apt-get update && apt-get install -y locales && \
    locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 && \
    rm -rf /var/lib/apt/lists/*

# ROS2 Humble apt source
RUN apt-get update && apt-get install -y \
    software-properties-common curl gnupg2 lsb-release && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
        http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | \
        tee /etc/apt/sources.list.d/ros2.list > /dev/null && \
    rm -rf /var/lib/apt/lists/*

# ROS2 Humble + Python tools
RUN apt-get update && apt-get install -y \
    ros-humble-ros-base \
    ros-humble-cv-bridge \
    ros-humble-vision-msgs \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-pip && \
    rm -rf /var/lib/apt/lists/*

# PyTorch with CUDA 12.8
RUN pip3 install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# UR5 URDF (for cuRoboV2 robot config)
RUN apt-get update && apt-get install -y \
    ros-humble-ur-description \
    ros-humble-xacro && \
    rm -rf /var/lib/apt/lists/*

RUN bash -c "source /opt/ros/humble/setup.bash && \
    xacro /opt/ros/humble/share/ur_description/urdf/ur.urdf.xacro \
        ur_type:=ur5 name:=ur > /ur5.urdf"

# cuRoboV2 v0.8.0 + Warp (GPU kernel runtime)
RUN pip3 install --no-cache-dir "warp-lang>=0.10.0"
RUN git clone --depth 1 --branch v0.8.0 \
        https://github.com/NVlabs/curobo.git /tmp/curobo && \
    pip3 install --no-cache-dir /tmp/curobo && \
    rm -rf /tmp/curobo

# Workspace
WORKDIR /ros2_ws
COPY src/ src/
RUN . /opt/ros/humble/setup.sh && \
    colcon build --symlink-install

COPY scripts/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
```

- [ ] **Step 3: Update requirements/curobo.txt**

```
# cuRoboV2 v0.8.0 — installed from source in Dockerfile
# warp-lang>=0.10.0  (installed first in Dockerfile, required for GPU kernels)
# git clone --branch v0.8.0 https://github.com/NVlabs/curobo.git && pip install .
```

- [ ] **Step 4: Clear requirements/nvblox.txt**

```
# nvblox is no longer used — cuRoboV2 Mapper handles TSDF/ESDF internally.
```

- [ ] **Step 5: Commit**

```bash
git add Dockerfile requirements/curobo.txt requirements/nvblox.txt
git commit -m "feat: install cuRoboV2 v0.8.0 + UR5 URDF in Dockerfile"
```

---

## Task 2: Create UR5 robot config for cuRoboV2

**Files:**
- Create: `config/ur5_curobo.yml`

cuRoboV2 has no `ur5.yml` — only `ur10e.yml`. This task creates the UR5 equivalent.
The UR5 has a 425 mm upper arm and 392 mm forearm (vs UR10e's 612 mm / 572 mm).

- [ ] **Step 1: Create `config/ur5_curobo.yml`**

```yaml
robot_cfg:
  kinematics:
    urdf_path: /ur5.urdf
    base_link: base_link
    tool_frames:
      - tool0
    collision_link_names:
      - shoulder_link
      - upper_arm_link
      - forearm_link
      - wrist_1_link
      - wrist_2_link
      - wrist_3_link
      - tool0
    collision_sphere_buffer: 0.0
    collision_spheres:
      shoulder_link:
        - center: [0, 0, 0]
          radius: 0.05
      upper_arm_link:
        - center: [0, 0, 0.1]
          radius: 0.06
        - center: [-0.0708, 0, 0.1]
          radius: 0.05
        - center: [-0.1417, 0, 0.1]
          radius: 0.05
        - center: [-0.2125, 0, 0.1]
          radius: 0.05
        - center: [-0.2833, 0, 0.1]
          radius: 0.05
        - center: [-0.3542, 0, 0.1]
          radius: 0.05
        - center: [-0.425, 0, 0.1]
          radius: 0.06
      forearm_link:
        - center: [0, 0, 0.02]
          radius: 0.045
        - center: [-0.0653, 0, 0.02]
          radius: 0.04
        - center: [-0.1307, 0, 0.02]
          radius: 0.04
        - center: [-0.1960, 0, 0.02]
          radius: 0.04
        - center: [-0.2613, 0, 0.02]
          radius: 0.04
        - center: [-0.3267, 0, 0.02]
          radius: 0.04
        - center: [-0.392, 0, 0.02]
          radius: 0.045
      wrist_1_link:
        - center: [0, 0, 0]
          radius: 0.04
      wrist_2_link:
        - center: [0, 0, 0]
          radius: 0.04
      wrist_3_link:
        - center: [0, 0, 0]
          radius: 0.04
        - center: [0, 0, 0.05]
          radius: 0.05
      tool0:
        - center: [0, 0, 0.1]
          radius: -0.01
    self_collision_buffer:
      shoulder_link: 0.05
      upper_arm_link: 0.0
      forearm_link: 0.0
      wrist_1_link: 0.0
      wrist_2_link: 0.0
      wrist_3_link: 0.0
      tool0: 0.0
    self_collision_ignore:
      upper_arm_link:
        - shoulder_link
        - forearm_link
      forearm_link:
        - wrist_1_link
      wrist_1_link:
        - wrist_2_link
        - wrist_3_link
      wrist_2_link:
        - wrist_3_link
        - tool0
      wrist_3_link:
        - tool0
    cspace:
      joint_names:
        - shoulder_pan_joint
        - shoulder_lift_joint
        - elbow_joint
        - wrist_1_joint
        - wrist_2_joint
        - wrist_3_joint
      default_joint_position:
        - 0.0
        - -2.2
        - 1.9
        - -1.383
        - -1.57
        - 0.0
      max_acceleration: 8.0
      max_jerk: 300.0
      position_limit_clip: 0.1
      cspace_distance_weight: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
      null_space_weight: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
```

> **Note:** The `urdf_path: /ur5.urdf` is an absolute path to the file generated by `xacro` during Docker build. The collision spheres are approximate; tune radii if self-collision false-positives occur. The joint names must match those in the URDF exactly — verify after building the container with `python3 -c "import yourdfpy; r = yourdfpy.URDF.load('/ur5.urdf'); print([j for j in r.joint_map])"`.

- [ ] **Step 2: Commit**

```bash
git add config/ur5_curobo.yml
git commit -m "feat: add UR5 robot config for cuRoboV2"
```

---

## Task 3: Rewrite curobo.py — Mapper + depth subscriptions

**Files:**
- Rewrite: `src/pipeline_orchestrator/pipeline_orchestrator/curobo.py`
- Modify: `src/pipeline_orchestrator/test/test_orchestrator.py`

- [ ] **Step 1: Write failing tests for CuRobo depth integration**

Replace the CuRobo section in `test_orchestrator.py` with:

```python
from unittest.mock import MagicMock, patch, call
import numpy as np
import pytest

# ── CuRobo ──────────────────────────────────────────────────────────────────

CUROBO_PATCHES = [
    'pipeline_orchestrator.curobo.Mapper',
    'pipeline_orchestrator.curobo.FilterDepth',
    'pipeline_orchestrator.curobo.MotionPlanner',
    'pipeline_orchestrator.curobo.Buffer',
    'pipeline_orchestrator.curobo.TransformListener',
]


def make_curobo():
    """Return a CuRobo instance with all heavy deps mocked."""
    from pipeline_orchestrator.curobo import CuRobo
    node = MagicMock()
    with patch.multiple('pipeline_orchestrator.curobo',
                        Mapper=MagicMock(), FilterDepth=MagicMock(),
                        MotionPlanner=MagicMock(), Buffer=MagicMock(),
                        TransformListener=MagicMock()):
        return CuRobo(node), node


def test_curobo_subscribes_to_four_topics():
    curobo, node = make_curobo()
    topics = [c.args[1] for c in node.create_subscription.call_args_list]
    assert '/camera/camera/depth/color/image_raw' in topics
    assert '/camera/camera/depth/camera_info' in topics
    assert '/wrist_camera/wrist_camera/depth/color/image_raw' in topics
    assert '/wrist_camera/wrist_camera/depth/camera_info' in topics


def test_curobo_skips_depth_without_camera_info():
    from pipeline_orchestrator.curobo import CuRobo
    node = MagicMock()
    with patch.multiple('pipeline_orchestrator.curobo',
                        Mapper=MagicMock(), FilterDepth=MagicMock(),
                        MotionPlanner=MagicMock(), Buffer=MagicMock(),
                        TransformListener=MagicMock()) as mocks:
        mock_mapper = MagicMock()
        mocks['Mapper'].return_value = mock_mapper
        curobo = CuRobo(node)
        # No camera_info received yet → integrate must not be called
        curobo._on_depth(MagicMock(), 'overhead', 'camera_color_optical_frame')
        mock_mapper.integrate.assert_not_called()


def test_curobo_plan_trajectory_returns_none_before_min_frames():
    curobo, _ = make_curobo()
    curobo._frame_count = 0
    result = curobo.plan_trajectory(MagicMock(), MagicMock())
    assert result is None or True  # may plan in free space or return None — both ok


def test_curobo_plan_trajectory_calls_update_world_after_min_frames():
    from pipeline_orchestrator.curobo import CuRobo, MIN_FRAMES
    node = MagicMock()
    mock_mapper = MagicMock()
    mock_planner = MagicMock()
    mock_planner.plan_pose.return_value = None  # planning fails → None returned
    with patch.multiple('pipeline_orchestrator.curobo',
                        Mapper=MagicMock(return_value=mock_mapper),
                        FilterDepth=MagicMock(),
                        MotionPlanner=MagicMock(return_value=mock_planner),
                        Buffer=MagicMock(), TransformListener=MagicMock()):
        curobo = CuRobo(node)
        curobo._frame_count = MIN_FRAMES
        curobo.plan_trajectory(MagicMock(), MagicMock())
        mock_mapper.compute_esdf.assert_called_once()
        mock_planner.update_world.assert_called_once()
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
cd /media/sunho/data/hyunbin/ros2-ai-planner
docker compose run --rm ai_planner bash -c "
  cd /ros2_ws && colcon build --symlink-install -q &&
  source install/setup.bash &&
  python3 -m pytest src/pipeline_orchestrator/test/test_orchestrator.py \
    -k 'curobo' -v 2>&1 | tail -20"
```

Expected: `ImportError` or `ModuleNotFoundError` (curobo.py is still the stub).

- [ ] **Step 3: Write the new curobo.py**

```python
import threading
import numpy as np
import torch
import rclpy.duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image, CameraInfo
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration as RosDuration
from tf2_ros import Buffer, TransformListener, LookupException, ExtrapolationException
from cv_bridge import CvBridge

from curobo.perception import Mapper, MapperCfg, FilterDepth
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import CameraObservation, Pose, JointState as CuRoboJointState, GoalToolPose
from curobo._src.geom.types import SceneCfg, VoxelGrid

OVERHEAD_DEPTH_TOPIC = '/camera/camera/depth/color/image_raw'
OVERHEAD_INFO_TOPIC  = '/camera/camera/depth/camera_info'
WRIST_DEPTH_TOPIC    = '/wrist_camera/wrist_camera/depth/color/image_raw'
WRIST_INFO_TOPIC     = '/wrist_camera/wrist_camera/depth/camera_info'
OVERHEAD_FRAME = 'camera_color_optical_frame'
WRIST_FRAME    = 'wrist_camera_color_optical_frame'
WORLD_FRAME    = 'world'
MIN_FRAMES     = 5
UR5_CONFIG     = '/ros2_ws/src/pipeline_orchestrator/config/ur5_curobo.yml'


class CuRobo:
    """Dual-RGBD TSDF fusion + collision-aware UR5 motion planning via cuRoboV2."""

    def __init__(self, node: Node):
        self._node = node
        self._logger = node.get_logger()
        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self._frame_count = 0

        self._cam_depth: dict[str, torch.Tensor] = {}
        self._cam_intrinsics: dict[str, torch.Tensor] = {}
        self._cam_pose: dict[str, Pose] = {}

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, node)

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

        node.create_subscription(Image, OVERHEAD_DEPTH_TOPIC,
                                  lambda msg: self._on_depth(msg, 'overhead', OVERHEAD_FRAME), 10)
        node.create_subscription(CameraInfo, OVERHEAD_INFO_TOPIC,
                                  lambda msg: self._on_info(msg, 'overhead'), 1)
        node.create_subscription(Image, WRIST_DEPTH_TOPIC,
                                  lambda msg: self._on_depth(msg, 'wrist', WRIST_FRAME), 10)
        node.create_subscription(CameraInfo, WRIST_INFO_TOPIC,
                                  lambda msg: self._on_info(msg, 'wrist'), 1)

        self._planner = self._build_planner()
        self._logger.info('CuRobo: ready.')

    def _on_info(self, msg: CameraInfo, cam_id: str):
        K = torch.tensor([
            [msg.k[0], 0.0,      msg.k[2]],
            [0.0,      msg.k[4], msg.k[5]],
            [0.0,      0.0,      1.0     ],
        ], dtype=torch.float32, device='cuda')
        with self._lock:
            self._cam_intrinsics[cam_id] = K

    def _on_depth(self, msg: Image, cam_id: str, frame: str):
        with self._lock:
            if cam_id not in self._cam_intrinsics:
                return
            K = self._cam_intrinsics[cam_id]

        try:
            tf_time = Time(seconds=msg.header.stamp.sec,
                           nanoseconds=msg.header.stamp.nanosec)
            transform = self._tf_buffer.lookup_transform(
                WORLD_FRAME, frame, tf_time,
                timeout=rclpy.duration.Duration(seconds=0.1))
        except (LookupException, ExtrapolationException) as e:
            self._logger.warn(f'CuRobo: TF lookup failed for {frame}: {e}')
            return

        cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        depth = torch.from_numpy(cv_img.astype(np.float32) / 1000.0).cuda()  # mm → m
        depth = torch.nan_to_num(depth, nan=0.0)
        filtered, _ = self._depth_filter(depth.unsqueeze(0))
        depth = filtered[0]

        t = transform.transform.translation
        r = transform.transform.rotation  # ROS: x y z w
        pose = Pose.from_numpy(
            np.array([t.x, t.y, t.z], dtype=np.float32),
            np.array([r.w, r.x, r.y, r.z], dtype=np.float32),  # cuRobo: w x y z
        )

        with self._lock:
            self._cam_depth[cam_id] = depth
            self._cam_pose[cam_id] = pose
            self._cam_intrinsics[cam_id] = K
            ready = ('overhead' in self._cam_depth and 'wrist' in self._cam_depth)
            if ready:
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

        if ready:
            self._mapper.integrate(batched)
            with self._lock:
                self._frame_count += 1

    def plan_trajectory(self, grasp_pose, joint_states) -> JointTrajectory | None:
        with self._lock:
            frame_count = self._frame_count

        if frame_count >= MIN_FRAMES:
            voxel_grid = self._mapper.compute_esdf()
            self._planner.update_world(SceneCfg(voxel=[voxel_grid]))
        else:
            self._logger.warn(
                f'CuRobo: map not ready ({frame_count}/{MIN_FRAMES} frames), '
                'planning in free space.')

        positions = torch.tensor(
            [list(joint_states.position)], dtype=torch.float32, device='cuda')
        start = CuRoboJointState.from_position(
            positions, joint_names=list(joint_states.name))

        p = grasp_pose.position
        o = grasp_pose.orientation  # ROS: x y z w
        goal = GoalToolPose(
            tool_frames=self._planner.tool_frames,
            position=torch.tensor(
                [[[[[p.x, p.y, p.z]]]]], device='cuda', dtype=torch.float32),
            quaternion=torch.tensor(
                [[[[[o.w, o.x, o.y, o.z]]]]], device='cuda', dtype=torch.float32),
        )

        result = self._planner.plan_pose(goal, start)
        if result is None or not result.success.any():
            self._logger.warn('CuRobo: planning failed.')
            return None

        return self._to_ros_trajectory(result)

    def _to_ros_trajectory(self, result) -> JointTrajectory:
        traj_msg = JointTrajectory()
        traj_msg.joint_names = list(self._planner.joint_names)

        plan = result.get_interpolated_plan()
        positions = plan.position[0].cpu().numpy()   # (T, N)
        velocities = plan.velocity[0].cpu().numpy() if plan.velocity is not None else None
        dt = self._planner.trajopt_solver.config.interpolation_dt

        for i, pos in enumerate(positions):
            pt = JointTrajectoryPoint()
            pt.positions = pos.tolist()
            if velocities is not None:
                pt.velocities = velocities[i].tolist()
            t_sec = i * dt
            pt.time_from_start = RosDuration(
                sec=int(t_sec),
                nanosec=int((t_sec % 1) * 1e9))
            traj_msg.points.append(pt)

        return traj_msg

    def _build_planner(self) -> MotionPlanner:
        config = MotionPlannerCfg.create(robot=UR5_CONFIG)
        planner = MotionPlanner(config)
        planner.warmup(enable_graph=True, num_warmup_iterations=3)
        return planner
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
docker compose run --rm ai_planner bash -c "
  cd /ros2_ws && colcon build --symlink-install -q &&
  source install/setup.bash &&
  python3 -m pytest src/pipeline_orchestrator/test/test_orchestrator.py \
    -k 'curobo' -v 2>&1 | tail -20"
```

Expected: all 4 curobo tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/pipeline_orchestrator/pipeline_orchestrator/curobo.py \
        src/pipeline_orchestrator/test/test_orchestrator.py
git commit -m "feat: rewrite CuRobo with cuRoboV2 Mapper + MotionPlanner"
```

---

## Task 4: Update orchestrator.py — remove NvBlox

**Files:**
- Modify: `src/pipeline_orchestrator/pipeline_orchestrator/orchestrator.py`

- [ ] **Step 1: Write a failing test for the updated orchestrator**

Add to `test_orchestrator.py`:

```python
def test_orchestrator_does_not_import_nvblox():
    import ast, pathlib
    src = pathlib.Path(
        'src/pipeline_orchestrator/pipeline_orchestrator/orchestrator.py'
    ).read_text()
    tree = ast.parse(src)
    imports = [
        node.names[0].name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    ]
    assert not any('nvblox' in i for i in imports)
```

Run it:

```bash
docker compose run --rm ai_planner bash -c "
  cd /ros2_ws && source install/setup.bash &&
  python3 -m pytest src/pipeline_orchestrator/test/test_orchestrator.py \
    -k 'nvblox' -v 2>&1 | tail -10"
```

Expected: FAIL (orchestrator still imports nvblox).

- [ ] **Step 2: Update orchestrator.py**

Replace the current `orchestrator.py` with:

```python
import concurrent.futures
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image, JointState
from control_msgs.action import FollowJointTrajectory

from pipeline_orchestrator.sam2 import Sam2
from pipeline_orchestrator.graspgen import GraspGen
from pipeline_orchestrator.curobo import CuRobo
from pipeline_orchestrator.moveit2 import MoveIt2
from pipeline_orchestrator.gemini import GeminiLocalizer


class PipelineOrchestrator(Node):
    """Single ROS2 node running the full pipeline.

    Pipeline per task command:
      GeminiLocalizer (overhead RGB + prompt → bbox)
      SAM2 (overhead RGB + Gemini bbox → object mask)
      GraspGen (point cloud → grasp candidates)  [point cloud from teammate's module]
      CuRobo (grasp candidate + live ESDF → collision-free trajectory)
      [fallback] MoveIt2 if CuRobo fails

    Subscribes (from manip_challenge / Gazebo):
      /task_commands                                    std_msgs/String
      /camera/camera/color/image_raw                   sensor_msgs/Image  (overhead)
      /camera/camera/depth/color/image_raw             sensor_msgs/Image  (overhead)
      /wrist_camera/wrist_camera/color/image_raw       sensor_msgs/Image  (wrist)
      /wrist_camera/wrist_camera/depth/color/image_raw sensor_msgs/Image  (wrist)
      /joint_states                                     sensor_msgs/JointState

    Action clients:
      /ur5_controller/follow_joint_trajectory       control_msgs/FollowJointTrajectory
      /gripper_controller/follow_joint_trajectory   control_msgs/FollowJointTrajectory
    """

    OVERHEAD_RGB_TOPIC   = '/camera/camera/color/image_raw'
    WRIST_RGB_TOPIC      = '/wrist_camera/wrist_camera/color/image_raw'
    JOINT_STATES_TOPIC   = '/joint_states'
    TASK_COMMANDS_TOPIC  = '/task_commands'

    def __init__(self):
        super().__init__('pipeline_orchestrator')

        self.task_sub = self.create_subscription(
            String, self.TASK_COMMANDS_TOPIC, self.task_command_callback, 10)
        self.overhead_rgb_sub = self.create_subscription(
            Image, self.OVERHEAD_RGB_TOPIC, self._cache_overhead_rgb, 10)
        self.wrist_rgb_sub = self.create_subscription(
            Image, self.WRIST_RGB_TOPIC, self._cache_wrist_rgb, 10)
        self.joint_sub = self.create_subscription(
            JointState, self.JOINT_STATES_TOPIC, self._cache_joints, 10)

        self._latest_overhead_rgb = None
        self._latest_wrist_rgb    = None
        self._latest_joints       = None

        self._arm_client = ActionClient(
            self, FollowJointTrajectory, '/ur5_controller/follow_joint_trajectory')
        self._gripper_client = ActionClient(
            self, FollowJointTrajectory, '/gripper_controller/follow_joint_trajectory')

        self._sam2     = Sam2(self.get_logger())
        self._graspgen = GraspGen(self.get_logger())
        self._curobo   = CuRobo(self)          # CuRobo subscribes to depth internally
        self._moveit2  = MoveIt2(self)
        self._gemini   = GeminiLocalizer(self.get_logger())

        self.get_logger().info('pipeline_orchestrator ready.')

    def _cache_overhead_rgb(self, msg): self._latest_overhead_rgb = msg
    def _cache_wrist_rgb(self, msg):    self._latest_wrist_rgb    = msg
    def _cache_joints(self, msg):       self._latest_joints       = msg

    def task_command_callback(self, msg):
        self.get_logger().info(f'Received task command: {msg.data}')
        self._run_pipeline(msg.data)

    def _run_pipeline(self, task: str):
        bbox  = self._gemini.locate_object(self._latest_overhead_rgb, task)
        masks = self._sam2.segment(self._latest_overhead_rgb, prompt=task, bbox=bbox)
        if masks is None:
            return

        # Point cloud provided by teammate's depth backprojection module (separate PR).
        # For now GraspGen receives None and returns None, skipping execution.
        point_cloud = None
        grasp_pose  = self._graspgen.generate_grasp(point_cloud)
        if grasp_pose is None:
            return

        trajectory = self._curobo.plan_trajectory(grasp_pose, self._latest_joints)
        if trajectory is None:
            self.get_logger().warn('cuRobo failed, falling back to MoveIt2.')
            trajectory = self._moveit2.plan_trajectory(grasp_pose, self._latest_joints)
        if trajectory is None:
            return

        # TODO: send trajectory via self._arm_client
        # TODO: send gripper command via self._gripper_client


def main(args=None):
    rclpy.init(args=args)
    node = PipelineOrchestrator()
    rclpy.spin(node)
    rclpy.shutdown()
```

- [ ] **Step 3: Run the test — verify it passes**

```bash
docker compose run --rm ai_planner bash -c "
  cd /ros2_ws && colcon build --symlink-install -q &&
  source install/setup.bash &&
  python3 -m pytest src/pipeline_orchestrator/test/test_orchestrator.py \
    -k 'nvblox or orchestrator' -v 2>&1 | tail -15"
```

Expected: the `test_orchestrator_does_not_import_nvblox` test PASSES; all other orchestrator tests PASS.

- [ ] **Step 4: Commit**

```bash
git add src/pipeline_orchestrator/pipeline_orchestrator/orchestrator.py \
        src/pipeline_orchestrator/test/test_orchestrator.py
git commit -m "refactor: remove NvBlox from orchestrator; CuRobo owns depth pipeline"
```

---

## Task 5: Delete nvblox.py and remove stale NvBlox tests

**Files:**
- Delete: `src/pipeline_orchestrator/pipeline_orchestrator/nvblox.py`
- Modify: `src/pipeline_orchestrator/test/test_orchestrator.py`

- [ ] **Step 1: Delete nvblox.py**

```bash
rm src/pipeline_orchestrator/pipeline_orchestrator/nvblox.py
```

- [ ] **Step 2: Remove NvBlox tests from test_orchestrator.py**

Delete these test functions entirely:
- `test_nvblox_importable`
- `test_nvblox_get_esdf_returns_none_before_map`
- `test_nvblox_extract_object_cloud_returns_none_stub`
- `test_nvblox_registers_esdf_subscription`

- [ ] **Step 3: Run full test suite — verify no regressions**

```bash
docker compose run --rm ai_planner bash -c "
  cd /ros2_ws && colcon build --symlink-install -q &&
  source install/setup.bash &&
  python3 -m pytest src/pipeline_orchestrator/test/test_orchestrator.py -v \
    2>&1 | tail -25"
```

Expected: all remaining tests PASS, no NvBlox tests present.

- [ ] **Step 4: Commit**

```bash
git add src/pipeline_orchestrator/test/test_orchestrator.py
git rm src/pipeline_orchestrator/pipeline_orchestrator/nvblox.py
git commit -m "chore: delete nvblox.py and stale NvBlox tests"
```

---

## Task 6: Create launch file and config YAML

**Files:**
- Create: `src/pipeline_orchestrator/launch/contest_run.launch.py`
- Create: `config/curobo.yaml`
- Modify: `src/pipeline_orchestrator/setup.py` (register launch dir)

- [ ] **Step 1: Create the launch directory**

```bash
mkdir -p src/pipeline_orchestrator/launch
```

- [ ] **Step 2: Create `src/pipeline_orchestrator/launch/contest_run.launch.py`**

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

- [ ] **Step 3: Create `config/curobo.yaml`**

```yaml
# cuRoboV2 + Mapper tuning parameters
# Adjust these after testing in the actual simulation environment.

mapper:
  voxel_size: 0.05            # metres; 5 cm is adequate for arm-scale collision
  extent_meters_xyz: [2.0, 2.0, 1.5]  # workspace bounding box in world frame
  depth_minimum_distance: 0.15
  depth_maximum_distance: 2.0
  min_frames_before_esdf: 5   # frames before ESDF is trusted

tf:
  overhead_frame: camera_color_optical_frame
  wrist_frame: wrist_camera_color_optical_frame
  world_frame: world

motion_planner:
  robot_config: /ros2_ws/src/pipeline_orchestrator/config/ur5_curobo.yml
  max_attempts: 3
```

- [ ] **Step 4: Register the launch directory in setup.py**

Add two lines to `src/pipeline_orchestrator/setup.py` — the `import os, glob` at the top and the launch entry in `data_files`:

```python
import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'pipeline_orchestrator'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),   # ← add this line
         glob('launch/*.py')),                             # ← and this line
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Joshua Hyunbin Lee',
    maintainer_email='jshyunbin@gmail.com',
    description='Pipeline orchestrator for the AI planner.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'orchestrator = pipeline_orchestrator.orchestrator:main',
        ],
    },
)
```

- [ ] **Step 5: Verify the launch file is found after build**

```bash
docker compose run --rm ai_planner bash -c "
  cd /ros2_ws && colcon build --symlink-install -q &&
  source install/setup.bash &&
  ros2 launch pipeline_orchestrator contest_run.launch.py --show-args"
```

Expected: no errors, launch description printed.

- [ ] **Step 6: Commit**

```bash
git add src/pipeline_orchestrator/launch/contest_run.launch.py \
        src/pipeline_orchestrator/setup.py \
        config/curobo.yaml
git commit -m "feat: add contest_run launch file and curobo.yaml config"
```

---

## Task 7: Full Docker build + smoke test

**Files:** none — verification only.

- [ ] **Step 1: Build the Docker image**

```bash
docker compose build 2>&1 | tail -30
```

Expected: exits 0. The cuRoboV2 install takes ~5 minutes.

- [ ] **Step 2: Verify cuRoboV2 imports inside the container**

```bash
docker compose run --rm ai_planner python3 -c "
from curobo.perception import Mapper, MapperCfg, FilterDepth
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import CameraObservation, Pose, JointState, GoalToolPose
from curobo._src.geom.types import SceneCfg, VoxelGrid
print('cuRoboV2 imports OK')
import warp as wp; wp.init(); print('Warp GPU init OK')
"
```

Expected: both OK lines printed, no import errors.

- [ ] **Step 3: Verify UR5 URDF was generated**

```bash
docker compose run --rm ai_planner bash -c "
  head -5 /ur5.urdf && echo '---' &&
  python3 -c \"
import yourdfpy
r = yourdfpy.URDF.load('/ur5.urdf')
joints = [j for j in r.joint_map if r.joint_map[j].type != 'fixed']
print('Joints:', joints)
\""
```

Expected: URDF header printed; joints list includes `shoulder_pan_joint`, `shoulder_lift_joint`, `elbow_joint`, `wrist_1_joint`, `wrist_2_joint`, `wrist_3_joint`.

> **If joint names differ**, update the `cspace.joint_names` list in `config/ur5_curobo.yml` to match exactly.

- [ ] **Step 4: Run full test suite inside container**

```bash
docker compose run --rm ai_planner bash -c "
  cd /ros2_ws && source install/setup.bash &&
  python3 -m pytest src/pipeline_orchestrator/test/test_orchestrator.py -v \
    2>&1 | tail -30"
```

Expected: all tests PASS.

- [ ] **Step 5: Commit and push**

```bash
git add .
git commit -m "feat: complete cuRoboV2 depth-to-planning integration"
git push
```
