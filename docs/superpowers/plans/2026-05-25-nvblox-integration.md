# nvblox Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate nvblox as a persistent scene map providing ESDF to cuRobo for collision avoidance and object point clouds to GraspGen, with a GeminiLocalizer stub seeding SAM2 with bounding boxes.

**Architecture:** The Isaac ROS nvblox node runs inside the container and fuses overhead + wrist RGBD continuously. A new `NvBlox` Python class subscribes to its ESDF topic. On each task command, GeminiLocalizer and nvblox map readiness run in parallel via `ThreadPoolExecutor`; SAM2 uses Gemini's bounding box as a spatial prompt; `NvBlox.extract_object_cloud()` carves the target object from the nvblox mesh for GraspGen; cuRobo plans with the live ESDF.

**Tech Stack:** ROS2 Humble, Isaac ROS nvblox, Python 3.10, numpy, concurrent.futures, pytest, unittest.mock

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `Dockerfile` | Modify | Add Isaac ROS apt repo + nvblox package |
| `requirements/nvblox.txt` | Create | Python-side nvblox deps placeholder |
| `src/pipeline_orchestrator/pipeline_orchestrator/gemini.py` | Create | `GeminiLocalizer` stub — teammate implements |
| `src/pipeline_orchestrator/pipeline_orchestrator/nvblox.py` | Create | `NvBlox` — ESDF subscription + object cloud extraction stub |
| `src/pipeline_orchestrator/pipeline_orchestrator/sam2.py` | Modify | Add optional `bbox` parameter to `segment()` |
| `src/pipeline_orchestrator/pipeline_orchestrator/graspgen.py` | Modify | Change `generate_grasp()` input to `(N,3)` point cloud |
| `src/pipeline_orchestrator/pipeline_orchestrator/curobo.py` | Modify | Add `esdf` parameter to `plan_trajectory()` |
| `src/pipeline_orchestrator/pipeline_orchestrator/orchestrator.py` | Modify | Instantiate `NvBlox` + `GeminiLocalizer`, parallel pipeline flow |
| `src/pipeline_orchestrator/test/test_orchestrator.py` | Modify | Replace placeholder tests with interface contract tests |

---

### Task 1: Dockerfile — add Isaac ROS nvblox

**Files:**
- Modify: `Dockerfile`
- Create: `requirements/nvblox.txt`

- [ ] **Step 1: Add Isaac ROS apt repository and nvblox package to Dockerfile**

In `Dockerfile`, insert the following block after the PyTorch install step (after the `--index-url https://download.pytorch.org/whl/cu128` line):

```dockerfile
# Isaac ROS apt repository (required for nvblox)
RUN apt-get update && apt-get install -y curl gnupg && \
    curl -sSL https://isaac.download.nvidia.com/isaac-ros/repos.key \
        | gpg --dearmor -o /usr/share/keyrings/isaac-ros.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/isaac-ros.gpg] \
        https://isaac.download.nvidia.com/isaac-ros/release-3 \
        $(. /etc/os-release && echo $VERSION_CODENAME) release" \
        | tee /etc/apt/sources.list.d/isaac-ros.list > /dev/null && \
    rm -rf /var/lib/apt/lists/*

# nvblox ROS2 node
RUN apt-get update && apt-get install -y \
    ros-humble-isaac-ros-nvblox \
    && rm -rf /var/lib/apt/lists/*
```

> If the build fails with a key or 404 error, verify the current Isaac ROS apt setup against https://nvidia-isaac-ros.github.io/getting_started/index.html — the key URL and release path may differ for newer Isaac ROS versions.

- [ ] **Step 2: Create requirements/nvblox.txt**

```
# Python-side nvblox dependencies
# nvblox Python bindings are installed via the ros-humble-isaac-ros-nvblox apt package.
# Add pip packages here when implementing NvBlox.extract_object_cloud.
```

- [ ] **Step 3: Verify Docker build**

```bash
docker compose build 2>&1 | tail -20
```

Expected: `Successfully built <hash>` with no errors.

- [ ] **Step 4: Commit**

```bash
git add Dockerfile requirements/nvblox.txt
git commit -m "feat: add Isaac ROS nvblox to Dockerfile

Written By: Claude Sonnet 4.6"
```

---

### Task 2: Write interface tests (TDD — run before implementing)

**Files:**
- Modify: `src/pipeline_orchestrator/test/test_orchestrator.py`

- [ ] **Step 1: Replace full contents of test file**

```python
import numpy as np
import pytest
from unittest.mock import MagicMock


# --- GeminiLocalizer ---

def test_gemini_localizer_importable():
    from pipeline_orchestrator.gemini import GeminiLocalizer
    assert GeminiLocalizer is not None


def test_gemini_locate_object_returns_none_stub():
    from pipeline_orchestrator.gemini import GeminiLocalizer
    localizer = GeminiLocalizer(MagicMock())
    result = localizer.locate_object(MagicMock(), "pick up the red cube")
    assert result is None


# --- NvBlox ---

def test_nvblox_importable():
    from pipeline_orchestrator.nvblox import NvBlox
    assert NvBlox is not None


def test_nvblox_get_esdf_returns_none_before_map():
    from pipeline_orchestrator.nvblox import NvBlox
    node = MagicMock()
    nvblox = NvBlox(node)
    assert nvblox.get_esdf() is None


def test_nvblox_extract_object_cloud_returns_none_stub():
    from pipeline_orchestrator.nvblox import NvBlox
    node = MagicMock()
    nvblox = NvBlox(node)
    mask = np.zeros((480, 640), dtype=bool)
    result = nvblox.extract_object_cloud(mask)
    assert result is None


def test_nvblox_registers_esdf_subscription():
    from pipeline_orchestrator.nvblox import NvBlox
    node = MagicMock()
    NvBlox(node)
    assert node.create_subscription.called


# --- Sam2 ---

def test_sam2_segment_accepts_bbox():
    from pipeline_orchestrator.sam2 import Sam2
    sam = Sam2(MagicMock())
    result = sam.segment(MagicMock(), prompt="red cube", bbox=(10, 20, 100, 200))
    assert result is None  # stub


def test_sam2_segment_works_without_bbox():
    from pipeline_orchestrator.sam2 import Sam2
    sam = Sam2(MagicMock())
    result = sam.segment(MagicMock(), prompt="red cube")
    assert result is None  # stub


# --- GraspGen ---

def test_graspgen_accepts_point_cloud():
    from pipeline_orchestrator.graspgen import GraspGen
    graspgen = GraspGen(MagicMock())
    point_cloud = np.zeros((100, 3), dtype=np.float32)
    result = graspgen.generate_grasp(point_cloud)
    assert result is None  # stub


# --- CuRobo ---

def test_curobo_accepts_esdf():
    from pipeline_orchestrator.curobo import CuRobo
    curobo = CuRobo(MagicMock())
    result = curobo.plan_trajectory(MagicMock(), MagicMock(), esdf=None)
    assert result is None  # stub


def test_curobo_esdf_optional():
    from pipeline_orchestrator.curobo import CuRobo
    curobo = CuRobo(MagicMock())
    result = curobo.plan_trajectory(MagicMock(), MagicMock())
    assert result is None  # stub


# --- Orchestrator ---

def test_orchestrator_importable():
    from pipeline_orchestrator.orchestrator import PipelineOrchestrator
    assert PipelineOrchestrator is not None


def test_orchestrator_has_task_command_callback():
    from pipeline_orchestrator.orchestrator import PipelineOrchestrator
    assert callable(PipelineOrchestrator.task_command_callback)


def test_orchestrator_has_run_pipeline():
    from pipeline_orchestrator.orchestrator import PipelineOrchestrator
    assert callable(PipelineOrchestrator._run_pipeline)
```

- [ ] **Step 2: Run tests — confirm they fail for the right reasons**

```bash
docker compose run --rm ai_planner bash -c \
  "cd /ros2_ws && pytest src/pipeline_orchestrator/test/test_orchestrator.py -v 2>&1"
```

Expected: `gemini` and `nvblox` tests fail with `ModuleNotFoundError`; `sam2`, `graspgen`, `curobo` tests fail with `TypeError` (wrong signatures); orchestrator tests pass.

- [ ] **Step 3: Commit**

```bash
git add src/pipeline_orchestrator/test/test_orchestrator.py
git commit -m "test: replace placeholder tests with interface contract tests

Written By: Claude Sonnet 4.6"
```

---

### Task 3: Create gemini.py stub

**Files:**
- Create: `src/pipeline_orchestrator/pipeline_orchestrator/gemini.py`

- [ ] **Step 1: Create gemini.py**

```python
from sensor_msgs.msg import Image


class GeminiLocalizer:
    """Gemini Vision API — locates target object in overhead RGB image.

    Returns a bounding box (x1, y1, x2, y2) in pixel coordinates.
    Implementation handled by separate teammate.
    """

    def __init__(self, logger):
        self._logger = logger
        # TODO: initialize google-generativeai client with API key

    def locate_object(
        self,
        rgb_image: Image,
        task_prompt: str,
    ) -> tuple[int, int, int, int] | None:
        """Query Gemini Vision with overhead RGB and task prompt.

        Returns (x1, y1, x2, y2) bounding box in pixel coords, or None on failure.
        """
        # TODO: convert sensor_msgs/Image to PIL Image, call Gemini Vision API
        self._logger.warn('GeminiLocalizer.locate_object not yet implemented.')
        return None
```

- [ ] **Step 2: Run gemini tests**

```bash
docker compose run --rm ai_planner bash -c \
  "cd /ros2_ws && pytest src/pipeline_orchestrator/test/test_orchestrator.py \
   -k 'gemini' -v 2>&1"
```

Expected: `test_gemini_localizer_importable` PASS, `test_gemini_locate_object_returns_none_stub` PASS.

- [ ] **Step 3: Commit**

```bash
git add src/pipeline_orchestrator/pipeline_orchestrator/gemini.py
git commit -m "feat: add GeminiLocalizer stub

Written By: Claude Sonnet 4.6"
```

---

### Task 4: Update sam2.py — add bbox parameter

**Files:**
- Modify: `src/pipeline_orchestrator/pipeline_orchestrator/sam2.py`

- [ ] **Step 1: Replace full contents of sam2.py**

```python
from sensor_msgs.msg import Image


class Sam2:
    """SAM2 segmentation module."""

    def __init__(self, logger):
        self._logger = logger
        # TODO: load SAM2 model (hydra config + checkpoint)

    def segment(
        self,
        rgb: Image,
        prompt: str,
        bbox: tuple[int, int, int, int] | None = None,
    ):
        """Run SAM2 on rgb image.

        If bbox is provided (x1, y1, x2, y2 in pixels), uses it as a spatial
        prompt for higher accuracy. Otherwise falls back to text prompt alone.
        Returns masks or None on failure.
        """
        # TODO: convert sensor_msgs/Image to numpy array
        # TODO: if bbox provided, use predictor.set_image() + predictor.predict(box=bbox)
        # TODO: else use SAM2AutomaticMaskGenerator with text prompt
        self._logger.warn('Sam2.segment not yet implemented.')
        return None
```

- [ ] **Step 2: Run SAM2 tests**

```bash
docker compose run --rm ai_planner bash -c \
  "cd /ros2_ws && pytest src/pipeline_orchestrator/test/test_orchestrator.py \
   -k 'sam2' -v 2>&1"
```

Expected: `test_sam2_segment_accepts_bbox` PASS, `test_sam2_segment_works_without_bbox` PASS.

- [ ] **Step 3: Commit**

```bash
git add src/pipeline_orchestrator/pipeline_orchestrator/sam2.py
git commit -m "feat: add optional bbox parameter to Sam2.segment

Written By: Claude Sonnet 4.6"
```

---

### Task 5: Update graspgen.py — point cloud input

**Files:**
- Modify: `src/pipeline_orchestrator/pipeline_orchestrator/graspgen.py`

- [ ] **Step 1: Replace full contents of graspgen.py**

```python
import numpy as np


class GraspGen:
    """GraspGen grasp pose generation module."""

    def __init__(self, logger):
        self._logger = logger
        # TODO: load GraspGen diffusion model

    def generate_grasp(self, point_cloud: np.ndarray):
        """Generate grasp pose from segmented object point cloud.

        Args:
            point_cloud: (N, 3) float32 array of object surface points
                         in robot base frame.

        Returns grasp pose or None on failure.
        """
        # TODO: run GraspGen diffusion model on point_cloud
        self._logger.warn('GraspGen.generate_grasp not yet implemented.')
        return None
```

- [ ] **Step 2: Run GraspGen tests**

```bash
docker compose run --rm ai_planner bash -c \
  "cd /ros2_ws && pytest src/pipeline_orchestrator/test/test_orchestrator.py \
   -k 'graspgen' -v 2>&1"
```

Expected: `test_graspgen_accepts_point_cloud` PASS.

- [ ] **Step 3: Commit**

```bash
git add src/pipeline_orchestrator/pipeline_orchestrator/graspgen.py
git commit -m "feat: update GraspGen to accept (N,3) point cloud instead of depth image

Written By: Claude Sonnet 4.6"
```

---

### Task 6: Update curobo.py — add esdf parameter

**Files:**
- Modify: `src/pipeline_orchestrator/pipeline_orchestrator/curobo.py`

- [ ] **Step 1: Replace full contents of curobo.py**

```python
from sensor_msgs.msg import JointState


class CuRobo:
    """cuRobo motion planning module."""

    def __init__(self, logger):
        self._logger = logger
        # TODO: initialize cuRobo RobotConfig and MotionGenConfig

    def plan_trajectory(self, grasp_pose, joint_states: JointState, esdf=None):
        """Plan joint trajectory from current state to grasp pose.

        Args:
            grasp_pose: Target end-effector pose.
            joint_states: Current robot joint state.
            esdf: nvblox ESDF PointCloud2 from NvBlox.get_esdf(). When provided,
                  used to construct WorldNvbloxCollision for collision checking.
                  If None, plans without a dynamic collision world.

        Returns trajectory or None on failure.
        """
        # TODO: if esdf provided, construct WorldNvbloxCollision from esdf
        # TODO: initialize MotionGen with WorldNvbloxCollision or WorldPrimitiveCollision
        # TODO: call MotionGen.plan_single(start_state, goal_pose)
        self._logger.warn('CuRobo.plan_trajectory not yet implemented.')
        return None
```

- [ ] **Step 2: Run cuRobo tests**

```bash
docker compose run --rm ai_planner bash -c \
  "cd /ros2_ws && pytest src/pipeline_orchestrator/test/test_orchestrator.py \
   -k 'curobo' -v 2>&1"
```

Expected: `test_curobo_accepts_esdf` PASS, `test_curobo_esdf_optional` PASS.

- [ ] **Step 3: Commit**

```bash
git add src/pipeline_orchestrator/pipeline_orchestrator/curobo.py
git commit -m "feat: add esdf parameter to CuRobo.plan_trajectory for WorldNvbloxCollision

Written By: Claude Sonnet 4.6"
```

---

### Task 7: Create nvblox.py — NvBlox class

**Files:**
- Create: `src/pipeline_orchestrator/pipeline_orchestrator/nvblox.py`

- [ ] **Step 1: Create nvblox.py**

```python
import threading
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2


class NvBlox:
    """Wraps the Isaac ROS nvblox ROS2 node interface.

    The nvblox node runs as a separate process inside the container and fuses
    overhead + wrist RGBD streams into a persistent TSDF/ESDF. This class
    subscribes to the published ESDF topic and exposes two interfaces:
      - get_esdf()              → cuRobo WorldNvbloxCollision
      - extract_object_cloud()  → GraspGen (N, 3) point cloud in robot frame
    """

    ESDF_TOPIC = '/nvblox_node/static_esdf_pointcloud'

    def __init__(self, node: Node):
        self._node = node
        self._logger = node.get_logger()
        self._esdf_msg = None
        self._lock = threading.Lock()

        self._esdf_sub = node.create_subscription(
            PointCloud2,
            self.ESDF_TOPIC,
            self._on_esdf,
            10,
        )
        self._logger.info('NvBlox: subscribing to %s' % self.ESDF_TOPIC)

    def _on_esdf(self, msg: PointCloud2):
        with self._lock:
            self._esdf_msg = msg

    def get_esdf(self) -> PointCloud2 | None:
        """Return latest ESDF PointCloud2 from nvblox.

        Returns None until the nvblox node publishes its first map.
        Subsequent calls return the cached, auto-updating message.
        """
        with self._lock:
            return self._esdf_msg

    def extract_object_cloud(
        self,
        mask_2d: np.ndarray,
        camera: str = 'overhead',
    ) -> np.ndarray | None:
        """Extract object point cloud from nvblox mesh using a 2D segmentation mask.

        Args:
            mask_2d: Boolean (H, W) mask from SAM2 in the specified camera's image space.
            camera: Which camera the mask belongs to ('overhead' or 'wrist').

        Returns (N, 3) float32 point cloud in robot base frame, or None on failure.
        """
        # TODO: subscribe to /nvblox_node/mesh (nvblox_msgs/Mesh)
        # TODO: project mask_2d pixel rays into nvblox mesh via raycasting
        # TODO: transform resulting 3D points to robot base frame using TF2
        self._logger.warn('NvBlox.extract_object_cloud not yet implemented.')
        return None
```

- [ ] **Step 2: Run nvblox tests**

```bash
docker compose run --rm ai_planner bash -c \
  "cd /ros2_ws && pytest src/pipeline_orchestrator/test/test_orchestrator.py \
   -k 'nvblox' -v 2>&1"
```

Expected: All four nvblox tests PASS.

- [ ] **Step 3: Commit**

```bash
git add src/pipeline_orchestrator/pipeline_orchestrator/nvblox.py
git commit -m "feat: add NvBlox class with ESDF subscription and extract_object_cloud stub

Written By: Claude Sonnet 4.6"
```

---

### Task 8: Update orchestrator.py — parallel pipeline + NvBlox/Gemini

**Files:**
- Modify: `src/pipeline_orchestrator/pipeline_orchestrator/orchestrator.py`

- [ ] **Step 1: Replace full contents of orchestrator.py**

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
from pipeline_orchestrator.nvblox import NvBlox
from pipeline_orchestrator.gemini import GeminiLocalizer


class PipelineOrchestrator(Node):
    """Single ROS2 node running the full pipeline.

    Pipeline per task command:
      [parallel] GeminiLocalizer (overhead RGB + prompt → bbox)
                 NvBlox.get_esdf() (ensure map ready)
      SAM2 (overhead RGB + Gemini bbox → object mask)
      NvBlox.extract_object_cloud (mask → object point cloud in robot frame)
      GraspGen (point cloud → grasp candidates)
      cuRobo (grasp candidates + ESDF → collision-free trajectory)
      [fallback] MoveIt2 if cuRobo fails

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

    OVERHEAD_RGB_TOPIC = '/camera/camera/color/image_raw'
    OVERHEAD_DEPTH_TOPIC = '/camera/camera/depth/color/image_raw'
    WRIST_RGB_TOPIC = '/wrist_camera/wrist_camera/color/image_raw'
    WRIST_DEPTH_TOPIC = '/wrist_camera/wrist_camera/depth/color/image_raw'
    JOINT_STATES_TOPIC = '/joint_states'
    TASK_COMMANDS_TOPIC = '/task_commands'

    def __init__(self):
        super().__init__('pipeline_orchestrator')

        self.task_sub = self.create_subscription(
            String, self.TASK_COMMANDS_TOPIC, self.task_command_callback, 10)
        self.overhead_rgb_sub = self.create_subscription(
            Image, self.OVERHEAD_RGB_TOPIC, self._cache_overhead_rgb, 10)
        self.overhead_depth_sub = self.create_subscription(
            Image, self.OVERHEAD_DEPTH_TOPIC, self._cache_overhead_depth, 10)
        self.wrist_rgb_sub = self.create_subscription(
            Image, self.WRIST_RGB_TOPIC, self._cache_wrist_rgb, 10)
        self.wrist_depth_sub = self.create_subscription(
            Image, self.WRIST_DEPTH_TOPIC, self._cache_wrist_depth, 10)
        self.joint_sub = self.create_subscription(
            JointState, self.JOINT_STATES_TOPIC, self._cache_joints, 10)

        self._latest_overhead_rgb = None
        self._latest_overhead_depth = None
        self._latest_wrist_rgb = None
        self._latest_wrist_depth = None
        self._latest_joints = None

        self._arm_client = ActionClient(
            self, FollowJointTrajectory, '/ur5_controller/follow_joint_trajectory')
        self._gripper_client = ActionClient(
            self, FollowJointTrajectory, '/gripper_controller/follow_joint_trajectory')

        self._sam2 = Sam2(self.get_logger())
        self._graspgen = GraspGen(self.get_logger())
        self._curobo = CuRobo(self.get_logger())
        self._moveit2 = MoveIt2(self)
        self._nvblox = NvBlox(self)
        self._gemini = GeminiLocalizer(self.get_logger())

        self.get_logger().info('pipeline_orchestrator ready.')

    def _cache_overhead_rgb(self, msg): self._latest_overhead_rgb = msg
    def _cache_overhead_depth(self, msg): self._latest_overhead_depth = msg
    def _cache_wrist_rgb(self, msg): self._latest_wrist_rgb = msg
    def _cache_wrist_depth(self, msg): self._latest_wrist_depth = msg
    def _cache_joints(self, msg): self._latest_joints = msg

    def task_command_callback(self, msg):
        self.get_logger().info(f'Received task command: {msg.data}')
        self._run_pipeline(msg.data)

    def _run_pipeline(self, task: str):
        # Gemini API call and nvblox map readiness run in parallel — both are
        # I/O-bound on first call (~1-2s each); subsequent calls are instant.
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_esdf = executor.submit(self._nvblox.get_esdf)
            future_bbox = executor.submit(
                self._gemini.locate_object, self._latest_overhead_rgb, task)

        esdf = future_esdf.result()
        bbox = future_bbox.result()

        masks = self._sam2.segment(
            self._latest_overhead_rgb, prompt=task, bbox=bbox)
        if masks is None:
            return

        point_cloud = self._nvblox.extract_object_cloud(masks)
        if point_cloud is None:
            return

        grasp_pose = self._graspgen.generate_grasp(point_cloud)
        if grasp_pose is None:
            return

        trajectory = self._curobo.plan_trajectory(
            grasp_pose, self._latest_joints, esdf=esdf)
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

- [ ] **Step 2: Run all tests**

```bash
docker compose run --rm ai_planner bash -c \
  "cd /ros2_ws && pytest src/pipeline_orchestrator/test/test_orchestrator.py -v 2>&1"
```

Expected: All tests PASS.

- [ ] **Step 3: Commit**

```bash
git add src/pipeline_orchestrator/pipeline_orchestrator/orchestrator.py
git commit -m "feat: wire NvBlox and GeminiLocalizer into orchestrator pipeline

Written By: Claude Sonnet 4.6"
```

---

### Task 9: Rebuild workspace and smoke test

- [ ] **Step 1: Rebuild colcon workspace**

```bash
docker compose run --rm ai_planner bash /ros2_ws/scripts/build.sh
```

Expected: `Summary: X packages finished` with no errors or warnings about missing modules.

- [ ] **Step 2: Run full test suite**

```bash
docker compose run --rm ai_planner bash -c \
  "cd /ros2_ws && pytest src/pipeline_orchestrator/test/test_orchestrator.py -v 2>&1"
```

Expected: All tests PASS.

- [ ] **Step 3: Verify node launches cleanly**

```bash
docker compose run --rm ai_planner bash -c \
  "source /opt/ros/humble/setup.bash && \
   source /ros2_ws/install/setup.bash && \
   timeout 5 ros2 run pipeline_orchestrator orchestrator 2>&1 || true"
```

Expected: `pipeline_orchestrator ready.` appears before the 5s timeout. No `ImportError` or `ModuleNotFoundError`.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: verify nvblox integration builds and node launches cleanly

Written By: Claude Sonnet 4.6"
```
