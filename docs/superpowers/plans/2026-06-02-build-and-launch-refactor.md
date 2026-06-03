# Build & Launch Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse the two-Dockerfile/two-script build into a single layer-ordered Dockerfile, reduce the pipeline to two one-command modes (deploy / debug), and add a dedicated `debug_viz` node that serves segmented clouds, ranked grasp poses, and the live TSDF over one viser server.

**Architecture:** One Dockerfile with heavy/stable layers first and `COPY src` last (deploy bakes src; debug live-mounts over it). Two compose files (`docker-compose.yml`, `docker-compose.debug.yml`) invoked by documented one-liners. Three launch files: `pipeline_common.launch.py` holds the nodes; `deploy.launch.py` and `debug.launch.py` set the mode-specific toggles. Viz is decoupled: `graspgen_service` publishes a `PoseArray`, `curobo_service` publishes TSDF voxel centers as a `PointCloud2`, and a new `debug_viz` node subscribes to those plus the existing cloud topics and renders them in viser.

**Tech Stack:** Docker / docker compose, ROS2 Humble (rclpy, launch), viser, numpy, pytest.

**Reference spec:** `docs/superpowers/specs/2026-06-02-build-and-launch-refactor-design.md`

---

## File Structure

**Docker / compose (Task 1–2):**
- Modify→replace: `Dockerfile` (merged, layer-ordered)
- Delete: `Dockerfile.base`, `scripts/build.sh`, `scripts/build_base_image.sh`, `scripts/build_image.sh`, `scripts/run.sh`
- Modify: `docker-compose.yml` (deploy command, drop base-image build arg)
- Create: `docker-compose.debug.yml` (replaces `docker-compose.dev.yml`)
- Delete: `docker-compose.dev.yml`

**Python nodes (Task 3–6):**
- Modify: `src/pipeline_orchestrator/pipeline_orchestrator/pipeline_utils.py` (add `quat_from_rotation_matrix`, `pose_from_grasp_row`)
- Modify: `src/pipeline_orchestrator/pipeline_orchestrator/orchestrator.py` (use the shared helpers)
- Modify: `src/pipeline_orchestrator/pipeline_orchestrator/graspgen_service.py` (publish `/graspgen/grasp_poses`)
- Modify: `src/pipeline_orchestrator/pipeline_orchestrator/curobo_service.py` (publish `/curobo/tsdf_voxels`)
- Create: `src/pipeline_orchestrator/pipeline_orchestrator/debug_viz.py` (new node)
- Modify: `src/pipeline_orchestrator/setup.py` (add `debug_viz` entry point)

**Launch + docs (Task 7):**
- Create: `src/pipeline_orchestrator/launch/pipeline_common.launch.py`
- Create: `src/pipeline_orchestrator/launch/deploy.launch.py`
- Create: `src/pipeline_orchestrator/launch/debug.launch.py`
- Delete: `src/pipeline_orchestrator/launch/planner_pipeline.launch.py`
- Modify: `.claude/CLAUDE.md` (one-liners, node/topic tables, key files)

**Tests:**
- Create: `src/pipeline_orchestrator/test/test_pipeline_utils.py`
- Modify: `src/pipeline_orchestrator/test/test_graspgen_service.py`
- Create: `src/pipeline_orchestrator/test/test_debug_viz.py`
- Modify: `src/pipeline_orchestrator/test/test_orchestrator.py` (curobo_service voxel publish test lives here alongside existing curobo tests)

**Testing note:** Unit tests run inside the container where ROS is available. From the repo root, run a single test module with:
`docker compose -f docker-compose.yml -f docker-compose.debug.yml run --rm ai_planner bash -lc "cd /ros2_ws && python3 -m pytest src/pipeline_orchestrator/test/<file> -v"`
The existing tests use `unittest.mock` + import-skip guards so most run without a full ROS graph.

---

## Task 1: Merge the two Dockerfiles into one

**Files:**
- Replace: `Dockerfile`
- Delete: `Dockerfile.base`

- [ ] **Step 1: Write the merged `Dockerfile`**

Replace the entire contents of `Dockerfile` with the merged definition below. It is `Dockerfile.base` (steps 1–13) followed by the old app stage (workspace copy + colcon build + entrypoints), with the `FROM ${PLANNER_BASE_IMAGE}` indirection removed. `COPY src` is the last expensive layer so edits never invalidate the heavy layers above it.

```dockerfile
FROM nvcr.io/nvidia/pytorch:23.07-py3

ARG GRASPGEN_REPO_URL=https://github.com/pianojay/GraspGen.git
ARG GRASPGEN_BRANCH=jaeuk
ARG GRASPGEN_COMMIT=31b67f65f3cb88928887edd2ee24e302c30cab70
ARG GRASPGEN_MODELS_REPO_URL=https://huggingface.co/adithyamurali/GraspGenModels
ARG GRASPGEN_MODELS_COMMIT=ec1ccbb5eec0680db669246ac312a3636f16ee43
ARG GRIPPER_CONFIG_NAME=graspgen_robotiq_2f_140.yml
ARG GRASPGEN_MODEL_FILES=checkpoints/graspgen_robotiq_2f_140.yml,checkpoints/graspgen_robotiq_2f_140_gen.pth,checkpoints/graspgen_robotiq_2f_140_dis.pth

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
ENV PIP_ROOT_USER_ACTION=ignore
ENV GRASPGEN_REPO_DIR=/opt/GraspGen
ENV GRASPGEN_MODELS_DIR=/opt/GraspGenModels
ENV SAM2_MODEL_DIR=/opt/models/sam2
ENV SAM2_MODEL_PATH=/opt/models/sam2/sam2_t.pt

COPY requirements/ /tmp/requirements/

# Locale
RUN apt-get update && apt-get install -y locales && \
    locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 && \
    rm -rf /var/lib/apt/lists/*

# Base system packages, ROS2 apt source, and git-lfs for Hugging Face model pulls.
RUN apt-get update && apt-get install -y \
    curl \
    git \
    git-lfs \
    gnupg2 \
    lsb-release \
    locales \
    software-properties-common \
    tmux \
    libosmesa6-dev && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
        http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | \
        tee /etc/apt/sources.list.d/ros2.list > /dev/null && \
    git lfs install --system && \
    rm -rf /var/lib/apt/lists/*

# ROS2 Humble + Python tools for planner-side nodes.
RUN apt-get update && apt-get install -y \
    ros-humble-ros-base \
    ros-humble-control-msgs \
    ros-humble-cv-bridge \
    ros-humble-rosidl-default-generators \
    ros-humble-ur-description \
    ros-humble-vision-msgs \
    ros-humble-xacro \
    python3-opencv \
    python3-colcon-common-extensions \
    python3-numpy \
    python3-rosdep \
    python3-pip && \
    rm -rf /var/lib/apt/lists/*

# UR5 URDF used by the cuRobo robot configuration.
RUN bash -c "source /opt/ros/humble/setup.bash && \
    xacro /opt/ros/humble/share/ur_description/urdf/ur.urdf.xacro \
      ur_type:=ur5 name:=ur > /ur5.urdf"

# Additional non-ROS system packages for pointnet2_ops compilation.
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Planner-side Python runtime. Keep this separate from GraspGen so Gemini/SAM2
# issues do not get conflated with GraspGen inference issues.
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir -r /tmp/requirements/planner-runtime.txt && \
    python3 -m pip install --no-cache-dir --no-deps -r /tmp/requirements/sam2.txt

# Clone the fork only after ROS2/system and planner runtime are established.
RUN git clone --recursive --branch ${GRASPGEN_BRANCH} ${GRASPGEN_REPO_URL} ${GRASPGEN_REPO_DIR} && \
    cd ${GRASPGEN_REPO_DIR} && \
    git checkout ${GRASPGEN_COMMIT} && \
    git submodule update --init --recursive

# GraspGen runtime. This is intentionally separated from the planner runtime above.
RUN python3 -m pip install --no-cache-dir -r ${GRASPGEN_REPO_DIR}/requirements.zmq_pointnet_viser.txt

# Official GraspGen pointnet installation pattern, adapted for the reduced runtime.
RUN cd ${GRASPGEN_REPO_DIR}/pointnet2_ops && \
    python3 -m pip install --no-cache-dir --no-build-isolation .

# Install the fork itself without re-resolving the broad upstream dependency set.
RUN cd ${GRASPGEN_REPO_DIR} && \
    python3 -m pip install --no-cache-dir --no-deps -e .

# cuRoboV2 runtime. The install order mirrors the validated smoke test against
# the current planner image; re-pin numpy afterwards to preserve cv_bridge ABI.
RUN python3 -m pip install --no-cache-dir "numpy<2" uv && \
    git clone --branch v0.8.0 --depth 1 https://github.com/NVlabs/curobo.git /tmp/curobo && \
    cd /tmp/curobo && \
    uv pip install --system ".[cu12]" && \
    cd / && rm -rf /tmp/curobo && \
    python3 -m pip install --no-cache-dir --force-reinstall "numpy<2"

# Bake the Ultralytics SAM2 checkpoint into the image to avoid first-run downloads.
RUN mkdir -p ${SAM2_MODEL_DIR} && \
    curl -L https://github.com/ultralytics/assets/releases/download/v8.4.0/sam2_t.pt \
      -o ${SAM2_MODEL_PATH} && \
    test -s ${SAM2_MODEL_PATH}

# Download only the pinned GraspGen model assets required by the planner.
RUN export GIT_LFS_SKIP_SMUDGE=1 && \
    git clone ${GRASPGEN_MODELS_REPO_URL} /tmp/GraspGenModels && \
    cd /tmp/GraspGenModels && \
    git checkout ${GRASPGEN_MODELS_COMMIT} && \
    git lfs pull --include="${GRASPGEN_MODEL_FILES}" && \
    mkdir -p ${GRASPGEN_MODELS_DIR}/checkpoints && \
    cp checkpoints/graspgen_robotiq_2f_140.yml ${GRASPGEN_MODELS_DIR}/checkpoints/ && \
    cp checkpoints/graspgen_robotiq_2f_140_gen.pth ${GRASPGEN_MODELS_DIR}/checkpoints/ && \
    cp checkpoints/graspgen_robotiq_2f_140_dis.pth ${GRASPGEN_MODELS_DIR}/checkpoints/ && \
    rm -rf /tmp/GraspGenModels && \
    test -f "${GRASPGEN_MODELS_DIR}/checkpoints/${GRIPPER_CONFIG_NAME}" && \
    test -f "${GRASPGEN_MODELS_DIR}/checkpoints/graspgen_robotiq_2f_140_gen.pth" && \
    test -f "${GRASPGEN_MODELS_DIR}/checkpoints/graspgen_robotiq_2f_140_dis.pth"

# ── Application layer (frequently changing; kept last so the heavy layers
#    above stay cached across src edits) ──────────────────────────────────────
WORKDIR /ros2_ws
COPY src/ src/
COPY scripts/ scripts/
COPY misc/ misc/
RUN . /opt/ros/humble/setup.sh && \
    colcon build --symlink-install

COPY scripts/entrypoint.sh /entrypoint.sh
COPY scripts/start_graspgen_server.sh /start_graspgen_server.sh
RUN chmod +x /entrypoint.sh && chmod +x /start_graspgen_server.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
```

- [ ] **Step 2: Delete the old base Dockerfile**

```bash
git rm Dockerfile.base
```

- [ ] **Step 3: Verify the Dockerfile parses (no full build yet)**

Run: `docker build --check -f Dockerfile .`
Expected: `Check complete, no warnings found.` (or only pre-existing warnings; no syntax errors). If `--check` is unavailable on the installed Docker, instead run `docker buildx build --print=outline -f Dockerfile .` or simply confirm `docker compose config` (Task 2) succeeds.

- [ ] **Step 4: Commit**

```bash
git add Dockerfile
git commit -m "Merge Dockerfile.base into a single layer-ordered Dockerfile

Heavy/stable layers (CUDA, ROS, pip, GraspGen, cuRobo, model downloads)
come first; COPY src + colcon build is last so code edits never rebuild
them. Removes the PLANNER_BASE_IMAGE build-arg indirection.

Written By: Claude Opus 4.8"
```

---

## Task 2: Rework compose files and remove wrapper scripts

**Files:**
- Modify: `docker-compose.yml`
- Create: `docker-compose.debug.yml`
- Delete: `docker-compose.dev.yml`, `scripts/build.sh`, `scripts/build_base_image.sh`, `scripts/build_image.sh`, `scripts/run.sh`

- [ ] **Step 1: Rewrite `docker-compose.yml` (deploy mode)**

Replace the `build` block (drop the `PLANNER_BASE_IMAGE` arg — there is no base image anymore) and add a `command` that launches deploy mode. Replace the file contents with:

```yaml
services:
  ai_planner:
    build:
      context: .
    network_mode: host
    runtime: nvidia
    environment:
      NVIDIA_VISIBLE_DEVICES: all
      NVIDIA_DRIVER_CAPABILITIES: all
      ROS_DOMAIN_ID: ${ROS_DOMAIN_ID:-0}
      ROS_LOCALHOST_ONLY: ${ROS_LOCALHOST_ONLY:-0}
      RMW_IMPLEMENTATION: ${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}
      FASTDDS_BUILTIN_TRANSPORTS: ${FASTDDS_BUILTIN_TRANSPORTS:-UDPv4}
      FASTRTPS_DEFAULT_PROFILES_FILE: /ros2_ws/config/fastdds_no_shm.xml
      GEMINI_API_KEY: ${GEMINI_API_KEY:-}
      GRASPGEN_REPO_DIR: /opt/GraspGen
      GRASPGEN_MODELS_DIR: /opt/GraspGenModels
    volumes:
      - ./artifacts:/artifacts
      - ./config:/ros2_ws/config:ro
    ipc: host
    stdin_open: true
    tty: true
    command: ["ros2", "launch", "pipeline_orchestrator", "deploy.launch.py"]
```

- [ ] **Step 2: Create `docker-compose.debug.yml` (debug override)**

This override live-mounts source over the baked image and runs debug mode (viser viz on). Create `docker-compose.debug.yml`:

```yaml
services:
  ai_planner:
    volumes:
      - ./src:/ros2_ws/src
      - ./scripts:/ros2_ws/scripts
      - ./config:/ros2_ws/config:ro
      - ./artifacts:/artifacts
    command: ["ros2", "launch", "pipeline_orchestrator", "debug.launch.py"]
```

- [ ] **Step 3: Delete the old dev override and wrapper scripts**

```bash
git rm docker-compose.dev.yml scripts/build.sh scripts/build_base_image.sh scripts/build_image.sh scripts/run.sh
```

- [ ] **Step 4: Verify both compose configurations resolve**

Run: `docker compose config >/dev/null && echo DEPLOY_OK`
Expected: `DEPLOY_OK`
Run: `docker compose -f docker-compose.yml -f docker-compose.debug.yml config >/dev/null && echo DEBUG_OK`
Expected: `DEBUG_OK`

- [ ] **Step 5: Commit**

```bash
git add docker-compose.yml docker-compose.debug.yml
git commit -m "Replace dev override + build scripts with deploy/debug compose

docker-compose.yml runs deploy.launch.py against the baked image;
docker-compose.debug.yml live-mounts src and runs debug.launch.py.
Removes build.sh/build_base_image.sh/build_image.sh/run.sh and the
docker-compose.dev.yml override in favor of documented one-liners.

Written By: Claude Opus 4.8"
```

---

## Task 3: Extract shared pose/quaternion helpers into `pipeline_utils`

The row→Pose and rotation-matrix→quaternion conversions currently live only in `orchestrator.py`. `graspgen_service` (Task 4) needs the same conversion to publish a `PoseArray`. Move them to `pipeline_utils` (DRY) and have the orchestrator delegate.

**Files:**
- Modify: `src/pipeline_orchestrator/pipeline_orchestrator/pipeline_utils.py`
- Modify: `src/pipeline_orchestrator/pipeline_orchestrator/orchestrator.py`
- Test: `src/pipeline_orchestrator/test/test_pipeline_utils.py`

- [ ] **Step 1: Write the failing test**

Create `src/pipeline_orchestrator/test/test_pipeline_utils.py`:

```python
import math

import pytest

from pipeline_orchestrator.pipeline_utils import quat_from_rotation_matrix


def test_quat_from_identity_is_unit_quaternion():
    # (w, x, y, z) ordering, matching the orchestrator's original convention.
    w, x, y, z = quat_from_rotation_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert w == pytest.approx(1.0)
    assert (x, y, z) == pytest.approx((0.0, 0.0, 0.0))


def test_quat_from_180_deg_z_rotation():
    # 180° about Z: w=0, z=±1.
    w, x, y, z = quat_from_rotation_matrix([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    assert w == pytest.approx(0.0, abs=1e-6)
    assert abs(z) == pytest.approx(1.0, abs=1e-6)
    assert (x, y) == pytest.approx((0.0, 0.0), abs=1e-6)


def test_quat_is_normalized():
    rot = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]  # 90° about Z
    q = quat_from_rotation_matrix(rot)
    norm = math.sqrt(sum(c * c for c in q))
    assert norm == pytest.approx(1.0, abs=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest src/pipeline_orchestrator/test/test_pipeline_utils.py -v`
Expected: FAIL with `ImportError: cannot import name 'quat_from_rotation_matrix'`

- [ ] **Step 3: Add the helpers to `pipeline_utils.py`**

Append to `src/pipeline_orchestrator/pipeline_orchestrator/pipeline_utils.py`. Add `import math` near the top (next to `import os`), and add a guarded `Pose` import after the existing `from std_msgs.msg import Header` line:

```python
try:  # pragma: no cover - geometry_msgs only present in the ROS runtime
    from geometry_msgs.msg import Pose
except ImportError:  # pragma: no cover - import-only test fallback
    Pose = None
```

Then append these functions at the end of the file:

```python
def quat_from_rotation_matrix(rotation) -> tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to a (w, x, y, z) quaternion."""
    r00, r01, r02 = [float(v) for v in rotation[0]]
    r10, r11, r12 = [float(v) for v in rotation[1]]
    r20, r21, r22 = [float(v) for v in rotation[2]]
    trace = r00 + r11 + r22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return (0.25 * s, (r21 - r12) / s, (r02 - r20) / s, (r10 - r01) / s)
    if r00 > r11 and r00 > r22:
        s = math.sqrt(1.0 + r00 - r11 - r22) * 2.0
        return ((r21 - r12) / s, 0.25 * s, (r01 + r10) / s, (r02 + r20) / s)
    if r11 > r22:
        s = math.sqrt(1.0 + r11 - r00 - r22) * 2.0
        return ((r02 - r20) / s, (r01 + r10) / s, 0.25 * s, (r12 + r21) / s)
    s = math.sqrt(1.0 + r22 - r00 - r11) * 2.0
    return ((r10 - r01) / s, (r02 + r20) / s, (r12 + r21) / s, 0.25 * s)


def pose_from_grasp_row(row: dict):
    """Build a geometry_msgs/Pose from a GraspGen rank row dict.

    Returns None when geometry_msgs is unavailable or the row lacks a valid
    3-vector translation / 3x3 rotation matrix.
    """
    if Pose is None:
        return None
    translation = row.get("translation")
    rotation = row.get("rotation_matrix")
    if translation is None or rotation is None:
        return None
    if len(translation) != 3 or len(rotation) != 3:
        return None
    w, x, y, z = quat_from_rotation_matrix(rotation)
    pose = Pose()
    pose.position.x = float(translation[0])
    pose.position.y = float(translation[1])
    pose.position.z = float(translation[2])
    pose.orientation.w = w
    pose.orientation.x = x
    pose.orientation.y = y
    pose.orientation.z = z
    return pose
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest src/pipeline_orchestrator/test/test_pipeline_utils.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Point the orchestrator at the shared helpers**

In `src/pipeline_orchestrator/pipeline_orchestrator/orchestrator.py`, update the import block near line 53–54 to also import the helpers:

```python
from pipeline_orchestrator.pipeline_utils import as_bool as _as_bool
from pipeline_orchestrator.pipeline_utils import env_float as _env_float
from pipeline_orchestrator.pipeline_utils import pose_from_grasp_row as _pose_from_grasp_row_util
```

Replace the body of `PipelineOrchestrator._pose_from_grasp_row` so it delegates (keep the method so existing call sites and tests still work):

```python
    @staticmethod
    def _pose_from_grasp_row(row: dict):
        """Build geometry_msgs/Pose from a GraspGen rank row dict."""
        return _pose_from_grasp_row_util(row)
```

Then delete the now-unused `PipelineOrchestrator._quat_from_rotation_matrix` static method (the helper moved to `pipeline_utils`). Confirm no other references remain:

Run: `grep -n "_quat_from_rotation_matrix" src/pipeline_orchestrator/pipeline_orchestrator/orchestrator.py`
Expected: no output.

- [ ] **Step 6: Run the orchestrator tests to verify no regression**

Run: `python3 -m pytest src/pipeline_orchestrator/test/test_orchestrator.py -v`
Expected: PASS (all existing tests still green)

- [ ] **Step 7: Commit**

```bash
git add src/pipeline_orchestrator/pipeline_orchestrator/pipeline_utils.py \
        src/pipeline_orchestrator/pipeline_orchestrator/orchestrator.py \
        src/pipeline_orchestrator/test/test_pipeline_utils.py
git commit -m "Move grasp-row pose/quaternion helpers into pipeline_utils

Shared by the orchestrator and (next) graspgen_service's PoseArray
publisher. Orchestrator delegates to the new pose_from_grasp_row.

Written By: Claude Opus 4.8"
```

---

## Task 4: Publish ranked grasp poses from `graspgen_service`

**Files:**
- Modify: `src/pipeline_orchestrator/pipeline_orchestrator/graspgen_service.py`
- Test: `src/pipeline_orchestrator/test/test_graspgen_service.py`

- [ ] **Step 1: Write the failing test**

Append to `src/pipeline_orchestrator/test/test_graspgen_service.py`:

```python
def test_maybe_publish_grasp_poses_builds_ranked_pose_array():
    from unittest.mock import MagicMock
    from builtin_interfaces.msg import Time

    svc = GraspGenService.__new__(GraspGenService)
    pub = MagicMock()
    svc._grasp_poses_pub = pub
    # Real Time() so PoseArray.header.stamp accepts the assignment.
    svc.get_clock = MagicMock(
        return_value=MagicMock(now=lambda: MagicMock(to_msg=lambda: Time()))
    )

    rows = [
        {"translation": [0.1, 0.2, 0.3],
         "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
         "confidence": 0.9},
        {"translation": [0.4, 0.5, 0.6],
         "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
         "confidence": 0.5},
    ]
    svc._maybe_publish_grasp_poses(rows, "base_link")

    pub.publish.assert_called_once()
    msg = pub.publish.call_args.args[0]
    assert msg.header.frame_id == "base_link"
    assert len(msg.poses) == 2
    # Rank order preserved: first row maps to first pose.
    assert msg.poses[0].position.x == pytest.approx(0.1)


def test_maybe_publish_grasp_poses_noop_without_publisher():
    svc = GraspGenService.__new__(GraspGenService)
    svc._grasp_poses_pub = None
    # Must not raise when publishing is disabled.
    svc._maybe_publish_grasp_poses([{"translation": [0, 0, 0],
                                     "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                     "confidence": 1.0}], "base_link")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest src/pipeline_orchestrator/test/test_graspgen_service.py -v -k grasp_poses`
Expected: FAIL with `AttributeError: ... has no attribute '_maybe_publish_grasp_poses'`

- [ ] **Step 3: Implement the publisher in `graspgen_service.py`**

Add the import near the top of the file (with the other `pipeline_orchestrator` / message imports):

```python
from geometry_msgs.msg import PoseArray
from pipeline_orchestrator.pipeline_utils import pose_from_grasp_row
```

In `GraspGenService.__init__`, add two parameter declarations alongside the existing ones (after the `debug_dir` declaration near line 63):

```python
        self.declare_parameter("publish_grasp_poses", False)
        self.declare_parameter("grasp_poses_topic", "/graspgen/grasp_poses")
```

Then, after the cloud publishers/subscriptions are created (after the `create_service(...)` call near line 107), add:

```python
        self._grasp_poses_pub = None
        if bool(self.get_parameter("publish_grasp_poses").value):
            self._grasp_poses_pub = self.create_publisher(
                PoseArray,
                str(self.get_parameter("grasp_poses_topic").value),
                qos,
            )
```

Add the method (place it next to `_run_inference`):

```python
    def _maybe_publish_grasp_poses(self, rows: list[dict], frame_id: str) -> None:
        """Publish ranked grasps as a PoseArray when debug viz is enabled."""
        if self._grasp_poses_pub is None:
            return
        msg = PoseArray()
        msg.header.frame_id = frame_id
        msg.header.stamp = self.get_clock().now().to_msg()
        for row in rows:
            pose = pose_from_grasp_row(row)
            if pose is not None:
                msg.poses.append(pose)
        self._grasp_poses_pub.publish(msg)
```

Finally, call it in `_run_inference` right before the successful `return payload` (after `_save_debug_artifacts(...)` and the success log line):

```python
        self._maybe_publish_grasp_poses(top_rows, segmented_frame)
        return payload
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest src/pipeline_orchestrator/test/test_graspgen_service.py -v -k grasp_poses`
Expected: PASS (2 passed)

- [ ] **Step 5: Run the full graspgen test module (no regression)**

Run: `python3 -m pytest src/pipeline_orchestrator/test/test_graspgen_service.py -v`
Expected: PASS (existing cloud-sync tests still green)

- [ ] **Step 6: Commit**

```bash
git add src/pipeline_orchestrator/pipeline_orchestrator/graspgen_service.py \
        src/pipeline_orchestrator/test/test_graspgen_service.py
git commit -m "Publish ranked grasps as /graspgen/grasp_poses PoseArray

Gated by the publish_grasp_poses param (debug mode only); rank order is
preserved so debug_viz can color by candidate index.

Written By: Claude Opus 4.8"
```

---

## Task 5: Publish TSDF voxel centers from `curobo_service`

`curobo.py` already caches occupied voxel centers and exposes `get_tsdf_centers()` when `enable_viz` is set. Add a timer-driven publisher in the service node.

**Files:**
- Modify: `src/pipeline_orchestrator/pipeline_orchestrator/curobo_service.py`
- Test: `src/pipeline_orchestrator/test/test_orchestrator.py` (curobo_service tests live in this module)

- [ ] **Step 1: Write the failing test**

Append to `src/pipeline_orchestrator/test/test_orchestrator.py`:

```python
def test_curobo_service_publishes_tsdf_voxels_when_centers_present():
    import threading
    import numpy as np
    from unittest.mock import MagicMock
    from builtin_interfaces.msg import Time
    from pipeline_orchestrator.curobo_service import CuRoboService

    svc = CuRoboService.__new__(CuRoboService)
    pub = MagicMock()
    svc._tsdf_pub = pub
    svc._init_lock = threading.Lock()
    svc._curobo = MagicMock(
        get_tsdf_centers=MagicMock(return_value=np.zeros((4, 3), dtype=np.float32))
    )
    svc.get_clock = MagicMock(
        return_value=MagicMock(now=lambda: MagicMock(to_msg=lambda: Time()))
    )

    svc._publish_tsdf_voxels()

    pub.publish.assert_called_once()
    cloud = pub.publish.call_args.args[0]
    assert cloud.width == 4
    assert cloud.header.frame_id == "base_link"


def test_curobo_service_tsdf_publish_noop_without_centers():
    import threading
    from unittest.mock import MagicMock
    from pipeline_orchestrator.curobo_service import CuRoboService

    svc = CuRoboService.__new__(CuRoboService)
    pub = MagicMock()
    svc._tsdf_pub = pub
    svc._init_lock = threading.Lock()
    svc._curobo = MagicMock(get_tsdf_centers=MagicMock(return_value=None))
    svc._publish_tsdf_voxels()
    pub.publish.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest src/pipeline_orchestrator/test/test_orchestrator.py -v -k tsdf`
Expected: FAIL with `AttributeError: ... has no attribute '_publish_tsdf_voxels'`

- [ ] **Step 3: Implement the publisher in `curobo_service.py`**

Add imports near the existing imports at the top:

```python
from sensor_msgs.msg import PointCloud2
from pipeline_orchestrator.pipeline_utils import make_xyz_cloud
```

Add a module-level constant near the top of the file (after the imports):

```python
BASE_FRAME = 'base_link'
```

In `CuRoboService.__init__`, after the existing `enable_viz` parameter declaration (near line 47), add:

```python
        self.declare_parameter('tsdf_voxels_topic', '/curobo/tsdf_voxels')
```

After the service is advertised (after the `self.create_service(...)` / log block, before starting `_init_thread`), add:

```python
        self._tsdf_pub = None
        if _as_bool(self.get_parameter('enable_viz').value):
            self._tsdf_pub = self.create_publisher(
                PointCloud2,
                str(self.get_parameter('tsdf_voxels_topic').value),
                1,
            )
            self.create_timer(1.0, self._publish_tsdf_voxels)
```

Add the method to the class:

```python
    def _publish_tsdf_voxels(self) -> None:
        """Publish occupied TSDF voxel centers as a PointCloud2 (debug viz)."""
        if self._tsdf_pub is None:
            return
        with self._init_lock:
            curobo = self._curobo
        if curobo is None:
            return
        centers = curobo.get_tsdf_centers()
        if centers is None or len(centers) == 0:
            return
        cloud = make_xyz_cloud(
            np.asarray(centers, dtype=np.float32),
            BASE_FRAME,
            self.get_clock().now().to_msg(),
        )
        self._tsdf_pub.publish(cloud)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest src/pipeline_orchestrator/test/test_orchestrator.py -v -k tsdf`
Expected: PASS (2 passed)

- [ ] **Step 5: Run the full module (no regression)**

Run: `python3 -m pytest src/pipeline_orchestrator/test/test_orchestrator.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/pipeline_orchestrator/pipeline_orchestrator/curobo_service.py \
        src/pipeline_orchestrator/test/test_orchestrator.py
git commit -m "Publish TSDF voxel centers as /curobo/tsdf_voxels in debug mode

Timer-driven PointCloud2 of occupied voxel centers (from the existing
enable_viz cache) so debug_viz can render the collision world the planner
sees, without serializing the full voxel grid.

Written By: Claude Opus 4.8"
```

---

## Task 6: Add the `debug_viz` node

A single rclpy node hosting one viser server, subscribing to the segmented/background clouds, `/graspgen/grasp_poses`, and `/curobo/tsdf_voxels`. The pure logic (rank coloring, pose → viser frame args) is unit-tested; the viser/ROS wiring is verified at integration time.

**Files:**
- Create: `src/pipeline_orchestrator/pipeline_orchestrator/debug_viz.py`
- Modify: `src/pipeline_orchestrator/setup.py`
- Test: `src/pipeline_orchestrator/test/test_debug_viz.py`

- [ ] **Step 1: Write the failing test**

Create `src/pipeline_orchestrator/test/test_debug_viz.py`:

```python
from types import SimpleNamespace

import pytest

try:
    from pipeline_orchestrator.debug_viz import colors_by_rank, pose_to_position_wxyz
    _IMPORT_ERROR = None
except Exception as exc:  # noqa: BLE001 - missing runtime dep should skip
    colors_by_rank = None
    pose_to_position_wxyz = None
    _IMPORT_ERROR = exc

pytestmark = pytest.mark.skipif(
    colors_by_rank is None,
    reason=f"debug_viz import unavailable: {_IMPORT_ERROR}",
)


def test_colors_by_rank_empty():
    assert colors_by_rank(0) == []


def test_colors_by_rank_single_is_green():
    assert colors_by_rank(1) == [(0, 255, 0)]


def test_colors_by_rank_first_green_last_red():
    colors = colors_by_rank(4)
    assert len(colors) == 4
    assert colors[0] == (0, 255, 0)      # best rank → green
    assert colors[-1] == (255, 0, 0)     # worst rank → red


def test_pose_to_position_wxyz_extracts_fields():
    pose = SimpleNamespace(
        position=SimpleNamespace(x=1.0, y=2.0, z=3.0),
        orientation=SimpleNamespace(w=1.0, x=0.0, y=0.0, z=0.0),
    )
    position, wxyz = pose_to_position_wxyz(pose)
    assert position == (1.0, 2.0, 3.0)
    assert wxyz == (1.0, 0.0, 0.0, 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest src/pipeline_orchestrator/test/test_debug_viz.py -v`
Expected: FAIL/skip-then-fail — the module does not exist yet, so collection errors or skips with `No module named 'pipeline_orchestrator.debug_viz'`.

- [ ] **Step 3: Create `debug_viz.py`**

Create `src/pipeline_orchestrator/pipeline_orchestrator/debug_viz.py`:

```python
"""Debug visualization node.

Hosts a single viser server and renders the live pipeline state by subscribing
to lightweight ROS topics:

  /graspgen/segmented_object, /graspgen/background  (sensor_msgs/PointCloud2)
  /graspgen/grasp_poses                             (geometry_msgs/PoseArray)
  /curobo/tsdf_voxels                               (sensor_msgs/PointCloud2)

Visualization is fully decoupled: the heavy nodes publish; this node only reads.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
import viser
from geometry_msgs.msg import PoseArray
from sensor_msgs.msg import PointCloud2


def colors_by_rank(n: int) -> list[tuple[int, int, int]]:
    """Green (best rank) → red (worst rank) RGB tuples for n grasps."""
    if n <= 0:
        return []
    colors = []
    for i in range(n):
        t = i / max(n - 1, 1)
        colors.append((int(round(255 * t)), int(round(255 * (1 - t))), 0))
    return colors


def pose_to_position_wxyz(pose):
    """Extract ((x, y, z), (w, x, y, z)) from a geometry_msgs/Pose."""
    position = (float(pose.position.x), float(pose.position.y), float(pose.position.z))
    wxyz = (
        float(pose.orientation.w),
        float(pose.orientation.x),
        float(pose.orientation.y),
        float(pose.orientation.z),
    )
    return position, wxyz


def _cloud_to_xyz(msg: PointCloud2) -> np.ndarray:
    """Decode an XYZ(+padding) PointCloud2 into an (N, 3) float32 array."""
    if msg.width * msg.height == 0:
        return np.empty((0, 3), dtype=np.float32)
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    raw = raw.reshape(msg.height * msg.width, msg.point_step)
    xyz = raw[:, 0:12].copy().view(np.float32).reshape(-1, 3)
    return np.nan_to_num(xyz, nan=0.0)


class DebugVizNode(Node):
    """Subscribes to pipeline viz topics and renders them in viser."""

    def __init__(self) -> None:
        super().__init__('debug_viz')
        self.declare_parameter('viser_port', 8080)
        self.declare_parameter('segmented_point_cloud_topic', '/graspgen/segmented_object')
        self.declare_parameter('background_point_cloud_topic', '/graspgen/background')
        self.declare_parameter('grasp_poses_topic', '/graspgen/grasp_poses')
        self.declare_parameter('tsdf_voxels_topic', '/curobo/tsdf_voxels')
        self.declare_parameter('grasp_frame_axes_length', 0.05)

        self._server = viser.ViserServer(port=int(self.get_parameter('viser_port').value))
        self.get_logger().info(
            f'debug_viz viser server on http://0.0.0.0:'
            f'{int(self.get_parameter("viser_port").value)}'
        )

        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.RELIABLE

        self.create_subscription(
            PointCloud2,
            str(self.get_parameter('segmented_point_cloud_topic').value),
            lambda m: self._on_cloud(m, '/segmented', (50, 220, 50)),
            qos,
        )
        self.create_subscription(
            PointCloud2,
            str(self.get_parameter('background_point_cloud_topic').value),
            lambda m: self._on_cloud(m, '/background', (140, 140, 140)),
            qos,
        )
        self.create_subscription(
            PointCloud2,
            str(self.get_parameter('tsdf_voxels_topic').value),
            lambda m: self._on_cloud(m, '/tsdf', (60, 120, 255)),
            qos,
        )
        self.create_subscription(
            PoseArray,
            str(self.get_parameter('grasp_poses_topic').value),
            self._on_grasp_poses,
            qos,
        )

    def _on_cloud(self, msg: PointCloud2, name: str, color: tuple) -> None:
        xyz = _cloud_to_xyz(msg)
        if len(xyz) == 0:
            return
        colors = np.tile(np.asarray(color, dtype=np.uint8), (len(xyz), 1))
        self._server.scene.add_point_cloud(name, points=xyz, colors=colors, point_size=0.004)

    def _on_grasp_poses(self, msg: PoseArray) -> None:
        # Clear stale grasp frames, then re-add the current ranked set.
        axes_length = float(self.get_parameter('grasp_frame_axes_length').value)
        colors = colors_by_rank(len(msg.poses))
        for i, pose in enumerate(msg.poses):
            position, wxyz = pose_to_position_wxyz(pose)
            self._server.scene.add_frame(
                f'/grasps/{i:03d}',
                wxyz=wxyz,
                position=position,
                axes_length=axes_length,
                axes_radius=axes_length * 0.1,
            )
        self.get_logger().info(
            f'debug_viz rendered {len(msg.poses)} grasp frames '
            f'(colors={len(colors)})'
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DebugVizNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest src/pipeline_orchestrator/test/test_debug_viz.py -v`
Expected: PASS (4 passed). If `viser` is importable in the container, the tests run; otherwise they skip cleanly via the import guard.

- [ ] **Step 5: Register the entry point**

In `src/pipeline_orchestrator/setup.py`, add to the `console_scripts` list:

```python
            'debug_viz = pipeline_orchestrator.debug_viz:main',
```

- [ ] **Step 6: Commit**

```bash
git add src/pipeline_orchestrator/pipeline_orchestrator/debug_viz.py \
        src/pipeline_orchestrator/setup.py \
        src/pipeline_orchestrator/test/test_debug_viz.py
git commit -m "Add debug_viz node serving clouds, grasps, and TSDF in viser

One viser server subscribing to the segmented/background clouds, the new
/graspgen/grasp_poses (colored by rank), and /curobo/tsdf_voxels. Pure
helpers (colors_by_rank, pose_to_position_wxyz, cloud decode) are unit
tested; viser/ROS wiring verified at integration.

Written By: Claude Opus 4.8"
```

---

## Task 7: Mode launch files + docs

Replace the single overloaded `planner_pipeline.launch.py` with a shared core plus two thin mode files, and update `CLAUDE.md`.

**Files:**
- Create: `src/pipeline_orchestrator/launch/pipeline_common.launch.py`
- Create: `src/pipeline_orchestrator/launch/deploy.launch.py`
- Create: `src/pipeline_orchestrator/launch/debug.launch.py`
- Delete: `src/pipeline_orchestrator/launch/planner_pipeline.launch.py`
- Modify: `.claude/CLAUDE.md`

- [ ] **Step 1: Create `pipeline_common.launch.py`**

This holds every node, with the mode-varying toggles exposed as launch arguments (`enable_motion_execution`, `start_graspgen_server`, `enable_viz`, `publish_grasp_poses`, `enable_debug_viz`). All other params keep their current defaults. Create `src/pipeline_orchestrator/launch/pipeline_common.launch.py`:

```python
"""Shared pipeline node graph, parameterized by mode toggles.

deploy.launch.py / debug.launch.py include this file and set the toggles;
neither requires the user to pass any argument.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    enable_motion_execution = LaunchConfiguration("enable_motion_execution")
    start_graspgen_server = LaunchConfiguration("start_graspgen_server")
    enable_viz = LaunchConfiguration("enable_viz")
    publish_grasp_poses = LaunchConfiguration("publish_grasp_poses")
    enable_debug_viz = LaunchConfiguration("enable_debug_viz")
    graspgen_port = LaunchConfiguration("graspgen_port")

    # Stable defaults (previously all individually exposed; now internal).
    use_sim_time = True
    segmentation_service_name = "/segmentation/segment_prompt"
    graspgen_service_name = "/graspgen/infer"
    curobo_service_name = "/curobo/plan_trajectory"
    rgb_topic = "/wrist_camera/wrist_camera/color/image_raw"
    depth_topic = "/wrist_camera/wrist_camera/depth/color/image_raw"
    camera_info_topic = "/wrist_camera/wrist_camera/depth/color/camera_info"
    segmented_point_cloud_topic = "/graspgen/segmented_object"
    background_point_cloud_topic = "/graspgen/background"
    grasp_poses_topic = "/graspgen/grasp_poses"
    tsdf_voxels_topic = "/curobo/tsdf_voxels"
    segmentation_output_frame = "base_link"

    return LaunchDescription(
        [
            DeclareLaunchArgument("enable_motion_execution", default_value="true"),
            DeclareLaunchArgument("start_graspgen_server", default_value="true"),
            DeclareLaunchArgument("enable_viz", default_value="false"),
            DeclareLaunchArgument("publish_grasp_poses", default_value="false"),
            DeclareLaunchArgument("enable_debug_viz", default_value="false"),
            DeclareLaunchArgument("graspgen_port", default_value="5556"),
            ExecuteProcess(
                cmd=["/start_graspgen_server.sh"],
                name="embedded_graspgen_server",
                output="screen",
                additional_env={
                    "GRASPGEN_HOST": "0.0.0.0",
                    "GRASPGEN_PORT": graspgen_port,
                },
                condition=IfCondition(start_graspgen_server),
            ),
            Node(
                package="pipeline_orchestrator",
                executable="segmentation_service",
                name="segmentation_service",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": use_sim_time,
                        "service_name": segmentation_service_name,
                        "rgb_topic": rgb_topic,
                        "depth_topic": depth_topic,
                        "camera_info_topic": camera_info_topic,
                        "segmented_point_cloud_topic": segmented_point_cloud_topic,
                        "background_point_cloud_topic": background_point_cloud_topic,
                        "output_frame": segmentation_output_frame,
                        "overlay_topic": "/segmentation/overlay",
                        "mask_topic": "/segmentation/mask",
                        "debug_dir": "/artifacts/segmentation_service",
                    }
                ],
            ),
            Node(
                package="pipeline_orchestrator",
                executable="graspgen_service",
                name="graspgen_service",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": use_sim_time,
                        "segmented_point_cloud_topic": segmented_point_cloud_topic,
                        "background_point_cloud_topic": background_point_cloud_topic,
                        "service_name": graspgen_service_name,
                        "server_host": "127.0.0.1",
                        "server_port": graspgen_port,
                        "remove_outliers": False,
                        "rank_mode": "approach_alignment",
                        "target_approach_dir": [0.0, 0.0, -1.0],
                        "expected_frame": segmentation_output_frame,
                        "debug_dir": "/artifacts/graspgen_service",
                        "publish_grasp_poses": publish_grasp_poses,
                        "grasp_poses_topic": grasp_poses_topic,
                    }
                ],
            ),
            Node(
                package="pipeline_orchestrator",
                executable="curobo_service",
                name="curobo_service",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": use_sim_time,
                        "service_name": curobo_service_name,
                        "enable_viz": enable_viz,
                        "tsdf_voxels_topic": tsdf_voxels_topic,
                    }
                ],
                condition=IfCondition(enable_motion_execution),
            ),
            Node(
                package="pipeline_orchestrator",
                executable="orchestrator",
                name="pipeline_orchestrator",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": use_sim_time,
                        "segmentation_service_name": segmentation_service_name,
                        "graspgen_service_name": graspgen_service_name,
                        "curobo_service_name": curobo_service_name,
                        "enable_motion_execution": enable_motion_execution,
                        "auto_run_on_task_command": True,
                    }
                ],
            ),
            Node(
                package="pipeline_orchestrator",
                executable="debug_viz",
                name="debug_viz",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": use_sim_time,
                        "segmented_point_cloud_topic": segmented_point_cloud_topic,
                        "background_point_cloud_topic": background_point_cloud_topic,
                        "grasp_poses_topic": grasp_poses_topic,
                        "tsdf_voxels_topic": tsdf_voxels_topic,
                    }
                ],
                condition=IfCondition(enable_debug_viz),
            ),
        ]
    )
```

- [ ] **Step 2: Create `deploy.launch.py`**

Create `src/pipeline_orchestrator/launch/deploy.launch.py`:

```python
"""Deploy mode: full pipeline, executes on the UR5, no visualization."""

import os

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory


def generate_launch_description() -> LaunchDescription:
    common = os.path.join(
        get_package_share_directory("pipeline_orchestrator"),
        "launch",
        "pipeline_common.launch.py",
    )
    return LaunchDescription(
        [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(common),
                launch_arguments={
                    "enable_motion_execution": "true",
                    "start_graspgen_server": "true",
                    "enable_viz": "false",
                    "publish_grasp_poses": "false",
                    "enable_debug_viz": "false",
                }.items(),
            )
        ]
    )
```

- [ ] **Step 3: Create `debug.launch.py`**

Create `src/pipeline_orchestrator/launch/debug.launch.py`:

```python
"""Debug mode: same pipeline as deploy (executes on the arm) plus a viser
server visualizing segmented clouds, ranked grasp poses, and the live TSDF."""

import os

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory


def generate_launch_description() -> LaunchDescription:
    common = os.path.join(
        get_package_share_directory("pipeline_orchestrator"),
        "launch",
        "pipeline_common.launch.py",
    )
    return LaunchDescription(
        [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(common),
                launch_arguments={
                    "enable_motion_execution": "true",
                    "start_graspgen_server": "true",
                    "enable_viz": "true",
                    "publish_grasp_poses": "true",
                    "enable_debug_viz": "true",
                }.items(),
            )
        ]
    )
```

- [ ] **Step 4: Delete the legacy launch file**

```bash
git rm src/pipeline_orchestrator/launch/planner_pipeline.launch.py
```

- [ ] **Step 5: Verify the launch files parse**

After a `colcon build` (run inside the container, see Task 8), both mode files must enumerate their arguments without raising:

Run: `ros2 launch pipeline_orchestrator deploy.launch.py --show-args`
Expected: lists the included arguments (e.g. `enable_motion_execution`, `start_graspgen_server`, …) and exits 0.
Run: `ros2 launch pipeline_orchestrator debug.launch.py --show-args`
Expected: same, exits 0.

- [ ] **Step 6: Update `.claude/CLAUDE.md`**

Make these edits to `.claude/CLAUDE.md`:

1. In the **Build & Run** section, replace the build/launch commands with the new one-liners:

```bash
# Build the image (heavy layers cached after first run; ~20 min first time)
docker compose build

# Deploy mode: full pipeline, executes on the UR5, no visualization
docker compose up

# Debug mode: same pipeline + viser visualization, src live-mounted
docker compose -f docker-compose.yml -f docker-compose.debug.yml up

# Interactive shell (debug override gives the live-mounted src)
docker compose -f docker-compose.yml -f docker-compose.debug.yml run --rm ai_planner bash
```

Remove the stale lines referencing `planner_pipeline.launch.py`,
`enable_motion_execution:=true`, and the `scripts/build.sh` rebuild recipe.

2. In the **Package Architecture** node table, add a row:

```
| `debug_viz.py` | `debug_viz` | Hosts one viser server; subscribes to the segmented/background clouds, `/graspgen/grasp_poses`, and `/curobo/tsdf_voxels` and renders them. Debug mode only. |
```

3. In the **Internal** topics table, add:

```
| `/graspgen/grasp_poses` | `geometry_msgs/PoseArray` | graspgen_service → debug_viz (debug only) |
| `/curobo/tsdf_voxels` | `sensor_msgs/PointCloud2` | curobo_service → debug_viz (debug only) |
```

4. In **Key Files**, replace the `Dockerfile` / `Dockerfile.base` / `docker-compose*.yml` bullets with:

```
- `Dockerfile` — single layer-ordered image (CUDA 12.8 + ROS2 Humble + PyTorch + SAM2/GraspGen/cuRobo + baked models; `COPY src` last). No separate base image.
- `docker-compose.yml` — deploy mode (baked image, runs `deploy.launch.py`).
- `docker-compose.debug.yml` — override that live-mounts `./src`/`./scripts`/`./config` and runs `debug.launch.py`.
- `src/pipeline_orchestrator/launch/{pipeline_common,deploy,debug}.launch.py` — shared node graph + the two mode entry points.
```

- [ ] **Step 7: Commit**

```bash
git add src/pipeline_orchestrator/launch/ .claude/CLAUDE.md
git commit -m "Split launch into deploy/debug over a shared core; refresh CLAUDE.md

pipeline_common.launch.py holds the node graph behind mode toggles;
deploy.launch.py and debug.launch.py set them (debug adds debug_viz +
grasp/TSDF publishing). Removes the overloaded planner_pipeline.launch.py
and documents the new one-command workflow.

Written By: Claude Opus 4.8"
```

---

## Task 8: Full integration verification

**Files:** none (verification only)

- [ ] **Step 1: Build the merged image**

Run: `docker compose build`
Expected: completes successfully. On a second run after touching only a `src` file, the heavy layers report `CACHED` and only the `COPY src` + `colcon build` layers re-run.

- [ ] **Step 2: Run the full unit-test suite inside the container**

Run:
```bash
docker compose -f docker-compose.yml -f docker-compose.debug.yml run --rm ai_planner \
  bash -lc "cd /ros2_ws && colcon build --symlink-install && \
            python3 -m pytest src/pipeline_orchestrator/test -v"
```
Expected: all tests pass (or skip cleanly where a heavy runtime dep is absent). No errors/failures.

- [ ] **Step 3: Verify both launch files enumerate arguments**

Run:
```bash
docker compose -f docker-compose.yml -f docker-compose.debug.yml run --rm ai_planner \
  bash -lc "ros2 launch pipeline_orchestrator deploy.launch.py --show-args && \
            ros2 launch pipeline_orchestrator debug.launch.py --show-args"
```
Expected: both print their arguments and exit 0.

- [ ] **Step 4: Smoke-test debug mode against the running sim (manual)**

With the `manip_challenge` sim up (see the Gazebo launch recipe in memory), run:
`docker compose -f docker-compose.yml -f docker-compose.debug.yml up`
Then open the viser URL printed by `debug_viz` (default `http://0.0.0.0:8080`) and publish a task command. Confirm the viser scene shows: segmented + background clouds, ranked grasp frames (green→red by rank), and the TSDF voxel cloud. Confirm the arm executes (debug also executes).

- [ ] **Step 5: Final no-op commit check**

Run: `git status`
Expected: clean working tree (all changes already committed across Tasks 1–7).

---

## Notes for the implementer

- **DRY:** the pose/quaternion conversion has a single home in `pipeline_utils` (Task 3); do not reintroduce a copy in `graspgen_service` or `debug_viz`.
- **Gating discipline:** `publish_grasp_poses` and `enable_viz` default to `false`; only `debug.launch.py` turns them on. Deploy mode publishes nothing extra.
- **Launch-file edits in debug:** because launch files are loaded from the colcon `install/share` tree (not `src`), editing a `*.launch.py` while live-mounted requires a `colcon build` inside the container before relaunch. Python node edits propagate live via the entrypoint's `PYTHONPATH`.
- **ASCII-only configs:** do not introduce non-ASCII bytes into `config/ur5_curobo.yml` (cuRobo's `load_yaml` uses the container's ASCII codec).
```
