# ros2-ai-planner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scaffold a new standalone repo at `~/dev/cs477_ws/ros2-ai-planner/` that runs as a Dockerized external ROS2 node implementing a SAM2 → GraspGen → cuRobo perception-to-action pipeline for the manip_challenge system.

**Architecture:** Single Docker container (CUDA 12.8 + ROS2 Humble) using host networking to transparently share ROS2 topics with the host machine. Four ROS2 packages: three AI stub nodes (sam2_node, graspgen_node, curobo_node) each exposing an internal service, and a pipeline_orchestrator that owns all external topic I/O and chains the three services.

**Tech Stack:** nvidia/cuda:12.8.0-cudnn9-devel-ubuntu22.04, ROS2 Humble, PyTorch (CUDA 12.8 wheels), colcon, docker compose with nvidia runtime.

---

## File Map

```
~/dev/cs477_ws/ros2-ai-planner/
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── scripts/
│   ├── entrypoint.sh          # source ROS2 + workspace, exec "$@"
│   ├── build.sh               # colcon build --symlink-install
│   └── run.sh                 # docker compose up
├── requirements/
│   ├── sam2.txt               # placeholder deps for SAM2
│   ├── graspgen.txt           # placeholder deps for GraspGen
│   └── curobo.txt             # placeholder deps for cuRobo
└── src/
    ├── sam2_node/
    │   ├── package.xml
    │   ├── setup.py
    │   ├── setup.cfg
    │   ├── sam2_node/
    │   │   ├── __init__.py
    │   │   └── sam2_node.py   # service server stub: ~/segment
    │   └── test/
    │       └── test_sam2_node.py
    ├── graspgen_node/
    │   ├── package.xml
    │   ├── setup.py
    │   ├── setup.cfg
    │   ├── graspgen_node/
    │   │   ├── __init__.py
    │   │   └── graspgen_node.py  # service server stub: ~/generate_grasp
    │   └── test/
    │       └── test_graspgen_node.py
    ├── curobo_node/
    │   ├── package.xml
    │   ├── setup.py
    │   ├── setup.cfg
    │   ├── curobo_node/
    │   │   ├── __init__.py
    │   │   └── curobo_node.py    # service server stub: ~/plan_trajectory
    │   └── test/
    │       └── test_curobo_node.py
    └── pipeline_orchestrator/
        ├── package.xml
        ├── setup.py
        ├── setup.cfg
        ├── pipeline_orchestrator/
        │   ├── __init__.py
        │   └── orchestrator.py   # subscribes to /task_commands, calls 3 services
        └── test/
            └── test_orchestrator.py
```

---

## Task 1: Initialize the repo

**Files:**
- Create: `~/dev/cs477_ws/ros2-ai-planner/.gitignore`

- [ ] **Step 1: Create the directory and initialize git**

```bash
mkdir -p ~/dev/cs477_ws/ros2-ai-planner
cd ~/dev/cs477_ws/ros2-ai-planner
git init
```

Expected output: `Initialized empty Git repository in .../ros2-ai-planner/.git/`

- [ ] **Step 2: Create `.gitignore`**

```
# ROS2 build artifacts
build/
install/
log/

# Python
__pycache__/
*.py[cod]
*.egg-info/
.eggs/

# Docker
.dockerignore

# Editor
.vscode/
.idea/
*.swp
```

- [ ] **Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore: init repo"
```

---

## Task 2: Dockerfile

**Files:**
- Create: `~/dev/cs477_ws/ros2-ai-planner/Dockerfile`
- Create: `~/dev/cs477_ws/ros2-ai-planner/scripts/entrypoint.sh`

- [ ] **Step 1: Create `Dockerfile`**

```dockerfile
FROM nvidia/cuda:12.8.0-cudnn9-devel-ubuntu22.04

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

- [ ] **Step 2: Create `scripts/entrypoint.sh`**

```bash
#!/bin/bash
set -e
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash
exec "$@"
```

- [ ] **Step 3: Commit**

```bash
git add Dockerfile scripts/entrypoint.sh
git commit -m "feat: add Dockerfile with CUDA 12.8 and ROS2 Humble"
```

---

## Task 3: docker-compose, scripts, and requirements

**Files:**
- Create: `~/dev/cs477_ws/ros2-ai-planner/docker-compose.yml`
- Create: `~/dev/cs477_ws/ros2-ai-planner/scripts/build.sh`
- Create: `~/dev/cs477_ws/ros2-ai-planner/scripts/run.sh`
- Create: `~/dev/cs477_ws/ros2-ai-planner/requirements/sam2.txt`
- Create: `~/dev/cs477_ws/ros2-ai-planner/requirements/graspgen.txt`
- Create: `~/dev/cs477_ws/ros2-ai-planner/requirements/curobo.txt`

- [ ] **Step 1: Create `docker-compose.yml`**

```yaml
services:
  ai_planner:
    build: .
    network_mode: host
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=all
      - ROS_DOMAIN_ID=0
    volumes:
      - ./src:/ros2_ws/src
    ipc: host
    stdin_open: true
    tty: true
```

- [ ] **Step 2: Create `scripts/build.sh`**

```bash
#!/bin/bash
set -e
source /opt/ros/humble/setup.bash
cd /ros2_ws
colcon build --symlink-install
```

- [ ] **Step 3: Create `scripts/run.sh`**

```bash
#!/bin/bash
set -e
docker compose up
```

- [ ] **Step 4: Make scripts executable**

```bash
chmod +x scripts/build.sh scripts/run.sh scripts/entrypoint.sh
```

- [ ] **Step 5: Create `requirements/sam2.txt`**

```
# SAM2 (Segment Anything Model 2) dependencies
# Ref: https://github.com/facebookresearch/segment-anything-2
# Fill in when implementing sam2_node
# torch>=2.3.1          # already installed in Dockerfile
# torchvision>=0.18.1   # already installed in Dockerfile
# hydra-core>=1.3.2
# iopath
```

- [ ] **Step 6: Create `requirements/graspgen.txt`**

```
# GraspGen dependencies
# Ref: diffusion-based grasp pose generation model
# Fill in when implementing graspgen_node
# torch>=2.3.1    # already installed in Dockerfile
# open3d
# trimesh
```

- [ ] **Step 7: Create `requirements/curobo.txt`**

```
# cuRobo dependencies
# Ref: https://curobo.org/
# Fill in when implementing curobo_node
# curobo requires CUDA 11.8+ (12.8 supported)
# Install instructions: https://curobo.org/get_started/1_install_instructions.html
# torch>=2.3.1    # already installed in Dockerfile
# curobo (install from source per official docs)
```

- [ ] **Step 8: Commit**

```bash
git add docker-compose.yml scripts/build.sh scripts/run.sh requirements/
git commit -m "feat: add docker-compose, helper scripts, and requirements stubs"
```

---

## Task 4: sam2_node package

**Files:**
- Create: `src/sam2_node/package.xml`
- Create: `src/sam2_node/setup.py`
- Create: `src/sam2_node/setup.cfg`
- Create: `src/sam2_node/sam2_node/__init__.py`
- Create: `src/sam2_node/sam2_node/sam2_node.py`
- Create: `src/sam2_node/test/test_sam2_node.py`

- [ ] **Step 1: Write the failing test first**

Create `src/sam2_node/test/test_sam2_node.py`:

```python
import pytest


def test_sam2_node_importable():
    from sam2_node.sam2_node import Sam2Node
    assert Sam2Node is not None


def test_sam2_node_has_segment_service():
    from sam2_node.sam2_node import Sam2Node
    import inspect
    assert hasattr(Sam2Node, 'segment_callback')
    assert callable(Sam2Node.segment_callback)
```

- [ ] **Step 2: Verify the test fails (import error expected)**

```bash
cd ~/dev/cs477_ws/ros2-ai-planner
python3 -m pytest src/sam2_node/test/test_sam2_node.py -v
```

Expected: `ModuleNotFoundError: No module named 'sam2_node'`

- [ ] **Step 3: Create `src/sam2_node/package.xml`**

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>sam2_node</name>
  <version>0.1.0</version>
  <description>SAM2 segmentation node stub for the AI planner pipeline.</description>
  <maintainer email="you@example.com">Your Name</maintainer>
  <license>MIT</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>sensor_msgs</depend>
  <depend>std_srvs</depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

- [ ] **Step 4: Create `src/sam2_node/setup.cfg`**

```ini
[develop]
script_dir=$base/lib/sam2_node

[install]
install_scripts=$base/lib/sam2_node
```

- [ ] **Step 5: Create `src/sam2_node/setup.py`**

```python
from setuptools import find_packages, setup

package_name = 'sam2_node'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='you@example.com',
    description='SAM2 segmentation node stub.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sam2_node = sam2_node.sam2_node:main',
        ],
    },
)
```

- [ ] **Step 6: Create the resource marker file (required by ament_python)**

```bash
mkdir -p src/sam2_node/resource
touch src/sam2_node/resource/sam2_node
```

- [ ] **Step 7: Create `src/sam2_node/sam2_node/__init__.py`**

```python
```

(empty file)

- [ ] **Step 8: Create `src/sam2_node/sam2_node/sam2_node.py`**

```python
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger


class Sam2Node(Node):
    """Stub node for SAM2 segmentation.

    Exposes service ~/segment.
    TODO: Replace Trigger with a custom service type that accepts
          (sensor_msgs/Image rgb, string prompt) and returns segmentation masks.
    """

    def __init__(self):
        super().__init__('sam2_node')
        self.srv = self.create_service(
            Trigger,
            '~/segment',
            self.segment_callback,
        )
        self.get_logger().info('sam2_node ready (stub).')

    def segment_callback(self, request, response):
        # TODO: load SAM2 model and run inference on RGB image + text prompt
        # Inputs:  request.rgb   (sensor_msgs/Image)
        #          request.prompt (string)
        # Outputs: response.masks (segmentation masks)
        self.get_logger().warn('SAM2 not yet implemented.')
        response.success = False
        response.message = 'SAM2 not yet implemented'
        return response


def main(args=None):
    rclpy.init(args=args)
    node = Sam2Node()
    rclpy.spin(node)
    rclpy.shutdown()
```

- [ ] **Step 9: Run tests — expect PASS**

```bash
cd ~/dev/cs477_ws/ros2-ai-planner
python3 -m pytest src/sam2_node/test/test_sam2_node.py -v
```

Expected:
```
test_sam2_node_importable PASSED
test_sam2_node_has_segment_service PASSED
```

- [ ] **Step 10: Commit**

```bash
git add src/sam2_node/
git commit -m "feat: add sam2_node stub package"
```

---

## Task 5: graspgen_node package

**Files:**
- Create: `src/graspgen_node/package.xml`
- Create: `src/graspgen_node/setup.py`
- Create: `src/graspgen_node/setup.cfg`
- Create: `src/graspgen_node/graspgen_node/__init__.py`
- Create: `src/graspgen_node/graspgen_node/graspgen_node.py`
- Create: `src/graspgen_node/test/test_graspgen_node.py`

- [ ] **Step 1: Write the failing test**

Create `src/graspgen_node/test/test_graspgen_node.py`:

```python
import pytest


def test_graspgen_node_importable():
    from graspgen_node.graspgen_node import GraspGenNode
    assert GraspGenNode is not None


def test_graspgen_node_has_generate_grasp_callback():
    from graspgen_node.graspgen_node import GraspGenNode
    assert hasattr(GraspGenNode, 'generate_grasp_callback')
    assert callable(GraspGenNode.generate_grasp_callback)
```

- [ ] **Step 2: Verify the test fails**

```bash
python3 -m pytest src/graspgen_node/test/test_graspgen_node.py -v
```

Expected: `ModuleNotFoundError: No module named 'graspgen_node'`

- [ ] **Step 3: Create `src/graspgen_node/package.xml`**

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>graspgen_node</name>
  <version>0.1.0</version>
  <description>GraspGen grasp pose generation node stub for the AI planner pipeline.</description>
  <maintainer email="you@example.com">Your Name</maintainer>
  <license>MIT</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>sensor_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>std_srvs</depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

- [ ] **Step 4: Create `src/graspgen_node/setup.cfg`**

```ini
[develop]
script_dir=$base/lib/graspgen_node

[install]
install_scripts=$base/lib/graspgen_node
```

- [ ] **Step 5: Create `src/graspgen_node/setup.py`**

```python
from setuptools import find_packages, setup

package_name = 'graspgen_node'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='you@example.com',
    description='GraspGen grasp pose generation node stub.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'graspgen_node = graspgen_node.graspgen_node:main',
        ],
    },
)
```

- [ ] **Step 6: Create the resource marker**

```bash
mkdir -p src/graspgen_node/resource
touch src/graspgen_node/resource/graspgen_node
```

- [ ] **Step 7: Create `src/graspgen_node/graspgen_node/__init__.py`**

```python
```

(empty file)

- [ ] **Step 8: Create `src/graspgen_node/graspgen_node/graspgen_node.py`**

```python
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger


class GraspGenNode(Node):
    """Stub node for GraspGen grasp pose generation.

    Exposes service ~/generate_grasp.
    TODO: Replace Trigger with a custom service type that accepts
          (segmentation masks, sensor_msgs/Image depth) and returns
          geometry_msgs/PoseStamped grasp pose.
    """

    def __init__(self):
        super().__init__('graspgen_node')
        self.srv = self.create_service(
            Trigger,
            '~/generate_grasp',
            self.generate_grasp_callback,
        )
        self.get_logger().info('graspgen_node ready (stub).')

    def generate_grasp_callback(self, request, response):
        # TODO: run GraspGen diffusion model on masks + depth image
        # Inputs:  request.masks (segmentation masks from sam2_node)
        #          request.depth (sensor_msgs/Image)
        # Outputs: response.grasp_pose (geometry_msgs/PoseStamped)
        self.get_logger().warn('GraspGen not yet implemented.')
        response.success = False
        response.message = 'GraspGen not yet implemented'
        return response


def main(args=None):
    rclpy.init(args=args)
    node = GraspGenNode()
    rclpy.spin(node)
    rclpy.shutdown()
```

- [ ] **Step 9: Run tests — expect PASS**

```bash
python3 -m pytest src/graspgen_node/test/test_graspgen_node.py -v
```

Expected:
```
test_graspgen_node_importable PASSED
test_graspgen_node_has_generate_grasp_callback PASSED
```

- [ ] **Step 10: Commit**

```bash
git add src/graspgen_node/
git commit -m "feat: add graspgen_node stub package"
```

---

## Task 6: curobo_node package

**Files:**
- Create: `src/curobo_node/package.xml`
- Create: `src/curobo_node/setup.py`
- Create: `src/curobo_node/setup.cfg`
- Create: `src/curobo_node/curobo_node/__init__.py`
- Create: `src/curobo_node/curobo_node/curobo_node.py`
- Create: `src/curobo_node/test/test_curobo_node.py`

- [ ] **Step 1: Write the failing test**

Create `src/curobo_node/test/test_curobo_node.py`:

```python
import pytest


def test_curobo_node_importable():
    from curobo_node.curobo_node import CuRoboNode
    assert CuRoboNode is not None


def test_curobo_node_has_plan_trajectory_callback():
    from curobo_node.curobo_node import CuRoboNode
    assert hasattr(CuRoboNode, 'plan_trajectory_callback')
    assert callable(CuRoboNode.plan_trajectory_callback)
```

- [ ] **Step 2: Verify the test fails**

```bash
python3 -m pytest src/curobo_node/test/test_curobo_node.py -v
```

Expected: `ModuleNotFoundError: No module named 'curobo_node'`

- [ ] **Step 3: Create `src/curobo_node/package.xml`**

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>curobo_node</name>
  <version>0.1.0</version>
  <description>cuRobo motion planning node stub for the AI planner pipeline.</description>
  <maintainer email="you@example.com">Your Name</maintainer>
  <license>MIT</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>sensor_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>trajectory_msgs</depend>
  <depend>std_srvs</depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

- [ ] **Step 4: Create `src/curobo_node/setup.cfg`**

```ini
[develop]
script_dir=$base/lib/curobo_node

[install]
install_scripts=$base/lib/curobo_node
```

- [ ] **Step 5: Create `src/curobo_node/setup.py`**

```python
from setuptools import find_packages, setup

package_name = 'curobo_node'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='you@example.com',
    description='cuRobo motion planning node stub.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'curobo_node = curobo_node.curobo_node:main',
        ],
    },
)
```

- [ ] **Step 6: Create the resource marker**

```bash
mkdir -p src/curobo_node/resource
touch src/curobo_node/resource/curobo_node
```

- [ ] **Step 7: Create `src/curobo_node/curobo_node/__init__.py`**

```python
```

(empty file)

- [ ] **Step 8: Create `src/curobo_node/curobo_node/curobo_node.py`**

```python
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger


class CuRoboNode(Node):
    """Stub node for cuRobo motion planning.

    Exposes service ~/plan_trajectory.
    TODO: Replace Trigger with a custom service type that accepts
          (geometry_msgs/PoseStamped grasp_pose, sensor_msgs/JointState joint_states)
          and returns trajectory_msgs/JointTrajectory.
    """

    def __init__(self):
        super().__init__('curobo_node')
        self.srv = self.create_service(
            Trigger,
            '~/plan_trajectory',
            self.plan_trajectory_callback,
        )
        self.get_logger().info('curobo_node ready (stub).')

    def plan_trajectory_callback(self, request, response):
        # TODO: run cuRobo motion planner
        # Inputs:  request.grasp_pose   (geometry_msgs/PoseStamped)
        #          request.joint_states (sensor_msgs/JointState)
        # Outputs: response.trajectory  (trajectory_msgs/JointTrajectory)
        self.get_logger().warn('cuRobo not yet implemented.')
        response.success = False
        response.message = 'cuRobo not yet implemented'
        return response


def main(args=None):
    rclpy.init(args=args)
    node = CuRoboNode()
    rclpy.spin(node)
    rclpy.shutdown()
```

- [ ] **Step 9: Run tests — expect PASS**

```bash
python3 -m pytest src/curobo_node/test/test_curobo_node.py -v
```

Expected:
```
test_curobo_node_importable PASSED
test_curobo_node_has_plan_trajectory_callback PASSED
```

- [ ] **Step 10: Commit**

```bash
git add src/curobo_node/
git commit -m "feat: add curobo_node stub package"
```

---

## Task 7: pipeline_orchestrator package

**Files:**
- Create: `src/pipeline_orchestrator/package.xml`
- Create: `src/pipeline_orchestrator/setup.py`
- Create: `src/pipeline_orchestrator/setup.cfg`
- Create: `src/pipeline_orchestrator/pipeline_orchestrator/__init__.py`
- Create: `src/pipeline_orchestrator/pipeline_orchestrator/orchestrator.py`
- Create: `src/pipeline_orchestrator/test/test_orchestrator.py`

- [ ] **Step 1: Write the failing test**

Create `src/pipeline_orchestrator/test/test_orchestrator.py`:

```python
import pytest


def test_orchestrator_importable():
    from pipeline_orchestrator.orchestrator import PipelineOrchestrator
    assert PipelineOrchestrator is not None


def test_orchestrator_has_task_command_callback():
    from pipeline_orchestrator.orchestrator import PipelineOrchestrator
    assert hasattr(PipelineOrchestrator, 'task_command_callback')
    assert callable(PipelineOrchestrator.task_command_callback)


def test_orchestrator_has_pipeline_methods():
    from pipeline_orchestrator.orchestrator import PipelineOrchestrator
    assert hasattr(PipelineOrchestrator, 'call_sam2')
    assert hasattr(PipelineOrchestrator, 'call_graspgen')
    assert hasattr(PipelineOrchestrator, 'call_curobo')
```

- [ ] **Step 2: Verify the test fails**

```bash
python3 -m pytest src/pipeline_orchestrator/test/test_orchestrator.py -v
```

Expected: `ModuleNotFoundError: No module named 'pipeline_orchestrator'`

- [ ] **Step 3: Create `src/pipeline_orchestrator/package.xml`**

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>pipeline_orchestrator</name>
  <version>0.1.0</version>
  <description>Orchestrates the SAM2 → GraspGen → cuRobo planning pipeline.</description>
  <maintainer email="you@example.com">Your Name</maintainer>
  <license>MIT</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>sensor_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>trajectory_msgs</depend>
  <depend>control_msgs</depend>
  <depend>std_srvs</depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

- [ ] **Step 4: Create `src/pipeline_orchestrator/setup.cfg`**

```ini
[develop]
script_dir=$base/lib/pipeline_orchestrator

[install]
install_scripts=$base/lib/pipeline_orchestrator
```

- [ ] **Step 5: Create `src/pipeline_orchestrator/setup.py`**

```python
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
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='you@example.com',
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

- [ ] **Step 6: Create the resource marker**

```bash
mkdir -p src/pipeline_orchestrator/resource
touch src/pipeline_orchestrator/resource/pipeline_orchestrator
```

- [ ] **Step 7: Create `src/pipeline_orchestrator/pipeline_orchestrator/__init__.py`**

```python
```

(empty file)

- [ ] **Step 8: Create `src/pipeline_orchestrator/pipeline_orchestrator/orchestrator.py`**

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image, JointState
from std_srvs.srv import Trigger


class PipelineOrchestrator(Node):
    """Orchestrates the full SAM2 → GraspGen → cuRobo pipeline.

    External subscriptions (from manip_challenge / Gazebo):
      /task_commands              std_msgs/String
      /wrist_camera/.../color/image_raw  sensor_msgs/Image
      /wrist_camera/.../depth/image_raw  sensor_msgs/Image
      /joint_states               sensor_msgs/JointState

    Internal service clients:
      sam2_node/segment           std_srvs/Trigger  (TODO: custom type)
      graspgen_node/generate_grasp std_srvs/Trigger (TODO: custom type)
      curobo_node/plan_trajectory  std_srvs/Trigger (TODO: custom type)

    Sends planned trajectory to:
      /ur5_controller/follow_joint_trajectory  (action, control_msgs)
    """

    CAMERA_RGB_TOPIC = '/wrist_camera/wrist_camera/color/image_raw'
    CAMERA_DEPTH_TOPIC = '/wrist_camera/wrist_camera/depth/image_raw'
    JOINT_STATES_TOPIC = '/joint_states'
    TASK_COMMANDS_TOPIC = '/task_commands'

    def __init__(self):
        super().__init__('pipeline_orchestrator')

        # External subscribers
        self.task_sub = self.create_subscription(
            String, self.TASK_COMMANDS_TOPIC, self.task_command_callback, 10)
        self.rgb_sub = self.create_subscription(
            Image, self.CAMERA_RGB_TOPIC, self._cache_rgb, 10)
        self.depth_sub = self.create_subscription(
            Image, self.CAMERA_DEPTH_TOPIC, self._cache_depth, 10)
        self.joint_sub = self.create_subscription(
            JointState, self.JOINT_STATES_TOPIC, self._cache_joints, 10)

        # Internal service clients
        self.sam2_client = self.create_client(Trigger, '/sam2_node/segment')
        self.graspgen_client = self.create_client(Trigger, '/graspgen_node/generate_grasp')
        self.curobo_client = self.create_client(Trigger, '/curobo_node/plan_trajectory')

        # Cached sensor data
        self._latest_rgb = None
        self._latest_depth = None
        self._latest_joints = None

        self.get_logger().info('pipeline_orchestrator ready (stub).')

    def _cache_rgb(self, msg):
        self._latest_rgb = msg

    def _cache_depth(self, msg):
        self._latest_depth = msg

    def _cache_joints(self, msg):
        self._latest_joints = msg

    def task_command_callback(self, msg):
        self.get_logger().info(f'Received task command: {msg.data}')
        # TODO: parse task command, then run pipeline
        self.call_sam2()

    def call_sam2(self):
        # TODO: send cached RGB + parsed prompt to sam2_node/segment service
        # On response: call self.call_graspgen(masks)
        self.get_logger().warn('call_sam2 not yet implemented.')

    def call_graspgen(self, masks=None):
        # TODO: send masks + cached depth to graspgen_node/generate_grasp service
        # On response: call self.call_curobo(grasp_pose)
        self.get_logger().warn('call_graspgen not yet implemented.')

    def call_curobo(self, grasp_pose=None):
        # TODO: send grasp_pose + cached joint states to curobo_node/plan_trajectory
        # On response: send trajectory to /ur5_controller/follow_joint_trajectory
        self.get_logger().warn('call_curobo not yet implemented.')


def main(args=None):
    rclpy.init(args=args)
    node = PipelineOrchestrator()
    rclpy.spin(node)
    rclpy.shutdown()
```

- [ ] **Step 9: Run tests — expect PASS**

```bash
python3 -m pytest src/pipeline_orchestrator/test/test_orchestrator.py -v
```

Expected:
```
test_orchestrator_importable PASSED
test_orchestrator_has_task_command_callback PASSED
test_orchestrator_has_pipeline_methods PASSED
```

- [ ] **Step 10: Commit**

```bash
git add src/pipeline_orchestrator/
git commit -m "feat: add pipeline_orchestrator stub package"
```

---

## Task 8: Verify Docker build

- [ ] **Step 1: Build the Docker image**

```bash
cd ~/dev/cs477_ws/ros2-ai-planner
docker compose build
```

Expected: Build completes with `FINISHED` — no errors. PyTorch install will take several minutes on first build.

- [ ] **Step 2: Verify ROS2 packages are found inside the container**

```bash
docker compose run --rm ai_planner ros2 pkg list | grep -E 'sam2|graspgen|curobo|orchestrator'
```

Expected output:
```
curobo_node
graspgen_node
pipeline_orchestrator
sam2_node
```

- [ ] **Step 3: Verify each package is importable inside the container**

```bash
docker compose run --rm ai_planner python3 -c "
from sam2_node.sam2_node import Sam2Node
from graspgen_node.graspgen_node import GraspGenNode
from curobo_node.curobo_node import CuRoboNode
from pipeline_orchestrator.orchestrator import PipelineOrchestrator
print('All packages import OK')
"
```

Expected output: `All packages import OK`

- [ ] **Step 4: Commit**

```bash
git add .
git commit -m "chore: verify docker build — all stub nodes launch cleanly"
```
