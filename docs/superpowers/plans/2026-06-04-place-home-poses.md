# Place/Home Poses Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a validated `config/place_poses.yml` holding `tool0`-in-`base_link` cartesian targets for home, storage 1, storage 2, and bookshelf floors 1 & 2, plus a `pose_probe` debug node to plan/execute/tune each pose in the live sim.

**Architecture:** A static YAML config carries the named poses + a `transit_z` safe-carry height. A pure loader/builder module (`place_pose_utils.py`, yaml + geometry_msgs only, unit-tested) parses and validates it. A thin ROS debug node (`pose_probe.py`, live-only, mirrors `graspgen_service_caller`) loads one target, plans to it via the existing `/curobo/plan_trajectory` single-pose path, optionally executes on the arm, and prints the achieved `tool0` pose for tuning. Orchestrator integration is a separate later plan.

**Tech Stack:** ROS2 Humble (rclpy, tf2_ros, control_msgs action), `riro_srvs/PlanTrajectory`, PyYAML, pytest. Runs against the live manip_challenge sim; `world == base_link` (verified via TF).

**Reference:** spec `docs/superpowers/specs/2026-06-04-place-home-poses-design.md`.

---

## File Structure

- **Create** `src/pipeline_orchestrator/config/place_poses.yml` — static named poses + `transit_z`. Installed via the existing `config/*.yml` glob in `setup.py`.
- **Create** `src/pipeline_orchestrator/pipeline_orchestrator/place_pose_utils.py` — pure loader/validator + `geometry_msgs/Pose` builder. Dependency-light (yaml + guarded geometry_msgs), mirroring `pipeline_utils.py`.
- **Create** `src/pipeline_orchestrator/test/test_place_pose_utils.py` — unit tests for the loader/builder.
- **Create** `src/pipeline_orchestrator/pipeline_orchestrator/pose_probe.py` — live-only debug node (no unit test, consistent with `graspgen_service_caller.py` / `graspgen_probe.py`).
- **Modify** `src/pipeline_orchestrator/setup.py:27-37` — add the `pose_probe` console_scripts entry point.

---

## Task 1: Create the place_poses.yml config

**Files:**
- Create: `src/pipeline_orchestrator/config/place_poses.yml`

- [ ] **Step 1: Write the config with computed starting-guess values**

Create `src/pipeline_orchestrator/config/place_poses.yml`:

```yaml
# Hardcoded place/home poses for the manip_challenge environment.
#
# All poses are the tool0 link expressed in base_link (world == base_link,
# verified via TF). tool0 +z is the approach axis; the gripper fingertips are
# +0.1034 m along +z, so a tool0 target sits 0.1034 m behind the fingertip
# contact point along the approach axis.
#
#   quat_xyzw [1, 0, 0, 0]        -> tool0 +z points down (world -z): top-down
#                                    drops / wrist-cam home.
#   quat_xyzw [0.5, 0.5, 0.5, 0.5] -> tool0 +z points +x (world): horizontal,
#                                    pointing into the bookshelf opening.
#
# Values below are STARTING GUESSES. Tune them in the live sim with the
# pose_probe node (see docs/superpowers/plans/2026-06-04-place-home-poses.md).

transit_z: 0.80          # safe height for carrying an object laterally

home:      {xyz: [0.55, 0.07, 0.90], quat_xyzw: [1.0, 0.0, 0.0, 0.0]}
storage_1: {xyz: [0.0,  0.55, 0.85], quat_xyzw: [1.0, 0.0, 0.0, 0.0]}
storage_2: {xyz: [0.0, -0.55, 0.85], quat_xyzw: [1.0, 0.0, 0.0, 0.0]}

bookshelf_floor1:
  pre_insert:      {xyz: [0.60, -0.30, 0.55], quat_xyzw: [0.5, 0.5, 0.5, 0.5]}
  insert_depth_m:  0.22
  retract_depth_m: 0.22

bookshelf_floor2:
  pre_insert:      {xyz: [0.60, -0.30, 0.76], quat_xyzw: [0.5, 0.5, 0.5, 0.5]}
  insert_depth_m:  0.22
  retract_depth_m: 0.22
```

- [ ] **Step 2: Verify it is valid YAML**

Run: `python3 -c "import yaml; print(sorted(yaml.safe_load(open('src/pipeline_orchestrator/config/place_poses.yml'))))"`
Expected: `['bookshelf_floor1', 'bookshelf_floor2', 'home', 'storage_1', 'storage_2', 'transit_z']`

- [ ] **Step 3: Commit**

```bash
git add src/pipeline_orchestrator/config/place_poses.yml
git commit -m "Add place_poses.yml starting-guess targets

Written By: Claude Opus 4.8"
```

---

## Task 2: place_pose_utils loader + Pose builder (TDD)

**Files:**
- Create: `src/pipeline_orchestrator/pipeline_orchestrator/place_pose_utils.py`
- Test: `src/pipeline_orchestrator/test/test_place_pose_utils.py`

- [ ] **Step 1: Write the failing tests**

Create `src/pipeline_orchestrator/test/test_place_pose_utils.py`:

```python
import textwrap

import pytest

from pipeline_orchestrator.place_pose_utils import (
    load_place_poses,
    pose_from_xyzquat,
    resolve_target_pose,
)

_YAML = textwrap.dedent("""
    transit_z: 0.80
    home:      {xyz: [0.55, 0.07, 0.90], quat_xyzw: [1.0, 0.0, 0.0, 0.0]}
    storage_1: {xyz: [0.0,  0.55, 0.85], quat_xyzw: [1.0, 0.0, 0.0, 0.0]}
    storage_2: {xyz: [0.0, -0.55, 0.85], quat_xyzw: [1.0, 0.0, 0.0, 0.0]}
    bookshelf_floor1:
      pre_insert:      {xyz: [0.60, -0.30, 0.55], quat_xyzw: [0.5, 0.5, 0.5, 0.5]}
      insert_depth_m:  0.22
      retract_depth_m: 0.22
    bookshelf_floor2:
      pre_insert:      {xyz: [0.60, -0.30, 0.76], quat_xyzw: [0.5, 0.5, 0.5, 0.5]}
      insert_depth_m:  0.22
      retract_depth_m: 0.22
""")


def _write(tmp_path, text):
    path = tmp_path / "place_poses.yml"
    path.write_text(text)
    return path


def test_load_parses_transit_and_simple_target(tmp_path):
    data = load_place_poses(_write(tmp_path, _YAML))
    assert data["transit_z"] == pytest.approx(0.80)
    assert data["storage_1"]["xyz"] == [0.0, 0.55, 0.85]
    assert data["storage_1"]["quat_xyzw"] == [1.0, 0.0, 0.0, 0.0]


def test_load_parses_bookshelf(tmp_path):
    data = load_place_poses(_write(tmp_path, _YAML))
    shelf = data["bookshelf_floor2"]
    assert shelf["pre_insert"]["xyz"] == [0.60, -0.30, 0.76]
    assert shelf["insert_depth_m"] == pytest.approx(0.22)


def test_load_rejects_missing_transit_z(tmp_path):
    bad = _YAML.replace("transit_z: 0.80\n", "")
    with pytest.raises(ValueError):
        load_place_poses(_write(tmp_path, bad))


def test_load_rejects_bad_xyz_length(tmp_path):
    bad = _YAML.replace("xyz: [0.55, 0.07, 0.90]", "xyz: [0.55, 0.07]")
    with pytest.raises(ValueError):
        load_place_poses(_write(tmp_path, bad))


def test_pose_from_xyzquat_sets_fields():
    pose = pose_from_xyzquat([0.1, 0.2, 0.3], [0.0, 0.0, 0.0, 1.0])
    assert (pose.position.x, pose.position.y, pose.position.z) == pytest.approx(
        (0.1, 0.2, 0.3))
    assert (pose.orientation.x, pose.orientation.y,
            pose.orientation.z, pose.orientation.w) == pytest.approx(
        (0.0, 0.0, 0.0, 1.0))


def test_resolve_simple_target_returns_pose(tmp_path):
    data = load_place_poses(_write(tmp_path, _YAML))
    pose = resolve_target_pose(data, "storage_1")
    assert pose.position.y == pytest.approx(0.55)


def test_resolve_bookshelf_returns_pre_insert(tmp_path):
    data = load_place_poses(_write(tmp_path, _YAML))
    pose = resolve_target_pose(data, "bookshelf_floor1")
    assert pose.position.z == pytest.approx(0.55)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd src/pipeline_orchestrator && python3 -m pytest test/test_place_pose_utils.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'pipeline_orchestrator.place_pose_utils'`

- [ ] **Step 3: Write the implementation**

Create `src/pipeline_orchestrator/pipeline_orchestrator/place_pose_utils.py`:

```python
"""Loader, validator, and Pose builder for the hardcoded place/home poses.

Reads config/place_poses.yml. Kept dependency-light (yaml + a guarded
geometry_msgs import) so it can be unit-tested without a live ROS graph,
mirroring pipeline_utils.py.
"""

from pathlib import Path

import yaml

try:  # pragma: no cover - geometry_msgs only present in the ROS runtime
    from geometry_msgs.msg import Pose
except ImportError:  # pragma: no cover - import-only test fallback
    Pose = None


SIMPLE_TARGETS = ("home", "storage_1", "storage_2")
BOOKSHELF_TARGETS = ("bookshelf_floor1", "bookshelf_floor2")


def load_place_poses(path) -> dict:
    """Parse place_poses.yml into a plain dict, validating its structure.

    Raises ValueError if a required key is missing or malformed.
    """
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"place_poses file is not a mapping: {path}")
    if not isinstance(data.get("transit_z"), (int, float)):
        raise ValueError("place_poses: 'transit_z' must be a number")
    for name in SIMPLE_TARGETS:
        _validate_xyzquat(data.get(name), name)
    for name in BOOKSHELF_TARGETS:
        shelf = data.get(name)
        if not isinstance(shelf, dict):
            raise ValueError(f"place_poses: missing/invalid '{name}'")
        _validate_xyzquat(shelf.get("pre_insert"), f"{name}.pre_insert")
        for key in ("insert_depth_m", "retract_depth_m"):
            if not isinstance(shelf.get(key), (int, float)):
                raise ValueError(f"place_poses: '{name}.{key}' must be a number")
    return data


def _validate_xyzquat(entry, label) -> None:
    if not isinstance(entry, dict):
        raise ValueError(f"place_poses: '{label}' must be a mapping")
    xyz = entry.get("xyz")
    quat = entry.get("quat_xyzw")
    if not (isinstance(xyz, list) and len(xyz) == 3):
        raise ValueError(f"place_poses: '{label}.xyz' must be a 3-list")
    if not (isinstance(quat, list) and len(quat) == 4):
        raise ValueError(f"place_poses: '{label}.quat_xyzw' must be a 4-list")


def pose_from_xyzquat(xyz, quat_xyzw):
    """Build a geometry_msgs/Pose from xyz + (x, y, z, w) quaternion.

    Returns None when geometry_msgs is unavailable (import-only test fallback).
    """
    if Pose is None:
        return None
    pose = Pose()
    pose.position.x = float(xyz[0])
    pose.position.y = float(xyz[1])
    pose.position.z = float(xyz[2])
    pose.orientation.x = float(quat_xyzw[0])
    pose.orientation.y = float(quat_xyzw[1])
    pose.orientation.z = float(quat_xyzw[2])
    pose.orientation.w = float(quat_xyzw[3])
    return pose


def resolve_target_pose(data, target):
    """Return the geometry_msgs/Pose for a target name.

    Bookshelf targets resolve to their pre_insert pose. Raises KeyError for an
    unknown target name.
    """
    if target in BOOKSHELF_TARGETS:
        entry = data[target]["pre_insert"]
    elif target in SIMPLE_TARGETS:
        entry = data[target]
    else:
        raise KeyError(f"unknown target '{target}'")
    return pose_from_xyzquat(entry["xyz"], entry["quat_xyzw"])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd src/pipeline_orchestrator && python3 -m pytest test/test_place_pose_utils.py -q`
Expected: PASS — 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/pipeline_orchestrator/pipeline_orchestrator/place_pose_utils.py \
        src/pipeline_orchestrator/test/test_place_pose_utils.py
git commit -m "Add place_pose_utils loader/validator + Pose builder

Written By: Claude Opus 4.8"
```

---

## Task 3: pose_probe debug node + entry point

**Files:**
- Create: `src/pipeline_orchestrator/pipeline_orchestrator/pose_probe.py`
- Modify: `src/pipeline_orchestrator/setup.py:27-37`

> No unit test for this node: it requires a live cuRobo service + arm action and is validated in the sim in Task 4, consistent with `graspgen_service_caller.py` / `graspgen_probe.py`.

- [ ] **Step 1: Write the node**

Create `src/pipeline_orchestrator/pipeline_orchestrator/pose_probe.py`:

```python
"""pose_probe: validate one place/home pose against the live cuRobo planner.

Debug utility (not part of the runtime pipeline). Loads a single named target
from config/place_poses.yml, plans to it via /curobo/plan_trajectory in
single-pose mode (grasp_pose set, grasp_poses empty), optionally executes the
trajectory on the arm, and prints the achieved tool0 pose so the YAML can be
tuned in the live sim. One-shot run() mirrors graspgen_service_caller.

Run (inside the container, workspace sourced):

  ros2 run pipeline_orchestrator pose_probe --ros-args \
    -p target:=home -p execute:=false
"""

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState
from riro_srvs.srv import PlanTrajectory
import tf2_ros

from pipeline_orchestrator.place_pose_utils import (
    load_place_poses,
    resolve_target_pose,
)


class PoseProbe(Node):
    """One-shot validator for a single place/home pose."""

    def __init__(self) -> None:
        super().__init__("pose_probe")

        self.declare_parameter(
            "poses_file",
            "/ros2_ws/src/pipeline_orchestrator/config/place_poses.yml",
        )
        self.declare_parameter("target", "home")
        self.declare_parameter("execute", True)
        self.declare_parameter("service_name", "/curobo/plan_trajectory")
        self.declare_parameter(
            "arm_action_name", "/ur5_controller/follow_joint_trajectory")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("tool_frame", "tool0")
        self.declare_parameter("service_timeout_sec", 120.0)

        self._poses_file = str(self.get_parameter("poses_file").value)
        self._target = str(self.get_parameter("target").value)
        self._execute = bool(self.get_parameter("execute").value)
        self._service_name = str(self.get_parameter("service_name").value)
        self._service_timeout = float(
            self.get_parameter("service_timeout_sec").value)

        self._latest_joints = None
        self.create_subscription(
            JointState, "/joint_states", self._cache_joints, 10)
        self._client = self.create_client(PlanTrajectory, self._service_name)
        self._arm = ActionClient(
            self,
            FollowJointTrajectory,
            str(self.get_parameter("arm_action_name").value),
        )
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

    def _cache_joints(self, msg: JointState) -> None:
        self._latest_joints = msg

    def run(self) -> int:
        try:
            data = load_place_poses(self._poses_file)
            pose = resolve_target_pose(data, self._target)
        except (OSError, ValueError, KeyError) as exc:
            self.get_logger().error(f"Could not load target: {exc}")
            return 1
        if pose is None:
            self.get_logger().error("geometry_msgs unavailable; cannot build pose.")
            return 1
        self.get_logger().info(
            f"target={self._target} "
            f"xyz=({pose.position.x:.3f},{pose.position.y:.3f},"
            f"{pose.position.z:.3f}) "
            f"quat_xyzw=({pose.orientation.x:.3f},{pose.orientation.y:.3f},"
            f"{pose.orientation.z:.3f},{pose.orientation.w:.3f})")

        for _ in range(50):
            if self._latest_joints is not None:
                break
            rclpy.spin_once(self, timeout_sec=0.1)
        if self._latest_joints is None:
            self.get_logger().error("No /joint_states received.")
            return 1

        if not self._client.wait_for_service(timeout_sec=30.0):
            self.get_logger().error(f"Service unavailable: {self._service_name}")
            return 1

        request = PlanTrajectory.Request()
        request.grasp_pose = pose          # single-pose mode (grasp_poses empty)
        request.joint_state = self._latest_joints
        future = self._client.call_async(request)
        rclpy.spin_until_future_complete(
            self, future, timeout_sec=self._service_timeout)
        if not future.done() or future.result() is None:
            self.get_logger().error("plan_trajectory call failed or timed out.")
            return 1
        result = future.result()
        if not result.success:
            self.get_logger().error(f"Planning failed: {result.message}")
            return 1
        self.get_logger().info(
            f"Planned OK: {result.message} "
            f"({len(result.trajectory.points)} points)")

        if not self._execute:
            self.get_logger().info("execute=false; not moving the arm.")
            return 0
        if not self._send_trajectory(result.trajectory):
            return 1
        self._report_tool_pose()
        return 0

    def _send_trajectory(self, trajectory) -> bool:
        if not self._arm.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Arm action server unavailable.")
            return False
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory
        send_future = self._arm.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=10.0)
        handle = send_future.result()
        if handle is None or not handle.accepted:
            self.get_logger().error("Arm goal rejected by action server.")
            return False
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=60.0)
        if result_future.result() is None:
            self.get_logger().error("Arm action returned no result.")
            return False
        self.get_logger().info("Trajectory executed.")
        return True

    def _report_tool_pose(self) -> None:
        base = str(self.get_parameter("base_frame").value)
        tool = str(self.get_parameter("tool_frame").value)
        for _ in range(10):              # let TF accumulate a couple of frames
            rclpy.spin_once(self, timeout_sec=0.1)
        try:
            tf = self._tf_buffer.lookup_transform(base, tool, rclpy.time.Time())
        except Exception as exc:         # tf2 raises several exception types
            self.get_logger().warning(f"TF {base}<-{tool} lookup failed: {exc}")
            return
        t = tf.transform.translation
        q = tf.transform.rotation
        self.get_logger().info(
            f"ACHIEVED {tool} in {base}: "
            f"xyz=({t.x:.3f},{t.y:.3f},{t.z:.3f}) "
            f"quat_xyzw=({q.x:.3f},{q.y:.3f},{q.z:.3f},{q.w:.3f})")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PoseProbe()
    try:
        raise SystemExit(node.run())
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
```

- [ ] **Step 2: Add the entry point**

In `src/pipeline_orchestrator/setup.py`, add one line to the `console_scripts` list (after the `orchestrator` line):

```python
            'orchestrator = pipeline_orchestrator.orchestrator:main',
            'pose_probe = pipeline_orchestrator.pose_probe:main',
            'curobo_service = pipeline_orchestrator.curobo_service:main',
```

- [ ] **Step 3: Commit**

```bash
git add src/pipeline_orchestrator/pipeline_orchestrator/pose_probe.py \
        src/pipeline_orchestrator/setup.py
git commit -m "Add pose_probe debug node for place/home pose validation

Written By: Claude Opus 4.8"
```

---

## Task 4: Validate & tune each pose in the live sim

This task is interactive (no unit tests): plan/execute each target in the running sim, observe, and tune the YAML. The manip_challenge sim and the AI-planner debug stack must both be running (see CLAUDE.md "Debug mode" and the Gazebo launch recipe in memory). All commands run **inside the container debug shell** with the workspace sourced.

**Files:**
- Modify: `src/pipeline_orchestrator/config/place_poses.yml` (tune values in place)

- [ ] **Step 1: Build the workspace (new entry point + config)**

Inside the container (`/ros2_ws`):
Run: `colcon build --packages-select pipeline_orchestrator riro_srvs && source install/setup.bash`
Expected: build finishes; `ros2 pkg executables pipeline_orchestrator | grep pose_probe` prints `pose_probe`.

> The node defaults `poses_file` to the live-mounted source path
> (`/ros2_ws/src/pipeline_orchestrator/config/place_poses.yml`), so YAML edits in
> later steps take effect on the next run **without** rebuilding.

- [ ] **Step 2: Dry-plan `home` (no motion)**

Run: `ros2 run pipeline_orchestrator pose_probe --ros-args -p target:=home -p execute:=false`
Expected: `Planned OK: ... (<N> points)`. If it reports `Planning failed`, the pose is unreachable/in collision — note the message before tuning.

- [ ] **Step 3: Execute `home` and check the wrist-camera view**

Run: `ros2 run pipeline_orchestrator pose_probe --ros-args -p target:=home -p execute:=true`
Expected: arm moves; node prints `ACHIEVED tool0 in base_link: xyz=(...) quat_xyzw=(...)`.
Verify: the wrist camera frames the workspace basket — e.g. `ros2 run rqt_image_view rqt_image_view /wrist_camera/wrist_camera/color/image_raw`, confirming the basket at `(0.55, 0, 0.5)` is roughly centered.
Tune: if the basket is off-center or too small/large, edit `home.xyz` in `place_poses.yml` (raise/lower z for zoom; adjust x/y to recenter, remembering the wrist cam sits ~−0.07 m in tool0 y) and re-run this step.

- [ ] **Step 4: Validate `storage_1` and `storage_2`**

For each `target` in `storage_1`, `storage_2`:
Run (dry): `ros2 run pipeline_orchestrator pose_probe --ros-args -p target:=<target> -p execute:=false`
Run (exec): `ros2 run pipeline_orchestrator pose_probe --ros-args -p target:=<target> -p execute:=true`
Expected: arm hovers over the basket, gripper pointing straight down, fingertips clearing the basket rim with margin.
Tune: adjust `<target>.xyz` z (drop height) so the fingertips sit just above the rim of the basket at `(0, ±0.55, 0.6)`; nudge x/y if the EEF is not centered over the basket. Re-run until safe.

- [ ] **Step 5: Validate `bookshelf_floor1` and `bookshelf_floor2` pre_insert reach**

For each `target` in `bookshelf_floor1`, `bookshelf_floor2`:
Run (dry): `ros2 run pipeline_orchestrator pose_probe --ros-args -p target:=<target> -p execute:=false`
Run (exec): `ros2 run pipeline_orchestrator pose_probe --ros-args -p target:=<target> -p execute:=true`
Expected: arm reaches a pose in front of the shelf opening, gripper pointing horizontally (+x, into the shelf), aligned with the correct shelf board (floor1 board top ≈ z 0.51, floor2 ≈ 0.73).
Tune in `place_poses.yml`:
- `<target>.pre_insert.xyz` — x is the standoff in front of the shelf front face (≈ x 0.815); z aligns the gripper with the target board so a straight +x push lands the object on it.
- `<target>.pre_insert.quat_xyzw` — adjust only if the gripper fingers foul the shelf boards (the `[0.5, 0.5, 0.5, 0.5]` start puts tool0 +z = world +x, tool0 +y = world up).
- `<target>.insert_depth_m` — eyeball: from the achieved `pre_insert`, a straight +x push of this distance should land the fingertips well onto the board without hitting the back wall (shelf depth 0.27 m). Record the value; pose_probe does not execute the push (that is the collision-off insert handled at integration time).

- [ ] **Step 6: Sanity-check the final YAML parses**

Run: `cd src/pipeline_orchestrator && python3 -m pytest test/test_place_pose_utils.py -q`
Then: `python3 -c "from pipeline_orchestrator.place_pose_utils import load_place_poses; load_place_poses('src/pipeline_orchestrator/config/place_poses.yml'); print('valid')"`
Expected: tests pass; prints `valid`.

- [ ] **Step 7: Commit the tuned values**

```bash
git add src/pipeline_orchestrator/config/place_poses.yml
git commit -m "Tune place_poses.yml from live-sim validation

Written By: Claude Opus 4.8"
```

---

## Out of scope (later plan)

Orchestrator integration — pick → rule-based transit (lift to `transit_z` → traverse → descend) → release → home, destination selection from the task command, and the collision-off bookshelf +x insert / −x retract execution — is a separate plan once these pose values are locked.
