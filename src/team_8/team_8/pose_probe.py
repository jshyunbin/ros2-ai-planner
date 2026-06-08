"""pose_probe: a single self-contained tool to test whether a tool0 pose is
reachable and watch the arm move there in Gazebo.

Debug utility (not part of the runtime pipeline). It builds its OWN cuRobo
planner in-process -- no curobo_service, no orchestrator, just this one node --
so you give it a pose, it plans to it, and (by default) drives the arm via
``/ur5_controller/follow_joint_trajectory`` so you can see how the robot reaches
the end-effector pose. If the pose is unreachable it says so and does not move.

At the ``pose>`` prompt, enter either:

  * seven numbers ``x y z qx qy qz qw``  -> a raw tool0 pose in base_link, or
  * a place_poses.yml key (``bookshelf``, ``storage_1``, ...) -> that config
    pose (bookshelf resolves to its ``pre_insert`` staging pose).

``q`` / ``quit`` / ``exit`` leaves. ``-p target:=<key>`` runs one target once and
exits; an empty ``target`` (the default) starts the REPL. ``-p execute:=false``
plans only (reachability check) without moving the arm.

Run inside the container, with manip_challenge (Gazebo) up so there is a robot
to move and ``/joint_states`` to read -- and nothing else of ours running (this
tool owns the arm and its own planner):

  docker compose run --rm ai_planner ros2 run team_8 pose_probe

The first plan pays a one-time cuRobo/CUDA warmup (tens of seconds); after that
the REPL is interactive. Planning is collision-aware against whatever map this
node has built, but since it does not continuously stream the cameras it usually
falls back to a collision-OFF plan -- fine for a reachability/visualization
tool, but the motion is not guaranteed obstacle-free.
"""

import math
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState
import tf2_ros

try:  # cuRobo pulls in torch/CUDA; guard so the pure helpers import on a bare host
    from team_8.curobo import CuRobo
except Exception:  # noqa: BLE001 - any missing runtime dep disables planning
    CuRobo = None

from team_8.place_pose_utils import (
    load_place_poses,
    pose_from_xyzquat,
    resolve_target_pose,
)


# --- pure helpers (no live ROS graph required; unit-tested) -------------------

def _is_number(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def parse_pose_input(raw: str):
    """Parse one REPL line into a pose request.

    Returns one of:
      ``("pose", [x, y, z, qx, qy, qz, qw])`` -- seven numbers,
      ``("key", name)``                       -- a single non-numeric token,
      ``("error", message)``                  -- malformed input,
      ``None``                                -- blank line (ignore).
    """
    text = raw.strip()
    if not text:
        return None
    tokens = text.replace(",", " ").split()
    if len(tokens) == 1 and not _is_number(tokens[0]):
        return ("key", tokens[0])
    if len(tokens) == 7 and all(_is_number(t) for t in tokens):
        return ("pose", [float(t) for t in tokens])
    return ("error",
            "Enter 7 numbers `x y z qx qy qz qw` or a single place_poses key.")


def position_error_m(req_xyz, ach_xyz) -> float:
    """Euclidean distance (m) between requested and achieved positions."""
    return math.sqrt(
        sum((float(a) - float(b)) ** 2 for a, b in zip(req_xyz, ach_xyz)))


def _normalize_quat(quat):
    norm = math.sqrt(sum(float(c) ** 2 for c in quat))
    if norm == 0.0:
        return [0.0, 0.0, 0.0, 1.0]
    return [float(c) / norm for c in quat]


def orientation_error_deg(req_quat_xyzw, ach_quat_xyzw) -> float:
    """Angle (deg) between two orientations given as quat_xyzw.

    Sign- and scale-insensitive: ``q`` and ``-q`` (and any positive multiple)
    are the same rotation. ``angle = 2 * acos(|dot(unit_a, unit_b)|)``.
    """
    a = _normalize_quat(req_quat_xyzw)
    b = _normalize_quat(ach_quat_xyzw)
    dot = abs(sum(x * y for x, y in zip(a, b)))
    dot = min(1.0, max(-1.0, dot))
    return math.degrees(2.0 * math.acos(dot))


# --- node ---------------------------------------------------------------------

class PoseProbe(Node):
    """REPL / one-shot: plan to a tool0 pose with an in-process cuRobo and move."""

    def __init__(self) -> None:
        super().__init__("pose_probe")

        self.declare_parameter(
            "poses_file", "/ros2_ws/src/team_8/config/place_poses.yml")
        # Empty target -> interactive REPL; a non-empty key -> one-shot run.
        self.declare_parameter("target", "")
        self.declare_parameter("execute", True)
        self.declare_parameter(
            "arm_action_name", "/ur5_controller/follow_joint_trajectory")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("tool_frame", "tool0")
        self.declare_parameter("execute_timeout_sec", 60.0)
        self.declare_parameter("pos_tol_m", 0.02)
        self.declare_parameter("orient_tol_deg", 5.0)

        self._poses_file = str(self.get_parameter("poses_file").value)
        self._target = str(self.get_parameter("target").value).strip()
        self._execute = bool(self.get_parameter("execute").value)
        self._execute_timeout = float(
            self.get_parameter("execute_timeout_sec").value)
        self._base_frame = str(self.get_parameter("base_frame").value)
        self._tool_frame = str(self.get_parameter("tool_frame").value)
        self._pos_tol = float(self.get_parameter("pos_tol_m").value)
        self._orient_tol = float(self.get_parameter("orient_tol_deg").value)

        self._place_poses = None
        self._latest_joints = None
        self._curobo = None

        self.create_subscription(
            JointState, "/joint_states", self._cache_joints, 10)
        self._arm = ActionClient(
            self,
            FollowJointTrajectory,
            str(self.get_parameter("arm_action_name").value),
        )
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

    def _cache_joints(self, msg: JointState) -> None:
        self._latest_joints = msg

    # --- entry points ---------------------------------------------------------

    def run(self) -> int:
        if CuRobo is None:
            self.get_logger().error(
                "cuRobo unavailable (torch/CUDA missing?); cannot plan. "
                "Run this inside the ai_planner container.")
            return 1
        try:
            self._place_poses = load_place_poses(self._poses_file)
        except (OSError, ValueError) as exc:
            self.get_logger().warning(
                f"Could not load {self._poses_file}: {exc}. "
                "Key lookups disabled (raw poses still work).")
            self._place_poses = None

        if not self._wait_for_joints():
            return 1

        self.get_logger().info("Building cuRobo planner (one-time warmup)...")
        self._curobo = CuRobo(self)
        self.get_logger().info("Planner ready.")

        if self._target:                     # one-shot
            return 0 if self._process_key(self._target) else 1
        return self._repl()

    def _repl(self) -> int:
        self.get_logger().info(
            "Enter `x y z qx qy qz qw` (tool0 in base_link) or a place_poses "
            "key; `q` to quit.")
        while rclpy.ok():
            try:
                raw = input("\npose> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if raw.strip().lower() in ("q", "quit", "exit"):
                break
            parsed = parse_pose_input(raw)
            if parsed is None:
                continue
            kind, value = parsed
            if kind == "error":
                self.get_logger().warning(value)
            elif kind == "key":
                self._process_key(value)
            else:                            # "pose"
                self._process_pose_vec(value, label=f"raw {value}")
        return 0

    # --- per-pose pipeline ----------------------------------------------------

    def _process_key(self, key: str) -> bool:
        if self._place_poses is None:
            self.get_logger().error(
                f"No place_poses loaded; cannot resolve key {key!r}.")
            return False
        try:
            pose = resolve_target_pose(self._place_poses, key)
        except (KeyError, ValueError, TypeError, IndexError) as exc:
            # TypeError/IndexError: a non-pose key like `transit_z` or
            # `home_joint_config` (a scalar/list, not an xyz+quat entry).
            self.get_logger().error(f"Could not resolve key {key!r}: {exc}")
            return False
        if pose is None:
            self.get_logger().error(
                "geometry_msgs unavailable; cannot build pose.")
            return False
        vec = [pose.position.x, pose.position.y, pose.position.z,
               pose.orientation.x, pose.orientation.y, pose.orientation.z,
               pose.orientation.w]
        return self._process(pose, vec, label=key)

    def _process_pose_vec(self, vec, label: str) -> bool:
        pose = pose_from_xyzquat(vec[:3], vec[3:])
        if pose is None:
            self.get_logger().error(
                "geometry_msgs unavailable; cannot build pose.")
            return False
        return self._process(pose, vec, label=label)

    def _process(self, pose, requested_vec, label: str) -> bool:
        self.get_logger().info(
            f"target={label} "
            f"xyz=({pose.position.x:.3f},{pose.position.y:.3f},"
            f"{pose.position.z:.3f}) "
            f"quat_xyzw=({pose.orientation.x:.3f},{pose.orientation.y:.3f},"
            f"{pose.orientation.z:.3f},{pose.orientation.w:.3f})")
        trajectory = self._curobo.plan_trajectory(pose, self._latest_joints)
        if trajectory is None or not trajectory.points:
            self.get_logger().error(
                "UNREACHABLE/FAILED: cuRobo found no trajectory to this pose "
                "(IK infeasible or no collision-free plan).")
            return False
        self.get_logger().info(
            f"Planned OK ({len(trajectory.points)} points).")
        if not self._execute:
            self.get_logger().info("execute=false; not moving the arm.")
            return True
        if not self._send_trajectory(trajectory):
            return False
        self._report_accuracy(requested_vec)
        return True

    def _send_trajectory(self, trajectory) -> bool:
        if not self._arm.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(
                "Arm action server unavailable. Is manip_challenge running?")
            return False
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory
        send_future = self._arm.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=10.0)
        if not send_future.done() or send_future.result() is None:
            self.get_logger().error("Arm goal-send timed out.")
            return False
        handle = send_future.result()
        if not handle.accepted:
            self.get_logger().error("Arm goal rejected by action server.")
            return False
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(
            self, result_future, timeout_sec=self._execute_timeout)
        if not result_future.done() or result_future.result() is None:
            self.get_logger().error("Arm action returned no result.")
            return False
        self.get_logger().info("Trajectory executed.")
        return True

    def _report_accuracy(self, requested_vec) -> None:
        for _ in range(10):                  # let TF accumulate a couple of frames
            rclpy.spin_once(self, timeout_sec=0.1)
        try:
            tf = self._tf_buffer.lookup_transform(
                self._base_frame, self._tool_frame, rclpy.time.Time())
        except Exception as exc:             # tf2 raises several exception types
            self.get_logger().warning(
                f"TF {self._base_frame}<-{self._tool_frame} lookup failed: {exc}")
            return
        t = tf.transform.translation
        q = tf.transform.rotation
        self.get_logger().info(
            f"ACHIEVED {self._tool_frame} in {self._base_frame}: "
            f"xyz=({t.x:.3f},{t.y:.3f},{t.z:.3f}) "
            f"quat_xyzw=({q.x:.3f},{q.y:.3f},{q.z:.3f},{q.w:.3f})")
        pos_err = position_error_m(requested_vec[:3], [t.x, t.y, t.z])
        ori_err = orientation_error_deg(requested_vec[3:], [q.x, q.y, q.z, q.w])
        verdict = ("PASS" if pos_err <= self._pos_tol
                   and ori_err <= self._orient_tol else "FAIL")
        self.get_logger().info(
            f"{verdict}: pos_err={pos_err * 1000:.1f}mm "
            f"(tol {self._pos_tol * 1000:.0f}mm) "
            f"orient_err={ori_err:.1f}deg (tol {self._orient_tol:.0f}deg)")

    # --- prerequisites --------------------------------------------------------

    def _wait_for_joints(self) -> bool:
        for _ in range(50):
            if self._latest_joints is not None:
                break
            rclpy.spin_once(self, timeout_sec=0.1)
        if self._latest_joints is None:
            self.get_logger().error(
                "No /joint_states received. Is manip_challenge running?")
            return False
        return True


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PoseProbe()
    try:
        raise SystemExit(node.run())
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


# Allows running without a colcon rebuild via `python3 -m team_8.pose_probe`
# (the entrypoint puts src on PYTHONPATH), convenient for the live-sim loop.
if __name__ == "__main__":
    main()
