#!/usr/bin/env python3
"""
CuRobo pipeline test: real ROS2 depth → CuRobo class → action deploy → viser.

Unlike test_live_viz.py (which re-implements the cuRobo perception/planning
logic inline), this script exercises the *actual* ``pipeline_orchestrator.CuRobo``
class end to end, so it doubles as an integration test of the orchestrator's
motion-planning module:

  1. perception        — CuRobo subscribes to both depth cameras and fuses a
                         live TSDF/ESDF (CuRobo runs in enable_viz mode so it
                         also caches point clouds + surface voxels for viser).
  2. motion planning   — CuRobo.plan_trajectory(grasp_pose, joint_states)
                         returns a collision-free JointTrajectory to GOAL_XYZ.
  3. action deployment — the harness (standing in for the orchestrator) sends
                         the trajectory to /ur5_controller/follow_joint_trajectory.
                         Deployment is the driver's job, not CuRobo's.

viser renders the live clouds, reconstructed voxels, the planned trajectory
(animated on the URDF) and the goal marker.

Run inside the container (with the manip_challenge sim running on the host):
  docker compose run --rm -p 8080:8080 ai_planner \\
    python3 /ros2_ws/src/pipeline_orchestrator/scripts/test_curobo_pipeline_viz.py

Then open http://localhost:8080 in your browser.
SSH users — forward the port first:
  ssh -L 8080:localhost:8080 user@host
"""
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState

from pipeline_orchestrator.curobo import CuRobo
from pipeline_orchestrator.live_viz_helpers import resolve_urdf, resolve_urdf_string

# ── constants ─────────────────────────────────────────────────────────────────
URDF_PATH    = '/ur5.urdf'
ARM_TRAJ_ACTION = '/ur5_controller/follow_joint_trajectory'
JOINT_NAMES  = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
]
HOME_CFG     = [0.0, -2.2, 1.9, -1.383, -1.57, 0.0]
GOAL_XYZ     = (0.3, 0.0, 0.4)
GOAL_QUAT    = (1.0, 0.0, 0.0, 0.0)    # w x y z
MIN_FRAMES   = 5
REPLAN_EVERY = 10                       # frames between replans
VIZ_HZ       = 10
WORLD_FRAME  = 'base_link'
TICK_SEC     = 0.5                       # how often the plan timer checks in
SETTLE_SEC   = 1.0                       # pause after a leg finishes before the next


# ── shared state (planner thread → viser) ──────────────────────────────────────
@dataclass
class SharedState:
    """Only the planned trajectory needs sharing; clouds/voxels/joints are read
    straight from the thread-safe CuRobo accessors."""
    lock: threading.Lock         = field(default_factory=threading.Lock)
    traj: Optional[np.ndarray]   = None   # (T, J) float32 of the 6 arm joints
    deployed: bool               = False
    leg: str                     = 'goal'  # which leg was last commanded


# ── goal / joint-state construction ─────────────────────────────────────────────

def pose_from(xyz, quat_wxyz):
    """A geometry_msgs/Pose — matches the .position / .orientation attribute
    access CuRobo.plan_trajectory expects."""
    from geometry_msgs.msg import Pose
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = xyz
    pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z = quat_wxyz
    return pose


def arm_joint_state(latest):
    """Build a 6-DOF sensor_msgs/JointState for the UR5 arm only.

    /joint_states also carries the gripper joints; the planner only knows the
    6 UR5 joints, so extract them in canonical order (HOME_CFG for any missing).
    """
    from sensor_msgs.msg import JointState
    if latest is not None:
        by_name = dict(zip(latest.name, latest.position))
        ordered = [by_name.get(n, HOME_CFG[i]) for i, n in enumerate(JOINT_NAMES)]
    else:
        ordered = list(HOME_CFG)
    js = JointState()
    js.name = list(JOINT_NAMES)
    js.position = ordered
    return js


# ── action deployment (the driver's job, not CuRobo's) ──────────────────────────

def deploy_trajectory(arm_client, traj, timeout_sec: float = 2.0):
    """Send a planned JointTrajectory to the UR5 arm controller.

    CuRobo only plans; the driver (here the harness, in production the
    orchestrator via its FollowJointTrajectory action clients) deploys.
    """
    if traj is None or not traj.points:
        return None
    if not arm_client.wait_for_server(timeout_sec=timeout_sec):
        print(f'  arm action server {ARM_TRAJ_ACTION} unavailable; plan-only.')
        return None
    goal = FollowJointTrajectory.Goal()
    goal.trajectory = traj
    return arm_client.send_goal_async(goal)


# ── planning tick (runs on the ROS executor thread) ──────────────────────────────

def make_plan_tick(curobo: CuRobo, state: SharedState, arm_client):
    """Build a timer callback that ping-pongs the arm: goal → home → goal …

    The first tick captures the robot's current end-effector pose (via FK) as
    the "home" target, then alternates planning to the goal and back to home,
    deploying each leg and waiting out its trajectory duration before the next.

    Crucially this runs on the single-threaded ROS executor, so it is
    serialized with CuRobo's depth callbacks. Running plan_trajectory
    (which captures a CUDA graph in compute_esdf) on a *separate* thread
    races the segmenter's CUDA ops in the depth callback and invalidates
    the capture (cudaErrorStreamCaptureInvalidated) — exactly how the
    orchestrator avoids it by planning inside a subscription callback.
    """
    goal_pose = pose_from(GOAL_XYZ, GOAL_QUAT)
    ctx = {'targets': None, 'idx': 0, 'next_at': 0.0}

    def _tick():
        if curobo.frame_count < MIN_FRAMES:
            return
        now = time.time()
        if now < ctx['next_at']:
            return   # previous leg still executing — let the arm finish

        js = arm_joint_state(curobo.get_latest_joints())

        # Define the two endpoints once: the goal pose, and the end-effector
        # pose of HOME_CFG (a distinct, known-reachable joint configuration —
        # not the arm's current pose, which may already sit at the goal from a
        # previous run). FK gives HOME_CFG's full pose+orientation so plan_pose
        # always has an IK solution.
        if ctx['targets'] is None:
            home = curobo.tool_pose(arm_joint_state(None))   # FK of HOME_CFG
            if home is None:
                return   # FK not ready yet; try again next tick
            ctx['targets'] = [('goal', goal_pose),
                              ('home', pose_from(home[0], home[1]))]
            print(f'  Endpoints set: goal {GOAL_XYZ} ↔ home '
                  f'{tuple(round(v, 3) for v in home[0])}; ping-ponging.')

        label, target = ctx['targets'][ctx['idx']]
        traj = curobo.plan_trajectory(target, js)
        if traj is None or not traj.points:
            # Planning failed (often a transient ESDF phantom from the arm's own
            # sweep). Back off ~1 s so perception can refresh before retrying the
            # same leg, rather than hammering every tick.
            ctx['next_at'] = now + 1.0
            return

        arr = np.array([list(p.positions) for p in traj.points], dtype=np.float32)
        future = deploy_trajectory(arm_client, traj)

        # Hold off the next leg until this trajectory's own duration elapses.
        last = traj.points[-1].time_from_start
        dur = last.sec + last.nanosec * 1e-9
        ctx['next_at'] = now + dur + SETTLE_SEC
        ctx['idx'] = 1 - ctx['idx']

        with state.lock:
            state.traj = arr
            state.deployed = future is not None
            state.leg = label

    return _tick


# ── viser update loop ───────────────────────────────────────────────────────────

def update_loop(server, curobo: CuRobo, state: SharedState, robot) -> None:
    """Main-thread loop: refresh the viser scene at VIZ_HZ from CuRobo state."""
    traj_idx = 0
    period = 1.0 / VIZ_HZ

    # yourdfpy raises KeyError on joints the loaded URDF doesn't know (e.g. the
    # gripper joints in /joint_states when only the UR5 URDF loaded), so filter
    # every cfg to the URDF's actuated joints.
    valid_joints = None
    if robot is not None:
        try:
            valid_joints = set(robot._urdf.actuated_joint_names)
        except Exception:
            valid_joints = set(JOINT_NAMES)

    def _apply_cfg(cfg):
        if valid_joints is not None:
            cfg = {k: v for k, v in cfg.items() if k in valid_joints}
        if cfg:
            robot.update_cfg(cfg)

    while True:
        t0 = time.time()

        clouds    = curobo.get_point_clouds()
        tsdf_pts  = curobo.get_tsdf_centers()
        n_frames  = curobo.frame_count
        latest_js = curobo.get_latest_joints()
        with state.lock:
            traj     = state.traj
            deployed = state.deployed
            leg      = state.leg

        # ── status ──────────────────────────────────────────────────────────
        traj_len = 0 if traj is None else len(traj)
        if n_frames < MIN_FRAMES:
            status = f'Perception: fusing depth ({n_frames}/{MIN_FRAMES})…'
        elif traj_len == 0:
            status = f'Frames: {n_frames}  |  planning…'
        else:
            status = (f'Frames: {n_frames}  |  →{leg.upper()}  |  '
                      f'Traj: {traj_len} pts  |  '
                      f'{"DEPLOYED" if deployed else "plan only"}')
        server.scene.add_label('/status', status, position=(0.0, 0.0, 1.6))

        # ── live point clouds ─────────────────────────────────────────────────
        for cam_id, pts in clouds.items():
            if pts is None or len(pts) == 0:
                continue
            color = (200, 200, 200) if cam_id == 'overhead' else (100, 150, 255)
            server.scene.add_point_cloud(
                f'/depth/{cam_id}',
                points=pts,
                colors=np.tile(color, (len(pts), 1)).astype(np.uint8),
                point_size=0.005,
            )

        # ── reconstructed TSDF surface voxels (shaded by height) ───────────────
        if tsdf_pts is not None and len(tsdf_pts) > 0:
            z = tsdf_pts[:, 2]
            z_norm = np.clip((z - z.min()) / max(z.max() - z.min(), 1e-6), 0, 1)
            colors = np.stack([
                (255 * (1 - z_norm)).astype(np.uint8),
                (255 * z_norm).astype(np.uint8),
                np.full_like(z_norm, 120, dtype=np.uint8),
            ], axis=1)
            server.scene.add_point_cloud(
                '/tsdf/voxels', points=tsdf_pts, colors=colors, point_size=0.02)

        # ── robot: mirror the *live* /joint_states so the URDF reflects the
        # real Gazebo arm actually moving goal↔home. Fall back to animating the
        # planned trajectory only if no live joints are available yet.
        if robot is not None:
            if latest_js is not None:
                _apply_cfg(dict(zip(latest_js.name, latest_js.position)))
            elif traj is not None and len(traj) > 0:
                waypoint = traj[traj_idx % len(traj)]
                _apply_cfg(dict(zip(JOINT_NAMES, waypoint.tolist())))
                traj_idx += 1

        elapsed = time.time() - t0
        time.sleep(max(0.0, period - elapsed))


# ── main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    import sys
    sys.stdout.reconfigure(line_buffering=True)   # flush progress logs promptly
    print('=== CuRobo Pipeline Test (perception → plan → deploy → viser) ===\n')

    print('[1/4] Initializing CUDA / Warp…')
    import warp as wp
    wp.init()
    print('  Warp OK.\n')

    print('[2/4] Starting ROS2 node + CuRobo (planner warmup ~30 s)…')
    rclpy.init()
    node = Node('curobo_pipeline_viz')
    curobo = CuRobo(node, enable_viz=True)

    # The harness plays the orchestrator's role: it owns /joint_states (feeding
    # CuRobo's robot segmenter) and the arm action client used to deploy plans.
    node.create_subscription(
        JointState, '/joint_states', curobo.update_joint_state, 10)
    arm_client = ActionClient(node, FollowJointTrajectory, ARM_TRAJ_ACTION)

    # Grab the full robot URDF (UR5 + Robotiq gripper) so the gripper joints in
    # /joint_states resolve in viser. TRANSIENT_LOCAL: the latched message
    # arrives even though it was published before we subscribed. Create the
    # subscription before spin starts so the spinning thread delivers it.
    from std_msgs.msg import String
    from rclpy.qos import (QoSProfile, QoSDurabilityPolicy,
                           QoSReliabilityPolicy, QoSHistoryPolicy)
    urdf_holder: dict = {}

    def _on_robot_description(msg):
        try:
            urdf_holder['path'] = resolve_urdf_string(msg.data)
        except Exception as exc:
            print(f'  /robot_description resolve failed: {exc}')

    node.create_subscription(
        String, '/robot_description', _on_robot_description,
        QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                   reliability=QoSReliabilityPolicy.RELIABLE,
                   history=QoSHistoryPolicy.KEEP_LAST))

    # Plan + deploy from a timer on the executor thread so it's serialized with
    # the depth callbacks (shared CUDA stream — see make_plan_tick).
    state = SharedState()
    node.create_timer(TICK_SEC, make_plan_tick(curobo, state, arm_client))

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    print('  CuRobo ready, ROS2 spinning.\n')

    print('[3/4] Starting viser…')
    import viser
    server = viser.ViserServer(port=8080, verbose=False)
    server.scene.add_frame('/base_link', axes_length=0.3, axes_radius=0.01)
    server.scene.add_icosphere(
        '/target', radius=0.03, color=(255, 80, 80), position=GOAL_XYZ)
    print('  Viser running — open http://localhost:8080\n')

    # Wait up to 5 s for the latched /robot_description; fall back to UR5-only.
    for _ in range(50):
        if 'path' in urdf_holder:
            break
        time.sleep(0.1)
    urdf_source = urdf_holder.get('path')
    if urdf_source is not None:
        print('  Full robot URDF received (UR5 + gripper).')
    else:
        print('  /robot_description not received; falling back to UR5-only URDF.')
        urdf_source = resolve_urdf(URDF_PATH)

    try:
        from viser.extras import ViserUrdf
        robot = ViserUrdf(
            server, urdf_or_path=urdf_source, root_node_name='/robot')
        print('  Robot model loaded.')
    except Exception as exc:
        print(f'  Robot model unavailable ({exc}), skipping URDF.')
        robot = None

    print('[4/4] Planning on the ROS executor; entering update loop. Ctrl+C to stop.')
    try:
        update_loop(server, curobo, state, robot)
    except KeyboardInterrupt:
        print('\nStopped.')
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
