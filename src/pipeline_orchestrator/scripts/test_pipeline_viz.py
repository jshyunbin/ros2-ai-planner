#!/usr/bin/env python3
"""
Pipeline visualization test: synthetic depth → Mapper → MotionPlanner → viser

Run inside the container:
  docker compose run --rm -p 8080:8080 ai_planner python3 /ros2_ws/src/pipeline_orchestrator/scripts/test_pipeline_viz.py

Then open http://localhost:8080 in your browser.
SSH users — forward the port first:
  ssh -L 8080:localhost:8080 user@host
"""
import time
import numpy as np
import torch
import viser

UR5_CONFIG  = '/ros2_ws/src/pipeline_orchestrator/config/ur5_curobo.yml'
URDF_PATH   = '/ur5.urdf'
JOINT_NAMES = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
]
HOME_CFG = [0.0, -2.2, 1.9, -1.383, -1.57, 0.0]


# ── synthetic scene ───────────────────────────────────────────────────────────

def synthetic_depth(H=480, W=640):
    """Flat table at 0.6 m with a 10 cm box in the centre."""
    depth = torch.full((H, W), 0.6, dtype=torch.float32, device='cuda')
    cx, cy, bw, bh = W // 2, H // 2, 60, 60
    depth[cy - bh // 2:cy + bh // 2, cx - bw // 2:cx + bw // 2] = 0.5
    return depth


# ── stage 1: build voxel map ─────────────────────────────────────────────────

def build_map():
    from curobo.perception import Mapper, MapperCfg
    from curobo.types import CameraObservation, Pose

    mapper = Mapper(MapperCfg(
        extent_meters_xyz=(2.0, 2.0, 1.5),
        voxel_size=0.02,
        esdf_voxel_size=0.05,
        truncation_distance=0.1,
        depth_minimum_distance=0.15,
        depth_maximum_distance=2.0,
        num_cameras=1,
    ))

    K = torch.tensor(
        [[500, 0, 320], [0, 500, 240], [0, 0, 1]],
        dtype=torch.float32, device='cuda',
    )
    pose = Pose.from_numpy(
        np.array([0.0, 0.0, 1.5], dtype=np.float32),
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )

    print('  Integrating 10 synthetic depth frames...')
    for _ in range(10):
        obs = CameraObservation(
            depth_image=synthetic_depth().unsqueeze(0),
            intrinsics=K.unsqueeze(0),
            pose=pose,
        )
        mapper.integrate(obs)

    voxel_grid = mapper.compute_esdf()
    print('  ESDF ready.')
    return voxel_grid


# ── stage 2: motion planner ───────────────────────────────────────────────────

def build_planner(voxel_grid):
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo._src.geom.types import SceneCfg

    print('  Loading MotionPlanner (warmup ~30 s)...')
    planner = MotionPlanner(MotionPlannerCfg.create(robot=UR5_CONFIG))
    planner.warmup(enable_graph=True, num_warmup_iterations=3)
    planner.update_world(SceneCfg(voxel=[voxel_grid]))
    print('  MotionPlanner ready.')
    return planner


# ── stage 3: plan trajectory ──────────────────────────────────────────────────

def plan(planner):
    from curobo.types import JointState as CJS, GoalToolPose

    start = CJS.from_position(
        torch.tensor([HOME_CFG], dtype=torch.float32, device='cuda'),
        joint_names=JOINT_NAMES,
    )
    goal = GoalToolPose(
        tool_frames=planner.tool_frames,
        position=torch.tensor(
            [[[[[0.3, 0.0, 0.4]]]]], device='cuda', dtype=torch.float32),
        quaternion=torch.tensor(
            [[[[[1.0, 0.0, 0.0, 0.0]]]]], device='cuda', dtype=torch.float32),
    )

    result = planner.plan_pose(goal, start)
    if result is None or not result.success.any():
        print('  Planning failed — visualizing home pose only.')
        return [HOME_CFG]

    traj = result.get_interpolated_plan().position[0].cpu().numpy().tolist()
    print(f'  Planned {len(traj)}-waypoint trajectory.')
    return traj


# ── stage 4: viser visualization ─────────────────────────────────────────────

def visualize(traj):
    server = viser.ViserServer(port=8080, verbose=False)
    print('\nViser running — open http://localhost:8080')

    try:
        from viser.extras import ViserUrdf
        robot = ViserUrdf(server, urdf_or_path=URDF_PATH, root_node_name='/ur5')
        has_robot = True
        print('  Robot model loaded.')
    except Exception as e:
        print(f'  Robot model unavailable ({e}), showing end-effector frames only.')
        has_robot = False

    # World frame origin
    server.scene.add_frame('/world', axes_length=0.3, axes_radius=0.01)

    # Target position marker
    server.scene.add_icosphere(
        '/target',
        radius=0.03,
        color=(255, 80, 80),
        position=(0.3, 0.0, 0.4),
    )

    print(f'Animating {len(traj)}-waypoint trajectory. Ctrl+C to stop.')
    try:
        while True:
            for waypoint in traj:
                if has_robot:
                    robot.update_cfg(dict(zip(JOINT_NAMES, waypoint)))
                time.sleep(0.05)
            time.sleep(1.0)
    except KeyboardInterrupt:
        print('Stopped.')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print('=== cuRoboV2 Pipeline Visualization Test ===\n')

    print('[1/4] Initializing CUDA / Warp...')
    import warp as wp
    wp.init()
    print('  Warp OK.\n')

    print('[2/4] Building voxel map from synthetic depth...')
    voxel_grid = build_map()

    print('\n[3/4] Setting up motion planner...')
    planner = build_planner(voxel_grid)

    print('\n[4/4] Planning trajectory to (0.3, 0.0, 0.4)...')
    traj = plan(planner)

    print('\n[viz] Starting viser...')
    visualize(traj)


if __name__ == '__main__':
    main()
