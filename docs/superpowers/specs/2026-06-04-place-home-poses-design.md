# Design: Hardcoded place/home poses for storage & bookshelf

**Date:** 2026-06-04
**Status:** Approved (design); scope = *finding & recording the poses*. Orchestrator
integration is a separate, later spec.

## Problem

The pipeline can pick an object but has no defined targets for *placing* it. The
mission requires dropping a grasped object into one of three destinations —
**storage 1** (left basket), **storage 2** (right basket), or the two-story
**bookshelf** — safely. We need hardcoded end-effector poses for each
destination, plus an **initial/home wrist pose** that gives the wrist camera a
clear top-down view of the workspace basket for segmentation.

This spec covers determining and recording those pose values. Wiring them into
the orchestrator task flow is out of scope here.

## Verified ground truth (from the live sim + world file)

Environment file:
`cs477_ws/src/cs477_IIR/manip_challenge/data/worlds/ur5_picking_challenge2.world`

World-frame model placements (robot `ur5_base` at world origin):

| Model | World pose (x, y, z) | Notes |
|---|---|---|
| `storage_left` (storage 1) | `(0, 0.55, 0.6)` | `storage_a_basket`, drop from above |
| `storage_right` (storage 2) | `(0, -0.55, 0.6)` | `storage_b_basket`, drop from above |
| `workspace_basket` (pick area) | `(0.55, 0, 0.5)` | objects start here |
| `bookshelf` | `(0.95, -0.3, 0.5)`, yaw `-1.57` | opening faces robot (−x) |

Bookshelf shelf boards (model local z above the `0.5` base; board size z=0.02):
base `+0.01` → world z `0.51`, middle `+0.23` → `0.73`, top `+0.45` → `0.95`.
Footprint 0.3 m (opening width) × 0.27 m (depth). Back wall is on the +x side, so
the opening faces −x (toward the robot).

Live TF facts (read via `tf2_echo` while the sim ran):
- **`world → base_link` is identity** (translation 0, rotation 0). So world-frame
  coordinates equal `base_link` coordinates directly — no offset to apply.
- Wrist camera optical frame relative to `tool0`: `~(-0.03, -0.07, 0.03)` with
  near-zero relative rotation, so the camera optical axis ≈ `tool0` +z. When
  `tool0` points straight down, the wrist camera looks straight down too.

## Conventions

All poses are the **`tool0`** link expressed in **`base_link`** (cuRobo's frame
contract; `BASE_FRAME = 'base_link'`, planner targets `tool0`).

- `tool0` **+z is the approach axis**. The gripper TCP (fingertips) is
  **`GRIPPER_TCP_Z_OFFSET = 0.1034 m`** beyond `tool0` along +z. A cartesian
  target for `tool0` therefore sits 0.1034 m *behind* the desired fingertip
  contact point, measured along the approach axis.
- **Top-down** orientation (storage drops, home): `tool0` +z = world −z (pointing
  down). Quaternion `(x, y, z, w) = (1, 0, 0, 0)` (180° about base x: +z → −z,
  +x → +x, +y → −y).
- **Horizontal-into-shelf** orientation (bookshelf): `tool0` +z = world +x
  (gripper points into the shelf). Exact quaternion determined during validation.

## Artifact

New config file: `src/pipeline_orchestrator/config/place_poses.yml`

A named-pose config holding cartesian `tool0`-in-`base_link` poses plus scalar
insertion params. Schema:

```yaml
home:      {xyz: [0.55, 0.07, 0.90], quat_xyzw: [1, 0, 0, 0]}   # wrist-cam over basket
storage_1: {xyz: [0.0,  0.55, 0.85], quat_xyzw: [1, 0, 0, 0]}   # left basket, drop above
storage_2: {xyz: [0.0, -0.55, 0.85], quat_xyzw: [1, 0, 0, 0]}   # right basket, drop above

bookshelf_floor1:
  pre_insert:     {xyz: [0.60, -0.30, 0.55], quat_xyzw: [<+x-forward>]}
  insert_depth_m: 0.22     # straight +x push after reaching pre_insert
  retract_depth_m: 0.22    # straight -x retract after releasing

bookshelf_floor2:
  pre_insert:     {xyz: [0.60, -0.30, 0.76], quat_xyzw: [<+x-forward>]}
  insert_depth_m: 0.22
  retract_depth_m: 0.22
```

The numeric values above are **computed starting guesses**. The validation
workflow below tunes them in place; the file is the deployable artifact.

Notes on the starting guesses:
- Storage z `0.85`: basket origin `0.6` + clearance, with `tool0` `0.1034 m`
  above the intended fingertip drop point (~`0.75`). Tune so fingertips clear the
  rim before release.
- Home `xyz [0.55, 0.07, 0.90]`: above `workspace_basket (0.55, 0, 0.5)`, shifted
  +0.07 in y to compensate the wrist-camera −y offset so the camera centers on
  the basket; height gives the wrist cam a full-basket view.
- Bookshelf `pre_insert` x `0.60`: standoff in front of the shelf front face
  (`x ≈ 0.95 − 0.135 = 0.815`); floor1 z `0.55` (above base board `0.51`),
  floor2 z `0.76` (above middle board `0.73`).

## Bookshelf insertion sequence (mirrors the pick path)

The bookshelf place mirrors pick's approach → grasp → lift:

1. **pre_insert** — planned, collision-aware motion to the `pre_insert` pose in
   front of the shelf opening (cuRobo `plan_trajectory`, single-pose mode).
2. **insert** — straight **+x push** of `insert_depth_m`, collision-off linear
   move (the same mechanism the pick path uses for the collision-off descent),
   so the gripper enters the shelf without the planner refusing on shelf contact.
3. **open gripper** — release the object onto the board.
4. **retract** — straight **−x** linear move of `retract_depth_m` to withdraw.

For pose *finding*, validating `pre_insert` reachability via the planner is
sufficient; the +x push depth is validated by eye in the sim and recorded as
`insert_depth_m`.

## Discovery workflow (hybrid compute + validate)

A standalone debug utility, `pose_probe.py` (not part of the runtime pipeline,
alongside the other debug callers), drives one tuning loop:

1. Read a target name and load its pose from `place_poses.yml`.
2. Capture the current `/joint_states`.
3. Call `/curobo/plan_trajectory` in **single-pose mode** — set `grasp_pose`,
   leave `grasp_poses` empty, pass the joint state.
4. Execute the returned `trajectory` on
   `/ur5_controller/follow_joint_trajectory`.
5. Print the resulting `tool0` pose (from TF) for inspection.

Per target: run → observe in Gazebo → edit the YAML number (drop height /
insertion depth / standoff / home framing) → re-run until safe. Drop to manual
teleop only when cuRobo cannot reach a spot; then read the feasible `tool0` pose
back from TF and record it in the YAML.

## Components & boundaries

| Unit | Responsibility | Depends on |
|---|---|---|
| `config/place_poses.yml` | Static named cartesian targets + insertion params | — |
| `pose_probe.py` | One-shot: load pose → plan via cuRobo → execute → report | `/curobo/plan_trajectory`, `/joint_states`, arm action, TF |

`pose_probe.py` reuses the existing single-pose `plan_trajectory` service path
and the arm `FollowJointTrajectory` action; it adds no new planner logic.

## Out of scope (later phase)

Orchestrator integration — the full pick → move-to-place → release → home task
flow, destination selection from the task command, and the collision-off bookshelf
insert/retract execution — is a separate spec, written once these pose values are
locked.

## Success criteria

- `place_poses.yml` exists with validated, safe values for `home`, `storage_1`,
  `storage_2`, `bookshelf_floor1`, `bookshelf_floor2`.
- Each storage/home pose is reachable: cuRobo plans and the arm executes to it in
  the live sim without collision, and (home) the wrist camera frames the
  workspace basket.
- Each bookshelf `pre_insert` pose is reachable and positioned so a straight +x
  push of `insert_depth_m` lands the object on the correct shelf board.
