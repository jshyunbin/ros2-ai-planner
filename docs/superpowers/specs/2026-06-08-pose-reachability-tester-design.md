# Pose reachability tester (extend `pose_probe`) — design

**Date:** 2026-06-08
**Status:** approved

## Problem

The bookshelf place goal keeps failing to plan. Root cause (confirmed from live
logs): the bookshelf `pre_insert` pose `[0.75, -0.3, 0.78]` is **outside the
UR5's reachable workspace** (≈1.06 m from the shoulder vs ~0.85–0.9 m max reach);
the place transit's descend leg reports `transit_2 IK found no feasible solution`
and `plan_place` fails. We need a fast interactive way to probe candidate place
poses for reachability + placement accuracy in the live sim, so a reachable
bookshelf pose (or the conclusion that none exists) can be found empirically.

## Decision

Extend the existing one-shot `src/team_8/team_8/pose_probe.py` into an
interactive REPL tester (the pose analog of `home_config_tuner`), and make it
**self-contained**: it builds its **own** in-process `CuRobo` planner instead of
calling the `curobo_service`. (Revised after first draft: requiring both
`curobo_service` and `pose_probe` to be running was rejected as too operationally
heavy — the whole point is a single command to check one pose.) Nothing
operational depends on `pose_probe` (referenced only in its own 2026-06-04 docs +
`setup.py`), so repurposing it is safe.

The planner is reused via `CuRobo.plan_trajectory(pose, joint_states)`, which
turns a tool0 pose into a smooth executable trajectory (with a collision-off
retry) and returns `None` when IK is infeasible — exactly the
reachable/unreachable signal we want. No new method is added to `curobo.py`.

## Behavior

REPL prompt `pose>` (mirrors `home_config_tuner`). Each entry is one of:

- **7 numbers** `x y z qx qy qz qw` → a raw `tool0` pose in `base_link`.
- **a `place_poses.yml` key** (`bookshelf`, `storage_1`, …) → resolved with
  `resolve_target_pose` (bookshelf → its `pre_insert` pose).
- `q` / `quit` / `exit` → leave.

Per entry:

1. Call the in-process `CuRobo.plan_trajectory(pose, latest_joints)`.
2. **No trajectory** (`None`/empty) → print `UNREACHABLE/FAILED`. Do **not**
   move the arm.
3. **Plan ok** → if `execute`, run the returned trajectory on
   `/ur5_controller/follow_joint_trajectory` (so you watch the arm reach the
   EEF in Gazebo), then look up `base_link→tool0` from TF and report **requested
   vs achieved** xyz + quat, **position error (mm)**, **orientation error
   (deg)**, and a `PASS/FAIL` against tolerances. If `execute=false`, report
   planned-ok and skip motion.

### Backward-compatible one-shot

If the `target` param is non-empty, run that single target once and exit (the
existing 2026-06-04 tuning recipe). Empty `target` (new default) → REPL.

## Params

Existing: `poses_file`, `target` (default now `""`), `execute`, `service_name`,
`arm_action_name`, `base_frame`, `tool_frame`, `service_timeout_sec`,
`execute_timeout_sec`. New: `pos_tol_m` (0.02), `orient_tol_deg` (5.0).

## Pure, unit-tested helpers (module-level, no rclpy)

- `parse_pose_input(raw) -> ("pose", [7 floats]) | ("key", name) | ("error", msg) | None`
- `position_error_m(req_xyz, ach_xyz) -> float`
- `orientation_error_deg(req_quat_xyzw, ach_quat_xyzw) -> float`
  (angle between quaternions, `2·acos(min(1, |dot|))`, math-only).

Tested in `src/team_8/test/test_pose_probe.py`, mirroring `test_place_pose_utils.py`.

## Scope notes

- Single-pose mode is **collision-aware** and uses free `plan_pose` (not the
  collision-off in-branch IK the bookshelf place leg uses). For pure
  reachability the IK solver is the same, so an unreachable pose fails
  identically; a collision-blocked or branch-flipped pose may differ. Noted in
  the docstring.
- Tests **one pose** (the staging/`pre_insert` pose), not the full
  lift→traverse→descend→insert→retract transit. That is the right diagnostic for
  "is this pose reachable," which is where the bookshelf failure occurs.
- No wrist-image capture (not needed).

## Run recipe

Single command, inside the container, with manip_challenge (Gazebo) up and
nothing else of ours running (this tool owns the arm and builds its own planner):

```
docker compose run --rm ai_planner ros2 run team_8 pose_probe                 # REPL
docker compose run --rm ai_planner ros2 run team_8 pose_probe --ros-args -p target:=bookshelf -p execute:=false
```

First plan pays a one-time cuRobo/CUDA warmup (tens of seconds), then the REPL is
interactive.
