# Design: Move grasped item to goal + return home (pipeline phases 3 & 4)

**Date:** 2026-06-05
**Branch:** `move_item`
**Status:** Approved (design)

## Problem

The pick-and-place pipeline can pick an object (phase 1 approach pre-grasp +
phase 2 grasp & lift) but stops after the lift. The mission needs the grasped
object **moved to a goal destination** (a storage basket or the bookshelf),
**released**, and the arm **returned home**. This spec covers those two
remaining phases:

- **Phase 3 — move to goal:** non-collision-aware, rule-based safe-`transit_z`
  routing through intermediate waypoints (the held object is invisible to
  collision, so the arm cannot be collision-checked against it — see the
  held-object constraint below).
- **Phase 4 — return home:** collision-aware, since the arm is empty after
  release.

The destination **poses already exist and are authoritative** in
`config/place_poses.yml` (provided by a teammate). This spec does **not** find or
tune poses.

## Held-object collision constraint (why phase 3 is non-collision-aware)

cuRobo cannot account for the carried object during planning: there is no
ground-truth mesh and no way to attach an arbitrary object to the single TSDF
voxel map. Collision-aware planning would protect only the **arm**, leaving the
**held object** free to swing into baskets/shelf/other objects while the planner
reports success. So motion while carrying must be **rule-based**: lift straight
up to a safe `transit_z`, traverse laterally at that height to above the
destination, and only then descend / insert. Those legs are planned
**collision-off** (the same `plan_pose` primitive the pick path already uses for
its collision-off grasp-descent and lift).

## Scope decisions (from brainstorming)

- **NL parsing is out of scope.** The orchestrator receives the destination as a
  resolved **key/string** (e.g. `storage_1`) from upstream. For this work a
  placeholder `target_goal` ROS parameter (default `storage_1`) feeds the key so
  the flow is runnable end-to-end.
- **All place/home knowledge lives in the cuRobo side.** `curobo_service` loads
  `place_poses.yml`, resolves the key, builds the transit route, and returns the
  trajectory segments. The orchestrator only forwards the key and executes the
  returned segments, opening the gripper at the right step.
- **Home is a cartesian pose** from `place_poses.yml`, planned with the existing
  collision-aware single-pose path.
- **Data-driven destinations.** A target is treated as **bookshelf-style** (needs
  insert/retract) when its YAML entry has `pre_insert` + `insert_depth_m`;
  otherwise it is a **simple top-down drop**. The code does **not** hardcode
  `floor1`/`floor2`. Only one bookshelf destination is required for now; floor1
  may be ignored.

## Architecture & data flow

```
orchestrator (after existing pick lift)
  │  forwards goal key (target_goal param / upstream)
  ▼
PlanTrajectory service  (goal_name set)
  ├─ storage_* / bookshelf_*  → CuRobo.plan_place()   [collision-OFF, phase 3]
  │     returns: trajectory (transit), insert_trajectory, retract_trajectory
  └─ home                     → CuRobo.plan_trajectory() [collision-ON, phase 4]
        returns: trajectory (home)

orchestrator execution sequence:
  plan_place(goal) → execute trajectory (transit)
    if insert_trajectory:  execute it            # bookshelf only
    open gripper (release)
    if retract_trajectory: execute it            # bookshelf only
  plan_home('home')  → execute trajectory (home, collision-aware)
```

Two service calls (place, then home). Home is planned **after release** from the
**actual** post-release joint state against the **fresh** TSDF — cleaner than
planning everything up front, and it cleanly separates the non-collision-aware
phase 3 from the collision-aware phase 4.

The orchestrator reacts to **which segments are populated** (it executes
`insert_trajectory`/`retract_trajectory` only if non-empty), so it needs no place
semantics beyond the key it forwards.

## Component changes

### 1. `riro_srvs/srv/PlanTrajectory.srv`
Add (requires a `colcon build` of `riro_srvs`):

- Request: `string goal_name` — place/home key. Empty ⇒ existing pick
  (`grasp_poses`) and single-pose (`grasp_pose`) modes are unchanged.
- Response: `trajectory_msgs/JointTrajectory insert_trajectory`,
  `trajectory_msgs/JointTrajectory retract_trajectory` — populated only for
  bookshelf-style destinations.

`trajectory` is reused for the transit/move (place) and for the home motion.

### 2. `place_pose_utils.py`
- Relax `load_place_poses` validation to be **data-driven**: require
  `transit_z`; validate the `home` simple target; for any other entry, accept
  either a simple `xyz`/`quat_xyzw` target **or** a bookshelf-style entry
  (`pre_insert` xyz/quat + numeric `insert_depth_m`/`retract_depth_m`). Do **not**
  require specific `bookshelf_floor*` keys.
- Add a helper to classify an entry: `is_bookshelf_target(data, key)` (true when
  the entry has `pre_insert` + `insert_depth_m`).
- `resolve_target_pose` already returns the `pre_insert` pose for bookshelf
  entries and the `xyz`/`quat` pose for simple ones — keep that.

### 3. `curobo.py` — `CuRobo.plan_place(...)` (new) + transit helpers
- `plan_place(place_pose, transit_z, bookshelf=False, insert_depth=0, retract_depth=0, joint_state)`
  returns a `PlacePlan(move, insert, retract)` (insert/retract `None` for simple
  drops). All legs are **collision-off** (`_clear_collision_world()` first, then
  chained `plan_pose` from each leg's final joint state — mirrors
  `_plan_pose_segment`).
- **Transit waypoint builder** (pure geometry, unit-tested): from the current
  `tool0` pose (FK via `compute_kinematics`) + the place pose + `transit_z`,
  produce three `tool0` targets:
  1. **lift** → `(current_x, current_y, transit_z)`, current orientation.
  2. **traverse + reorient** → `(place_x, place_y, transit_z)`, place orientation.
  3. **descend** → the full place pose.
  Concatenate the three planned legs into the `move` trajectory.
- **Bookshelf insert/retract:** `insert` = collision-off `plan_pose` to
  `pre_insert + (insert_depth, 0, 0)` (straight +x); `retract` = collision-off
  `plan_pose` to the inserted pose `- (retract_depth, 0, 0)` (straight −x).
- Home reuses the existing collision-aware `plan_trajectory(home_pose, joints)`
  (TSDF with its relaxed fallback). No change there.

*Known limitation:* `plan_pose` between two high configs keeps the object roughly
high but does not hard-constrain the path to `z = transit_z`; more sub-waypoints
can be added during live tuning if the object dips.

### 4. `curobo_service.py`
- Load `place_poses.yml` at startup (share-dir path via
  `ament_index_python.get_package_share_directory('team_8')`, overridable by a
  `place_poses_path` param).
- In `_handle_plan`, add a branch: when `request.goal_name` is non-empty →
  `_handle_place_or_home`:
  - `goal_name == 'home'` → `plan_trajectory(home_pose, joints)` → `trajectory`.
  - otherwise → resolve key, classify simple vs bookshelf, call `plan_place(...)`
    → populate `trajectory` (+ `insert_trajectory`/`retract_trajectory`).
- Existing pick/single-pose branches unchanged.

### 5. `orchestrator.py`
- After the existing pick lift succeeds, run the place→home sequence:
  - obtain the goal key from the `target_goal` param (placeholder seam;
    upstream resolution is out of scope),
  - call the service with `goal_name=key`, execute `trajectory`,
  - if `insert_trajectory` non-empty, execute it,
  - `_send_gripper(closed=False)` to release,
  - if `retract_trajectory` non-empty, execute it,
  - call the service with `goal_name='home'`, execute the returned `trajectory`.
- On any planning/execution failure mid-place: log, best-effort open gripper,
  reset pipeline state (object may be left held — acceptable for now).

## Error handling

- Place planning failure → service returns `success=False`; orchestrator aborts
  the place, opens the gripper (best effort), resets.
- Home planning failure → log; arm is left wherever it released (empty gripper),
  pipeline resets.
- Unknown goal key → service `success=False` with a clear message.

## Testing

- **TDD (host, no GPU):**
  - `place_pose_utils`: data-driven loader/validator + `is_bookshelf_target` +
    `resolve_target_pose` (extend existing `test_place_pose_utils.py`).
  - Transit-waypoint builder: pure-geometry function (current tool pose + place
    pose + `transit_z` → three `tool0` targets) factored so it imports without
    cuRobo/torch, unit-tested for both orientations.
  - Orchestrator place→home sequencing via the existing mock-based
    `test_orchestrator.py` (assert execution order, gripper-release timing,
    insert/retract gating on populated segments).
- **Live (in-sim):** cuRobo planning itself — storage drop, bookshelf
  insert/retract, and home — validated in Gazebo (no GPU on host).

## Out of scope

- NL → goal-key resolution (upstream).
- Pose finding / tuning (poses are authoritative in `place_poses.yml`).
- Bookshelf floor1 (one bookshelf destination is sufficient for now).
- Multi-object orchestration (one item → one goal per task).

## Success criteria

- With a grasped object, the arm lifts to `transit_z`, traverses to above the
  goal, descends, releases, and returns home — without the held object striking
  mapped obstacles on the storage path.
- Bookshelf destinations additionally perform a collision-off +x insert before
  release and −x retract after, then home.
- Phase 3 legs are collision-off; phase 4 (home) is collision-aware.
- Pick (`grasp_poses`) and existing single-pose modes are unchanged.
