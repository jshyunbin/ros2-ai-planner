# GraspGen Test Plan

Date: 2026-05-25

## Purpose

This document records our current understanding, decisions, risks, and next steps for testing and potentially integrating `GraspGen` into the CS477 IIR manipulation challenge pipeline.

This is meant for:

- current team members
- future teammates
- future me / future Codex sessions


## Workspace Layout

Current root: `/home/user/JW/iir`

Relevant folders:

- `GraspGen/`
- `GraspDataGen/`
- `CS477_IIR_2026S/`
- `ros2-ai-planner/`

Relevant documents:

- `Manipulation_Challenge.pdf`
- `CS477_ IIR Picking Challenge.pdf`
- `[2026-1]CS477 지능로봇공학개론 352a850fabe7801c9565f7a320e503cc.md`


## Challenge Understanding

The challenge environment is a ROS2 + Gazebo pick-and-place setup with:

- fixed robot / storage / shelf / camera layout
- randomly placed known object categories
- high-level text instructions delivered over `/task_commands`
- prohibition on using Gazebo internal object-state topics during the real competition

Expected team-side behavior:

- launch in standby
- wait for task command
- perceive target objects using RGB-D sensors
- execute pick-and-place actions via ROS2 interfaces

Important available sensing and control interfaces from `manip_challenge`:

- `/task_commands`
- top-down RGB-D camera topics under `/camera/camera/...`
- wrist RGB-D camera topics under `/wrist_camera/wrist_camera/...`
- organized point cloud topics under `.../depth/color/points`
- `/joint_states`
- `/ur5_controller/follow_joint_trajectory`
- `/gripper_controller/follow_joint_trajectory`


## Current Technical Conclusion

### Main question

Can we use GraspGen in this challenge?

### Current answer

Yes, but only as one module in a larger perception-to-grasp pipeline.

GraspGen is feasible for:

- receiving an object-centric partial point cloud
- generating 6-DOF grasp candidates
- returning grasp poses + scores

GraspGen is not a full solution for:

- text parsing
- object detection / segmentation
- point cloud extraction from the challenge cameras
- planner feasibility
- motion execution


## Key Design Decision So Far

### Chosen baseline

Use the pretrained `robotiq_2f_140` GraspGen checkpoint first.

### Reason

This is lower risk than trying to:

- generate a brand-new `robotiq_2f_85` dataset
- train a new model from scratch
- debug IsaacLab / GraspDataGen / GraspGen training all at once

### Interpretation

The right first experiment is not:

- "can we train everything from scratch?"

It is:

- "can GraspGen provide useful grasp candidates in the challenge simulation when given a masked partial point cloud?"

If the answer is no, then training a custom model may not be worth the time.


## Current GraspGen Position

### What we know

- `GraspGen` supports:
  - partial point clouds
  - object-centric point clouds
  - scene point clouds
  - collision filtering
- `GraspGen` also supports a standalone ZMQ server mode
- `GraspDataGen` now exists locally and supports `robotiq_2f_85`
- `GraspDataGen` can generate GraspGen-format datasets and auto-create missing `robotiq_2f_85` config files

### But current recommendation remains

Do not start with custom `2f_85` training.

Start with:

- pretrained `2f_140`
- point-cloud input from challenge sensors
- geometric retargeting / filtering
- simulation testing


## Segmentation / Point Cloud Plan

### Current perception assumption

We do not need native 3D instance segmentation first.

Instead, the intended near-term pipeline is:

1. object detection / 2D segmentation in image space
2. use aligned depth or organized point cloud to mask object points
3. build an object-centric point cloud
4. feed that point cloud to GraspGen

This is simpler and more realistic for the current challenge setup.

### Why this is acceptable

The challenge already gives:

- RGB
- depth
- organized point clouds

Therefore we only need:

- 2D mask
- mask-to-point-cloud conversion

not a dedicated point-cloud instance segmentation model.


## Current Integration Recommendation

### Recommended architecture

Use two containers first:

1. `ros2-ai-planner` container as ROS2-side integration client
2. `GraspGen` container as GPU inference server

### Why

This reduces early integration risk.

If we try to put the whole GraspGen stack directly inside `ros2-ai-planner` immediately, failure modes multiply:

- ROS2 issues
- CUDA / PyTorch / model issues
- dependency conflicts
- perception bugs
- planner bugs

The two-container approach isolates concerns:

- ROS2 side handles sensors, masking, transforms, logging
- GraspGen side only handles point cloud -> grasps

### Intended first vertical slice

1. launch `manip_challenge`
2. launch GraspGen server with pretrained `robotiq_2f_140`
3. subscribe to wrist point cloud from ROS2
4. send point cloud to the GPU model
5. receive grasp poses + confidence scores
6. inspect and visualize results

At this stage, no motion execution is required yet.

### Important deployment clarification

The challenge documentation requires team code and dependencies to run inside Docker, and the final submission must be a single Docker image.

Therefore:

- a two-container setup is acceptable as a development/debugging step
- but the final target should be treated as a Docker-contained challenge deployment
- the first technical gate is Docker-side GraspGen inference itself, before ROS2 integration


## ros2-ai-planner Status

Current repo role:

- useful as integration scaffold
- not yet a working end-to-end system

Current limitations:

- GraspGen module is still a stub
- SAM2 module is still a stub
- cuRobo module is still a stub
- MoveIt2 module is still a stub

Branch created for free experimentation:

- `ros2-ai-planner` branch: `jaeuk`

Use this branch for development without worrying about disturbing `main`.


## Immediate Technical Goals

There are now two experiments, depending on the status of the rest of the pipeline.

### Experiment 1: Without segmentation

Goal:

- check that GraspGen inference itself works inside Docker on this machine

Minimal success criteria:

- `GraspGen` Docker image builds
- pretrained `robotiq_2f_140` checkpoint loads
- sample mesh or point-cloud inference returns grasp poses and scores inside Docker

Why this comes first:

- if Docker-side inference fails, ROS2 integration is premature
- this is the smallest test that still matches the challenge deployment constraints

### Experiment 2: With segmentation

Goal:

- check whether pretrained `robotiq_2f_140` GraspGen inference is feasible in the challenge setting when given an object-centric partial point cloud

Minimal success criteria:

- ROS2 node receives `/wrist_camera/wrist_camera/depth/color/points`
- 2D segmentation provides an object mask
- point cloud is converted to `numpy float32 (N, 3)`
- masked object point cloud is sent to GraspGen
- server returns grasp transforms and confidence scores
- outputs are plausible enough to justify filtering and planner testing

If this fails, stop and debug before adding:

- filtering
- planner integration
- training


## Current Blocker

### NVIDIA driver mismatch

At the time of writing, Docker-side GPU inference is blocked by an NVIDIA driver/library mismatch:

- loaded kernel module: `535.288.01`
- installed user-space libraries: `535.309.01`

Additional verification:

- the `535.309.01` DKMS module is already built for the running kernel `6.8.0-90-generic`
- `modinfo nvidia` points to the correct new module on disk

Interpretation:

- the driver installation appears complete
- the system is still running the older loaded NVIDIA module
- a clean reboot is the most likely fix

Practical consequence:

- both Experiment 1 and Experiment 2 are blocked until reboot permission is obtained and the machine is restarted
- after reboot, confirm `nvidia-smi` works before resuming GraspGen work


## Short-Term Plan

### Phase 1: Docker-only GraspGen inference check

Target: as soon as possible

Tasks:

- obtain permission to reboot the server
- reboot the machine
- confirm `nvidia-smi` works correctly after reboot
- obtain or clone `GraspGenModels`
- build `GraspGen` Docker image
- run pretrained `robotiq_2f_140` Docker inference on sample data

Deliverable:

- one working Docker-side GraspGen inference result


### Phase 2: Thin vertical slice with ROS2 point cloud

Tasks:

- run `manip_challenge`
- run GraspGen serve container
- use `ros2-ai-planner` branch `jaeuk`
- subscribe to wrist point cloud
- call GraspGen server from ROS2 side
- print / save returned grasp candidates

Deliverable:

- one working inference path from ROS2 wrist point cloud to grasp candidates


### Phase 3: Object masking

Tasks:

- attach 2D segmentation pipeline
- use 2D mask to filter aligned point cloud
- feed masked object point cloud instead of full wrist cloud
- compare outputs against unmasked scene cloud

Deliverable:

- object-centric point cloud inference


### Phase 4: 2F-85 retargeting and filtering

Tasks:

- transform / align the `2f_140` predicted grasp frame to challenge gripper frame
- reject obviously impossible grasps for the Robotiq 85
- filter by:
  - collision with scene points
  - table / shelf / bin constraints
  - reachability / IK feasibility

Deliverable:

- filtered grasp candidates that are realistic for the challenge robot


### Phase 5: Planner check

Tasks:

- use filtered grasp candidates as planner goals
- test MoveIt2 or cuRobo feasibility
- determine whether the candidates are actually executable

Deliverable:

- at least one successful grasp execution in challenge simulation


## Long-Term Plan

### If pretrained `2f_140` baseline works reasonably well

Continue with:

- stronger filtering
- better segmentation
- task-command parsing
- planner robustness
- multi-object sequential task execution

In this case, avoid training unless clearly necessary.


### If pretrained `2f_140` baseline partially works

Consider:

- adding better frame offsets
- stronger geometry-aware filtering
- object-category-specific heuristics
- fine-tuning or partial retraining later


### If pretrained `2f_140` baseline clearly fails

Then escalate to:

- `GraspDataGen` dataset generation with `robotiq_2f_85`
- training `GraspGen` generator + discriminator for `2f_85`

This is now technically possible because `GraspDataGen` provides:

- a built-in `robotiq_2f_85` gripper config
- GraspGen-format output generation
- auto-generation of required `GraspGen/config/grippers/robotiq_2f_85.yaml` and `.py`

But this should be treated as a second-stage effort, not first priority.


## Risk Assessment

### Low-risk assumptions

- RGB-D + point cloud access from the challenge environment is available
- masking a point cloud from 2D segmentation is straightforward
- GraspGen can accept partial point clouds
- two-container client/server inference is a reasonable architecture

### Medium-risk items

- `2f_140` grasps may need significant retargeting for `2f_85`
- challenge clutter may make static-view partial clouds unreliable
- planners may reject many visually good grasp candidates
- final single-container packaging may be less convenient than the two-container debug setup

### High-risk items

- full training from scratch within the available time
- IsaacLab / GraspDataGen environment issues
- GPU / NVIDIA driver mismatch issues on the current machine
- assuming a visually plausible grasp is physically executable in challenge sim


## Resource Notes

### System RAM

Host RAM appears sufficient for this work.

Point cloud memory itself is not a major concern.

A typical organized RGB-D point cloud is relatively small compared with model memory.

### Actual likely bottleneck

GPU runtime / CUDA / driver health is a more serious concern than host RAM.

At one point, `nvidia-smi` was failing with:

- `Driver/library version mismatch`

This must be kept in mind whenever Docker / GPU inference behaves unexpectedly.


## Open Questions

- How well do `robotiq_2f_140` grasp priors transfer to `robotiq_2f_85` in the challenge environment?
- Is wrist view alone sufficient, or will we need top-down masking first and wrist view only for refinement?
- Which planner is the first practical execution target: MoveIt2 or cuRobo?
- How much scene collision filtering is needed before grasp candidates become planner-usable?
- What is the cleanest path from the two-container debug setup to the required final single-image submission?


## Current Recommendation Summary

If we do only one thing next, it should be this:

- get reboot permission, reboot the machine, and verify Docker-side GraspGen inference works with the pretrained `robotiq_2f_140` checkpoint

Why:

- current GPU driver mismatch blocks reliable inference
- Docker-only inference is the smallest valid technical gate
- it avoids mixing ROS2, segmentation, and model issues too early
- it gives the clearest answer about whether GraspGen is usable here at all


## Decision Log

### Decided

- We will not begin with native point-cloud instance segmentation.
- We will use 2D segmentation -> point cloud masking first.
- We will not prioritize full `2f_85` training immediately.
- We will first test pretrained `robotiq_2f_140`.
- We will use `ros2-ai-planner` as the ROS2-side development workspace.
- We created branch `jaeuk` for this work.
- We will separate the work into two experiments:
  - Docker-only GraspGen inference without segmentation
  - challenge-side feasibility test with segmentation
- We will treat reboot permission as the immediate prerequisite for both experiments.

### Not decided yet

- final deployment architecture for competition submission
- whether `2f_85` retraining is necessary
- whether cuRobo or MoveIt2 becomes the primary planner
