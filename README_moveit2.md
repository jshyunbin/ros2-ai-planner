# MoveIt2 + Manip Challenge Pipeline Setup

This guide explains how to run the manip challenge Gazebo/controller environment on the host machine and use the Docker-based `pipeline_orchestrator` MoveIt2 code to plan and execute robot motions.

host
```commandline
cd /home/jiseong/cs477_ws/src/cs477_IIR

source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch manip_challenge ur5_setup_set2_picking.launch.py
```
docker terminal 1
```commandline
cd ~/manip/ros2-ai-planner/src

sudo docker compose run --rm ai_planner bash

ros2 launch pipeline_orchestrator moveit2_stack.launch.py use_sim_time:=true
```

docker terminal 2
```commandline
sudo docker ps

sudo docker exec -it <container_id_or_name> bash

cd /ros2_ws

source /opt/ros/humble/setup.bash
source /home/jiseong/cs477_ws/src/cs477_IIR/install/setup.bash

colcon build --symlink-install --packages-select pipeline_orchestrator

source /ros2_ws/install/setup.bash

ros2 run pipeline_orchestrator add_static_scene_to_moveit
```

move to bookshelf
```commandline
ros2 run pipeline_orchestrator move_to_bookshelf_center --ros-args \
  -p use_sim_time:=true \
  -p moveit2.group_name:=ur5_arm \
  -p moveit2.end_effector_link:=tool0 \
  -p moveit2.planning_frame:=world \
  -p moveit2.move_action_name:=/move_action \
  -p moveit2.pipeline_id:=move_group \
  -p 'moveit2.planner_id:=ur5_arm[RRTConnectkConfigDefault]' \
  -p moveit2.allowed_planning_time:=15.0 \
  -p moveit2.num_planning_attempts:=10 \
  -p moveit2.velocity_scaling:=0.3 \
  -p moveit2.acceleration_scaling:=0.3 \
  -p moveit2.use_orientation_constraint:=true \
  -p moveit2.orientation_tolerance:=1.0 \
  -p moveit2.position_tolerance:=0.005 \
  -p shelf_level:=upper \
  -p bookshelf_center_x:=0.95 \
  -p bookshelf_center_y:=-0.30 \
  -p upper_target_z:=1.15
```


```commandline
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
```

towards the bookshelf
```commandline
ros2 run pipeline_orchestrator move_to_pose --ros-args \
  -p use_sim_time:=true \
  -p moveit2.group_name:=ur5_arm \
  -p moveit2.end_effector_link:=tool0 \
  -p moveit2.planning_frame:=world \
  -p moveit2.move_action_name:=/move_action \
  -p moveit2.pipeline_id:=move_group \
  -p 'moveit2.planner_id:=ur5_arm[RRTConnectkConfigDefault]' \
  -p moveit2.allowed_planning_time:=15.0 \
  -p moveit2.num_planning_attempts:=10 \
  -p moveit2.velocity_scaling:=0.3 \
  -p moveit2.acceleration_scaling:=0.3 \
  -p moveit2.use_orientation_constraint:=true \
  -p moveit2.orientation_tolerance:=1.0 \
  -p moveit2.position_tolerance:=0.005 \
  -p use_approach_pose:=false \
  -p target_x:=0.65 \
  -p target_y:=-0.30 \
  -p target_z:=1.05 \
  -p qx:=0.0 \
  -p qy:=0.7071 \
  -p qz:=0.0 \
  -p qw:=0.7071
```
robotiq_85_left_finger_tip_link
tool0

45 degree down
```commandline
ros2 run pipeline_orchestrator move_to_pose --ros-args \
  -p use_sim_time:=true \
  -p moveit2.group_name:=ur5_arm \
  -p moveit2.end_effector_link:=robotiq_85_left_finger_tip_link\
  -p moveit2.planning_frame:=world \
  -p moveit2.move_action_name:=/move_action \
  -p moveit2.pipeline_id:=move_group \
  -p 'moveit2.planner_id:=ur5_arm[RRTConnectkConfigDefault]' \
  -p moveit2.allowed_planning_time:=15.0 \
  -p moveit2.num_planning_attempts:=10 \
  -p moveit2.velocity_scaling:=0.3 \
  -p moveit2.acceleration_scaling:=0.3 \
  -p moveit2.use_orientation_constraint:=true \
  -p moveit2.orientation_tolerance:=1.0 \
  -p moveit2.position_tolerance:=0.005 \
  -p use_approach_pose:=false \
  -p target_x:=0.65 \
  -p target_y:=-0.30 \
  -p target_z:=1.05 \
  -p qx:=0.0 \
  -p qy:=0.9239 \
  -p qz:=0.0 \
  -p qw:=0.3827
```

add object
```commandline
ros2 run pipeline_orchestrator fixed_object_scene --ros-args   -p object_name:=moveit_test_box   -p object_x:=0.50   -p object_y:=0.00   -p object_z:=0.535   -p size_x:=0.05   -p size_y:=0.05   -p size_z:=0.05   -p add_to_moveit_scene:=false
```

## 1. docker-compose.yml

The container must use host networking so that Docker can see the ROS graph from the host Gazebo simulation.

Example:

```yaml
services:
  ai_planner:
    build: .
    network_mode: host
    ipc: host
    environment:
      - ROS_DOMAIN_ID=0
      - ROS_LOCALHOST_ONLY=0
    volumes:
      - ./src:/ros2_ws/src
      - /home/jiseong/cs477_ws/src/cs477_IIR:/home/jiseong/cs477_ws/src/cs477_IIR:ro
    stdin_open: true
    tty: true
```

Change this line for each computer:

```yaml
- /your/workspace/dir:/home/jiseong/cs477_ws/src/cs477_IIR:ro
```
---

## 2. Run Gazebo/controller on the host

Open a host terminal, not Docker.

```bash
cd ~/cs477_ws/src/cs477_IIR

source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch manip_challenge ur5_setup_set2_picking.launch.py
```

When the prompt appears, select the position controller:

```text
1
1
```
## 3. Build pipeline_orchestrator inside Docker

Open a Docker shell:

```bash
cd ~/manip/ros2-ai-planner/src
sudo docker compose run --rm ai_planner bash
```

Inside Docker:

```bash
cd /ros2_ws

source /opt/ros/humble/setup.bash
source /home/jiseong/cs477_ws/src/cs477_IIR/install/setup.bash

colcon build --symlink-install --packages-select pipeline_orchestrator
source /ros2_ws/install/setup.bash
```

Check that the host simulation and controllers are active:

```bash
ros2 topic echo /joint_states --once
ros2 action list | grep follow_joint
ros2 control list_controllers
```

Expected actions:

```text
/ur5_controller/follow_joint_trajectory
/gripper_controller/follow_joint_trajectory
```

Expected controller state:

```text
ur5_controller          active
gripper_controller      active
joint_state_broadcaster active
```

Do not run a host launch file that also starts MoveIt2.
The host should run Gazebo and controllers only.

---

## 4. Run MoveIt2 stack in Docker

Open a Docker terminal.

```bash
cd ~/manip/ros2-ai-planner/src
sudo docker compose run --rm ai_planner bash
```

Inside Docker:

```bash
source /opt/ros/humble/setup.bash
source /home/jiseong/cs477_ws/src/cs477_IIR/install/setup.bash
source /ros2_ws/install/setup.bash

ros2 launch pipeline_orchestrator moveit2_stack.launch.py use_sim_time:=true
```

In another Docker terminal, enter the same container:

```bash
sudo docker ps
sudo docker exec -it <container_id_or_name> bash
```

Then check:

```bash
source /opt/ros/humble/setup.bash
source /home/jiseong/cs477_ws/src/cs477_IIR/install/setup.bash
source /ros2_ws/install/setup.bash

ros2 action info /move_action
ros2 service list | grep planning_scene
```

Expected:

```text
Action servers: 1
/apply_planning_scene
/get_planning_scene
```

If `Action servers: 2` appears, two `move_group` processes are running. Stop one of them. Usually this means MoveIt2 was launched both on the host and inside Docker.

---

## 5. Pose Generation in `orchestrator.py`

The orchestrator does not move directly to the raw grasp pose returned by GraspGen. Instead, it converts each grasp candidate into a sequence of checkpoint poses for a safer pick-and-place motion.

The generated sequence is:

```text
pregrasp
→ grasp
→ lift
→ place_above
→ place
→ retreat
```

### 1. GraspGen output format

`GraspGen` may return one grasp pose or multiple grasp pose candidates.

The orchestrator accepts the following formats:

```text
PoseStamped
Pose
list[PoseStamped]
list[Pose]
dict with x, y, z, qx, qy, qz, qw
list[dict]
```

A dictionary grasp pose should look like this:

```python
{
    "frame_id": "world",
    "x": 0.50,
    "y": 0.00,
    "z": 0.76,
    "qx": -0.019,
    "qy": 1.000,
    "qz": 0.014,
    "qw": 0.007,
}
```

If `frame_id` is not provided, the orchestrator uses the MoveIt2 planning frame. The default planning frame is usually `world`.

All quaternions are normalized before planning.

---

### 2. How checkpoint poses are generated

The raw GraspGen pose becomes the base `grasp` pose.

From this pose, the orchestrator generates the following checkpoint poses:

```text
grasp:
  The actual grasp pose from GraspGen.

pregrasp:
  Same x, y, orientation as grasp.
  z is placed above grasp.

lift:
  Same x, y, orientation as grasp.
  z is placed above grasp after closing the gripper.

place_above:
  Uses the target basket x, y.
  z is set to a safe height above the basket.

place:
  Uses the target basket x, y.
  z is lowered into the basket.

retreat:
  Uses the target basket x, y.
  z is raised again after releasing the object.
```

The sequence is then executed as:

```text
gripper_open()
move_to(pregrasp)
move_to(grasp)
gripper_close()
move_to(lift)
move_to(place_above)
move_to(place)
gripper_open()
move_to(retreat)
```

The object collision removal step is not used in the current version. The orchestrator simply plans through the generated checkpoint poses and controls the gripper separately.

---

### 3. Pose generation parameters

These parameters control how the checkpoint poses are generated.

```bash
sequence.grasp_z_override
sequence.pregrasp_z_override
sequence.lift_z_override
sequence.pregrasp_offset_z
sequence.lift_offset_z
sequence.place_x
sequence.place_y
sequence.place_above_z
sequence.place_z
sequence.retreat_z
```

#### Grasp pose

By default, the grasp pose uses the `x`, `y`, `z`, and orientation returned by GraspGen.

If `sequence.grasp_z_override > 0`, the grasp z value is replaced:

```text
grasp.z = sequence.grasp_z_override
```

This is useful when GraspGen returns a noisy or unsafe z value.

#### Pregrasp pose

If `sequence.pregrasp_z_override > 0`:

```text
pregrasp.z = sequence.pregrasp_z_override
```

Otherwise:

```text
pregrasp.z = grasp.z + sequence.pregrasp_offset_z
```

Default:

```text
sequence.pregrasp_offset_z = 0.15
```

#### Lift pose

If `sequence.lift_z_override > 0`:

```text
lift.z = sequence.lift_z_override
```

Otherwise:

```text
lift.z = grasp.z + sequence.lift_offset_z
```

Default:

```text
sequence.lift_offset_z = 0.30
```

#### Place poses

The basket target is controlled by:

```text
sequence.place_x
sequence.place_y
```

## Default Pose Offsets and Placement Targets

This section summarizes the default pose generation values used by `orchestrator.py`.

The orchestrator generates pick-and-place checkpoint poses from a single GraspGen grasp pose:

```text
pregrasp
→ grasp
→ lift
→ place_above
→ place
→ retreat
```

---

### 1. Default pose offset values

By default, the raw GraspGen output becomes the `grasp` pose.

The other poses are generated using these default offsets and target heights:

| Parameter                    | Default | Meaning                                               |
| ---------------------------- | ------: | ----------------------------------------------------- |
| `sequence.pregrasp_offset_z` |  `0.15` | Height added above `grasp.z` to create `pregrasp.z`   |
| `sequence.lift_offset_z`     |  `0.30` | Height added above `grasp.z` to create `lift.z`       |
| `sequence.place_above_z`     |  `1.10` | Safe height above the placement target                |
| `sequence.place_z`           |  `0.84` | Lower placement height inside the target basket/shelf |
| `sequence.retreat_z`         |  `1.05` | Retreat height after releasing the object             |

Default generation rule:

```text
grasp.z      = GraspGen result z
pregrasp.z   = grasp.z + 0.15
lift.z       = grasp.z + 0.30
place_above.z = 1.10
place.z       = 0.84
retreat.z     = 1.05
```

If override parameters are set to a positive value, they replace the generated z value:

```text
sequence.grasp_z_override > 0
  → grasp.z = sequence.grasp_z_override

sequence.pregrasp_z_override > 0
  → pregrasp.z = sequence.pregrasp_z_override

sequence.lift_z_override > 0
  → lift.z = sequence.lift_z_override
```

Recommended default command values:

```bash
-p sequence.pregrasp_offset_z:=0.15 \
-p sequence.lift_offset_z:=0.30 \
-p sequence.place_above_z:=1.10 \
-p sequence.place_z:=0.84 \
-p sequence.retreat_z:=1.05
```

For early testing, keep `place_above_z`, `lift_z`, and `retreat_z` high enough to avoid collision with the table, basket walls, or bookshelf.

---

### 2. Default placement targets in set2

The set2 environment has fixed basket and bookshelf locations. Use these as default place targets.

#### Storage basket A

Use this when placing into the left/positive-y storage basket:

```text
storage_a_basket center:
  x = 0.00
  y = 0.55
```

Recommended parameters:

```bash
-p sequence.place_x:=0.00 \
-p sequence.place_y:=0.55 \
-p sequence.place_above_z:=1.10 \
-p sequence.place_z:=0.84 \
-p sequence.retreat_z:=1.05
```

#### Storage basket B

Use this when placing into the right/negative-y storage basket:

```text
storage_b_basket center:
  x = 0.00
  y = -0.55
```

Recommended parameters:

```bash
-p sequence.place_x:=0.00 \
-p sequence.place_y:=-0.55 \
-p sequence.place_above_z:=1.10 \
-p sequence.place_z:=0.84 \
-p sequence.retreat_z:=1.05
```

#### Workspace basket

Use this when placing into the workspace basket:

```text
workspace_basket center:
  x = 0.55
  y = 0.00
```

Recommended starting parameters:

```bash
-p sequence.place_x:=0.55 \
-p sequence.place_y:=0.00 \
-p sequence.place_above_z:=1.00 \
-p sequence.place_z:=0.70 \
-p sequence.retreat_z:=0.95
```

The workspace basket is lower than the storage baskets, so `place_z` can usually be lower than the storage basket placement height. If the robot collides with the basket wall, increase `place_z` first.

---

### 3. Bookshelf placement targets

The bookshelf is located around:

```text
bookshelf:
  x = 0.95
  y = -0.30
  yaw ≈ -1.57
```

Recommended target positions depend on which shelf is used.

#### Lower shelf

```bash
-p sequence.place_x:=0.95 \
-p sequence.place_y:=-0.30 \
-p sequence.place_above_z:=0.80 \
-p sequence.place_z:=0.58 \
-p sequence.retreat_z:=0.85
```

#### Middle shelf

```bash
-p sequence.place_x:=0.95 \
-p sequence.place_y:=-0.30 \
-p sequence.place_above_z:=1.00 \
-p sequence.place_z:=0.80 \
-p sequence.retreat_z:=1.05
```

#### Upper shelf

```bash
-p sequence.place_x:=0.95 \
-p sequence.place_y:=-0.30 \
-p sequence.place_above_z:=1.20 \
-p sequence.place_z:=1.02 \
-p sequence.retreat_z:=1.25
```

For the bookshelf, begin with the middle shelf because it is usually easier to reach than the lower or upper shelf.

---


