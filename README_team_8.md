# Team 8 — Launch Instructions

This image (`image_team_8`) is a **self-contained** AI planning stack for the
`manip_challenge` pick-and-place task. Everything it needs (DDS profile, model
checkpoints, API key, config) is baked in — no compose file, mount, or `.env`
is required. It launches into **standby** and only begins processing once it
receives a command on `/task_commands`.

## Prerequisites (host)

- Docker + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- NVIDIA GPU, driver ≥ 580 (CUDA 12), VRAM ≥ 4 GB
- ROS2 Humble on the host (to run `manip_challenge`)

## Step 1 — Load the image

```bash
docker load -i image_team_8.tar
```

Verify it loaded:

```bash
docker images | grep image_team_8
```

## Step 2 — Launch the simulator (host)

```bash
ros2 launch manip_challenge ur5_setup.launch.py
```

## Step 3 — Launch the planner in standby mode

Run the image. The default command launches the full pipeline
(`ros2 launch pipeline_orchestrator contest_run.launch.py`) and waits for a
task command — this is the **standby** state.

```bash
docker run --rm -it \
  --runtime nvidia --gpus all \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  --network host \
  --ipc host \
  image_team_8:latest
```

The container shares the host network (`--network host`), so all ROS2 topics
from the simulator are visible immediately. Wait until cuRobo finishes
initializing — the log goes quiet and the system idles, waiting on
`/task_commands`. Requests that arrive before initialization completes are not
dropped; they block until the planner is ready.

> **ROS domain:** the image defaults to `ROS_DOMAIN_ID=0`. If the simulator
> runs on a different domain, add `-e ROS_DOMAIN_ID=<id>` to the `docker run`
> command.

## Step 4 — Send a task command

From the host (or any sourced ROS2 shell):

```bash
ros2 topic pub --once /task_commands std_msgs/msg/String "{data: 'banana'}"
```

The pipeline then runs segmentation → grasp generation → motion planning and
executes the trajectory on the UR5.
