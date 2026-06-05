# GraspGenX Standalone Server

GraspGenX can be run as a standalone ZMQ server so that any application — on the same machine or across the network — can request **cross-embodiment** 6-DOF grasp predictions without importing the model code or needing a GPU on the client side.

```
┌──────────────────────┐         ZMQ (tcp)            ┌──────────────────────┐
│   Client (any lang)  │  ── point cloud + gripper ─▶ │  GraspGenX Server    │
│   - Python / C++ / … │  ◀── grasps + scores ──────  │  - GPU, model loaded │
│   - No CUDA needed   │                              │  - One process, many │
│                      │                              │    grippers          │
└──────────────────────┘                              └──────────────────────┘
```

Unlike GraspGen (single gripper per server), GraspGenX loads a **single cross-embodiment model** and supports **any gripper** at inference time. The client specifies `gripper_name` per request; the server loads the gripper's **sweep volume v2** data from the assets directory lazily, then caches the sampler.

## Install

The serving layer is gated behind an optional extra so inference-only users don't pay the ZMQ/msgpack deps. On the server-side machine:

```bash
# From the GraspGenX repo root (inside the uv venv from the main install):
uv pip install pyzmq msgpack msgpack-numpy
# — or, more durably (survives uv sync):
uv sync --extra serve
```

The client-side machine only needs `pyzmq`, `msgpack`, `msgpack-numpy`, `numpy`, and `trimesh` — no PyTorch or CUDA.

## Start the server

```bash
# Activate the GraspGenX uv venv first.
python client-server/graspgenx_server.py \
    --config <repo>/ext/graspgenx_checkpoints/release \
    --assets_dir <repo>/assets \
    --default_gripper franka_panda \
    --port 5556
```

`--config` is the **checkpoint root** containing `gen/` and `dis/` subdirectories — *not* the inner `config.yaml`. For the default checkpoint shipped with the release, that's `<repo>/ext/graspgenx_checkpoints/release/`.

`--default_gripper` is optional. If set, the gripper is pre-loaded at startup and used when a client omits `gripper_name`.

## Run the client

```bash
# Run inference from a mesh file:
python client-server/graspgenx_client.py \
    --mesh_file /path/to/box.obj --mesh_scale 1.0 \
    --gripper_name franka_panda \
    --host localhost --port 5556

# Or from a point cloud file (.pcd / .ply / .xyz / .npy):
python client-server/graspgenx_client.py \
    --pcd_file assets/sample_data/object_mesh/banana.obj \
    --gripper_name robotiq_2f_140 \
    --host localhost --port 5556

# Add --visualize to render the result in a viser web viewer at :8080.
```

## Python Client API

```python
from graspgenx.serving import GraspGenXClient

with GraspGenXClient(host="localhost", port=5556) as client:
    # Server info (cached after first call).
    print(client.server_metadata)
    # {
    #   "default_gripper": "franka_panda",
    #   "loaded_grippers": ["franka_panda"],
    #   "model": {
    #     "generator_backbone": "ptv3vanilla",
    #     "discriminator_backbone": "ptv3vanilla",
    #     "grasp_repr": "r3_so3",
    #     "num_diffusion_iters_eval": 20,
    #   },
    #   "assets_dir": "/.../graspgenx/assets",
    # }

    # Run inference — specify the gripper for each request, or rely on default.
    grasps, confidences = client.infer(
        point_cloud,                  # (N, 3) numpy float32
        gripper_name="franka_panda",  # gripper whose sweep volume v2 conditions the model
        num_grasps=200,               # diffusion samples
        grasp_threshold=-1.0,         # -1.0 ⇒ use top-k instead of threshold
        topk_num_grasps=100,          # return top-k by confidence
    )
    # grasps:       (M, 4, 4) float32 — SE(3) poses
    # confidences:  (M,)      float32 — discriminator scores in [0, 1]

    # Try a different gripper on the same object — no server restart needed.
    grasps_rq, conf_rq = client.infer(point_cloud, gripper_name="robotiq_2f_140")
```

## Supported Grippers

Anything with a directory under your assets root. The default `gripper_descriptions` checkout ships with:

`abb_yumi`, `arx_x5`, `barrett_hand`, `bd_spot`, `dh_ag95`, `ezgripper`, `fetch_robot`, `franka_panda`, `franka_umi`, `galaxea_g1`, `inspire_hand`, `onrobot_RG2`, `onrobot_RG6`, `piper_hand`, `robotiq_2f_85`, `robotiq_2f_140`, `robotiq_3f`, `robotiq_hande`, `sawyer_hand`, `schunk_wsg50`, `sharpa_wave`, `surge_hand`, `tesollo_delto2f`, `unitree_g1`, `wuji_hand`, `xarm_hand`, `yam_4310`.

Plus 32 procedural grippers under `assets/proc_grippers/` (the training set; see the main README for the full list).

## Protocol Reference

Wire format: **msgpack** (with `msgpack_numpy.patch()` so numpy arrays travel natively) over a **ZMQ REQ/REP** socket.

| Request | Fields | Response |
|---------|--------|----------|
| `{"action": "health"}` | — | `{"status": "ok"}` |
| `{"action": "metadata"}` | — | `{"default_gripper": str?, "loaded_grippers": [str], "model": {...}, "assets_dir": str}` |
| `{"action": "infer", ...}` | `point_cloud` (N,3 float32), `gripper_name` (str, optional if server has a default), `num_grasps` (int=200), `grasp_threshold` (float=-1.0), `topk_num_grasps` (int=100) | `{"grasps": (K,4,4) float32, "confidences": (K,) float32, "gripper_name": str, "timing": {"infer_ms": float}}` |

On error: `{"error": "<ExceptionType>: <message>"}`. The Python client raises `RuntimeError`.

The protocol is dead-simple to drive from C++/Rust/etc. with any ZMQ + msgpack bindings.
