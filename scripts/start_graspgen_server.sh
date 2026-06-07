#!/bin/bash
set -euo pipefail

GRASPGENX_REPO_DIR="${GRASPGENX_REPO_DIR:-/opt/GraspGenX}"
GRASPGENX_CHECKPOINT_DIR="${GRASPGENX_CHECKPOINT_DIR:-/opt/GraspGenXModel/release}"
GRASPGENX_GRIPPER="${GRASPGENX_GRIPPER:-robotiq_2f_85}"
GRASPGENX_HOST="${GRASPGENX_HOST:-0.0.0.0}"
GRASPGENX_PORT="${GRASPGENX_PORT:-5556}"

if [ ! -d "${GRASPGENX_REPO_DIR}" ]; then
    echo "GraspGenX repo not found: ${GRASPGENX_REPO_DIR}" >&2
    exit 1
fi

if [ ! -d "${GRASPGENX_CHECKPOINT_DIR}/gen" ]; then
    echo "GraspGenX checkpoints not found: ${GRASPGENX_CHECKPOINT_DIR}/gen" >&2
    exit 1
fi

# Prefer the downloaded gripper_descriptions as assets_dir if available,
# because it contains processed robotiq_2f_85 assets (tsdf.npy, coll_mesh.obj).
# Fall back to the built-in assets directory.
if [ -d "${GRASPGENX_REPO_DIR}/ext/gripper_descriptions" ]; then
    ASSETS_DIR="${GRASPGENX_REPO_DIR}/ext/gripper_descriptions"
else
    ASSETS_DIR="${GRASPGENX_REPO_DIR}/assets"
fi

# Generate missing LFS assets (tsdf.npy, coll_mesh.obj, vis_mesh.obj) for the
# active gripper from config.json if they were not downloaded (LFS budget issue).
GRIPPER_ASSET_DIR="${ASSETS_DIR}/gripper_descriptions/assets/x_grippers/${GRASPGENX_GRIPPER}"
if [ ! -d "${GRIPPER_ASSET_DIR}" ]; then
    # Fallback path used by some gripper_descriptions layouts
    GRIPPER_ASSET_DIR="${ASSETS_DIR}/assets/x_grippers/${GRASPGENX_GRIPPER}"
fi
if [ -f "${GRIPPER_ASSET_DIR}/config.json" ]; then
    python3 - "${GRIPPER_ASSET_DIR}" << 'PYEOF'
import sys, json, os
import numpy as np
import trimesh

d = sys.argv[1]
cfg = json.load(open(f'{d}/config.json'))

for fname in ('coll_mesh.obj', 'vis_mesh.obj'):
    if not os.path.exists(f'{d}/{fname}') or os.path.getsize(f'{d}/{fname}') == 0:
        bbox_min = np.array(cfg['bbox'][0])
        bbox_max = np.array(cfg['bbox'][1])
        mesh = trimesh.creation.box(
            extents=bbox_max - bbox_min,
            transform=trimesh.transformations.translation_matrix((bbox_min + bbox_max) / 2))
        mesh.export(f'{d}/{fname}')
        print(f'[start_graspgen_server] Generated {fname} from bbox.')

tsdf_path = f'{d}/tsdf.npy'
if not os.path.exists(tsdf_path) or os.path.getsize(tsdf_path) == 0:
    np.save(tsdf_path, {
        'open_tsdf':  np.zeros((64, 32, 64), dtype=np.float16),
        'close_tsdf': np.zeros((64, 32, 64), dtype=np.float16),
    })
    print('[start_graspgen_server] Generated tsdf.npy (zero dummy).')
PYEOF
fi

exec python3 "${GRASPGENX_REPO_DIR}/client-server/graspgenx_server.py" \
    --config "${GRASPGENX_CHECKPOINT_DIR}/gen/config.yaml" \
    --assets_dir "${ASSETS_DIR}" \
    --default_gripper "${GRASPGENX_GRIPPER}" \
    --host "${GRASPGENX_HOST}" \
    --port "${GRASPGENX_PORT}"
