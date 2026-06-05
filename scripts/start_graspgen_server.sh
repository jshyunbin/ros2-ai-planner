#!/bin/bash
set -euo pipefail

# Launch the GraspGenX ZMQ inference server (robotiq_2f_85 by default).
#
# Paths/assets are populated by the Docker build (GraspGenX repo + GraspGenXModel
# checkpoints + gripper_descriptions). All values are overridable via env.

GRASPGENX_REPO_DIR="${GRASPGENX_REPO_DIR:-/opt/GraspGenX}"
# Checkpoints are auto-fetched by graspgenx into ${GRASPGENX_REPO_DIR}/ext/graspgenx_checkpoints.
GRASPGENX_MODELS_DIR="${GRASPGENX_MODELS_DIR:-${GRASPGENX_REPO_DIR}/ext/graspgenx_checkpoints}"
# Checkpoint version dir as published by adithyamurali/GraspGenXModel (default tag).
GRASPGENX_CHECKPOINT_VERSION="${GRASPGENX_CHECKPOINT_VERSION:-release}"
# --config wants the generator config YAML file, not a directory.
GRASPGENX_CONFIG="${GRASPGENX_CONFIG:-${GRASPGENX_MODELS_DIR}/${GRASPGENX_CHECKPOINT_VERSION}/gen/config.yaml}"
# --assets_dir must contain x_grippers/ (+ proc_grippers/), populated from the
# gripper_descriptions clone (GraspGenX's own assets/x_grippers is empty).
GRASPGENX_ASSETS_DIR="${GRASPGENX_ASSETS_DIR:-${GRASPGENX_REPO_DIR}/ext/gripper_descriptions/gripper_descriptions/assets}"
GRASPGENX_DEFAULT_GRIPPER="${GRASPGENX_DEFAULT_GRIPPER:-robotiq_2f_85}"
GRASPGEN_HOST="${GRASPGEN_HOST:-0.0.0.0}"
GRASPGEN_PORT="${GRASPGEN_PORT:-5556}"

if [ ! -d "${GRASPGENX_REPO_DIR}" ]; then
    echo "GraspGenX repo not found: ${GRASPGENX_REPO_DIR}" >&2
    exit 1
fi

if [ ! -f "${GRASPGENX_CONFIG}" ]; then
    echo "GraspGenX model config not found: ${GRASPGENX_CONFIG}" >&2
    exit 1
fi

if [ ! -d "${GRASPGENX_ASSETS_DIR}" ]; then
    echo "GraspGenX assets dir not found: ${GRASPGENX_ASSETS_DIR}" >&2
    exit 1
fi

cd "${GRASPGENX_REPO_DIR}"
exec python3 client-server/graspgenx_server.py \
    --config "${GRASPGENX_CONFIG}" \
    --assets_dir "${GRASPGENX_ASSETS_DIR}" \
    --default_gripper "${GRASPGENX_DEFAULT_GRIPPER}" \
    --host "${GRASPGEN_HOST}" \
    --port "${GRASPGEN_PORT}"
