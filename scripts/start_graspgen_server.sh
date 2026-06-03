#!/bin/bash
set -euo pipefail

GRASPGEN_REPO_DIR="${GRASPGEN_REPO_DIR:-/opt/GraspGen}"
GRASPGEN_MODELS_DIR="${GRASPGEN_MODELS_DIR:-/opt/GraspGenModels}"
GRIPPER_CONFIG="${GRIPPER_CONFIG:-${GRASPGEN_MODELS_DIR}/checkpoints/graspgen_robotiq_2f_140.yml}"
GRASPGEN_HOST="${GRASPGEN_HOST:-0.0.0.0}"
GRASPGEN_PORT="${GRASPGEN_PORT:-5556}"

if [ ! -d "${GRASPGEN_REPO_DIR}" ]; then
    echo "GraspGen repo not found: ${GRASPGEN_REPO_DIR}" >&2
    exit 1
fi

if [ ! -f "${GRIPPER_CONFIG}" ]; then
    echo "Gripper config not found: ${GRIPPER_CONFIG}" >&2
    exit 1
fi

cd "${GRASPGEN_REPO_DIR}"
exec python3 client-server/graspgen_server.py \
    --gripper_config "${GRIPPER_CONFIG}" \
    --host "${GRASPGEN_HOST}" \
    --port "${GRASPGEN_PORT}"
