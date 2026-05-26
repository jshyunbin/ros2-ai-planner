#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BASE_IMAGE_TAG="${BASE_IMAGE_TAG:-ros2-ai-planner-base:latest}"
GRASPGEN_REPO_URL="${GRASPGEN_REPO_URL:-https://github.com/pianojay/GraspGen.git}"
GRASPGEN_BRANCH="${GRASPGEN_BRANCH:-jaeuk}"

if ! docker image inspect "${BASE_IMAGE_TAG}" >/dev/null 2>&1; then
  "${SCRIPT_DIR}/build_base_image.sh"
fi

cd "${REPO_ROOT}"
PLANNER_BASE_IMAGE="${BASE_IMAGE_TAG}" \
GRASPGEN_REPO_URL="${GRASPGEN_REPO_URL}" \
GRASPGEN_BRANCH="${GRASPGEN_BRANCH}" \
docker compose build ai_planner
