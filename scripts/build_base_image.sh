#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BASE_IMAGE_TAG="${BASE_IMAGE_TAG:-ros2-ai-planner-base:latest}"
GRASPGEN_REPO_URL="${GRASPGEN_REPO_URL:-https://github.com/pianojay/GraspGen.git}"
GRASPGEN_BRANCH="${GRASPGEN_BRANCH:-jaeuk}"

docker build \
  -f "${REPO_ROOT}/Dockerfile.base" \
  -t "${BASE_IMAGE_TAG}" \
  --build-arg "GRASPGEN_REPO_URL=${GRASPGEN_REPO_URL}" \
  --build-arg "GRASPGEN_BRANCH=${GRASPGEN_BRANCH}" \
  "${REPO_ROOT}"
