#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BASE_IMAGE_TAG="${BASE_IMAGE_TAG:-ros2-ai-planner-base:latest}"
GRASPGEN_REPO_URL="${GRASPGEN_REPO_URL:-https://github.com/pianojay/GraspGen.git}"
GRASPGEN_BRANCH="${GRASPGEN_BRANCH:-jaeuk}"
GRASPGEN_COMMIT="${GRASPGEN_COMMIT:-beddd216a62781670a9b0938e7624b1ea10925f6}"
GRASPGEN_MODELS_REPO_URL="${GRASPGEN_MODELS_REPO_URL:-https://huggingface.co/adithyamurali/GraspGenModels}"
GRASPGEN_MODELS_COMMIT="${GRASPGEN_MODELS_COMMIT:-ec1ccbb5eec0680db669246ac312a3636f16ee43}"

if ! docker image inspect "${BASE_IMAGE_TAG}" >/dev/null 2>&1; then
  "${SCRIPT_DIR}/build_base_image.sh"
fi

cd "${REPO_ROOT}"
PLANNER_BASE_IMAGE="${BASE_IMAGE_TAG}" \
GRASPGEN_REPO_URL="${GRASPGEN_REPO_URL}" \
GRASPGEN_BRANCH="${GRASPGEN_BRANCH}" \
GRASPGEN_COMMIT="${GRASPGEN_COMMIT}" \
GRASPGEN_MODELS_REPO_URL="${GRASPGEN_MODELS_REPO_URL}" \
GRASPGEN_MODELS_COMMIT="${GRASPGEN_MODELS_COMMIT}" \
docker compose build ai_planner
