#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BASE_IMAGE_TAG="${BASE_IMAGE_TAG:-ros2-ai-planner-base:latest}"
GRASPGEN_REPO_URL="${GRASPGEN_REPO_URL:-https://github.com/pianojay/GraspGen.git}"
GRASPGEN_BRANCH="${GRASPGEN_BRANCH:-jaeuk}"
GRASPGEN_COMMIT="${GRASPGEN_COMMIT:-31b67f65f3cb88928887edd2ee24e302c30cab70}"
GRASPGEN_MODELS_REPO_URL="${GRASPGEN_MODELS_REPO_URL:-https://huggingface.co/adithyamurali/GraspGenModels}"
GRASPGEN_MODELS_COMMIT="${GRASPGEN_MODELS_COMMIT:-ec1ccbb5eec0680db669246ac312a3636f16ee43}"

echo "Building ${BASE_IMAGE_TAG}"
echo "GraspGen repo   : ${GRASPGEN_REPO_URL}"
echo "GraspGen branch : ${GRASPGEN_BRANCH}"
echo "GraspGen commit : ${GRASPGEN_COMMIT}"

docker build \
  -f "${REPO_ROOT}/Dockerfile.base" \
  -t "${BASE_IMAGE_TAG}" \
  --build-arg "GRASPGEN_REPO_URL=${GRASPGEN_REPO_URL}" \
  --build-arg "GRASPGEN_BRANCH=${GRASPGEN_BRANCH}" \
  --build-arg "GRASPGEN_COMMIT=${GRASPGEN_COMMIT}" \
  --build-arg "GRASPGEN_MODELS_REPO_URL=${GRASPGEN_MODELS_REPO_URL}" \
  --build-arg "GRASPGEN_MODELS_COMMIT=${GRASPGEN_MODELS_COMMIT}" \
  "${REPO_ROOT}"
