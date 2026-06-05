FROM nvcr.io/nvidia/pytorch:23.07-py3

# GraspGenX is vendored in-tree (./GraspGenX) and installed from the COPY below.
# Its gripper_descriptions + checkpoints are auto-fetched by graspgenx into
# ${GRASPGENX_REPO_DIR}/ext on first import (see graspgenx/_setup_dependencies.py).
ARG GRASPGENX_CHECKPOINT_VERSION=release
ARG GRASPGENX_DEFAULT_GRIPPER=robotiq_2f_85

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
ENV PIP_ROOT_USER_ACTION=ignore
ENV GRASPGENX_REPO_DIR=/opt/GraspGenX
# gripper_descriptions + checkpoints land under ${GRASPGENX_REPO_DIR}/ext (graspgenx defaults).
ENV GRASPGENX_CHECKPOINT_VERSION=release
# Dir containing x_grippers/ (gripper_descriptions assets). Shared by the GraspGenX
# server (--assets_dir) AND the ROS nodes' TCP/collision resolver (gripper_tcp.py)
# so both read the same robotiq_2f_85 config (config.json -> fingertip depth).
ENV GRASPGENX_ASSETS_DIR=/opt/GraspGenX/ext/gripper_descriptions/gripper_descriptions/assets
ENV SAM3_MODEL_DIR=/opt/models/sam3
ENV SAM3_MODEL_PATH=/opt/models/sam3/sam3.pt

COPY requirements/ /tmp/requirements/

# Locale
RUN apt-get update && apt-get install -y locales && \
    locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 && \
    rm -rf /var/lib/apt/lists/*

# Base system packages, ROS2 apt source, and git-lfs for Hugging Face model pulls.
RUN apt-get update && apt-get install -y \
    curl \
    git \
    git-lfs \
    gnupg2 \
    lsb-release \
    locales \
    software-properties-common \
    tmux \
    libosmesa6-dev && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
        http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | \
        tee /etc/apt/sources.list.d/ros2.list > /dev/null && \
    git lfs install --system && \
    rm -rf /var/lib/apt/lists/*

# ROS2 Humble + Python tools for planner-side nodes.
RUN apt-get update && apt-get install -y \
    ros-humble-ros-base \
    ros-humble-control-msgs \
    ros-humble-cv-bridge \
    ros-humble-rosidl-default-generators \
    ros-humble-ur-description \
    ros-humble-vision-msgs \
    ros-humble-xacro \
    python3-opencv \
    python3-colcon-common-extensions \
    python3-numpy \
    python3-rosdep \
    python3-pip && \
    rm -rf /var/lib/apt/lists/*

# UR5 URDF used by the cuRobo robot configuration.
RUN bash -c "source /opt/ros/humble/setup.bash && \
    xacro /opt/ros/humble/share/ur_description/urdf/ur.urdf.xacro \
      ur_type:=ur5 name:=ur > /ur5.urdf"

# Additional non-ROS system packages for pointnet2_ops compilation.
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Planner-side Python runtime. Keep this separate from GraspGen so Gemini/SAM3
# issues do not get conflated with GraspGen inference issues.
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir -r /tmp/requirements/planner-runtime.txt && \
    python3 -m pip install --no-cache-dir --no-deps -r /tmp/requirements/sam3.txt

# GraspGenX is vendored in-tree; copy it into the image and install it. GraspGenX
# bundles its own PointNet (graspgenx/models/pointnet/), so there is NO separate
# pointnet2_ops build step, and there is no [serving] extra (serving needs only
# msgpack/pyzmq, already provided). NOTE (build-time-unverified): GraspGenX
# dependency resolution against this image's CUDA/torch stack is validated at
# `docker compose build`, not in the static change pass; numpy is re-pinned
# afterwards to preserve the cv_bridge ABI (mirrors the cuRobo block below).
COPY GraspGenX/ ${GRASPGENX_REPO_DIR}/
RUN cd ${GRASPGENX_REPO_DIR} && \
    python3 -m pip install --no-cache-dir -e . && \
    python3 -m pip install --no-cache-dir --force-reinstall "numpy<2"

# cuRoboV2 runtime. The install order mirrors the validated smoke test against
# the current planner image; re-pin numpy afterwards to preserve cv_bridge ABI.
RUN python3 -m pip install --no-cache-dir "numpy<2" uv && \
    git clone --branch v0.8.0 --depth 1 https://github.com/NVlabs/curobo.git /tmp/curobo && \
    cd /tmp/curobo && \
    uv pip install --system ".[cu12]" && \
    cd / && rm -rf /tmp/curobo && \
    python3 -m pip install --no-cache-dir --force-reinstall "numpy<2"

# Bake the SAM 3 checkpoint into the image to avoid first-run downloads.
# SAM 3 weights are GATED on Hugging Face (facebook/sam3): they are NOT
# auto-downloaded by ultralytics — you must accept Meta's license, get approved,
# and download with an authenticated token. Pass it at build time:
#   docker compose build --build-arg HF_TOKEN=hf_xxx
# (the token account must already have approved access to https://huggingface.co/facebook/sam3)
ARG HF_TOKEN=""
RUN test -n "${HF_TOKEN}" || (echo "ERROR: HF_TOKEN build-arg required to download gated SAM3 weights (facebook/sam3). Build with: docker compose build --build-arg HF_TOKEN=hf_xxx" >&2; exit 1) && \
    python3 -m pip install --no-cache-dir "huggingface_hub>=0.34.0" && \
    mkdir -p ${SAM3_MODEL_DIR} && \
    HF_TOKEN="${HF_TOKEN}" python3 -c "import os, shutil; from huggingface_hub import hf_hub_download; p = hf_hub_download(repo_id='facebook/sam3', filename='sam3.pt', token=os.environ['HF_TOKEN']); shutil.copy(p, '${SAM3_MODEL_PATH}')" && \
    test -s ${SAM3_MODEL_PATH}

# Pre-fetch GraspGenX gripper_descriptions + checkpoints at build time so the
# first server start is fast. graspgenx auto-clones them into
# ${GRASPGENX_REPO_DIR}/ext on import (gripper_descriptions from github;
# checkpoints >1GB from HF via git-lfs, laid out as <version>/{gen,dis} with
# version "${GRASPGENX_CHECKPOINT_VERSION}"). Best-effort: if the prefetch is
# skipped/fails, the server fetches the assets on first import at runtime.
# NOTE (build-time-unverified): asset availability/layout is validated at
# `docker compose build`, not in the static change pass.
RUN cd ${GRASPGENX_REPO_DIR} && python3 -c "import graspgenx" || \
    echo "WARNING: GraspGenX asset prefetch incomplete; assets will be fetched on first import."

# ── Application layer (frequently changing; kept last so the heavy layers
#    above stay cached across src edits) ──────────────────────────────────────
WORKDIR /ros2_ws
COPY src/ src/
COPY scripts/ scripts/
COPY misc/ misc/
RUN . /opt/ros/humble/setup.sh && \
    colcon build --symlink-install

COPY scripts/entrypoint.sh /entrypoint.sh
COPY scripts/start_graspgen_server.sh /start_graspgen_server.sh
RUN chmod +x /entrypoint.sh && chmod +x /start_graspgen_server.sh

# ── Self-contained contest submission layer ─────────────────────────────────
# Bake the DDS profile/config and runtime env so the image runs from a single
# `docker run` with no compose file, mounts, or .env. Kept last so the heavy
# layers above stay cached.
COPY config/ /ros2_ws/config/
RUN mkdir -p /artifacts/segmentation_service /artifacts/graspgen_service

# The Gemini key is supplied at build time so it is NOT stored in source
# control. Bake it into the submission image with:
#   docker build --build-arg GEMINI_API_KEY=<key> -t image_team_8:latest .
ARG GEMINI_API_KEY=""

ENV ROS_DOMAIN_ID=0
ENV ROS_LOCALHOST_ONLY=0
ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ENV FASTDDS_BUILTIN_TRANSPORTS=UDPv4
ENV FASTRTPS_DEFAULT_PROFILES_FILE=/ros2_ws/config/fastdds_no_shm.xml
ENV GEMINI_API_KEY=${GEMINI_API_KEY}

ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "launch", "team_8", "contest_run.launch.py"]
