# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04
ARG GRASPGENX_CHECKPOINT_VERSION=release
ARG GRASPGENX_DEFAULT_GRIPPER=robotiq_2f_85

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
ENV PIP_ROOT_USER_ACTION=ignore
ENV GRASPGENX_REPO_DIR=/opt/GraspGenX
ENV GRASPGENX_CHECKPOINT_VERSION=release
ENV GRASPGENX_ASSETS_DIR=/opt/GraspGenX/ext/gripper_descriptions/gripper_descriptions/assets
ENV GRASPGENX_GRIPPER_CFG_DIR=/opt/GraspGenX/ext/gripper_descriptions
ENV SAM3_MODEL_DIR=/opt/models/sam3
ENV SAM3_MODEL_PATH=/opt/models/sam3/sam3.pt

COPY requirements/ /tmp/requirements/

# Locale
RUN apt-get update && apt-get install -y locales && \
    locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    git \
    git-lfs \
    gnupg2 \
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

# C/C++ toolchain + Python headers — needed to compile cuRobo's CUDA extensions.
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev && \
    rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install \
      --index-url https://download.pytorch.org/whl/cu128 \
      torch==2.7.1 torchvision==0.22.1

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install -r /tmp/requirements/planner-runtime.txt && \
    python3 -m pip install --no-deps -r /tmp/requirements/sam3.txt

COPY GraspGenX/ ${GRASPGENX_REPO_DIR}/
RUN --mount=type=cache,target=/root/.cache/pip \
    cd ${GRASPGENX_REPO_DIR} && \
    python3 -m pip install -e . && \
    python3 -m pip install --force-reinstall "numpy<2"

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/uv \
    python3 -m pip install "numpy<2" uv && \
    git clone --branch v0.8.0 --depth 1 https://github.com/NVlabs/curobo.git /tmp/curobo && \
    cd /tmp/curobo && \
    TORCH_CUDA_ARCH_LIST="12.0+PTX" \
    uv pip install --system ".[cu12]" && \
    cd / && rm -rf /tmp/curobo && \
    python3 -m pip install --force-reinstall "numpy<2"

ARG HF_TOKEN=""
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/huggingface \
    test -n "${HF_TOKEN}" || (echo "ERROR: HF_TOKEN build-arg required to download gated SAM3 weights (facebook/sam3). Build with: docker compose build --build-arg HF_TOKEN=hf_xxx" >&2; exit 1) && \
    python3 -m pip install "huggingface_hub>=0.34.0" && \
    mkdir -p ${SAM3_MODEL_DIR} && \
    HF_TOKEN="${HF_TOKEN}" python3 -c "import os, shutil; from huggingface_hub import hf_hub_download; p = hf_hub_download(repo_id='facebook/sam3', filename='sam3.pt', token=os.environ['HF_TOKEN']); shutil.copy(p, '${SAM3_MODEL_PATH}')" && \
    test -s ${SAM3_MODEL_PATH}

RUN --mount=type=cache,target=/root/.cache/huggingface \
    HF_TOKEN="${HF_TOKEN}" python3 -c "import os; from huggingface_hub import snapshot_download; snapshot_download(repo_id='adithyamurali/GraspGenXModel', local_dir='${GRASPGENX_REPO_DIR}/ext/graspgenx_checkpoints', token=os.environ.get('HF_TOKEN') or None)"
# Sanity-check the import now that assets are present.
RUN cd ${GRASPGENX_REPO_DIR} && python3 -c "import graspgenx" || \
    echo "WARNING: GraspGenX import check failed."

WORKDIR /ros2_ws
COPY src/ src/
COPY scripts/ scripts/
COPY misc/ misc/
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install "setuptools==59.6.0" && \
    . /opt/ros/humble/setup.sh && \
    colcon build --symlink-install

COPY scripts/entrypoint.sh /entrypoint.sh
COPY scripts/start_graspgen_server.sh /start_graspgen_server.sh
RUN chmod +x /entrypoint.sh && chmod +x /start_graspgen_server.sh
COPY config/ /ros2_ws/config/
RUN mkdir -p /artifacts/segmentation_service /artifacts/graspgen_service
ARG GEMINI_API_KEY=""

ENV ROS_DOMAIN_ID=0
ENV ROS_LOCALHOST_ONLY=0
ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ENV FASTDDS_BUILTIN_TRANSPORTS=UDPv4
ENV FASTRTPS_DEFAULT_PROFILES_FILE=/ros2_ws/config/fastdds_no_shm.xml
ENV GEMINI_API_KEY=${GEMINI_API_KEY}

ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "launch", "team_8", "contest_run.launch.py"]
