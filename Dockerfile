# GraspGenX requires PyTorch>=2.1; nvcr.io/nvidia/pytorch:24.01-py3 ships 2.2.
FROM nvcr.io/nvidia/pytorch:24.01-py3

ARG GRASPGENX_REPO_URL=https://github.com/NVlabs/GraspGenX.git
ARG GRASPGENX_COMMIT=main
ARG GRASPGENX_MODELS_REPO_URL=https://huggingface.co/adithyamurali/GraspGenXModel
ARG GRASPGENX_MODELS_COMMIT=main
ARG GRASPGENX_GRIPPER=robotiq_2f_85

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
ENV PIP_ROOT_USER_ACTION=ignore
ENV GRASPGENX_REPO_DIR=/opt/GraspGenX
ENV GRASPGENX_CHECKPOINT_DIR=/opt/GraspGenXModel/release
ENV SAM2_MODEL_DIR=/opt/models/sam2
ENV SAM2_MODEL_PATH=/opt/models/sam2/sam2_t.pt

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
# The ROS2 apt mirror intermittently returns 400 for random packages, so we
# retry up to 5 times with a short sleep + apt-get update between attempts.
RUN for i in 1 2 3 4 5; do \
        apt-get update && \
        apt-get install -y --no-install-recommends \
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
            python3-pip && break || \
        (echo "apt attempt $i failed, retrying in 15s..." && sleep 15); \
    done && \
    rm -rf /var/lib/apt/lists/*

# UR5 URDF used by the cuRobo robot configuration.
RUN bash -c "source /opt/ros/humble/setup.bash && \
    xacro /opt/ros/humble/share/ur_description/urdf/ur.urdf.xacro \
      ur_type:=ur5 name:=ur > /ur5.urdf"

# System packages for build tools (GraspGenX has no C extension compilation).
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Planner-side Python runtime. Keep this separate from GraspGenX so Gemini/SAM2
# issues do not get conflated with GraspGenX inference issues.
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir -r /tmp/requirements/planner-runtime.txt && \
    python3 -m pip install --no-cache-dir --no-deps -r /tmp/requirements/sam2.txt

# Clone GraspGenX and install with the [serve] extra (adds pyzmq + msgpack).
# No C extension compilation needed — replaces GraspGen's pointnet2_ops build step.
RUN git clone ${GRASPGENX_REPO_URL} ${GRASPGENX_REPO_DIR} && \
    cd ${GRASPGENX_REPO_DIR} && \
    git checkout ${GRASPGENX_COMMIT}

RUN python3 -m pip install --no-cache-dir -e "${GRASPGENX_REPO_DIR}[serve]"

# cuRoboV2 runtime. The install order mirrors the validated smoke test against
# the current planner image; re-pin numpy afterwards to preserve cv_bridge ABI.
# GraspGenX pins numpy==1.26.4 which satisfies the <2 constraint.
RUN python3 -m pip install --no-cache-dir "numpy<2" uv && \
    git clone --branch v0.8.0 --depth 1 https://github.com/NVlabs/curobo.git /tmp/curobo && \
    cd /tmp/curobo && \
    uv pip install --system ".[cu12]" && \
    cd / && rm -rf /tmp/curobo && \
    python3 -m pip install --no-cache-dir --force-reinstall "numpy==1.26.4"

# Bake the Ultralytics SAM2 checkpoint into the image to avoid first-run downloads.
RUN mkdir -p ${SAM2_MODEL_DIR} && \
    curl -L https://github.com/ultralytics/assets/releases/download/v8.4.0/sam2_t.pt \
      -o ${SAM2_MODEL_PATH} && \
    test -s ${SAM2_MODEL_PATH}

# Download GraspGenX model checkpoints (gen + dis) from Hugging Face.
# GRASPGENX_CHECKPOINT_DIR points to the "release" version subdir that
# GraspGenXSampler expects (gen/ and dis/ subdirectories inside).
RUN export GIT_LFS_SKIP_SMUDGE=1 && \
    git clone ${GRASPGENX_MODELS_REPO_URL} /tmp/GraspGenXModel && \
    cd /tmp/GraspGenXModel && \
    git checkout ${GRASPGENX_MODELS_COMMIT} && \
    git lfs pull --include="release/gen/**,release/dis/**" && \
    mkdir -p /opt/GraspGenXModel && \
    cp -r release /opt/GraspGenXModel/release && \
    rm -rf /tmp/GraspGenXModel && \
    test -d "${GRASPGENX_CHECKPOINT_DIR}/gen" && \
    test -d "${GRASPGENX_CHECKPOINT_DIR}/dis"

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
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
