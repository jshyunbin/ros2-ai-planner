FROM nvcr.io/nvidia/pytorch:23.07-py3

ARG GRASPGEN_REPO_URL=https://github.com/pianojay/GraspGen.git
ARG GRASPGEN_BRANCH=jaeuk
ARG GRASPGEN_COMMIT=31b67f65f3cb88928887edd2ee24e302c30cab70
ARG GRASPGEN_MODELS_REPO_URL=https://huggingface.co/adithyamurali/GraspGenModels
ARG GRASPGEN_MODELS_COMMIT=ec1ccbb5eec0680db669246ac312a3636f16ee43
ARG GRIPPER_CONFIG_NAME=graspgen_robotiq_2f_140.yml
ARG GRASPGEN_MODEL_FILES=checkpoints/graspgen_robotiq_2f_140.yml,checkpoints/graspgen_robotiq_2f_140_gen.pth,checkpoints/graspgen_robotiq_2f_140_dis.pth

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
ENV PIP_ROOT_USER_ACTION=ignore
ENV GRASPGEN_REPO_DIR=/opt/GraspGen
ENV GRASPGEN_MODELS_DIR=/opt/GraspGenModels
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

# Planner-side Python runtime. Keep this separate from GraspGen so Gemini/SAM2
# issues do not get conflated with GraspGen inference issues.
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir -r /tmp/requirements/planner-runtime.txt && \
    python3 -m pip install --no-cache-dir --no-deps -r /tmp/requirements/sam2.txt

# Clone the fork only after ROS2/system and planner runtime are established.
RUN git clone --recursive --branch ${GRASPGEN_BRANCH} ${GRASPGEN_REPO_URL} ${GRASPGEN_REPO_DIR} && \
    cd ${GRASPGEN_REPO_DIR} && \
    git checkout ${GRASPGEN_COMMIT} && \
    git submodule update --init --recursive

# GraspGen runtime. This is intentionally separated from the planner runtime above.
RUN python3 -m pip install --no-cache-dir -r ${GRASPGEN_REPO_DIR}/requirements.zmq_pointnet_viser.txt

# Official GraspGen pointnet installation pattern, adapted for the reduced runtime.
RUN cd ${GRASPGEN_REPO_DIR}/pointnet2_ops && \
    python3 -m pip install --no-cache-dir --no-build-isolation .

# Install the fork itself without re-resolving the broad upstream dependency set.
RUN cd ${GRASPGEN_REPO_DIR} && \
    python3 -m pip install --no-cache-dir --no-deps -e .

# cuRoboV2 runtime. The install order mirrors the validated smoke test against
# the current planner image; re-pin numpy afterwards to preserve cv_bridge ABI.
RUN python3 -m pip install --no-cache-dir "numpy<2" uv && \
    git clone --branch v0.8.0 --depth 1 https://github.com/NVlabs/curobo.git /tmp/curobo && \
    cd /tmp/curobo && \
    uv pip install --system ".[cu12]" && \
    cd / && rm -rf /tmp/curobo && \
    python3 -m pip install --no-cache-dir --force-reinstall "numpy<2"

# Bake the Ultralytics SAM2 checkpoint into the image to avoid first-run downloads.
RUN mkdir -p ${SAM2_MODEL_DIR} && \
    curl -L https://github.com/ultralytics/assets/releases/download/v8.4.0/sam2_t.pt \
      -o ${SAM2_MODEL_PATH} && \
    test -s ${SAM2_MODEL_PATH}

# Download only the pinned GraspGen model assets required by the planner.
RUN export GIT_LFS_SKIP_SMUDGE=1 && \
    git clone ${GRASPGEN_MODELS_REPO_URL} /tmp/GraspGenModels && \
    cd /tmp/GraspGenModels && \
    git checkout ${GRASPGEN_MODELS_COMMIT} && \
    git lfs pull --include="${GRASPGEN_MODEL_FILES}" && \
    mkdir -p ${GRASPGEN_MODELS_DIR}/checkpoints && \
    cp checkpoints/graspgen_robotiq_2f_140.yml ${GRASPGEN_MODELS_DIR}/checkpoints/ && \
    cp checkpoints/graspgen_robotiq_2f_140_gen.pth ${GRASPGEN_MODELS_DIR}/checkpoints/ && \
    cp checkpoints/graspgen_robotiq_2f_140_dis.pth ${GRASPGEN_MODELS_DIR}/checkpoints/ && \
    rm -rf /tmp/GraspGenModels && \
    test -f "${GRASPGEN_MODELS_DIR}/checkpoints/${GRIPPER_CONFIG_NAME}" && \
    test -f "${GRASPGEN_MODELS_DIR}/checkpoints/graspgen_robotiq_2f_140_gen.pth" && \
    test -f "${GRASPGEN_MODELS_DIR}/checkpoints/graspgen_robotiq_2f_140_dis.pth"

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
