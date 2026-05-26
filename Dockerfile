FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# Locale
RUN apt-get update && apt-get install -y locales && \
    locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 && \
    rm -rf /var/lib/apt/lists/*

# ROS2 Humble apt source
RUN apt-get update && apt-get install -y \
    software-properties-common curl gnupg2 lsb-release && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
        http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | \
        tee /etc/apt/sources.list.d/ros2.list > /dev/null && \
    rm -rf /var/lib/apt/lists/*

# ROS2 Humble + Python tools
RUN apt-get update && apt-get install -y \
    ros-humble-ros-base \
    ros-humble-cv-bridge \
    ros-humble-vision-msgs \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-pip \
    git && \
    rm -rf /var/lib/apt/lists/*

# PyTorch with CUDA 12.8
RUN pip3 install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# UR5 URDF (for cuRoboV2 robot config)
RUN apt-get update && apt-get install -y \
    ros-humble-ur-description \
    ros-humble-xacro && \
    rm -rf /var/lib/apt/lists/*

RUN bash -c "source /opt/ros/humble/setup.bash && \
    xacro /opt/ros/humble/share/ur_description/urdf/ur.urdf.xacro \
        ur_type:=ur5 name:=ur > /ur5.urdf"

# cuRoboV2 v0.8.0 + Warp (GPU kernel runtime)
RUN pip3 install --no-cache-dir "warp-lang>=0.10.0" && \
    git clone --depth 1 --branch v0.8.0 \
        https://github.com/NVlabs/curobo.git /tmp/curobo && \
    pip3 install --no-cache-dir /tmp/curobo && \
    rm -rf /tmp/curobo

# Workspace
WORKDIR /ros2_ws
COPY src/ src/
RUN . /opt/ros/humble/setup.sh && \
    colcon build --symlink-install

COPY scripts/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
