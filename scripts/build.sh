#!/bin/bash
set -e
source /opt/ros/humble/setup.bash
cd /ros2_ws
colcon build --symlink-install
