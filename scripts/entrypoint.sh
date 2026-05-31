#!/bin/bash
set -e
unset CC CXX CUDAHOSTCXX
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash
exec "$@"
