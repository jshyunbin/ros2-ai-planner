#!/bin/bash
set -e
unset CC CXX CUDAHOSTCXX
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash
# Prefer host-mounted src/ over the install/ copy for Python imports so
# .py edits propagate live. install/ is left in place for ROS resource
# lookups (ament index, launch files, config data_files) that need it.
# colcon --symlink-install is not used because the bundled setuptools
# rejects setup.py's --editable option.
export PYTHONPATH="/ros2_ws/src/pipeline_orchestrator:${PYTHONPATH}"
exec "$@"
