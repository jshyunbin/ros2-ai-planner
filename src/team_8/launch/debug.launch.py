"""Debug mode: same pipeline as deploy (executes on the arm) plus a viser
server visualizing segmented clouds, ranked grasp poses, and the live TSDF."""

import os

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory


def generate_launch_description() -> LaunchDescription:
    common = os.path.join(
        get_package_share_directory("team_8"),
        "launch",
        "pipeline_common.launch.py",
    )
    return LaunchDescription(
        [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(common),
                launch_arguments={
                    "enable_motion_execution": "true",
                    "start_graspgen_server": "true",
                    "enable_viz": "true",
                    "publish_grasp_poses": "true",
                    "enable_debug_viz": "true",
                }.items(),
            )
        ]
    )
