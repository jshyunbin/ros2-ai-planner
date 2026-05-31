from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='pipeline_orchestrator',
            executable='orchestrator',
            name='pipeline_orchestrator',
            output='screen',
        ),
    ])
