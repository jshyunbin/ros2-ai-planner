from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    start_graspgen_server = LaunchConfiguration("start_graspgen_server")
    graspgen_host = LaunchConfiguration("graspgen_host")
    graspgen_port = LaunchConfiguration("graspgen_port")
    use_sim_time = LaunchConfiguration("use_sim_time")
    segmentation_service_name = LaunchConfiguration("segmentation_service_name")
    graspgen_service_name = LaunchConfiguration("graspgen_service_name")
    rgb_topic = LaunchConfiguration("rgb_topic")
    depth_topic = LaunchConfiguration("depth_topic")
    segmented_point_cloud_topic = LaunchConfiguration("segmented_point_cloud_topic")
    background_point_cloud_topic = LaunchConfiguration("background_point_cloud_topic")
    overlay_topic = LaunchConfiguration("overlay_topic")
    mask_topic = LaunchConfiguration("mask_topic")
    auto_run_on_task_command = LaunchConfiguration("auto_run_on_task_command")
    graspgen_remove_outliers = LaunchConfiguration("graspgen_remove_outliers")
    segmentation_debug_dir = LaunchConfiguration("segmentation_debug_dir")
    graspgen_debug_dir = LaunchConfiguration("graspgen_debug_dir")

    return LaunchDescription(
        [
            DeclareLaunchArgument("start_graspgen_server", default_value="false"),
            DeclareLaunchArgument("graspgen_host", default_value="127.0.0.1"),
            DeclareLaunchArgument("graspgen_port", default_value="5556"),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument(
                "segmentation_service_name", default_value="/segmentation/segment_prompt"
            ),
            DeclareLaunchArgument("graspgen_service_name", default_value="/graspgen/infer"),
            DeclareLaunchArgument("rgb_topic", default_value="/wrist_camera/wrist_camera/color/image_raw"),
            DeclareLaunchArgument(
                "depth_topic", default_value="/wrist_camera/wrist_camera/depth/color/image_raw"
            ),
            DeclareLaunchArgument(
                "segmented_point_cloud_topic", default_value="/graspgen/segmented_object"
            ),
            DeclareLaunchArgument(
                "background_point_cloud_topic", default_value="/graspgen/background"
            ),
            DeclareLaunchArgument("overlay_topic", default_value="/segmentation/overlay"),
            DeclareLaunchArgument("mask_topic", default_value="/segmentation/mask"),
            DeclareLaunchArgument("auto_run_on_task_command", default_value="true"),
            DeclareLaunchArgument("graspgen_remove_outliers", default_value="false"),
            DeclareLaunchArgument(
                "segmentation_debug_dir",
                default_value="/tmp/ros2-ai-planner/segmentation_service",
            ),
            DeclareLaunchArgument(
                "graspgen_debug_dir",
                default_value="/tmp/ros2-ai-planner/graspgen_service",
            ),
            ExecuteProcess(
                cmd=["/start_graspgen_server.sh"],
                name="embedded_graspgen_server",
                output="screen",
                additional_env={
                    "GRASPGEN_HOST": "0.0.0.0",
                    "GRASPGEN_PORT": graspgen_port,
                },
                condition=IfCondition(start_graspgen_server),
            ),
            Node(
                package="pipeline_orchestrator",
                executable="segmentation_service",
                name="segmentation_service",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": use_sim_time,
                        "service_name": segmentation_service_name,
                        "rgb_topic": rgb_topic,
                        "depth_topic": depth_topic,
                        "segmented_point_cloud_topic": segmented_point_cloud_topic,
                        "background_point_cloud_topic": background_point_cloud_topic,
                        "overlay_topic": overlay_topic,
                        "mask_topic": mask_topic,
                        "debug_dir": segmentation_debug_dir,
                    }
                ],
            ),
            Node(
                package="pipeline_orchestrator",
                executable="graspgen_service",
                name="graspgen_service",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": use_sim_time,
                        "segmented_point_cloud_topic": segmented_point_cloud_topic,
                        "background_point_cloud_topic": background_point_cloud_topic,
                        "service_name": graspgen_service_name,
                        "server_host": graspgen_host,
                        "server_port": graspgen_port,
                        "remove_outliers": graspgen_remove_outliers,
                        "debug_dir": graspgen_debug_dir,
                    }
                ],
            ),
            Node(
                package="pipeline_orchestrator",
                executable="orchestrator",
                name="pipeline_orchestrator",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": use_sim_time,
                        "segmentation_service_name": segmentation_service_name,
                        "graspgen_service_name": graspgen_service_name,
                        "auto_run_on_task_command": auto_run_on_task_command,
                    }
                ],
            ),
        ]
    )
