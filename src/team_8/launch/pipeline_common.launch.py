"""Shared pipeline node graph, parameterized by mode toggles.

deploy.launch.py / debug.launch.py include this file and set the toggles;
neither requires the user to pass any argument.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    enable_motion_execution = LaunchConfiguration("enable_motion_execution")
    start_graspgen_server = LaunchConfiguration("start_graspgen_server")
    enable_viz = LaunchConfiguration("enable_viz")
    publish_grasp_poses = LaunchConfiguration("publish_grasp_poses")
    enable_debug_viz = LaunchConfiguration("enable_debug_viz")
    graspgen_port = LaunchConfiguration("graspgen_port")

    # Stable defaults (previously all individually exposed; now internal).
    use_sim_time = True
    segmentation_service_name = "/segmentation/segment_prompt"
    graspgen_service_name = "/graspgen/infer"
    curobo_service_name = "/curobo/plan_trajectory"
    rgb_topic = "/wrist_camera/wrist_camera/color/image_raw"
    depth_topic = "/wrist_camera/wrist_camera/depth/color/image_raw"
    camera_info_topic = "/wrist_camera/wrist_camera/depth/color/camera_info"
    segmented_point_cloud_topic = "/graspgen/segmented_object"
    background_point_cloud_topic = "/graspgen/background"
    grasp_poses_topic = "/graspgen/grasp_poses"
    tsdf_voxels_topic = "/curobo/tsdf_voxels"
    overhead_cloud_topic = "/curobo/overhead_cloud"
    segmentation_output_frame = "base_link"

    return LaunchDescription(
        [
            DeclareLaunchArgument("enable_motion_execution", default_value="true"),
            DeclareLaunchArgument("start_graspgen_server", default_value="true"),
            DeclareLaunchArgument("enable_viz", default_value="false"),
            DeclareLaunchArgument("publish_grasp_poses", default_value="false"),
            DeclareLaunchArgument("enable_debug_viz", default_value="false"),
            DeclareLaunchArgument("graspgen_port", default_value="5556"),
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
                package="team_8",
                executable="segmentation_service",
                name="segmentation_service",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": use_sim_time,
                        "service_name": segmentation_service_name,
                        "rgb_topic": rgb_topic,
                        "depth_topic": depth_topic,
                        "camera_info_topic": camera_info_topic,
                        "segmented_point_cloud_topic": segmented_point_cloud_topic,
                        "background_point_cloud_topic": background_point_cloud_topic,
                        "output_frame": segmentation_output_frame,
                        "overlay_topic": "/segmentation/overlay",
                        "mask_topic": "/segmentation/mask",
                        "debug_dir": "/artifacts/segmentation_service",
                    }
                ],
            ),
            Node(
                package="team_8",
                executable="graspgen_service",
                name="graspgen_service",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": use_sim_time,
                        "segmented_point_cloud_topic": segmented_point_cloud_topic,
                        "background_point_cloud_topic": background_point_cloud_topic,
                        "service_name": graspgen_service_name,
                        "server_host": "127.0.0.1",
                        "server_port": graspgen_port,
                        "remove_outliers": False,
                        "rank_mode": "approach_alignment",
                        "target_approach_dir": [0.0, 0.0, -1.0],
                        "expected_frame": segmentation_output_frame,
                        "debug_dir": "/artifacts/graspgen_service",
                        "publish_grasp_poses": publish_grasp_poses,
                        "grasp_poses_topic": grasp_poses_topic,
                    }
                ],
            ),
            Node(
                package="team_8",
                executable="curobo_service",
                name="curobo_service",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": use_sim_time,
                        "service_name": curobo_service_name,
                        "enable_viz": enable_viz,
                        "tsdf_voxels_topic": tsdf_voxels_topic,
                        "overhead_cloud_topic": overhead_cloud_topic,
                    }
                ],
                condition=IfCondition(enable_motion_execution),
            ),
            Node(
                package="team_8",
                executable="orchestrator",
                name="team_8",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": use_sim_time,
                        "segmentation_service_name": segmentation_service_name,
                        "graspgen_service_name": graspgen_service_name,
                        "curobo_service_name": curobo_service_name,
                        "enable_motion_execution": enable_motion_execution,
                        "auto_run_on_task_command": True,
                        "place_goal": "storageA_1",
                        "auto_loop": True,
                    }
                ],
            ),
            Node(
                package="team_8",
                executable="debug_viz",
                name="debug_viz",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": use_sim_time,
                        "segmented_point_cloud_topic": segmented_point_cloud_topic,
                        "background_point_cloud_topic": background_point_cloud_topic,
                        "overhead_cloud_topic": overhead_cloud_topic,
                        "grasp_poses_topic": grasp_poses_topic,
                        "tsdf_voxels_topic": tsdf_voxels_topic,
                    }
                ],
                condition=IfCondition(enable_debug_viz),
            ),
        ]
    )
