import datetime
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from reachy_bomi.scenarios import SCENARIO_NAMES, resolve_world_for_scenario, resolve_bag_prefix_for_scenario

BAG_OUTPUT_DIR = os.path.expanduser("~/reachy_bomi_bags")


def launch_setup(context, *args, **kwargs):
    scenario = LaunchConfiguration("scenario").perform(context)
    start_rviz = LaunchConfiguration("start_rviz")
    record = LaunchConfiguration("record").perform(context).lower() == "true"

    world = resolve_world_for_scenario(scenario)

    reachy_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [FindPackageShare("reachy_bringup"), "/launch/reachy.launch.py"]
        ),
        launch_arguments={
            "gazebo": "true",
            "start_rviz": start_rviz,
            "start_sdk_server": "false",
            "foxglove": "false",
            "orbbec": "false",
            "world": world,
        }.items(),
    )

    cmd_vel_publisher_node = Node(
        package="reachy_bomi",
        executable="cmd_vel_publisher",
        output="screen",
    )

    actions = [
        reachy_sim,
        cmd_vel_publisher_node,
    ]

    if record:
        bag_prefix = resolve_bag_prefix_for_scenario(scenario)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        bag_output = os.path.join(BAG_OUTPUT_DIR, f"{bag_prefix}_{timestamp}")
        os.makedirs(BAG_OUTPUT_DIR, exist_ok=True)

        bag_record = ExecuteProcess(
            cmd=[
                "ros2", "bag", "record",
                "--storage", "mcap",
                "-o", bag_output,
                "/tf", "/odom", "/cmd_vel", "/scan",
            ],
            output="screen",
        )
        actions.append(bag_record)

    return actions


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "scenario",
                default_value="familiarization",
                description="Scenario to launch",
                choices=list(SCENARIO_NAMES),
            ),
            DeclareLaunchArgument(
                "start_rviz",
                default_value="true",
                description="Whether to start RViz",
                choices=["true", "false"],
            ),
            DeclareLaunchArgument(
                "record",
                default_value="true",
                description="Whether to record a ROS 2 bag of the run",
                choices=["true", "false"],
            ),
            OpaqueFunction(function=launch_setup),
        ],
    )