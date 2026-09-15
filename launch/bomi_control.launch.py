import datetime
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, OpaqueFunction, Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from reachy_bomi.scenarios import (
    SCENARIO_NAMES, resolve_world_for_scenario, resolve_bag_prefix_for_scenario, resolve_task_for_scenario,
)

BAG_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reachy_bomi_bags")

def launch_setup(context, *args, **kwargs):
    scenario = LaunchConfiguration("scenario").perform(context)
    start_rviz = LaunchConfiguration("start_rviz")
    record = LaunchConfiguration("record").perform(context).lower() == "true"

    world = resolve_world_for_scenario(scenario)
    task = resolve_task_for_scenario(scenario)
    bag_prefix = resolve_bag_prefix_for_scenario(scenario)

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

    if task == "reaching":
        # Center-out reaching task. The node exits when the session is over
        # (all targets done or session time up): on_exit=Shutdown() then stops
        # the whole launch -- Gazebo, cmd_vel_publisher and the bag recorder.
        reaching_node = Node(
            package="reachy_bomi",
            executable="reaching_task",
            output="screen",
            parameters=[{"results_dir": BAG_OUTPUT_DIR, "results_prefix": bag_prefix}],
            on_exit=Shutdown(),
        )
        actions.append(reaching_node)

    if record:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        bag_output = os.path.join(BAG_OUTPUT_DIR, f"{bag_prefix}_{timestamp}")
        os.makedirs(BAG_OUTPUT_DIR, exist_ok=True)

        topics = ["/tf", "/odom", "/cmd_vel", "/scan"]
        if task == "reaching":
            topics += ["/reaching/target", "/reaching/status", "/reaching/event",
                       "/reaching/trial_result", "/reaching/summary"]
        bag_record = ExecuteProcess(
            # Default storage (sqlite3): the mcap plugin is not installed in
            # the Reachy container, and "--storage mcap" made the recorder
            # exit immediately (code 2) without recording anything.
            cmd=[
                "ros2", "bag", "record",
                "-o", bag_output,
                *topics,
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