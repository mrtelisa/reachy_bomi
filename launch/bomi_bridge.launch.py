from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """
    Persistent bridge: only the socket server, listening for the operator PC's
    connection. Start this once on the robot PC and leave it running; it
    launches bomi_control.launch.py itself for each scenario it receives.
    """
    return LaunchDescription(
        [
            Node(
                package="reachy_bomi",
                executable="socket_server",
                output="screen",
            ),
        ]
    )
