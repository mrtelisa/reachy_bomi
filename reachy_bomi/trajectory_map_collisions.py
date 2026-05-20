#!/usr/bin/env python3
"""
Plot robot trajectory (and optionally collisions) from a ROS2 bag file.

Usage:
    python trajectory_map_collisions.py <bag_dir> [collision_topic]

    collision_topic  default: /reachy/contacts_state
                     Requires a gazebo_ros contact sensor plugin in the world file.
                     If the topic is not in the bag, collisions are simply not shown.

The scenario name is inferred from the bag directory name (e.g. Train1_20260520_143021 → Train1).
Pose is read from /odom (nav_msgs/msg/Odometry).
"""
import sys
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from bisect import bisect_left

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

DEFAULT_COLLISION_TOPIC = '/reachy/contacts_state'
POSE_TOPIC = '/odom'


def _open_reader(bag_path: str) -> rosbag2_py.SequentialReader:
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_path, storage_id='mcap'),
        rosbag2_py.ConverterOptions('', ''),
    )
    return reader


def _scenario_from_bag_dir(bag_dir: str) -> str:
    """Extract scenario prefix from bag directory name, e.g. 'Train1_20260520_143021' → 'Train1'."""
    name = os.path.basename(bag_dir.rstrip('/'))
    match = re.match(r'^(.+?)_\d{8}_\d{6}$', name)
    return match.group(1) if match else name


def main():
    if len(sys.argv) < 2:
        print("Usage: trajectory_map_collisions.py <bag_dir> [collision_topic]")
        sys.exit(1)

    bag_path = sys.argv[1]
    collision_topic = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_COLLISION_TOPIC
    scenario_name = _scenario_from_bag_dir(bag_path)

    reader = _open_reader(bag_path)
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}

    if collision_topic not in type_map:
        print(f"Info: '{collision_topic}' not in bag — collisions won't be shown.")

    pose_times: list = []
    pose_xy: list = []
    collision_times: list = []
    prev_had_contact = False

    topics_of_interest = {POSE_TOPIC, collision_topic}

    while reader.has_next():
        topic, data, ts_ns = reader.read_next()
        if topic not in topics_of_interest or topic not in type_map:
            continue

        msg = deserialize_message(data, get_message(type_map[topic]))
        t_sec = ts_ns * 1e-9

        if topic == POSE_TOPIC:
            pose_times.append(t_sec)
            pose_xy.append((msg.pose.pose.position.x, msg.pose.pose.position.y))

        elif topic == collision_topic:
            now_has_contact = hasattr(msg, 'states') and len(msg.states) > 0
            if (not prev_had_contact) and now_has_contact:
                collision_times.append(t_sec)
            prev_had_contact = now_has_contact

    pose_times_arr = np.array(pose_times)
    pose_xy_arr = np.array(pose_xy) if pose_xy else np.empty((0, 2))

    collision_xy = []
    for ct in collision_times:
        i = bisect_left(pose_times_arr, ct)
        if i >= len(pose_times_arr):
            i = len(pose_times_arr) - 1
        if i > 0 and abs(pose_times_arr[i - 1] - ct) < abs(pose_times_arr[i] - ct):
            i -= 1
        collision_xy.append(pose_xy_arr[i])
    collision_xy_arr = np.array(collision_xy) if collision_xy else np.empty((0, 2))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor('white')

    if pose_xy_arr.shape[0] > 0:
        ax.plot(pose_xy_arr[:, 0], pose_xy_arr[:, 1], 'r-', linewidth=2, label='Trajectory')
        ax.scatter(*pose_xy_arr[0], c='green', s=80, marker='o',
                   edgecolors='black', linewidths=1.5, label='Start', zorder=5)
        ax.scatter(*pose_xy_arr[-1], c='black', s=80, marker='o',
                   edgecolors='white', linewidths=1.5, label='End', zorder=5)

    if collision_xy_arr.shape[0] > 0:
        ax.scatter(collision_xy_arr[:, 0], collision_xy_arr[:, 1],
                   c='red', marker='x', s=100, linewidths=2, label=f'Collision ({len(collision_xy_arr)})')

    ax.set_title(f"Trajectory — {scenario_name}")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend(loc='upper right')
    ax.grid(True, color='lightgray')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
