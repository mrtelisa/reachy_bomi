#!/usr/bin/env python3
"""
Extract robot path from a ROS2 bag file and save it as CSV.

Usage:
    python extract_path1.py <bag_dir> <output_csv> [pose_topic]

    pose_topic  default: /odom
                Use /amcl_pose if Nav2 localization is running.

Both nav_msgs/msg/Odometry (/odom) and
geometry_msgs/msg/PoseWithCovarianceStamped (/amcl_pose) are supported
since both expose the same msg.pose.pose.position / orientation fields.
"""
import sys
import csv

from tf_transformations import euler_from_quaternion
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def _open_reader(bag_path: str) -> rosbag2_py.SequentialReader:
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_path, storage_id='mcap'),
        rosbag2_py.ConverterOptions('', ''),
    )
    return reader


def main():
    if len(sys.argv) < 3:
        print("Usage: extract_path1.py <bag_dir> <output_csv> [pose_topic]")
        print("  pose_topic default: /odom")
        sys.exit(1)

    bag_path = sys.argv[1]
    output_csv = sys.argv[2]
    pose_topic = sys.argv[3] if len(sys.argv) > 3 else '/odom'

    reader = _open_reader(bag_path)
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}

    if pose_topic not in type_map:
        nav_topics = [t for t in type_map if 'odom' in t or 'pose' in t or 'amcl' in t]
        print(f"Warning: '{pose_topic}' not in bag. Pose-related topics found: {nav_topics}")

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp_s', 'x', 'y', 'yaw'])

        while reader.has_next():
            topic, data, ts_ns = reader.read_next()
            if topic != pose_topic or topic not in type_map:
                continue
            msg = deserialize_message(data, get_message(type_map[topic]))
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            q = msg.pose.pose.orientation
            _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
            writer.writerow([f'{ts_ns * 1e-9:.6f}', f'{x:.6f}', f'{y:.6f}', f'{yaw:.6f}'])

    print(f"Path saved to {output_csv}")


if __name__ == '__main__':
    main()
