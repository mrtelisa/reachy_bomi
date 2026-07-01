#!/usr/bin/env python3
"""
Extract the robot path from a ROS 2 bag file and save it as CSV.

Usage:
    python3 extract_path.py <bag_dir> [output_csv] [options]

If output_csv is omitted, the file is written in the current 
working directory.
"""
import argparse
import csv
import os

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


def _bag_name(bag_dir: str) -> str:
    """Full bag directory name, e.g. 'Train1_20260520_143021'."""
    return os.path.basename(bag_dir.rstrip('/'))


def main():
    parser = argparse.ArgumentParser(description="Extract robot path from a ROS 2 bag to CSV.")
    parser.add_argument("bag_dir")
    parser.add_argument("output_csv", nargs="?", default=None)
    args = parser.parse_args()

    output_csv = args.output_csv or f"{_bag_name(args.bag_dir)}_path.csv"

    reader = _open_reader(args.bag_dir)
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}

    if args.pose_topic not in type_map:
        pose_like = [t for t in type_map if 'odom' in t or 'pose' in t or 'amcl' in t]
        print(f"Warning: '{args.pose_topic}' not in bag. Pose-related topics found: {pose_like}")

    n = 0
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp_s', 'x', 'y', 'yaw'])

        while reader.has_next():
            topic, data, ts_ns = reader.read_next()
            if topic != args.pose_topic or topic not in type_map:
                continue
            msg = deserialize_message(data, get_message(type_map[topic]))
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            q = msg.pose.pose.orientation
            _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
            writer.writerow([f'{ts_ns * 1e-9:.6f}', f'{x:.6f}', f'{y:.6f}', f'{yaw:.6f}'])
            n += 1

    print(f"Path saved to {output_csv} ({n} poses)")


if __name__ == '__main__':
    main()