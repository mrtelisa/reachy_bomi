#!/usr/bin/env python3
"""
Plot robot trajectory and laser-based near-wall events from a ROS 2 bag file.

Collisions are estimated geometrically from the laser scan: whenever the 
closest valid /scan reading drops below a threshold, the robot is considered 
to be touching (or about to touch) an obstacle.

Usage:
    python3 trajectory_map_collisions.py <bag_dir> [options]

Options:
    --touch-dist FLOAT   Distance (m) below which a near-wall event starts.
                         Roughly: base radius + small margin. Default: 0.30
    --hysteresis FLOAT   Extra distance (m) the reading must rise above
                         touch-dist before the event is considered over.
                         Prevents flicker. Default: 0.10

The scenario name is inferred from the bag directory name
(e.g. Train1_20260520_143021 -> Train1).

NOTE: this is an indicative estimate. A planar laser only sees obstacles in
its scan plane and field of view, and "very close" is not exactly "physical
contact". Tune --touch-dist against a run where the robot clearly hits a wall.
"""
import argparse
import os
import re
from bisect import bisect_left
 
import numpy as np
import matplotlib.pyplot as plt
 
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
 
DEFAULT_TOUCH_DIST = 0.30
DEFAULT_HYSTERESIS = 0.10

SCAN_TOPIC = "/scan"
POSE_TOPIC = "/odom"
 
 
def _open_reader(bag_path: str) -> rosbag2_py.SequentialReader:
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_path, storage_id='mcap'),
        rosbag2_py.ConverterOptions('', ''),
    )
    return reader
 
 
def _scenario_from_bag_dir(bag_dir: str) -> str:
    """Extract scenario prefix from bag dir name, e.g. 'Train1_20260520_143021' -> 'Train1'."""
    name = os.path.basename(bag_dir.rstrip('/'))
    match = re.match(r'^(.+?)_\d{8}_\d{6}$', name)
    return match.group(1) if match else name
 
 
def _min_valid_range(msg) -> float:
    """Smallest finite range within [range_min, range_max]; inf if none valid."""
    ranges = np.asarray(msg.ranges, dtype=float)
    valid = ranges[np.isfinite(ranges) & (ranges >= msg.range_min) & (ranges <= msg.range_max)]
    return float(valid.min()) if valid.size else float('inf')
 
 
def main():
    parser = argparse.ArgumentParser(description="Plot trajectory and laser-based near-wall events.")
    parser.add_argument("bag_dir")
    parser.add_argument("--touch-dist", type=float, default=DEFAULT_TOUCH_DIST)
    parser.add_argument("--hysteresis", type=float, default=DEFAULT_HYSTERESIS)
    args = parser.parse_args()
 
    scenario_name = _scenario_from_bag_dir(args.bag_dir)
 
    reader = _open_reader(args.bag_dir)
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
 
    if SCAN_TOPIC not in type_map:
        scan_like = [t for t in type_map if 'scan' in t.lower() or 'laser' in t.lower()]
        print(f"Error: scan topic '{SCAN_TOPIC}' not in bag. Laser topics found: {scan_like}")
        return
 
    pose_times: list = []
    pose_xy: list = []
    collision_times: list = []
    in_contact = False
 
    topics_of_interest = {POSE_TOPIC, SCAN_TOPIC}
 
    while reader.has_next():
        topic, data, ts_ns = reader.read_next()
        if topic not in topics_of_interest or topic not in type_map:
            continue
 
        msg = deserialize_message(data, get_message(type_map[topic]))
        t_sec = ts_ns * 1e-9
 
        if topic == POSE_TOPIC:
            pose_times.append(t_sec)
            pose_xy.append((msg.pose.pose.position.x, msg.pose.pose.position.y))
 
        elif topic == SCAN_TOPIC:
            d = _min_valid_range(msg)
            if not in_contact and d < args.touch_dist:
                in_contact = True
                collision_times.append(t_sec)            # rising edge = new event
            elif in_contact and d > args.touch_dist + args.hysteresis:
                in_contact = False                        # cleared
 
    pose_times_arr = np.array(pose_times)
    pose_xy_arr = np.array(pose_xy) if pose_xy else np.empty((0, 2))
 
    # Map each collision onset to the nearest recorded pose
    collision_xy = []
    for ct in collision_times:
        if pose_times_arr.size == 0:
            break
        i = bisect_left(pose_times_arr, ct)
        if i >= len(pose_times_arr):
            i = len(pose_times_arr) - 1
        if i > 0 and abs(pose_times_arr[i - 1] - ct) < abs(pose_times_arr[i] - ct):
            i -= 1
        collision_xy.append(pose_xy_arr[i])
    collision_xy_arr = np.array(collision_xy) if collision_xy else np.empty((0, 2))
 
    print(f"Scenario: {scenario_name}  |  near-wall events (laser): {len(collision_times)}  "
          f"(touch_dist={args.touch_dist} m)")
 
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
                   facecolors='none', edgecolors='orange', marker='o', s=140, linewidths=2,
                   label=f'Near-wall, laser ({len(collision_xy_arr)})', zorder=6)
 
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
