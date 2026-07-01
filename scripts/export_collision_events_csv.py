#!/usr/bin/env python3
"""
Export laser-based near-wall events from a ROS 2 bag to CSV.

Collisions are estimated geometrically from the laser scan: an event runs
from when the closest valid /scan reading drops below --touch-dist until it
rises back above --touch-dist + --hysteresis. For each event the script records
its start/end time, duration, closest approach distance, the bearing of the
closest beam at that moment, and the robot pose (from /odom) at start and end.

Usage:
    python3 export_collision_events_csv.py <bag_dir> [out.csv] [options]

Options:
    --touch-dist FLOAT   Distance (m) below which an event starts. Default: 0.30
    --hysteresis FLOAT   Clearance (m) above touch-dist to end an event. Default: 0.10
    --scan-topic STR     Laser topic. Default: /scan

NOTE: indicative estimate only. A planar laser sees obstacles just in its scan
plane / field of view; "very close" is not exactly "physical contact". Tune
--touch-dist on a run where the robot clearly hits a wall.
"""
import argparse
import csv
import math
import os
import re
 
import numpy as np
 
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
 
DEFAULT_TOUCH_DIST = 0.30
DEFAULT_HYSTERESIS = 0.10
 
# Fixed topics (read from these; change here if your bag uses different names)
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
    name = os.path.basename(bag_dir.rstrip('/'))
    match = re.match(r'^(.+?)_\d{8}_\d{6}$', name)
    return match.group(1) if match else name
 
 
def _bag_name(bag_dir: str) -> str:
    """Full bag directory name."""
    return os.path.basename(bag_dir.rstrip('/'))
 
 
def _min_range_and_bearing(msg):
    """Return (min_distance, bearing_rad) of the closest valid beam; (inf, nan) if none."""
    ranges = np.asarray(msg.ranges, dtype=float)
    mask = np.isfinite(ranges) & (ranges >= msg.range_min) & (ranges <= msg.range_max)
    if not mask.any():
        return float('inf'), float('nan')
    idx = np.where(mask)[0]
    i = idx[np.argmin(ranges[idx])]
    return float(ranges[i]), float(msg.angle_min + i * msg.angle_increment)
 
 
def _yaw_from_quat(q) -> float:
    """Yaw (rad) from a geometry_msgs quaternion."""
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))
 
 
def main():
    parser = argparse.ArgumentParser(description="Export laser-based near-wall events to CSV.")
    parser.add_argument("bag_dir")
    parser.add_argument("out_csv", nargs="?", default=None)
    parser.add_argument("--touch-dist", type=float, default=DEFAULT_TOUCH_DIST)
    parser.add_argument("--hysteresis", type=float, default=DEFAULT_HYSTERESIS)
    args = parser.parse_args()
 
    scenario_name = _scenario_from_bag_dir(args.bag_dir)
    out_csv = args.out_csv or f"{_bag_name(args.bag_dir)}_collision_events.csv"
 
    reader = _open_reader(args.bag_dir)
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
 
    if SCAN_TOPIC not in type_map:
        scan_like = [t for t in type_map if 'scan' in t.lower() or 'laser' in t.lower()]
        print(f"Error: scan topic '{SCAN_TOPIC}' not in bag. Laser topics found: {scan_like}")
        return
 
    topics_of_interest = {POSE_TOPIC, SCAN_TOPIC}
 
    events: list = []
    current = None
    last_pose = None          # (x, y, yaw) most recent /odom
    in_contact = False
    event_id = 0
 
    while reader.has_next():
        topic, data, ts_ns = reader.read_next()
        if topic not in topics_of_interest or topic not in type_map:
            continue
 
        msg = deserialize_message(data, get_message(type_map[topic]))
        t_sec = ts_ns * 1e-9
 
        if topic == POSE_TOPIC:
            p = msg.pose.pose
            last_pose = (p.position.x, p.position.y, _yaw_from_quat(p.orientation))
            continue
 
        # scan topic
        d, bearing = _min_range_and_bearing(msg)
 
        if not in_contact and d < args.touch_dist:
            in_contact = True
            event_id += 1
            current = {
                "event_id": event_id,
                "start_time": t_sec,
                "end_time": t_sec,
                "min_distance": d,
                "bearing_rad": bearing,
                "start_pose": last_pose,
                "end_pose": last_pose,
            }
        elif in_contact:
            current["end_time"] = t_sec
            current["end_pose"] = last_pose
            if d < current["min_distance"]:
                current["min_distance"] = d
                current["bearing_rad"] = bearing
            if d > args.touch_dist + args.hysteresis:
                in_contact = False
                events.append(current)
                current = None
 
    if current is not None:      # flush an event still open at end of bag
        events.append(current)
 
    def _fmt_pose(pose, key):
        if pose is None:
            return ""
        return f"{pose[key]:.4f}"
 
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "event_id", "start_time", "end_time", "duration_s",
            "min_distance_m", "obstacle_bearing_deg",
            "start_x", "start_y", "start_yaw_deg", "end_x", "end_y",
        ])
        for e in events:
            duration = e["end_time"] - e["start_time"]
            bearing_deg = math.degrees(e["bearing_rad"]) if not math.isnan(e["bearing_rad"]) else ""
            sp, ep = e["start_pose"], e["end_pose"]
            writer.writerow([
                e["event_id"],
                f"{e['start_time']:.3f}",
                f"{e['end_time']:.3f}",
                f"{duration:.3f}",
                f"{e['min_distance']:.4f}",
                f"{bearing_deg:.1f}" if bearing_deg != "" else "",
                _fmt_pose(sp, 0), _fmt_pose(sp, 1),
                f"{math.degrees(sp[2]):.1f}" if sp is not None else "",
                _fmt_pose(ep, 0), _fmt_pose(ep, 1),
            ])
 
    total_time = sum(e["end_time"] - e["start_time"] for e in events)
    print(f"Scenario: {scenario_name}")
    print(f"Near-wall events (laser, touch_dist={args.touch_dist} m): {len(events)}")
    print(f"Total near-wall time: {total_time:.2f} s")
    print(f"Written: {out_csv}")
 
 
if __name__ == '__main__':
    main()
