#!/usr/bin/env python3
"""
Export collision events from a ROS2 bag file to CSV.

Usage:
    python export_collision_events_csv1.py <bag_dir> <output_csv> [collision_topic]

    collision_topic  default: /reachy/contacts_state
                     Requires a gazebo_ros bumper/contact plugin in the world file.
                     If the topic is not in the bag, the script exits with a warning.

The collision topic must publish gazebo_msgs/msg/ContactsState.
Pose is taken from /odom (nav_msgs/msg/Odometry).
"""
import os
import sys
import csv

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


def _infer_wall_name(c1: str, c2: str) -> str:
    if 'reachy::' in c1 and 'reachy::' not in c2:
        return c2
    if 'reachy::' in c2 and 'reachy::' not in c1:
        return c1
    return c2


def main():
    if len(sys.argv) < 3:
        print("Usage: export_collision_events_csv1.py <bag_dir> <output_csv> [collision_topic]")
        sys.exit(1)

    bag_path = sys.argv[1]
    csv_path = sys.argv[2]
    collision_topic = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_COLLISION_TOPIC

    print(f"Opening bag: {bag_path}")
    reader = _open_reader(bag_path)
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}

    if collision_topic not in type_map:
        contact_topics = [t for t in type_map if 'contact' in t.lower() or 'collision' in t.lower()]
        print(f"Warning: collision topic '{collision_topic}' not found in bag.")
        print(f"Contact-related topics in bag: {contact_topics}")
        print("Add a gazebo_ros contact sensor plugin to the world files to enable collision recording.")
        sys.exit(1)

    last_pose = None
    active_events: dict = {}
    events: list = []
    next_event_id = 1
    first_time = None
    last_time = None

    topics_of_interest = {collision_topic, POSE_TOPIC}

    while reader.has_next():
        topic, data, ts_ns = reader.read_next()
        if topic not in topics_of_interest or topic not in type_map:
            continue

        msg = deserialize_message(data, get_message(type_map[topic]))
        cur_t = ts_ns * 1e-9

        if first_time is None:
            first_time = cur_t
        last_time = cur_t

        if topic == POSE_TOPIC:
            last_pose = msg.pose.pose

        elif topic == collision_topic and hasattr(msg, 'states'):
            current_pairs: set = set()
            for s in msg.states:
                c1 = getattr(s, 'collision1_name', '')
                c2 = getattr(s, 'collision2_name', '')
                current_pairs.add(tuple(sorted([c1, c2])))

            for pair in current_pairs:
                if pair not in active_events:
                    active_events[pair] = {
                        'event_id': next_event_id,
                        'pair': pair,
                        'start_time': cur_t,
                        'end_time': cur_t,
                        'start_pose': last_pose,
                        'end_pose': last_pose,
                        'states_count': sum(
                            1 for s in msg.states
                            if tuple(sorted([getattr(s, 'collision1_name', ''),
                                             getattr(s, 'collision2_name', '')])) == pair
                        ),
                    }
                    next_event_id += 1
                else:
                    ev = active_events[pair]
                    ev['end_time'] = cur_t
                    ev['end_pose'] = last_pose
                    ev['states_count'] += sum(
                        1 for s in msg.states
                        if tuple(sorted([getattr(s, 'collision1_name', ''),
                                         getattr(s, 'collision2_name', '')])) == pair
                    )

            for pair in list(active_events.keys()):
                if pair not in current_pairs:
                    events.append(active_events.pop(pair))

    events.extend(active_events.values())

    bag_duration = (last_time - first_time) if first_time and last_time else 0.0
    total_collision_time = sum(ev['end_time'] - ev['start_time'] for ev in events)

    print()
    print("=" * 40)
    print(f"  Scenario bag:      {os.path.basename(bag_path.rstrip('/'))}")
    print(f"  Bag duration:      {bag_duration:.1f} s")
    print(f"  Collisions:        {len(events)}")
    if events:
        print(f"  Total contact time:{total_collision_time:.2f} s")
        print(f"  Avg duration:      {total_collision_time / len(events):.2f} s")
        print()
        for ev in events:
            c1, c2 = ev['pair']
            wall = _infer_wall_name(c1, c2)
            dur = ev['end_time'] - ev['start_time']
            sp = ev['start_pose']
            pos = f"({sp.position.x:.2f}, {sp.position.y:.2f})" if sp else "(unknown)"
            print(f"  #{ev['event_id']:02d}  wall={wall:<30s}  dur={dur:.2f}s  pos={pos}")
    print("=" * 40)
    print()

    def _pose_to_row(p):
        if p is None:
            return [''] * 7
        return [
            f'{p.position.x:.6f}', f'{p.position.y:.6f}', f'{p.position.z:.6f}',
            f'{p.orientation.x:.6f}', f'{p.orientation.y:.6f}',
            f'{p.orientation.z:.6f}', f'{p.orientation.w:.6f}',
        ]

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'event_id', 'collision1', 'collision2', 'wall_name',
            'start_time', 'end_time', 'duration',
            'start_x', 'start_y', 'start_z', 'start_qx', 'start_qy', 'start_qz', 'start_qw',
            'end_x', 'end_y', 'end_z', 'end_qx', 'end_qy', 'end_qz', 'end_qw',
            'states_total',
        ])
        for ev in events:
            c1, c2 = ev['pair']
            duration = ev['end_time'] - ev['start_time']
            writer.writerow([
                ev['event_id'], c1, c2, _infer_wall_name(c1, c2),
                f"{ev['start_time']:.6f}", f"{ev['end_time']:.6f}", f"{duration:.6f}",
                *_pose_to_row(ev['start_pose']),
                *_pose_to_row(ev['end_pose']),
                ev.get('states_count', 0),
            ])

    print(f"CSV saved to {csv_path}")


if __name__ == '__main__':
    main()
