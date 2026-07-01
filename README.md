# reachy_bomi

A ROS 2 package that turns **hand movements into velocity commands for the Reachy 2 mobile base**.

A webcam tracks the operator's hand with [MediaPipe](https://developers.google.com/mediapipe), a calibrated PCA map converts the hand pose into a 2D cursor, and the cursor position is mapped to linear/angular velocities. Those velocities are streamed over a TCP socket to the robot side, republished on ROS 2 topics, and finally written to `/cmd_vel`, so the base moves in a Gazebo simulation (or on the real robot).

This work started from a ROS 1 implementation written for the TIAGo robot and was ported to ROS 2 for Reachy 2.

---

## How it works

The system is **distributed across two machines**:

```
  OPERATOR PC                          ROBOT / SIMULATION PC
  ───────────                          ─────────────────────
  webcam → MediaPipe                   socket_server (ROS 2 node)
        → PCA cursor                        ├─ receives socket messages
        → 9-region velocity                 └─ publishes socket_server/* topics
        → TCP socket  ───────────────▶  cmd_vel_publisher (ROS 2 node)
     (socket_client.py)                      └─ republishes to /cmd_vel
                                         Gazebo + Reachy (reachy_bringup)
```

1. **`socket_client.py`** runs on the **operator PC** (not on the robot). It needs a webcam and the MediaPipe stack. It calibrates a PCA hand-to-cursor map, then continuously sends velocity strings such as `lin_vel:0.500 ang_vel:-0.300` over TCP.
2. **`socket_server.py`** (ROS 2 node) runs on the **robot/simulation side**. It opens the TCP server, decodes the incoming messages, and publishes them on `socket_server/linear_vel`, `socket_server/angular_vel`, `socket_server/base_state`, etc.
3. **`cmd_vel_publisher.py`** (ROS 2 node) subscribes to those topics and, while the base is in velocity mode (`base_state == 1.0`), publishes a `geometry_msgs/Twist` on `/cmd_vel`.
4. The launch file also starts the **Reachy simulation in Gazebo** (via `reachy_bringup`) with the world chosen by the selected scenario, and records a **ROS 2 bag** of the run.

The **velocity computation does not depend on the scenario** — the scenario only selects which Gazebo world (obstacles) is loaded. See [Scenarios](#scenarios).

---

## Requirements

**Robot / simulation side (ROS 2):**

- ROS 2 (with `rclpy`, `std_msgs`, `geometry_msgs`)
- [`reachy_bringup`](https://github.com/pollen-robotics) and `reachy_utils` (provide `reachy.launch.py` and the Gazebo simulation)
- Gazebo

**...or** just download the docker image pollenrobotics/reachy2 and create a container

**Operator PC (client, plain Python — no ROS required):**

```bash
pip install mediapipe opencv-python scikit-learn numpy
```

Both machines must be on the **same network** and able to reach each other on the TCP port (default `5051`).

---

## Installation

Clone the package into the `src/` folder of a ROS 2 workspace and build it:

```bash
cd ~/ros2_ws/src
git clone https://github.com/mrtelisa/reachy_bomi.git
cd ~/ros2_ws
colcon build --packages-select reachy_bomi
source install/setup.bash
```

---

## Usage

### 1. On the robot / simulation PC — launch the simulation and the bridge nodes

```bash
ros2 launch reachy_bomi bomi_control.launch.py scenario:=familiarization
```

This starts the Gazebo simulation with the scenario's world, the `socket_server` and `cmd_vel_publisher` nodes, and a `ros2 bag` recording into `~/reachy_bomi_bags/`.

Launch arguments:

| Argument     | Default           | Description                                              |
|--------------|-------------------|---------------------------------------------------------|
| `scenario`   | `familiarization` | Scenario to run (selects the Gazebo world). See below.  |
| `start_rviz` | `true`            | Whether to start RViz (`true` / `false`).               |

When the node starts, note the **IP address** the socket server is listening on (printed in the log).

### 2. On the operator PC — run the hand-tracking client

```bash
python3 socket_client.py <robot_ip> [--port 5051] [--cam 0]
```

**Phase 1 — Calibration:** move your hand through all the positions you intend to use.
`SPACE` records a sample, `ENTER` finishes (minimum 30 samples), `Q` quits.

**Phase 2 — Control:** your hand drives the cursor; the cursor position is mapped to base velocities and streamed to the robot. Press `Q` to stop the robot and quit.

The control area is a 3×3 grid with a dead zone in the centre:

```
 1 | 2 | 3      center (5)        → stop
 4 | 5 | 6      middle column (2,8) → linear only
 7 | 8 | 9      middle row (4,6)    → angular only
                corners (1,3,7,9)   → linear + angular
```

---

## Scenarios

Scenarios are defined in [`config/scenarios.yaml`](config/scenarios.yaml). Each one maps a name to a `map_id`, a Gazebo `world`, and a `bag_prefix` used to name the recording.

| Scenario        | World                  |
|-----------------|------------------------|
| `familiarization` | `familiarization.world` |
| `train1`–`train4` | `labyrinth1.world`     |
| `test1`, `test2`  | `mid_tests.world`      |
| `train5`–`train8` | `labyrinth2.world`     |
| `final_test`      | `final_test.world`     |

---

## Package layout

```
reachy_bomi/
├── reachy_bomi/                    # ROS 2 Python package
│   ├── __init__.py
│   ├── socket_client.py            # operator-side client (MediaPipe → socket)
│   ├── socket_server.py            # ROS 2 node: socket → ROS topics
│   ├── cmd_vel_publisher.py        # ROS 2 node: ROS topics → /cmd_vel
│   └── scenarios.py                # loads scenarios.yaml, resolves worlds
├── config/
│   └── scenarios.yaml              # scenario definitions
├── launch/
│   └── bomi_control.launch.py      # simulation + bridge nodes + bag recording
├── scripts/                        # offline analysis tools (not part of runtime)
│   ├── extract_path.py
│   ├── trajectory_map_collisions.py
│   └── export_collision_events_csv.py
├── worlds/                         # Gazebo world files
├── resource/
│   └── reachy_bomi                 # ament resource marker
├── package.xml
├── setup.py
├── setup.cfg
├── .gitignore
└── README.md
```

### Offline analysis scripts

The tools in `scripts/` inspect a recorded bag **after** a run; they are standalone and not part of the runtime flow:

- `extract_path.py` — export the robot path from a bag to CSV.
- `trajectory_map_collisions.py` — plot the trajectory (and collisions, if available).
- `export_collision_events_csv.py` — export collision events to CSV.

Collisions are estimated using the `/scan` topic.

Run them directly, e.g.:

```bash
python3 scripts/extract_path.py ~/reachy_bomi_bags/Familiarization_<timestamp> path.csv
```

## TODOs
- Evaluate if it makes sense to implement a mechanism of collision evaluation using the `/odom` topic together with the `/scan` one 

## Known limitations
- The client and the robot side must be reachable on the same network; there is no automatic discovery — you pass the robot IP to the client manually.