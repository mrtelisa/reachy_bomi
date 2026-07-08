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
  webcam → MediaPipe                   socket_server (always-on bridge node)
        → PCA cursor                        ├─ receives socket messages
        → 9-region velocity                 ├─ publishes socket_server/* topics
        → TCP socket  ───────────────▶      └─ on "scenario:..." → launches
     (socket_client.py)                          bomi_control.launch.py
                                              ├─ Gazebo + Reachy (reachy_bringup)
                                              ├─ cmd_vel_publisher → /cmd_vel
                                              └─ ros2 bag record
```

1. **`socket_client.py`** runs on the **operator PC** (not on the robot). It needs a webcam and the MediaPipe stack. Optionally, it first sends a scenario request (`scenario:<name> rviz:<true|false> record:<true|false>`) and waits for the simulation to come up; it then calibrates a PCA hand-to-cursor map and continuously sends velocity strings such as `lin_vel:0.500 ang_vel:-0.300` over TCP at 20Hz.
2. **`socket_server.py`** (ROS 2 node, started once via `bomi_bridge.launch.py` and left running) opens the TCP server and decodes incoming messages. Velocity/state messages are published on `socket_server/linear_vel`, `socket_server/angular_vel`, `socket_server/base_state`. A `scenario:...` message instead makes it run `ros2 launch reachy_bomi bomi_control.launch.py` itself (terminating any scenario it had previously launched).
3. **`cmd_vel_publisher.py`** (ROS 2 node, started by `bomi_control.launch.py`) subscribes to those topics and, while the base is in velocity mode (`base_state == 1.0`), publishes a `geometry_msgs/Twist` on `/cmd_vel`.
4. **`bomi_control.launch.py`** starts the **Reachy simulation in Gazebo** (via `reachy_bringup`) with the world chosen by the selected scenario, `cmd_vel_publisher`, and (optionally) records a **ROS 2 bag** of the run. It can also be launched manually on the robot PC instead of being triggered remotely — see [Usage](#usage).

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
pip install mediapipe opencv-python scikit-learn numpy scipy
```
If necessary, create a virtual environment.

`socket_client.py` uses the MediaPipe **Tasks API** (`HandLandmarker`), which needs a `hand_landmarker.task` model file — it's not bundled with the `mediapipe` pip package. Download it once and point `--model` at it (default: `scripts/hand_landmarker.task` inside the package):

```bash
curl -o scripts/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
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

### 1. On the robot / simulation PC — start the bridge (once)

```bash
ros2 launch reachy_bomi bomi_bridge.launch.py
```

This starts only the `socket_server` node and leaves it running for the whole session. Note the **IP address** it prints in the log — that's what you pass to the client. You normally start this once and never touch it again; it launches/relaunches `bomi_control.launch.py` itself as scenario requests come in from the client (see step 2).

If you'd rather launch a scenario manually on the robot PC without going through the socket (e.g. for local testing), you can still run `bomi_control.launch.py` directly instead of the bridge:

```bash
ros2 launch reachy_bomi bomi_control.launch.py scenario:=familiarization
```

Launch arguments:

| Argument     | Default           | Description                                              |
|--------------|-------------------|-----------------------------------------------------------|
| `scenario`   | `familiarization` | Scenario to run (selects the Gazebo world). See below.       |
| `start_rviz` | `true`            | Whether to start RViz (`true` / `false`).                    |
| `record`     | `true`            | Whether to record a ROS 2 bag of the run (`true` / `false`). |

### 2. On the operator PC — run the hand-tracking client

```bash
# First time (or to recalibrate): run the calibration phase and save it
python3 socket_client.py <robot_ip> --calibrate [--calib bomi_calib.npz] [--model scripts/hand_landmarker.task] \
    [--port 5051] [--cam 0] \
    [--scenario familiarization] [--start-rviz true] [--record true] [--sim-wait 25]

# Next times: load the saved calibration, skip straight to control
python3 socket_client.py <robot_ip> [--model scripts/hand_landmarker.task] [--port 5051] [--cam 0] \
    [--scenario familiarization] [--start-rviz true] [--record true] [--sim-wait 25]
```

`<robot_ip>` is optional if you've set `DEFAULT_HOST` in `socket_client.py` to your robot's IP; otherwise pass it explicitly. If `--scenario` is given, the client sends `scenario:<name> rviz:<start_rviz> record:<record>` to the bridge right after connecting, waits `--sim-wait` seconds for the simulation to come up, and only then starts calibration/control. If `--scenario` is omitted, no scenario request is sent (useful when a scenario is already running, e.g. launched manually per step 1).

`--calibrate` is **opt-in**: without it, `socket_client.py` skips Phase 1 entirely and loads the saved calibration file (`--calib`, a bare filename is stored in the package's `calibrations/` folder; default `bomi_calib.npz`) — it fails immediately if that file doesn't exist yet. Pass `--calibrate` the first time, or whenever you want to redo it.

**Phase 1 — Calibration** (only with `--calibrate`): move your hand through all the positions you intend to use.
`SPACE` records a sample, `ENTER` finishes (minimum 30 samples), `Q`/`Esc`/closing the window quits.

**Phase 2 — Control:** your hand drives the cursor; the cursor position is mapped to base velocities and streamed to the robot. Two windows are shown: the webcam feed with the hand landmarks, and a map of the virtual screen with the 9-region grid lines and a dot at the current cursor position. Press `Q`/`Esc`, or close either window, to stop the robot and quit.

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
│   ├── bomi_bridge.launch.py       # persistent socket_server bridge (start once)
│   └── bomi_control.launch.py      # simulation + cmd_vel_publisher + bag recording
├── scripts/                        # offline analysis tools (not part of runtime)
│   ├── extract_path.py
│   ├── trajectory_map_collisions.py
│   ├── export_collision_events_csv.py
│   └── hand_landmarker.task        # MediaPipe model (download separately, see Requirements)
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