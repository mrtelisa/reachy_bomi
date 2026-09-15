# reachy_bomi

A ROS 2 package that turns **hand movements into velocity commands for the Reachy 2 mobile base**.

A webcam tracks the operator's hand with [MediaPipe](https://developers.google.com/mediapipe), a calibrated autoencoder map converts the hand pose into a 2D cursor, and the cursor position is mapped to linear/angular velocities. Those velocities are streamed over a TCP socket to the robot side, republished on ROS 2 topics, and finally written to `/cmd_vel`, so the base moves in a Gazebo simulation (or on the real robot).

This work started from a ROS 1 implementation written for the TIAGo robot and was ported to ROS 2 for Reachy 2.

---

## How it works

The system is split in two sides, **operator** and **robot**. They can be two machines, or -- for the simulation -- one PC where the robot side runs inside the Reachy Docker container (see [Single PC with Docker](#single-pc-with-docker)):

```
  OPERATOR PC                          ROBOT / SIMULATION PC
  ───────────                          ─────────────────────
  webcam → MediaPipe                   socket_server (always-on bridge node)
        → autoencoder cursor                ├─ receives socket messages
        → 9-region velocity                 ├─ publishes socket_server/* topics
        → TCP socket  ───────────────▶      └─ on "scenario:..." → launches
     (socket_client.py)                          bomi_control.launch.py
                                              ├─ Gazebo + Reachy (reachy_bringup)
                                              ├─ cmd_vel_publisher → /cmd_vel
                                              └─ ros2 bag record
```

1. **`socket_client.py`** runs on the **operator PC** (not on the robot). It needs a webcam and the MediaPipe stack. Optionally, it first sends a scenario request (`scenario:<name> rviz:<true|false> record:<true|false>`) and waits for the simulation to come up; it then calibrates an autoencoder hand-to-cursor map and continuously sends velocity strings such as `lin_vel:0.500 ang_vel:-0.300` over TCP at 20Hz.
2. **`socket_server.py`** (ROS 2 node, started once via `bomi_bridge.launch.py` and left running) opens the TCP server and decodes incoming messages. Velocity/state messages are published on `socket_server/linear_vel`, `socket_server/angular_vel`, `socket_server/base_state`. A `scenario:...` message instead makes it run `ros2 launch reachy_bomi bomi_control.launch.py` itself (terminating any scenario it had previously launched).
3. **`cmd_vel_publisher.py`** (ROS 2 node, started by `bomi_control.launch.py`) subscribes to those topics and, while the base is in velocity mode (`base_state == 1.0`), publishes a `geometry_msgs/Twist` on `/cmd_vel`.
4. **`bomi_control.launch.py`** starts the **Reachy simulation in Gazebo** (via `reachy_bringup`) with the world chosen by the selected scenario, `cmd_vel_publisher`, and (optionally) records a **ROS 2 bag** of the run. It can also be launched manually on the robot PC instead of being triggered remotely — see [Usage](#usage).

The **velocity computation does not depend on the scenario** — the scenario selects the Gazebo world and, for `reaching`, starts the reaching task node. See [Scenarios](#scenarios).

---

## Requirements

**Robot / simulation side (ROS 2):**

- ROS 2 Humble (with `rclpy`, `std_msgs`, `geometry_msgs`, `nav_msgs`, `gazebo_msgs`), `numpy`, `scipy`, `pyyaml`
- Pollen's `reachy2_core` stack (`reachy_bringup`, `reachy_gazebo`, `zuuu_hal`, ...) and Gazebo Classic 11

**...or** just use the Pollen Docker image (`pollenrobotics/reachy2`), which ships all of the above: it is what the simulation experiments run in, see [Single PC with Docker](#single-pc-with-docker).

**Operator PC (client, plain Python — no ROS required):**

```bash
pip install mediapipe opencv-python tensorflow numpy scipy
sudo apt install wmctrl   # optional: keeps the cursor map above the browser window
```
If necessary, create a virtual environment.

`socket_client.py` uses the MediaPipe **Tasks API** (`HandLandmarker`), which needs a `hand_landmarker.task` model file — it's not bundled with the `mediapipe` pip package. Download it once and point `--model` at it (default: `scripts/hand_landmarker.task` inside the package):

```bash
curl -o scripts/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

With two machines, both must be on the **same network** and able to reach each other on the TCP port (default `5051`).

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

### Single PC with Docker

The host runs the client (Ubuntu 24.04 / ROS Jazzy, or no ROS at all); the whole robot side runs in a container created from the Pollen image, which has ROS Humble and Gazebo Classic (not available natively on Ubuntu 24.04). The TCP socket makes the ROS version mismatch irrelevant.

```bash
# once: create the container (the repo is mounted, so the code inside is always the host's)
docker run -it --name reachy_sim -p 6080:6080 \
  -v ~/Desktop/Tesi/Code/reachy_bomi:/home/reachy/reachy_ws/src/reachy_bomi \
  reachy_completo:latest            # or pollenrobotics/reachy2:latest

# every session: open a shell in the container, build once, start the bridge
docker exec -it reachy_sim bash
source /opt/ros/humble/setup.bash && source ~/reachy_ws/install/setup.bash
cd ~/reachy_ws && colcon build --packages-select reachy_bomi --symlink-install && source install/setup.bash
ros2 launch reachy_bomi bomi_bridge.launch.py      # prints "Socket server listening on 172.17.0.2:5051"
```

Then run the client on the host with the container's IP (`172.17.0.2` on the default Docker bridge). Gazebo is shown through the container's noVNC page (`http://localhost:6080/vnc.html?autoconnect=1&resize=remote`), which the client opens by itself. Bags and reaching results land in `~/reachy_bomi_bags/` **inside the container**.

[`scripts/container_tweaks.sh`](scripts/container_tweaks.sh) contains the changes to Pollen's stack that the experiments rely on (lidar rays not drawn, simulated cameras shrunk to save CPU); re-apply it whenever the container is recreated from the image (instructions in the file). If the container was created without the `-v` mount, copy the changed files in with `docker cp` and rebuild.

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
| `start_rviz` | `true`            | Whether to start RViz (`true` / `false`). The client sends `false` by default. |
| `record`     | `true`            | Whether to record a ROS 2 bag of the run (`true` / `false`). |

### 2. On the operator PC — run the hand-tracking client

```bash
# First time (or to recalibrate): run the calibration phase and save it
python3 socket_client.py <robot_ip> --calibrate [--calib bomi_calib.npz] [--model scripts/hand_landmarker.task] \
    [--port 5051] [--cam 0] \
    [--scenario familiarization] [--start-rviz false] [--record true] [--sim-wait 10] [--sim-url URL] [--show-cam]

# Next times: load the saved calibration, skip straight to control
python3 socket_client.py <robot_ip> [--model scripts/hand_landmarker.task] [--port 5051] [--cam 0] \
    [--scenario familiarization] [--start-rviz false] [--record true] [--sim-wait 10] [--sim-url URL] [--show-cam]
```

`<robot_ip>` is optional if you've set `DEFAULT_HOST` in `socket_client.py` to your robot's IP; otherwise pass it explicitly. If `--scenario` is given, the client sends `scenario:<name> rviz:<start_rviz> record:<record>` to the bridge right after connecting, opens the simulation view (`--sim-url`, by default the noVNC page served by the Reachy container on `http://localhost:6080`) in the default browser, waits `--sim-wait` seconds for the simulation to come up, and only then starts calibration/control. If `--scenario` is omitted, no scenario request is sent (useful when a scenario is already running, e.g. launched manually per step 1).

`--calibrate` is **opt-in**: without it, `socket_client.py` skips Phase 1 entirely and loads the saved calibration file (`--calib`, a bare filename is stored in the package's `calibrations/` folder; default `bomi_calib.npz`) — it fails immediately if that file doesn't exist yet. Pass `--calibrate` the first time, or whenever you want to redo it.

**Phase 1 — Calibration** (only with `--calibrate`): move your hand through all the positions you intend to use.
`SPACE` records a sample, `ENTER` finishes (minimum 30 samples), `Q`/`Esc`/closing the window quits.

**Phase 2 — Cursor preview:** the cursor map is shown but nothing is sent to the robot, so you can get a feel for the cursor. Hold it in the centre region (5) for 5 s to start Control.

**Phase 3 — Control:** your hand drives the cursor; the cursor position is mapped to base velocities and streamed to the robot. In the `reaching` scenario the map also shows the task progress (`target 4/60`) in blue at the bottom, and the client exits by itself when the task is over. A small map of the virtual screen with the 9-region grid lines and a dot at the current cursor position is shown, pinned to the top-left corner of the screen and kept above the browser window with the simulation (`wmctrl` recommended: `sudo apt install wmctrl`). Add `--show-cam` to also see the webcam feed with the hand landmarks. Press `Q`/`Esc`, or close the window, to stop the robot and quit.

The control area is a 3×3 grid with a dead zone in the centre:

```
 1 | 2 | 3      center (5)        → stop
 4 | 5 | 6      middle column (2,8) → linear only
 7 | 8 | 9      middle row (4,6)    → angular only
                corners (1,3,7,9)   → linear + angular
```

### Calibration tools (operator PC, no robot needed)

Besides `--calibrate` in the client, three standalone tools in `reachy_bomi/` manage calibrations without connecting to anything (same webcam/MediaPipe chain, same `calibrations/` folder; names can be given without `.npz`):

```bash
python3 calibrate_bomi.py                 # calibrate, preview the cursor, S = save under a name
python3 customize_bomi.py <name>          # rotate / flip / scale / offset a saved map live, S = save as a new name
python3 load_bomi.py <name>               # try a saved calibration on the cursor map
```

The same calibration file is used by the client (`--calib <name>`), by the robot reaching task and by the cursor reaching tests.

---

## Reaching task

The `reaching` scenario runs a **center-out reaching test** with the mobile base, ported from markerlessBoMI's reaching test: targets on circles around the robot's start pose (home), presented in a fixed pseudo-random order, alternating peripheral target -> home -> target -> home ... The distance grows every `targets_per_distance` targets: by default 10 targets at 2 m, 10 at 4 m and 10 at 6 m (30 targets, 60 reaches with the home returns). Everything is configured in [`config/reaching.yaml`](config/reaching.yaml) (angular positions, `circle_radii`, `targets_per_distance`, target radius, dwell time, trial timeout, session duration, target order).

```bash
python3 socket_client.py <robot_ip> --calib <name> --scenario reaching
```

How it runs ([`reachy_bomi/reaching_task.py`](reachy_bomi/reaching_task.py), started by `bomi_control.launch.py` for scenarios with `task: reaching`):

1. Gazebo loads `worlds/reaching.world`: an empty room with two floor markers (white disc = home, green disc = current target; visual only, no collision).
2. The node waits for the operator to finish the **cursor preview**: as soon as control starts, the **session timer** (`session_duration`, default 5 min) starts and the first target appears.
3. A target is reached when the base centre stays within `target_radius` (0.35 m) for `dwell_time` (2 s); one not reached within `trial_timeout` (30 s) is missed and the next one is shown. The cursor map shows the progress (`target 4/60`) in blue at the bottom.
4. When all targets are done **or the session time is up**, the node exits and the launch shuts everything down (Gazebo, `cmd_vel_publisher`, bag); the client stops on its own.

Per-trial metrics are computed from `/odom` and recorded both in the bag (`/reaching/trial_result`, JSON) and in `~/reachy_bomi_bags/Reaching_<timestamp>_reaching.csv` (+ a `_summary.json`):

| Metric | Meaning |
|---|---|
| `reaction_time` | target shown -> movement onset (speed > `motion_onset_speed`) |
| `movement_time` | movement onset -> last entry into the target |
| `reach_time` | target shown -> last entry into the target |
| `success`, `end_reason` | reached / `trial_timeout` / `session_timeout` |
| `radius`, `distance_block`, `target_number` | distance of the current circle, which distance block (1-3) and which peripheral target (1-30) the trial belongs to |
| `path_length`, `normalized_path_length` | path / straight-line displacement (1 = perfectly straight) |
| `max_deviation` | max perpendicular distance from the ideal line onset -> target centre |
| `dimensionless_jerk`, `log_dimensionless_jerk` | smoothness (Hogan & Sternad 2009) |
| `n_speed_peaks` | local maxima of the speed profile above `speed_peak_threshold` |
| `mean_speed`, `peak_speed` | |

Other bag topics: `/reaching/target` (current target position), `/reaching/event` (target shown / reached / missed / finished, JSON), `/reaching/status` (progress text), `/reaching/summary` (success rate and mean metrics).

## Cursor reaching tests (no robot)

Two fullscreen tests run markerlessBoMI's reaching test on screen: same hand -> cursor chain as the client, but the cursor itself has to reach targets. No robot, socket or container involved.

```bash
cd reachy_bomi
python3 reaching_center_out.py --calib <name> --subject S001 [--max-minutes 4]
python3 reaching_random.py     --calib <name> --subject S001 [--max-minutes 4]
```

- [`reaching_center_out.py`](reachy_bomi/reaching_center_out.py): centre -> target -> centre -> target ... (248 targets, 496 goals). Results in `results_center_out/<subject>_center_out_{trials.csv,summary.json}`.
- [`reaching_random.py`](reachy_bomi/reaching_random.py): one goal at the centre, then the same 248 targets one after the other, no returns to the centre. Results in `results_random/<subject>_random_{trials.csv,summary.json}`. It reuses everything from the center-out script.

Common to both: 1200x650 canvas scaled to the screen (the map's 2550x1500 space is scaled onto it, so the same calibration works here and on the robot), target radius 40 px, targets spread over the **whole canvas** (a seeded random point in each cell of a 6x3 grid, every cell visited before any repeats); the 248 positions are frozen in [`config/cursor_targets.csv`](config/cursor_targets.csv), which is what the tests read, so the sequence is identical for every participant and every launch on any machine, dwell **1 s** inside a target, a target not reached within **10 s** is marked missed and skipped. The window shows the target (green ring, blue while the cursor is inside, filled green when reached), the score, a `target k/N` counter and the remaining time. The **session timer starts when the centre is reached for the first time**; the session ends when the sequence is over or after `--max-minutes` (default 4). Score, as in the original: 4/3/2/1 points per target by time from its appearance to entering it (< 2 s / < 3 s / < 4 s / more). A subject with previous sessions gets `_1`, `_2`, ... appended to the file names. The summary lists the targets that were not reached (`missed_targets`: trial, grid cell, position, reason).

## Scenarios

Scenarios are defined in [`config/scenarios.yaml`](config/scenarios.yaml): each maps a name to a Gazebo `world`, a `bag_prefix` used to name the recording and, optionally, a `task`.

| Scenario          | World                   | What happens |
|-------------------|-------------------------|--------------|
| `familiarization` | `familiarization.world` | free driving: the participant moves the robot around a room with a few obstacles |
| `reaching`        | `reaching.world`        | center-out reaching task with the mobile base (see [Reaching task](#reaching-task)) |

---

## Package layout

```
reachy_bomi/
├── reachy_bomi/                    # ROS 2 Python package
│   ├── socket_client.py            # operator-side client (MediaPipe → cursor → socket)
│   ├── socket_server.py            # ROS 2 node: socket → ROS topics (bridge)
│   ├── cmd_vel_publisher.py        # ROS 2 node: ROS topics → /cmd_vel
│   ├── reaching_task.py            # ROS 2 node: center-out reaching task with the robot
│   ├── reaching_center_out.py      # fullscreen cursor reaching test, center-out (no robot)
│   ├── reaching_random.py          # fullscreen cursor reaching test, random sequence (no robot)
│   ├── reaching_metrics.py         # per-trial kinematic metrics shared by the reaching tests
│   ├── calibrate_bomi.py           # standalone tools: create / customize / try a calibration
│   ├── customize_bomi.py
│   ├── load_bomi.py
│   └── scenarios.py                # loads scenarios.yaml
├── config/
│   ├── scenarios.yaml              # scenario definitions
│   ├── reaching.yaml               # robot reaching task parameters
│   └── cursor_targets.csv          # frozen target positions of the cursor reaching tests
├── launch/
│   ├── bomi_bridge.launch.py       # persistent socket_server bridge (start once)
│   └── bomi_control.launch.py      # simulation + cmd_vel_publisher (+ reaching_task) + bag recording
├── worlds/
│   ├── familiarization.world
│   └── reaching.world
├── scripts/
│   ├── hand_landmarker.task        # MediaPipe model (download separately, see Requirements)
│   └── container_tweaks.sh         # tweaks to re-apply inside the Reachy Docker container
├── calibrations/                   # saved hand-to-cursor calibrations (.npz, not versioned)
├── resource/reachy_bomi            # ament resource marker
├── package.xml, setup.py, setup.cfg
└── README.md
```

## Known limitations
- The client and the robot side must be reachable on the same network; there is no automatic discovery — you pass the robot IP to the client manually.
- `socket_server` binds to the machine's network IP (not `0.0.0.0`): connect to the address it prints, and the machine needs a default route.
- In the container Gazebo and its GUI are software-rendered (no GPU): the simulation is CPU-heavy and the robot's motion can look choppy. The base velocity goes through Pollen's `zuuu_hal`, which caps it at 0.61 m/s.
