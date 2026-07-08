# reachy_bomi

A package that turns **hand movements into velocity commands for the Reachy 2 mobile base**.

A webcam tracks the operator's hand with [MediaPipe](https://developers.google.com/mediapipe), a calibrated PCA map converts the hand pose into a 2D cursor, and the cursor position is mapped to linear/angular velocities. Those velocities are sent to the real Reachy 2 mobile base over [`reachy2_sdk`](https://github.com/pollen-robotics/reachy2-sdk) (gRPC over IP), so no ROS 2 networking is involved between the operator's PC and the robot.

This work started from a ROS 1 implementation written for the TIAGo robot and was ported to ROS 2 for Reachy 2, then moved off ROS 2 topics onto `reachy2_sdk` so the operator's PC doesn't need a ROS 2 distro matching the robot's.

---

## How it works

Everything runs in a **single process on a single PC** (any PC with a webcam and network access to the robot — no simulation running alongside means there's no need to split hand tracking and robot control across two machines):

```
webcam → MediaPipe → PCA cursor → 9-region velocity → reachy2_sdk (gRPC/IP) → mobile base
                    (reachy_bomi/bomi_teleop.py, one process)
```

**`bomi_teleop.py`** is the only script: it grabs webcam frames, runs MediaPipe hand tracking, maps the hand pose to a 2D cursor through a calibrated PCA map, converts the cursor position into a 3×3-region linear/angular velocity, and sends it to the robot via `reachy.mobile_base.set_goal_speed(...)` / `send_speed_command()` at 20Hz.

---

## Requirements

- The real Reachy 2 robot reachable over the network, with its SDK server running (mobile base enabled).
- A webcam.
- Python deps, in the same environment used to run the script:

```bash
pip install reachy2-sdk mediapipe opencv-python scikit-learn numpy scipy
```

`bomi_teleop.py` uses the MediaPipe **Tasks API** (`HandLandmarker`), which needs a `hand_landmarker.task` model file — it's not bundled with the `mediapipe` pip package. Download it once and point `--model` at it (default: `hand_landmarker.task` at the package root):

```bash
curl -o hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

No ROS 2 install is required to run `bomi_teleop.py` itself (it only talks to the robot via `reachy2_sdk`). The package is still packaged as an `ament_python` ROS 2 package for convenience if you keep it in a ROS 2 workspace, but nothing in the runtime code imports `rclpy`.

---

## Installation

Clone the package into the `src/` folder of a (ROS 2, optional) workspace and build it:

```bash
cd ~/ros2_ws/src
git clone https://github.com/mrtelisa/reachy_bomi.git
cd ~/ros2_ws
colcon build --packages-select reachy_bomi
source install/setup.bash
```

Or just run `bomi_teleop.py` directly with `python3` — no build step required, since it doesn't depend on ROS 2 at runtime.

---

## Usage

```bash
python3 reachy_bomi/bomi_teleop.py [robot_ip] [--model hand_landmarker.task] [--cam 0]
# or, from a built ROS 2 workspace:
ros2 run reachy_bomi bomi_teleop [robot_ip] [--model hand_landmarker.task] [--cam 0]
```

`robot_ip` is optional if you've set `DEFAULT_ROBOT_IP` in `bomi_teleop.py` to your robot's IP; otherwise pass it explicitly.

Every run starts with the calibration phase, then goes straight into control — the PCA map is only kept in memory for that run, not saved or reloaded.

**Phase 1 — Calibration** (always runs first): move your hand through all the positions you intend to use.
`SPACE` records a sample, `ENTER` finishes (minimum 30 samples), `Q`/`Esc`/closing the window quits.

**Phase 2 — Control:** your hand drives the cursor; the cursor position is mapped to base velocities and sent to the robot. Two windows are shown: the webcam feed with the hand landmarks, and a map of the virtual screen with the 9-region grid lines and a dot at the current cursor position. Press `Q`/`Esc`, or close either window, to stop the robot and quit.

The control area is a 3×3 grid with a dead zone in the centre:

```
 1 | 2 | 3      center (5)        → stop
 4 | 5 | 6      middle column (2,8) → linear only
 7 | 8 | 9      middle row (4,6)    → angular only
                corners (1,3,7,9)   → linear + angular
```

---

## Package layout

```
reachy_bomi/
├── reachy_bomi/                    # Python package
│   ├── __init__.py
│   └── bomi_teleop.py              # single script: webcam/MediaPipe → PCA cursor → reachy2_sdk mobile base
├── hand_landmarker.task             # MediaPipe model (download separately, see Requirements)
├── resource/
│   └── reachy_bomi                 # ament resource marker
├── package.xml
├── setup.py
├── setup.cfg
├── .gitignore
└── README.md
```

## Known limitations
- Requires a functioning webcam on the machine running `bomi_teleop`.
- The operator's PC and the robot just need network (IP) reachability to each other — no ROS 2 distro matching is required, since communication goes through `reachy2_sdk`'s gRPC interface.