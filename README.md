# reachy_bomi

A ROS 2 package that turns **hand movements into velocity commands for the Reachy 2 mobile base**.

A webcam tracks the operator's hand with [MediaPipe](https://developers.google.com/mediapipe), a calibrated PCA map converts the hand pose into a 2D cursor, and the cursor position is mapped to linear/angular velocities. Those velocities are published directly on `/cmd_vel`, so the real Reachy 2 mobile base moves.

This work started from a ROS 1 implementation written for the TIAGo robot and was ported to ROS 2 for Reachy 2.

---

## How it works

Everything runs in a **single process on a single PC** (the robot PC, or any PC with a webcam and access to the ROS 2 network — no simulation running alongside means there's no need to split hand tracking and robot control across two machines anymore):

```
webcam → MediaPipe → PCA cursor → 9-region velocity → /cmd_vel
                    (reachy_bomi/bomi_teleop.py, one ROS 2 node)
```

**`bomi_teleop.py`** is the only node: it grabs webcam frames, runs MediaPipe hand tracking, maps the hand pose to a 2D cursor through a calibrated PCA map, converts the cursor position into a 3×3-region linear/angular velocity, and publishes a `geometry_msgs/Twist` directly on `/cmd_vel` at 20Hz.

---

## Requirements

- ROS 2 (with `rclpy`, `geometry_msgs`)
- The real Reachy 2 stack already running (mobile base enabled) so something is listening on `/cmd_vel`.
- A webcam.
- Python deps for the hand-tracking side, in the same environment `ros2 run` uses:

```bash
pip install mediapipe opencv-python scikit-learn numpy scipy
```

`bomi_teleop.py` uses the MediaPipe **Tasks API** (`HandLandmarker`), which needs a `hand_landmarker.task` model file — it's not bundled with the `mediapipe` pip package. Download it once and point `--model` at it (default: `hand_landmarker.task` at the package root):

```bash
curl -o hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

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

```bash
ros2 run reachy_bomi bomi_teleop [--model hand_landmarker.task] [--cam 0]
```

Every run starts with the calibration phase, then goes straight into control — the PCA map is only kept in memory for that run, not saved or reloaded.

**Phase 1 — Calibration** (always runs first): move your hand through all the positions you intend to use.
`SPACE` records a sample, `ENTER` finishes (minimum 30 samples), `Q`/`Esc`/closing the window quits.

**Phase 2 — Control:** your hand drives the cursor; the cursor position is mapped to base velocities and published on `/cmd_vel`. Two windows are shown: the webcam feed with the hand landmarks, and a map of the virtual screen with the 9-region grid lines and a dot at the current cursor position. Press `Q`/`Esc`, or close either window, to stop the robot and quit.

The control area is a 3×3 grid with a dead zone in the centre:

```
 1 | 2 | 3      center (5)        → stop
 4 | 5 | 6      middle column (2,8) → linear only
 7 | 8 | 9      middle row (4,6)    → angular only
                corners (1,3,7,9)   → linear + angular
```

Verify the robot is receiving commands with:

```bash
ros2 topic echo /cmd_vel
```

---

## Package layout

```
reachy_bomi/
├── reachy_bomi/                    # ROS 2 Python package
│   ├── __init__.py
│   └── bomi_teleop.py              # single node: webcam/MediaPipe → PCA cursor → /cmd_vel
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
