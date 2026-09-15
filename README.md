# reachy_bomi

A package that turns **hand movements into control of a Reachy 2 robot**: driving its mobile base, and selecting + grasping objects it sees through its torso depth camera.

A webcam tracks the operator's hand with [MediaPipe](https://developers.google.com/mediapipe), a calibrated autoencoder map converts the hand pose into a 2D cursor, and the cursor position drives either base velocity commands or object hover-selection. Grasp planning turns a YOLOv8 detection + depth camera point cloud into pre-grasp/grasp/lift end-effector poses, executed over [`reachy2_sdk`](https://github.com/pollen-robotics/reachy2-sdk) (gRPC over IP) — no ROS 2 networking is involved between the operator's PC and the robot.

This work started from a ROS 1 implementation written for the TIAGo robot and was ported to ROS 2 for Reachy 2, then moved off ROS 2 topics onto `reachy2_sdk` so the operator's PC doesn't need a ROS 2 distro matching the robot's.

---

## How it works

Everything runs on a **single PC** (any PC with a webcam and network access to the robot); the head- or torso-camera live feed runs in a small helper subprocess (`camera_viewer.py`) so its network round-trips never block the main cursor/velocity loop, but there's no ROS 2/robot-side process split:

```
webcam → MediaPipe → autoencoder cursor → 9-region velocity → reachy2_sdk (gRPC/IP) → mobile base
                                        ↓ (cursor dwell)
                          arms → pre-grasping pose → object hover-select ⇄ repositioning
                                        ↓                    (min-speed nav, torso camera)
      torso depth camera → YOLOv8 → point cloud → grasp planning → arm execution
```

`reachy_control.py` is the only script with a CLI/`main()` for real robot use — `bomi_teleop.py`, `reachy_detection.py`, `reachy_selection.py`, `reachy_pregrasp.py`, `reachy_grasp.py`, `camera_viewer.py`, `graphs.py`, `stream.py`, and `safety.py` are library modules it's built from. Every other way of running things (mouse-driven grasp, dry runs, single-piece diagnostics) lives under `tests/`, see Usage below.

- **`bomi_teleop.py`** — hand tracking → autoencoder cursor → 9-region velocity building blocks (calibration/cursor-preview phases, the BoMI map, cursor filter, velocity helpers). `BoMIMap.save_map_bomi`/`load_map_bomi` (de)serialize a fitted map to/from a `.npz` file; `resolve_calib_path` turns a bare name into a path inside `CALIB_DIR` (a `calibrations/` folder next to the package, created on first save, not tracked by git).
- **`calibrate_bomi.py`** — standalone tool: run calibration, preview the fitted map live (nothing sent anywhere, no robot needed), then `S` prompts for a name and saves it, `Q` quits without saving. Stays in the preview loop after a cancelled save so you can retry.
- **`load_bomi.py`** — standalone tool: loads a calibration saved by `calibrate_bomi.py` by name and lets you try it live on the cursor map (again, no robot). Missing name → lists the `.npz` files actually found in `calibrations/`.
- **`reachy_detection.py`** — torso camera → YOLOv8 detection → depth point cloud → grasp geometry building blocks: `capture_and_detect` (grab frame + detect, optionally pre-filtered down to presentable candidates via an injected predicate) and `build_object_point_cloud` (crop/fuse/isolate/measure once an object is confirmed).
- **`reachy_selection.py`** — hover-to-select/confirm UI on top of `reachy_detection.py`'s boxes: dwell-to-select, a **Repositioning** button, Yes/No confirm, and `presentable_filter` (the reachability/gripper-size pre-filter fed into `capture_and_detect`). Dwelling on Repositioning doesn't act itself — `select_object_to_grasp_bomi` returns the `REPOSITION_REQUESTED` sentinel and lets `reachy_control.py` drive the actual mode switch, so this module never depends back on it. `select_object_to_grasp_bomi`/`confirm_grasp_bomi` drive the UI from the BoMI cursor (used by `reachy_control.py`); `MouseTracker` is a standalone mouse-driven alternative used by `tests/*.py`.
- **`reachy_pregrasp.py`** — moves both arms to the pre-grasping posture (non-blocking) and waits for it, showing a progress-bar window while holding the mobile base at zero speed and keeping the BoMI cursor alive.
- **`reachy_grasp.py`** — grasp planning/execution library: from an `ObjectGeometry` (position, axes, width/height, table normal), computes pre-grasp/grasp/lift end-effector poses and drives the arm through them. Also exposes `is_roughly_reachable`, the cheap single-point reachability check `reachy_selection.py` uses to pre-filter detections.
- **`camera_viewer.py`** — standalone script that shows a live feed from either the head/teleop camera or the torso/depth camera (`--camera teleop|torso`); spawned by `reachy_control.py` as its own OS process during Control/pre-grasping pose (teleop) and during repositioning (torso), so the camera's network round-trips stay out of the cursor/velocity loop. Its window is moved to a fixed screen position so it lands beside the cursor-map window instead of on top of it.
- **`graphs.py`** — matplotlib diagnostics (point cloud stages, planned grasp visualization).
- **`stream.py`** — camera-streaming primitives (non-blocking grab+show, blocking live feed) used by `camera_viewer.py` and, in-process, by `reachy_control.py`'s look-before-you-grasp checkpoint right before a grasp executes.
- **`safety.py`** — quit/shutdown safety net: a local `quit_requested` check (Q/ESC or window closed, while a cv2 window has focus) plus an OS-level global watcher (`pynput`, works regardless of focus, even mid-`arm.goto`) that triggers `emergency_shutdown`.

Dependencies between these run one way only, with no cycles: `reachy_grasp.py`/`bomi_teleop.py` have no dependency on the rest, `reachy_detection.py` depends only on `reachy_grasp.py` (for the shared `ObjectGeometry` type), `reachy_selection.py` depends on both, and `reachy_control.py` ties everything together.

---

## Requirements

- The real Reachy 2 robot reachable over the network, with its SDK server running (mobile base + arms + torso depth camera, depending on which script you run).
- A webcam (for `reachy_control.py`).
- Python deps, in the same environment used to run the scripts:

```bash
pip install reachy2-sdk mediapipe opencv-python tensorflow numpy scipy ultralytics matplotlib open3d pynput
```

`reachy_control.py` uses the MediaPipe **Tasks API** (`HandLandmarker`), which needs a `hand_landmarker.task` model file — it's not bundled with the `mediapipe` pip package. Download it once and point `--model` at it (default: `hand_landmarker.task` at the package root):

```bash
curl -o hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

`reachy_control.py` uses YOLOv8 (`ultralytics`) for object detection — `yolov8n.pt` (COCO-pretrained) is auto-downloaded on first run, or pass `--yolo-model` to use different weights.

No ROS 2 install is required to run these scripts (they only talk to the robot via `reachy2_sdk`). The package is still packaged as an `ament_python` ROS 2 package for convenience if you keep it in a ROS 2 workspace, but nothing in the runtime code imports `rclpy`.

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

Or just run the scripts directly with `python3` — no build step required, since none of them depend on ROS 2 at runtime.

---

## Usage

### Full BoMI flow: teleop + grasp — `reachy_control.py`

The real entry point — everything else in the package exists to support this.

```bash
python3 reachy_bomi/reachy_control.py [robot_ip] [--cam 0] [--model hand_landmarker.task] [--yolo-model yolov8n.pt] [--conf 0.5] [--calib NAME] [--subject ID]
# or, from a built ROS 2 workspace:
ros2 run reachy_bomi reachy_control [robot_ip] [--cam 0] [--model hand_landmarker.task] [--yolo-model yolov8n.pt] [--conf 0.5] [--calib NAME] [--subject ID]
```

`robot_ip` is optional if you've set `DEFAULT_ROBOT_IP` in `reachy_control.py` to your robot's IP; otherwise pass it explicitly. `--subject` names the session metrics file (see [Session metrics](#session-metrics)).

**Phase 1 — Calibration** (skipped if `--calib NAME` is given): move your hand through all the positions you intend to use.
`SPACE` records a sample, `ENTER` finishes (minimum 30 samples), `Q`/`Esc`/closing the window quits. The autoencoder map is only kept in memory for that run, not saved or reloaded — to reuse a calibration across runs, save one with `calibrate_bomi.py` first and pass `--calib NAME` here (see "Saving / reusing a calibration" below); a missing/bad name fails fast, before connecting to the robot.

**Phase 2 — Cursor preview:** the same cursor-map window used in Control is shown, but nothing is sent to the robot yet — also where Reachy's head/teleop camera live feed starts streaming, as its own subprocess (`camera_viewer.py`). Hold the cursor centered (region 5) for `SELECTION_HOLD_SECONDS` to proceed into Control, or `Q`/`Esc`/close a window to quit.

**Phase 3 — Control:** hand → autoencoder cursor → 9-region base velocity, mapped and sent to the robot. Hold the cursor centered (region 5) for `SELECTION_HOLD_SECONDS` straight: a Yes/No dialog ("Do you want to continue on the pipeline?") then either moves the arms into the pre-grasping pose or, on "No", goes back to driving through a new cursor preview.

```
 1 | 2 | 3      center (5)          → stop
 4 | 5 | 6      middle column (2,8) → linear only
 7 | 8 | 9      middle row (4,6)    → angular only
                corners (1,3,7,9)   → linear + angular
```

**Phase 3.5 — Pre-grasping pose:** both arms bend to `reachy_pregrasp.PRE_GRASP_ELBOW_PITCH_DEG` (`reachy_pregrasp.goto_pre_grasp_pose`/`wait_for_pre_grasp_pose`), the base holds zero speed; the head-camera feed from Phase 2 just keeps streaming throughout. The cursor/9-region map reappears as a cursor preview (no speed commands); holding the cursor centered resumes Control at reduced velocity (`HALVED_SPEED_FACTOR`). Hold the cursor centered again for `SELECTION_HOLD_SECONDS` (and answer "Yes") to open object selection (the head-camera subprocess is stopped right there, since Phase 4's own checkpoint streams the same camera differently, in-process); the base is held at zero speed here rather than powered off, so it's still ready to drive during Repositioning.

**Phase 4 — Object selection / grasp:** capture → hover-select → Yes/No confirm (`reachy_selection.select_object_to_grasp_bomi`/`confirm_grasp_bomi`) → build point cloud → live feed checkpoint → plan → execute, every hover point being the BoMI cursor (mapped into the window's pixel space) instead of a mouse. "No" re-offers hover-select on the same captured frame; the mobile base is only powered off once an object is confirmed with "Yes". After the object has been placed, a last Yes/No dialog asks whether to pick another object: "Yes" loops back to a fresh capture, "No" ends the session (`_finish_session`: back up, rotate, default posture). Quitting (`Q`/`Esc`/X) at any point in Phase 4, or a failed grasp/place, ends the run — there's no going back to Control.

**Phase 4.5 — Repositioning (opened from object selection):** dwelling on the **Repositioning** button for `REPOSITIONING_HOVER_SECONDS` hands off to `reachy_control.repositioning_navigation`, which reuses the same 9-region cursor UI as Control but caps velocity to `bomi_teleop.MIN_LINEAR`/`MIN_ANGULAR` only (no ramp-up), and streams the **torso/depth camera** instead of the head one (`camera_viewer.py --camera torso`), so you can see what you're driving toward. A cursor-preview sub-phase runs first (nothing sent) until the cursor is re-centered; holding it centered again for `SELECTION_HOLD_SECONDS` stops the base and returns to object selection with a **freshly recaptured** frame (the robot has moved, so the old detections/point positions no longer apply).

ESC/Q stop the robot from *any* window (including a `graphs.py` plot) or the terminal, at any point — see `safety.py`.

### Session metrics

Every run of `reachy_control.py` writes `results_robot/<subject>_session.json` (`<subject>_1`, `_2`, ... for later sessions of the same subject; `--subject`, default `S000`), collected by [`session_metrics.py`](reachy_bomi/session_metrics.py) from the mobile base odometry and the pipeline events. Written in `main()`'s `finally`, so it exists even after a quit or an abort (`end_reason`: `finished` / `quit` / `aborted: ...`).

| Field | Meaning |
|---|---|
| `test_duration` | from the start of Control after the cursor preview (Reachy starts moving) to the "No" to "pick another object?" |
| `navigation_duration` | from the same start to the first time object selection opens (`reached_object_selection` says whether it did) |
| `n_repositioning` | how many times repositioning navigation was used |
| `n_objects_moved`, `objects_moved` | objects picked **and** placed (YOLO class names, in order) |
| `path_length_max_speed` / `_reduced_speed` / `_repositioning` | base path [m] driven at full speed (before the pre-grasp pose), at reduced speed (after it) and during repositioning |
| `path_length_navigation`, `path_length_total` | max + reduced (the path up to the first object selection); all three |
| `optimal_path_length`, `normalized_path_length` | `session_metrics.DEFAULT_OPTIMAL_PATH_LENGTH` [m] — **set it in the file before the session** (depends on the room layout) — and `path_length_navigation / optimal_path_length` |
| `log_dimensionless_jerk` | smoothness of the navigation trajectory (start → first object selection) |
| `region_time_percent`, `region_time_seconds` | share of the driving time the cursor spent in each of the 9 regions — cursor previews and dialogs excluded; the time of every completed dwell (`SELECTION_HOLD_SECONDS × n_dwell`, reported as `region5_dwell_time_removed`) is subtracted from region 5 first |
| `n_dwell`, `n_dwell_declined` | dwells completed while driving (Control + repositioning), and how many of them were answered "No" to "Do you want to continue on the pipeline?" (dwell + switch with no change of state) |

The automatic back-up/rotation at the end is not part of any path length.

### Saving / reusing a calibration — `calibrate_bomi.py` / `customize_bomi.py` / `load_bomi.py`

None of them needs a robot connection — just the webcam and MediaPipe model, so they can run on the operator PC on their own.

```bash
python3 reachy_bomi/calibrate_bomi.py [--cam 0] [--model hand_landmarker.task]
python3 reachy_bomi/customize_bomi.py NAME [--cam 0] [--model hand_landmarker.task]
python3 reachy_bomi/load_bomi.py NAME [--cam 0] [--model hand_landmarker.task]
```

`calibrate_bomi.py` runs the same calibration phase as `reachy_control.py`, fits the map, then previews it live (cursor map only, nothing sent): move your hand to check it feels right, `S` prompts for a name and saves to `calibrations/<name>.npz` (staying in the preview loop if you cancel with a blank name), `Q` quits without saving. `customize_bomi.py NAME` loads a saved map and lets you rotate/flip/scale/offset it live (`[ ]`, `i`/`o`, `-`/`=`, `hjkl`, `r` resets), saving the result under a new name with `S`. `load_bomi.py NAME` loads a file back and shows the same live preview, so you can sanity-check a saved calibration before trusting it. All of them resolve `NAME` the same way `reachy_control.py --calib NAME` does (`bomi_teleop.resolve_calib_path`), so a name saved with one is usable by all three.

### Everything else — `tests/`

Standalone scripts: dry runs, mouse-driven equivalents, and single-piece diagnostics, none of them needing the full calibration → teleop → grasp flow above to exercise one part of the pipeline:

| Script | What it does |
|---|---|
| `test_teleop.py` | `bomi_teleop.py`'s hand-tracking → velocity pipeline, logged instead of sent (no robot needed) |
| `test_navigation.py` | `reachy_control.py`'s calibration → cursor → pre-grasping-pose flow, fully self-contained (no robot, no depth camera, no YOLO) |
| `test_object_dimensions.py` | Live YOLO + depth size estimation only (robot arms/base never power on) |
| `test_reachability.py` | Whether a single pasted 4×4 end-effector pose is IK-reachable for one arm, and moves there if so — isolates a rejection from the rest of the grasp-planning pipeline |
| `test_grasp.py` | Mouse-driven select → confirm → plan → execute grasp flow (the counterpart to `reachy_selection.py`'s BoMI-cursor-driven `select_object_to_grasp_bomi`/`confirm_grasp_bomi`), starting both arms directly in the `elbow_135` posture |
| `test_stream_camera.py` | Head-camera live feed via `reachy2_sdk` directly, no `camera_viewer.py` subprocess — currently a rough/incomplete scratch script, not runnable as-is |

---

## Package layout

```
reachy_bomi/
├── reachy_bomi/                     # Python package
│   ├── __init__.py
│   ├── reachy_control.py            # THE entry point: ties bomi_teleop/reachy_detection/reachy_selection/reachy_pregrasp/reachy_grasp together under one BoMI cursor
│   ├── bomi_teleop.py               # library: webcam/MediaPipe → autoencoder cursor → 9-region velocity building blocks, BoMIMap save/load/customize
│   ├── session_metrics.py           # library: session metrics (durations, path lengths, region shares, dwells) written to results_robot/
│   ├── calibrate_bomi.py            # standalone script: run calibration, preview it live, save it by name -- no robot needed
│   ├── customize_bomi.py            # standalone script: rotate/flip/scale/offset a saved calibration live, save as new -- no robot needed
│   ├── load_bomi.py                 # standalone script: load a saved calibration by name and preview/use it live -- no robot needed
│   ├── reachy_detection.py          # library: torso camera → YOLOv8 → point cloud → grasp geometry building blocks
│   ├── reachy_selection.py          # library: hover-to-select/confirm UI (dwell-select, repositioning, Yes/No confirm, presentable_filter)
│   ├── reachy_pregrasp.py           # library: pre-grasping-pose goto + wait-with-progress-bar UI
│   ├── reachy_grasp.py              # library: grasp planning/execution (pose math, IK search, execute_grasp)
│   ├── camera_viewer.py             # standalone script: head or torso camera live feed (--camera), spawned by reachy_control.py as its own process
│   ├── graphs.py                    # library: matplotlib diagnostics (point cloud stages, grasp plan visualization)
│   ├── stream.py                    # library: camera-streaming primitives (torso + teleop cameras)
│   ├── safety.py                    # library: quit/shutdown safety net (local check + OS-level global watcher)
│   ├── yolov8n.pt                   # YOLOv8 weights (auto-downloaded by ultralytics on first run)
│   └── tests/                       # standalone scripts (mouse-driven grasp, dry runs, diagnostics), see Usage above
├── calibrations/                     # saved BoMIMap .npz files (created on first save, not tracked by git)
├── results_robot/                    # session metrics JSON files (created on first run, not tracked by git)
├── hand_landmarker.task              # MediaPipe model (download separately, see Requirements)
├── reachy_vel.py                     # standalone script: manual mobile-base velocity/lidar-safety-distance calibration, not part of the package
├── resource/
│   └── reachy_bomi                  # ament resource marker
├── package.xml
├── setup.py
├── setup.cfg
├── .gitignore
└── README.md
```

## Known limitations
- Requires a functioning webcam on the machine running `reachy_control.py` (or the teleop-only `tests/test_teleop.py`).
- Grasp planning only handles round objects (`cylinder`/`sphere` shapes in `reachy_detection.SHAPE_BY_CLASS`) — a box's hidden depth can't be recovered from a single camera view the same way.
- The operator's PC and the robot just need network (IP) reachability to each other — no ROS 2 distro matching is required, since communication goes through `reachy2_sdk`'s gRPC interface.
