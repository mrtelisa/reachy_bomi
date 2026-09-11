#!/usr/bin/env python3
"""
BoMI teleop + grasp for Reachy2: entry point that drives bomi_teleop.py,
reachy_detection.py, reachy_selection.py and reachy_grasp.py through a
single hand-tracked PCA cursor. Driving the mobile base and hovering YOLO
boxes/buttons in the grasp UI both use that same cursor.

Runs entirely on a single PC with a webcam and network access to the robot;
mediapipe/opencv/ultralytics computation and the reachy2_sdk client live in
the same process.

Dependencies:
    pip install reachy2-sdk mediapipe opencv-python scikit-learn numpy scipy ultralytics matplotlib pynput

Usage:
    python3 reachy_control.py [robot_ip]

    <robot_ip> is optional; if omitted, DEFAULT_ROBOT_IP (in bomi_teleop.py) is used.

    Options:
        --cam INDEX          Webcam index. Default: 0
        --model PATH         Path to the MediaPipe hand_landmarker.task model.
        --yolo-model PATH    YOLOv8 weights (.pt). Default: yolov8n.pt
        --conf FLOAT         Minimum YOLO detection confidence. Default: 0.5

Camera streaming: Phase 3.5 and 4 spawn camera_viewer.py as their own OS process.

Phases 1-2 (Calibration, Cursor preview): see bomi_teleop.py.

Phase 3 - Control:
    Hand movement -> PCA cursor -> 9-region velocity -> mobile base, exactly
    as in bomi_teleop.py. Hold the cursor centered (region 5) for
    SELECTION_HOLD_SECONDS straight to move the arms into a pre-grasping pose.

Phase 3.5 - Pre-grasping pose (opened from Control):
    Both arms move to a pre-grasping posturethe base is held at zero
    speed. Once arms are in place, the cursor/9-region map reappears
    (same as Control); holding the cursor centered (region 5) for 
    SELECTION_HOLD_SECONDS straight resumes Control with linear/angular velocities 
    halved. Hold it centered to stop the base and open object selection.

Phase 4 - Object selection / grasp (opened from Control):
    Capture -> hover-to-select -> Yes/No confirm flow, see reachy_selection.py
    (select_object_to_grasp_bomi/confirm_grasp_bomi), every hover point being
    the BoMI cursor. Answering "No" re-offers hover-select on a fresh capture;
    quitting (Q/ESC/X) or a successful grasp both end the run.
    On a successful lift, _place_back_and_wind_down puts the
    object right back down where it was picked up, retracts both arms to
    the pre-grasping posture, rotates the base 180 deg, returns to the default posture, 
    then main()'s finally block powers everything off.
"""

import argparse
import math
import os
import subprocess
import sys
import time
from typing import Optional

# Force XWayland so cv2's fullscreen/topmost window hints actually work (must be
# set before cv2 creates a window; camera_viewer.py inherits this too)
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import numpy as np
from mediapipe.tasks.python.core import base_options
from mediapipe.tasks.python.vision import hand_landmarker
from mediapipe.tasks.python.vision.core import vision_task_running_mode
from reachy2_sdk import ReachySDK
from ultralytics import YOLO

import graphs
import reachy_detection
import reachy_grasp
import reachy_pregrasp
import reachy_selection
import safety
import bomi_teleop

# Placeholder — replace with the robot's actual IP
DEFAULT_ROBOT_IP = "192.168.0.121"

# How long the cursor must stay in region 5 before Control moves to the next step
SELECTION_HOLD_SECONDS = reachy_selection.DWELL_HOLD_SECONDS

# Linear/angular velocity multiplier for Control once the arms are in the
# pre-grasping pose
HALVED_SPEED_FACTOR = 0.75 # max_lin = 0.375 [m/s], max_ang = 0.825 [rad/s] 

# Neck pitch on power-on, applied on top of the "default" posture (whose own
# neck pitch is -10 deg, i.e. looking slightly up) -- negative = look down.
STARTUP_GAZE_PITCH_DEG = -20.0 

# How far the base translates backward before rotating 180 deg at the end of
# a successful grasp -- positive = backward
REVERSE_BASE_CM = 20.0 # TODO find the right value

# camera_viewer.py (head camera, LEFT eye), spawned as its own process by
# start_camera_viewer during Control/pre-grasping pose
CAMERA_VIEWER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_viewer.py")

# global param to let _on_emergency_quit/main()'s block work,
# however deep in the call stack the quit was triggered from.
_in_grasp_phase = False # True for the duration of _run_grasp_mode
_camera_viewer_proc: Optional[subprocess.Popen] = None # head-camera-viewer subprocess, if one is currently running


# --- Windows and camera streaming functions ---
def bring_window_to_front(window_name: str, pos: tuple = None) -> None:
    """Pins a cv2 window on top of every other window, staying pinned
    so the window stays visible in front of the camera-viewer subprocess windows.
    Qt backend only; a harmless no-op elsewhere. pos, if given, moves the
    window to a fixed (x, y) screen position first."""
    cv2.namedWindow(window_name)
    cv2.imshow(window_name, np.zeros((300, 510, 3), dtype="uint8"))  # match draw_cursor_map's size, or the move below gets undone
    cv2.waitKey(1)
    if pos is not None:
        cv2.moveWindow(window_name, *pos)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)


def make_window_fullscreen(window_name: str) -> None:
    """Creates (or resizes) a cv2 window, pins it fullscreen and brings it to the
    front. Re-call this on every newly captured frame -- other windows (e.g. the
    editor) can otherwise steal focus over it while repositioning runs."""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # A window needs at least one rendered frame before FULLSCREEN sticks
    cv2.imshow(window_name, np.zeros((2, 2, 3), dtype="uint8"))
    cv2.waitKey(1)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    safety.force_fullscreen(window_name)  # some Qt builds ignore the request above; ask the WM directly too
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)


def start_camera_viewer(robot_ip: str, camera: str = "teleop") -> None:
    """Spawns camera_viewer.py as its own OS process and returns
    immediately -- keeps its network round-trips (Camera.get_frame) out
    of this process's own loops. camera is "teleop" (head) or "torso" (chest/depth)."""
    global _camera_viewer_proc
    _camera_viewer_proc = subprocess.Popen(
        [sys.executable, CAMERA_VIEWER_SCRIPT, robot_ip, "--camera", camera]
    )


def stop_camera_viewer() -> None:
    global _camera_viewer_proc
    if _camera_viewer_proc is not None and _camera_viewer_proc.poll() is None:
        _camera_viewer_proc.terminate()
    _camera_viewer_proc = None


# --- Grasping flow ---
def _run_grasp_mode(cap, landmarker, bomi_map, cursor_filter, depth_cam, model, confidence, reachy, crs_x, crs_y,
                     mobile_base, robot_ip):
    """capture -> hover-select -> confirm, looping back on "No" until an object is
    confirmed (then builds its point cloud, streams the live feed as a
    checkpoint, then plans and executes the grasp) or the user quits"""
    global _in_grasp_phase
    _in_grasp_phase = True
    try:
        make_window_fullscreen(reachy_detection.CAM_WINDOW_NAME)
        captured = reachy_detection.capture_and_detect(
            depth_cam, model, confidence, reachy_selection.presentable_filter(reachy),
        )
        if captured is None:
            return crs_x, crs_y

        while True:
            class_name, box, captured, crs_x, crs_y = reachy_selection.select_object_to_grasp_bomi(
                cap, landmarker, bomi_map, cursor_filter, crs_x, crs_y,
                depth_cam, model, confidence, captured, reachy,
            )
            if class_name is reachy_selection.REPOSITION_REQUESTED:
                crs_x, crs_y, quit_now = repositioning_navigation(
                    cap, landmarker, bomi_map, cursor_filter, crs_x, crs_y, mobile_base, robot_ip,
                )
                if quit_now:
                    break
                make_window_fullscreen(reachy_detection.CAM_WINDOW_NAME)
                captured = reachy_detection.capture_and_detect(
                    depth_cam, model, confidence, reachy_selection.presentable_filter(reachy),
                )
                if captured is None:
                    break
                continue
            if class_name is None:
                break

            decision, crs_x, crs_y = reachy_selection.confirm_grasp_bomi(
                cap, landmarker, bomi_map, cursor_filter, crs_x, crs_y, class_name,
            )
            if decision is None:
                break
            if decision:
                mobile_base.set_goal_speed(vx=0, vy=0, vtheta=0)
                mobile_base.send_speed_command()
                mobile_base.turn_off()
                geometry = reachy_detection.build_object_point_cloud(depth_cam, class_name, box)
                if geometry is not None:
                    print(f"[{class_name}] estimated width={geometry.width_m * 100:.1f}cm  "
                          f"height={geometry.height_m * 100:.1f}cm")
                    # Live feed to see the scene while the arm moves
                    start_camera_viewer(robot_ip)
                    plan = reachy_grasp.plan_grasp(reachy, geometry)
                    if plan is None:
                        print(f"[{class_name}] no feasible grasp (too wide for the gripper, "
                              "or its pose couldn't be estimated)")
                    else:
                        graphs.show_grasp_plan(geometry, plan)
                        if reachy_grasp.execute_grasp(reachy, plan):
                            _place_back_and_wind_down(reachy, mobile_base, plan)
                break
            # if decision =  False -> back to the same captured frame/detections, all blue again

        cv2.destroyWindow(reachy_detection.CAM_WINDOW_NAME)
        return crs_x, crs_y
    finally:
        _in_grasp_phase = False


def _place_back_and_wind_down(reachy, mobile_base, plan: reachy_grasp.GraspPlan) -> None:
    """Runs once, right after a successful grasp+lift: puts the object back
    where it was picked up from and retreats to pregrasp. Then the base translates,
    rotates and evetually returns to the default posture"""
    reachy_grasp.place_back(reachy, plan)

    if reachy.head is not None:
        reachy.head.goto_posture(duration=1.0, wait=False)
        stop_camera_viewer() # no-op if already stopped (Phase 4 switch stops it itself)

    print(f"\nRetracting both arms to the pre-grasping posture "
          f"(elbow pitch {reachy_pregrasp.PRE_GRASP_ELBOW_PITCH_DEG:.0f} deg)...")
    goto_ids = reachy_pregrasp.goto_pre_grasp_pose(reachy)
    while not all(reachy.is_goto_finished(goto_id) for goto_id in goto_ids):
        time.sleep(0.1)

    print(f"\nRotating the base {safety.SHUTDOWN_ROTATION_DEG:.0f} deg...")
    try:
        mobile_base.turn_on()
        mobile_base.translate_by(x=-REVERSE_BASE_CM / 100.0, y=0.0, wait=True) 
        mobile_base.rotate_by(safety.SHUTDOWN_ROTATION_DEG, wait=True)
    except Exception as exc:
        print(f"[WARN] Could not rotate the base ({exc}).")

    print("\nReturning to default posture...")
    reachy.goto_posture("default", duration=3.0, wait=True)


# --- BoMI control/navigation, with a dwell-in-center switch into grasp mode ---
def teleop_with_grasp_switch(cap, landmarker, bomi_map, mobile_base, depth_cam, model, confidence, reachy, robot_ip,
                               cursor_filter=None, crs_x=None, crs_y=None) -> None:
    # Pass in the cursor_filter/crs_x/crs_y from the preceding cursor preview
    # to continue them, avoiding a velocity blip on entry.
    dt = 1.0 / bomi_teleop.PUBLISH_HZ
    last_publish = time.time()
    cursor_filter = cursor_filter or bomi_teleop.CursorFilter()
    map_window = bomi_teleop.MAP_WINDOW_NAME

    if crs_x is None or crs_y is None:
        crs_x, crs_y = bomi_teleop.BASE_WIDTH / 2.0, bomi_teleop.BASE_HEIGHT / 2.0
    region = bomi_teleop.check_region_cursor(crs_x, crs_y)
    message = "lin_vel:0.000 ang_vel:0.000"
    center_hold_start = None
    speed_scale = 1.0
    pre_grasp_reached = False

    def _hold_base_still() -> None:
        mobile_base.set_goal_speed(vx=0, vy=0, vtheta=0)
        mobile_base.send_speed_command()

    print("\n=== CONTROL ===  Q = quit  |  hold the cursor centered (region 5) "
          f"for {SELECTION_HOLD_SECONDS:.0f}s to move to the pre-grasping pose")

    while True:
        hand_frame, crs_x, crs_y, hand_detected = bomi_teleop.update_bomi_cursor(
            cap, landmarker, bomi_map, cursor_filter, crs_x, crs_y,
        )
        if hand_frame is None:
            continue

        if hand_detected:
            region = bomi_teleop.check_region_cursor(crs_x, crs_y)
            # speed_scale narrows the range (smaller max)
            lin_vel, ang_vel = bomi_teleop.compute_dynamic_vel_from_cursor(
                crs_x, crs_y,
                max_linear=bomi_teleop.MAX_LINEAR * speed_scale,
                max_angular=bomi_teleop.MAX_ANGULAR * speed_scale,
            )
            lin_vel, ang_vel = bomi_teleop.apply_region_velocity_mask(region, lin_vel, ang_vel)
        else:
            lin_vel, ang_vel = 0.0, 0.0

        now = time.time()
        center_hold_start = (center_hold_start or now) if (hand_detected and region == 5) else None
        center_progress = (
            min((now - center_hold_start) / SELECTION_HOLD_SECONDS, 1.0) if center_hold_start else 0.0
        )

        cv2.imshow(map_window, bomi_teleop.draw_cursor_map(crs_x, crs_y, region, message))
        cv2.moveWindow(map_window, *bomi_teleop.MAP_WINDOW_POS)  # a just-closed window can make the WM reclaim the position otherwise
        cv2.setWindowProperty(map_window, cv2.WND_PROP_TOPMOST, 1)  # re-pin (same-process windows only)
        safety.raise_window(map_window)  # actually wins over the cross-process fullscreen camera_viewer window

        if center_progress >= 1.0 and not pre_grasp_reached:
            decision, crs_x, crs_y = reachy_selection.confirm_bomi(
                cap, landmarker, bomi_map, cursor_filter, crs_x, crs_y,
                lines=["Do you want to continue on the pipeline?"],
                on_frame=_hold_base_still,
            )
            if decision is None:
                break
            center_hold_start = None
            if not decision:
                # Back to the previous navigation mode - through a cursor preview first
                cursor_filter.reset(crs_x, crs_y)
                crs_x, crs_y = bomi_teleop.cursor_preview_phase(
                    cap, landmarker, bomi_map, cursor_filter=cursor_filter, crs_x=crs_x, crs_y=crs_y, show_cam=False,
                    hold_seconds=SELECTION_HOLD_SECONDS,
                )
                center_hold_start = None
                continue

            # First dwell: stop and hold here while the arms move to pre-grasping pose,
            # then resume Control at limited speed
            mobile_base.set_goal_speed(vx=0, vy=0, vtheta=0)
            mobile_base.send_speed_command()
            mobile_base.lidar.safety_critical_distance = bomi_teleop.LIDAR_CRITICAL_DISTANCE_SLOWDOWN
                
            print("\nMoving arms to pre-grasping pose "
                  f"(elbow pitch {reachy_pregrasp.PRE_GRASP_ELBOW_PITCH_DEG:.0f} deg)...")
            goto_ids = reachy_pregrasp.goto_pre_grasp_pose(reachy)
            crs_x, crs_y, quit_now = reachy_pregrasp.wait_for_pre_grasp_pose(
                cap, landmarker, bomi_map, cursor_filter, crs_x, crs_y, reachy, mobile_base, goto_ids,
            )
            if quit_now:
                break
            cursor_filter.reset(crs_x, crs_y)

            crs_x, crs_y = bomi_teleop.cursor_preview_phase(
                cap, landmarker, bomi_map, cursor_filter=cursor_filter, crs_x=crs_x, crs_y=crs_y, show_cam=False,
                hold_seconds=SELECTION_HOLD_SECONDS,
            )

            pre_grasp_reached = True
            speed_scale = HALVED_SPEED_FACTOR
            center_hold_start = None
            print("\nPre-grasping pose reached. Control resumed at limited speed — "
                  f"hold the cursor centered (region 5) for {SELECTION_HOLD_SECONDS:.0f}s "
                  "to open object selection")
            continue

        if center_progress >= 1.0 and pre_grasp_reached:
            decision, crs_x, crs_y = reachy_selection.confirm_bomi(
                cap, landmarker, bomi_map, cursor_filter, crs_x, crs_y,
                lines=["Do you want to continue on the pipeline?"],
                on_frame=_hold_base_still,
            )
            if decision is None:
                break
            center_hold_start = None
            if not decision:
                # Back to the previous navigation mode - through a cursor preview first
                cursor_filter.reset(crs_x, crs_y)
                crs_x, crs_y = bomi_teleop.cursor_preview_phase(
                    cap, landmarker, bomi_map, cursor_filter=cursor_filter, crs_x=crs_x, crs_y=crs_y, show_cam=False,
                    hold_seconds=SELECTION_HOLD_SECONDS,
                )
                center_hold_start = None
                continue

            # Second dwell: switch to object selection mode. The base is held at
            # zero speed but stays powered on (needed for repositioning).
            mobile_base.set_goal_speed(vx=0, vy=0, vtheta=0)
            mobile_base.send_speed_command()
            print("\nMobile base held at zero speed. Switching to object selection.")
            cv2.destroyWindow(map_window)
            stop_camera_viewer()
            _run_grasp_mode(
                cap, landmarker, bomi_map, cursor_filter, depth_cam, model, confidence, reachy, crs_x, crs_y,
                mobile_base, robot_ip,
            )
            break

        if now - last_publish >= dt:
            message = f"lin_vel:{lin_vel:.3f} ang_vel:{ang_vel:.3f}"
            # vtheta is in degrees/s for reachy2_sdk, ang_vel is computed in rad/s
            mobile_base.set_goal_speed(vx=lin_vel, vy=0, vtheta=math.degrees(ang_vel))
            mobile_base.send_speed_command()
            last_publish = now

        key = cv2.waitKey(1) & 0xFF
        if safety.quit_requested(key, map_window):
            break

    mobile_base.set_goal_speed(vx=0, vy=0, vtheta=0)
    mobile_base.send_speed_command()
    mobile_base.turn_off()
    cv2.destroyWindow(map_window)
    stop_camera_viewer()  # no-op if already stopped (Phase 4 switch stops it itself)


# --- Repositioning: 9-region nav at minimum speed, torso camera, entered from object selection ---
def repositioning_navigation(cap, landmarker, bomi_map, cursor_filter, crs_x, crs_y, mobile_base, robot_ip) -> tuple:
    """Same 9-region cursor UI as Control, with velocities pinned to
    MIN_LINEAR/MIN_ANGULAR and torso/depth camera streaming. 
    Returns (crs_x, crs_y, quit_now) once the cursor has been
    held centered (region 5) for SELECTION_HOLD_SECONDS."""
    dt = 1.0 / bomi_teleop.PUBLISH_HZ
    last_publish = time.time()
    map_window = bomi_teleop.MAP_WINDOW_NAME
    region = bomi_teleop.check_region_cursor(crs_x, crs_y)
    message = "lin_vel:0.000 ang_vel:0.000"
    center_hold_start = None

    print("\n=== REPOSITIONING ===  Q = quit  |  hold the cursor centered (region 5) "
          f"for {SELECTION_HOLD_SECONDS:.0f}s to return to object selection")
    # Let the torso stream (and the map, re-pinned every frame below) cover the
    # object-selection window instead of fighting it for the top spot
    cv2.setWindowProperty(reachy_detection.CAM_WINDOW_NAME, cv2.WND_PROP_TOPMOST, 0)
    bring_window_to_front(map_window, bomi_teleop.MAP_WINDOW_POS)  # re-positions it, since it was destroyed on the last exit
    start_camera_viewer(robot_ip, camera="torso")

    try:
        # Preview first (robot not moving) so velocities only start once the
        # cursor is confirmed centered, same as Control on entry/resume.
        crs_x, crs_y = bomi_teleop.cursor_preview_phase(
            cap, landmarker, bomi_map, cursor_filter=cursor_filter, crs_x=crs_x, crs_y=crs_y, show_cam=False,
            hold_seconds=SELECTION_HOLD_SECONDS,
        )
        center_hold_start = None

        while True:
            _, crs_x, crs_y, hand_detected = bomi_teleop.update_bomi_cursor(
                cap, landmarker, bomi_map, cursor_filter, crs_x, crs_y,
            )

            if hand_detected:
                region = bomi_teleop.check_region_cursor(crs_x, crs_y)
                lin_vel, ang_vel = bomi_teleop.compute_dynamic_vel_from_cursor(
                    crs_x, crs_y, max_linear=bomi_teleop.MIN_LINEAR, max_angular=bomi_teleop.MIN_ANGULAR,
                )
                lin_vel, ang_vel = bomi_teleop.apply_region_velocity_mask(region, lin_vel, ang_vel)
            else:
                lin_vel, ang_vel = 0.0, 0.0

            now = time.time()
            center_hold_start = (center_hold_start or now) if (hand_detected and region == 5) else None
            center_progress = (
                min((now - center_hold_start) / SELECTION_HOLD_SECONDS, 1.0) if center_hold_start else 0.0
            )

            cv2.imshow(map_window, bomi_teleop.draw_cursor_map(crs_x, crs_y, region, message))
            cv2.moveWindow(map_window, *bomi_teleop.MAP_WINDOW_POS)  # a just-closed window can make the WM reclaim the position otherwise
            cv2.setWindowProperty(map_window, cv2.WND_PROP_TOPMOST, 1)  # re-pin (same-process windows only)
            safety.raise_window(map_window)  # actually wins over the cross-process fullscreen camera_viewer window

            if center_progress >= 1.0:
                mobile_base.set_goal_speed(vx=0, vy=0, vtheta=0)
                mobile_base.send_speed_command()
                cv2.destroyWindow(map_window)
                return crs_x, crs_y, False

            if now - last_publish >= dt:
                message = f"lin_vel:{lin_vel:.3f} ang_vel:{ang_vel:.3f}"
                mobile_base.set_goal_speed(vx=lin_vel, vy=0, vtheta=math.degrees(ang_vel))
                mobile_base.send_speed_command()
                last_publish = now

            key = cv2.waitKey(1) & 0xFF
            if safety.quit_requested(key, map_window):
                mobile_base.set_goal_speed(vx=0, vy=0, vtheta=0)
                mobile_base.send_speed_command()
                cv2.destroyWindow(map_window)
                return crs_x, crs_y, True
    finally:
        stop_camera_viewer()


# --- Entry point ---
def main() -> None:
    parser = argparse.ArgumentParser(
        description="BoMI teleop for Reachy2 that switches into BoMI-driven "
                     "object selection/grasp when the cursor is held centered."
    )
    parser.add_argument("robot_ip", nargs="?", default=DEFAULT_ROBOT_IP,
                        help=f"IP address of the Reachy robot (default: {DEFAULT_ROBOT_IP})")
    parser.add_argument("--cam", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--model", default=bomi_teleop.DEFAULT_MODEL_PATH,
                        help="Path to the MediaPipe hand_landmarker.task model "
                             f"(default: {bomi_teleop.DEFAULT_MODEL_PATH}).")
    parser.add_argument("--yolo-model", default=reachy_detection.YOLO_MODEL_PATH,
                        help=f"Path to YOLOv8 weights (default: {reachy_detection.YOLO_MODEL_PATH})")
    parser.add_argument("--conf", type=float, default=reachy_detection.YOLO_CONFIDENCE,
                        help=f"Minimum detection confidence (default: {reachy_detection.YOLO_CONFIDENCE})")
    parser.add_argument("--calib", default=None,
                        help="Name of a calibration saved by calibrate_bomi.py to load instead of "
                             "running the calibration phase")
    cli_args = parser.parse_args()

    if not os.path.exists(cli_args.model):
        print(f"[ERROR] MediaPipe model not found: '{cli_args.model}'")
        print("        Download hand_landmarker.task and pass its path with --model.")
        sys.exit(1)

    calib_path = None
    if cli_args.calib:
        calib_path = bomi_teleop.resolve_calib_path(cli_args.calib)
        if not os.path.exists(calib_path):
            print(f"[ERROR] No calibration file '{calib_path}' found.")
            if os.path.isdir(bomi_teleop.CALIB_DIR):
                available = [f for f in os.listdir(bomi_teleop.CALIB_DIR) if f.endswith(".npz")]
                if available:
                    print("        Available: " + ", ".join(sorted(available)))
            sys.exit(1)

    reachy = ReachySDK(host=cli_args.robot_ip)
    if reachy.mobile_base is None:
        print(f"[ERROR] No mobile base reported by the robot at '{cli_args.robot_ip}'")
        reachy.disconnect()
        sys.exit(1)
    if reachy.cameras is None or reachy.cameras.depth is None:
        print(f"[ERROR] No depth camera reported by the robot at '{cli_args.robot_ip}'")
        reachy.disconnect()
        sys.exit(1)
    if reachy.cameras.teleop is None:
        print(f"[ERROR] No head/teleop camera reported by the robot at '{cli_args.robot_ip}'")
        reachy.disconnect()
        sys.exit(1)
    if reachy.r_arm is None and reachy.l_arm is None:
        print(f"[ERROR] No arm reported by the robot at '{cli_args.robot_ip}'")
        reachy.disconnect()
        sys.exit(1)

    mobile_base = reachy.mobile_base
    depth_cam = reachy.cameras.depth

    reachy.turn_on()
    reachy.goto_posture("default", duration=3.0, wait=True)
    reachy.r_arm.gripper.open()
    reachy.l_arm.gripper.open()
    mobile_base.lidar.safety_enabled = True
    mobile_base.lidar.safety_slowdown_distance = bomi_teleop.LIDAR_SLOWDOWN_DISTANCE
    mobile_base.lidar.safety_critical_distance = bomi_teleop.LIDAR_CRITICAL_DISTANCE
    mobile_base.turn_on()


    def _on_emergency_quit() -> None:
        stop_camera_viewer()
        safety.emergency_shutdown(reachy, mobile_base, rotate_base_before_shutdown=_in_grasp_phase)

    safety.start_global_quit_watcher(_on_emergency_quit)
    stop_terminal_watcher = safety.start_terminal_quit_watcher(_on_emergency_quit)

    print(f"Loading YOLO model '{cli_args.yolo_model}'...")
    model = YOLO(cli_args.yolo_model)

    cap = None
    landmarker = None
    try:
        cap = cv2.VideoCapture(cli_args.cam, cv2.CAP_V4L2)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open camera {cli_args.cam}")
            sys.exit(1)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always the freshest frame, not a growing backlog

        landmarker_options = hand_landmarker.HandLandmarkerOptions(
            base_options=base_options.BaseOptions(model_asset_path=cli_args.model),
            running_mode=vision_task_running_mode.VisionTaskRunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        landmarker = hand_landmarker.HandLandmarker.create_from_options(landmarker_options)

        bomi_map = bomi_teleop.BoMIMap()
        if calib_path is not None:
            bomi_map.load_map_bomi(calib_path)
            print(f"Loaded calibration from {calib_path} (calibration phase skipped)")
        else:
            samples = bomi_teleop.calibration_phase(cap, landmarker)
            bomi_map.fit(samples)
            print("PCA map fitted")
        bring_window_to_front(bomi_teleop.MAP_WINDOW_NAME, bomi_teleop.MAP_WINDOW_POS)
        start_camera_viewer(cli_args.robot_ip)
        reachy.head.rotate_by(pitch=-STARTUP_GAZE_PITCH_DEG, yaw=0, roll=0, wait=False)  # look down

        cursor_filter = bomi_teleop.CursorFilter()
        crs_x, crs_y = bomi_teleop.cursor_preview_phase(
            cap, landmarker, bomi_map, cursor_filter=cursor_filter, show_cam=False,
            hold_seconds=SELECTION_HOLD_SECONDS,
        )
        teleop_with_grasp_switch(cap, landmarker, bomi_map, mobile_base, depth_cam, model,
                                   cli_args.conf, reachy, cli_args.robot_ip,
                                   cursor_filter=cursor_filter, crs_x=crs_x, crs_y=crs_y)
    finally:
        stop_camera_viewer()
        if stop_terminal_watcher is not None:
            stop_terminal_watcher()
        safety.safe_robot_shutdown(reachy, mobile_base, rotate_base_before_shutdown=_in_grasp_phase)
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        if landmarker is not None:
            landmarker.close()
        reachy.disconnect()


if __name__ == "__main__":
    main()
