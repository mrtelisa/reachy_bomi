#!/usr/bin/env python3
"""
BoMI teleop + grasp for Reachy2: the entry point. One hand-tracked cursor
(bomi_teleop) drives the mobile base, then the object selection / grasp UI
(reachy_detection, reachy_selection, reachy_grasp) over reachy2_sdk.

Usage:
    python3 reachy_control.py [robot_ip] [--cam 0] [--model PATH] [--yolo-model PATH]
                              [--conf 0.5] [--calib NAME] [--subject ID]

Phases:
  1. Calibration (skipped with --calib) and 2. cursor preview: see bomi_teleop.
  3. Control: cursor -> 9-region velocity -> mobile base. A dwell in region 5
     asks "continue on the pipeline?": Yes moves the arms to the pre-grasp pose
     and resumes Control at reduced speed; a second dwell + Yes opens object
     selection.
  4. Object selection / grasp: capture -> hover-select -> confirm -> pick a
     place point -> confirm -> grasp -> place, then "pick another object?".
     Repositioning (slow driving with the torso camera) can be requested from
     the selection screen. No at the end runs _finish_session; a failed
     grasp/place runs _abort_and_shutdown. main()'s finally block always powers
     the robot off and writes the session metrics (session_metrics.py).
"""

import argparse
import math
import os
import subprocess
import sys
import time
from typing import Optional

# XWayland: cv2's fullscreen/topmost hints only work there (inherited by camera_viewer.py)
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import numpy as np
from mediapipe.tasks.python.core import base_options
from mediapipe.tasks.python.vision import hand_landmarker
from mediapipe.tasks.python.vision.core import vision_task_running_mode
from reachy2_sdk import ReachySDK
from ultralytics import YOLO

import reachy_detection
import reachy_grasp
import reachy_pregrasp
import reachy_selection
import safety
import bomi_teleop
import graphs
import session_metrics

DEFAULT_ROBOT_IP = "192.168.0.121"

SELECTION_HOLD_SECONDS = reachy_selection.DWELL_HOLD_SECONDS

HALVED_SPEED_FACTOR = 0.75   # velocity multiplier once the arms are in the pre-grasp pose

STARTUP_GAZE_PITCH_DEG = -20.0   # neck pitch on power-on (negative = look down)

REVERSE_BASE_CM = 20.0   # backward translation before the final 180 deg rotation

CAMERA_VIEWER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_viewer.py")

# Latches True the first time object selection opens and never clears: from
# then on the robot is next to the table, so any shutdown (ESC/Q included)
# must back the base away and rotate it before the arms fold in.
_grasp_phase_entered = False
_camera_viewer_proc: Optional[subprocess.Popen] = None
_metrics: Optional[session_metrics.SessionMetrics] = None   # session metrics, created in main()


def _sample_odometry(mobile_base, mode: str) -> None:
    """Feed the base odometry to the session metrics. A failed read (base off,
    gRPC hiccup) is reported and skipped, never fatal."""
    if _metrics is None:
        return
    try:
        _metrics.sample(mobile_base.get_current_odometry(degrees=False), mode)
    except Exception as exc:
        print(f"[metrics] odometry read failed: {exc}")


# --- Windows and camera streaming functions ---
def bring_window_to_front(window_name: str, pos: tuple = None) -> None:
    """Create a cv2 window pinned on top of every other window (so it stays
    visible over the camera-viewer subprocess windows), optionally moved to pos."""
    cv2.namedWindow(window_name)
    cv2.imshow(window_name, np.zeros((300, 510, 3), dtype="uint8"))  # draw_cursor_map's size
    cv2.waitKey(1)
    if pos is not None:
        cv2.moveWindow(window_name, *pos)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)


def make_window_fullscreen(window_name: str) -> None:
    """Create (or resize) a cv2 window fullscreen and on top. Re-called on every
    new capture, since other windows can steal the focus in the meantime."""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, np.zeros((2, 2, 3), dtype="uint8"))  # FULLSCREEN only sticks after a first frame
    cv2.waitKey(1)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    safety.force_fullscreen(window_name)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)


def start_camera_viewer(robot_ip: str, camera: str = "teleop") -> None:
    """Spawn camera_viewer.py ("teleop" = head camera, "torso" = depth camera) as
    its own OS process, so its network round-trips stay out of this loop."""
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
    """Object selection / grasp loop for one object at a time: capture ->
    hover-select -> confirm -> pick a place point -> confirm -> grasp -> place,
    then "pick another object?" (Yes: fresh capture and loop, No: _finish_session).
    Repositioning can be requested from the selection screen. Quitting anywhere
    else just ends the run; main()'s finally block handles the shutdown."""
    global _grasp_phase_entered
    _grasp_phase_entered = True
    if _metrics is not None:
        _metrics.enter_object_selection()
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
                if _metrics is not None:
                    _metrics.repositioning()
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
            if not decision:
                continue  # back to the same captured frame/detections, all blue again

            mobile_base.set_goal_speed(vx=0, vy=0, vtheta=0)
            mobile_base.send_speed_command()
            mobile_base.turn_off()
            geometry = reachy_detection.build_object_point_cloud(depth_cam, class_name, box)
            if geometry is None:
                break
            print(f"[{class_name}] estimated width={geometry.width_m * 100:.1f}cm  "
                  f"height={geometry.height_m * 100:.1f}cm")

            # Plan for both arms: the placement cell the user picks decides which arm is used
            grasp_plans = reachy_grasp.plan_grasps_by_arm(reachy, geometry)
            if not grasp_plans:
                print(f"[{class_name}] no feasible grasp (too wide for the gripper, "
                      "out of reach, or its pose couldn't be estimated)")
                break

            #graphs.show_grasp_plan(geometry, next(iter(grasp_plans.values())))  # diagnostic plot
            target_point, place_arm, crs_x, crs_y = _resolve_and_confirm_place_point(
                cap, landmarker, bomi_map, cursor_filter, crs_x, crs_y,
                depth_cam, reachy, grasp_plans, geometry,
            )
            if target_point is None:
                break

            # Head-camera feed while the arm moves, torso window out of the way
            safety.destroy_window(reachy_detection.CAM_WINDOW_NAME)
            start_camera_viewer(robot_ip)

            plan, place_plan = _replan_and_execute_grasp(reachy, geometry, target_point, place_arm)
            if plan is None:
                _abort_and_shutdown(
                    reachy, mobile_base,
                    f"[{class_name}] {place_arm} couldn't execute a grasp after {MAX_GRASP_ATTEMPTS} attempts",
                )
                break
            if not _place_object(reachy, place_plan):
                _abort_and_shutdown(reachy, mobile_base, f"[{class_name}] execute_place failed")
                break
            if _metrics is not None:
                _metrics.object_moved(class_name)
            _retract_after_place(reachy)

            want_new_object, crs_x, crs_y = reachy_selection.confirm_new_object_bomi(
                cap, landmarker, bomi_map, cursor_filter, crs_x, crs_y,
            )
            if want_new_object is None:
                break
            if not want_new_object:
                if _metrics is not None:
                    _metrics.end_test("finished")   # the user does not want more objects: test over
                _finish_session(reachy, mobile_base)
                break

            mobile_base.turn_on()
            make_window_fullscreen(reachy_detection.CAM_WINDOW_NAME)
            captured = reachy_detection.capture_and_detect(
                depth_cam, model, confidence, reachy_selection.presentable_filter(reachy),
            )
            if captured is None:
                if _metrics is not None:
                    _metrics.end_test("finished")
                _finish_session(reachy, mobile_base)
                break

        return crs_x, crs_y
    finally:
        safety.destroy_window(reachy_detection.CAM_WINDOW_NAME)


def _resolve_and_confirm_place_point(
    cap, landmarker, bomi_map, cursor_filter, crs_x, crs_y,
    depth_cam, reachy, grasp_plans: dict, geometry: reachy_grasp.ObjectGeometry,
) -> tuple:
    """Build the placement grid once (every cell IK-checked per arm: unreachable
    cells are red and cannot be dwelled on), dwell-select a cell and ask a
    Yes/No confirm; "No" re-offers the same grid. Returns the raw target point
    and the arm that serves it -- not a plan: the caller re-plans right before
    moving. (None, None, crs_x, crs_y) on quit; nothing has been grasped yet."""
    grid = reachy_selection.build_place_grid(depth_cam, reachy, grasp_plans, geometry)
    if grid is None:
        return None, None, crs_x, crs_y

    while True:
        target_point, place_arm, crs_x, crs_y = reachy_selection.select_place_location_bomi(
            cap, landmarker, bomi_map, cursor_filter, crs_x, crs_y, grid,
        )
        if target_point is None:
            return None, None, crs_x, crs_y

        decision, crs_x, crs_y = reachy_selection.confirm_place_bomi(
            cap, landmarker, bomi_map, cursor_filter, crs_x, crs_y,
        )
        if decision is None:
            return None, None, crs_x, crs_y
        if decision:
            return target_point, place_arm, crs_x, crs_y


# reachy2_sdk's IK seeds its search from the last solution, so a pose that
# plan_grasp/plan_place just validated can still fail execute_grasp's own
# re-check a few IK calls later; re-planning from scratch usually finds an
# alternative right away, so the whole cycle is retried a few times.
MAX_GRASP_ATTEMPTS = 3


def _replan_and_execute_grasp(reachy, geometry, target_point, arm_name: str) -> tuple:
    """plan_grasp -> plan_place -> execute_grasp for arm_name (the arm the chosen
    placement cell needs), up to MAX_GRASP_ATTEMPTS. execute_grasp only moves
    once its IK pre-check passes, so a failed attempt never leaves the arm
    mid-motion. Returns (plan, place_plan), or (None, None) if all attempts failed."""
    for attempt in range(1, MAX_GRASP_ATTEMPTS + 1):
        plan = reachy_grasp.plan_grasp(reachy, geometry, arm_name=arm_name)
        if plan is None:
            continue
        place_plan = reachy_grasp.plan_place(reachy, plan, geometry.table_normal, target_point)
        if place_plan is None:
            continue
        if reachy_grasp.execute_grasp(reachy, plan):
            return plan, place_plan
        print(f"[grasp] attempt {attempt}/{MAX_GRASP_ATTEMPTS} failed to execute -- retrying with a fresh plan")
    return None, None


def _place_object(reachy, place_plan: reachy_grasp.GraspPlan) -> bool:
    """Move above place_plan's target, descend, open the gripper and retreat
    (reachy_grasp.execute_place, all cartesian so nothing is dragged sideways)."""
    return reachy_grasp.execute_place(reachy, place_plan)


def _retract_after_place(reachy) -> None:
    """After a placement: head back to its default posture, head-camera feed
    stopped, both arms retracted to the pre-grasp pose -- a neutral state to
    sit in while asking whether to pick another object."""
    if reachy.head is not None:
        reachy.head.goto_posture(duration=1.0, wait=False)
        stop_camera_viewer()

    print(f"\nRetracting both arms to the pre-grasping posture "
          f"(elbow pitch {reachy_pregrasp.PRE_GRASP_ELBOW_PITCH_DEG:.0f} deg)...")
    goto_ids = reachy_pregrasp.goto_pre_grasp_pose(reachy)
    while not all(reachy.is_goto_finished(goto_id) for goto_id in goto_ids):
        time.sleep(0.1)


def _finish_session(reachy, mobile_base) -> None:
    """End of a session (user declined another object): back the base up and
    rotate it 180 deg through rotate_base_once's shared gate (so an ESC
    shutdown cannot rotate it twice), then default posture. main()'s finally
    block powers everything off afterwards."""
    print(f"\nRotating the base {safety.SHUTDOWN_ROTATION_DEG:.0f} deg...")
    safety.rotate_base_once(mobile_base, reverse_cm=REVERSE_BASE_CM)

    print("\nReturning to default posture...")
    reachy.goto_posture("default", duration=3.0, wait=True)


def _abort_and_shutdown(reachy, mobile_base, reason: str) -> None:
    """A grasp/place failed to complete (unreachable pose or a RuntimeError
    mid-motion): report it, back the base up and rotate it, then power the
    robot off right away instead of pretending the run can continue."""
    print(f"[ERROR] {reason} -- plan aborted")
    if _metrics is not None:
        _metrics.end_test("aborted: " + reason)
    safety.rotate_base_once(mobile_base, reverse_cm=REVERSE_BASE_CM)
    reachy.turn_off_smoothly()


# --- BoMI control/navigation, with a dwell-in-center switch into grasp mode ---
def teleop_with_grasp_switch(cap, landmarker, bomi_map, mobile_base, depth_cam, model, confidence, reachy, robot_ip,
                               cursor_filter=None, crs_x=None, crs_y=None) -> None:
    """Control loop: cursor -> 9-region velocities -> mobile base. Holding the
    cursor in region 5 for SELECTION_HOLD_SECONDS opens a Yes/No dialog: the
    first Yes moves the arms to the pre-grasp pose and resumes Control at
    reduced speed, the second one opens object selection; No goes back to
    driving through a cursor preview. Pass the preview's cursor_filter/crs_x/
    crs_y to continue the cursor without a velocity blip on entry."""
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
    if _metrics is not None:
        _metrics.start_test()   # Reachy starts moving after the cursor preview: test starts here

    while True:
        # A quit watcher may already be shutting down (and rotating the base) on
        # another thread: stop publishing speed commands or they fight the rotation
        if safety.shutdown_started():
            return

        hand_frame, crs_x, crs_y, hand_detected = bomi_teleop.update_bomi_cursor(
            cap, landmarker, bomi_map, cursor_filter, crs_x, crs_y,
        )
        if hand_frame is None:
            continue

        if hand_detected:
            region = bomi_teleop.check_region_cursor(crs_x, crs_y)
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
        if _metrics is not None:
            _metrics.region_tick(region, now)

        cv2.imshow(map_window, bomi_teleop.draw_cursor_map(crs_x, crs_y, region, message))
        cv2.moveWindow(map_window, *bomi_teleop.MAP_WINDOW_POS)  # re-pin, the WM can move it
        cv2.setWindowProperty(map_window, cv2.WND_PROP_TOPMOST, 1)  # re-pin (same-process windows only)
        safety.raise_window(map_window)  # actually wins over the cross-process fullscreen camera_viewer window

        if center_progress >= 1.0 and not pre_grasp_reached:
            decision, crs_x, crs_y = reachy_selection.confirm_bomi(
                cap, landmarker, bomi_map, cursor_filter, crs_x, crs_y,
                lines=["Do you want to continue on the pipeline?"],
                on_frame=_hold_base_still,
            )
            if _metrics is not None:
                _metrics.dwell(decision)
            if decision is None:
                break
            center_hold_start = None
            if not decision:
                # No: back to driving, through a cursor preview
                cursor_filter.reset(crs_x, crs_y)
                crs_x, crs_y = bomi_teleop.cursor_preview_phase(
                    cap, landmarker, bomi_map, cursor_filter=cursor_filter, crs_x=crs_x, crs_y=crs_y, show_cam=False,
                    hold_seconds=SELECTION_HOLD_SECONDS,
                )
                center_hold_start = None
                continue

            # First dwell: hold the base while the arms move to the pre-grasp pose
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
            if _metrics is not None:
                _metrics.dwell(decision)
            if decision is None:
                break
            center_hold_start = None
            if not decision:
                # No: back to driving, through a cursor preview
                cursor_filter.reset(crs_x, crs_y)
                crs_x, crs_y = bomi_teleop.cursor_preview_phase(
                    cap, landmarker, bomi_map, cursor_filter=cursor_filter, crs_x=crs_x, crs_y=crs_y, show_cam=False,
                    hold_seconds=SELECTION_HOLD_SECONDS,
                )
                center_hold_start = None
                continue

            # Second dwell: object selection (base held at zero speed, still on for repositioning)
            mobile_base.set_goal_speed(vx=0, vy=0, vtheta=0)
            mobile_base.send_speed_command()
            print("\nMobile base held at zero speed. Switching to object selection.")
            safety.destroy_window(map_window)
            stop_camera_viewer()
            _run_grasp_mode(
                cap, landmarker, bomi_map, cursor_filter, depth_cam, model, confidence, reachy, crs_x, crs_y,
                mobile_base, robot_ip,
            )
            break

        if now - last_publish >= dt:
            message = f"lin_vel:{lin_vel:.3f} ang_vel:{ang_vel:.3f}"
            # vtheta is in deg/s for reachy2_sdk
            mobile_base.set_goal_speed(vx=lin_vel, vy=0, vtheta=math.degrees(ang_vel))
            mobile_base.send_speed_command()
            last_publish = now
            _sample_odometry(mobile_base,
                             session_metrics.MODE_REDUCED if pre_grasp_reached else session_metrics.MODE_MAX)

        key = cv2.waitKey(1) & 0xFF
        if safety.quit_requested(key, map_window):
            break

    # During a shutdown the watcher thread already zeroed the speed; turning the
    # base off here would cut its rotation short
    if not safety.shutdown_started():
        mobile_base.set_goal_speed(vx=0, vy=0, vtheta=0)
        mobile_base.send_speed_command()
        mobile_base.turn_off()
    safety.destroy_window(map_window)
    stop_camera_viewer()  # no-op if already stopped (Phase 4 switch stops it itself)


# --- Repositioning: 9-region nav at minimum speed, torso camera, entered from object selection ---
def repositioning_navigation(cap, landmarker, bomi_map, cursor_filter, crs_x, crs_y, mobile_base, robot_ip) -> tuple:
    """Same 9-region cursor UI as Control, with velocities pinned to
    MIN_LINEAR/MIN_ANGULAR and the torso/depth camera streaming, so the user
    can adjust the base in front of the objects. Starts with a cursor preview;
    returns (crs_x, crs_y, quit_now) once the cursor has been held centred
    for SELECTION_HOLD_SECONDS (back to object selection with a fresh capture)."""
    dt = 1.0 / bomi_teleop.PUBLISH_HZ
    last_publish = time.time()
    map_window = bomi_teleop.MAP_WINDOW_NAME
    region = bomi_teleop.check_region_cursor(crs_x, crs_y)
    message = "lin_vel:0.000 ang_vel:0.000"
    center_hold_start = None

    print("\n=== REPOSITIONING ===  Q = quit  |  hold the cursor centered (region 5) "
          f"for {SELECTION_HOLD_SECONDS:.0f}s to return to object selection")
    # Torso stream and map over the object-selection window
    cv2.setWindowProperty(reachy_detection.CAM_WINDOW_NAME, cv2.WND_PROP_TOPMOST, 0)
    bring_window_to_front(map_window, bomi_teleop.MAP_WINDOW_POS)  # re-positions it, since it was destroyed on the last exit
    start_camera_viewer(robot_ip, camera="torso")

    try:
        # Cursor preview first, as in Control
        crs_x, crs_y = bomi_teleop.cursor_preview_phase(
            cap, landmarker, bomi_map, cursor_filter=cursor_filter, crs_x=crs_x, crs_y=crs_y, show_cam=False,
            hold_seconds=SELECTION_HOLD_SECONDS,
        )
        center_hold_start = None

        while True:
            if safety.shutdown_started():
                return crs_x, crs_y, True

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
            if _metrics is not None:
                _metrics.region_tick(region, now)

            cv2.imshow(map_window, bomi_teleop.draw_cursor_map(crs_x, crs_y, region, message))
            cv2.moveWindow(map_window, *bomi_teleop.MAP_WINDOW_POS)  # re-pin, the WM can move it
            cv2.setWindowProperty(map_window, cv2.WND_PROP_TOPMOST, 1)  # re-pin (same-process windows only)
            safety.raise_window(map_window)  # actually wins over the cross-process fullscreen camera_viewer window

            if center_progress >= 1.0:
                if _metrics is not None:
                    _metrics.dwell(True)   # back to object selection
                mobile_base.set_goal_speed(vx=0, vy=0, vtheta=0)
                mobile_base.send_speed_command()
                safety.destroy_window(map_window)
                return crs_x, crs_y, False

            if now - last_publish >= dt:
                message = f"lin_vel:{lin_vel:.3f} ang_vel:{ang_vel:.3f}"
                mobile_base.set_goal_speed(vx=lin_vel, vy=0, vtheta=math.degrees(ang_vel))
                mobile_base.send_speed_command()
                last_publish = now
                _sample_odometry(mobile_base, session_metrics.MODE_REPOSITIONING)

            key = cv2.waitKey(1) & 0xFF
            if safety.quit_requested(key, map_window):
                mobile_base.set_goal_speed(vx=0, vy=0, vtheta=0)
                mobile_base.send_speed_command()
                safety.destroy_window(map_window)
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
    parser.add_argument("--subject", default="S000",
                        help="Subject id for the session metrics file in results_robot/ (default: S000)")
    cli_args = parser.parse_args()

    global _metrics
    # optimal path length for normalized_path_length: session_metrics.DEFAULT_OPTIMAL_PATH_LENGTH
    _metrics = session_metrics.SessionMetrics(cli_args.subject, dwell_seconds=SELECTION_HOLD_SECONDS)

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
        safety.emergency_shutdown(reachy, mobile_base, rotate_base_before_shutdown=_grasp_phase_entered)

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
        # Session metrics: a run that did not end with "no more objects" is
        # closed here as a quit; the file is written whatever happened
        _metrics.end_test("quit")
        summary = _metrics.save()
        print(f"[metrics] test {summary['test_duration'] or 0:.0f}s, navigation {summary['navigation_duration'] or 0:.0f}s, "
              f"path {summary['path_length_total']:.2f} m (nav {summary['path_length_navigation']:.2f} m), "
              f"{summary['n_repositioning']} repositioning, objects moved: {summary['objects_moved']}")
        safety.safe_robot_shutdown(reachy, mobile_base, rotate_base_before_shutdown=_grasp_phase_entered)
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        if landmarker is not None:
            landmarker.close()
        reachy.disconnect()


if __name__ == "__main__":
    main()
