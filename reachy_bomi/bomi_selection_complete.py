#!/usr/bin/env python3
"""
BoMI teleop + grasp selection for Reachy2: merges bomi_teleop.py and
bomi_grasp.py into a single BoMI cursor. Driving the mobile base and hovering
YOLO boxes/buttons in the grasp UI both use the same hand-tracked PCA cursor —
there's no mouse involved anywhere.

Runs entirely on a single PC with a webcam and network access to the robot;
mediapipe/opencv/ultralytics computation and the reachy2_sdk client live in
the same process.

Dependencies:
    pip install reachy2-sdk mediapipe opencv-python scikit-learn numpy scipy ultralytics

Usage:
    python3 bomi_selection_complete.py [robot_ip]

    <robot_ip> is optional; if omitted, DEFAULT_ROBOT_IP (in bomi_teleop.py) is used.

    Options:
        --cam INDEX          Webcam index. Default: 0
        --model PATH         Path to the MediaPipe hand_landmarker.task model.
        --yolo-model PATH    YOLOv8 weights (.pt). Default: yolov8n.pt
        --conf FLOAT         Minimum YOLO detection confidence. Default: 0.5

Phases 1-2 (Calibration, Cursor preview): identical to bomi_teleop.py.

Phase 3 - Control:
    Hand movement -> PCA cursor -> 9-region velocity -> mobile base, exactly
    as in bomi_teleop.py. Hold the cursor centered (region 5) for
    SELECTION_HOLD_SECONDS straight to stop the base and open object selection.

Phase 4 - Object selection / grasp (opened from Control):
    Same capture -> hover-to-select -> Yes/No confirm flow as bomi_grasp.py,
    and every hover point is the BoMI cursor (mapped into that window's pixel
    space). Answering "No" or quitting (Q/ESC/X) returns to Control.
"""

import argparse
import math
import os
import sys
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python.core import base_options
from mediapipe.tasks.python.vision import hand_landmarker
from mediapipe.tasks.python.vision.core import vision_task_running_mode
from reachy2_sdk import ReachySDK
from ultralytics import YOLO

import bomi_grasp as grasp
import bomi_teleop as teleop

# Placeholder — replace with the robot's actual IP.
DEFAULT_ROBOT_IP = "192.168.0.124"

# How long the cursor must stay in region 5 (dead-zone center) before Control
# hands off to the grasp/selection UI.
SELECTION_HOLD_SECONDS = 5.0

COLOR_CURSOR = (255, 0, 255)  # magenta marker for the BoMI cursor, distinct from grasp's box colors


def _update_bomi_cursor(cap, landmarker, bomi_map, cursor_filter, crs_x, crs_y):
    """One iteration of hand tracking: reads a webcam frame, runs the hand
    landmarker, and returns (frame_with_landmarks, crs_x, crs_y, hand_detected).
    Cursor position is carried over unchanged when no hand is detected;
    hand_detected tells callers that drive the robot to stop instead of
    coasting on a stale position."""
    ret, frame = cap.read()
    if not ret:
        return None, crs_x, crs_y, False

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = landmarker.detect_for_video(mp_image, int(time.time() * 1000))

    if not results.hand_landmarks:
        return frame, crs_x, crs_y, False

    hl = results.hand_landmarks[0]
    teleop._draw_hand_landmarks(frame, hl)
    mirror_x = results.handedness[0][0].category_name == "Right"
    crs_x, crs_y = bomi_map.transform(teleop._extract_hand_features(hl, mirror_x))
    crs_x, crs_y = cursor_filter.update(crs_x, crs_y)
    return frame, crs_x, crs_y, True


def _map_bomi_to_frame(crs_x: float, crs_y: float, width: int, height: int) -> tuple:
    """Rescale a cursor position from the BASE_WIDTH x BASE_HEIGHT BoMI screen
    space into an arbitrary window's pixel space (a captured camera frame or
    the fixed-size confirm canvas)."""
    return (
        int(crs_x / teleop.BASE_WIDTH * width),
        int(crs_y / teleop.BASE_HEIGHT * height),
    )


def _draw_bomi_cursor(frame, x: int, y: int) -> None:
    cv2.drawMarker(frame, (x, y), COLOR_CURSOR, markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)


# --- BoMI-cursor equivalents of bomi_grasp's mouse-driven UI ---

def _select_object_to_grasp_bomi(
    cap, landmarker, bomi_map, cursor_filter, crs_x, crs_y,
    depth_cam, model, confidence, captured,
):
    """Same hover-to-select/hover-Refresh loop as bomi_grasp._select_object_to_grasp,
    but the hover point is the BoMI cursor mapped into the captured frame."""
    base_frame, detections, labels = captured
    frame_h, frame_w = base_frame.shape[:2]

    hovered_box = None
    hover_start = None
    button_hover_start = None

    print(f"\n=== CAPTURED FRAME (RGB + YOLO) ===  Q = quit  |  "
          f"hold Refresh for {grasp.REFRESH_HOVER_SECONDS:.0f}s to recapture")

    while True:
        hand_frame, crs_x, crs_y, _ = _update_bomi_cursor(cap, landmarker, bomi_map, cursor_filter, crs_x, crs_y)
        if hand_frame is not None:
            cv2.imshow(teleop.CAM_WINDOW_NAME, hand_frame)

        gx, gy = _map_bomi_to_frame(crs_x, crs_y, frame_w, frame_h)
        now = time.time()

        on_button = grasp._box_contains(grasp.REFRESH_BUTTON_BOX, gx, gy)
        if not on_button:
            button_hover_start = None
        elif button_hover_start is None:
            button_hover_start = now
        elif now - button_hover_start >= grasp.REFRESH_HOVER_SECONDS:
            refreshed = grasp._capture_and_detect(depth_cam, model, confidence)
            if refreshed is not None:
                base_frame, detections, labels = refreshed
                frame_h, frame_w = base_frame.shape[:2]
            hovered_box, hover_start = None, None
            button_hover_start = None

        frame = base_frame.copy()
        hovered = grasp._find_hovered_detection(detections, gx, gy)

        if hovered is None:
            hovered_box, hover_start = None, None
        else:
            box = hovered[2]
            if hovered_box is None or grasp._iou(box, hovered_box) < grasp.HOVER_IOU_MATCH:
                hover_start = now
            hovered_box = box
        hover_duration = (now - hover_start) if hover_start is not None else 0.0
        is_held = hovered is not None and hover_duration >= grasp.HOVER_HOLD_SECONDS

        if is_held:
            box = hovered[2]
            grasp._draw_box(frame, box, labels[box], grasp.COLOR_GREEN)
            _draw_bomi_cursor(frame, gx, gy)
            cv2.imshow(grasp.CAM_WINDOW_NAME, frame)
            cv2.waitKey(1)
            return hovered[0], (base_frame, detections, labels), crs_x, crs_y

        hover_progress = min(hover_duration / grasp.HOVER_HOLD_SECONDS, 1.0) if hovered is not None else 0.0
        for class_name, conf, box in detections:
            is_hovered = hovered is not None and box == hovered[2]
            color = grasp.COLOR_YELLOW if is_hovered else grasp.COLOR_BLUE
            grasp._draw_box(frame, box, labels[box], color, hover_progress if is_hovered else 0.0)

        button_progress = min((now - button_hover_start) / grasp.REFRESH_HOVER_SECONDS, 1.0) if button_hover_start else 0.0
        grasp._draw_refresh_button(frame, button_progress)
        _draw_bomi_cursor(frame, gx, gy)
        cv2.imshow(grasp.CAM_WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if grasp._quit_requested(key, grasp.CAM_WINDOW_NAME) or grasp._quit_requested(key, teleop.CAM_WINDOW_NAME):
            return None, (base_frame, detections, labels), crs_x, crs_y


def _confirm_grasp_bomi(cap, landmarker, bomi_map, cursor_filter, crs_x, crs_y, class_name):
    """Same Yes/No dwell dialog as bomi_grasp._confirm_grasp, hovered with the
    BoMI cursor mapped into the confirm canvas instead of the mouse."""
    yes_hover_start = None
    no_hover_start = None

    while True:
        hand_frame, crs_x, crs_y, _ = _update_bomi_cursor(cap, landmarker, bomi_map, cursor_filter, crs_x, crs_y)
        if hand_frame is not None:
            cv2.imshow(teleop.CAM_WINDOW_NAME, hand_frame)

        gx, gy = _map_bomi_to_frame(crs_x, crs_y, grasp.CONFIRM_CANVAS_WIDTH, grasp.CONFIRM_CANVAS_HEIGHT)
        now = time.time()
        on_yes = grasp._box_contains(grasp.YES_BUTTON_BOX, gx, gy)
        on_no = grasp._box_contains(grasp.NO_BUTTON_BOX, gx, gy)

        yes_hover_start = (yes_hover_start or now) if on_yes else None
        no_hover_start = (no_hover_start or now) if on_no else None
        yes_progress = min((now - yes_hover_start) / grasp.CONFIRM_HOVER_SECONDS, 1.0) if yes_hover_start else 0.0
        no_progress = min((now - no_hover_start) / grasp.CONFIRM_HOVER_SECONDS, 1.0) if no_hover_start else 0.0

        canvas = grasp._draw_confirm_canvas(class_name, yes_progress, no_progress)
        _draw_bomi_cursor(canvas, gx, gy)
        cv2.imshow(grasp.CONFIRM_WINDOW_NAME, canvas)

        key = cv2.waitKey(1) & 0xFF
        result = None
        quit_now = grasp._quit_requested(key, grasp.CONFIRM_WINDOW_NAME)
        if yes_progress >= 1.0:
            result = True
        elif no_progress >= 1.0:
            result = False

        if quit_now or result is not None:
            cv2.destroyWindow(grasp.CONFIRM_WINDOW_NAME)
            return (None if quit_now else result), crs_x, crs_y


def _run_grasp_mode(cap, landmarker, bomi_map, cursor_filter, depth_cam, model, confidence, crs_x, crs_y):
    """BoMI-driven equivalent of bomi_grasp._show_torso_camera: capture ->
    hover-select -> confirm, looping back on "No" until an object is
    confirmed (then streams the live feed) or the user quits."""
    captured = grasp._capture_and_detect(depth_cam, model, confidence)
    if captured is None:
        return crs_x, crs_y

    while True:
        class_name, captured, crs_x, crs_y = _select_object_to_grasp_bomi(
            cap, landmarker, bomi_map, cursor_filter, crs_x, crs_y,
            depth_cam, model, confidence, captured,
        )
        if class_name is None:
            break

        decision, crs_x, crs_y = _confirm_grasp_bomi(
            cap, landmarker, bomi_map, cursor_filter, crs_x, crs_y, class_name,
        )
        if decision is None:
            break
        if decision:
            grasp._stream_torso_camera(depth_cam)
            break
        # No -> back to the same captured frame/detections, all blue again

    cv2.destroyWindow(grasp.CAM_WINDOW_NAME)
    return crs_x, crs_y


# --- Control, with a dwell-in-center switch into grasp mode ---

def _teleop_with_grasp_switch(cap, landmarker, bomi_map, mobile_base, depth_cam, model, confidence) -> None:
    dt = 1.0 / teleop.PUBLISH_HZ
    last_publish = time.time()
    cursor_filter = teleop.CursorFilter()
    cam_window = teleop.CAM_WINDOW_NAME
    map_window = teleop.MAP_WINDOW_NAME

    crs_x, crs_y = teleop.BASE_WIDTH / 2.0, teleop.BASE_HEIGHT / 2.0
    region = teleop.check_region_cursor(crs_x, crs_y)
    message = "lin_vel:0.000 ang_vel:0.000"
    center_hold_start = None

    print("\n=== CONTROL ===  Q = quit  |  hold the cursor centered (region 5) "
          f"for {SELECTION_HOLD_SECONDS:.0f}s to open object selection")

    while True:
        hand_frame, crs_x, crs_y, hand_detected = _update_bomi_cursor(
            cap, landmarker, bomi_map, cursor_filter, crs_x, crs_y,
        )
        if hand_frame is None:
            continue

        if hand_detected:
            region = teleop.check_region_cursor(crs_x, crs_y)
            lin_vel, ang_vel = teleop.compute_dynamic_vel_from_cursor(crs_x, crs_y)
            lin_vel, ang_vel = teleop.apply_region_velocity_mask(region, lin_vel, ang_vel)
        else:
            lin_vel, ang_vel = 0.0, 0.0

        now = time.time()
        # Only accrue dwell time while actively tracked and centered, so a
        # dropped hand while the stale cursor happens to sit in region 5
        # can't silently trigger the switch into grasp mode.
        center_hold_start = (center_hold_start or now) if (hand_detected and region == 5) else None
        center_progress = (
            min((now - center_hold_start) / SELECTION_HOLD_SECONDS, 1.0) if center_hold_start else 0.0
        )

        cv2.putText(hand_frame, f"region={region}  cursor=({crs_x:.0f},{crs_y:.0f})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(hand_frame, f"-> mobile base: {message}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if center_progress > 0:
            cv2.putText(hand_frame, f"hold to select object: {center_progress * 100:.0f}%",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_CURSOR, 2)

        cv2.imshow(cam_window, hand_frame)
        cv2.imshow(map_window, teleop._draw_cursor_map(crs_x, crs_y, region, message))

        if center_progress >= 1.0:
            # One-way switch: once object selection opens, the base is powered
            # off for good, not just zeroed — Control never runs again after
            # this, so there's no loop iteration left that could re-drive it.
            mobile_base.set_goal_speed(vx=0, vy=0, vtheta=0)
            mobile_base.send_speed_command()
            mobile_base.turn_off()
            print("\nMobile base powered off. Switching to object selection for good.")
            _run_grasp_mode(
                cap, landmarker, bomi_map, cursor_filter, depth_cam, model, confidence, crs_x, crs_y,
            )
            break

        if now - last_publish >= dt:
            message = f"lin_vel:{lin_vel:.3f} ang_vel:{ang_vel:.3f}"
            # vtheta is in degrees/s for reachy2_sdk, ang_vel is computed in rad/s
            mobile_base.set_goal_speed(vx=lin_vel, vy=0, vtheta=math.degrees(ang_vel))
            mobile_base.send_speed_command()
            last_publish = now

        key = cv2.waitKey(1) & 0xFF
        if teleop._quit_requested(key, cam_window) or teleop._quit_requested(key, map_window):
            break

    mobile_base.set_goal_speed(vx=0, vy=0, vtheta=0)
    mobile_base.send_speed_command()
    mobile_base.turn_off()
    cv2.destroyWindow(cam_window)
    cv2.destroyWindow(map_window)


# --- Entry point ---

def main() -> None:
    parser = argparse.ArgumentParser(
        description="BoMI teleop for Reachy2 that switches into BoMI-driven "
                     "object selection/grasp when the cursor is held centered."
    )
    parser.add_argument("robot_ip", nargs="?", default=DEFAULT_ROBOT_IP,
                        help=f"IP address of the Reachy robot (default: {DEFAULT_ROBOT_IP})")
    parser.add_argument("--cam", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--model", default=teleop.DEFAULT_MODEL_PATH,
                        help="Path to the MediaPipe hand_landmarker.task model "
                             f"(default: {teleop.DEFAULT_MODEL_PATH}).")
    parser.add_argument("--yolo-model", default=grasp.YOLO_MODEL_PATH,
                        help=f"Path to YOLOv8 weights (default: {grasp.YOLO_MODEL_PATH})")
    parser.add_argument("--conf", type=float, default=grasp.YOLO_CONFIDENCE,
                        help=f"Minimum detection confidence (default: {grasp.YOLO_CONFIDENCE})")
    cli_args = parser.parse_args()

    if not os.path.exists(cli_args.model):
        print(f"[ERROR] MediaPipe model not found: '{cli_args.model}'")
        print("        Download hand_landmarker.task and pass its path with --model.")
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

    mobile_base = reachy.mobile_base
    depth_cam = reachy.cameras.depth

    reachy.turn_on()
    mobile_base.lidar.safety_enabled = True
    mobile_base.lidar.safety_slowdown_distance = teleop.LIDAR_SLOWDOWN_DISTANCE
    mobile_base.lidar.safety_critical_distance = teleop.LIDAR_CRITICAL_DISTANCE
    mobile_base.turn_on()

    print(f"Loading YOLO model '{cli_args.yolo_model}'...")
    model = YOLO(cli_args.yolo_model)

    cap = None
    landmarker = None
    try:
        cap = cv2.VideoCapture(cli_args.cam)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open camera {cli_args.cam}")
            sys.exit(1)

        landmarker_options = hand_landmarker.HandLandmarkerOptions(
            base_options=base_options.BaseOptions(model_asset_path=cli_args.model),
            running_mode=vision_task_running_mode.VisionTaskRunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        landmarker = hand_landmarker.HandLandmarker.create_from_options(landmarker_options)

        bomi_map = teleop.BoMIMap()
        samples = teleop._calibration_phase(cap, landmarker)
        bomi_map.fit(samples)
        print("PCA map fitted")

        teleop._cursor_preview_phase(cap, landmarker, bomi_map)
        _teleop_with_grasp_switch(cap, landmarker, bomi_map, mobile_base, depth_cam, model, cli_args.conf)
    finally:
        mobile_base.set_goal_speed(vx=0, vy=0, vtheta=0)
        mobile_base.send_speed_command()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        if landmarker is not None:
            landmarker.close()
        reachy.disconnect()


if __name__ == "__main__":
    main()
