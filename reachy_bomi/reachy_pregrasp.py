#!/usr/bin/env python3
"""
Pre-grasp pose: send both arms to the pre-grasp posture (non-blocking) and
wait for it with a progress bar, holding the base still and the cursor alive.
"""

import time

import cv2
import numpy as np

import bomi_teleop
import safety

PRE_GRASP_ELBOW_PITCH_DEG = -135.0  # [deg], elbow pitch for the pre-grasping posture
PRE_GRASP_MOVE_DURATION = 5.0       # [s], arm goto duration into the pre-grasping pose

WAIT_WINDOW_NAME = "BoMI - Waiting"
WAIT_CANVAS_WIDTH = 640
WAIT_CANVAS_HEIGHT = 220


def goto_pre_grasp_pose(reachy, duration: float = PRE_GRASP_MOVE_DURATION) -> list:
    """Send both arms towards the pre-grasping posture, non-blocking. Returns the GoToIds
    to poll for completion, skipping any arm that isn't reported or powered on."""
    goto_ids = []
    for arm in (reachy.r_arm, reachy.l_arm):
        if arm is None or not arm.is_on():
            continue
        joints = arm.get_default_posture_joints(common_posture="elbow_90")
        joints[3] = PRE_GRASP_ELBOW_PITCH_DEG
        goto_ids.append(arm.goto(joints, duration=duration, wait=False))
    return goto_ids


def _draw_wait_canvas(progress: float) -> np.ndarray:
    """Canvas shown while the arms move into the pre-grasping pose"""
    canvas = np.full((WAIT_CANVAS_HEIGHT, WAIT_CANVAS_WIDTH, 3), 30, dtype=np.uint8)
    cv2.putText(canvas, "Waiting for the robot",
                (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(canvas, "to finished its movement",
                (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    bar_x1, bar_y1, bar_x2, bar_y2 = 30, 160, WAIT_CANVAS_WIDTH - 30, 190
    cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x2, bar_y2), (90, 90, 90), 1)
    filled_x = bar_x1 + int((bar_x2 - bar_x1) * progress)
    cv2.rectangle(canvas, (bar_x1, bar_y1), (filled_x, bar_y2), (0, 255, 255), -1)
    return canvas


def wait_for_pre_grasp_pose(cap, landmarker, bomi_map, cursor_filter, crs_x, crs_y, reachy, mobile_base, goto_ids):
    """Blocks until every goto in goto_ids has finished, showing a dedicated
    waiting window with a progress bar. The base is repeatedly held at zero
    speed throughout. Returns the possibly updated cursor position and whether
    the user quit."""
    dt = 1.0 / bomi_teleop.PUBLISH_HZ
    last_publish = time.time()
    start = time.time()

    while True:
        # A shutdown may already be rotating the base on another thread
        if safety.shutdown_started():
            safety.destroy_window(WAIT_WINDOW_NAME)
            return crs_x, crs_y, True

        _, crs_x, crs_y, _ = bomi_teleop.update_bomi_cursor(cap, landmarker, bomi_map, cursor_filter, crs_x, crs_y)

        progress = min((time.time() - start) / PRE_GRASP_MOVE_DURATION, 1.0)
        cv2.imshow(WAIT_WINDOW_NAME, _draw_wait_canvas(progress))
        cv2.setWindowProperty(WAIT_WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)  # re-pin (same-process windows only)
        safety.raise_window(WAIT_WINDOW_NAME)  # actually wins over the cross-process fullscreen camera_viewer window

        now = time.time()
        if now - last_publish >= dt:
            mobile_base.set_goal_speed(vx=0, vy=0, vtheta=0)
            mobile_base.send_speed_command()
            last_publish = now

        key = cv2.waitKey(1) & 0xFF
        if safety.quit_requested(key, WAIT_WINDOW_NAME):
            safety.destroy_window(WAIT_WINDOW_NAME)
            return crs_x, crs_y, True

        if all(reachy.is_goto_finished(goto_id) for goto_id in goto_ids):
            safety.destroy_window(WAIT_WINDOW_NAME)
            return crs_x, crs_y, False
