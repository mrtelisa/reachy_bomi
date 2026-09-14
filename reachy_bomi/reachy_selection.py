#!/usr/bin/env python3
"""
Library module -- hover-to-select/confirm UI for a detected object: maps a
hover point onto reachy_detection.py's YOLO boxes, drives the
dwell-to-select/repositioning/confirm flow, and draws the corresponding overlays.

select_object_to_grasp_bomi/confirm_grasp_bomi drive that hover point from a
BoMI cursor (bomi_teleop.py). 

MouseTracker is a standalone alternative hover-point source, for driving the 
same box_contains/find_hovered_detection geometry from a real mouse instead.
"""

import math
import time
from typing import List, Optional

import cv2
import numpy as np

import bomi_teleop
import reachy_detection
import reachy_grasp
import safety

Box = reachy_detection.Box

# [s], single dwell time for all hover-to-select/confirm interactions
DWELL_HOLD_SECONDS = 3.0

HOVER_HOLD_SECONDS = DWELL_HOLD_SECONDS
HOVER_IOU_MATCH = 0.3     # min overlap between frames to count as "still hovering the same object"

REPOSITIONING_BUTTON_BOX: Box = (10, 10, 260, 90)
COLOR_BUTTON = (255, 0, 0)
COLOR_BUTTON_TEXT = (255, 255, 255)
REPOSITIONING_HOVER_SECONDS = DWELL_HOLD_SECONDS

# Sentinel class_name for "user dwelled on Repositioning" - handled by reachy_control.py
REPOSITION_REQUESTED = object()

# --- Free-point placement dwell (e.g. "put it in the box") ---
PLACE_HOVER_SECONDS = DWELL_HOLD_SECONDS

# Radius (px) of the tolerance circle the cursor must stay inside to keep
# dwelling on a placement point. Big enough to absorb the hand tremor the
# BoMI cursor carries through even after cursor_filter's smoothing (nobody can
# hold a hand-tracked cursor to an exact pixel for 3s); small enough that the
# confirmed point still reads as "here", not "somewhere over there".
PLACE_TARGET_RADIUS_PX = 40

# EMA weight applied to the anchor each frame the cursor stays inside the
# tolerance circle, so it drifts along with a slow, unintentional hand
# movement instead of comparing forever against the very first sample (which
# would spuriously reset the dwell for a tremor that wanders gradually).
PLACE_ANCHOR_SMOOTHING = 0.15

COLOR_PLACE_TARGET = (0, 255, 255)

CONFIRM_WINDOW_NAME = "BoMI - Confirm Grasp"
CONFIRM_CANVAS_WIDTH = 520
CONFIRM_CANVAS_HEIGHT = 260
CONFIRM_HOVER_SECONDS = DWELL_HOLD_SECONDS
YES_BUTTON_BOX: Box = (60, 150, 240, 220)
NO_BUTTON_BOX: Box = (280, 150, 460, 220)
COLOR_YES = (0, 200, 0)
COLOR_NO = (0, 0, 255)

COLOR_CURSOR = (255, 0, 255)

# BGR colors
COLOR_BLUE = (255, 0, 0)
COLOR_YELLOW = (0, 255, 255)


# --- Drawing windows and cursor ---
def draw_repositioning_button(frame: np.ndarray, progress: float = 0.0) -> None:
    x1, y1, x2, y2 = REPOSITIONING_BUTTON_BOX
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BUTTON, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BUTTON_TEXT, 1)

    text = "Repositioning"
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    text_x = x1 + ((x2 - x1) - text_w) // 2
    text_y = y1 + ((y2 - y1) + text_h) // 2
    cv2.putText(frame, text, (text_x, text_y), font, scale, COLOR_BUTTON_TEXT, thickness)

    if progress > 0:
        bar_width = int((x2 - x1) * progress)
        cv2.rectangle(frame, (x1, y2 - 8), (x1 + bar_width, y2), COLOR_BUTTON_TEXT, -1)


def draw_confirm_canvas(lines: List[str], yes_progress: float, no_progress: float) -> np.ndarray:
    canvas = np.full((CONFIRM_CANVAS_HEIGHT, CONFIRM_CANVAS_WIDTH, 3), 30, dtype=np.uint8)

    for i, line in enumerate(lines):
        cv2.putText(canvas, line, (20, 60 + i * 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_BUTTON_TEXT, 2)

    for box, label, color, progress in (
        (YES_BUTTON_BOX, "Yes", COLOR_YES, yes_progress),
        (NO_BUTTON_BOX, "No", COLOR_NO, no_progress),
    ):
        x1, y1, x2, y2 = box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), COLOR_BUTTON_TEXT, 1)
        cv2.putText(canvas, label, (x1 + 60, y2 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_BUTTON_TEXT, 2)
        if progress > 0:
            bar_width = int((x2 - x1) * progress)
            cv2.rectangle(canvas, (x1, y2 - 8), (x1 + bar_width, y2), COLOR_BUTTON_TEXT, -1)

    return canvas


def _draw_bomi_cursor(frame: np.ndarray, x: int, y: int) -> None:
    cv2.drawMarker(frame, (x, y), COLOR_CURSOR, markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)


def _draw_place_target(frame: np.ndarray, anchor_x: int, anchor_y: int, progress: float) -> None:
    """Tolerance circle for select_place_location_bomi: an outline the user
    dwells inside, filled clockwise as a pie to show progress -- same
    affordance as the hover progress bars, just circular since there's no
    YOLO box to draw it against."""
    cv2.circle(frame, (anchor_x, anchor_y), PLACE_TARGET_RADIUS_PX, COLOR_PLACE_TARGET, 2)
    if progress > 0:
        cv2.ellipse(
            frame, (anchor_x, anchor_y), (PLACE_TARGET_RADIUS_PX, PLACE_TARGET_RADIUS_PX),
            -90, 0, 360 * progress, COLOR_PLACE_TARGET, -1,
        )


# --- Hover geometry ---
def box_contains(box: Box, x: int, y: int) -> bool:
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def iou(box_a: Box, box_b: Box) -> float:
    """ Computes the Intersection Over Union of two bboxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter_area / float(area_a + area_b - inter_area)


def find_hovered_detection(
    detections: List["reachy_detection.Detection"], x: int, y: int,
) -> Optional["reachy_detection.Detection"]:
    """Smallest-area detection whose box contains (x, y), or None if none does."""
    hovered = None
    hovered_area = None
    for detection in detections:
        _, _, box = detection
        if not box_contains(box, x, y):
            continue
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        if hovered_area is None or area < hovered_area:
            hovered, hovered_area = detection, area
    return hovered


# --- Filter for detections ---
def presentable_filter(reachy):
    """is_selectable predicate for reachy_detection.capture_and_detect: keeps
    only detections plausibly reachable and narrow enough for the gripper"""
    def is_selectable(position: Optional[np.ndarray], width: Optional[float]) -> bool:
        return (
            position is not None
            and reachy_grasp.is_roughly_reachable(reachy, position)
            and width is not None
            and width <= reachy_grasp.GRIPPER_MAX_OPENING_M
        )
    return is_selectable


# --- BoMI-cursor-driven selection flow ---
def select_object_to_grasp_bomi(
    cap, landmarker, bomi_map, cursor_filter, crs_x, crs_y,
    depth_cam, model, confidence, captured, reachy,
):
    """hover-to-select loop, where the hover point is the BoMI cursor mapped
    into the captured frame"""
    base_frame, detections, labels = captured
    frame_h, frame_w = base_frame.shape[:2]

    hovered_box = None
    hover_start = None
    button_hover_start = None

    print(f"\n=== CAPTURED FRAME (RGB + YOLO) ===  Q = quit  |  "
          f"hold Repositioning for {REPOSITIONING_HOVER_SECONDS:.0f}s to reposition")

    while True:
        _, crs_x, crs_y, _ = bomi_teleop.update_bomi_cursor(cap, landmarker, bomi_map, cursor_filter, crs_x, crs_y)

        gx, gy = bomi_teleop.map_bomi_to_frame(crs_x, crs_y, frame_w, frame_h)
        now = time.time()

        on_button = box_contains(REPOSITIONING_BUTTON_BOX, gx, gy)
        button_hover_start = (button_hover_start or now) if on_button else None
        button_progress = min((now - button_hover_start) / REPOSITIONING_HOVER_SECONDS, 1.0) if button_hover_start else 0.0
        if button_progress >= 1.0:
            return REPOSITION_REQUESTED, None, (base_frame, detections, labels), crs_x, crs_y

        frame = base_frame.copy()
        hovered = find_hovered_detection(detections, gx, gy)

        if hovered is None:
            hovered_box, hover_start = None, None
        else:
            box = hovered[2]
            if hovered_box is None or iou(box, hovered_box) < HOVER_IOU_MATCH:
                hover_start = now
            hovered_box = box
        hover_duration = (now - hover_start) if hover_start is not None else 0.0
        is_held = hovered is not None and hover_duration >= HOVER_HOLD_SECONDS

        if is_held:
            box = hovered[2]
            reachy_detection.draw_box(frame, box, labels[box], reachy_detection.COLOR_GREEN)
            _draw_bomi_cursor(frame, gx, gy)
            cv2.imshow(reachy_detection.CAM_WINDOW_NAME, frame)
            cv2.waitKey(1)
            return hovered[0], box, (base_frame, detections, labels), crs_x, crs_y

        hover_progress = min(hover_duration / HOVER_HOLD_SECONDS, 1.0) if hovered is not None else 0.0
        for class_name, conf, box in detections:
            is_hovered = hovered is not None and box == hovered[2]
            color = COLOR_YELLOW if is_hovered else COLOR_BLUE
            reachy_detection.draw_box(frame, box, labels[box], color, hover_progress if is_hovered else 0.0)

        draw_repositioning_button(frame, button_progress)
        _draw_bomi_cursor(frame, gx, gy)
        cv2.imshow(reachy_detection.CAM_WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if safety.quit_requested(key, reachy_detection.CAM_WINDOW_NAME):
            return None, None, (base_frame, detections, labels), crs_x, crs_y


def select_place_location_bomi(cap, landmarker, bomi_map, cursor_filter, crs_x, crs_y, depth_cam):
    """Dwell-to-pick-a-point loop for placing an already-grasped object (e.g.
    "put it in the box"): unlike select_object_to_grasp_bomi, the target
    isn't a YOLO box, so the tolerance region is a fixed-radius circle
    (_draw_place_target) that recenters itself on the cursor (PLACE_ANCHOR_SMOOTHING)
    while the cursor stays inside it, and resets on a bigger, intentional move.

    Confirmed by dwelling inside that circle for PLACE_HOVER_SECONDS. The
    returned point is the median of every valid depth reading taken at the
    anchor while dwelling, for the same per-pixel noise robustness
    reachy_detection.build_object_point_cloud gets from fusing several depth
    frames.

    Returns (target point [xyz, m, Reachy world frame] or None if the user
    quit, crs_x, crs_y)."""
    anchor = None
    hover_start = None
    position_samples: List[np.ndarray] = []

    print(f"\n=== PLACE LOCATION ===  Q = quit  |  hold the cursor inside the "
          f"circle for {PLACE_HOVER_SECONDS:.0f}s to confirm where to place it")

    while True:
        _, crs_x, crs_y, _ = bomi_teleop.update_bomi_cursor(cap, landmarker, bomi_map, cursor_filter, crs_x, crs_y)

        capture = reachy_detection.capture_rgb_and_depth(depth_cam)
        if capture is None:
            key = cv2.waitKey(1) & 0xFF
            if safety.quit_requested(key, reachy_detection.CAM_WINDOW_NAME):
                return None, crs_x, crs_y
            continue
        base_frame, depth_frame = capture
        frame_h, frame_w = base_frame.shape[:2]

        gx, gy = bomi_teleop.map_bomi_to_frame(crs_x, crs_y, frame_w, frame_h)
        now = time.time()

        if anchor is None:
            anchor = (gx, gy)
            hover_start = now
            position_samples = []
        elif math.hypot(gx - anchor[0], gy - anchor[1]) > PLACE_TARGET_RADIUS_PX:
            anchor = (gx, gy)
            hover_start = now
            position_samples = []
        else:
            anchor = (
                anchor[0] + PLACE_ANCHOR_SMOOTHING * (gx - anchor[0]),
                anchor[1] + PLACE_ANCHOR_SMOOTHING * (gy - anchor[1]),
            )

        if depth_frame is not None:
            position = reachy_detection.estimate_position_at_pixel(
                depth_cam, depth_frame, int(anchor[0]), int(anchor[1]),
            )
            if position is not None:
                position_samples.append(position)

        hover_progress = min((now - hover_start) / PLACE_HOVER_SECONDS, 1.0)
        if hover_progress >= 1.0:
            if position_samples:
                return np.median(np.stack(position_samples), axis=0), crs_x, crs_y
            # Dwelled long enough but never got a valid depth reading at the
            # anchor (e.g. hovering off the table/box edge) -- keep trying
            # instead of returning a made-up point.
            print("[place] no valid depth at that point -- keep dwelling somewhere else")
            anchor, hover_start, position_samples = None, None, []

        frame = base_frame.copy()
        _draw_place_target(frame, int(anchor[0]), int(anchor[1]), hover_progress)
        _draw_bomi_cursor(frame, gx, gy)
        cv2.imshow(reachy_detection.CAM_WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if safety.quit_requested(key, reachy_detection.CAM_WINDOW_NAME):
            return None, crs_x, crs_y


def confirm_bomi(cap, landmarker, bomi_map, cursor_filter, crs_x, crs_y, lines: List[str], on_frame=None):
    """Generic Yes/No dwell dialog hovered with the BoMI cursor mapped into the
    confirm canvas, lines drawn top to bottom as the prompt.
    on_frame, if given, is called with no arguments once per loop iteration."""
    yes_hover_start = None
    no_hover_start = None

    while True:
        _, crs_x, crs_y, _ = bomi_teleop.update_bomi_cursor(cap, landmarker, bomi_map, cursor_filter, crs_x, crs_y)
        if on_frame is not None:
            on_frame()

        gx, gy = bomi_teleop.map_bomi_to_frame(crs_x, crs_y, CONFIRM_CANVAS_WIDTH, CONFIRM_CANVAS_HEIGHT)
        now = time.time()
        on_yes = box_contains(YES_BUTTON_BOX, gx, gy)
        on_no = box_contains(NO_BUTTON_BOX, gx, gy)

        yes_hover_start = (yes_hover_start or now) if on_yes else None
        no_hover_start = (no_hover_start or now) if on_no else None
        yes_progress = min((now - yes_hover_start) / CONFIRM_HOVER_SECONDS, 1.0) if yes_hover_start else 0.0
        no_progress = min((now - no_hover_start) / CONFIRM_HOVER_SECONDS, 1.0) if no_hover_start else 0.0

        canvas = draw_confirm_canvas(lines, yes_progress, no_progress)
        _draw_bomi_cursor(canvas, gx, gy)
        cv2.imshow(CONFIRM_WINDOW_NAME, canvas)
        cv2.setWindowProperty(CONFIRM_WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)  # re-pin (same-process windows only)
        safety.raise_window(CONFIRM_WINDOW_NAME)  # actually wins over the cross-process fullscreen camera_viewer window

        key = cv2.waitKey(1) & 0xFF
        result = None
        quit_now = safety.quit_requested(key, CONFIRM_WINDOW_NAME)
        if yes_progress >= 1.0:
            result = True
        elif no_progress >= 1.0:
            result = False

        if quit_now or result is not None:
            cv2.destroyWindow(CONFIRM_WINDOW_NAME)
            return (None if quit_now else result), crs_x, crs_y


def confirm_grasp_bomi(cap, landmarker, bomi_map, cursor_filter, crs_x, crs_y, class_name):
    """Yes/No dwell dialog confirming the selected object, before grasping."""
    return confirm_bomi(
        cap, landmarker, bomi_map, cursor_filter, crs_x, crs_y,
        lines=[f"You have select the {class_name} to be grasped.", "Do you want to confirm?"],
    )


def confirm_place_bomi(cap, landmarker, bomi_map, cursor_filter, crs_x, crs_y):
    """Yes/No dwell dialog confirming the dwell-picked placement point, before
    the grasp is actually executed."""
    return confirm_bomi(
        cap, landmarker, bomi_map, cursor_filter, crs_x, crs_y,
        lines=["You have selected where to place the object.", "Do you want to confirm?"],
    )


# --- Used only for tests, when the BoMI cursor is replaced by the mouse --- 
class MouseTracker:
    """Tracks the latest mouse position over the given OpenCV window --
    a standalone alternative to the BoMI cursor for driving the hover
    geometry above from a real mouse (see tests/*.py)."""

    def __init__(self, window_name: str) -> None:
        self.x = -1
        self.y = -1
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self._on_mouse)

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        self.x, self.y = x, y
