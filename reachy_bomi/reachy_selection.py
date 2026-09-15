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

# --- Free-point placement dwell, on a fixed reachability grid laid out on the
# table plane itself (e.g. "put it in the box") ---
PLACE_HOVER_SECONDS = DWELL_HOLD_SECONDS

# Real-world side length (m) of each grid cell, laid out on the table plane
# (not the screen) -- two cells look the same physical size to the robot
# whether they're near or far in the shot, unlike a screen-space grid where a
# far cell covers far more real table than a near one. Big enough to dwell on
# despite BoMI cursor tremor (comparable to a small YOLO detection box in
# screen terms, at a typical distance); small enough that the placement still
# lands close to where the user pointed. Tune this first if cells feel too
# coarse or too fiddly on the real robot.
PLACE_GRID_CELL_SIZE_M = 0.12

# Every PLACE_GRID_SAMPLE_STRIDE_PXth pixel (in each axis) is classified into
# a table-plane cell, instead of every pixel -- table cells are big enough
# (PLACE_GRID_CELL_SIZE_M) relative to any reasonable camera resolution that
# this loses no visible detail, while cutting the one-time setup cost
# (dominated by reachy_detection.estimate_world_points_for_frame) by
# stride**2.
PLACE_GRID_SAMPLE_STRIDE_PX = 10

# How many approach directions plan_place tries per cell while building the
# grid -- fewer than its default, since it runs once per cell (hundreds of
# times) rather than once. Lower this first if "Computing reachable area..."
# starts taking too long; raise it if too much of the table reads as red.
PLACE_GRID_CANDIDATE_COUNT = reachy_grasp.QUICK_REACHABILITY_CANDIDATE_COUNT

COLOR_UNREACHABLE = (0, 0, 200)   # BGR red, translucent fill over unreachable/unknown table area
UNREACHABLE_TINT_ALPHA = 0.35

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


def _table_plane_basis(table_normal: np.ndarray) -> tuple:
    """Orthonormal in-plane basis perpendicular to table_normal: basis_u is
    Reachy world X ("forward", per reachy_grasp.py's frame convention)
    projected onto the plane, basis_v completes a right-handed frame with
    normal. Falls back to world Y as the reference if the table is (bizarrely)
    near-vertical relative to X. Returns (normal, basis_u, basis_v), all unit
    vectors."""
    normal = table_normal / np.linalg.norm(table_normal)
    reference = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(reference, normal)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    basis_u = reference - normal * np.dot(reference, normal)
    basis_u /= np.linalg.norm(basis_u)
    basis_v = np.cross(normal, basis_u)
    return normal, basis_u, basis_v


def _draw_place_grid(
    frame: np.ndarray, unreachable_mask: np.ndarray,
    coarse_row_img: np.ndarray, coarse_col_img: np.ndarray, stride: int,
    hovered_cell: Optional[tuple], hover_progress: float, hovered_reachable: bool,
) -> None:
    """Tints unreachable/unknown table area red (translucent) -- unreachable_mask
    is full-resolution and precomputed once in _build_place_grid, since it
    doesn't change during the dwell. Then outlines hovered_cell's actual
    projected shape, found via cv2.findContours on the coarse table-plane
    classification image (so it follows the table's real perspective --
    smaller/more skewed the farther away it is -- instead of being a flat
    screen rectangle): yellow and filled with dwell progress if reachable,
    red (no fill) otherwise, so hovering unreachable table area still gives
    feedback without ever accumulating progress there."""
    red = np.full_like(frame, COLOR_UNREACHABLE)
    blended = cv2.addWeighted(red, UNREACHABLE_TINT_ALPHA, frame, 1 - UNREACHABLE_TINT_ALPHA, 0)
    frame[unreachable_mask] = blended[unreachable_mask]

    if hovered_cell is None:
        return
    row, col = hovered_cell
    mask = ((coarse_row_img == row) & (coarse_col_img == col)).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return
    contour = max(contours, key=cv2.contourArea) * stride
    color = COLOR_YELLOW if hovered_reachable else COLOR_UNREACHABLE
    cv2.drawContours(frame, [contour], -1, color, 3)
    if hovered_reachable and hover_progress > 0:
        x, y, w, h = cv2.boundingRect(contour)
        bar_width = int(w * hover_progress)
        cv2.rectangle(frame, (x, y + h - 6), (x + bar_width, y + h), color, -1)


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


def _arm_that_can_place_at(reachy, grasp_plans: dict, geometry, target_point: np.ndarray) -> Optional[str]:
    """Which of grasp_plans' arms could carry the object to target_point and
    set it down there, or None if neither can. The arm on the target's own
    side is tried first, so it's the one that wins when both could -- and so
    most cells settle on the first try, since the grid runs this once per
    cell."""
    preferred = "r_arm" if target_point[1] < 0 else "l_arm"
    for arm_name in sorted(grasp_plans, key=lambda name: name != preferred):
        place_plan = reachy_grasp.plan_place(
            reachy, grasp_plans[arm_name], geometry.table_normal, target_point,
            candidate_count=PLACE_GRID_CANDIDATE_COUNT,
        )
        if place_plan is not None:
            return arm_name
    return None


def _build_place_grid(depth_cam, depth_frame, frame_w, frame_h, reachy, grasp_plans, geometry):
    """Classifies (a stride-sampled subset of) the frame's pixels by which
    PLACE_GRID_CELL_SIZE_M cell of the table plane they land in -- a grid
    laid out in real-world meters on the table itself, not on the screen, so
    it looks like it's actually resting on the table (perspective-correct:
    smaller/more skewed toward the horizon) instead of a flat rectangular
    overlay. The table plane's origin is geometry.centroid (near where the
    object was picked up) if available, else world origin -- either way just
    a fixed reference for cell indexing, since only the components
    perpendicular to table_normal (basis_u/basis_v) affect it.

    For each cell that appears in the frame, computes its real-world center
    and asks reachy_grasp.plan_place whether ANY of grasp_plans' arms can put
    the object down there, trying the arm on the cell's own side first (so
    that's the one that wins when both can). A cell is reachable if at least
    one arm can, which makes the map the union of both arms' reach when the
    object can be picked up with either, and just that arm's reach when only
    one can pick it up.

    The plans themselves are only used to color the grid and to record which
    arm serves each cell; the caller re-derives the actual grasp and place
    plans fresh right before moving (see select_place_location_bomi), to
    minimize how many reachy.inverse_kinematics calls happen between
    validating them and acting on them -- reachy2_sdk's IK solver appears to
    seed its search from the last computed solution, so the many speculative
    IK checks this grid needs can otherwise leave it unable to re-solve a
    pose it validated only moments earlier.

    Returns (stride, coarse_row_img, coarse_col_img, cell_targets, cell_arms,
    unreachable_mask):
      - coarse_row_img/coarse_col_img: int32 arrays, shape
        (ceil(frame_h/stride), ceil(frame_w/stride)) -- each entry is the
        table-plane cell (row, col) the corresponding stride x stride screen
        block landed in, or -1 where there was no valid depth.
      - cell_targets: {(row, col): xyz point} for every cell that appeared.
      - cell_arms: {(row, col): arm_name or None} -- which arm would place
        there, None if neither can.
      - unreachable_mask: full-resolution (frame_h, frame_w) bool array,
        True wherever the table area is unknown (no depth) or unreachable --
        precomputed once here since it doesn't change during the dwell."""
    stride = PLACE_GRID_SAMPLE_STRIDE_PX
    table_normal = geometry.table_normal if geometry.table_normal is not None else reachy_grasp.DEFAULT_TABLE_NORMAL
    _, basis_u, basis_v = _table_plane_basis(table_normal)
    origin = geometry.centroid if geometry.centroid is not None else np.zeros(3)

    rows, cols, points = reachy_detection.estimate_world_points_for_frame(depth_cam, depth_frame, stride=stride)

    coarse_h, coarse_w = math.ceil(frame_h / stride), math.ceil(frame_w / stride)
    coarse_row_img = np.full((coarse_h, coarse_w), -1, dtype=np.int32)
    coarse_col_img = np.full((coarse_h, coarse_w), -1, dtype=np.int32)
    cell_targets: dict = {}
    cell_arms: dict = {}
    unreachable_mask = np.ones((frame_h, frame_w), dtype=bool)  # default: no data at all -> unknown/unreachable

    if points.shape[0] > 0:
        rel = points - origin
        cell_col = np.floor((rel @ basis_u) / PLACE_GRID_CELL_SIZE_M).astype(np.int32)
        cell_row = np.floor((rel @ basis_v) / PLACE_GRID_CELL_SIZE_M).astype(np.int32)
        coarse_row_img[rows // stride, cols // stride] = cell_row
        coarse_col_img[rows // stride, cols // stride] = cell_col

        for rr, cc in np.unique(np.stack([cell_row, cell_col], axis=1), axis=0):
            rr, cc = int(rr), int(cc)
            center = origin + (cc + 0.5) * PLACE_GRID_CELL_SIZE_M * basis_u + (rr + 0.5) * PLACE_GRID_CELL_SIZE_M * basis_v
            cell_targets[(rr, cc)] = center
            cell_arms[(rr, cc)] = _arm_that_can_place_at(reachy, grasp_plans, geometry, center)

        coarse_unreachable = coarse_row_img == -1
        valid = ~coarse_unreachable
        if np.any(valid):
            reachable_flags = np.fromiter(
                (cell_arms[(int(r), int(c))] is not None
                 for r, c in zip(coarse_row_img[valid], coarse_col_img[valid])),
                dtype=bool, count=int(np.count_nonzero(valid)),
            )
            coarse_unreachable[valid] = ~reachable_flags
        unreachable_mask = cv2.resize(
            coarse_unreachable.astype(np.uint8), (frame_w, frame_h), interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    return stride, coarse_row_img, coarse_col_img, cell_targets, cell_arms, unreachable_mask


def select_place_location_bomi(
    cap, landmarker, bomi_map, cursor_filter, crs_x, crs_y, depth_cam, reachy, grasp_plans, geometry,
):
    """Dwell-to-pick-a-cell loop for placing an already-grasped object (e.g.
    "put it in the box"), the cells laid out on the table plane itself
    (_build_place_grid) rather than the screen. Captures a single torso
    RGB+depth frame up front -- same as select_object_to_grasp_bomi does for
    its YOLO detections -- then builds the grid and precomputes each cell's
    reachability once. Unlike a free-floating anchor, dwelling is then just
    "is the cursor still over the same table cell as last frame" -- cells
    don't move, so cursor tremor within one never resets the dwell, only
    actually crossing into a different cell does.

    Unreachable/unknown table area is tinted red and never accumulates dwell
    progress (hovering it behaves like hovering nothing), which steers the
    user away from it without a separate error dialog.

    grasp_plans is {arm_name: GraspPlan} for every arm that can pick the
    object up (reachy_grasp.plan_grasps_by_arm) -- the grid shows the union
    of their reach, and the confirmed cell is what decides which arm
    actually does the job.

    Confirmed by dwelling on the same reachable cell for PLACE_HOVER_SECONDS.
    Returns that cell's raw 3D target point and the arm that serves it, not
    the GraspPlan _build_place_grid computed -- the caller should re-derive
    the actual grasp and place plans right before moving, per
    _build_place_grid's docstring.

    Returns (target point [xyz, m, Reachy world frame], arm_name, crs_x,
    crs_y), or (None, None, crs_x, crs_y) if the user quit, the initial
    capture failed, or there's no depth frame to build a grid from."""
    capture = reachy_detection.capture_rgb_and_depth(depth_cam)
    if capture is None:
        print("[place] could not capture a frame from the depth camera")
        return None, None, crs_x, crs_y
    base_frame, depth_frame = capture
    frame_h, frame_w = base_frame.shape[:2]
    if depth_frame is None:
        print("[place] no depth frame available -- can't compute a placement grid")
        return None, None, crs_x, crs_y

    wait_frame = base_frame.copy()
    cv2.putText(wait_frame, "Computing reachable area...", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_BUTTON_TEXT, 2)
    cv2.imshow(reachy_detection.CAM_WINDOW_NAME, wait_frame)
    cv2.waitKey(1)
    stride, coarse_row_img, coarse_col_img, cell_targets, cell_arms, unreachable_mask = _build_place_grid(
        depth_cam, depth_frame, frame_w, frame_h, reachy, grasp_plans, geometry,
    )
    coarse_h, coarse_w = coarse_row_img.shape

    tracked_cell = None  # the reachable cell currently accumulating dwell, or None
    hover_start = None

    print(f"\n=== PLACE LOCATION ===  Q = quit  |  hold the cursor over a spot on the "
          f"table for {PLACE_HOVER_SECONDS:.0f}s to confirm where to place it (red area isn't reachable)")

    while True:
        _, crs_x, crs_y, _ = bomi_teleop.update_bomi_cursor(cap, landmarker, bomi_map, cursor_filter, crs_x, crs_y)

        gx, gy = bomi_teleop.map_bomi_to_frame(crs_x, crs_y, frame_w, frame_h)
        coarse_y = min(max(gy // stride, 0), coarse_h - 1)
        coarse_x = min(max(gx // stride, 0), coarse_w - 1)
        rr, cc = int(coarse_row_img[coarse_y, coarse_x]), int(coarse_col_img[coarse_y, coarse_x])
        current_cell = (rr, cc) if rr != -1 else None
        now = time.time()

        place_arm = cell_arms.get(current_cell) if current_cell is not None else None
        if place_arm is None:
            tracked_cell, hover_start = None, None
        elif tracked_cell != current_cell:
            tracked_cell, hover_start = current_cell, now
        hover_progress = (
            min((now - hover_start) / PLACE_HOVER_SECONDS, 1.0) if tracked_cell is not None else 0.0
        )

        if hover_progress >= 1.0:
            print(f"[place] spot confirmed -- {place_arm} will do it")
            return cell_targets[current_cell], place_arm, crs_x, crs_y

        frame = base_frame.copy()
        _draw_place_grid(
            frame, unreachable_mask, coarse_row_img, coarse_col_img, stride,
            current_cell, hover_progress, place_arm is not None,
        )
        _draw_bomi_cursor(frame, gx, gy)
        cv2.imshow(reachy_detection.CAM_WINDOW_NAME, frame)
        cv2.setWindowProperty(reachy_detection.CAM_WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)  # re-pin (same-process windows only)
        safety.raise_window(reachy_detection.CAM_WINDOW_NAME)  # wins over a cross-process fullscreen window too

        key = cv2.waitKey(1) & 0xFF
        if safety.quit_requested(key, reachy_detection.CAM_WINDOW_NAME):
            return None, None, crs_x, crs_y


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
            safety.destroy_window(CONFIRM_WINDOW_NAME)
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


def confirm_new_object_bomi(cap, landmarker, bomi_map, cursor_filter, crs_x, crs_y):
    """Yes/No dwell dialog asked right after a successful placement: Yes loops
    back into object selection on a fresh capture, No (or quitting) starts the
    end-of-session wind-down (back up, rotate, power off)."""
    return confirm_bomi(
        cap, landmarker, bomi_map, cursor_filter, crs_x, crs_y,
        lines=["Object placed.", "Do you want to select a new object?"],
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
