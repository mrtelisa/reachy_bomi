#!/usr/bin/env python3
"""
BoMI grasp for Reachy2: turns the robot on, captures a single frame from the
torso camera at launch, runs YOLOv8 on it to spot graspable objects, and
(when depth is available at the object's center pixel) estimates its 3D
position in Reachy's coordinate system, over reachy2_sdk (gRPC/IP).

Dependencies:
    pip install reachy2-sdk opencv-python numpy ultralytics

Usage:
    python3 bomi_grasp.py [robot_ip]

    <robot_ip> is optional; if omitted, DEFAULT_ROBOT_IP (set in this file) is used.

    Options:
        --yolo-model PATH   YOLOv8 weights (.pt). Default: yolov8n.pt (COCO
                             pretrained, auto-downloaded by ultralytics on
                             first run).
        --conf FLOAT        Minimum detection confidence. Default: 0.5

    Bounding boxes are blue by default. Hover the mouse over one to turn it
    yellow. Keep hovering the same object for HOVER_HOLD_SECONDS straight and
    it turns green, hiding the other boxes.

    Q, ESC, or closing the window with the X = quit.
"""

import argparse
import sys
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
from reachy2_sdk import ReachySDK
from reachy2_sdk.media.camera import CameraView, DepthCamera
from ultralytics import YOLO

# Placeholder — replace with the robot's actual IP.
DEFAULT_ROBOT_IP = "192.168.0.120"

CAM_WINDOW_NAME = "BoMI - Depth Camera (RGB)"

YOLO_MODEL_PATH = "yolov8n.pt"
YOLO_CONFIDENCE = 0.5

# Curated subset of COCO classes small/light enough for Reachy's gripper.
GRASPABLE_CLASSES = {
    "bottle", "cup", "wine glass", "bowl", "banana", "apple", "orange",
    "sandwich", "cell phone", "remote", "scissors", "book", "spoon",
    "fork", "knife", "teddy bear", "toothbrush", "mouse", "cake",
    "donut", "carrot", "broccoli", "hair drier", "vase", "book"
}

Detection = Tuple[str, float, Tuple[int, int, int, int]]  # class_name, confidence, (x1, y1, x2, y2)
Box = Tuple[int, int, int, int]

# BGR colors
COLOR_BLUE = (255, 0, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_GREEN = (0, 255, 0)

HOVER_HOLD_SECONDS = 5.0
HOVER_IOU_MATCH = 0.3  # min overlap between frames to count as "still hovering the same object"

REFRESH_BUTTON_BOX: Box = (10, 10, 260, 90)
COLOR_BUTTON = (0, 0, 255)
COLOR_BUTTON_TEXT = (255, 255, 255)
REFRESH_HOVER_SECONDS = 5.0

CONFIRM_WINDOW_NAME = "BoMI - Confirm Grasp"
CONFIRM_CANVAS_WIDTH = 520
CONFIRM_CANVAS_HEIGHT = 260
CONFIRM_HOVER_SECONDS = 5.0
YES_BUTTON_BOX: Box = (60, 150, 240, 220)
NO_BUTTON_BOX: Box = (280, 150, 460, 220)
COLOR_YES = (0, 200, 0)
COLOR_NO = (0, 0, 255)


def _quit_requested(key: int, window_name: str) -> bool:
    """True if Q/ESC was pressed, or the window was closed with the X button."""
    if key in (ord('q'), ord('Q'), 27):
        return True
    try:
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return False


class _MouseTracker:
    """Tracks the latest mouse position over the given OpenCV window."""

    def __init__(self, window_name: str) -> None:
        self.x = -1
        self.y = -1
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self._on_mouse)

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        self.x, self.y = x, y


def _box_contains(box: Box, x: int, y: int) -> bool:
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def _iou(box_a: Box, box_b: Box) -> float:
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


def _find_hovered_detection(detections: List[Detection], x: int, y: int) -> Optional[Detection]:
    """Smallest-area detection whose box contains (x, y), or None if none does."""
    hovered = None
    hovered_area = None
    for detection in detections:
        _, _, box = detection
        if not _box_contains(box, x, y):
            continue
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        if hovered_area is None or area < hovered_area:
            hovered, hovered_area = detection, area
    return hovered


def _detect_graspable_objects(model: YOLO, frame: np.ndarray, confidence: float) -> List[Detection]:
    """Run YOLO on frame, keeping only detections in GRASPABLE_CLASSES above confidence."""
    results = model(frame, verbose=False)[0]
    detections = []
    for box in results.boxes:
        class_name = model.names[int(box.cls[0])]
        conf = float(box.conf[0])
        if class_name not in GRASPABLE_CLASSES or conf < confidence:
            continue
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        detections.append((class_name, conf, (x1, y1, x2, y2)))
    return detections


def _estimate_object_position(
    depth_cam: DepthCamera, depth_frame: np.ndarray, u: int, v: int
) -> Optional[np.ndarray]:
    """XYZ position (meters, Reachy coordinate system) of the pixel (u, v), or None if depth is invalid there."""
    height, width = depth_frame.shape
    if not (0 <= v < height and 0 <= u < width):
        return None

    depth_mm = int(depth_frame[v, u])
    if depth_mm <= 0:
        return None

    return depth_cam.pixel_to_world(u, v, z_c=depth_mm / 1000.0, view=CameraView.LEFT)


def _label_for_detection(
    class_name: str, conf: float, box: Box,
    depth_cam: DepthCamera, depth_frame: Optional[np.ndarray],
) -> str:
    label = f"{class_name} {conf:.2f}"

    if depth_frame is not None:
        x1, y1, x2, y2 = box
        u, v = (x1 + x2) // 2, (y1 + y2) // 2
        position = _estimate_object_position(depth_cam, depth_frame, u, v)
        if position is not None:
            label += f"  xyz=({position[0]:.2f},{position[1]:.2f},{position[2]:.2f})m"

    return label


def _draw_box(
    frame: np.ndarray, box: Box, label: str, color: Tuple[int, int, int], progress: float = 0.0,
) -> None:
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    if progress > 0:
        bar_width = int((x2 - x1) * progress)
        cv2.rectangle(frame, (x1, y2 - 6), (x1 + bar_width, y2), color, -1)


def _draw_refresh_button(frame: np.ndarray, progress: float = 0.0) -> None:
    x1, y1, x2, y2 = REFRESH_BUTTON_BOX
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BUTTON, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BUTTON_TEXT, 1)

    text = "Refresh"
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    text_x = x1 + ((x2 - x1) - text_w) // 2
    text_y = y1 + ((y2 - y1) + text_h) // 2
    cv2.putText(frame, text, (text_x, text_y), font, scale, COLOR_BUTTON_TEXT, thickness)

    if progress > 0:
        bar_width = int((x2 - x1) * progress)
        cv2.rectangle(frame, (x1, y2 - 8), (x1 + bar_width, y2), COLOR_BUTTON_TEXT, -1)


def _draw_confirm_canvas(class_name: str, yes_progress: float, no_progress: float) -> np.ndarray:
    canvas = np.full((CONFIRM_CANVAS_HEIGHT, CONFIRM_CANVAS_WIDTH, 3), 30, dtype=np.uint8)

    cv2.putText(canvas, f"You have select the {class_name} to be grasped.",
                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_BUTTON_TEXT, 1)
    cv2.putText(canvas, "Do you want to confirm?",
                (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_BUTTON_TEXT, 2)

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


def _confirm_grasp(class_name: str) -> Optional[bool]:
    """Blocking Yes/No dialog. Returns True (Yes), False (No), or None if the user quit."""
    mouse = _MouseTracker(CONFIRM_WINDOW_NAME)
    yes_hover_start: Optional[float] = None
    no_hover_start: Optional[float] = None

    while True:
        now = time.time()
        on_yes = _box_contains(YES_BUTTON_BOX, mouse.x, mouse.y)
        on_no = _box_contains(NO_BUTTON_BOX, mouse.x, mouse.y)

        yes_hover_start = (yes_hover_start or now) if on_yes else None
        no_hover_start = (no_hover_start or now) if on_no else None

        yes_progress = min((now - yes_hover_start) / CONFIRM_HOVER_SECONDS, 1.0) if yes_hover_start else 0.0
        no_progress = min((now - no_hover_start) / CONFIRM_HOVER_SECONDS, 1.0) if no_hover_start else 0.0

        cv2.imshow(CONFIRM_WINDOW_NAME, _draw_confirm_canvas(class_name, yes_progress, no_progress))

        key = cv2.waitKey(1) & 0xFF
        result: Optional[bool] = None
        quit_now = _quit_requested(key, CONFIRM_WINDOW_NAME)
        if yes_progress >= 1.0:
            result = True
        elif no_progress >= 1.0:
            result = False

        if quit_now or result is not None:
            cv2.destroyWindow(CONFIRM_WINDOW_NAME)
            return None if quit_now else result


def _stream_torso_camera(depth_cam: DepthCamera) -> None:
    """Plain live RGB feed from the torso camera, no detection."""
    print("\n=== LIVE RGB STREAM (no detection) ===  Q = quit")
    while True:
        result = depth_cam.get_frame(view=CameraView.LEFT)
        if result is None:
            continue
        frame, _timestamp = result

        cv2.imshow(CAM_WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if _quit_requested(key, CAM_WINDOW_NAME):
            break


Capture = Tuple[np.ndarray, List[Detection], dict]


def _capture_and_detect(depth_cam: DepthCamera, model: YOLO, confidence: float) -> Optional[Capture]:
    """Grab one RGB + depth frame and run YOLO once, returning the frame, detections, and their labels."""
    result = depth_cam.get_frame(view=CameraView.LEFT)
    if result is None:
        print("[ERROR] Could not capture a frame from the camera")
        return None
    base_frame, _timestamp = result

    depth_result = depth_cam.get_depth_frame(view=CameraView.DEPTH)
    depth_frame = depth_result[0] if depth_result is not None else None

    detections = _detect_graspable_objects(model, base_frame, confidence)
    labels = {
        box: _label_for_detection(class_name, conf, box, depth_cam, depth_frame)
        for class_name, conf, box in detections
    }
    return base_frame, detections, labels


def _select_object_to_grasp(
    depth_cam: DepthCamera, model: YOLO, confidence: float, captured: Capture,
) -> Tuple[Optional[str], Capture]:
    """Detect-and-hover loop on an already-captured frame. Returns the class name once a box
    is held green for HOVER_HOLD_SECONDS (or None if the user quit), plus the possibly-refreshed
    (frame, detections, labels), so a later re-entry (e.g. after answering "No") can reuse them
    without recapturing."""
    base_frame, detections, labels = captured

    mouse = _MouseTracker(CAM_WINDOW_NAME)
    hovered_box: Optional[Box] = None
    hover_start: Optional[float] = None
    button_hover_start: Optional[float] = None

    print(f"\n=== CAPTURED FRAME (RGB + YOLO) ===  Q = quit  |  "
          f"hover Refresh for {REFRESH_HOVER_SECONDS:.0f}s to recapture")
    while True:
        now = time.time()
        on_button = _box_contains(REFRESH_BUTTON_BOX, mouse.x, mouse.y)
        if not on_button:
            button_hover_start = None
        elif button_hover_start is None:
            button_hover_start = now
        elif now - button_hover_start >= REFRESH_HOVER_SECONDS:
            refreshed = _capture_and_detect(depth_cam, model, confidence)
            if refreshed is not None:
                base_frame, detections, labels = refreshed
            hovered_box, hover_start = None, None
            button_hover_start = None

        frame = base_frame.copy()

        hovered = _find_hovered_detection(detections, mouse.x, mouse.y)

        if hovered is None:
            hovered_box, hover_start = None, None
        else:
            box = hovered[2]
            if hovered_box is None or _iou(box, hovered_box) < HOVER_IOU_MATCH:
                hover_start = now
            hovered_box = box
        hover_duration = (now - hover_start) if hover_start is not None else 0.0

        is_held = hovered is not None and hover_duration >= HOVER_HOLD_SECONDS

        if is_held:
            box = hovered[2]
            _draw_box(frame, box, labels[box], COLOR_GREEN)
            cv2.imshow(CAM_WINDOW_NAME, frame)
            cv2.waitKey(1)
            return hovered[0], (base_frame, detections, labels)

        hover_progress = min(hover_duration / HOVER_HOLD_SECONDS, 1.0) if hovered is not None else 0.0
        for class_name, conf, box in detections:
            is_hovered = hovered is not None and box == hovered[2]
            color = COLOR_YELLOW if is_hovered else COLOR_BLUE
            _draw_box(frame, box, labels[box], color, hover_progress if is_hovered else 0.0)

        button_progress = min((now - button_hover_start) / REFRESH_HOVER_SECONDS, 1.0) if button_hover_start else 0.0
        _draw_refresh_button(frame, button_progress)
        cv2.imshow(CAM_WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if _quit_requested(key, CAM_WINDOW_NAME):
            return None, (base_frame, detections, labels)


def _show_torso_camera(reachy: ReachySDK, model: YOLO, confidence: float) -> None:
    depth_cam = reachy.cameras.depth
    if depth_cam is None:
        print("[ERROR] No depth camera reported by the robot")
        return

    captured = _capture_and_detect(depth_cam, model, confidence)
    if captured is None:
        return

    while True:
        class_name, captured = _select_object_to_grasp(depth_cam, model, confidence, captured)
        if class_name is None:
            break

        decision = _confirm_grasp(class_name)
        if decision is None:
            break
        if decision:
            _stream_torso_camera(depth_cam)
            break
        # No -> back to the same captured frame/detections, all blue again

    cv2.destroyWindow(CAM_WINDOW_NAME)


def main() -> None:
    parser = argparse.ArgumentParser(description="BoMI grasp for Reachy2")
    parser.add_argument("robot_ip", nargs="?", default=DEFAULT_ROBOT_IP,
                        help=f"IP address of the Reachy robot (default: {DEFAULT_ROBOT_IP})")
    parser.add_argument("--yolo-model", default=YOLO_MODEL_PATH,
                        help=f"Path to YOLOv8 weights (default: {YOLO_MODEL_PATH})")
    parser.add_argument("--conf", type=float, default=YOLO_CONFIDENCE,
                        help=f"Minimum detection confidence (default: {YOLO_CONFIDENCE})")
    cli_args = parser.parse_args()

    reachy = ReachySDK(host=cli_args.robot_ip)
    if reachy.cameras is None:
        print(f"[ERROR] No camera service reported by the robot at '{cli_args.robot_ip}'")
        reachy.disconnect()
        sys.exit(1)

    reachy.turn_on()

    print(f"Loading YOLO model '{cli_args.yolo_model}'...")
    model = YOLO(cli_args.yolo_model)

    try:
        _show_torso_camera(reachy, model, cli_args.conf)
    finally:
        cv2.destroyAllWindows()
        reachy.disconnect()


if __name__ == "__main__":
    main()