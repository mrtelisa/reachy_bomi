#!/usr/bin/env python3
"""
BoMI teleop for Reachy2: hand tracking directly to the mobile base, over
reachy2_sdk (gRPC/IP) — no ROS 2 required on this machine, so its ROS 2
distro (if any) doesn't need to match the robot's.

Runs entirely on a single PC with a webcam and network access to the robot;
mediapipe/opencv computation and the reachy2_sdk client live in the same
process.

Dependencies:
    pip install reachy2-sdk mediapipe opencv-python scikit-learn numpy scipy

Usage:
    # Every run starts with calibration, then goes straight into control.
    python3 bomi_teleop.py [robot_ip]

    <robot_ip> is optional; if omitted, DEFAULT_ROBOT_IP (set in this file) is used.

    Options:
        --model PATH           Path to the MediaPipe hand_landmarker.task model.
                               Default: hand_landmarker.task inside the package.
        --cam INDEX            Webcam index. Default: 0

Phase 1 - Calibration (always runs first):
    Move your hand through all positions you intend to use.
    SPACE = record sample   |   ENTER = finish (min 30 samples required)

Phase 2 - Cursor preview (no robot motion):
    Same cursor/region view as Control, but nothing is sent to the robot.
    Use this to get a feel for the cursor before it starts driving the base.
    Opens the same cam/map windows used by Control; they stay open across the
    ENTER press below rather than closing and reopening.
    ENTER = proceed to Control   |   Q, ESC, or closing a window = quit

Phase 3 - Control:
    Hand movement -> PCA cursor -> 9-region velocity -> mobile base.
    Opens two windows: the webcam feed with landmarks, and a map of the
    virtual screen with the 9-region grid lines and a dot at the current
    cursor position.
    Q, ESC, or closing a window with the X = quit and stop the robot.
"""

import argparse
import math
import os
import sys
import time

import cv2
import mediapipe as mp
import numpy as np
import scipy.signal as sgn
from mediapipe.tasks.python.core import base_options
from mediapipe.tasks.python.vision import hand_landmarker
from mediapipe.tasks.python.vision.core import vision_task_running_mode
from reachy2_sdk import ReachySDK
from sklearn.decomposition import PCA

HAND_CONNECTIONS = hand_landmarker.HandLandmarksConnections.HAND_CONNECTIONS

# --- Lidar safety ---
LIDAR_SLOWDOWN_DISTANCE = 0.7   # m
LIDAR_CRITICAL_DISTANCE = 0.55  # m

# --- Virtual screen dimensions ---
BASE_WIDTH = 2550
BASE_HEIGHT = 1500

MAX_LINEAR = 0.6     # m/s
MAX_ANGULAR = 0.8     # rad/s
DEAD_ZONE_PX = 200    # pixel radius around screen center before motion starts

PUBLISH_HZ = 20  # speed-command rate (Hz) — comfortably under the mobile base's 0.2s command duration

# Cursor low-pass filter: 3rd-order
# Butterworth, coefficients derived from the actual control loop rate below
# rather than hardcoded for 50Hz.
CURSOR_FILTER_HZ = 30.0        # assumed webcam/control loop sample rate
CURSOR_FILTER_CUTOFF_HZ = 4.0  # cutoff frequency

# MediaPipe Tasks hand-landmarker model (.task), lives at the package root by default.
DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "hand_landmarker.task"
)

# Placeholder — replace with the robot's actual IP.
DEFAULT_ROBOT_IP = "192.168.0.120"

# Shared window names for the cursor preview and control phases, so the same
# OS windows stay open across the transition instead of closing and reopening.
CAM_WINDOW_NAME = "BoMI - Camera"
MAP_WINDOW_NAME = "BoMI - Cursor Map"


# --- Velocity helpers (adapted from reaching_functions.py) ---------

def check_region_cursor(crs_x: float, crs_y: float) -> int:
    """
    Returns region 1-9 based on 3x3 grid over BASE_WIDTH x BASE_HEIGHT.
    Layout:
        1 | 2 | 3   (top row)
        4 | 5 | 6   (middle row)
        7 | 8 | 9   (bottom row)
    """
    if crs_x < 847:
        col = 0
    elif crs_x <= 1697:
        col = 1
    else:
        col = 2

    if crs_y < 497:
        row = 0
    elif crs_y <= 997:
        row = 1
    else:
        row = 2

    return row * 3 + col + 1


def compute_dynamic_vel_from_cursor(
    crs_x: float,
    crs_y: float,
    max_linear: float = MAX_LINEAR,
    max_angular: float = MAX_ANGULAR,
    dead_zone_px: float = DEAD_ZONE_PX,
    ang_right_is_negative: bool = True,
) -> tuple:
    """
    Continuous linear/angular velocity from cursor position.
    Cursor at screen center -> zero velocity (dead zone).
    Up from center -> positive linear; right from center -> negative angular.
    """
    cx = BASE_WIDTH / 2.0
    cy = BASE_HEIGHT / 2.0
    dx = crs_x - cx
    dy = crs_y - cy

    if np.hypot(dx, dy) < dead_zone_px:
        return 0.0, 0.0

    x_norm = float(np.clip(dx / cx, -1.0, 1.0))
    y_norm = float(np.clip(-dy / cy, -1.0, 1.0))  # up = positive

    if abs(x_norm) < dead_zone_px / cx:
        x_norm = 0.0
    if abs(y_norm) < dead_zone_px / cy:
        y_norm = 0.0

    lin_vel = max_linear * y_norm
    ang_sign = -1.0 if ang_right_is_negative else 1.0
    ang_vel = ang_sign * max_angular * x_norm
    return lin_vel, ang_vel


def apply_region_velocity_mask(region: int, lin_vel: float, ang_vel: float) -> tuple:
    """
    Enforce active DOFs per region:
      center (5)        -> stop
      middle col (2, 8) -> linear only
      middle row (4, 6) -> angular only
      corners (1,3,7,9) -> both
    """
    if region == 5:
        return 0.0, 0.0
    if region in (2, 8):
        ang_vel = 0.0
    if region in (4, 6):
        lin_vel = 0.0
    return lin_vel, ang_vel


# --- PCA forward map ---
def _extract_hand_features(hand_landmarks, mirror_x: bool = False) -> np.ndarray:
    """
    Flatten all 21 hand landmarks (x, y) into a 42-element vector.
    If mirror_x, x is mirrored (1 - x) so the right hand maps to the same
    feature space as the left hand.

    hand_landmarks is the list of NormalizedLandmark returned by MediaPipe
    Tasks (e.g. results.hand_landmarks[0]).
    """
    coords = [[1.0 - lm.x if mirror_x else lm.x, lm.y] for lm in hand_landmarks]
    return np.array(coords).flatten()


def _draw_hand_landmarks(frame, landmarks) -> None:
    """Draw MediaPipe Tasks hand landmarks/connections on a BGR OpenCV frame."""
    height, width = frame.shape[:2]
    points = []

    for landmark in landmarks:
        x = min(max(int(landmark.x * width), 0), width - 1)
        y = min(max(int(landmark.y * height), 0), height - 1)
        points.append((x, y))

    for connection in HAND_CONNECTIONS:
        cv2.line(frame, points[connection.start], points[connection.end], (0, 200, 255), 2)

    for point in points:
        cv2.circle(frame, point, 4, (0, 255, 0), -1)


def _quit_requested(key: int, window_name: str) -> bool:
    """True if Q/ESC was pressed, or the window was closed with the X button."""
    if key in (ord('q'), ord('Q'), 27):
        return True
    try:
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return False


def _draw_cursor_map(crs_x: float, crs_y: float, region: int, message: str,
                      map_width: int = 850, map_height: int = 500):
    """Rectangle representing the BASE_WIDTH x BASE_HEIGHT virtual screen, with
    the 9-region grid lines, a dot at the current cursor position, and the
    lin_vel/ang_vel message currently being sent to the mobile base."""
    canvas = np.full((map_height, map_width, 3), 30, dtype=np.uint8)
    sx = map_width / BASE_WIDTH
    sy = map_height / BASE_HEIGHT

    x1, x2 = int(847 * sx), int(1697 * sx)
    y1, y2 = int(497 * sy), int(997 * sy)
    for x in (x1, x2):
        cv2.line(canvas, (x, 0), (x, map_height), (90, 90, 90), 1)
    for y in (y1, y2):
        cv2.line(canvas, (0, y), (map_width, y), (90, 90, 90), 1)

    cx, cy = int(crs_x * sx), int(crs_y * sy)
    cv2.circle(canvas, (cx, cy), 10, (0, 0, 255), -1)

    cv2.putText(canvas, f"region={region}  cursor=({crs_x:.0f},{crs_y:.0f})",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(canvas, f"-> mobile base: {message}",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return canvas


class CursorFilter:
    """
    3rd-order Butterworth low-pass filter for the (crs_x, crs_y) cursor position.
    Coefficients are derived from the actual sample rate (CURSOR_FILTER_HZ) instead
    of being hardcoded for a fixed frequency loop.
    """

    ORDER = 3

    def __init__(self, sample_hz: float = CURSOR_FILTER_HZ, cutoff_hz: float = CURSOR_FILTER_CUTOFF_HZ) -> None:
        nyquist = sample_hz / 2.0
        self._b, self._a = sgn.butter(self.ORDER, cutoff_hz / nyquist, btype="low")
        self._in_history = np.zeros((self.ORDER, 2))
        self._out_history = np.zeros((self.ORDER, 2))

    def update(self, crs_x: float, crs_y: float) -> tuple:
        new_input = np.array([crs_x, crs_y])
        new_output = self._b[0] * new_input
        for i in range(self.ORDER):
            new_output += self._b[i + 1] * self._in_history[i]
        for i in range(self.ORDER):
            new_output -= self._a[i + 1] * self._out_history[i]

        self._in_history = np.roll(self._in_history, 1, axis=0)
        self._in_history[0] = new_input
        self._out_history = np.roll(self._out_history, 1, axis=0)
        self._out_history[0] = new_output

        return float(new_output[0]), float(new_output[1])


class BoMIMap:
    """
    PCA forward map: raw hand landmarks -> 2D cursor in screen space.

    PCA(2 components) fitted directly on the raw landmark samples. Scale/offset
    map the calibration scores' peak-to-peak range onto the screen size, centered
    on the mean.
    """

    def __init__(self) -> None:
        self._mean = None         # PCA mean (42,)
        self._components = None   # PCA components (2, 42)
        self._scale = np.ones(2)
        self._offset = np.zeros(2)
        self.fitted = False

    def fit(self, samples: list) -> None:
        X = np.array(samples)

        pca = PCA(n_components=2)
        scores = pca.fit_transform(X)

        extent = np.ptp(scores, axis=0)
        extent = np.where(extent > 1e-6, extent, 1.0)

        screen = np.array([BASE_WIDTH, BASE_HEIGHT], dtype=float)

        # Store the plain arrays needed for inference (decoupled from the sklearn object)
        self._mean = pca.mean_
        self._components = pca.components_
        self._scale = screen / extent
        self._offset = screen / 2.0 - (scores * self._scale).mean(axis=0)
        self.fitted = True

    def transform(self, features: np.ndarray) -> tuple:
        """
        Linear BoMI map: raw hand landmarks -> 2D cursor in screen space.
        Returns (crs_x, crs_y) in pixels, clipped to the screen size
        """
        cu = np.dot(features - self._mean, self._components.T) # Linear projection of the features onto the PCA components
        cu = cu * self._scale + self._offset # Scaling operation to map the PCA scores to the screen size
        crs_x = float(np.clip(cu[0], 0, BASE_WIDTH))
        crs_y = float(np.clip(cu[1], 0, BASE_HEIGHT))
        return crs_x, crs_y


# --- Phases ---
def _calibration_phase(cap, landmarker) -> list:
    MIN_SAMPLES = 30
    samples = []

    print("\n=== CALIBRATION ===")
    print("Move your hand through all positions you intend to use.")
    print("SPACE = record sample   |   ENTER = finish (need >= 30)   |   Q = quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = landmarker.detect_for_video(mp_image, int(time.time() * 1000))

        if results.hand_landmarks:
            _draw_hand_landmarks(frame, results.hand_landmarks[0])

        label = f"Samples: {len(samples)}/{MIN_SAMPLES}  SPACE=add  ENTER=done  Q=quit"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        window_name = "BoMI - Calibration"
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' ') and results.hand_landmarks:
            mirror_x = results.handedness[0][0].category_name == "Right"
            samples.append(_extract_hand_features(results.hand_landmarks[0], mirror_x))
            print(f"  Sample {len(samples)} recorded")

        elif key == 13:  # ENTER
            if len(samples) >= MIN_SAMPLES:
                print(f"  Calibration done ({len(samples)} samples)")
                break
            else:
                print(f"  Need at least {MIN_SAMPLES} samples (have {len(samples)})")

        elif _quit_requested(key, window_name):
            print("Aborted.")
            sys.exit(0)

    cv2.destroyWindow(window_name)
    return samples

def _cursor_preview_phase(cap, landmarker, bomi_map: BoMIMap) -> None:
    """
    Shows the same cursor/region view as the control phase, but never talks to
    the robot. Lets the user get a feel for the cursor and see where it starts
    out before enabling motion.
    """
    cursor_filter = CursorFilter()
    cam_window = CAM_WINDOW_NAME
    map_window = MAP_WINDOW_NAME

    crs_x, crs_y = BASE_WIDTH / 2.0, BASE_HEIGHT / 2.0
    region = check_region_cursor(crs_x, crs_y)

    print("\n=== CURSOR PREVIEW (robot not moving) ===")
    print("Get a feel for the cursor. ENTER = start Control   |   Q = quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = landmarker.detect_for_video(mp_image, int(time.time() * 1000))

        if results.hand_landmarks:
            hl = results.hand_landmarks[0]
            _draw_hand_landmarks(frame, hl)
            mirror_x = results.handedness[0][0].category_name == "Right"
            crs_x, crs_y = bomi_map.transform(_extract_hand_features(hl, mirror_x))
            crs_x, crs_y = cursor_filter.update(crs_x, crs_y)
            region = check_region_cursor(crs_x, crs_y)

        cv2.putText(
            frame, f"region={region}  cursor=({crs_x:.0f},{crs_y:.0f})",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
        )
        cv2.putText(
            frame, "PREVIEW - robot not moving. ENTER=start control  Q=quit",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
        )

        cv2.imshow(cam_window, frame)
        cv2.imshow(map_window, _draw_cursor_map(crs_x, crs_y, region, "(preview - not sent)"))

        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # ENTER
            break
        if _quit_requested(key, cam_window) or _quit_requested(key, map_window):
            print("Aborted.")
            sys.exit(0)

    # Windows are intentionally left open (no destroyWindow) so the same cam/map
    # windows carry straight into the control phase instead of flickering shut.


def _control_phase(cap, landmarker, bomi_map: BoMIMap, mobile_base) -> None:
    dt = 1.0 / PUBLISH_HZ
    last_publish = time.time()
    cursor_filter = CursorFilter()
    cam_window = CAM_WINDOW_NAME
    map_window = MAP_WINDOW_NAME

    # Start centered (region 5) until the first hand detection updates it.
    crs_x, crs_y = BASE_WIDTH / 2.0, BASE_HEIGHT / 2.0
    region = check_region_cursor(crs_x, crs_y)
    message = "lin_vel:0.000 ang_vel:0.000"

    print("\n=== CONTROL ===  Q = quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = landmarker.detect_for_video(mp_image, int(time.time() * 1000))

        lin_vel, ang_vel = 0.0, 0.0

        if results.hand_landmarks:
            hl = results.hand_landmarks[0]
            _draw_hand_landmarks(frame, hl)
            mirror_x = results.handedness[0][0].category_name == "Right"
            crs_x, crs_y = bomi_map.transform(_extract_hand_features(hl, mirror_x))
            crs_x, crs_y = cursor_filter.update(crs_x, crs_y)
            region = check_region_cursor(crs_x, crs_y)
            lin_vel, ang_vel = compute_dynamic_vel_from_cursor(crs_x, crs_y)
            lin_vel, ang_vel = apply_region_velocity_mask(region, lin_vel, ang_vel)

            cv2.putText(
                frame, f"region={region}  cursor=({crs_x:.0f},{crs_y:.0f})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )
            cv2.putText(
                frame, f"-> mobile base: {message}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
            )

        now = time.time()
        if now - last_publish >= dt:
            message = f"lin_vel:{lin_vel:.3f} ang_vel:{ang_vel:.3f}"
            # vtheta is in degrees/s for reachy2_sdk, ang_vel is computed in rad/s
            mobile_base.set_goal_speed(vx=lin_vel, vy=0, vtheta=math.degrees(ang_vel))
            mobile_base.send_speed_command()
            last_publish = now

        cv2.imshow(cam_window, frame)
        cv2.imshow(map_window, _draw_cursor_map(crs_x, crs_y, region, message))

        key = cv2.waitKey(1) & 0xFF
        if _quit_requested(key, cam_window) or _quit_requested(key, map_window):
            break

    mobile_base.set_goal_speed(vx=0, vy=0, vtheta=0)
    mobile_base.send_speed_command()
    cv2.destroyWindow(cam_window)
    cv2.destroyWindow(map_window)


# --- Entry point ---
def main() -> None:
    parser = argparse.ArgumentParser(description="BoMI teleop for Reachy2")
    parser.add_argument("robot_ip", nargs="?", default=DEFAULT_ROBOT_IP,
                        help=f"IP address of the Reachy robot (default: {DEFAULT_ROBOT_IP})")
    parser.add_argument("--cam", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH,
                        help="Path to the MediaPipe hand_landmarker.task model "
                             f"(default: {DEFAULT_MODEL_PATH}).")
    cli_args = parser.parse_args()

    # Fail early if the hand-landmarker model is missing
    if not os.path.exists(cli_args.model):
        print(f"[ERROR] MediaPipe model not found: '{cli_args.model}'")
        print("        Download hand_landmarker.task and pass its path with --model.")
        sys.exit(1)

    reachy = ReachySDK(host=cli_args.robot_ip)
    if reachy.mobile_base is None:
        print(f"[ERROR] No mobile base reported by the robot at '{cli_args.robot_ip}'")
        reachy.disconnect()
        sys.exit(1)
    mobile_base = reachy.mobile_base
    mobile_base.lidar.safety_enabled = True
    mobile_base.lidar.safety_slowdown_distance = LIDAR_SLOWDOWN_DISTANCE
    mobile_base.lidar.safety_critical_distance = LIDAR_CRITICAL_DISTANCE
    mobile_base.turn_on()

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

        bomi_map = BoMIMap()
        samples = _calibration_phase(cap, landmarker)
        bomi_map.fit(samples)
        print("PCA map fitted")

        _cursor_preview_phase(cap, landmarker, bomi_map)
        _control_phase(cap, landmarker, bomi_map, mobile_base)
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
