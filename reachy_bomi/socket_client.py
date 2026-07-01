#!/usr/bin/env python3
"""
BoMI client for Reachy2 teleoperation.
Runs on the operator PC, NOT on the robot.

Dependencies:
    pip install mediapipe opencv-python scikit-learn numpy

Usage:
    # First time: run calibration and save it to the calib file
    python3 socket_client.py <server_ip> --calibrate

    # Next times (default): load the saved calibration, no calibration phase
    python3 socket_client.py <server_ip>

    Options:
        --calibrate            Run the calibration phase and save it. If omitted
                               (default), the saved calibration is loaded instead.
        --calib PATH           Calibration file. A bare filename is saved in the
                               calibrations/ folder inside the package.
                               Default: bomi_calib.npz
        --port PORT            Robot socket port. Default: 5051
        --cam INDEX            Webcam index. Default: 0

Phase 1 - Calibration (only with --calibrate):
    Move your hand through all positions you intend to use.
    SPACE = record sample   |   ENTER = finish (min 30 samples required)

Phase 2 - Control:
    Hand movement -> PCA cursor -> 9-region velocity -> TCP socket to robot.
    Q = quit and stop robot.
"""

import argparse
import os
import socket
import sys
import time

import cv2
import mediapipe as mp
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- Virtual screen dimensions (must match socket_server expectations) ---
BASE_WIDTH = 2550
BASE_HEIGHT = 1500

MAX_LINEAR = 1.0      # m/s
MAX_ANGULAR = 0.8     # rad/s
DEAD_ZONE_PX = 200    # pixel radius around screen center before motion starts

FORMAT = "utf-8"
DISCONNECT_MESSAGE = "!DISCONNECT"

# Calibration files live in a 'calibrations/' folder next to this script
# (i.e. inside the reachy_bomi package). A bare --calib filename is placed there;
# a --calib value that already contains a path is used as-is.
CALIB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calibrations")
DEFAULT_CALIB_FILE = "bomi_calib.npz"


def _resolve_calib_path(calib_arg: str) -> str:
    """Bare filename -> calibrations/ folder; explicit path -> used as given."""
    if os.path.dirname(calib_arg):
        return calib_arg
    return os.path.join(CALIB_DIR, calib_arg)


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
def _extract_hand_features(hand_landmarks) -> np.ndarray:
    """Flatten all 21 hand landmarks (x, y) into a 42-element vector."""
    return np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark]).flatten()


class BoMIMap:
    """
    PCA forward map: raw hand landmarks -> 2D cursor in screen space.

    Calibration fits StandardScaler + PCA(2 components) on collected samples,
    then computes a linear scale/offset so the PCA output spans the full screen.

    The fitted parameters (scaler mean/std, PCA components, scale, offset) are
    plain numpy arrays, so the map can be saved to / loaded from a .npz file and
    reused without repeating calibration.
    """

    def __init__(self) -> None:
        self._mean = None         # scaler mean (42,)
        self._std = None          # scaler std  (42,)
        self._components = None   # PCA components (2, 42)
        self._scale = np.ones(2)
        self._offset = np.zeros(2)
        self.fitted = False

    def fit(self, samples: list, margin: float = 100.0) -> None:
        X = np.array(samples)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=2)
        scores = pca.fit_transform(X_scaled)

        min_s = scores.min(axis=0)
        max_s = scores.max(axis=0)
        extent = np.where(max_s - min_s > 1e-6, max_s - min_s, 1.0)

        screen = np.array([BASE_WIDTH, BASE_HEIGHT], dtype=float)

        # Store the plain arrays needed for inference (decoupled from sklearn objects)
        self._mean = scaler.mean_
        self._std = scaler.scale_
        self._components = pca.components_
        self._scale = (screen - 2 * margin) / extent
        self._offset = margin - min_s * self._scale
        self.fitted = True

    def transform(self, features: np.ndarray) -> tuple:
        body = (features - self._mean) / self._std
        cu = np.dot(body, self._components.T)
        cu = cu * self._scale + self._offset
        crs_x = float(np.clip(cu[0], 0, BASE_WIDTH))
        crs_y = float(np.clip(cu[1], 0, BASE_HEIGHT))
        return crs_x, crs_y

    def save(self, path: str) -> None:
        if not self.fitted:
            raise RuntimeError("Cannot save an unfitted BoMIMap.")
        np.savez(
            path,
            mean=self._mean,
            std=self._std,
            components=self._components,
            scale=self._scale,
            offset=self._offset,
        )

    def load(self, path: str) -> None:
        data = np.load(path)
        self._mean = data["mean"]
        self._std = data["std"]
        self._components = data["components"]
        self._scale = data["scale"]
        self._offset = data["offset"]
        self.fitted = True


# --- Socket ---
class RobotSocket:
    def __init__(self, host: str, port: int) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.connect((host, port))
        print(f"[SOCKET] Connected to {host}:{port}")

    def send(self, msg: str) -> None:
        self._sock.sendall(msg.encode(FORMAT))

    def close(self) -> None:
        try:
            self.send(DISCONNECT_MESSAGE)
        except OSError:
            pass
        self._sock.close()


# --- Phases ---
def _calibration_phase(cap, hands) -> list:
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
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.multi_hand_landmarks[0],
                mp.solutions.hands.HAND_CONNECTIONS,
            )

        label = f"Samples: {len(samples)}/{MIN_SAMPLES}  SPACE=add  ENTER=done  Q=quit"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.imshow("BoMI - Calibration", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' ') and results.multi_hand_landmarks:
            samples.append(_extract_hand_features(results.multi_hand_landmarks[0]))
            print(f"  Sample {len(samples)} recorded")

        elif key == 13:  # ENTER
            if len(samples) >= MIN_SAMPLES:
                print(f"  Calibration done ({len(samples)} samples)")
                break
            else:
                print(f"  Need at least {MIN_SAMPLES} samples (have {len(samples)})")

        elif key == ord('q'):
            print("Aborted.")
            sys.exit(0)

    cv2.destroyWindow("BoMI - Calibration")
    return samples


def _control_phase(cap, hands, bomi_map: BoMIMap, robot: RobotSocket) -> None:
    SEND_HZ = 20
    dt = 1.0 / SEND_HZ
    last_send = time.time()

    print("\n=== CONTROL ===  Q = quit")
    robot.send("nine region")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        lin_vel, ang_vel = 0.0, 0.0

        if results.multi_hand_landmarks:
            hl = results.multi_hand_landmarks[0]
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hl, mp.solutions.hands.HAND_CONNECTIONS
            )
            crs_x, crs_y = bomi_map.transform(_extract_hand_features(hl))
            region = check_region_cursor(crs_x, crs_y)
            lin_vel, ang_vel = compute_dynamic_vel_from_cursor(crs_x, crs_y)
            lin_vel, ang_vel = apply_region_velocity_mask(region, lin_vel, ang_vel)

            cv2.putText(
                frame,
                f"region={region}  lin={lin_vel:.2f}  ang={ang_vel:.2f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )

        now = time.time()
        if now - last_send >= dt:
            robot.send(f"lin_vel:{lin_vel:.3f} ang_vel:{ang_vel:.3f}")
            last_send = now

        cv2.imshow("BoMI - Control", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    robot.send("lin_vel:0.000 ang_vel:0.000")
    cv2.destroyWindow("BoMI - Control")


# --- Entry point ---
def main() -> None:
    parser = argparse.ArgumentParser(description="BoMI client for Reachy2")
    parser.add_argument("server_ip", help="IP address of the Reachy robot")
    parser.add_argument("--port", type=int, default=5051)
    parser.add_argument("--cam", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--calibrate", action="store_true",
                        help="Run calibration and save it. Default: load saved calibration.")
    parser.add_argument("--calib", default=DEFAULT_CALIB_FILE,
                        help="Calibration file. A bare filename is stored in the "
                             f"calibrations/ folder inside the package (default: {DEFAULT_CALIB_FILE}).")
    args = parser.parse_args()

    calib_path = _resolve_calib_path(args.calib)

    # Fail early if we are supposed to load but there is no calibration file
    if not args.calibrate and not os.path.exists(calib_path):
        print(f"[ERROR] No calibration file '{calib_path}' found.")
        print("        Run once with --calibrate to create it, e.g.:")
        print(f"        python3 socket_client.py {args.server_ip} --calibrate")
        sys.exit(1)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {args.cam}")
        sys.exit(1)

    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
    )

    robot = RobotSocket(args.server_ip, args.port)
    try:
        bomi_map = BoMIMap()
        if args.calibrate:
            samples = _calibration_phase(cap, hands)
            bomi_map.fit(samples)
            os.makedirs(os.path.dirname(calib_path) or ".", exist_ok=True)
            bomi_map.save(calib_path)
            print(f"PCA map fitted and saved to {calib_path}")
        else:
            bomi_map.load(calib_path)
            print(f"Loaded calibration from {calib_path} (no calibration phase)")

        _control_phase(cap, hands, bomi_map, robot)
    finally:
        robot.close()
        cap.release()
        cv2.destroyAllWindows()
        hands.close()


if __name__ == "__main__":
    main()