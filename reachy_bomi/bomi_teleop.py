#!/usr/bin/env python3
"""
Library module -- BoMI teleop building blocks for Reachy2: hand tracking
(MediaPipe) to a calibrated autoencoder cursor to 9-region base velocity, over
reachy2_sdk (gRPC/IP).

Phase 1 - Calibration (calibration_phase): move your hand through all
    positions you intend to use. SPACE = record sample, ENTER = finish
    (min 30 samples required).

Phase 2 - Cursor preview (cursor_preview_phase): same cursor/region view
    as Control, but nothing is sent to the robot -- lets the caller get a
    feel for the cursor before it starts driving anything.
"""

import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import mediapipe as mp
import numpy as np
import scipy.signal as sgn
import tensorflow as tf
from mediapipe.tasks.python.vision import hand_landmarker
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

import safety

HAND_CONNECTIONS = hand_landmarker.HandLandmarksConnections.HAND_CONNECTIONS

# --- Lidar safety ---
LIDAR_SLOWDOWN_DISTANCE = 0.7   # [m]
LIDAR_CRITICAL_DISTANCE = 0.55  # [m], when navigating freely
LIDAR_CRITICAL_DISTANCE_SLOWDOWN = 0.15  # [m], when approaching the object

# --- Virtual screen dimensions ---
BASE_WIDTH = 2550
BASE_HEIGHT = 1500

# MIN_LINEAR/MIN_ANGULAR to let the mobile base's wheels overcome their own resistance 
MIN_LINEAR = 0.2      # [m/s]
MAX_LINEAR = 0.5      # [m/s]
MIN_ANGULAR = 0.8     # [rad/s]
MAX_ANGULAR = 1.1     # [rad/s]

DEAD_ZONE_PX = 200    # pixel radius around screen center before motion starts

PUBLISH_HZ = 20  # [Hz] speed-command rate — comfortably under the mobile base's 0.2s command duration

# MediaPipe Tasks hand-landmarker model (.task), lives at the package root by default.
DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "hand_landmarker.task"
)

# Shared window names for the cursor preview and control phases, so the same
# OS windows stay open across the transition instead of closing and reopening.
CAM_WINDOW_NAME = "BoMI - Camera"
MAP_WINDOW_NAME = "BoMI - Cursor Map"

# Screen position (top-left corner) the map window is pinned to
MAP_WINDOW_POS = (0, 0)

# Saved BoMIMap calibrations (see BoMIMap.save_map_bomi/load_map_bomi) live in
# a 'calibrations/' folder next to the package.
CALIB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calibrations")


def resolve_calib_path(name: str) -> str:
    """Bare name -> calibrations/<name>.npz; a name that already contains a
    path is used as-is (.npz appended either way if missing)."""
    if not name.endswith(".npz"):
        name += ".npz"
    if os.path.dirname(name):
        return name
    return os.path.join(CALIB_DIR, name)


class CursorFilter:
    """
    3rd-order Butterworth low-pass filter for the (crs_x, crs_y) cursor position.
    Coefficients are derived from the actual sample rate (SAMPLE_HZ) instead
    of being hardcoded for a fixed frequency loop.
    """

    ORDER = 3
    SAMPLE_HZ = 30.0  # assumed webcam/control loop sample rate
    CUTOFF_HZ = 4.0   # cutoff frequency

    def __init__(self, sample_hz: float = SAMPLE_HZ, cutoff_hz: float = CUTOFF_HZ) -> None:
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

    def reset(self, crs_x: float, crs_y: float) -> None:
        """Reinit history to steady-state at (crs_x, crs_y), so tracking
        resumed after a gap doesn't overshoot on stale samples."""
        steady = np.array([crs_x, crs_y])
        self._in_history[:] = steady
        self._out_history[:] = steady


# --- Autoencoder forward map parameters ---
# Same architecture/hyperparameters as Naji's compute_bomi_map.Autoencoder.train_network,
# as used for dr_mode="ae" in main_reaching_FullHand_MOD_ae.py's train_ae():
# Input -> Dense(32, tanh) -> Dense(32, tanh) -> Dense(2, linear) [latent/cursor]
#       -> Dense(32, tanh) -> Dense(32, tanh) -> Dense(n_features, linear) [reconstruction]
AE_N_STEPS = 3001      # training epochs (n_steps)
AE_LR = 0.02           # Adam learning rate
AE_HIDDEN_UNITS = 32   # nh1 = nh2
AE_ACTIVATION = "tanh"
AE_SEED = 20
AE_LATENT_DIM = 2      # cu: 2 code units -> (crs_x, crs_y)


class BoMIMap:
    """
    Autoencoder forward map: raw hand landmarks -> 2D cursor in screen space.

    A dense autoencoder is trained on the raw landmark calibration samples
    (reconstruction loss only, no separate normalization), and its 2-unit
    latent layer is used standalone as the cursor space at inference time --
    exactly the dr_mode="ae" path in Naji's BoMI pipeline
    (compute_bomi_map.Autoencoder.train_network + reaching_functions.get_mapped_values).
    Scale/offset map the latent codes' peak-to-peak range onto the screen size,
    centered on the mean, same as train_ae()'s post-training step (with rot=0).

    The screen-space map is stored as a single affine transform cu_screen = A @
    cu_latent + b (A starts out diagonal, i.e. plain per-axis scale). customize()
    folds an extra rotation/gain/offset -- the same operation as Naji's
    rotation_custom/scale_custom/offset_custom in reaching_functions.get_mapped_values
    -- into that same (A, b), so a customized map is still just a BoMIMap that
    save_map_bomi/load_map_bomi handle unchanged.
    """

    def __init__(self) -> None:
        self._w1 = self._b1 = None  # encoder layer 1 (Dense, tanh)
        self._w2 = self._b2 = None  # encoder layer 2 (Dense, tanh)
        self._w3 = self._b3 = None  # encoder layer 3 (Dense, linear) -> latent/cursor
        self._A = np.eye(2)     # latent -> screen affine map (starts diagonal = plain scale)
        self._b = np.zeros(2)
        self.fitted = False

    def fit(self, samples: list) -> None:
        X = np.array(samples, dtype=np.float32)
        n_features = X.shape[1]

        tf.keras.backend.clear_session()
        np.random.seed(AE_SEED)
        tf.random.set_seed(AE_SEED)
        initializer = tf.keras.initializers.GlorotNormal(seed=AE_SEED)

        inputs = Input(shape=(n_features,))
        hidden1 = Dense(AE_HIDDEN_UNITS, activation=AE_ACTIVATION, kernel_initializer=initializer)(inputs)
        hidden1 = Dense(AE_HIDDEN_UNITS, activation=AE_ACTIVATION, kernel_initializer=initializer)(hidden1)
        latent = Dense(AE_LATENT_DIM, kernel_initializer=initializer)(hidden1)
        hidden2 = Dense(AE_HIDDEN_UNITS, activation=AE_ACTIVATION, kernel_initializer=initializer)(latent)
        hidden2 = Dense(AE_HIDDEN_UNITS, activation=AE_ACTIVATION, kernel_initializer=initializer)(hidden2)
        predictions = Dense(n_features, kernel_initializer=initializer)(hidden2)

        encoder = Model(inputs=inputs, outputs=latent)
        autoencoder = Model(inputs=inputs, outputs=predictions)
        autoencoder.compile(loss="mse", optimizer=Adam(learning_rate=AE_LR))

        print(f"Training autoencoder BoMI map ({AE_N_STEPS} epochs)...")
        autoencoder.fit(x=X, y=X, epochs=AE_N_STEPS, verbose=0, batch_size=len(X), shuffle=False)
        print("Autoencoder training done.")

        # Keep only the encoder half (first 3 Dense layers) for standalone inference,
        # same as train_ae() only persisting weights1/2/3 + biases1/2/3.
        dense_layers = [layer for layer in autoencoder.layers if layer.get_weights()]
        self._w1, self._b1 = dense_layers[0].get_weights()
        self._w2, self._b2 = dense_layers[1].get_weights()
        self._w3, self._b3 = dense_layers[2].get_weights()

        train_cu = encoder.predict(X, verbose=0)

        extent = np.ptp(train_cu, axis=0)
        extent = np.where(extent > 1e-6, extent, 1.0)

        screen = np.array([BASE_WIDTH, BASE_HEIGHT], dtype=float)

        scale = screen / extent
        self._A = np.diag(scale)
        self._b = screen / 2.0 - (train_cu * scale).mean(axis=0)
        self.fitted = True

    def transform(self, features: np.ndarray) -> tuple:
        """
        Autoencoder BoMI map: raw hand landmarks -> 2D cursor in screen space,
        via the trained encoder's forward pass (2 tanh hidden layers + linear
        latent layer), same as reaching_functions.get_mapped_values(dr_mode="ae").
        Returns (crs_x, crs_y) in pixels, clipped to the screen size.
        """
        h = np.tanh(np.dot(features, self._w1) + self._b1)
        h = np.tanh(np.dot(h, self._w2) + self._b2)
        cu = np.dot(h, self._w3) + self._b3  # latent code == raw cursor position
        cu = self._A @ cu + self._b  # map latent extent onto the screen size
        crs_x = float(np.clip(cu[0], 0, BASE_WIDTH))
        crs_y = float(np.clip(cu[1], 0, BASE_HEIGHT))
        return crs_x, crs_y

    def customize(self, rot_deg: float = 0.0, gain_x: float = 1.0, gain_y: float = 1.0,
                  off_x: float = 0.0, off_y: float = 0.0) -> None:
        """
        Compose an extra screen-space rotation/gain/offset on top of the current
        map, exactly like Naji's rotation_custom/scale_custom/offset_custom
        (applied in reaching_functions.get_mapped_values after the base AE/PCA
        map): recentre on the screen middle, rotate (screen space is left-handed,
        hence the sign flip), apply a per-axis gain -- negative flips that axis --
        then offset. Composes into the same (A, b) used by transform(), so calling
        this repeatedly (e.g. from a live tuning UI) keeps stacking correctly, and
        the result is saved/loaded like any other BoMIMap.
        """
        center = np.array([BASE_WIDTH, BASE_HEIGHT]) / 2.0
        rad = -np.radians(rot_deg)  # left-handed screen space
        rot = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
        gain_rot = np.array([gain_x, gain_y])[:, None] * rot  # diag(gain) @ rot

        self._A = gain_rot @ self._A
        self._b = gain_rot @ (self._b - center) + center + np.array([off_x, off_y])

    def save_map_bomi(self, path: str) -> None:
        if not self.fitted:
            raise RuntimeError("Cannot save an unfitted BoMIMap.")
        np.savez(
            path,
            w1=self._w1, b1=self._b1,
            w2=self._w2, b2=self._b2,
            w3=self._w3, b3=self._b3,
            A=self._A,
            b=self._b,
        )

    def load_map_bomi(self, path: str) -> None:
        data = np.load(path)
        self._w1, self._b1 = data["w1"], data["b1"]
        self._w2, self._b2 = data["w2"], data["b2"]
        self._w3, self._b3 = data["w3"], data["b3"]
        self._A = data["A"]
        self._b = data["b"]
        self.fitted = True


# --- Velocity helpers ---
def check_region_cursor(crs_x: float, crs_y: float) -> int:
    """
    Returns region 1-9 based on 3x3 grid over BASE_WIDTH x BASE_HEIGHT.
    Layout:
        1 | 2 | 3   
        4 | 5 | 6   
        7 | 8 | 9   
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


def _ramped_axis_velocity(delta: float, half_extent: float, dead_zone_px: float, min_v: float, max_v: float) -> float:
    """Signed velocity for one axis: 0 inside dead_zone_px of center, then
    ramps linearly from min_v to max_v"""
    norm = delta / half_extent
    dead_zone_norm = dead_zone_px / half_extent
    magnitude = abs(norm)
    if magnitude <= dead_zone_norm:
        return 0.0
    t = min((magnitude - dead_zone_norm) / (1.0 - dead_zone_norm), 1.0)
    sign = 1.0 if norm > 0 else -1.0
    return sign * (min_v + t * (max_v - min_v))


def compute_dynamic_vel_from_cursor(
    crs_x: float,
    crs_y: float,
    min_linear: float = MIN_LINEAR,
    max_linear: float = MAX_LINEAR,
    min_angular: float = MIN_ANGULAR,
    max_angular: float = MAX_ANGULAR,
    dead_zone_px: float = DEAD_ZONE_PX,
    ang_right_is_negative: bool = True,
) -> tuple:
    """
    Linear/angular velocity from cursor position, each axis independent: 
    0 inside dead_zone_px of screen center on that axis, then min_*..max_* 
    linearly from there to the screen edge.
    Up from center -> positive linear; right from center -> negative angular.
    """
    cx = BASE_WIDTH / 2.0
    cy = BASE_HEIGHT / 2.0
    dx = crs_x - cx
    dy = crs_y - cy

    lin_vel = _ramped_axis_velocity(-dy, cy, dead_zone_px, min_linear, max_linear)  # up = positive
    ang_sign = -1.0 if ang_right_is_negative else 1.0
    ang_vel = ang_sign * _ramped_axis_velocity(dx, cx, dead_zone_px, min_angular, max_angular)
    return lin_vel, ang_vel


# --- Autoencoder forward map ---
def _extract_hand_features(hand_landmarks) -> np.ndarray:
    """
    Flatten all 21 hand landmarks (x, y) into a 42-element vector.
    hand_landmarks is the list of NormalizedLandmark returned by MediaPipe
    Tasks (e.g. results.hand_landmarks[0]).
    """
    coords = [[lm.x, lm.y] for lm in hand_landmarks]
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


def draw_cursor_map(crs_x: float, crs_y: float, region: int, message: str,
                      map_width: int = 510, map_height: int = 300):
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
    cv2.circle(canvas, (cx, cy), 8, (0, 0, 255), -1)

    cv2.putText(canvas, f"region={region}  cursor=({crs_x:.0f},{crs_y:.0f})",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(canvas, f"-> mobile base: {message}",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return canvas


def update_bomi_cursor(cap, landmarker, bomi_map: BoMIMap, cursor_filter: CursorFilter, crs_x: float, crs_y: float):
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
    _draw_hand_landmarks(frame, hl)
    crs_x, crs_y = bomi_map.transform(_extract_hand_features(hl))
    crs_x, crs_y = cursor_filter.update(crs_x, crs_y)
    return frame, crs_x, crs_y, True


def map_bomi_to_frame(crs_x: float, crs_y: float, width: int, height: int) -> tuple:
    """Rescale a cursor position from the BASE_WIDTH x BASE_HEIGHT BoMI screen
    space into an arbitrary window's pixel space"""
    return (
        int(crs_x / BASE_WIDTH * width),
        int(crs_y / BASE_HEIGHT * height),
    )


# --- Phases ---
def calibration_phase(cap, landmarker) -> list:
    MIN_SAMPLES = 30
    samples = []

    print("\n=== CALIBRATION ===")
    print("Move your hand through all positions you intend to use.")
    print("SPACE = record sample   |   ENTER = finish (need >= 30)   |   Q = quit")

    last_landmarks = None  # most recent detection, so a keypress landing on a
                            # dropout frame (tracking flickers frame to frame) doesn't silently lose the sample
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = landmarker.detect_for_video(mp_image, int(time.time() * 1000))

        if results.hand_landmarks:
            last_landmarks = results.hand_landmarks[0]
            _draw_hand_landmarks(frame, last_landmarks)

        label = f"Samples: {len(samples)}/{MIN_SAMPLES}  SPACE=add  ENTER=done  Q=quit"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        window_name = "BoMI - Calibration"
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            if last_landmarks is not None:
                samples.append(_extract_hand_features(last_landmarks))
                print(f"  Sample {len(samples)} recorded")
            else:
                print("  SPACE ignored: no hand detected yet")

        elif key == 13:  # ENTER
            if len(samples) >= MIN_SAMPLES:
                print(f"  Calibration done ({len(samples)} samples)")
                break
            else:
                print(f"  Need at least {MIN_SAMPLES} samples (have {len(samples)})")

        elif safety.quit_requested(key, window_name):
            print("Aborted.")
            sys.exit(0)

    cv2.destroyWindow(window_name)
    return samples

def cursor_preview_phase(cap, landmarker, bomi_map: BoMIMap, cursor_filter: CursorFilter = None,
                           crs_x: float = None, crs_y: float = None, show_cam: bool = True,
                           on_frame=None, hold_seconds: float = 5.0) -> tuple:
    """
    Shows the same cursor/region view as the control phase, but lets the user 
    where the cursor is before enabling motion.

    Pass an existing cursor_filter/crs_x/crs_y to continue them; returns the 
    final (crs_x, crs_y). 
    - show_cam=False skips the raw camera/landmarks window 
    entirely, for callers that only want the cursor map on screen. 
    - on_frame, if given, is called with no arguments once per loop iteration.
    """
    cursor_filter = cursor_filter or CursorFilter()
    cam_window = CAM_WINDOW_NAME
    map_window = MAP_WINDOW_NAME

    if crs_x is None or crs_y is None:
        crs_x, crs_y = BASE_WIDTH / 2.0, BASE_HEIGHT / 2.0
    region = check_region_cursor(crs_x, crs_y)
    center_hold_start = None

    print("\n=== CURSOR PREVIEW (robot not moving) ===")
    print(f"Get a feel for the cursor. Hold it centered (region 5) for {hold_seconds:.0f}s "
          "to start Control   |   Q = quit")

    while True:
        frame, crs_x, crs_y, hand_detected = update_bomi_cursor(cap, landmarker, bomi_map, cursor_filter, crs_x, crs_y)
        if frame is None:
            continue
        if hand_detected:
            region = check_region_cursor(crs_x, crs_y)

        now = time.time()
        # Only accrue while actively tracked and centered, so a dropped
        # hand while the stale cursor happens to sit in region 5 can't
        # silently trigger the switch.
        center_hold_start = (center_hold_start or now) if (hand_detected and region == 5) else None
        center_progress = min((now - center_hold_start) / hold_seconds, 1.0) if center_hold_start else 0.0

        cv2.putText(
            frame, f"region={region}  cursor=({crs_x:.0f},{crs_y:.0f})",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
        )
        cv2.putText(
            frame, f"PREVIEW - robot not moving. Hold centered: {center_progress * 100:.0f}%  Q=quit",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
        )

        if show_cam:
            cv2.imshow(cam_window, frame)
        cv2.imshow(map_window, draw_cursor_map(crs_x, crs_y, region, "(preview - not sent)"))
        cv2.moveWindow(map_window, *MAP_WINDOW_POS)  # a just-closed window can make the WM reclaim the position otherwise
        cv2.setWindowProperty(map_window, cv2.WND_PROP_TOPMOST, 1)  # re-pin (same-process windows only)
        safety.raise_window(map_window)  # actually wins over the cross-process fullscreen camera_viewer window
        if on_frame is not None:
            on_frame()

        key = cv2.waitKey(1) & 0xFF
        if center_progress >= 1.0:
            break
        if (show_cam and safety.quit_requested(key, cam_window)) or safety.quit_requested(key, map_window):
            print("Aborted.")
            sys.exit(0)

    # Windows are intentionally left open so the same cam/map windows carry 
    # straight into the control phase instead of flickering shut.
    return crs_x, crs_y


