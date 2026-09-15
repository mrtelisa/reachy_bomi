#!/usr/bin/env python3
"""
Standalone BoMI calibration loader -- no robot connection needed.

Loads a calibration previously saved with calibrate_bomi.py (or socket_client.py --calibrate) and lets you
try/use it live on the cursor map. Q = quit.

Usage:
    python3 load_bomi.py NAME [--cam INDEX] [--model PATH]

    NAME is the calibration to load, e.g. "elisa" for
    calibrations/elisa.npz (the .npz extension is optional).
"""

import argparse
import os
import sys

import cv2
from mediapipe.tasks.python.core import base_options
from mediapipe.tasks.python.vision import hand_landmarker
from mediapipe.tasks.python.vision.core import vision_task_running_mode

import socket_client as bomi


def _use_map(cap, landmarker, bomi_map: bomi.BoMIMap) -> None:
    cursor_filter = bomi.CursorFilter()
    crs_x, crs_y = bomi.BASE_WIDTH / 2.0, bomi.BASE_HEIGHT / 2.0
    map_window = bomi.MAP_WINDOW_NAME

    print("\n=== USING LOADED CALIBRATION (nothing is sent anywhere) ===  Q = quit")

    while True:
        _, crs_x, crs_y, _ = bomi.update_bomi_cursor(
            cap, landmarker, bomi_map, cursor_filter, crs_x, crs_y,
        )
        region = bomi.check_region_cursor(crs_x, crs_y)
        cv2.imshow(map_window, bomi._draw_cursor_map(crs_x, crs_y, region, "(loaded)"))

        key = cv2.waitKey(1) & 0xFF
        if bomi._quit_requested(key, map_window):
            return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="Calibration to load, e.g. 'elisa' for calibrations/elisa.npz")
    parser.add_argument("--cam", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--model", default=bomi.DEFAULT_MODEL_PATH,
                        help="Path to the MediaPipe hand_landmarker.task model.")
    cli_args = parser.parse_args()

    calib_path = bomi._resolve_calib_path(cli_args.name)
    if not os.path.exists(calib_path):
        print(f"[ERROR] No calibration file '{calib_path}' found.")
        if os.path.isdir(bomi.CALIB_DIR):
            available = [f for f in os.listdir(bomi.CALIB_DIR) if f.endswith(".npz")]
            if available:
                print("        Available: " + ", ".join(sorted(available)))
        sys.exit(1)

    if not os.path.exists(cli_args.model):
        print(f"[ERROR] MediaPipe model not found: '{cli_args.model}'")
        print("        Download hand_landmarker.task and pass its path with --model.")
        sys.exit(1)

    bomi_map = bomi.BoMIMap()
    bomi_map.load(calib_path)
    print(f"Loaded calibration from {calib_path}")

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

        _use_map(cap, landmarker, bomi_map)
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        if landmarker is not None:
            landmarker.close()


if __name__ == "__main__":
    main()
