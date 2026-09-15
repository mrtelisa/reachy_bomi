#!/usr/bin/env python3
"""
Standalone BoMI calibration tool -- no robot connection needed.

Runs the calibration phase, fits the cursor map, then previews it live so
you can move your hand and see how it responds before deciding whether to
keep it: S = save (prompts for a name, stored in calibrations/), Q = quit
without saving.

Usage:
    python3 calibrate_bomi.py [--cam INDEX] [--model PATH]
"""

import argparse
import os
import sys

import cv2
from mediapipe.tasks.python.core import base_options
from mediapipe.tasks.python.vision import hand_landmarker
from mediapipe.tasks.python.vision.core import vision_task_running_mode

import socket_client as bomi


def _prompt_and_save(bomi_map: bomi.BoMIMap) -> bool:
    """Returns True once the map is actually saved (False if the user cancels
    with a blank name, so the caller keeps previewing)."""
    name = input("Calibration name (blank = cancel): ").strip()
    if not name:
        print("Cancelled.")
        return False
    path = bomi._resolve_calib_path(name)
    os.makedirs(bomi.CALIB_DIR, exist_ok=True)
    bomi_map.save(path)
    print(f"Saved to {path}")
    return True


def _preview_and_save(cap, landmarker, bomi_map: bomi.BoMIMap) -> None:
    cursor_filter = bomi.CursorFilter()
    crs_x, crs_y = bomi.BASE_WIDTH / 2.0, bomi.BASE_HEIGHT / 2.0
    map_window = bomi.MAP_WINDOW_NAME

    print("\n=== PREVIEW (nothing is sent anywhere) ===")
    print("Move your hand to try the calibration.  S = save   |   Q = quit without saving")

    while True:
        _, crs_x, crs_y, _ = bomi.update_bomi_cursor(
            cap, landmarker, bomi_map, cursor_filter, crs_x, crs_y,
        )
        region = bomi.check_region_cursor(crs_x, crs_y)
        cv2.imshow(map_window, bomi._draw_cursor_map(crs_x, crs_y, region, "(preview)"))

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("s"), ord("S")):
            if _prompt_and_save(bomi_map):
                return
            continue
        if bomi._quit_requested(key, map_window):
            print("Closed without saving.")
            return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cam", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--model", default=bomi.DEFAULT_MODEL_PATH,
                        help="Path to the MediaPipe hand_landmarker.task model.")
    cli_args = parser.parse_args()

    if not os.path.exists(cli_args.model):
        print(f"[ERROR] MediaPipe model not found: '{cli_args.model}'")
        print("        Download hand_landmarker.task and pass its path with --model.")
        sys.exit(1)

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

        samples = bomi._calibration_phase(cap, landmarker)
        bomi_map = bomi.BoMIMap()
        bomi_map.fit(samples)
        print("Autoencoder map fitted")

        _preview_and_save(cap, landmarker, bomi_map)
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        if landmarker is not None:
            landmarker.close()


if __name__ == "__main__":
    main()
