#!/usr/bin/env python3
"""Blocking live feed of a Reachy camera view (used by camera_viewer.py)."""

from typing import Callable

import cv2
from reachy2_sdk.media.camera import CameraView, DepthCamera


def stream_blocking(
    depth_cam: DepthCamera, window_name: str, quit_requested: Callable[[int, str], bool],
    view: CameraView = CameraView.LEFT,
) -> None:
    """Show frames until quit_requested(key, window_name) is True."""
    print("\n=== LIVE RGB STREAM (no detection) ===  Q = quit")
    while True:
        result = depth_cam.get_frame(view=view)
        if result is None:
            continue
        frame, _timestamp = result
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if quit_requested(key, window_name):
            break
