#!/usr/bin/env python3
"""Quit/shutdown safety net for the whole pipeline.

Two layers, because a single cv2.waitKey-based check only ever fires while
a cv2 window has OS focus and the caller's own loop is actively polling it:
  - quit_requested: local, in-loop check (Q/ESC while a cv2 window has
    focus, or that window closed) for graceful "go back a step" UI flow.
  - start_global_quit_watcher / start_terminal_quit_watcher: OS-level
    watchers on a background thread, so ESC/Q stops the robot regardless
    of what the main thread is doing or which window has focus.
"""

import os
import select
import subprocess
import sys
import termios
import threading
import time
import tty

import cv2
from reachy2_sdk import ReachySDK

# How far the base translates backward, then how much it rotates in place,
# before powering down, when rotate_base_before_shutdown is set to True
SHUTDOWN_REVERSE_CM = 20.0
SHUTDOWN_ROTATION_DEG = 180.0

_wmctrl_missing_warned = False

_rotation_lock = threading.Lock()
_rotation_done = False


def rotate_base_once(mobile_base, reverse_cm: float = SHUTDOWN_REVERSE_CM) -> None:
    """Translate back + rotate SHUTDOWN_ROTATION_DEG, but only the first time
    this is called in the whole process -- guards against the wind-down
    rotation and an ESC-triggered shutdown (or two quit watchers) firing at
    the same time and rotating the base twice."""
    global _rotation_done
    with _rotation_lock:
        if _rotation_done or mobile_base is None:
            return
        _rotation_done = True
    try:
        mobile_base.turn_on()
        mobile_base.translate_by(x=-reverse_cm / 100.0, y=0.0, wait=True)
        mobile_base.rotate_by(SHUTDOWN_ROTATION_DEG, wait=True)
    except Exception as exc:
        print(f"[WARN] Could not rotate the base ({exc}).")


def force_fullscreen(window_name: str) -> None:
    """Requests the EWMH fullscreen state via wmctrl. Some Qt/opencv-python
    builds report cv2's own WND_PROP_FULLSCREEN as set without Mutter ever
    actually resizing the window -- this bypasses that by asking the WM
    directly. Call once, after the window is first shown. No-ops (after one
    warning) if wmctrl isn't installed."""
    global _wmctrl_missing_warned
    try:
        subprocess.run(["wmctrl", "-r", window_name, "-b", "add,fullscreen"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        if not _wmctrl_missing_warned:
            print("[WARN] wmctrl not installed -- camera windows may not go fullscreen "
                  "(sudo apt install wmctrl).")
            _wmctrl_missing_warned = True


# Throttle for raise_window -- meant to be called every display-loop frame,
# but should only actually spawn wmctrl a few times a second
_RAISE_WINDOW_INTERVAL_S = 0.5
_last_raise_time: dict = {}


def raise_window(window_name: str) -> None:
    """Actively re-activates a window by title via wmctrl -- the same action
    as alt-tabbing to it. Needed because cv2's own WND_PROP_TOPMOST only
    beats another window from the *same process*; confirmed it silently
    loses to a fullscreen window from a different process (e.g.
    camera_viewer.py's subprocess), which this fixes. Safe to call every
    frame -- throttled internally. No-ops (after one warning) if wmctrl
    isn't installed."""
    global _wmctrl_missing_warned
    now = time.time()
    if now - _last_raise_time.get(window_name, 0.0) < _RAISE_WINDOW_INTERVAL_S:
        return
    _last_raise_time[window_name] = now
    try:
        subprocess.run(["wmctrl", "-a", window_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        if not _wmctrl_missing_warned:
            print("[WARN] wmctrl not installed -- windows may not come to the front "
                  "(sudo apt install wmctrl).")
            _wmctrl_missing_warned = True


def quit_requested(key: int, window_name: str) -> bool:
    """True if Q/ESC was pressed, or the window was closed with the X button."""
    if key in (ord('q'), ord('Q'), 27):
        return True
    try:
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return False


def start_global_quit_watcher(on_quit):
    """Hooks ESC/Q system-wide via pynput, so quitting works with no cv2
    window focused. Returns the listener, or None if pynput is unavailable."""
    try:
        from pynput import keyboard
    except ImportError:
        print("[WARN] pynput not installed: ESC/Q only quits with a window focused.")
        return None

    def _on_press(key) -> None:
        if key == keyboard.Key.esc or getattr(key, "char", None) in ("q", "Q"):
            on_quit()

    try:
        listener = keyboard.Listener(on_press=_on_press)
        listener.daemon = True
        listener.start()
        return listener
    except Exception as exc:
        print(f"[WARN] Could not start the global ESC/Q watcher ({exc}).")
        return None


def start_terminal_quit_watcher(on_quit):
    """Watches this process's own terminal for ESC/Q, for when the terminal
    is what actually has keyboard focus. Returns a stop() to restore the terminal,
    or None if stdin isn't an interactive terminal."""
    if not sys.stdin.isatty():
        return None

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    stop_flag = threading.Event()

    def _restore() -> None:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except Exception:
            pass

    def _watch() -> None:
        tty.setcbreak(fd)
        try:
            while not stop_flag.is_set():
                ready, _, _ = select.select([sys.stdin], [], [], 0.2)
                if ready and sys.stdin.read(1) in ("q", "Q", "\x1b"):
                    _restore()
                    on_quit()
                    return
        finally:
            _restore()

    threading.Thread(target=_watch, daemon=True).start()

    def stop() -> None:
        stop_flag.set()
        _restore()

    return stop


def safe_robot_shutdown(reachy: ReachySDK, mobile_base=None, rotate_base_before_shutdown: bool = False) -> None:
    """Stop the base, then power down smoothly. Swallows exceptions.

    rotate_base_before_shutdown=True translates the base back SHUTDOWN_REVERSE_CM
    then rotates it SHUTDOWN_ROTATION_DEG in place first, via rotate_base_once
    (so it never double-fires against a concurrent wind-down rotation) -- pass
    this whenever the robot may be sitting close to a table with its arms
    about to fold in, so they don't fold into it."""
    if mobile_base is not None:
        try:
            mobile_base.set_goal_speed(vx=0, vy=0, vtheta=0)
            mobile_base.send_speed_command()
        except Exception:
            pass

    if rotate_base_before_shutdown:
        rotate_base_once(mobile_base)

    try:
        reachy.turn_off_smoothly()
    except Exception:
        try:
            reachy.turn_off()
        except Exception:
            pass

    if mobile_base is not None:
        try:
            mobile_base.turn_off()
        except Exception:
            pass


def emergency_shutdown(reachy: ReachySDK, mobile_base=None, rotate_base_before_shutdown: bool = False) -> None:
    """safe_robot_shutdown + disconnect + close every cv2 window, then a
    hard process exit. Meant as the on_quit callback for the watchers
    above, so ESC/Q stops the robot no matter what the main thread is
    currently blocked doing."""
    print("\n[QUIT] ESC/Q pressed — stopping the robot and exiting.")
    try:
        safe_robot_shutdown(reachy, mobile_base, rotate_base_before_shutdown=rotate_base_before_shutdown)
        reachy.turn_off()
        cv2.destroyAllWindows()
    finally:
        os._exit(0)
