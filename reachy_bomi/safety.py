#!/usr/bin/env python3
"""
Quit/shutdown safety net: in-loop quit checks (Q/ESC, window closed) plus
OS-level ESC/Q watchers on a background thread, and the one-shot base
back-up + rotation and robot power-off they trigger.
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

# Back-up distance and in-place rotation before powering down next to the table
SHUTDOWN_REVERSE_CM = 20.0
SHUTDOWN_ROTATION_DEG = 180.0

_wmctrl_missing_warned = False

_rotation_lock = threading.Lock()
_rotation_done = False

_shutdown_started = threading.Event()


def shutdown_started() -> bool:
    """True once safe_robot_shutdown has begun, on any thread. The quit watchers
    run on their own threads while the teleop loops are still running: those
    loops must stop commanding the base as soon as this is True, or their
    speed commands fight the shutdown rotation and it overshoots."""
    return _shutdown_started.is_set()


def rotate_base_once(mobile_base, reverse_cm: float = SHUTDOWN_REVERSE_CM) -> None:
    """Translate back + rotate SHUTDOWN_ROTATION_DEG, only the first time it is
    called in the process: the wind-down rotation and an ESC shutdown can fire
    together. The lock is held for the whole movement, so a concurrent caller
    waits instead of powering the robot off mid-rotation."""
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
    """Ask the window manager (wmctrl) for fullscreen: some Qt builds ignore cv2's
    WND_PROP_FULLSCREEN. No-op if wmctrl is missing."""
    global _wmctrl_missing_warned
    try:
        subprocess.run(["wmctrl", "-r", window_name, "-b", "add,fullscreen"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        if not _wmctrl_missing_warned:
            print("[WARN] wmctrl not installed -- camera windows may not go fullscreen "
                  "(sudo apt install wmctrl).")
            _wmctrl_missing_warned = True


# raise_window throttle (called every frame, wmctrl spawned a few times a second)
_RAISE_WINDOW_INTERVAL_S = 0.5
_last_raise_time: dict = {}


def raise_window(window_name: str) -> None:
    """Re-activate a window via wmctrl (cv2's TOPMOST loses to windows of other
    processes, e.g. camera_viewer.py). Throttled; no-op if wmctrl is missing."""
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


def destroy_window(window_name: str) -> None:
    """cv2.destroyWindow tolerant of the window already being gone (on the Qt
    backend destroying a missing window raises instead of no-op'ing)."""
    try:
        cv2.destroyWindow(window_name)
    except cv2.error:
        pass


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
    """Stop the base and power the robot down smoothly (exceptions swallowed).
    rotate_base_before_shutdown backs the base up and rotates it first, for
    when the robot sits next to the table. Flags shutdown_started() first."""
    _shutdown_started.set()

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
