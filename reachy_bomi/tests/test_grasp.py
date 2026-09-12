#!/usr/bin/env python3
"""
Test script for the object selection + grasp planning + execution pipeline,
mouse-driven (no BoMI cursor/webcam needed) -- the counterpart to
reachy_selection.py's BoMI-cursor-driven select_object_to_grasp_bomi/
confirm_grasp_bomi.

Connects to Reachy, sends both arms straight to the elbow_135 starting
posture, then hover-selects an object with the mouse and confirms it.
Once confirmed, its point cloud is built, reachy_grasp.plan_grasp computes
pre-grasp/grasp/lift poses (plotted via graphs.show_grasp_plan), and
reachy_grasp.execute_grasp drives the arm through them for real. On a
successful lift, _place_back_and_wind_down puts the object right back
down where it was picked up, retracts to elbow_135, returns to the
default posture, and powers off -- ending the run (see there for why,
unlike a failed/skipped grasp, which re-captures and offers selection
again).

Usage:
    python3 test_grasp.py [robot_ip]

    <robot_ip> is optional; if omitted, bomi_teleop.DEFAULT_ROBOT_IP is used.

    Options:
        --yolo-model PATH   YOLOv8 weights (.pt). Default: yolov8n.pt
        --conf FLOAT        Minimum detection confidence. Default: 0.5

    Q, ESC, or closing a window with the X = quit / go back.
"""

import argparse
import os
import sys
import time
from typing import Optional, Tuple

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from reachy2_sdk import ReachySDK
from ultralytics import YOLO

import bomi_teleop
import graphs
import reachy_detection
import reachy_grasp
import reachy_selection
import safety

# Same convention as tests/elbow_135.py.
ELBOW_PITCH_DEG = -135.0
ELBOW_MOVE_DURATION_S = 3.0


def _goto_elbow_135(reachy: ReachySDK, duration: float = ELBOW_MOVE_DURATION_S) -> None:
    """Sends both arms to the elbow_135 posture (elbow_90 base, pitch
    overridden), blocking. Used both as the starting posture and, after a
    successful grasp, to retract before winding down."""
    for arm in (reachy.r_arm, reachy.l_arm):
        if arm is None or not arm.is_on():
            continue
        joints = arm.get_default_posture_joints(common_posture="elbow_90")
        joints[3] = ELBOW_PITCH_DEG
        print(f"Moving {arm._part_id.name} to elbow_135 posture...")
        arm.goto(joints, duration=duration, wait=True)


def _select_object_to_grasp(
    depth_cam, model: YOLO, confidence: float, captured: "reachy_detection.Capture", reachy: ReachySDK,
) -> Tuple[Optional[str], Optional["reachy_detection.Box"], "reachy_detection.Capture"]:
    """Mouse-driven equivalent of reachy_selection.select_object_to_grasp_bomi
    (which hovers with the BoMI cursor instead) -- detect-and-hover loop on
    an already-captured frame. Returns the class name and its box once held
    green for HOVER_HOLD_SECONDS (or None, None if the user quit), plus the
    (frame, detections, labels), so a later re-entry (e.g. after answering
    "No") can reuse them without recapturing."""
    base_frame, detections, labels = captured

    mouse = reachy_selection.MouseTracker(reachy_detection.CAM_WINDOW_NAME)
    hovered_box = None
    hover_start: Optional[float] = None
    button_hover_start: Optional[float] = None

    print(f"\n=== CAPTURED FRAME (RGB + YOLO) ===  Q = quit  |  "
          f"hover Reposition for {reachy_selection.REPOSITIONING_HOVER_SECONDS:.0f}s to reposition")
    while True:
        now = time.time()
        on_button = reachy_selection.box_contains(reachy_selection.REPOSITIONING_BUTTON_BOX, mouse.x, mouse.y)
        button_hover_start = (button_hover_start or now) if on_button else None

        frame = base_frame.copy()

        hovered = reachy_selection.find_hovered_detection(detections, mouse.x, mouse.y)

        if hovered is None:
            hovered_box, hover_start = None, None
        else:
            box = hovered[2]
            if hovered_box is None or reachy_selection.iou(box, hovered_box) < reachy_selection.HOVER_IOU_MATCH:
                hover_start = now
            hovered_box = box
        hover_duration = (now - hover_start) if hover_start is not None else 0.0

        is_held = hovered is not None and hover_duration >= reachy_selection.HOVER_HOLD_SECONDS

        if is_held:
            box = hovered[2]
            reachy_detection.draw_box(frame, box, labels[box], reachy_detection.COLOR_GREEN)
            cv2.imshow(reachy_detection.CAM_WINDOW_NAME, frame)
            cv2.waitKey(1)
            return hovered[0], box, (base_frame, detections, labels)

        hover_progress = min(hover_duration / reachy_selection.HOVER_HOLD_SECONDS, 1.0) if hovered is not None else 0.0
        for class_name, conf, box in detections:
            is_hovered = hovered is not None and box == hovered[2]
            color = reachy_detection.COLOR_YELLOW if is_hovered else reachy_detection.COLOR_BLUE
            reachy_detection.draw_box(frame, box, labels[box], color, hover_progress if is_hovered else 0.0)

        button_progress = (
            min((now - button_hover_start) / reachy_selection.REPOSITIONING_HOVER_SECONDS, 1.0) if button_hover_start else 0.0
        )
        reachy_selection.draw_repositioning_button(frame, button_progress)
        cv2.imshow(reachy_detection.CAM_WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if safety.quit_requested(key, reachy_detection.CAM_WINDOW_NAME):
            return None, None, (base_frame, detections, labels)


def _confirm_grasp(class_name: str) -> Optional[bool]:
    """Mouse-driven equivalent of reachy_selection.confirm_grasp_bomi.
    Blocking Yes/No dialog. Returns True (Yes), False (No), or None if the
    user quit."""
    mouse = reachy_selection.MouseTracker(reachy_selection.CONFIRM_WINDOW_NAME)
    yes_hover_start: Optional[float] = None
    no_hover_start: Optional[float] = None

    while True:
        now = time.time()
        on_yes = reachy_selection.box_contains(reachy_selection.YES_BUTTON_BOX, mouse.x, mouse.y)
        on_no = reachy_selection.box_contains(reachy_selection.NO_BUTTON_BOX, mouse.x, mouse.y)

        yes_hover_start = (yes_hover_start or now) if on_yes else None
        no_hover_start = (no_hover_start or now) if on_no else None

        yes_progress = min((now - yes_hover_start) / reachy_selection.CONFIRM_HOVER_SECONDS, 1.0) if yes_hover_start else 0.0
        no_progress = min((now - no_hover_start) / reachy_selection.CONFIRM_HOVER_SECONDS, 1.0) if no_hover_start else 0.0

        cv2.imshow(
            reachy_selection.CONFIRM_WINDOW_NAME,
            reachy_selection.draw_confirm_canvas(
                [f"You have select the {class_name} to be grasped.", "Do you want to confirm?"],
                yes_progress, no_progress,
            ),
        )

        key = cv2.waitKey(1) & 0xFF
        result: Optional[bool] = None
        quit_now = safety.quit_requested(key, reachy_selection.CONFIRM_WINDOW_NAME)
        if yes_progress >= 1.0:
            result = True
        elif no_progress >= 1.0:
            result = False

        if quit_now or result is not None:
            cv2.destroyWindow(reachy_selection.CONFIRM_WINDOW_NAME)
            return None if quit_now else result


def _place_back_and_wind_down(reachy: ReachySDK, plan: reachy_grasp.GraspPlan) -> None:
    """Runs once, right after a successful grasp+lift: puts the object back
    down exactly where it was picked up from and retreats to pregrasp
    along a straight line (reachy_grasp.place_back, head watching the
    end-effector throughout), then -- before retracting -- sends the head
    back to its own default posture (reachy.head.goto_posture), so it's no
    longer tracking the end-effector once the arms start moving to elbow_135
    (now a safe joint-space move since the gripper already cleared the
    object), returns to the default posture, then powers off """
    reachy_grasp.place_back(reachy, plan)

    if reachy.head is not None:
        reachy.head.goto_posture(duration=1.0, wait=False)

    print("\nRetracting both arms to the elbow_135 posture...")
    _goto_elbow_135(reachy)

    reachy.mobile_base.turn_on()
    reachy.mobile_base.translate_by(x=-0.1, y=0.0, wait=True)
    reachy.mobile_base.rotate_by(180.0, wait=True)

    print("\nReturning to default posture...")
    reachy.goto_posture("default", duration=3.0, wait=True)

    print("\nPowering off...")
    safety.safe_robot_shutdown(reachy)


def _test_grasp_planning(reachy: ReachySDK, model: YOLO, confidence: float) -> None:
    depth_cam = reachy.cameras.depth
    if depth_cam is None:
        print("[ERROR] No depth camera reported by the robot")
        return

    print("\n=== Capturing frame ===")
    captured = reachy_detection.capture_and_detect(depth_cam, model, confidence, reachy_selection.presentable_filter(reachy))
    if captured is None:
        return

    while True:
        class_name, box, captured = _select_object_to_grasp(depth_cam, model, confidence, captured, reachy)
        if class_name is None:
            break

        decision = _confirm_grasp(class_name)
        if decision is None:
            break
        if decision:
            geometry = reachy_detection.build_object_point_cloud(depth_cam, class_name, box)
            if geometry is not None:
                print(f"[{class_name}] estimated width={geometry.width_m * 100:.1f}cm  "
                      f"height={geometry.height_m * 100:.1f}cm")
                plan = reachy_grasp.plan_grasp(reachy, geometry)
                if plan is None:
                    print(f"[{class_name}] no feasible grasp (too wide for the gripper, "
                          "or its pose couldn't be estimated)")
                else:
                    # Same matrix as pregrasp_matrix below, printed so it can
                    # be pasted into tests/test_reachability.py to check a
                    # rejection in isolation from the rest of the pipeline.
                    rounded = [[round(float(v), 6) for v in row] for row in plan.pregrasp_matrix.tolist()]
                    print(f"[{class_name}] pregrasp target for {plan.arm_name}: np.array({rounded})")

                    #graphs.show_grasp_plan(geometry, plan)
                    if reachy_grasp.execute_grasp(reachy, plan):
                        _place_back_and_wind_down(reachy, plan)
                        break

            # Re-capture before offering selection again -- building the
            # point cloud takes real time, so the scene may have changed.
            print("\n=== Re-capturing ===")
            captured = reachy_detection.capture_and_detect(depth_cam, model, confidence, reachy_selection.presentable_filter(reachy))
            if captured is None:
                break
        # "No" -> same, back to hover-select

    cv2.destroyWindow(reachy_detection.CAM_WINDOW_NAME)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test the object selection + grasp planning/execution pipeline."
    )
    parser.add_argument("robot_ip", nargs="?", default=bomi_teleop.DEFAULT_ROBOT_IP,
                        help=f"IP address of the Reachy robot (default: {bomi_teleop.DEFAULT_ROBOT_IP})")
    parser.add_argument("--yolo-model", default=reachy_detection.YOLO_MODEL_PATH,
                        help=f"Path to YOLOv8 weights (default: {reachy_detection.YOLO_MODEL_PATH})")
    parser.add_argument("--conf", type=float, default=reachy_detection.YOLO_CONFIDENCE,
                        help=f"Minimum reachy_detection confidence (default: {reachy_detection.YOLO_CONFIDENCE})")
    cli_args = parser.parse_args()

    reachy = ReachySDK(host=cli_args.robot_ip)

    if reachy.cameras is None or reachy.cameras.depth is None:
        print(f"[ERROR] No depth camera reported by the robot at '{cli_args.robot_ip}'")
        reachy.disconnect()
        sys.exit(1)
    if reachy.r_arm is None and reachy.l_arm is None:
        print(f"[ERROR] No arm reported by the robot at '{cli_args.robot_ip}'")
        reachy.disconnect()
        sys.exit(1)

    reachy.turn_on()

    def _on_emergency_quit() -> None:
        safety.emergency_shutdown(reachy)

    safety.start_global_quit_watcher(_on_emergency_quit)
    stop_terminal_watcher = safety.start_terminal_quit_watcher(_on_emergency_quit)

    # Straight to elbow_135's final joint configuration, no intermediate
    # "default" posture first (unlike tests/elbow_135.py itself).
    _goto_elbow_135(reachy)

    print(f"Loading YOLO model '{cli_args.yolo_model}'...")
    model = YOLO(cli_args.yolo_model)

    try:
        time.sleep(2.0)
        _test_grasp_planning(reachy, model, cli_args.conf)
    finally:
        if stop_terminal_watcher is not None:
            stop_terminal_watcher()
        cv2.destroyAllWindows()
        reachy.disconnect()


if __name__ == "__main__":
    main()
