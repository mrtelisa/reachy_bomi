#!/usr/bin/env python3
"""
Fullscreen cursor reaching test -- no robot, no socket, no velocities.

The hand drives an on-screen cursor through the same chain as socket_client.py
(webcam -> MediaPipe -> autoencoder BoMI map -> Butterworth filter) and the
participant has to bring the cursor onto targets. Geometry and trial sequence
are those of markerlessBoMI's main_reaching.py / reaching.py:

  virtual canvas 1200 x 650 (scaled to the whole screen, aspect preserved),
  home at the centre, targets spread over the whole canvas (seeded random
  point in each cell of a 6x3 grid, all cells before any repeat; the 248
  positions are frozen in config/cursor_targets.csv), target
  radius 40 px, cursor radius 15 px, dwell 1 s inside the target,
  N_TARGETS (248, as in the original) targets, center-out: centre -> target
  -> centre -> target ... (496 goals in total), score 4/3/2/1 by reach time.
  The original's blocks/repetitions and "blind" trials are not reproduced.

The session timer starts when the centre (first goal) is reached for the
first time. The session ends when the sequence is over or after --max-minutes
(default 4), whichever comes first; the window closes and the results are
saved (a subject with previous sessions gets _1, _2, ... appended):
  results_center_out/<subject>_center_out_trials.csv     one row per trial (metrics)
  results_center_out/<subject>_center_out_summary.json   success rate, mean metrics, config

Usage:
    python3 reaching_center_out.py --calib <name> --subject S001 [--cam 0] [--max-minutes 4]
(reaching_random.py runs the same test with targets in random order and no
returns to the centre; it reuses everything in this file.)
Keys: Q / ESC = abort (results so far are still saved).
"""

import argparse
import collections
import csv
import datetime
import json
import math
import os
import re
import sys
import time

import cv2
import numpy as np

import socket_client as bomi
from reaching_metrics import compute_trial_metrics, summarize

# --- Geometry (markerlessBoMI reaching.py) ---
CANVAS_W, CANVAS_H = 1200, 650
CRS_RADIUS = 15
TGT_RADIUS = 40
# Targets are spread over the whole canvas instead of a 260 px circle, to
# check that every part of the interface can be reached: the canvas is split
# into GRID_COLS x GRID_ROWS cells, visited in a seeded random order (all cells
# once before any repeats), with a random point inside each cell.
GRID_COLS, GRID_ROWS = 6, 3
TARGET_MARGIN = TGT_RADIUS + 10   # keep targets fully on screen
TARGET_SEED = 20
# The generated positions are frozen in this file (versioned with the code):
# it is what every session actually uses, so the sequence can never drift.
TARGETS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "cursor_targets.csv")

# --- Sequence ---
N_TARGETS = 248   # as many as the original's 11 blocks (8+28+28+28+28+8+28+28+28+28+8)

# --- Timing (markerlessBoMI reaching_functions.py) ---
DWELL_S = 1.0            # cursor must stay inside the target this long (original: 250 ms)
OUT_OF_TIME_S = 1.0      # state 1 ("out of time") after this, only affects the score
TRIAL_TIMEOUT_S = 10.0   # a target not reached within this time is marked missed and skipped
                         # (not applied to the very first centre goal, before the session starts)
# Score, as in the original: points by the time from target shown to entering
# it (the dwell is not counted): < 2 s -> 4, < 3 s -> 3, < 4 s -> 2, else 1.
SCORE_THRESHOLDS = ((2.0, 4), (3.0, 3), (4.0, 2))

# --- Metrics (pixels of the 1200x650 canvas) ---
MOTION_ONSET_SPEED = 40.0     # [px/s]
SPEED_PEAK_THRESHOLD = 80.0   # [px/s]
RESAMPLE_HZ = 50.0

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results_center_out")
WINDOW = "BoMI - Reaching (center-out)"

# Colours (BGR)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLUE = (255, 80, 0)
CURSOR = (int(0.4 * 255), int(0.65 * 255), int(0.19 * 255))   # markerlessBoMI CURSOR, RGB -> BGR
REACHED_FLASH_S = 0.25   # target filled green for this long after a reach
# Target ring: green, blue while the cursor is inside, filled green once reached.


def generate_target_positions(n_targets: int, seed: int = TARGET_SEED) -> list:
    """n_targets (cell, x, y) spread over the canvas: cells of a
    GRID_COLS x GRID_ROWS grid in seeded random order (every cell once before
    any cell repeats), a uniformly random point inside each cell, never the
    same cell twice in a row. Only used to create TARGETS_FILE once."""
    rng = np.random.default_rng(seed)
    cell_w = (CANVAS_W - 2 * TARGET_MARGIN) / GRID_COLS
    cell_h = (CANVAS_H - 2 * TARGET_MARGIN) / GRID_ROWS
    n_cells = GRID_COLS * GRID_ROWS
    cells, last = [], None
    while len(cells) < n_targets:
        perm = rng.permutation(n_cells).tolist()
        if perm[0] == last:
            perm.append(perm.pop(0))
        cells += perm
        last = cells[-1]
    out = []
    for cell in cells[:n_targets]:
        col, row = cell % GRID_COLS, cell // GRID_COLS
        x = TARGET_MARGIN + (col + rng.random()) * cell_w
        y = TARGET_MARGIN + (row + rng.random()) * cell_h
        out.append((int(cell), float(x), float(y)))
    return out


def target_positions(n_targets: int = N_TARGETS) -> list:
    """The fixed list of (cell, x, y) targets, read from TARGETS_FILE so that
    every participant and every launch -- on any machine, with any NumPy
    version -- gets exactly the same sequence. The file is created once from
    generate_target_positions() if it does not exist (delete it to regenerate)."""
    if not os.path.exists(TARGETS_FILE):
        positions = generate_target_positions(n_targets)
        with open(TARGETS_FILE, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["target_number", "cell", "x", "y"])
            w.writerows((k, cell, f"{x:.3f}", f"{y:.3f}") for k, (cell, x, y) in enumerate(positions, start=1))
        print(f"[targets] generated {n_targets} target positions -> {TARGETS_FILE} (commit this file)")
    with open(TARGETS_FILE, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) < n_targets:
        raise ValueError(f"{TARGETS_FILE} has {len(rows)} targets, {n_targets} needed: delete it to regenerate")
    return [(int(r["cell"]), float(r["x"]), float(r["y"])) for r in rows[:n_targets]]


def build_trials() -> list:
    """Flat list of goals: centre -> target -> centre -> target ..., the
    N_TARGETS targets from target_positions() (whole canvas). Each entry:
    kind, target_number (1..N_TARGETS, the target this goal belongs to),
    index (grid cell, -1 = centre), x, y (canvas px)."""
    cx, cy = CANVAS_W / 2.0, CANVAS_H / 2.0
    trials = []
    for k, (cell, x, y) in enumerate(target_positions(N_TARGETS), start=1):
        trials.append({"kind": "home", "target_number": k, "index": -1, "x": cx, "y": cy})
        trials.append({"kind": "target", "target_number": k, "index": cell, "x": x, "y": y})
    return trials


def session_name(subject: str, sequence: str, results_dir: str) -> str:
    """<subject>_<sequence> for the first session, then <subject>_<sequence>_1,
    _2, ... (any existing <results_dir>/<subject>_<sequence>* file counts as a
    previous session)."""
    prefix = f"{subject}_{sequence}"
    existing = [f for f in os.listdir(results_dir) if f.startswith(prefix)]
    if not existing:
        return prefix
    used = {int(m.group(1)) for f in existing
            for m in [re.match(rf"{re.escape(prefix)}_(\d+)_(trials\.csv|summary\.json)$", f)] if m}
    return f"{prefix}_{max(used, default=0) + 1}"


class ReachingCursorTest:
    def __init__(self, subject: str, max_minutes: float, trials: list = None,
                 results_dir: str = RESULTS_DIR, sequence: str = "center_out") -> None:
        """trials/results_dir/sequence let another script (reaching_random.py)
        reuse the whole test with a different goal sequence and output folder."""
        self.subject = subject
        self.max_seconds = max_minutes * 60.0
        self.trials = trials if trials is not None else build_trials()
        self.sequence = sequence
        self.results = []
        self.score = 0

        os.makedirs(results_dir, exist_ok=True)
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.base = os.path.join(results_dir, session_name(subject, sequence, results_dir))

        # Session/trial state (times are time.time())
        self.t_session0 = None
        self.trial_i = -1
        self.trial = None
        self.t_shown = None
        self.t_enter = None
        self.samples = []
        self.flash_until = 0.0
        self.end_reason = None

    # --- trial flow ---
    def start(self, t: float) -> None:
        # The first goal is the centre: the session timer only starts once it
        # has been reached (see _end_trial), so t_session0 stays None until then.
        self.t_start = t
        self._next_trial(t)

    def _next_trial(self, t: float) -> None:
        self.trial_i += 1
        if self.trial_i >= len(self.trials):
            self.end_reason = "completed"
            return
        self.trial = self.trials[self.trial_i]
        self.t_shown = t
        self.t_enter = None
        self.samples = []

    def update(self, t: float, x: float, y: float) -> None:
        """One frame with the current (filtered) cursor position in canvas px."""
        if self.trial is None or self.end_reason:
            return
        self.samples.append((t, x, y))

        inside = math.hypot(x - self.trial["x"], y - self.trial["y"]) < TGT_RADIUS
        if inside:
            if self.t_enter is None:
                self.t_enter = t
            elif t - self.t_enter >= DWELL_S:
                self._end_trial(t)
                self.flash_until = t + REACHED_FLASH_S
                self._next_trial(t)
        else:
            self.t_enter = None
            if self.t_session0 is not None and t - self.t_shown >= TRIAL_TIMEOUT_S:
                self._record_trial(t, success=False, reason="trial_timeout")
                print(f"  trial {self.trial_i + 1}/{len(self.trials)} NOT reached ({TRIAL_TIMEOUT_S:.0f}s timeout)")
                self._next_trial(t)

        if self.t_session0 is not None and t - self.t_session0 >= self.max_seconds:
            self.end_reason = "session_timeout"

    def t0(self) -> float:
        """Time origin for the logs: session start if it has begun, else the launch."""
        return self.t_session0 if self.t_session0 is not None else self.t_start

    def state(self, t: float, x: float, y: float) -> int:
        """0 = out of target in time, 1 = out of target out of time, 2 = inside (as in the original)."""
        if self.t_enter is not None:
            return 2
        return 1 if t - self.t_shown > OUT_OF_TIME_S else 0

    def _end_trial(self, t: float) -> None:
        if self.t_session0 is None:
            self.t_session0 = t   # first reach of the centre: the session (and its timer) starts now
            print("  centre reached: session timer started")
        reach_time = self.t_enter - self.t_shown
        points = next((p for limit, p in SCORE_THRESHOLDS if reach_time < limit), 1)
        self.score += points
        self._record_trial(t, success=True, reason="reached", points=points)
        print(f"  trial {self.trial_i + 1}/{len(self.trials)} reached in {reach_time:.2f}s (+{points}, score {self.score})")

    def _record_trial(self, t: float, success: bool, reason: str, points: int = 0) -> None:
        metrics = compute_trial_metrics(
            self.samples, (self.trial["x"], self.trial["y"]), self.t_shown, self.t_enter if success else None,
            MOTION_ONSET_SPEED, SPEED_PEAK_THRESHOLD, RESAMPLE_HZ,
        )
        self.results.append({
            "trial": self.trial_i + 1, "kind": self.trial["kind"],
            "target_number": self.trial["target_number"], "index": self.trial["index"],
            "target_x": self.trial["x"], "target_y": self.trial["y"],
            "success": success, "end_reason": reason, "points": points, "score": self.score,
            "t_shown": self.t_shown - self.t0(), "t_end": t - self.t0(),
            "trial_duration": t - self.t_shown,
            **metrics,
        })

    def finish(self, t: float, reason: str = None) -> dict:
        """Close the current (unfinished) trial as missed and write the results."""
        reason = reason or self.end_reason or "aborted"
        if self.trial is not None and self.trial_i < len(self.trials) and reason != "completed":
            self._record_trial(t, success=False, reason=reason)
        # One entry per missed target (same position = same target, e.g. the
        # centre), with a counter of how many times it was missed and in which trials
        missed = collections.OrderedDict()
        for r in self.results:
            if r["success"]:
                continue
            key = (r["target_x"], r["target_y"])
            if key not in missed:
                missed[key] = {"kind": r["kind"], "cell": r["index"],
                               "x": round(r["target_x"], 1), "y": round(r["target_y"], 1),
                               "times_missed": 0, "trials": [], "reasons": []}
            missed[key]["times_missed"] += 1
            missed[key]["trials"].append(r["trial"])
            missed[key]["reasons"].append(r["end_reason"])
        missed_by_cell = collections.Counter(r["index"] for r in self.results if not r["success"])
        shown_by_cell = collections.Counter(r["index"] for r in self.results)
        summary = {
            "subject": self.subject, "sequence": self.sequence, "end_reason": reason,
            "n_trials_total": len(self.trials),
            "session_duration": (t - self.t_session0) if self.t_session0 is not None else 0.0,
            "timestamp": self.timestamp,
            "score": self.score,
            **summarize(self.results),
            # Which targets were not reached: grid cell (-1 = centre), canvas
            # position, how many times it was missed, in which trials and why
            # (trial_timeout / session_timeout / aborted)
            "missed_targets": list(missed.values()),
            # Misses per grid cell (-1 = centre): how many goals in that cell were
            # shown and how many of them were missed
            "missed_by_cell": {
                str(cell): {"presented": shown_by_cell[cell], "missed": missed_by_cell.get(cell, 0)}
                for cell in sorted(missed_by_cell)
            },
            "config": {
                "canvas": [CANVAS_W, CANVAS_H], "crs_radius": CRS_RADIUS, "tgt_radius": TGT_RADIUS,
                "grid": [GRID_COLS, GRID_ROWS], "target_seed": TARGET_SEED,
                "dwell_s": DWELL_S, "max_seconds": self.max_seconds,
                "n_targets": N_TARGETS,
                "motion_onset_speed": MOTION_ONSET_SPEED, "speed_peak_threshold": SPEED_PEAK_THRESHOLD,
            },
        }
        if self.results:
            with open(self.base + "_trials.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(self.results[0].keys()))
                w.writeheader()
                w.writerows(self.results)
        with open(self.base + "_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary


# --- Drawing ---
class Screen:
    """Fullscreen window; the 1200x650 canvas is scaled uniformly and centred."""

    def __init__(self, title: str = WINDOW) -> None:
        self.window = title
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(self.window, np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8))
        cv2.waitKey(50)
        _, _, w, h = cv2.getWindowImageRect(self.window)
        if w <= 0 or h <= 0:
            w, h = CANVAS_W, CANVAS_H
        self.w, self.h = w, h
        self.scale = min(w / CANVAS_W, h / CANVAS_H)
        self.ox = (w - CANVAS_W * self.scale) / 2.0
        self.oy = (h - CANVAS_H * self.scale) / 2.0

    def to_screen(self, x: float, y: float) -> tuple:
        return int(self.ox + x * self.scale), int(self.oy + y * self.scale)

    def px(self, r: float) -> int:
        return max(1, int(round(r * self.scale)))

    def draw(self, test: ReachingCursorTest, crs_x: float, crs_y: float, t: float, hand_detected: bool) -> None:
        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        tr = test.trial
        if tr is not None and not test.end_reason:
            state = test.state(t, crs_x, crs_y)
            centre = self.to_screen(tr["x"], tr["y"])
            colour = BLUE if state == 2 else GREEN
            cv2.circle(img, centre, self.px(TGT_RADIUS), colour, self.px(2))
        # Reached feedback: previous target position is gone, flash a filled disc at the cursor's goal
        if t < test.flash_until and test.results:
            last = test.results[-1]
            cv2.circle(img, self.to_screen(last["target_x"], last["target_y"]), self.px(TGT_RADIUS), GREEN, -1)
        cv2.circle(img, self.to_screen(crs_x, crs_y), self.px(CRS_RADIUS), CURSOR if hand_detected else (90, 90, 90), -1)

        font, fs = cv2.FONT_HERSHEY_SIMPLEX, 0.9 * self.scale
        cv2.putText(img, str(test.score), (self.w - int(120 * self.scale), int(60 * self.scale)), font, 1.8 * self.scale, RED, 3)
        cv2.putText(img, f"target {min(test.trial_i + 1, len(test.trials))}/{len(test.trials)}",
                    (int(20 * self.scale), int(40 * self.scale)), font, fs, BLUE, 2)
        elapsed = (t - test.t_session0) if test.t_session0 is not None else 0.0
        remaining = max(0.0, test.max_seconds - elapsed)
        cv2.putText(img, f"{int(remaining // 60)}:{int(remaining % 60):02d}",
                    (int(20 * self.scale), self.h - int(20 * self.scale)), font, fs, WHITE, 2)
        cv2.imshow(self.window, img)


# --- Entry point ---
def main(build=build_trials, results_dir: str = RESULTS_DIR, sequence: str = "center_out",
         description: str = __doc__) -> None:
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calib", required=True, help="Calibration to load, e.g. 'elisa' for calibrations/elisa.npz")
    parser.add_argument("--subject", default="S000", help="Subject id used in the result file names (default: S000)")
    parser.add_argument("--cam", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--model", default=bomi.DEFAULT_MODEL_PATH, help="MediaPipe hand_landmarker.task model")
    parser.add_argument("--max-minutes", type=float, default=4.0, help="Session hard stop (default: 4)")
    args = parser.parse_args()

    calib_path = bomi._resolve_calib_path(args.calib)
    if not os.path.exists(calib_path):
        print(f"[ERROR] No calibration file '{calib_path}' found.")
        sys.exit(1)
    if not os.path.exists(args.model):
        print(f"[ERROR] MediaPipe model not found: '{args.model}'")
        sys.exit(1)

    bomi_map = bomi.BoMIMap()
    bomi_map.load(calib_path)
    print(f"Loaded calibration from {calib_path}")

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {args.cam}")
        sys.exit(1)
    landmarker = bomi.hand_landmarker.HandLandmarker.create_from_options(
        bomi.hand_landmarker.HandLandmarkerOptions(
            base_options=bomi.base_options.BaseOptions(model_asset_path=args.model),
            running_mode=bomi.vision_task_running_mode.VisionTaskRunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )

    test = ReachingCursorTest(args.subject, args.max_minutes, trials=build(),
                              results_dir=results_dir, sequence=sequence)
    screen = Screen(title=f"BoMI - Reaching ({sequence})")
    cursor_filter = bomi.CursorFilter()
    # Map space (BASE_WIDTH x BASE_HEIGHT) -> canvas, same calibration as the robot
    sx, sy = CANVAS_W / bomi.BASE_WIDTH, CANVAS_H / bomi.BASE_HEIGHT
    crs_x, crs_y = bomi.BASE_WIDTH / 2.0, bomi.BASE_HEIGHT / 2.0

    print(f"\n=== CURSOR REACHING ({sequence}) === {len(test.trials)} goals, max {args.max_minutes:.0f} min. Q = abort")
    test.start(time.time())
    try:
        while not test.end_reason:
            _, crs_x, crs_y, hand_detected = bomi.update_bomi_cursor(cap, landmarker, bomi_map, cursor_filter, crs_x, crs_y)
            t = time.time()
            cx = min(max(crs_x * sx, 0.0), CANVAS_W)
            cy = min(max(crs_y * sy, 0.0), CANVAS_H)
            test.update(t, cx, cy)
            screen.draw(test, cx, cy, t, hand_detected)
            key = cv2.waitKey(1) & 0xFF
            if bomi._quit_requested(key, screen.window):
                test.end_reason = "aborted"
    finally:
        summary = test.finish(time.time())
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()

    print(f"\nSession over ({summary['end_reason']}): {summary['n_success']}/{summary['n_trials_done']} reached, "
          f"score {summary['score']}, {summary['session_duration']:.0f}s")
    if summary["missed_targets"]:
        print("Missed targets:")
        for m in summary["missed_targets"]:
            print(f"  {m['kind']:6s} cell {m['cell']:2d}  at ({m['x']:.0f}, {m['y']:.0f})  missed {m['times_missed']}x "
                  f"(trials {m['trials']}, {', '.join(sorted(set(m['reasons'])))})")


if __name__ == "__main__":
    main()
