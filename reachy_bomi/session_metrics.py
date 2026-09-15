"""
Session metrics for the Reachy teleop + grasp experiment (reachy_control.py).

One SessionMetrics object per run collects, from the moment Control starts
after the cursor preview ("test start") to the moment the user declines to
pick another object ("test end"):

  test_duration              test start -> test end
  navigation_duration        test start -> first time object selection opens
  n_repositioning            how many times repositioning navigation was used
  objects_moved              names of the objects picked AND placed (count + list)
  path_length_max_speed      base path [m] while driving at full speed (before the pre-grasp pose)
  path_length_reduced_speed  base path [m] while driving at reduced speed (after the pre-grasp pose)
  path_length_repositioning  base path [m] during repositioning navigation
  path_length_total          sum of the three
  path_length_navigation     max + reduced speed, i.e. the path up to the first object selection
  optimal_path_length        DEFAULT_OPTIMAL_PATH_LENGTH (metres), set below before the session
  normalized_path_length     path_length_navigation / optimal_path_length
  log_dimensionless_jerk     smoothness of the navigation trajectory (test start -> first
                             object selection): -ln( sqrt(0.5 * int |jerk|^2 dt * T^5 / L^2) ),
                             Hogan & Sternad 2009, L = straight-line displacement start -> end
  region_time_percent        share of the driving time the cursor spent in each of the 9
                             regions (cursor previews and confirm dialogs excluded); the
                             dwell time of every completed dwell (dwell_seconds * n_dwell) is
                             removed from region 5 first
  n_dwell                    dwells completed while driving (Control + repositioning)
  n_dwell_declined           of those, "Do you want to continue on the pipeline?" answered No,
                             i.e. dwell + switch without a change of state

Positions come from the mobile base odometry (mobile_base.get_current_odometry),
sampled by the control loops through sample(). Everything is written to
results_robot/<subject>_session.json (a subject with previous sessions gets
_1, _2, ... appended).
"""

import datetime
import json
import math
import os
import re
import time

import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results_robot")

# Length [m] of the optimal navigation path from the start pose to the
# objects, used for normalized_path_length. Set it here before a session
# (it depends on how the room is laid out); None = not known, no normalization.
DEFAULT_OPTIMAL_PATH_LENGTH = None

MODE_MAX = "max_speed"
MODE_REDUCED = "reduced_speed"
MODE_REPOSITIONING = "repositioning"
MODES = (MODE_MAX, MODE_REDUCED, MODE_REPOSITIONING)

JERK_RESAMPLE_HZ = 20.0   # odometry is sampled at the control loop rate (PUBLISH_HZ)
REGIONS = tuple(range(1, 10))
# A gap between two region ticks longer than this means the control loop was
# not running (cursor preview, confirm dialog, arms moving...): not counted.
REGION_TICK_MAX_GAP_S = 0.5


def log_dimensionless_jerk(samples) -> float:
    """samples: list of (t, x, y). Returns -ln(dimensionless jerk) of the
    trajectory, or None if it is too short / the base did not move."""
    if len(samples) < 5:
        return None
    arr = np.asarray(samples, dtype=float)
    t, x, y = arr[:, 0], arr[:, 1], arr[:, 2]
    if np.any(np.diff(t) <= 0):
        keep = np.concatenate([[True], np.diff(t) > 0])
        t, x, y = t[keep], x[keep], y[keep]
        if t.size < 5:
            return None
    dt = 1.0 / JERK_RESAMPLE_HZ
    tu = np.arange(t[0], t[-1], dt)
    if tu.size < 5:
        return None
    xu, yu = np.interp(tu, t, x), np.interp(tu, t, y)
    vx, vy = np.gradient(xu, dt), np.gradient(yu, dt)
    ax, ay = np.gradient(vx, dt), np.gradient(vy, dt)
    jx, jy = np.gradient(ax, dt), np.gradient(ay, dt)
    duration = tu[-1] - tu[0]
    amplitude = math.hypot(xu[-1] - xu[0], yu[-1] - yu[0])
    if duration <= 0 or amplitude < 1e-3:
        return None
    jerk_int = float(np.sum(jx ** 2 + jy ** 2) * dt)
    dj = math.sqrt(0.5 * jerk_int * duration ** 5 / amplitude ** 2)
    return -math.log(dj) if dj > 0 else None


def _session_name(subject: str, results_dir: str) -> str:
    """<subject> for the first session, then <subject>_1, <subject>_2, ..."""
    existing = [f for f in os.listdir(results_dir) if f.startswith(subject + "_") or f == f"{subject}_session.json"]
    if not existing:
        return subject
    used = {int(m.group(1)) for f in existing for m in [re.match(rf"{re.escape(subject)}_(\d+)_session\.json$", f)] if m}
    return f"{subject}_{max(used, default=0) + 1}"


class SessionMetrics:
    def __init__(self, subject: str, optimal_path_m: float = DEFAULT_OPTIMAL_PATH_LENGTH,
                 dwell_seconds: float = 0.0, results_dir: str = RESULTS_DIR) -> None:
        self.subject = subject
        self.optimal_path_m = optimal_path_m
        self.dwell_seconds = dwell_seconds
        os.makedirs(results_dir, exist_ok=True)
        self.path_json = os.path.join(results_dir, _session_name(subject, results_dir) + "_session.json")
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self.t_test_start = None
        self.t_test_end = None
        self.end_reason = None
        self.t_first_selection = None
        self.n_repositioning = 0
        self.objects_moved = []
        self.path = {m: 0.0 for m in MODES}
        self.n_samples = {m: 0 for m in MODES}
        self._last_xy = None
        self._nav_samples = []   # (t, x, y) from test start to the first object selection
        self.region_time = {r: 0.0 for r in REGIONS}
        self._last_region_tick = None
        self.n_dwell = 0
        self.n_dwell_declined = 0
        self.saved = False

    # --- events ---
    def start_test(self) -> None:
        if self.t_test_start is None:
            self.t_test_start = time.time()
            print(f"[metrics] test started ({self.subject})")

    def enter_object_selection(self) -> None:
        if self.t_first_selection is None and self.t_test_start is not None:
            self.t_first_selection = time.time()
            print(f"[metrics] navigation over after {self.t_first_selection - self.t_test_start:.1f}s, "
                  f"{self.path_length_navigation:.2f} m")

    def repositioning(self) -> None:
        self.n_repositioning += 1

    def object_moved(self, name: str) -> None:
        self.objects_moved.append(name)

    def dwell(self, accepted) -> None:
        """A dwell in region 5 completed while driving. accepted: True = it
        changed state (pre-grasp pose / object selection / back to selection),
        False = the user answered No and kept driving, None = quit."""
        self.n_dwell += 1
        if accepted is False:
            self.n_dwell_declined += 1

    def end_test(self, reason: str) -> None:
        if self.t_test_end is None:
            self.t_test_end = time.time()
            self.end_reason = reason

    # --- odometry ---
    def sample(self, odom: dict, mode: str) -> None:
        """odom: mobile_base.get_current_odometry() dict (x, y in metres);
        mode: MODE_MAX / MODE_REDUCED / MODE_REPOSITIONING, the driving mode the
        base is in right now. Call it at the control-loop rate while driving."""
        if self.t_test_start is None or self.t_test_end is not None:
            return
        x, y = float(odom["x"]), float(odom["y"])
        t = time.time()
        if self._last_xy is not None:
            self.path[mode] += math.hypot(x - self._last_xy[0], y - self._last_xy[1])
        self._last_xy = (x, y)
        self.n_samples[mode] += 1
        if self.t_first_selection is None:
            self._nav_samples.append((t, x, y))

    def region_tick(self, region: int, now: float = None) -> None:
        """Called every control-loop iteration with the cursor's current
        region: accumulates the time since the previous tick on that region.
        Gaps longer than REGION_TICK_MAX_GAP_S (previews, dialogs) are skipped."""
        if self.t_test_start is None or self.t_test_end is not None:
            return
        now = time.time() if now is None else now
        if self._last_region_tick is not None:
            gap = now - self._last_region_tick
            if 0.0 < gap <= REGION_TICK_MAX_GAP_S and region in self.region_time:
                self.region_time[region] += gap
        self._last_region_tick = now

    def reset_path_origin(self) -> None:
        """Call after a base motion that is not the user's driving (e.g. the
        automatic back-up/rotation), so the jump is not counted as path."""
        self._last_xy = None

    # --- results ---
    @property
    def path_length_navigation(self) -> float:
        return self.path[MODE_MAX] + self.path[MODE_REDUCED]

    def region_shares(self) -> tuple:
        """(seconds per region after removing the dwells from region 5, percent per region)."""
        secs = dict(self.region_time)
        removed = min(secs[5], self.dwell_seconds * self.n_dwell)
        secs[5] -= removed
        total = sum(secs.values())
        pct = {f"region{r}": (100.0 * secs[r] / total if total > 0 else 0.0) for r in REGIONS}
        return secs, pct, removed

    def summary(self) -> dict:
        t_end = self.t_test_end or time.time()
        nav_end = self.t_first_selection or t_end
        nav_path = self.path_length_navigation
        region_secs, region_pct, dwell_removed = self.region_shares()
        return {
            "subject": self.subject,
            "timestamp": self.timestamp,
            "end_reason": self.end_reason,
            "test_duration": (t_end - self.t_test_start) if self.t_test_start else None,
            "navigation_duration": (nav_end - self.t_test_start) if self.t_test_start else None,
            "reached_object_selection": self.t_first_selection is not None,
            "n_repositioning": self.n_repositioning,
            "n_objects_moved": len(self.objects_moved),
            "objects_moved": list(self.objects_moved),
            "path_length_max_speed": self.path[MODE_MAX],
            "path_length_reduced_speed": self.path[MODE_REDUCED],
            "path_length_repositioning": self.path[MODE_REPOSITIONING],
            "path_length_navigation": nav_path,
            "path_length_total": sum(self.path.values()),
            "optimal_path_length": self.optimal_path_m,
            "normalized_path_length": (nav_path / self.optimal_path_m) if self.optimal_path_m else None,
            "log_dimensionless_jerk": log_dimensionless_jerk(self._nav_samples),
            "region_time_percent": {k: round(v, 1) for k, v in region_pct.items()},
            "region_time_seconds": {f"region{r}": round(region_secs[r], 2) for r in REGIONS},
            "region5_dwell_time_removed": round(dwell_removed, 2),
            "n_dwell": self.n_dwell,
            "n_dwell_declined": self.n_dwell_declined,
            "n_odometry_samples": dict(self.n_samples),
        }

    def save(self) -> dict:
        s = self.summary()
        with open(self.path_json, "w", encoding="utf-8") as f:
            json.dump(s, f, indent=2)
        self.saved = True
        print(f"[metrics] saved {self.path_json}")
        return s
