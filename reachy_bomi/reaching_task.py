#!/usr/bin/env python3
"""
Center-out reaching task for the Reachy mobile base in Gazebo.

Ported from markerlessBoMI's reaching test (reaching.py / reaching_functions.py):
targets on circles around a home position, presented in a fixed pseudo-random
order, alternating peripheral target -> home -> peripheral target -> ...; the
circle radius grows every `targets_per_distance` targets (circle_radii). A
target counts as reached when the base centre stays within `target_radius` of
it for `dwell_time` seconds; a target not reached within `trial_timeout` is
missed and the task moves on.

Timeline:
  1. Node starts with bomi_control.launch.py, waits for Gazebo (/odom) and
     for the operator to finish the cursor preview: socket_client.py sends
     "nine region" then, which socket_server republishes as
     socket_server/base_state == 1.0.
  2. Session starts: `session_duration` timer + first target shown.
  3. Session ends when all targets are done or the timer expires. The node
     publishes a summary, writes a CSV next to the bag, and exits; the launch
     file shuts the whole simulation down on its exit.

Per-trial metrics (computed from /odom, published as JSON on
/reaching/trial_result so they end up in the bag):
  reaction_time      target shown -> movement onset (speed > motion_onset_speed)
  movement_time      movement onset -> entering the target for the last time
  reach_time         target shown -> entering the target for the last time
  success            reached within trial_timeout
  path_length, normalized_path_length (path / straight-line displacement onset -> end, 1 = straight)
  max_deviation      max perpendicular distance from the ideal line onset -> target centre
  dimensionless_jerk sqrt(0.5 * int |jerk|^2 dt * T^5 / L^2), and log_dimensionless_jerk = -ln(DJ)
  n_speed_peaks      local maxima of the speed profile above speed_peak_threshold
  mean_speed, peak_speed
"""

import csv
import datetime
import json
import math
import os

import numpy as np
import yaml
import rclpy
from ament_index_python.packages import get_package_share_directory
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from scipy.signal import find_peaks
from std_msgs.msg import Float32, String

TARGET_MODEL = "reaching_target"   # marker model in worlds/reaching.world
HIDDEN_Z = -2.0                    # marker parked below the ground plane (pole included)
SHOWN_Z = 0.025                    # marker (5 cm thick) resting on the ground

DEFAULT_CONFIG = os.path.join(get_package_share_directory("reachy_bomi"), "config", "reaching.yaml")
DEFAULT_RESULTS_DIR = os.path.expanduser("~/reachy_bomi_bags")


def _yaw_from_quaternion(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


# --- Metrics ---
def compute_trial_metrics(samples, goal, t_shown, t_reach, onset_speed, peak_threshold, resample_hz):
    """
    samples: list of (t, x, y, vx, vy) in the odom frame, from target shown to
    trial end. goal: (x, y). t_reach: time the target was entered for the last
    time (None if missed -> the whole trial is used).
    Returns a dict of metrics (None where not computable).
    """
    m = {
        "reaction_time": None, "movement_time": None, "reach_time": None,
        "path_length": None, "straight_distance": None, "normalized_path_length": None,
        "max_deviation": None, "dimensionless_jerk": None, "log_dimensionless_jerk": None,
        "n_speed_peaks": None, "mean_speed": None, "peak_speed": None,
    }
    if len(samples) < 3:
        return m
    arr = np.asarray(samples, dtype=float)
    t, x, y, vx, vy = arr.T
    speed = np.hypot(vx, vy)

    if t_reach is not None:
        m["reach_time"] = float(t_reach - t_shown)

    moving = np.flatnonzero(speed > onset_speed)
    if moving.size == 0:
        return m
    i_onset = int(moving[0])
    t_onset = float(t[i_onset])
    m["reaction_time"] = t_onset - t_shown

    t_end = t_reach if t_reach is not None else float(t[-1])
    i_end = int(np.searchsorted(t, t_end, side="right"))
    if i_end - i_onset < 3:
        return m
    m["movement_time"] = t_end - t_onset

    xy = np.column_stack([x[i_onset:i_end], y[i_onset:i_end]])
    seg = np.diff(xy, axis=0)
    path_length = float(np.sum(np.hypot(seg[:, 0], seg[:, 1])))
    p0 = xy[0]
    g = np.asarray(goal, dtype=float)
    # Straight-line distance actually covered (onset -> where the movement
    # ended, i.e. the target's edge when reached), so a perfectly straight
    # reach gives normalized_path_length == 1.
    straight = float(np.hypot(*(xy[-1] - p0)))
    m["path_length"] = path_length
    m["straight_distance"] = straight
    if straight > 1e-6:
        m["normalized_path_length"] = path_length / straight
    # Max deviation from the ideal line onset -> target centre
    ideal = float(np.hypot(*(g - p0)))
    if ideal > 1e-6:
        d = (g - p0) / ideal
        rel = xy - p0
        m["max_deviation"] = float(np.max(np.abs(rel[:, 0] * d[1] - rel[:, 1] * d[0])))

    sp = speed[i_onset:i_end]
    m["mean_speed"] = float(np.mean(sp))
    m["peak_speed"] = float(np.max(sp))

    # Uniform resampling before differentiating: /odom timestamps jitter.
    dt = 1.0 / resample_hz
    tt = t[i_onset:i_end]
    tu = np.arange(tt[0], tt[-1], dt)
    if tu.size < 5:
        return m
    xu = np.interp(tu, tt, x[i_onset:i_end])
    yu = np.interp(tu, tt, y[i_onset:i_end])
    vxu, vyu = np.gradient(xu, dt), np.gradient(yu, dt)
    axu, ayu = np.gradient(vxu, dt), np.gradient(vyu, dt)
    jxu, jyu = np.gradient(axu, dt), np.gradient(ayu, dt)
    duration = tu[-1] - tu[0]
    # Dimensionless jerk (Hogan & Sternad 2009), amplitude = straight-line displacement
    if straight > 1e-6 and duration > 0:
        jerk_int = float(np.sum(jxu ** 2 + jyu ** 2) * dt)
        dj = math.sqrt(0.5 * jerk_int * duration ** 5 / straight ** 2)
        m["dimensionless_jerk"] = dj
        m["log_dimensionless_jerk"] = -math.log(dj) if dj > 0 else None

    # Speed peaks: light moving-average smoothing, then local maxima above threshold
    su = np.hypot(vxu, vyu)
    k = max(1, int(round(0.1 * resample_hz)))  # 100 ms window
    su_s = np.convolve(su, np.ones(k) / k, mode="same")
    peaks, _ = find_peaks(su_s, height=peak_threshold, prominence=peak_threshold * 0.25)
    m["n_speed_peaks"] = int(peaks.size)
    return m


# --- Node ---
class ReachingTask(Node):
    def __init__(self) -> None:
        super().__init__("reaching_task")
        self.declare_parameter("config", DEFAULT_CONFIG)
        self.declare_parameter("results_dir", DEFAULT_RESULTS_DIR)
        self.declare_parameter("results_prefix", "Reaching")

        cfg_path = self.get_parameter("config").get_parameter_value().string_value
        with open(cfg_path, "r", encoding="utf-8") as f:
            self.cfg = (yaml.safe_load(f) or {}).get("reaching", {})
        c = self.cfg
        self.home = tuple(float(v) for v in c.get("home", [0.0, 0.0]))
        self.target_radius = float(c.get("target_radius", 0.35))
        self.dwell_time = float(c.get("dwell_time", 2.0))
        self.trial_timeout = float(c.get("trial_timeout", 30.0))
        self.session_duration = float(c.get("session_duration", 300.0))
        self.onset_speed = float(c.get("motion_onset_speed", 0.05))
        self.peak_threshold = float(c.get("speed_peak_threshold", 0.10))
        self.resample_hz = float(c.get("resample_hz", 50.0))

        self.trials = self._build_trials()  # list of dicts: kind, index, x, y
        self.n_trials = len(self.trials)

        results_dir = os.path.expanduser(self.get_parameter("results_dir").get_parameter_value().string_value)
        prefix = self.get_parameter("results_prefix").get_parameter_value().string_value
        os.makedirs(results_dir, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_csv = os.path.join(results_dir, f"{prefix}_{stamp}_reaching.csv")
        self.results_json = os.path.join(results_dir, f"{prefix}_{stamp}_reaching_summary.json")

        self.target_pub = self.create_publisher(PointStamped, "/reaching/target", 10)
        self.status_pub = self.create_publisher(String, "/reaching/status", 10)
        self.event_pub = self.create_publisher(String, "/reaching/event", 10)
        self.result_pub = self.create_publisher(String, "/reaching/trial_result", 10)
        self.summary_pub = self.create_publisher(String, "/reaching/summary", 10)

        self.create_subscription(Odometry, "/odom", self._odom_cb, 50)
        self.create_subscription(Float32, "socket_server/base_state", self._base_state_cb, 10)

        self.set_state_cli = self.create_client(SetEntityState, "/gazebo/set_entity_state")

        # Session / trial state (all times are /odom header stamps, i.e. sim time)
        self.session_started = False
        self.session_t0 = None
        self.control_ready = False       # "nine region" received
        self.last_odom_t = None
        self.trial_i = -1
        self.trial = None
        self.t_shown = None
        self.t_enter = None              # time the base last entered the target
        self.samples = []
        self.results = []
        self.finished = False
        self.exit_requested = False

        self.get_logger().info(
            f"Reaching task: {self.n_trials} reaches ({len(self.trials) // 2} targets + home returns, "
            f"distances {[t['radius'] for t in self.trials[::2 * int(c.get('targets_per_distance', 10))]]} m), "
            f"radius {self.target_radius} m, dwell {self.dwell_time} s, timeout {self.trial_timeout} s, "
            f"session {self.session_duration:.0f} s. Waiting for the cursor preview to end..."
        )

    def _build_trials(self) -> list:
        """Peripheral targets alternated with home returns. The circle radius
        steps through `circle_radii`, `targets_per_distance` targets each;
        `order` gives, per distance, the angular indices to present."""
        c = self.cfg
        n = int(c.get("n_targets", 8))
        radii = [float(r) for r in c.get("circle_radii", [2.0])]
        per_dist = int(c.get("targets_per_distance", 10))
        order = c.get("order")
        if not order:
            rng = np.random.default_rng(int(c.get("seed", 20)))
            order = []
            for _ in radii:
                block, last = [], None
                while len(block) < per_dist:
                    perm = rng.permutation(n).tolist()
                    if perm[0] == last:
                        continue
                    block += perm
                    last = block[-1]
                order.append(block[:per_dist])
        if len(order) != len(radii):
            raise ValueError(f"reaching.yaml: 'order' has {len(order)} lists but circle_radii has {len(radii)} entries")

        trials = []
        for dist_i, (radius, indices) in enumerate(zip(radii, order)):
            for k, idx in enumerate(indices[:per_dist]):
                ang = 2.0 * math.pi * idx / n + math.pi / n
                trials.append({
                    "kind": "target", "distance_block": dist_i + 1, "radius": radius,
                    "target_number": dist_i * per_dist + k + 1, "index": int(idx),
                    "x": self.home[0] + radius * math.cos(ang),
                    "y": self.home[1] + radius * math.sin(ang),
                })
                trials.append({
                    "kind": "home", "distance_block": dist_i + 1, "radius": radius,
                    "target_number": dist_i * per_dist + k + 1, "index": -1,
                    "x": self.home[0], "y": self.home[1],
                })
        return trials

    # --- Callbacks ---
    def _base_state_cb(self, msg: Float32) -> None:
        if msg.data == 1.0 and not self.control_ready:
            self.control_ready = True
            self.get_logger().info("Control started (cursor preview over): session starts at next /odom")

    def _odom_cb(self, msg: Odometry) -> None:
        if self.finished:
            return
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.last_odom_t = t

        if not self.session_started:
            if self.control_ready:
                self._start_session(t)
            return

        # Session hard stop
        if t - self.session_t0 >= self.session_duration:
            self.get_logger().info("Session time is up")
            self._end_trial(t, success=False, reason="session_timeout")
            self._finish("session_timeout")
            return
        if self.trial is None:
            return

        p = msg.pose.pose.position
        yaw = _yaw_from_quaternion(msg.pose.pose.orientation)
        vx_b, vy_b = msg.twist.twist.linear.x, msg.twist.twist.linear.y
        vx = vx_b * math.cos(yaw) - vy_b * math.sin(yaw)
        vy = vx_b * math.sin(yaw) + vy_b * math.cos(yaw)
        self.samples.append((t, p.x, p.y, vx, vy))

        dist = math.hypot(p.x - self.trial["x"], p.y - self.trial["y"])
        if dist < self.target_radius:
            if self.t_enter is None:
                self.t_enter = t
            elif t - self.t_enter >= self.dwell_time:
                self._end_trial(t, success=True, reason="reached")
                self._next_trial(t)
                return
        else:
            self.t_enter = None

        if t - self.t_shown >= self.trial_timeout:
            self._end_trial(t, success=False, reason="trial_timeout")
            self._next_trial(t)

    # --- Session / trial flow ---
    def _start_session(self, t: float) -> None:
        self.session_started = True
        self.session_t0 = t
        self._publish_event({"event": "session_start", "t": t, "n_trials": self.n_trials,
                             "session_duration": self.session_duration})
        self._next_trial(t)

    def _next_trial(self, t: float) -> None:
        self.trial_i += 1
        if self.trial_i >= self.n_trials:
            self._finish("completed")
            return
        self.trial = self.trials[self.trial_i]
        self.t_shown = t
        self.t_enter = None
        self.samples = []

        self._move_target(self.trial["x"], self.trial["y"], SHOWN_Z)
        pt = PointStamped()
        pt.header.stamp = self.get_clock().now().to_msg()
        pt.header.frame_id = "odom"
        pt.point.x, pt.point.y = self.trial["x"], self.trial["y"]
        self.target_pub.publish(pt)
        self._publish_status(f"target {self.trial_i + 1}/{self.n_trials}")
        self._publish_event({"event": "target_shown", "t": t, "trial": self.trial_i + 1,
                             "kind": self.trial["kind"], "index": self.trial["index"],
                             "x": self.trial["x"], "y": self.trial["y"]})
        self.get_logger().info(
            f"Trial {self.trial_i + 1}/{self.n_trials}: {self.trial['kind']} "
            f"#{self.trial['index']} at ({self.trial['x']:.2f}, {self.trial['y']:.2f}), "
            f"distance {self.trial['radius']:.1f} m"
        )

    def _end_trial(self, t: float, success: bool, reason: str) -> None:
        if self.trial is None:
            return
        t_reach = self.t_enter if success else None
        metrics = compute_trial_metrics(
            self.samples, (self.trial["x"], self.trial["y"]), self.t_shown, t_reach,
            self.onset_speed, self.peak_threshold, self.resample_hz,
        )
        result = {
            "trial": self.trial_i + 1, "kind": self.trial["kind"], "index": self.trial["index"],
            "target_number": self.trial["target_number"], "distance_block": self.trial["distance_block"],
            "radius": self.trial["radius"],
            "target_x": self.trial["x"], "target_y": self.trial["y"],
            "success": bool(success), "end_reason": reason,
            "t_shown": self.t_shown, "t_end": t, "trial_duration": t - self.t_shown,
            "session_time": t - self.session_t0,
            **metrics,
        }
        self.results.append(result)
        self.result_pub.publish(String(data=json.dumps(result)))
        self._publish_event({"event": reason, "t": t, "trial": self.trial_i + 1, "success": bool(success)})
        mt = result["movement_time"]
        self.get_logger().info(
            f"Trial {self.trial_i + 1} {'reached' if success else 'MISSED'} ({reason}) "
            f"reach {result['reach_time'] if success else float('nan'):.2f}s, "
            f"movement {mt if mt is not None else float('nan'):.2f}s"
        )
        self.trial = None

    def _finish(self, reason: str) -> None:
        if self.finished:
            return
        self.finished = True
        self._move_target(self.home[0], self.home[1], HIDDEN_Z)

        n = len(self.results)
        n_targets = [r for r in self.results if r["kind"] == "target"]
        summary = {
            "end_reason": reason,
            "n_trials_total": self.n_trials,
            "n_trials_done": n,
            "n_success": sum(r["success"] for r in self.results),
            "success_rate": (sum(r["success"] for r in self.results) / n) if n else None,
            "success_rate_targets": (sum(r["success"] for r in n_targets) / len(n_targets)) if n_targets else None,
            "session_duration": (self.last_odom_t - self.session_t0) if self.session_t0 is not None else None,
            "results_csv": self.results_csv,
        }
        for key in ("reach_time", "movement_time", "normalized_path_length", "max_deviation",
                    "log_dimensionless_jerk", "n_speed_peaks"):
            vals = [r[key] for r in self.results if r["success"] and r.get(key) is not None]
            summary[f"mean_{key}"] = float(np.mean(vals)) if vals else None

        self._write_results(summary)
        self.summary_pub.publish(String(data=json.dumps(summary)))
        self._publish_event({"event": "finished", "reason": reason})
        self._publish_status("done")
        self.get_logger().info(f"Reaching task finished ({reason}): {summary['n_success']}/{n} reached. "
                               f"Results in {self.results_csv}")
        # Give the bag recorder a moment to receive the last messages, then exit
        # (bomi_control.launch.py shuts the simulation down on our exit).
        self.create_timer(1.5, self._request_exit)

    def _request_exit(self) -> None:
        self.exit_requested = True

    # --- Helpers ---
    def _move_target(self, x: float, y: float, z: float) -> None:
        if not self.set_state_cli.service_is_ready():
            self.get_logger().warning("/gazebo/set_entity_state not available: target marker not moved")
            return
        req = SetEntityState.Request()
        req.state.name = TARGET_MODEL
        req.state.pose.position.x = x
        req.state.pose.position.y = y
        req.state.pose.position.z = z
        req.state.pose.orientation.w = 1.0
        req.state.reference_frame = "world"
        self.set_state_cli.call_async(req)

    def _publish_status(self, text: str) -> None:
        self.status_pub.publish(String(data=text))

    def _publish_event(self, payload: dict) -> None:
        self.event_pub.publish(String(data=json.dumps(payload)))

    def _write_results(self, summary: dict) -> None:
        if self.results:
            with open(self.results_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(self.results[0].keys()))
                writer.writeheader()
                writer.writerows(self.results)
        with open(self.results_json, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "config": self.cfg}, f, indent=2)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ReachingTask()
    try:
        while rclpy.ok() and not node.exit_requested:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
