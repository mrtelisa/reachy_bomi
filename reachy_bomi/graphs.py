#!/usr/bin/env python3
"""All matplotlib plotting for the grasp pipeline."""

import os
import re
import time

import matplotlib.pyplot as plt
import numpy as np

GRAPHS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "graphs")


def _save_fig(fig, name: str) -> None:
    """Save fig to GRAPHS_DIR as <timestamp>_<slug>.png, creating the dir if needed."""
    os.makedirs(GRAPHS_DIR, exist_ok=True)
    slug = re.sub(r"[^\w.-]+", "_", name).strip("_")
    filename = f"{time.strftime('%Y%m%d_%H%M%S')}_{slug}.png"
    fig.savefig(os.path.join(GRAPHS_DIR, filename), dpi=150, bbox_inches="tight")


def show_point_cloud(point_cloud: np.ndarray, class_name: str) -> None:
    """Non-blocking 3D scatter view of a point cloud (Reachy coords,
    [m]), colored by height. Used for the intermediate pipeline stages
    (raw capture, distortion-corrected, isolated, final) -- see the
    commented-out calls in reachy_detection.build_object_point_cloud."""
    if point_cloud.shape[0] == 0:
        print("[WARN] Point cloud is empty, nothing to show")
        return

    fig = plt.figure(f"Point cloud - {class_name}")
    ax = fig.add_subplot(projection="3d")
    xs, ys, zs = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]
    scatter = ax.scatter(xs, ys, zs, c=zs, cmap="viridis", s=4)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(f"{class_name}: {len(point_cloud)} points")
    fig.colorbar(scatter, ax=ax, shrink=0.6, label="z (m)")

    # Equal aspect ratio on all three axes
    ranges = point_cloud.max(axis=0) - point_cloud.min(axis=0)
    half_range = max(ranges.max() / 2.0, 1e-3)
    mid = point_cloud.mean(axis=0)
    ax.set_xlim(mid[0] - half_range, mid[0] + half_range)
    ax.set_ylim(mid[1] - half_range, mid[1] + half_range)
    ax.set_zlim(mid[2] - half_range, mid[2] + half_range)

    _save_fig(fig, f"point_cloud_{class_name}")

    plt.show(block=False)
    plt.pause(0.001)


def show_grasp_plan(geometry, plan) -> None:
    """Non-blocking 3D scatter of the object's point cloud with the planned 
    pre-grasp/grasp/lift EE positions marked on top."""
    point_cloud = geometry.point_cloud
    if point_cloud.shape[0] == 0:
        print("[WARN] Point cloud is empty, nothing to show")
        return

    fig = plt.figure(f"Grasp plan - {geometry.class_name} ({plan.arm_name})")
    ax = fig.add_subplot(projection="3d")

    xs, ys, zs = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]
    ax.scatter(xs, ys, zs, c=zs, cmap="viridis", s=4, label="object point cloud")

    pregrasp_pos = plan.pregrasp_matrix[:3, 3]
    grasp_pos = plan.grasp_matrix[:3, 3]
    lift_pos = plan.lift_matrix[:3, 3]

    ax.plot(*zip(pregrasp_pos, grasp_pos), c="blue", linestyle="--", linewidth=1.5, label="approach path")
    ax.scatter(*pregrasp_pos, c="orange", marker="^", s=80, label="pre-grasp EE")
    ax.scatter(*grasp_pos, c="red", marker="X", s=100, label="grasp EE")
    ax.scatter(*lift_pos, c="green", marker="^", s=80, label="lift EE")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(f"{geometry.class_name}: planned grasp ({plan.arm_name})")
    ax.legend(loc="upper left", fontsize=8)

    # Equal aspect ratio over the point cloud + waypoints
    all_points = np.vstack([point_cloud, [pregrasp_pos, grasp_pos, lift_pos]])
    ranges = all_points.max(axis=0) - all_points.min(axis=0)
    half_range = max(ranges.max() / 2.0, 1e-3)
    mid = all_points.mean(axis=0)
    ax.set_xlim(mid[0] - half_range, mid[0] + half_range)
    ax.set_ylim(mid[1] - half_range, mid[1] + half_range)
    ax.set_zlim(mid[2] - half_range, mid[2] + half_range)

    _save_fig(fig, f"grasp_plan_{geometry.class_name}_{plan.arm_name}")

    plt.show(block=False)
    # Several short pauses, not one: gives the window manager more chances
    # to map/raise the window before cv2.imshow + waitKey resume and starve
    #  matplotlib's event loop. 
    for _ in range(10):
        plt.pause(0.1)
