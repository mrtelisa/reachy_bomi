#!/usr/bin/env python3
"""
Library module-- Grasp planning + execution: from reachy_detection.py's point
cloud (position, axes, width/height, table normal) for a confirmed object,
computes pre-grasp/grasp/lift end-effector poses and drives the arm through
them.

World pose matrix convention: X forward, Y left, Z up; verified at
https://docs.pollen-robotics.com/developing-with-reachy-2/basics/4-use-arm-kinematics/.
"""

from typing import List, NamedTuple, Optional

import numpy as np
import numpy.typing as npt
from reachy2_sdk import ReachySDK
from reachy2_sdk.parts.arm import Arm

GRIPPER_MAX_OPENING_M = 0.1 # [m], parallel gripper's max fingertip opening

# How far back from the grasp point, opposite the approach direction, the
# pre-grasp waypoint sits
PREGRASP_STANDOFF_M = 0.10

# How far straight up the arm lifts once the gripper has closed 
GRASP_LIFT_M = 0.20

ARM_GOTO_DURATION_S = 5.0 #[s]

# How far outside the object's near surface (beyond its radius) the
# commanded EE position sits, per arm -- the left arm has been observed to
# fall consistently a bit short of the commanded pose, dropping the object,
# so it gets a negative margin (reaches past the surface) to compensate
GRASP_APPROACH_MARGIN_DX_M = 0.0  # r_arm
GRASP_APPROACH_MARGIN_SX_M = -0.04  # l_arm TODO: find right value
_GRASP_APPROACH_MARGIN_BY_ARM = {"r_arm": GRASP_APPROACH_MARGIN_DX_M, "l_arm": GRASP_APPROACH_MARGIN_SX_M}

# Fallback "up" direction (Reachy world frame) when the table plane fit fails 
DEFAULT_TABLE_NORMAL: npt.NDArray[np.float64] = np.array([0.0, 0.0, 1.0])

# How many horizontal approach directions _approach_candidates spreads
# across the full circle around the object when searching for one IK accepts
APPROACH_CANDIDATE_COUNT = 16

# Grasp height as a fraction of the object's height, measured from its
# base (0 = bottom, 1 = top)
GRASP_HEIGHT_FRACTION = 3 / 5

# How many horizontal approach directions is_roughly_reachable tries
QUICK_REACHABILITY_CANDIDATE_COUNT = 8


class ObjectGeometry(NamedTuple):
    """Everything reachy_detection.py's point cloud pipeline knows about a
    confirmed object, in Reachy's world frame (meters). centroid/axes/
    table_normal are None if too few points survived isolation to fit them"""

    class_name: str
    shape: str
    width_m: float
    height_m: float
    centroid: Optional[npt.NDArray[np.float64]]
    axes: Optional[npt.NDArray[np.float64]]  # 3x3, columns = principal axes, sorted long -> short extent
    table_normal: Optional[npt.NDArray[np.float64]]
    point_cloud: npt.NDArray[np.float64]  # (N, 3), isolated object points, for show_grasp_plan


class GraspPlan(NamedTuple):
    arm_name: str  # "r_arm" or "l_arm"
    pregrasp_matrix: npt.NDArray[np.float64]  # 4x4, Reachy world frame
    grasp_matrix: npt.NDArray[np.float64]  # 4x4, Reachy world frame
    lift_matrix: npt.NDArray[np.float64]  # 4x4, Reachy world frame


def _pose_matrix(rotation: npt.NDArray[np.float64], position: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = position
    return matrix


# --- Approach and grasping parameters ---
def _orientation_from_approach(
    approach: npt.NDArray[np.float64], closing_axis: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """3x3 rotation for an end-effector reaching along world-frame `approach`
    with its gripper opening/closing along world-frame `closing_axis`."""
    local_z = -approach / np.linalg.norm(approach)

    local_y = closing_axis - local_z * np.dot(closing_axis, local_z)
    local_y_norm = np.linalg.norm(local_y)
    if local_y_norm < 1e-6:
        # closing_axis was parallel to approach 
        fallback = np.eye(3)[np.argmin(np.abs(approach))]
        local_y = fallback - local_z * np.dot(fallback, local_z)
        local_y_norm = np.linalg.norm(local_y)
    local_y /= local_y_norm

    local_x = np.cross(local_y, local_z)
    return np.column_stack([local_x, local_y, local_z])


def _side_grasp_closing_axis(
    approach: npt.NDArray[np.float64], up: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """The one closing axis, for a horizontal `approach`, that keeps the
    gripper's local X axis pointing straight up (parallel to the table,
    aligned with the object's own axis) -- derived directly as cross(-approach, up)."""
    closing_axis = np.cross(-approach, up)
    return closing_axis / np.linalg.norm(closing_axis)


def _approach_candidates(
    mid_position: npt.NDArray[np.float64], table_normal: npt.NDArray[np.float64], count: int = APPROACH_CANDIDATE_COUNT,
) -> List[npt.NDArray[np.float64]]:
    """`count` horizontal approach directions spanning the full circle
    around the object."""
    radial_xy = mid_position[:2]
    radial_norm = np.linalg.norm(radial_xy)
    if radial_norm < 1e-6:
        default = np.array([1.0, 0.0, 0.0])  # object directly below the origin -- arbitrary but valid 
    else:
        default = np.array([radial_xy[0] / radial_norm, radial_xy[1] / radial_norm, 0.0])

    up = table_normal / np.linalg.norm(table_normal)
    perp = np.cross(up, default)
    perp_norm = np.linalg.norm(perp)
    if perp_norm < 1e-6:
        # table_normal isn't close to vertical (a bad table-plane fit)
        # fall back to an arbitrary horizontal-ish axis.
        perp = np.cross(up, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(up, np.array([0.0, 1.0, 0.0]))
    perp /= np.linalg.norm(perp)

    return [default * np.cos(theta) + perp * np.sin(theta) for theta in np.linspace(0, 2 * np.pi, count, endpoint=False)]


def _grasp_height_position(
    geometry: ObjectGeometry, table_normal: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """geometry.centroid recentered along the object's long axis to
    GRASP_HEIGHT_FRACTION of its observed height range, measured from the
    base."""
    long_axis = geometry.axes[:, 0]
    offsets = (geometry.point_cloud - geometry.centroid) @ long_axis
    base_offset, top_offset = (
        (float(offsets.min()), float(offsets.max()))
        if np.dot(long_axis, table_normal) >= 0
        else (float(offsets.max()), float(offsets.min()))
    )
    target_offset = base_offset + GRASP_HEIGHT_FRACTION * (top_offset - base_offset)
    return geometry.centroid + target_offset * long_axis


# --- Cheap reachability pre-filter ---
def is_roughly_reachable(
    reachy: ReachySDK, position: npt.NDArray[np.float64], count: int = QUICK_REACHABILITY_CANDIDATE_COUNT,
) -> bool:
    """Cheap reachability pre-filter. Ignores the object's actual radius/orientation 
    (unknown at detection time) and checks only the pregrasp position, not grasp too.
    True if IK accepts a horizontal-approach pregrasp pose, from either arm, for at 
    least one of `count` candidate directions."""
    near_side = "r_arm" if position[1] < 0 else "l_arm"
    far_side = "l_arm" if near_side == "r_arm" else "r_arm"

    for arm_name in (near_side, far_side):
        arm = getattr(reachy, arm_name, None)
        if arm is None:
            continue
        for approach in _approach_candidates(position, DEFAULT_TABLE_NORMAL, count=count):
            pregrasp_position = position - approach * PREGRASP_STANDOFF_M
            rotation = _orientation_from_approach(approach, _side_grasp_closing_axis(approach, DEFAULT_TABLE_NORMAL))
            try:
                arm.inverse_kinematics(_pose_matrix(rotation, pregrasp_position))
                return True
            except ValueError:
                continue
    return False


# --- Grasp planning ---
def plan_grasp(reachy: ReachySDK, geometry: ObjectGeometry) -> Optional[GraspPlan]:
    """Pre-grasp + grasp + lift end-effector poses for `geometry`, or None
    if its pose couldn't be estimated or it's too wide for the gripper.

    Approaches horizontally at GRASP_HEIGHT_FRACTION of the object's
    height. For each arm, searches _approach_candidates for a direction IK accepts 
    for both pregrasp and grasp. Falls back to the near-side arm at the default 
    direction if nothing is found, so a plan is always returned to inspect even if 
    execute_grasp will then refuse to move."""
    if geometry.centroid is None or geometry.axes is None:
        return None
    if not (0 < geometry.width_m <= GRIPPER_MAX_OPENING_M):
        return None

    table_normal = geometry.table_normal if geometry.table_normal is not None else DEFAULT_TABLE_NORMAL
    table_normal = table_normal / np.linalg.norm(table_normal)

    mid_position = (
        geometry.centroid if geometry.shape == "sphere"
        else _grasp_height_position(geometry, table_normal)
    )
    radius = geometry.width_m / 2.0

    def _pregrasp_grasp_rotation_for(approach: npt.NDArray[np.float64], arm_name: str):
        margin = _GRASP_APPROACH_MARGIN_BY_ARM[arm_name]
        grasp_position = mid_position - (radius + margin) * approach
        pregrasp_position = grasp_position - approach * PREGRASP_STANDOFF_M
        rotation = _orientation_from_approach(approach, _side_grasp_closing_axis(approach, table_normal))
        return pregrasp_position, grasp_position, rotation

    near_side = "r_arm" if mid_position[1] < 0 else "l_arm"
    far_side = "l_arm" if near_side == "r_arm" else "r_arm"

    # Fallback if nothing below is confirmed reachable: the single most
    # direct line, near-side arm
    arm_name = near_side
    approach = _approach_candidates(mid_position, table_normal)[0]
    pregrasp_position, grasp_position, rotation = _pregrasp_grasp_rotation_for(approach, arm_name)

    found = False
    for candidate_arm_name in (near_side, far_side):
        arm = getattr(reachy, candidate_arm_name, None)
        if arm is None:
            continue
        for candidate_approach in _approach_candidates(mid_position, table_normal):
            candidate_pregrasp_position, candidate_grasp_position, candidate_rotation = \
                _pregrasp_grasp_rotation_for(candidate_approach, candidate_arm_name)
            try:
                arm.inverse_kinematics(_pose_matrix(candidate_rotation, candidate_pregrasp_position))
                arm.inverse_kinematics(_pose_matrix(candidate_rotation, candidate_grasp_position))
            except ValueError:
                continue
            arm_name = candidate_arm_name
            approach = candidate_approach
            rotation = candidate_rotation
            pregrasp_position, grasp_position = candidate_pregrasp_position, candidate_grasp_position
            found = True
            break
        else:
            continue
        break

    lift_position = grasp_position + table_normal * GRASP_LIFT_M

    if found:
        print(f"[plan_grasp] found a reachable approach line for {arm_name} "
              f"({APPROACH_CANDIDATE_COUNT} directions x 2 arms tried)")
    else:
        print(f"[plan_grasp] WARNING: no approach line reachable on either arm -- "
              f"falling back to {arm_name} at the default line, UNVALIDATED.")

    return GraspPlan(
        arm_name=arm_name,
        pregrasp_matrix=_pose_matrix(rotation, pregrasp_position),
        grasp_matrix=_pose_matrix(rotation, grasp_position),
        lift_matrix=_pose_matrix(rotation, lift_position),
    )


def plan_place(
    reachy: ReachySDK, plan: GraspPlan, object_height_m: float,
    table_normal: Optional[npt.NDArray[np.float64]], target_point: npt.NDArray[np.float64],
) -> Optional[GraspPlan]:
    """Place GraspPlan for the object plan already grasped, releasing it so
    its base lands on target_point (e.g. a BoMI-dwelled point inside a box)
    while it's held at GRASP_HEIGHT_FRACTION of its height -- the same
    convention _grasp_height_position used to pick the original grasp point,
    just aimed at a new surface instead of the one it was picked up from.

    Keeps plan's arm and orientation unchanged (the gripper doesn't need to
    reorient to place what it's already holding); the approach direction for
    the pregrasp standoff is recovered from plan.grasp_matrix's rotation.
    Returns None if either pose is unreachable for plan.arm_name, so the
    caller can ask the user to dwell on a different point instead."""
    arm = getattr(reachy, plan.arm_name, None)
    if arm is None:
        return None

    normal = table_normal if table_normal is not None else DEFAULT_TABLE_NORMAL
    normal = normal / np.linalg.norm(normal)

    rotation = plan.grasp_matrix[:3, :3]
    approach = -rotation[:, 2]  # inverse of _orientation_from_approach's local_z = -approach/||approach||

    place_position = target_point + normal * (GRASP_HEIGHT_FRACTION * object_height_m)
    pregrasp_position = place_position + approach * PREGRASP_STANDOFF_M

    pregrasp_matrix = _pose_matrix(rotation, pregrasp_position)
    place_matrix = _pose_matrix(rotation, place_position)
    for name, matrix in (("pre-place", pregrasp_matrix), ("place", place_matrix)):
        try:
            arm.inverse_kinematics(matrix)
        except ValueError:
            print(f"[plan_place] {name} pose unreachable for {plan.arm_name}")
            return None

    return GraspPlan(
        arm_name=plan.arm_name,
        pregrasp_matrix=pregrasp_matrix,
        grasp_matrix=place_matrix,
        lift_matrix=pregrasp_matrix,  # unused by place_back; kept only to satisfy GraspPlan's shape
    )


def _look_at_matrix(reachy: ReachySDK, matrix: npt.NDArray[np.float64], duration: float) -> None:
    """Turns the head to look at a pose matrix's position, non-blocking."""
    if reachy.head is not None:
        reachy.head.look_at(*matrix[:3, 3], duration=duration, wait=False)


def execute_grasp(reachy: ReachySDK, plan: GraspPlan, duration: float = ARM_GOTO_DURATION_S) -> bool:
    """Drives plan.arm_name through open -> pregrasp -> grasp -> close ->
    lift, the head turning to watch the end-effector at each arm.
    Returns False without moving if the arm/gripper isn't available or 
    any pose is unreachable from the arm's current joints."""
    arm: Optional[Arm] = getattr(reachy, plan.arm_name)
    if arm is None or arm.gripper is None:
        print(f"[ERROR] {plan.arm_name} or its gripper is not available -- grasp not executed")
        return False

    for name, matrix in (("pregrasp", plan.pregrasp_matrix), ("grasp", plan.grasp_matrix), ("lift", plan.lift_matrix)):
        try:
            arm.inverse_kinematics(matrix)
        except ValueError:
            print(f"[ERROR] {name} pose unreachable for {plan.arm_name} -- grasp not executed")
            return False

    try:
        print(f"[{plan.arm_name}] opening gripper...")
        arm.gripper.open()
        print(f"[{plan.arm_name}] moving to pregrasp...")
        _look_at_matrix(reachy, plan.pregrasp_matrix, duration)
        arm.goto(plan.pregrasp_matrix, duration=duration, wait=True)
        print(f"[{plan.arm_name}] moving to grasp...")
        _look_at_matrix(reachy, plan.grasp_matrix, duration)
        arm.goto(plan.grasp_matrix, duration=duration, interpolation_space="cartesian_space", wait=True)
        print(f"[{plan.arm_name}] closing gripper...")
        arm.gripper.close()
        print(f"[{plan.arm_name}] moving to lift...")
        _look_at_matrix(reachy, plan.lift_matrix, duration)
        arm.goto(plan.lift_matrix, duration=duration, wait=True)
    except RuntimeError as exc:
        print(f"[ERROR] {plan.arm_name} grasp aborted: {exc}")
        return False

    print(f"[{plan.arm_name}] grasp sequence done")
    return True


def place_back(reachy: ReachySDK, plan: GraspPlan, duration: float = ARM_GOTO_DURATION_S) -> bool:
    """Reverses execute_grasp: moves plan.arm_name from lift back down to
    plan.grasp_matrix, opens the gripper, then retreats back out to
    plan.pregrasp_matrix. Returns False without moving if the arm/gripper 
    isn't available or either pose is unreachable from the arm's current joints."""
    arm: Optional[Arm] = getattr(reachy, plan.arm_name)
    if arm is None or arm.gripper is None:
        print(f"[ERROR] {plan.arm_name} or its gripper is not available -- can't place back")
        return False

    for name, matrix in (("grasp", plan.grasp_matrix), ("pregrasp", plan.pregrasp_matrix)):
        try:
            arm.inverse_kinematics(matrix)
        except ValueError:
            print(f"[ERROR] {name} pose unreachable for {plan.arm_name} -- can't place back")
            return False

    try:
        print(f"[{plan.arm_name}] placing back down...")
        _look_at_matrix(reachy, plan.grasp_matrix, duration)
        arm.goto(plan.grasp_matrix, duration=duration, interpolation_space="cartesian_space", wait=True)
        print(f"[{plan.arm_name}] opening gripper...")
        arm.gripper.open()
        print(f"[{plan.arm_name}] retreating to pregrasp...")
        _look_at_matrix(reachy, plan.pregrasp_matrix, duration)
        arm.goto(plan.pregrasp_matrix, duration=duration, interpolation_space="cartesian_space", wait=True)
    except RuntimeError as exc:
        print(f"[ERROR] {plan.arm_name} place-back aborted: {exc}")
        return False

    print(f"[{plan.arm_name}] placed back down")
    return True
