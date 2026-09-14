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
GRASP_LIFT_M = 0.10

ARM_GOTO_DURATION_S = 3.0 

# How far outside the object's near surface (beyond its radius) the
# commanded EE position sits, per arm -- the left arm has been observed to
# fall consistently a bit short of the commanded pose, dropping the object,
# so it gets a negative margin (reaches past the surface) to compensate
GRASP_APPROACH_MARGIN_DX_M = 0.0  # r_arm
GRASP_APPROACH_MARGIN_SX_M = -0.02  # l_arm TODO: find right value
_GRASP_APPROACH_MARGIN_BY_ARM = {"r_arm": GRASP_APPROACH_MARGIN_DX_M, "l_arm": GRASP_APPROACH_MARGIN_SX_M}

# Empirical correction for plan_place: releasing exactly at the original
# grasp height lands a bit lower than intended (object slip in the gripper
# during transit and/or arm sag under the object's weight) -- release this
# much higher instead. TODO: find the right value / whether it's per-arm too
PLACE_HEIGHT_MARGIN_M = 0.009

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
def _grasp_targets(geometry: ObjectGeometry) -> Optional[tuple]:
    """(mid_position, radius, table_normal) -- where along the object to
    close the gripper and how far out its surface is -- or None if the
    object's pose couldn't be estimated or it's too wide for the gripper."""
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
    return mid_position, geometry.width_m / 2.0, table_normal


def _plan_grasp_for_arm(
    reachy: ReachySDK, arm_name: str, mid_position: npt.NDArray[np.float64], radius: float,
    table_normal: npt.NDArray[np.float64], count: int = APPROACH_CANDIDATE_COUNT,
) -> Optional[GraspPlan]:
    """Pre-grasp + grasp + lift poses for one specific arm: the first of
    `count` horizontal approach directions whose pregrasp and grasp poses IK
    both accept, or None if that arm can't reach the object on any of them."""
    arm = getattr(reachy, arm_name, None)
    if arm is None:
        return None
    margin = _GRASP_APPROACH_MARGIN_BY_ARM[arm_name]

    for approach in _approach_candidates(mid_position, table_normal, count=count):
        grasp_position = mid_position - (radius + margin) * approach
        pregrasp_position = grasp_position - approach * PREGRASP_STANDOFF_M
        rotation = _orientation_from_approach(approach, _side_grasp_closing_axis(approach, table_normal))
        try:
            arm.inverse_kinematics(_pose_matrix(rotation, pregrasp_position))
            arm.inverse_kinematics(_pose_matrix(rotation, grasp_position))
        except ValueError:
            continue

        return GraspPlan(
            arm_name=arm_name,
            pregrasp_matrix=_pose_matrix(rotation, pregrasp_position),
            grasp_matrix=_pose_matrix(rotation, grasp_position),
            lift_matrix=_pose_matrix(rotation, grasp_position + table_normal * GRASP_LIFT_M),
        )
    return None


def plan_grasps_by_arm(
    reachy: ReachySDK, geometry: ObjectGeometry, count: int = APPROACH_CANDIDATE_COUNT,
) -> dict:
    """{arm_name: GraspPlan} for every arm that can actually pick `geometry`
    up -- empty if neither can (or its pose couldn't be estimated / it's too
    wide for the gripper). Near-side arm first.

    The placement grid needs all of them, not a single winner: which arm
    ends up doing the job depends on where the user chooses to put the
    object down, which isn't known yet at grasp-planning time. An object
    only one arm can pick up filters the grid to that arm's reach; one both
    can pick up shows the union, and the chosen cell decides the arm."""
    targets = _grasp_targets(geometry)
    if targets is None:
        return {}
    mid_position, radius, table_normal = targets

    near_side = "r_arm" if mid_position[1] < 0 else "l_arm"
    far_side = "l_arm" if near_side == "r_arm" else "r_arm"

    plans = {}
    for arm_name in (near_side, far_side):
        plan = _plan_grasp_for_arm(reachy, arm_name, mid_position, radius, table_normal, count=count)
        if plan is not None:
            plans[arm_name] = plan

    print(f"[plan_grasp] can be picked up by: {', '.join(plans) if plans else 'neither arm'} "
          f"({count} approach directions x 2 arms tried)")
    return plans


def plan_grasp(
    reachy: ReachySDK, geometry: ObjectGeometry, arm_name: Optional[str] = None,
) -> Optional[GraspPlan]:
    """Pre-grasp + grasp + lift end-effector poses for `geometry`, or None if
    its pose couldn't be estimated, it's too wide for the gripper, or no
    approach direction is reachable.

    Approaches horizontally at GRASP_HEIGHT_FRACTION of the object's height.
    arm_name pins which arm to plan for -- which is what the placement grid
    decides, see plan_grasps_by_arm; without it the near-side arm is tried
    first, then the far-side one."""
    targets = _grasp_targets(geometry)
    if targets is None:
        return None
    mid_position, radius, table_normal = targets

    if arm_name is not None:
        candidates = (arm_name,)
    else:
        near_side = "r_arm" if mid_position[1] < 0 else "l_arm"
        candidates = (near_side, "l_arm" if near_side == "r_arm" else "r_arm")

    for candidate in candidates:
        plan = _plan_grasp_for_arm(reachy, candidate, mid_position, radius, table_normal)
        if plan is not None:
            return plan
    return None


def plan_place(
    reachy: ReachySDK, plan: GraspPlan,
    table_normal: Optional[npt.NDArray[np.float64]], target_point: npt.NDArray[np.float64],
    candidate_count: int = APPROACH_CANDIDATE_COUNT,
) -> Optional[GraspPlan]:
    """Place GraspPlan for the object plan already grasped: released at the
    same height it was grasped from (plan.grasp_matrix's height along
    table_normal), directly above target_point's horizontal position --
    mirroring the lift exactly (climb X, carry across, descend the same X)
    instead of computing an absolute release height from the target surface
    + object height, which repeatedly proved fragile in practice (wrong
    direction, IK solver flakiness) and is redundant anyway: moving
    perpendicular to table_normal is by definition horizontal, so wherever
    target_point sits along table_normal doesn't matter for a level table.

    Searches orientations the way plan_grasp does, rather than inheriting
    the grasp's: the orientation the object is currently held in is tried
    first (so a wrist that doesn't need to move doesn't move), then
    `candidate_count` horizontal approaches around the circle. Every
    candidate keeps the gripper's local X up (_side_grasp_closing_axis), so
    switching between them only yaws the held object about its own vertical
    axis, never tips it -- and the retreat follows whichever orientation
    won, so it always withdraws along the gripper's own axis, clear of the
    object's sides.

    Returns three waypoints, reusing GraspPlan's fields:
      - lift_matrix: "transit" -- above target_point's horizontal position,
        at plan.lift_matrix's height, so the carry ends at the same height
        the lift ended at.
      - grasp_matrix: "place" -- same horizontal position, dropped back down
        by the same distance plan.lift_matrix climbed above plan.grasp_matrix
        (where the gripper opens), plus PLACE_HEIGHT_MARGIN_M -- releasing
        exactly at the original grasp height was empirically landing a bit
        lower than intended (slip in the gripper during transit and/or arm
        sag under the object's weight).
      - pregrasp_matrix: "retreat" -- PREGRASP_STANDOFF_M back out from
        there once the gripper has opened, along the winning orientation's
        own forward axis rather than up along table_normal or any direction
        picked independently of orientation: that's the only direction
        guaranteed clear of the object's sides for the still-open fingers to
        withdraw along without dragging/knocking it.

    Returns None (silently -- _build_place_grid calls this once per grid
    cell) if no orientation puts all three poses within reach of
    plan.arm_name, so the caller can ask the user for a different point."""
    arm = getattr(reachy, plan.arm_name, None)
    if arm is None:
        return None

    normal = table_normal if table_normal is not None else DEFAULT_TABLE_NORMAL
    normal = normal / np.linalg.norm(normal)

    target_inplane = target_point - normal * np.dot(target_point, normal)
    lift_height = np.dot(plan.lift_matrix[:3, 3], normal)
    grasp_height = np.dot(plan.grasp_matrix[:3, 3], normal) + PLACE_HEIGHT_MARGIN_M

    transit_position = target_inplane + normal * lift_height
    place_position = target_inplane + normal * grasp_height

    # -grasp_matrix's local z is the approach it was built from
    # (_orientation_from_approach), so this re-derives the current orientation
    # as candidate #0 rather than special-casing it.
    held_approach = -plan.grasp_matrix[:3, 2]
    for approach in [held_approach] + _approach_candidates(place_position, normal, count=candidate_count):
        rotation = _orientation_from_approach(approach, _side_grasp_closing_axis(approach, normal))
        pregrasp_position = place_position - approach * PREGRASP_STANDOFF_M
        # place first: a cell well outside the workspace fails it for every
        # candidate, so it bails after one IK call each instead of three
        place_matrix = _pose_matrix(rotation, place_position)
        transit_matrix = _pose_matrix(rotation, transit_position)
        pregrasp_matrix = _pose_matrix(rotation, pregrasp_position)
        try:
            arm.inverse_kinematics(place_matrix)
            arm.inverse_kinematics(transit_matrix)
            arm.inverse_kinematics(pregrasp_matrix)
        except ValueError:
            continue

        return GraspPlan(
            arm_name=plan.arm_name,
            pregrasp_matrix=pregrasp_matrix,  # retreat, after the gripper opens
            grasp_matrix=place_matrix,        # where the gripper opens
            lift_matrix=transit_matrix,       # constant-height transit waypoint before descending
        )

    return None


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


def execute_place(reachy: ReachySDK, place_plan: GraspPlan, duration: float = ARM_GOTO_DURATION_S) -> bool:
    """Drives place_plan.arm_name -- already holding the object after
    execute_grasp -- from wherever it currently is (the lift pose, near the
    original pickup spot) through: over to the target's XY at the same
    height it was lifted to (place_plan.lift_matrix, plan_place's "transit"
    waypoint), straight down to the target, dropping by the same distance
    the lift climbed (place_plan.grasp_matrix), open the gripper, straight
    back out PREGRASP_STANDOFF_M along the gripper's own forward axis
    (place_plan.pregrasp_matrix -- withdrawing along the same line it
    approached on, clear of the object's sides, not lifting away over it or
    pulling off at some other angle that could drag/knock it).

    The transit leg interpolates in joint space, so the wrist is free to
    rotate on the way (plan_place may pick a different orientation than the
    object is currently held in) as long as it arrives at the chosen pose --
    a straight cartesian line there can't always accommodate both the
    translation and the reorientation. The two legs near the object, where
    the path itself matters, stay cartesian: a clean vertical drop and a
    clean straight withdrawal. The head tracks the end-effector throughout.

    Returns False without moving if the arm/gripper isn't available or any
    pose is unreachable from the arm's current joints; returns False
    (partway through, object already lowered/released) if a goto raises
    RuntimeError mid-sequence -- both cases are reported, never left to
    propagate and crash the caller."""
    arm: Optional[Arm] = getattr(reachy, place_plan.arm_name, None)
    if arm is None or arm.gripper is None:
        print(f"[ERROR] {place_plan.arm_name} or its gripper is not available -- place not executed")
        return False

    for name, matrix in (
        ("transit", place_plan.lift_matrix), ("place", place_plan.grasp_matrix), ("retreat", place_plan.pregrasp_matrix),
    ):
        try:
            arm.inverse_kinematics(matrix)
        except ValueError:
            print(f"[ERROR] {name} pose unreachable for {place_plan.arm_name} -- place not executed")
            return False

    try:
        print(f"[{place_plan.arm_name}] carrying to above the placement point...")
        _look_at_matrix(reachy, place_plan.lift_matrix, duration)
        arm.goto(place_plan.lift_matrix, duration=duration, wait=True)
        print(f"[{place_plan.arm_name}] lowering to place...")
        _look_at_matrix(reachy, place_plan.grasp_matrix, duration)
        arm.goto(place_plan.grasp_matrix, duration=duration, interpolation_space="cartesian_space", wait=True)
        print(f"[{place_plan.arm_name}] opening gripper...")
        arm.gripper.open()
        print(f"[{place_plan.arm_name}] retreating...")
        _look_at_matrix(reachy, place_plan.pregrasp_matrix, duration)
        arm.goto(place_plan.pregrasp_matrix, duration=duration, interpolation_space="cartesian_space", wait=True)
    except RuntimeError as exc:
        print(f"[ERROR] {place_plan.arm_name} place aborted: {exc}")
        return False

    print(f"[{place_plan.arm_name}] place sequence done")
    return True
