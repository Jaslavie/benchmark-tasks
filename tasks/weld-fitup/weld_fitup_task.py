"""Fit up two bevelled steel coupons for a single-V groove butt weld, on the station table.

A dark steel welding-table top (30 x 32 x 1 cm) is fixed on the left half of the table. On it lie two steel coupons
(1/2 in thick, 3 x 3 in, 0.56 kg each, dark mill scale), loose, at its far side. Each has one edge prepared for
welding: a bright ground 30-degree bevel down to a 1/8 in root face. coupon_a (the left one) has its bevel pointing
away from the arms; coupon_b has its bevel pointing to the right, away from coupon_a.

Fit-up: both coupons flat, their bevels facing each other across the joint so together they make a 60-degree V, a
root opening of 1/16 to 1/8 in (1.6-3.2 mm) at both ends of the joint, and their ends in line within 5 mm.

The coupons are custom bodies (a bevel isn't a box), so they come from the task-local plugin ``weld_fitup``, which
also scores the trial (embodiment ``yam_weld_fitup``). This file holds the fixed welding-table top and the scripted
demo that ``yam_scripted`` plays.
"""

from __future__ import annotations

import math

from yam_arena.constants import FINGER_OPEN, PAD_TIP_PAST_TCP
from yam_arena.demo import Waypoint
from yam_arena.task_file import Box

from weld_fitup import design as d

SCENE = "station"
BENCH_STEEL = (0.10, 0.10, 0.11)

OBJECTS = [Box("welding_table", pos=(*d.BENCH_XY, d.TOP + d.BENCH_SIZE[2] / 2), size=d.BENCH_SIZE,
               color=BENCH_STEEL)]


def success(objects):
    return False  # scored by the embodiment yam_weld_fitup (weld_fitup/embodiment.py), from the coupons' poses


# The demo. The left arm grips each coupon at the middle, the fingers on the two square edges either side of the
# bevel (the grip turned to the coupon's heading), turns it while carrying, and sets it down: coupon_a with its bevel
# facing right, then coupon_b 3/32 in from it, facing left, turned end for end.
PAD_ABOVE = 0.003  # the pad tips 3 mm above the welding table
GRIP_Z = d.BENCH_TOP + PAD_ABOVE + PAD_TIP_PAST_TCP
TRAVEL_Z = d.BENCH_TOP + 0.10
DROP = 0.001  # each coupon is let go 1 mm above the table
GRIP_FINGER = FINGER_OPEN * 68 / 94  # aim the fingers 68 mm apart, 4 mm inside each 76.2 mm face
RELEASE_FINGER = FINGER_OPEN * 90 / 94  # open slowly to 90 mm, 7 mm clear each side
# Where each coupon lands against where it is let go, in mm (x, y) and degrees of turn, measured on the official
# stack; the demo lets go that much the other way. First official trace (2026-11-08, aims uncorrected): the arm, a
# 0.56 kg coupon in hand, stops 6-7 mm short of the aim and each coupon comes out turned 1.5 degrees anticlockwise;
# coupon_b also lands 3.3 mm to the left, which closed the root opening to 0.4-0.5 mm (credit 0.5).
# With those offsets the next trace scored 1.0 (opening 2.60-2.75 mm, ends 0.22 mm apart), coupon_b 0.31 mm further
# from coupon_a than aimed; its y offset takes that back too.
DEMO_LANDING = {"coupon_a": (-7.12, 1.37, 1.45), "coupon_b": (-6.22, 2.98, 1.53)}
DEMO_OPENING_SHIFT_MM = 0.0  # coupon_b's aim moved this far away from coupon_a (the opening wider), for margin runs
# Margin runs (2026-11-08, traced on the official stack with the plugin): shifted 0, +0.5 and -0.5 mm, all three 1.0,
# openings 2.44-2.57, 2.94-3.07 and 1.94-2.06 mm (limits 1.6-3.2), ends 0.35-0.39 mm apart, the coupons parallel within
# 0.15 degree; coupon_a landed within 0.01 mm of the same place each time.


def _move(pick, pick_yaw, place, place_yaw, wait_s: float = 0.0) -> list[Waypoint]:
    (px, py), (qx, qy) = pick, place
    pz, qz = GRIP_Z, GRIP_Z + DROP
    a, b = math.radians(pick_yaw), math.radians(place_yaw)
    out = [Waypoint(wait_s, (px, py, TRAVEL_Z), FINGER_OPEN, "wait", a)] if wait_s else []
    return out + [
        Waypoint(2.0, (px, py, TRAVEL_Z), FINGER_OPEN, "to the coupon", a),
        Waypoint(1.5, (px, py, pz), FINGER_OPEN, "descend", a),
        Waypoint(0.5, (px, py, pz), FINGER_OPEN, "steady", a),
        Waypoint(0.8, (px, py, pz), GRIP_FINGER, "close", a),
        Waypoint(0.5, (px, py, pz), GRIP_FINGER, "settle the grip", a),
        Waypoint(1.5, (px, py, TRAVEL_Z), GRIP_FINGER, "lift", a),
        Waypoint(4.0, (qx, qy, TRAVEL_Z), GRIP_FINGER, "carry and turn", b),
        Waypoint(2.0, (qx, qy, TRAVEL_Z), GRIP_FINGER, "hover", b),
        Waypoint(3.0, (qx, qy, qz + 0.02), GRIP_FINGER, "lower", b),
        Waypoint(2.0, (qx, qy, qz), GRIP_FINGER, "set down", b),
        Waypoint(0.3, (qx, qy, qz), GRIP_FINGER, "hold", b),
        Waypoint(1.5, (qx, qy, qz), RELEASE_FINGER, "release", b),
        Waypoint(0.3, (qx, qy, qz), RELEASE_FINGER, "settle", b),
        Waypoint(1.5, (qx, qy, TRAVEL_Z), RELEASE_FINGER, "retreat", b),
    ]


def demo_targets() -> dict:
    """Where the demo lets go of each coupon: (x, y) and heading (degrees), allowing for DEMO_LANDING."""
    out = {}
    for n, p in d.target_poses().items():
        dx, dy, dturn = DEMO_LANDING[n]
        shift = -DEMO_OPENING_SHIFT_MM / 1000 if n == "coupon_b" else 0.0  # coupon_b lies on the joint's right (-y)
        out[n] = ((p[0] - dx / 1000, p[1] - dy / 1000 + shift), d.heading_deg(p) - dturn)
    return out


def demo():
    """The left arm sets coupon_a, then coupon_b beside it; the right arm stays at the ready pose."""
    tgt = demo_targets()
    (a_xy, a_yaw), (b_xy, b_yaw) = d.START["coupon_a"], d.START["coupon_b"]
    return {"left": _move(a_xy, a_yaw, *tgt["coupon_a"]) + _move(b_xy, b_yaw, *tgt["coupon_b"])}
