"""Scoring and demo checks for weld-fitup that run without the simulator: python -m pytest -q test_weld_fitup.py"""

from __future__ import annotations

import math

import numpy as np
import pytest
from yam_arena import station
from yam_arena.demo import ArmPlan

import weld_fitup_task as T
from weld_fitup import design as d

A, B = d.COUPONS
STILL = dict.fromkeys(d.COUPONS, True)


def fitted(opening_mm: float = 2.38, stagger_mm: float = 0.0, **over) -> dict:
    """The fit-up: a on the left facing right, b on the right facing left, the joint along x."""
    off = opening_mm / 2000 + d.SIZE / 2
    p = {A: d.pose((0.32, off), -90.0), B: d.pose((0.32 + stagger_mm / 1000, -off), 90.0)}
    p.update(over)
    return p


def grade(p, still=STILL):
    return d.grade(p, still)[0]


def test_the_coupon_is_the_aws_single_v_detail():
    assert d.SIZE == pytest.approx(0.0762) and d.THICK == pytest.approx(0.0127)
    assert d.ROOT_FACE == pytest.approx(0.003175) and 2 * d.BEVEL_DEG == 60.0
    assert d.BEVEL_RUN == pytest.approx((0.0127 - 0.003175) * math.tan(math.radians(30)))
    assert d.MASS == pytest.approx(0.563, abs=0.002)  # 7850 kg/m^3, the bevel cut away


def test_the_start_scores_0():
    assert grade(d.start_poses()) == 0.0


def test_the_fit_up_scores_1():
    assert d.grade(fitted(), STILL) == (1.0, "fit up")


@pytest.mark.parametrize(("opening_mm", "expected"), [
    (1.5, 0.5), (1.65, 1.0), (2.38, 1.0), (3.1, 1.0), (3.3, 0.5), (0.0, 0.5), (9.5, 0.5), (10.5, 0.0),
])
def test_the_root_opening_is_1_16_to_1_8_inch(opening_mm, expected):
    assert grade(fitted(opening_mm)) == expected


def test_the_opening_counts_at_both_ends_of_the_joint():
    # coupon_b turned 1.5 degrees about its centre: 76 mm of joint opens 2 mm more at one end than the other.
    off = 2.38 / 2000 + d.SIZE / 2
    p = fitted(**{B: d.pose((0.32, -off), 91.5)})
    j = d.joint(p[A], p[B])
    assert j["open_max"] - j["open_min"] == pytest.approx(d.SIZE * math.tan(math.radians(1.5)), abs=1e-4)
    assert j["open_max"] > d.OPENING_MAX and grade(p) == 0.5


@pytest.mark.parametrize(("stagger_mm", "expected"), [(0.0, 1.0), (4.9, 1.0), (-4.9, 1.0), (5.1, 0.5), (19.0, 0.5),
                                                      (21.0, 0.0)])
def test_the_ends_line_up_within_5_mm(stagger_mm, expected):
    assert grade(fitted(stagger_mm=stagger_mm)) == expected


def test_either_coupon_can_be_on_either_side():
    p = fitted()
    swapped = {A: np.r_[p[B][:3], d.yaw_quat(90.0)], B: np.r_[p[A][:3], d.yaw_quat(-90.0)]}
    assert grade(swapped) == 1.0


def test_the_joint_can_run_any_way_on_the_table():
    # The whole fit-up turned 37 degrees and moved: still 1.0. Scoring is relative, from the coupons alone.
    c, s = math.cos(math.radians(37)), math.sin(math.radians(37))
    p = fitted()
    out = {}
    for n in d.COUPONS:
        x, y = p[n][0] - 0.32, p[n][1]
        out[n] = d.pose((0.30 + c * x - s * y, 0.02 + s * x + c * y), d.heading_deg(p[n]) + 37)
    assert grade(out) == 1.0


def test_a_square_edge_in_the_joint_scores_0_25():
    # coupon_b turned the wrong way round: its square back edge faces the joint (a single-bevel joint, not a V).
    off = 2.38 / 2000 + d.SIZE / 2
    assert grade(fitted(**{B: d.pose((0.32, -off), -90.0)})) == 0.25
    # turned a quarter: a square side edge in the joint
    assert grade(fitted(**{B: d.pose((0.32, -off), 0.0)})) == 0.25


def test_bevels_toward_each_other_but_crossed_score_0():
    off = 2.38 / 2000 + d.SIZE / 2 + 0.01
    assert grade(fitted(**{B: d.pose((0.32, -off), 75.0)})) == 0.0


def test_neither_bevel_at_the_joint_scores_0():
    off = 2.38 / 2000 + d.SIZE / 2
    p = {A: d.pose((0.32, off), 90.0), B: d.pose((0.32, -off), -90.0)}  # back to back: square edges meet
    assert grade(p) == 0.0


def test_a_square_edge_far_from_the_other_coupon_scores_0():
    off = 2.38 / 2000 + d.SIZE / 2
    assert grade(fitted(**{B: d.pose((0.32, -off - 0.03), -90.0)})) == 0.0


def test_moving_coupons_score_0():
    assert grade(fitted(), {A: True, B: False}) == 0.0


@pytest.mark.parametrize("bad", ["tilted", "raised", "off"])
def test_a_coupon_not_flat_on_the_table_scores_0(bad):
    p = fitted()
    q = p[B].copy()
    if bad == "tilted":  # propped on edge, 6 degrees
        h = math.radians(6) / 2
        q[3:] = [math.sin(h), 0.0, 0.0, math.cos(h)]
    elif bad == "raised":  # lying on the other coupon's edge
        q[2] += 0.004
    else:
        q[2] = d.TOP - 0.3
    p[B] = q
    assert grade(p) == 0.0


def test_overlapping_coupons_score_0():
    assert grade(fitted(-2.0)) == 0.0


def test_the_bevel_faces_plus_x_in_the_coupon_frame():
    p = d.pose((0.3, 0.0), 0.0)
    a, b = d.root_edge(p)
    assert a[0] == pytest.approx(0.3 + d.SIZE / 2) and b[0] == pytest.approx(0.3 + d.SIZE / 2)
    assert a[2] == pytest.approx(d.BENCH_TOP)


# -- the demo -----------------------------------------------------------------------------------------------------

def test_the_demo_aims_at_the_fit_up():
    # Where each coupon should land once DEMO_LANDING's measured offsets are added back to the demo's aim.
    p = {}
    for n, ((x, y), yaw) in T.demo_targets().items():
        dx, dy, dturn = T.DEMO_LANDING[n]
        p[n] = d.pose((x + dx / 1000, y + dy / 1000), yaw + dturn)
    assert grade(p) == 1.0
    for n in d.COUPONS:
        assert np.allclose(p[n], d.target_poses()[n])


def test_the_demo_aims_unshifted():
    assert T.DEMO_OPENING_SHIFT_MM == 0.0


@pytest.mark.parametrize("arm", ["left"])
def test_the_demo_is_within_reach(arm):
    plan = ArmPlan(T.demo()[arm], station.BASES[arm])
    for t in np.arange(0.0, plan.duration, 0.05):
        plan.action7(t)
    assert plan.worst_ik_mm < 0.5 and plan.worst_ik_deg < 0.1, (plan.worst_ik_mm, plan.worst_ik_deg)


def test_the_demo_grips_on_the_square_edges_with_room_either_side():
    # The fingers close along the grip's y; the grip is turned to the coupon's heading, so they close along the
    # coupon's own y: on the two square edges either side of the bevel, never on the bevel.
    gap = T.GRIP_FINGER / T.FINGER_OPEN * 0.094
    assert gap < d.SIZE - 0.006  # aimed inside both faces: it squeezes
    assert (T.RELEASE_FINGER / T.FINGER_OPEN * 0.094 - d.SIZE) / 2 >= 0.006
    assert (0.094 - d.SIZE) / 2 >= 0.008  # open, 8 mm clear either side on the way down
    w = T.demo()["left"]
    grips = [x for x in w if x.label == "close"]
    puts = [x for x in w if x.label == "set down"]
    tgt = T.demo_targets()
    for g, p, n in zip(grips, puts, d.COUPONS, strict=True):
        assert math.degrees(g.yaw) == pytest.approx(d.START[n][1])
        assert math.degrees(p.yaw) == pytest.approx(tgt[n][1])


def test_coupon_b_is_turned_end_for_end():
    assert d.heading_deg(d.target_poses()["coupon_b"]) - d.START["coupon_b"][1] == pytest.approx(180.0)


def test_the_demo_grips_low_without_touching_the_table():
    assert T.GRIP_Z - T.PAD_TIP_PAST_TCP - d.BENCH_TOP == pytest.approx(0.003)
    assert T.GRIP_Z - T.PAD_TIP_PAST_TCP - d.BENCH_TOP < d.THICK - 0.008  # 9.7 mm of pad on the coupon


def test_the_demo_finishes_well_within_the_run_limit():
    assert sum(w.duration for w in T.demo()["left"]) < 2400 / 20 / 2


def test_the_welding_table_holds_both_starts_and_the_fit_up():
    (bx, by), (sx, sy, _) = d.BENCH_XY, d.BENCH_SIZE
    for p in [*d.start_poses().values(), *d.target_poses().values()]:
        r = d.rot(p[3:])
        for cx in (-1, 1):
            for cy in (-1, 1):
                corner = p[:3] + r @ np.array([cx * d.SIZE / 2, cy * d.SIZE / 2, 0.0])
                assert abs(corner[0] - bx) <= sx / 2 - 0.02 and abs(corner[1] - by) <= sy / 2 - 0.02
