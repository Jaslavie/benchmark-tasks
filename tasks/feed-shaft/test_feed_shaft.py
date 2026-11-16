"""Checks of feed_shaft.py's scoring and demo, without the simulator. Needs yam_arena (the official stack):

    python -m pytest -q test_feed_shaft.py
"""

import math
from pathlib import Path

import numpy as np
import pytest

from yam_arena import station
from yam_arena.constants import FINGER_OPEN, PAD_TIP_PAST_TCP
from yam_arena.demo import ArmPlan
from yam_arena.task_file import ObjectState, load

TASK = load(Path(__file__).with_name("feed_shaft.py"))
M = TASK.module
A, B = M.PLATE_Y["plate_a"], M.PLATE_Y["plate_b"]
MIDDLE = (A + B) / 2


def _quat(turn_deg, pitch_deg=0.0):
    """A turn about the vertical times a tilt about world x (the shaft's far end rising for a positive pitch)."""
    s, t = math.radians(turn_deg) / 2, math.radians(pitch_deg) / 2
    return np.array([math.sin(t) * math.cos(s), -math.sin(t) * math.sin(s), math.cos(t) * math.sin(s),
                     math.cos(t) * math.cos(s)])


def shaft_state(centre, turn_deg=0.0, pitch_deg=0.0, moving=False, spinning=False):
    q = _quat(turn_deg, pitch_deg)
    half = np.array([M.BAR, M.SHAFT_LEN, M.BAR]) / 2
    corners = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]) * half @ M._rot(q).T
    world = corners + np.asarray(centre, dtype=float)
    return ObjectState(pos=np.asarray(centre, dtype=float), quat_xyzw=q,
                       lin_vel=np.array([0.0, 0.05, 0.0]) if moving else np.zeros(3),
                       ang_vel=np.array([0.0, 0.0, 0.6]) if spinning else np.zeros(3),
                       low=world.min(axis=0), high=world.max(axis=0))


def on_line(y, dx=0.0, dz=0.0, **kw):
    """The shaft along the bores' line with its middle at ``y``, moved ``dx`` forward and ``dz`` up."""
    return {"shaft": shaft_state((M.AXIS_X + dx, y, M.AXIS_Z + dz), **kw)}


def test_the_parts_are_wcp_sizes():
    assert M.BAR == pytest.approx(0.0127) and M.SHAFT_LEN == pytest.approx(0.254)
    assert M.BORE == pytest.approx(0.0147) and M.BEARING_OD == pytest.approx(0.028575) and M.BEARING_W == pytest.approx(0.00795, abs=1e-6)
    assert (A - M.PLATE_T / 2) - (B + M.PLATE_T / 2) == pytest.approx(8 * 0.0254)  # the 8" standoffs


def test_the_bores_leave_a_millimetre_each_side():
    assert (M.BORE - M.BAR) / 2 == pytest.approx(0.001)


def test_each_bore_is_an_opening_in_a_closed_frame():
    for plate in M.PLATE_Y:
        parts = [b for b in M.FIXED if b.name.startswith(f"{plate}_bearing")]
        assert len(parts) == 4
        lows = [(np.array(b.pos) - np.array(b.size) / 2) for b in parts]
        highs = [(np.array(b.pos) + np.array(b.size) / 2) for b in parts]
        for lo, hi in zip(lows, highs, strict=True):  # no part inside the bore
            inside_x = lo[0] < M.AXIS_X + M.BORE / 2 - 1e-9 and hi[0] > M.AXIS_X - M.BORE / 2 + 1e-9
            inside_z = lo[2] < M.AXIS_Z + M.BORE / 2 - 1e-9 and hi[2] > M.AXIS_Z - M.BORE / 2 + 1e-9
            assert not (inside_x and inside_z)


def test_the_plates_stand_on_the_table_and_the_shaft_starts_on_it():
    for b in M.FIXED:
        assert b.pos[2] - b.size[2] / 2 >= M.TOP - 1e-9
    assert M.SHAFT.pos[2] - M.BAR / 2 == pytest.approx(M.TOP + 0.001)


def test_the_start_scores_zero():
    s = {"shaft": shaft_state((*M.START_XY, M.TOP + M.BAR / 2), M.START_TURN)}
    assert M.score(s) == 0.0 and not M.success(s)


def test_through_both_with_room_for_both_collars_is_success():
    s = on_line(MIDDLE)
    assert M.success(s) and M.score(s) == M.COMPLETE_SCORE
    assert min(M.sticking_out(s["shaft"])) == pytest.approx(0.0182, abs=1e-4)


def test_either_way_round_is_the_same():
    assert M.score(on_line(MIDDLE, turn_deg=180.0)) == M.COMPLETE_SCORE


@pytest.mark.parametrize(("shift_mm", "expected"), [(0.0, 0.9), (8.0, 0.9), (-8.0, 0.9), (9.0, 0.5), (-9.0, 0.5)])
def test_each_end_needs_1_cm_for_its_collar(shift_mm, expected):
    # Centred, each end is out 18.2 mm, so the middle can sit 8.2 mm either way and still leave 1 cm at both ends.
    assert M.score(on_line(MIDDLE + shift_mm / 1000)) == expected


def test_resting_in_the_bores_counts():
    assert M.score(on_line(MIDDLE, dz=-(M.BORE - M.BAR) / 2)) == M.COMPLETE_SCORE  # sagging onto the bores' floors


@pytest.mark.parametrize("kw", [{"moving": True}, {"spinning": True}])
def test_a_moving_or_turning_shaft_is_not_yet_done(kw):
    s = on_line(MIDDLE, **kw)  # 5 cm/s, or turning 0.6 rad/s: both over the limits for still
    assert not M.success(s) and M.score(s) == 0.5


@pytest.mark.parametrize("y", [A + M.REACH_PAST, A, A - 0.03])
def test_through_one_bearing_is_a_quarter(y):
    # Where stroke 1 leaves it (leading end 9.9 cm through), and further on but still short of plate_b.
    assert M.score(on_line(y)) == 0.25


def test_fed_from_the_other_side_counts_the_same():
    assert M.score(on_line(B)) == 0.25


def test_short_of_the_first_bearing_scores_nothing():
    assert M.score(on_line(A + M.SHAFT_LEN / 2 + 0.01)) == 0.0


@pytest.mark.parametrize(("dx_mm", "dz_mm"), [(20.0, 0.0), (0.0, 40.0), (-60.0, 0.0)])
def test_beside_or_over_the_plates_is_not_through(dx_mm, dz_mm):
    # A shaft lying alongside the plates or across their tops crosses their planes outside the bores.
    assert M.score(on_line(MIDDLE, dx=dx_mm / 1000, dz=dz_mm / 1000)) == 0.0


def test_a_shaft_crossing_the_line_is_not_through():
    assert M.score(on_line(MIDDLE, turn_deg=90.0)) == 0.0


def test_a_shaft_off_the_table_scores_zero():
    assert M.score({"shaft": shaft_state((0.3, 0.6, M.TOP - 0.3))}) == 0.0


@pytest.mark.parametrize("axis", ["dx", "dz"])
@pytest.mark.parametrize("sign", [1, -1])
def test_the_bore_window_is_the_bore(axis, sign):
    # The centre line must cross each plate's middle within half the 13.75 mm bore, both ways, in x and in z.
    inside, outside = M.BORE / 2 - 1e-4, M.BORE / 2 + 1e-4
    assert M.score(on_line(MIDDLE, **{axis: sign * inside})) == M.COMPLETE_SCORE
    assert M.score(on_line(MIDDLE, **{axis: sign * outside})) == 0.0


def test_a_pitched_shaft_is_judged_where_it_crosses_each_plate():
    # Centred on the line but pitched 5 degrees, it crosses each plate 9.2 mm off the line: outside both bores.
    assert M.score(on_line(MIDDLE, pitch_deg=5.0)) == 0.0
    # Raised 5 mm and pitched 2.5 degrees down towards plate_a (+y): 0.4 mm off at plate_a, 9.6 mm off at plate_b.
    s = on_line(MIDDLE, dz=0.005, pitch_deg=-2.5)["shaft"]
    assert M.through(s, "plate_a") and not M.through(s, "plate_b")


@pytest.mark.parametrize(("past_mm", "through"), [(1.0, True), (-1.0, False)])
def test_the_shaft_must_reach_past_a_plate_to_be_through_it(past_mm, through):
    # Its plate_b end 1 mm past plate_b's middle, or 1 mm short of it.
    s = on_line(B + M.SHAFT_LEN / 2 - past_mm / 1000)["shaft"]
    assert M.through(s, "plate_b") == through


def _demo_poses():
    left = M.demo()["left"]
    return {w.label: w for w in left}, left


def test_the_demo_picks_the_shaft_at_its_middle():
    # A pinch grip barely resists a turn about the line between the pads: gripped off its middle the shaft swung.
    _, left = _demo_poses()
    assert left[3].label == "close" and left[3].pos[:2] == pytest.approx(M.START_XY)


def test_stroke_1_stops_with_the_fingers_clear_of_plate_a_and_the_shaft_short_of_plate_b():
    by_label, _ = _demo_poses()
    grip_y = by_label["stroke 1"].pos[1]
    assert grip_y - A == pytest.approx(M.REACH_PAST)
    lead_end = grip_y - M.SHAFT_LEN / 2
    assert A - lead_end == pytest.approx(0.099, abs=1e-3) and lead_end > A - M.SPAN  # 9.9 cm through, short of b


def test_the_shaft_starts_its_first_stroke_short_of_the_bearing():
    by_label, _ = _demo_poses()
    lead_end = by_label["lower beside the plate"].pos[1] - M.SHAFT_LEN / 2
    assert lead_end - (A + M.BEARING_W / 2) == pytest.approx(M.LEAD_IN)


def test_a_shaft_let_go_outside_its_middle_tips_onto_the_table():
    # It could tip 20.9 degrees in the bore, but its far end, 15.5 cm out, meets the table first, at 12.2 degrees.
    assert math.degrees(M.WEDGE) == pytest.approx(20.89, abs=0.01)
    assert M.BAR * math.cos(M.WEDGE) + M.BEARING_W * math.sin(M.WEDGE) == pytest.approx(M.BORE, abs=1e-7)
    assert math.degrees(M.RELEASE_TILT) == pytest.approx(12.25, abs=0.01) and M.RELEASE_TILT < M.WEDGE
    far = M.REACH_PAST + M.SHAFT_LEN / 2
    assert M.AXIS_Z - far * math.tan(M.RELEASE_TILT) - M.BAR / 2 == pytest.approx(M.TOP)


def test_stroke_2_grips_the_wedged_shaft_between_the_plates_and_ends_centred():
    by_label, _ = _demo_poses()
    grip, level, end = by_label["close"], by_label["level onto the line"], by_label["stroke 2"]
    assert B + M.REACH_PAST <= grip.pos[1] <= A - M.REACH_PAST
    assert B + M.REACH_PAST - M.DEMO_STROKE_2_FURTHER <= end.pos[1] <= A - M.REACH_PAST
    assert end.pos[1] - B >= M.REACH_PAST - 0.0035  # at most 3.5 mm of the 4 mm spare, in case the arm doesn't lag
    rise = (A - M.STROKE_2_GRIP_Y) * math.tan(M.RELEASE_TILT)
    extra = (M.DEMO_STROKE_2_AIM_MM[1] + M.DEMO_TCP_ERROR_MM[1]) / 1000  # stroke 2 aims higher than the regrasp
    assert grip.pos[2] - level.pos[2] == pytest.approx(rise - extra)
    offset = M.STROKE_2_GRIP_Y - (A + M.REACH_PAST)  # the grip's place on the shaft against its middle
    assert end.pos[1] - offset == pytest.approx(MIDDLE - M.DEMO_STROKE_2_FURTHER)


def test_the_regrasp_closes_on_the_shaft_where_it_lies():
    by_label, _ = _demo_poses()
    grip, level = by_label["close"], by_label["level onto the line"]
    assert grip.yaw == pytest.approx(M.INSERT_YAW + math.radians(M.DEMO_WEDGED_YAW_DEG))
    assert grip.pos[0] == pytest.approx(M._wedged_line_x(M.STROKE_2_GRIP_Y) - M.DEMO_REGRASP_ERROR_MM / 1000)
    assert M._wedged_line_x(M.DEMO_WEDGED_MIDDLE_Y) == M.DEMO_WEDGED_X
    assert level.yaw == pytest.approx(M.INSERT_YAW - math.radians(M.DEMO_STROKE_2_TURN_DEG))  # then turns onto it


def test_stroke_2_holds_the_shaft_on_the_bores_line():
    by_label, _ = _demo_poses()
    w = by_label["stroke 2"]
    assert w.pos == pytest.approx(M._aim_2(M.AXIS_X, w.pos[1], M.AXIS_Z + M.CARRY_DROP))


def test_stroke_1_enters_level_with_the_leading_end_on_the_line_then_lowers():
    by_label, _ = _demo_poses()
    enter, stroke = by_label["enter"], by_label["stroke 1"]
    lift = M.SHAFT_LEN / 2 * math.tan(math.radians(M.DEMO_SLIP_DEG[1]))
    assert enter.pos == pytest.approx(M._aim(M.AXIS_X, enter.pos[1], M.AXIS_Z + M.CARRY_DROP + lift))
    assert stroke.pos == pytest.approx(M._aim(M.AXIS_X, stroke.pos[1], M.AXIS_Z + M.CARRY_DROP))
    lead_end = enter.pos[1] - M.SHAFT_LEN / 2
    assert (M.PLATE_Y["plate_a"] + M.BEARING_W / 2) - lead_end == pytest.approx(M.ENTER - M.LEAD_IN)  # 2 cm in
    assert enter.yaw == pytest.approx(M.INSERT_YAW - math.radians(M.DEMO_SLIP_DEG[0]))


def test_the_pads_reach_the_shaft_but_not_the_table():
    tips = M.TABLE_GRIP_Z - PAD_TIP_PAST_TCP
    assert M.TOP < tips < M.TOP + M.BAR / 2


def test_a_carried_shaft_clears_the_plates():
    assert M.TRAVEL_Z - M.CARRY_DROP - M.BAR / 2 - (M.TOP + M.PLATE_H) >= 0.03


def test_the_grip_centres_the_shaft_and_lets_go_with_room():
    from yam_arena.constants import GRIPPER

    squeeze = (M.BAR - 0.094 * M.GRIP_FINGER / FINGER_OPEN) / 2
    force = GRIPPER["stiffness"] * squeeze
    assert 0.002 <= squeeze and force < 0.7 * GRIPPER["effort"]
    assert 2 * force * 1.0 > 10 * M.SHAFT_KG * 9.81
    assert 94 * M.RELEASE_FINGER / FINGER_OPEN - M.BAR * 1000 >= 15


def test_the_demo_grips_across_the_shaft():
    by_label, left = _demo_poses()
    pick = left[3]
    grip_y = np.array([-math.sin(pick.yaw), math.cos(pick.yaw)])  # the direction the fingers close along
    turn = math.radians(M.START_TURN)
    axis = np.array([-math.sin(turn), math.cos(turn)])
    assert abs(float(grip_y @ axis)) < 1e-9
    assert abs(math.cos(M.INSERT_YAW)) < 1e-9  # closing along x, across a shaft along y


@pytest.mark.parametrize("arm", list(M.demo()))
def test_the_demo_is_within_reach(arm):
    plan = ArmPlan(M.demo()[arm], station.BASES[arm])
    for t in np.arange(0.0, plan.duration, 0.05):
        plan.action7(t)
    assert plan.worst_ik_mm < 0.5 and plan.worst_ik_deg < 0.1, (plan.worst_ik_mm, plan.worst_ik_deg)


def test_the_demo_aims_unshifted():
    assert M.DEMO_AIM_SHIFT_MM == 0.0


def test_the_demo_finishes_well_within_the_run_limit():
    assert sum(w.duration for w in M.demo()["left"]) < 2400 / 20 / 2


def test_the_complete_state_scores_0_9_until_success_has_held_1_s():
    # 1.0 comes only from success() holding for 1 s, which ends the trial. score() is what task_score takes if time
    # runs out first, so a complete state that only settles in the final second earns 0.9.
    s = on_line(MIDDLE)
    assert M.success(s) and M.score(s) == 0.9 < 1.0
