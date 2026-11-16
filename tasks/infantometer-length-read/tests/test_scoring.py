# SPDX-License-Identifier: MIT
# Copyright (c) 2026 the task authors
"""Scorer edge cases for infantometer-length-read, pure Python (no simulator). Run from the task folder:
pytest -q tests

The scorer is record() (fed every control step into a linkkit History, which only the simulator reset clears) and
ladder(); these tests feed it synthetic steps, and check the scene geometry with the kit's own kinematics.
"""

import math
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import linkkit as lk  # noqa: E402
import task as T  # noqa: E402

DT = 0.05  # 20 Hz control
CURLED = dict(hips=(80.0, 80.0), knees=(95.0, 95.0), gaps=(0.06, 0.06))
FLAT = dict(hips=(1.0, 1.0), knees=(2.0, 2.0))


def step(**kw):
    """One control step's measurement; defaults: legs flat, crown on the headpiece, footboard far, hands on."""
    m = dict(hips=(1.0, 1.0), knees=(2.0, 2.0), head_gap=0.0008, gaps=(0.03, 0.03), feet_x=(0.174, 0.226),
             reading=0.450, qd=0.0, low=0.0, trunk=(0.200, 0.279), clear={"left": 0.0, "right": 0.0},
             speed={"left": 0.0, "right": 0.0})
    m.update(kw)
    return m


def run(steps, h=None):
    h = h or lk.History()
    for m in steps:
        h._tick(h.t + DT)  # what the kit does before every on_step
        T.record(h, m)
    return h


def rep(n_or_s, **kw):
    n = n_or_s if isinstance(n_or_s, int) else int(round(n_or_s / DT))
    return [step(**kw) for _ in range(n)]


def demo_steps(reading=0.4505, release=True, touch_s=1.0, end_clear=0.05, final=None):
    """What the scripted demo does: curled 3 s, legs pressed flat 2 s, footboard closes 2 s, touch at ``reading``,
    release with the footboard left at ``final`` (default: the reading)."""
    final = reading if final is None else final
    s = rep(3.0, **CURLED, reading=0.52)
    s += rep(2.0, reading=0.52, gaps=(0.07, 0.07))
    s += [step(reading=0.52 - (0.52 - reading) * k / 40, gaps=(0.07 * (1 - k / 40) + 0.004,) * 2, qd=0.025)
          for k in range(1, 41)]
    s += rep(touch_s, reading=reading, gaps=(0.0004, 0.0006))
    if release:
        s += rep(1.0, **CURLED, reading=final, clear={"left": 0.01, "right": 0.06}, speed={"left": 0.05, "right": 0.0})
        s += rep(1.5, **CURLED, reading=final, clear={"left": end_clear, "right": 0.08})
    else:
        s += rep(2.5, reading=reading, gaps=(0.0004, 0.0006))
    return s


def score(h):
    return T.ladder(h)[0]


def test_demo_trace_scores_one():
    h = run(demo_steps())
    assert score(h) == 1.0, T.ladder(h)
    assert h.when("measured") is not None


def test_no_steps_scores_zero():
    assert score(lk.History()) == 0.0


def test_nothing_scores_zero():
    assert score(run(rep(10.0, **CURLED, reading=0.52))) == 0.0


@pytest.mark.parametrize("final,expect", [(0.4455, 1.0), (0.4555, 1.0), (0.4454, 0.5), (0.4556, 0.5)])
def test_footboard_left_within_5_mm_of_its_reading(final, expect):
    assert score(run(demo_steps(reading=0.4505, final=final))) == expect


@pytest.mark.parametrize("reading", [0.444, 0.456, 0.459])
def test_valid_touch_off_450_left_in_place_scores_one(reading):
    # the touch limits allow readings off 450 mm (bent-knee tilt, crown and sole gaps); leaving the footboard at the
    # reading taken is what the instruction asks, so it scores 1.0
    assert score(run(demo_steps(reading=reading))) == 1.0


def test_brief_touch_does_not_widen_the_window():
    # a 0.3 s touch at 459 mm is not a measurement and adds no reading; the measured touch read 450.5 mm
    brief = rep(0.3, reading=0.459, gaps=(0.0004, 0.0006)) + rep(1.0, **CURLED, reading=0.52)
    assert score(run(brief + demo_steps(final=0.4555))) == 1.0
    assert score(run(brief + demo_steps(final=0.4560))) == 0.5


def test_window_spans_every_measured_reading():
    first = demo_steps(reading=0.447, release=False)
    second = rep(1.0, reading=0.456, gaps=(0.0004, 0.0006))
    end = rep(2.5, **CURLED, reading=0.4610, clear={"left": 0.05, "right": 0.08})
    h = run(first + second + end)
    assert h.lows["measured_reading"] == pytest.approx(0.447) and h.peaks["measured_reading"] == pytest.approx(0.456)
    assert score(h) == 1.0
    end = rep(2.5, **CURLED, reading=0.4415, clear={"left": 0.05, "right": 0.08})
    assert score(run(first + second + end)) == 0.5


def test_touch_must_hold_half_a_second():
    assert run(demo_steps(touch_s=0.45)).when("measured") is None
    assert run(demo_steps(touch_s=0.50)).when("measured") is not None


@pytest.mark.parametrize("gaps", [(0.0021, 0.0), (0.0, 0.0021), (-0.0031, 0.0)])
def test_one_sole_off_the_footboard_blocks_touch(gaps):
    s = rep(2.0) + rep(1.0, gaps=gaps)
    h = run(s)
    assert h.when("measured") is None
    assert score(h) == 0.25  # the legs were still held straight with the crown on for 1 s


def test_foot_beside_the_plate_blocks_touch():
    h = run(rep(2.0, gaps=(0.0, 0.0), feet_x=(0.174, 0.305)))
    assert h.when("measured") is None


def test_head_gap_blocks_touch_and_straight():
    h = run(rep(3.0, gaps=(0.0, 0.0), head_gap=0.0021))
    assert h.when("measured") is None and score(h) == 0.0
    assert run(rep(3.0, gaps=(0.0, 0.0), head_gap=0.0020)).when("measured") is not None


@pytest.mark.parametrize("hips,knees,ok", [((12.0, 0.0), (15.0, 0.0), True), ((12.1, 0.0), (0.0, 0.0), False),
                                           ((0.0, 0.0), (0.0, 15.1), False), ((0.0, 12.1), (0.0, 0.0), False)])
def test_legs_straight_limits(hips, knees, ok):
    h = run(rep(2.0, hips=hips, knees=knees, gaps=(0.0, 0.0)))
    assert (h.when("measured") is not None) == ok
    assert (h.best_runs["straight"] >= 1.0) == ok


def test_moving_footboard_is_not_a_touch():
    assert run(rep(2.0, gaps=(0.0, 0.0), qd=0.0051)).when("measured") is None
    assert run(rep(2.0, gaps=(0.0, 0.0), qd=0.0049)).when("measured") is not None


def test_lift_gate_scores_zero_even_after_a_perfect_measurement():
    s = demo_steps()
    s[10] = dict(s[10], low=0.0201)
    assert score(run(s)) == 0.0
    s[10] = dict(s[10], low=0.0199)
    assert score(run(s)) == 1.0


def test_off_board_gate():
    s = demo_steps()
    s[5] = dict(s[5], trunk=(0.33, 0.279))
    assert score(run(s)) == 0.0


def test_release_restarts_when_a_pad_comes_within_29_mm():
    s = demo_steps()
    s[-5] = dict(s[-5], clear={"left": 0.029, "right": 0.08})  # 4 steps before the end: run is 0.2 s
    h = run(s)
    assert score(h) == 0.75 and h.runs["released"] == pytest.approx(0.2)


def test_moving_gripper_blocks_release():
    s = demo_steps()
    s[-3] = dict(s[-3], speed={"left": 0.0, "right": 0.006})
    assert score(run(s)) == 0.75


def test_hold_on_scores_three_quarters():
    assert score(run(demo_steps(release=False))) == 0.75


def test_measured_then_moved_off_the_mark_scores_half():
    s = demo_steps()
    s += rep(2.0, **CURLED, reading=0.47, clear={"left": 0.05, "right": 0.08})
    h = run(s)
    assert score(h) == 0.5 and h.when("measured") is not None


def test_press_then_fetch_scores_quarter():
    s = rep(1.0, **CURLED, reading=0.52) + rep(2.0, reading=0.52) + rep(3.0, **CURLED, reading=0.391)
    s += rep(1.0, hips=(80.0, 80.0), knees=(95.0, 95.0), gaps=(0.0, 0.0005), reading=0.391,
             clear={"left": 0.05, "right": 0.08})
    assert score(run(s)) == 0.25


def test_footboard_on_the_mark_without_touching_scores_zero():
    assert score(run(rep(8.0, **CURLED, reading=0.450, clear={"left": 0.1, "right": 0.1}))) == 0.0


def test_measured_survives_recurl_and_history_clears_only_on_reset():
    h = run(demo_steps())
    t_measured = h.when("measured")
    run(rep(2.0, **CURLED, reading=0.4505, clear={"left": 0.05, "right": 0.08}), h)
    assert h.when("measured") == t_measured and score(h) == 1.0
    run([step(low=0.03)] + rep(2.0), h)  # lifted once, put back: still 0
    assert score(h) == 0.0
    h.reset()  # what the simulator reset does
    assert score(h) == 0.0 and h.when("lifted") is None and h.when("measured") is None


# -- scene geometry with the kit's own kinematics ---------------------------------------------------------------


def kin_state(spec, q):
    poses = spec.link_poses(q)
    z = np.zeros(3)
    root = poses[spec.links[0].name]
    return lk.LinkState(pos=root[:3, 3].copy(), quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]), lin_vel=z, ang_vel=z,
                        low=z, high=z, spec=spec, poses=poses, lin_vels={n: z for n in poses},
                        ang_vels={n: z for n in poses}, q={j.name: q.get(j.name, j.init) for j in spec.moving_joints},
                        qd={j.name: 0.0 for j in spec.moving_joints})


def test_true_length_is_450_mm():
    assert T._true_length() == pytest.approx(0.450, abs=5e-4)


def test_design_pose_with_footboard_on_the_soles_is_a_touch():
    baby = kin_state(T.BABY, {j.name: 0.0 for j in T.BABY.moving_joints})
    board = kin_state(T.BOARD, {"slide": -0.001})  # face on the soles (the crown sits 1 mm off the headpiece)
    m = T.measure(baby, board)
    assert m["gaps"] == pytest.approx((0.0, 0.0), abs=1e-9)
    assert m["head_gap"] == pytest.approx(0.001, abs=1e-9)
    assert m["reading"] == pytest.approx(0.451)
    assert m["low"] == pytest.approx(0.0, abs=1e-9)
    assert max(m["hips"] + m["knees"]) == pytest.approx(0.0, abs=1e-9)
    h = run([dict(m, clear={"left": 0.0}, speed={"left": 0.0})] * 12)
    assert h.when("measured") is not None


def test_touch_limits_keep_crown_to_sole_within_448_to_456_mm():
    # every hip/knee pair inside the touch limits (hips 0 to 12 deg, knees -3 to 15 deg) with no collider below the
    # board: the crown-to-nearest-sole length stays near the true 450 mm, so a touch reads about 445 to 460 mm once the
    # 2 mm crown and sole allowances are added
    lengths = []
    for hip in np.arange(0.0, T.HIP_MAX + 1e-9, 0.5):
        for knee in np.arange(-3.0, T.KNEE_MAX + 1e-9, 0.5):
            q = {f"{j}_{s}": math.radians(a) for s in T.LEGS for j, a in (("hip", hip), ("knee", knee))}
            baby = kin_state(T.BABY, q)
            if T.lowest(baby) < T.Z_B - 0.001:
                continue  # a heel or shin through the board
            crown = float(baby.point("trunk", T.HEAD_LOCAL)[1]) + T.HEAD_R
            lengths.append(crown - min(float(baby.point("shin_n", c)[1]) for c in T.SOLE_CORNERS))
    assert len(lengths) > 200
    assert 0.447 <= min(lengths) and max(lengths) <= 0.457, (min(lengths), max(lengths))


def test_curled_legs_first_touch_reads_about_391_mm():
    baby = kin_state(T.BABY, {})  # every joint at its reset value: hips 80, knees 95
    foot = T.BABY.link("shin_n").shapes[1]
    corners = np.array(np.meshgrid(*[[-1, 1]] * 3)).T.reshape(-1, 3) * foot.half_extents() + np.asarray(foot.pos)
    y_most = min(float(baby.point(f"shin_{s}", c)[1]) for s in T.LEGS for c in corners)
    first_touch = T.Y_HEADPIECE - y_most
    assert 0.385 <= first_touch <= 0.400, first_touch
    assert T.L_TRUE - first_touch > 0.045  # far outside the 5 mm tolerance
    assert T.lowest(baby) - T.Z_B == pytest.approx(0.0, abs=1e-9)  # trunk on the board, feet in the air
    feet_low = min(float(baby.point("shin_n", c)[2]) for c in corners) - T.Z_B
    assert feet_low > 0.04


def test_demo_poses_reachable():
    ik = pytest.importorskip("yam_arena.ik")
    up = np.array([0.0, 0.0, 0.013])
    left_b, right_b = (0.0, 0.305, 0.75), (0.0, -0.305, 0.75)
    hip = np.array([0.2, T.HIP_Y, T.AXIS_Z])
    knee0 = np.array([0.2, T.HIP_Y - 0.085 * math.cos(math.radians(80)),
                      T.AXIS_Z + 0.085 * math.sin(math.radians(80))])
    poses = [
        ("left sweep start", left_b, np.array([0.2, knee0[1] + T.D_START, knee0[2]]) + up, T.LEFT_ROT),
        ("left phase A end", left_b, hip + lk.axis_rotation((1.0, 0.0, 0.0), math.radians(T.A_MAX)) @ (np.array([0.2, knee0[1] + T.D_START, knee0[2]]) - hip)
         + up, T.LEFT_ROT),
        ("left press, 8 mm extra", left_b, T._thigh_top(hip, T.B_END) + up - [0.0, 0.0, 0.008], T.LEFT_ROT),
        ("left opens at 40 deg", left_b, T._thigh_top(hip, T.RELEASE_TO) + up, T.LEFT_ROT),
        ("left lifted", left_b, T._thigh_top(hip, T.RELEASE_TO) + up + [0.0, 0.0, 0.10], T.LEFT_ROT),
        ("left away", left_b, np.array([0.20, 0.17, 0.97]), T.LEFT_ROT),
        ("right handle at 520", right_b, np.array([0.2, T.FACE_Y0 - 0.07 - 0.072, T.Z_B + 0.124]), T.RIGHT_ROT),
        ("right handle at 450", right_b, np.array([0.2, T.FACE_Y0 - 0.072, T.Z_B + 0.124]), T.RIGHT_ROT),
    ]
    for name, base, p, rot in poses:
        best = min((ik.solve(p, ik.matrix_to_quat_xyzw(rot), np.array(q0, float), base_pos=base, iters=400)
                    for q0 in ([0, 1, 1, -0.6, 0, 0], [0.3, 1.2, 1.2, -0.4, 0, 0], [-0.3, 1.2, 1.2, -0.4, 0, 0])),
                   key=lambda r: r[1] + 0.05 * r[2])
        assert best[1] < 0.002 and math.degrees(best[2]) < 1.0, (name, best[1], math.degrees(best[2]))
