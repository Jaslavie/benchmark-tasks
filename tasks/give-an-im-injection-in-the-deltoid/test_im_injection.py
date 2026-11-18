"""Offline tests: the scorer on hand-built readings (every tier and the negatives), the layout, and the whole demo stepped
offline with yam_arena's own IK on yam.urdf: reach, the arms' real meshes swept against every fixture, the scoring views,
and the demo scored end to end by the same rule. No simulator, no GPU.

    python -m pytest -q test_im_injection.py      (with the harness's Python: yam_arena must import)
"""

from __future__ import annotations

import json
import math
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from yam_arena import YAM_URDF, ik, station
from yam_arena.constants import FINGER_OPEN
from yam_arena.task_file import load
from yam_arena.urdf import forward_kinematics, rpy_matrix

HERE = Path(__file__).parent
TASK = load(HERE / "im_injection.py")
T = TASK.module
DT = 1.0 / 240.0
X = np.array([1.0, 0.0, 0.0])


def _rot_z(deg):
    a = math.radians(deg)
    return np.array([math.cos(a), math.sin(a), 0.0])


def _axis(pitch=0.0, yaw=0.0):
    """+x turned by yaw (about z) and pitch (about -y: positive tips the needle up)."""
    p, y = math.radians(pitch), math.radians(yaw)
    return np.array([math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), math.sin(p)])


# -- a hand-driven trial ------------------------------------------------------------------------------------------------


class Run:
    """A syringe driven by hand through the rule, one physics step at a time (240 Hz)."""

    def __init__(self, geom=None, site=None, axis=(1.0, 0.0, 0.0), cap_on=False):
        self.g = geom or T.Geometry()
        self.tr = T.Trial(geom=self.g)
        self.tr.cap_on = cap_on
        self.site = self.g.site if site is None else np.asarray(site, float)
        self.a = np.asarray(axis, float) / np.linalg.norm(axis)
        self.hub = self.site - self.a * (T.NEEDLE_L + 0.010)
        self.travel = 0.0
        self.cap_slide = 0.0
        self.needle_axis = None
        self.touch = None
        self.thumb = 0.0  # N on the thumb rest along the axis

    def force(self):
        if self.tr.engaged or self.tr.cap_on:
            return 0.0
        tip = self.hub + T.NEEDLE_L * self.a
        if abs(tip[1]) > T.TRAINER_W / 2 or not (self.g.z_bottom <= tip[2] <= self.g.z_top):
            return 0.0
        pen = tip[0] - self.g.surface_x(tip[1], tip[2])
        return float(np.clip(T.K_SENSE * pen, 0.0, 2.0))

    def step(self, n=1, v_tr=0.0):
        for _ in range(n):
            self.tr.feed(T.Sample(DT, hub=tuple(self.hub), axis=tuple(self.a), tip_force=self.force(), travel=self.travel, travel_vel=v_tr,
                                  thumb_n=self.thumb, cap_on=True, cap_slide=self.cap_slide, needle_axis=self.needle_axis, touch=self.touch))
        return self

    def uncap(self):
        self.cap_slide = 0.003
        self.step()
        self.cap_slide = 0.0
        return self

    def move(self, d, v=0.02):
        d = np.asarray(d, float)
        n = max(1, round(np.linalg.norm(d) / v / DT))
        for _ in range(n):
            self.hub = self.hub + d / n
            self.step()
        return self

    def insert(self, to_gap=-0.0003, v=0.010):
        """Along the axis until the hub is `to_gap` from the skin (negative: pressed on it)."""
        n_in = -self.g.normal(self.site[1], self.site[2])
        target = self.site - self.a * (to_gap / max(float(self.a @ n_in), 0.2))
        return self.move(target - self.hub, v)

    def push(self, ml, s_per_02=2.0):
        """Push `ml` at 0.2 mL per `s_per_02` seconds."""
        v = 0.2 * T.ML_M / s_per_02
        n = round(ml * T.ML_M / v / DT)
        self.thumb = 5.0  # past the stopper's breakaway, short of the 0 line's 10 N
        for _ in range(n):
            self.travel = min(self.travel + v * DT, T.TRAVEL_MAX)
            self.step(v_tr=v)
        return self

    def retract(self):
        v = 0.002
        self.thumb = 12.0  # through the 0 line's stop
        while self.travel < T.RETRACT_AT:
            self.travel = min(self.travel + v * DT, T.TRAVEL_MAX)
            self.step(v_tr=v)
        self.step(24)
        return self

    def full(self, s_per_02=2.0):
        """The whole injection: insert, push 1 mL, retract."""
        self.uncap().insert().step(60)
        return self.push(1.0, s_per_02).retract()

    def end(self, steps=25, **kw):
        st = {"in_container": True, "cap_in_cup": True, "clear": 0.10} | kw
        for _ in range(steps):
            self.tr.status(T.Status(**st))
        return self.tr.score


def test_perfect_run_scores_full_after_one_second():
    r = Run().full()
    p = r.tr.first()
    assert p.engaged and p.full_depth and abs(p.D - T.SITE_D) < 1e-6 and p.L < 1e-6 and p.A < 1e-6
    assert r.tr.retracted and r.tr.full_dose() and r.tr.steady() and r.tr.continuous and not r.tr.moved
    assert r.end(steps=19) == 0.75 and r.end(steps=1) == 1.0 and r.tr.full


def test_untouched_scores_nothing():
    assert Run(cap_on=True).end() == 0.0


def test_joints_switch_in_order():
    r = Run()
    acts = []
    orig = r.tr.feed
    r.tr.feed = lambda s: acts.extend(orig(s)) or []
    r.full()
    assert acts == ["lock_off", "tissue_on", "unstick", "detent", "tissue_off", "retract"]
    assert T.JOINT_OF["retract"] == [("J_lock", False)]
    assert T.JOINT_OF["unstick"] == [("J_stick", False)] and T.JOINT_OF["detent"] == [("J_zero", False)]


def test_engagement_needs_1_n_and_the_needle_bare():
    r = Run(cap_on=True)
    r.insert(to_gap=0.0)  # capped: the cap covers the needle, nothing happens
    assert not r.tr.punctures
    r2 = Run().uncap()
    r2.move(0.0115 * r2.a)  # the tip 1.5 mm into the skin: 0.75 N
    assert not r2.tr.punctures and not r2.tr.engaged
    r2.move(0.001 * r2.a)
    assert r2.tr.engaged


def test_s_is_one_graduation_below_the_start():
    """S: the edge reaches (on or past) the printed graduation 0.1 mL below its reading, to the nearest graduation, at
    the first in-muscle frame: from 1.0 mL, the 0.9 line; from 0.97 or 1.0002 mL, also the 0.9 line."""
    r = Run().uncap().insert().step(30)
    assert r.tr.first().travel_first == 0.0 and r.tr.t_S is None
    r.push(0.099)
    assert r.tr.t_S is None
    r.push(0.002)
    assert r.tr.t_S is not None and r.tr.vol_S <= 0.9 + 1e-9 and r.tr.vol_S == pytest.approx(0.9, abs=0.002)
    assert T.grad_below(1.0) == pytest.approx(0.9) and T.grad_below(0.97) == pytest.approx(0.9) and T.grad_below(0.9) == pytest.approx(0.8)
    for vol, line in ((1.0, 0.9), (1.0002, 0.9), (1.04, 0.9), (0.97, 0.9), (0.95, 0.9), (0.9, 0.8), (0.9003, 0.8), (0.94, 0.8)):
        assert T.s_line(vol) == pytest.approx(line), vol
    assert T.grad_index(0.95) == 10 and T.grad_index(0.9) == 9 and T.GRAD_ML == 0.1


@pytest.mark.parametrize("hair", [-3e-6, -1e-7, 0.0, 2e-6])
def test_s_does_not_fire_at_insertion_with_the_stopper_a_hair_off_the_line(hair):
    """The recorded fault: the stopper read a hair above the 1.0 mL line (travel about -0.003 mm) at the first in-muscle
    frame, the 'first graduation below' was then the 1.0 line itself, and S fired at the insertion: the rest of the
    insertion counted as the needle moving and the wait before the push as a stop. S is now the 0.9 line for any reading
    that rounds to 1.0, so the same run has no S until the push, and scores 1.0."""
    assert T.grad_below(T.volume(-3e-6)) == pytest.approx(1.0)  # the old rule's line for that reading: the fault
    r = Run().uncap()
    r.travel = hair
    r.insert()
    assert r.tr.first().travel_first == hair and r.tr.in_muscle_now
    r.travel = 1e-6  # the stopper settles back across the 1.0 line with nothing pushing
    r.step(round(5.5 / DT))  # and the hand waits 5.5 s before the push
    assert r.tr.t_S is None and not r.tr.moved and not r.tr.stopped
    r.travel = 0.0
    r.push(0.099)
    assert r.tr.t_S is None
    r.push(0.002)
    assert r.tr.t_S is not None and r.tr.vol_S == pytest.approx(0.9, abs=0.002)
    r.push(0.899).retract()
    assert r.tr.steady() and r.tr.still() and not r.tr.moved and not r.tr.stopped and r.end() == 1.0


def test_a_push_during_the_insertion_starts_s_there():
    """S is the push's start wherever the needle is: pushing 0.1 mL with the hub still 4 mm off the skin starts S there,
    and the last 4 mm of insertion then count as the needle moving (0.75)."""
    r = Run().uncap().insert(to_gap=0.004)
    assert r.tr.in_muscle_now and r.tr.t_S is None
    r.push(0.11)
    assert r.tr.t_S is not None
    r.insert().push(0.89).retract()
    assert r.tr.moved and r.end() == 0.75


def test_insertion_before_s_is_not_counted_as_moving():
    """The hub travels 25 mm in before S: needle-still compares from the pose at S only."""
    r = Run().full()
    assert not r.tr.moved and r.end() == 1.0


@pytest.mark.parametrize("kind,amount,want", [("pitch", 6.0, 0.75), ("yaw", 6.0, 0.75), ("y", 0.004, 0.75), ("pitch", 4.0, 1.0), ("yaw", 4.0, 1.0), ("y", 0.002, 1.0)])
def test_needle_moved_after_s(kind, amount, want):
    """Over S to the retraction: 6 deg of pitch or yaw, or 4 mm in y, is 'moved' (0.75); a 4 deg wobble or 2 mm is not."""
    r = Run().uncap().insert().step(30).push(0.2)
    assert r.tr.t_S is not None
    if kind == "y":
        r.move((0, amount, 0), v=0.01)
        r.push(0.4)
        r.move((0, -amount, 0), v=0.01)
    else:
        r.a = _axis(**{kind: amount})
        r.push(0.4)
        r.a = X.copy()
    r.push(0.4).retract()
    assert r.tr.moved == (want < 1.0)
    assert r.end() == want


def test_pulled_out_mid_push_loses_the_top_tiers():
    """A 7 mm hub-to-skin gap mid-push (out of muscle): no 1.0 or 0.75; 0.5 if 0.5 mL went in while in muscle."""
    r = Run().uncap().insert().step(30).push(0.6)
    r.move(-0.007 * r.a, v=0.01)
    r.push(0.1)
    r.move(0.007 * r.a, v=0.01)
    r.push(0.3).retract()
    assert not r.tr.continuous and len(r.tr.punctures) == 1
    assert r.end() == 0.5


def test_pulled_right_out_and_back_in_is_a_second_puncture():
    r = Run().uncap().insert().step(30).push(0.3)
    r.move(-(T.NEEDLE_L + 0.005) * r.a)
    assert not r.tr.engaged
    r.insert().push(0.7).retract()
    assert len(r.tr.punctures) == 2 and r.end() == 0.25


def test_second_puncture_elsewhere_caps_at_025():
    g = T.Geometry()
    r = Run(site=g.site + [0.0, 0.0, 0.006]).uncap().insert().step(10)
    r.move(-(T.NEEDLE_L + 0.008) * r.a)
    r.site = g.site
    r.move(g.site - [T.NEEDLE_L + 0.010, 0, 0] - r.hub)
    r.insert().push(1.0).retract()
    assert len(r.tr.punctures) == 2 and r.end() == 0.25


def test_push_before_puncture_is_a_permanent_zero():
    r = Run().uncap()
    r.push(0.25)
    assert r.tr.zero
    r.insert().push(0.75).retract()
    assert r.end() == 0.0


def test_small_push_before_puncture_is_not_a_zero_but_misses_full_dose():
    """0.15 mL out before the needle goes in: no zero, but the edge is below 0.9 mL at the first in-muscle frame: 0.5."""
    r = Run().uncap().push(0.15).insert().step(30).push(0.85).retract()
    assert r.tr.zero is None and not r.tr.full_dose() and r.end() == 0.5


@pytest.mark.parametrize("D_mm,want", [(37.5, 1.0), (26.0, 1.0), (49.0, 1.0), (22.0, 0.5), (53.0, 0.5), (19.0, 0.0), (57.0, 0.0)])
def test_site_depth_below_the_acromion(D_mm, want):
    g = T.Geometry()
    r = Run(site=(g.skin_x, 0.0, g.z_edge - D_mm / 1000)).full()
    assert r.end() == want, r.tr.report()


def test_site_too_high_is_a_zero_whatever_follows():
    g = T.Geometry()
    r = Run(site=(g.skin_x, 0.0, g.z_edge - 0.015)).full()
    assert r.tr.zero and "20 mm" in r.tr.zero and r.end() == 0.0


@pytest.mark.parametrize("L_mm,want", [(8.0, 1.0), (12.0, 0.5), (16.0, 0.0)])
def test_site_off_the_midline(L_mm, want):
    g = T.Geometry()
    r = Run(site=(g.skin_x, L_mm / 1000, g.site[2])).full()
    assert r.end() == want


@pytest.mark.parametrize("pitch,yaw,want", [(8.0, 0.0, 1.0), (0.0, 9.0, 1.0), (12.0, 0.0, 0.75), (0.0, 14.0, 0.75), (17.0, 0.0, 0.5), (0.0, 19.0, 0.5)])
def test_angle_tiers(pitch, yaw, want):
    r = Run(axis=_axis(pitch, yaw)).full()
    assert r.tr.first().A == pytest.approx(max(abs(pitch), abs(yaw)), abs=0.05)
    assert r.end() == want


def test_tilted_more_than_20_never_goes_in_and_still_punctures():
    """25 deg off square: it presses and punctures (a real arm), but never goes in; a later square one is a second
    puncture (0.25 at most)."""
    r = Run(axis=_axis(0.0, 25.0)).uncap()
    r.move(0.014 * r.a)
    assert not r.tr.engaged and len(r.tr.punctures) == 1 and not r.tr.punctures[0].engaged
    assert r.end() == 0.0
    r.move(-0.02 * r.a)
    r.a = X.copy()
    r.move(r.site - (T.NEEDLE_L + 0.010) * X - r.hub)
    r.insert().push(1.0).retract()
    assert len(r.tr.punctures) == 2 and r.end() == 0.25


def test_tier_needs_a_within_20_deg():
    base = {"zero": False, "punctures": 1, "D": 0.0375, "L": 0.0, "A": 19.0, "full_depth": True, "continuous": False, "still": True, "full_dose": False, "steady": True,
                "stowed": True, "held": True, "pushed_ml": 0.6, "any_025": True}
    assert T.tier(**base) == 0.5
    assert T.tier(**(base | {"A": 21.0, "any_025": False})) == 0.0


def test_not_full_depth_is_075():
    r = Run().uncap().insert(to_gap=0.003).step(30).push(1.0).retract()
    assert not r.tr.first().full_depth
    assert r.end() == 0.75


@pytest.mark.parametrize("s_per_02,want", [(1.1, 1.0), (3.8, 1.0), (0.6, 0.75), (4.5, 0.75)])
def test_steady_push(s_per_02, want):
    """1-4 s per 0.2 mL: 2 s (10 s per mL) and the ends of the band pass; too fast (0.6 s) or too slow (4.5 s) is 0.75."""
    r = Run().full(s_per_02=s_per_02)
    assert r.tr.steady() == (want == 1.0)
    assert r.end() == want


def test_first_interval_is_prorated_from_s():
    """S at the 0.9 line; the first line reached is 0.8: 0.1 mL, allowed 0.5-2 s (1-4 s per 0.2 mL, prorated)."""
    r = Run().full()
    first_line, t1 = r.tr.crossings[0]
    assert first_line == 0.8
    frac = (r.tr.vol_S - 0.8) / 0.2
    assert frac == pytest.approx(0.5, abs=0.01)
    assert T.STEP_S[0] * frac <= t1 - r.tr.t_S <= T.STEP_S[1] * frac


def test_a_stall_mid_push_is_not_steady():
    """A 5 s stall mid-dose, thumb on: a stop (3 s or more with no new graduation reached) and an interval over 4 s."""
    r = Run().uncap().insert().step(30).push(0.5)
    r.step(round(5.0 / DT))
    r.push(0.5).retract()
    assert r.tr.stopped and not r.tr.rate_ok() and not r.tr.steady() and r.end() == 0.75


@pytest.mark.parametrize("pause,want", [(1.5, 1.0), (2.0, 1.0), (3.2, 0.75)])
def test_a_pause_of_3_s_is_a_stop_even_inside_the_rate_band(pause, want):
    """S at the 0.9 line. Push to the 0.6 line at 2 s per 0.2 mL, then pause (thumb on), then the next 0.2 mL in 0.5 s:
    that interval is pause + 0.5 s, inside 1-4 s, so the rate band alone passes all three. A 3.2 s pause reaches no new
    0.1 mL graduation for over 3 s: a stop, 0.75; 1.5 and 2 s (with the graduation reached just before) are not: 1.0."""
    r = Run().uncap().insert().step(60)
    r.push(0.4)  # to the 0.6 line
    r.step(round(pause / DT))  # the thumb stays on (5 N), nothing moves
    r.push(0.2, s_per_02=0.5).push(0.4).retract()
    assert r.tr.rate_ok(), r.tr.report()["crossings"]
    assert r.tr.stopped == (want < 1.0) and r.end() == want


def test_a_pause_at_the_0_line_is_not_timed():
    """The timing, the stop and the thumb-on clause end when the edge reaches the 0 line: a 6 s wait there with the thumb
    off, before pushing on through the detent (10 N), is not a stop, not a gap, and still scores 1.0."""
    r = Run().uncap().insert().step(60).push(1.0)
    t0 = r.tr.t_zero
    assert t0 is not None and r.tr.crossings[-1][0] == 0.0
    r.thumb = 0.0
    r.step(round(6.0 / DT))
    r.retract()
    assert r.tr.t_zero == t0 and not r.tr.stopped and not r.tr.thumb_lost and r.tr.steady() and r.end() == 1.0


@pytest.mark.parametrize("gap,want", [(0.05, 1.0), (0.15, 0.75)])
def test_thumb_off_the_thumb_rest_from_s_to_the_0_line(gap, want):
    """Thumb on from S to the 0 line: the contact load under 0.2 N for 0.1 s or more (3 frames of the side video) is the
    thumb off: 0.75. A shorter dip (0.05 s, contact noise) is not."""
    r = Run().uncap().insert().step(60).push(0.5)
    r.thumb = 0.1  # under THUMB_ON_N while the edge still advances (as if pushed by something else)
    v = 0.1 * T.ML_M
    for _ in range(round(gap / DT)):
        r.travel += v * DT
        r.step(v_tr=v)
    r.push(0.5 - gap * 0.1).retract()
    assert r.tr.thumb_lost == (want < 1.0) and r.tr.rate_ok() and r.end() == want


def test_thumb_off_before_s_is_not_scored():
    """The thumb-on clause starts at S: the hand easing off at the breakaway (before S) does not count."""
    r = Run().uncap().insert().step(60)
    r.thumb = T.BREAKAWAY_F
    r.step()
    r.thumb = 0.0
    r.step(round(0.5 / DT))  # off the thumb rest, the stopper still (no S yet)
    assert r.tr.t_S is None
    r.push(1.0).retract()
    assert not r.tr.thumb_lost and r.end() == 1.0


def test_the_slam_fails_the_rate():
    """A near-bolus (0.8 mL in about 2 s, then a wait before the 0 line) must fail. Here: to the 0.2 line
    at 0.5 s per 0.2 mL (4 times the taught rate), then 0.2 mL in 2 s: the fast intervals fail 1-4 s: 0.75."""
    r = Run().uncap().insert().step(60).push(0.8, s_per_02=0.5).push(0.2).retract()
    assert not r.tr.rate_ok() and r.end() == 0.75
    r2 = Run().uncap().insert().step(60).push(1.0, s_per_02=0.3).retract()  # the whole dose in about 1.5 s
    assert not r2.tr.rate_ok() and r2.end() == 0.75


def test_the_first_interval_is_prorated_and_can_fail_alone():
    """S at the 0.9 line: the first interval is the 0.1 mL from S to the 0.8 line, allowed 1-4 s per 0.2 mL prorated
    (0.5-2 s). That stretch at 0.5 s per 0.2 mL (0.25 s) fails though every later interval is 2 s."""
    r = Run().uncap().insert().step(60)
    r.push(0.101, s_per_02=2.0)  # to S
    assert r.tr.t_S is not None
    first = r.tr.vol_S - 0.8 + 0.001
    r.push(first, s_per_02=0.5).push(0.8).retract()
    later = [b[1] - a[1] for a, b in zip(r.tr.crossings, r.tr.crossings[1:])]
    assert all(1.0 <= g <= 4.0 for g in later), later
    assert not r.tr.rate_ok() and r.end() == 0.75


def test_breakaway_is_the_thumb_load_never_the_travel():
    """On the GPU J_stick's limit gave 0.4 mm under 5.9 N (short of the 8 N breakaway then in use) and the demo, reading travel,
    took that for the breakaway; the rule's stuck flag cleared only later. The rule breaks the stopper loose only on the
    measured thumb load, and records when and at what load; the demo's PushCtl waits for that flag."""
    import demo as D

    r = Run().uncap()
    r.thumb = T.BREAKAWAY_F - 0.1
    r.travel = T.STICK_PLAY + 0.0004  # the stop giving under load
    r.step(48)
    assert r.tr.stuck and r.tr.t_unstick is None
    r.thumb = T.BREAKAWAY_F
    r.step()
    assert not r.tr.stuck and r.tr.unstick_n == pytest.approx(T.BREAKAWAY_F) and r.tr.t_unstick == pytest.approx(r.tr.t)
    ctl = D.PushCtl(0.0002, ZERO_TRAVEL=T.ZERO_TRAVEL, TRAVEL_MAX=T.TRAVEL_MAX, DETENT_F=T.DETENT_F, BREAKAWAY_F=T.BREAKAWAY_F)
    ctl.step(0.0002, 1.0, False)  # touched
    for _ in range(40):
        ctl.step(0.0010, T.BREAKAWAY_F - 0.1, False)  # 0.8 mm past its start, the rule still stuck: still pressing
    assert ctl.phase == "press" and not ctl.released
    ctl.step(0.0010, T.BREAKAWAY_F, True)
    assert ctl.phase == "push" and ctl.released


def test_press_reaches_the_breakaway_and_the_detent_on_the_softest_arm():
    """Every PushCtl travel cap is a force over the softest arm measured (165 N/m): the press can reach BREAKAWAY_F + 1 N,
    the push holds more than the glide, and the lead reaches DETENT_F + 2 N against the 0 line's stop."""
    import demo as D

    ctl = D.PushCtl(0.0, ZERO_TRAVEL=T.ZERO_TRAVEL, TRAVEL_MAX=T.TRAVEL_MAX, DETENT_F=T.DETENT_F, BREAKAWAY_F=T.BREAKAWAY_F)
    k = D.ARM_K_MIN
    assert k * ctl.press_max >= T.BREAKAWAY_F + D.PRESS_OVER - 1e-9
    push_f = T.F_GLIDE + T.FLOW_C * D.PUSH_V  # the glide and the liquid's resistance at the taught rate (about 3.3 N)
    ctl = D.PushCtl(0.0, ZERO_TRAVEL=T.ZERO_TRAVEL, TRAVEL_MAX=T.TRAVEL_MAX, DETENT_F=T.DETENT_F, BREAKAWAY_F=T.BREAKAWAY_F, PUSH_F=push_f)
    assert k * ctl.lead_cap >= push_f + D.PUSH_OVER - 1e-9 and k * ctl.lead_detent >= T.DETENT_F + D.DETENT_OVER - 1e-9
    cross, _flips, _glide, retracted = _arm_push(k, T.FLOW_C, 100.0, 0.5)
    press_max, surge, off_max = _arm_push.last
    gaps = [b[1] - a[1] for a, b in zip(cross, cross[1:])]
    assert retracted and len(cross) == 5 and all(1.6 <= g <= 2.4 for g in gaps), gaps
    assert T.BREAKAWAY_F <= press_max <= T.BREAKAWAY_F + 1.5 and off_max < T.THUMB_GAP_S


def test_not_retracted_misses_full_dose():
    """The edge to the 0 line but not on through the detent: the needle stays out, no full dose: 0.5."""
    r = Run().uncap().insert().step(30).push(1.0)
    r.step(240)
    assert not r.tr.retracted and r.end() == 0.5


def test_detent_gives_only_to_10_n_on_the_0_line():
    r = Run().uncap().insert().step(30).push(1.0)
    assert r.tr.at_stop and not r.tr.stuck
    r.thumb = T.DETENT_F - 0.5  # 9.5 N on the 0 line: the stop holds
    r.step(24)
    assert r.tr.at_stop and not r.tr.retracted
    r.retract()
    assert not r.tr.at_stop and r.tr.retracted and abs(r.tr.plunger_n) <= T.F_GLIDE + 1e-9 and not r.tr.engaged  # the glide holds it


def test_retraction_cannot_latch_without_the_thumb_through_the_0_line():
    """Spike 1: the stopper ran to its stop with no thumb on it (the barrel loaded along its axis) and the retraction
    latched. Travel alone never retracts: the stop at the 0 line gives only to 10 N of thumb load there."""
    r = Run().uncap()
    r.travel = T.RETRACT_AT + 0.0003
    r.step(240, v_tr=0.05)
    assert not r.tr.retracted and r.tr.at_stop and r.tr.t_retract is None
    r2 = Run().uncap().insert().step(30)
    r2.thumb = 15.0  # a hard press well short of the 0 line does not release the stop
    r2.travel = T.ZERO_TRAVEL - 0.002
    r2.step(24)
    assert r2.tr.at_stop
    r2.thumb = 0.0  # on through the 0 line with nothing on the thumb rest
    r2.travel = T.RETRACT_AT
    r2.step(24)
    assert r2.tr.at_stop and not r2.tr.retracted


def test_stopper_held_until_a_thumb_breaks_it_loose():
    """The cap pull and the grip test load the barrel with nothing on the rod: J_stick stays on, so travel stays 0."""
    r = Run(cap_on=True)
    acts = []
    orig = r.tr.feed
    r.tr.feed = lambda s: acts.extend(orig(s)) or acts
    r.thumb = T.BREAKAWAY_F - 0.5
    r.cap_slide = 0.002
    r.step(240)
    r.cap_slide = 0.003  # the cap pulled off
    r.step(24)
    assert r.tr.stuck and "unstick" not in acts and "cap_off" in acts and r.tr.plunger_n == 0.0
    r.thumb = T.BREAKAWAY_F
    r.step()
    assert not r.tr.stuck and acts[-1] == "unstick"


def test_plunger_stiction_and_the_0_line_stop_are_both_ranged_limits():
    """The stopper's static friction: a hard stop with 0.2 mm of play (J_stick), released by the rule at 3 N of
    measured thumb load; the 0 line's stop: a D6 limit on the slide. Probe 1: a 2e4 N/m drive acted at about 1 kN/m on
    the GPU and the pinch drove the plunger to J_zero's limit, which held; so both are limits of the same form (lower
    -1 mm, upper above it, no drive, no break force, re-armed at a reset alike)."""
    js = T.yb._JOINTS
    hold = js["J_stick"]
    assert hold.kind == "d6" and hold.on and hold.body1 == "plunger" and hold.drive is None
    assert dict(hold.d6_limits) == {"transX": (-0.001, T.STICK_PLAY)} and 0.0 < T.STICK_PLAY < T.ZERO_TOL
    assert [n for n, _, _ in T.STOPS] == ["J_zero"]  # spike 4: seven joints on the pair jammed the slide
    for n, at, need in T.STOPS:
        j = js[n]
        assert j.kind == "d6" and j.on and j.break_force is None and not j.remake and dict(j.d6_limits) == {"transX": (-0.001, at)}
        assert need == T.DETENT_F
        assert (j.anchor, j.anchor1, j.frame_quat, j.remake, j.rebuilt) == (hold.anchor, hold.anchor1, hold.frame_quat, hold.remake, hold.rebuilt)
    assert hold.break_force is None and not hold.rebuilt and "J_end" not in js and "J_hold" not in js


def test_stops_pass_only_in_order_and_only_at_their_load():
    tr = T.Trial()
    tr.cap_on = False
    feed = lambda travel, thumb: tr.feed(T.Sample(DT, travel=travel, thumb_n=thumb))
    assert feed(0.0, 2.9) == [] and tr.limit() == 0.0  # under the breakaway: still stuck
    assert feed(0.0, 3.0) == ["unstick"] and tr.limit() == pytest.approx(T.STOPS[0][1])
    assert feed(T.ZERO_TRAVEL - 0.001, 15.0) == [] and tr.passed == 0  # pressed hard 1 mm short of the 0 line: holds
    assert feed(T.ZERO_TRAVEL, 9.5) == [] and tr.passed == 0  # on the stop, under 10 N
    assert feed(T.ZERO_TRAVEL, 10.0) == ["detent"] and tr.limit() == T.TRAVEL_MAX


def test_slide_velocity_is_read_from_its_position():
    """Spikes 3-4: the centre-of-mass velocities read 5-10 mm/s (and about 1 m/s) along the axis with the slide still;
    that offset, fed to the glide friction, pushed the stopper at 2 N. The sensing differentiates the slide's position."""
    src = Path(T.__file__).read_text()
    assert "tv = 0.0 if prev is None else (travel - float(prev)) / dt" in src
    assert "lin_vel - sy.lin_vel" not in src


def test_gpu_caveat_no_runtime_enable_of_plunger_or_authored_off_joint_but_the_tissue():
    """CPU vs GPU caveat (spikes 2-3, and the chainsaw build): switching a joint OFF at run time reached the GPU (the
    carrier's lock, the cap's fit), PhysX break forces broke under barrel loads with no thumb, and runtime drive targets
    did not reach it. So the plunger's joints are unbreakable and only ever switched off by the rule; the only joints
    switched ON at run time or authored off are the needle's carrier lock (re-locked after a withdrawal) and the tissue
    joint, whose effect is a spike gate (needle in at 90 deg; the retraction's carrier travel)."""
    switched_on = {j for acts in T.JOINT_OF.values() for j, on in acts if on}
    plunger = {"J_stick", "J_plunger", *[n for n, _, _ in T.STOPS]}
    assert switched_on == {"J_lock", "J_tissue"} and not switched_on & plunger
    assert {n for n, j in T.yb._JOINTS.items() if not j.on} == {"J_tissue"}
    # spike 7: not toggling the hold at the reset changed nothing, so J_stick is re-armed as J_zero is (no joint is
    # re-made). No joint carries a break force (spike 6: 3e38 N broke both locks).
    assert {n for n, j in T.yb._JOINTS.items() if j.rebuilt} == set()
    assert all(j.break_force is None for j in T.yb._JOINTS.values())


def test_no_break_force_above_1e9_n():
    """Spike 6: a 3e38 N 'unbreakable' break force (near float32's 3.4e38) left both plunger locks not holding on the
    GPU. Any break force a task file authors stays at or below 1e9 N (use 1e6 N for 'unbreakable')."""
    assert all(j.break_force is None or j.break_force <= 1e9 for j in T.yb._JOINTS.values())


def test_every_limit_has_low_at_most_high():
    """No limit uses the 'low > high means locked' convention (GPU PhysX may read that axis as free); a lock is an
    explicit (0, 0) range."""
    for name, j in T.yb._JOINTS.items():
        lims = [*dict(j.d6_limits).values()]
        if j.limits is not None:
            lims.append(j.limits)
        if j.limits_deg is not None:
            lims.append(j.limits_deg)
        for lo, hi in lims:
            assert lo <= hi, (name, lo, hi)
    assert dict(T.yb._JOINTS["J_stick"].d6_limits)["transX"][0] < dict(T.yb._JOINTS["J_stick"].d6_limits)["transX"][1]  # a range, not (0, 0)


def test_rebuilt_joints_are_not_toggled_at_reset():
    """yam_bodies re-arms ordinary joints at a reset by switching them off and on, but leaves joints with a break force
    to be re-made from scratch (the path spike 3's working hold took)."""
    src = (HERE / "yam_bodies.py").read_text()
    assert "if j.on and not j.rebuilt:" in src and "if not j.rebuilt:  # breakable and remake joints" in src


def test_plunger_glide_is_an_applied_2_n_friction_from_breakaway_to_retraction():
    """The stopper's ~2 N glide is an applied force pair: none while J_stick holds; after the breakaway F_GLIDE against
    the slide's motion either way once the rubber's GLIDE_Z0 of presliding is taken up, whatever the speed; the barrel
    takes minus it; and on after the retraction, holding the retracted plunger (a VanishPoint's plunger is locked down)."""
    assert T.F_GLIDE == 2.0 and T.F_GLIDE < T.BREAKAWAY_F
    r = Run().uncap()
    r.step(10, v_tr=0.00169)
    assert r.tr.plunger_n == 0.0  # stuck: static friction (a joint), no force
    r.thumb = T.BREAKAWAY_F
    v = 0.1 * T.ML_M  # 0.1 mL/s
    r.step(round(T.GLIDE_Z0 / v / DT) + 2, v_tr=v)
    assert r.tr.plunger_n == pytest.approx(-T.F_GLIDE)
    assert r.tr.glide.mean_sliding() > 1.0 and r.tr.glide.flips == 0
    a = r.a
    base = np.asarray(r.tr.syr_force, float) - r.tr.carrier_n * a - r.tr.needle_n * a
    np.testing.assert_allclose(T.syringe_push(r.tr, a) - base, -r.tr.plunger_n * a)  # the reaction on the barrel
    r.step(round(2 * T.GLIDE_Z0 / 0.002 / DT) + 2, v_tr=-0.002)
    assert r.tr.plunger_n == pytest.approx(T.F_GLIDE)  # drawn back: against that motion
    r.step(5, v_tr=0.0)
    assert r.tr.plunger_n == pytest.approx(T.F_GLIDE)  # at rest the rubber's shear holds (it springs back at most Z0)
    r.step(240, v_tr=0.02)
    assert r.tr.plunger_n == pytest.approx(-T.F_GLIDE)  # ten times faster: the same (the rate term is the joint's damping)
    g = T.Glide()
    g.step(0.001, DT, True)
    assert g.step(0.0, DT, False) == 0.0 and g.z == 0.0  # off (stuck): no force, the shear let go
    src = Path(T.__file__).read_text()
    assert "self.glide.step(float(smp.travel_vel) * smp.dt, smp.dt, not self.stuck)" in src  # on after the retraction too


def test_flow_resistance_is_the_articulation_joints_damping():
    """The liquid through the 25 G x 1 in needle (Hagen-Poiseuille, water) is J_plunger's zero-velocity damping drive, and
    the barrel and plunger are one reduced-coordinate articulation so the solver integrates it exactly (as two free
    bodies on a maximal joint it acted at about 63 N s/m of the authored value with the barrel held by contact, and the
    glide's force pair cancelled). The constant is the formula's, not a tuned one."""
    mu, L, r, A = 1.0e-3, 0.0254, 0.13e-3, 1.0e-6 / T.ML_M  # Pa s, m, m (bore 0.26 mm), m2 (1 mL per ML_M of travel)
    c = 8.0 * mu * L * A * A / (math.pi * r ** 4)
    assert c == pytest.approx(793.0, abs=1.0) and T.FLOW_C == pytest.approx(c, rel=0.01)
    assert T.FLOW_C * 0.1 * T.ML_M == pytest.approx(1.34, abs=0.01)  # N at the taught 0.1 mL/s
    j = T.yb._JOINTS["J_plunger"]
    assert j.kind == "prismatic" and j.drive == (0.0, T.FLOW_C, 0.0) and j.articulated
    assert T.SYRINGE.articulation_root and not T.PLUNGER.articulation_root
    others = [x for x in T.yb._JOINTS.values() if x.name != "J_plunger"]
    assert others and not any(x.articulated for x in others)  # the stops, the lock, the cap, the tissue: maximal joints as before
    src = (HERE / "yam_bodies.py").read_text()
    assert "joint.CreateExcludeFromArticulationAttr(not j.articulated)" in src and "UsdPhysics.ArticulationRootAPI.Apply(prim)" in src
    assert "view.set_dof_positions(zeros, idx)" in src  # a reset puts the plunger back through the articulation
    assert T.yb._art_root("plunger") == "syringe" == T.yb._art_root("syringe") and T.yb._art_root("cap") is None
    T.yb._ART_FORCES.clear()
    T.yb.push(None, "plunger", (-2.0, 0.0, 0.0))  # a link's force is queued for the articulation's single call, not sent per body
    T.yb.push(None, "syringe", (2.0, 0.0, 0.0), (0.0, 0.1, 0.0))
    q = T.yb._ART_FORCES["syringe"]
    assert q["plunger"][0].tolist() == [-2.0, 0.0, 0.0] and q["syringe"][0].tolist() == [2.0, 0.0, 0.0] and q["syringe"][1].tolist() == [0.0, 0.1, 0.0]
    T.yb._ART_FORCES.clear()
    assert "_flush_art_forces(env)" in src and "is_global=True)" in src


def test_glide_is_explicitly_stable_at_240_hz():
    """The velocity law F = -F_GLIDE tanh(v / V_EPS) at V_EPS 0.5-1 mm/s has slope*dt/m of thousands on the 2 g plunger and
    10-50 with a 0.2-1 kg hand on it (it must stay under 1), and clamped to m |v| / dt it gives under 0.001 N at 0.1 mL/s.
    The elasto-plastic glide's stiffness F_GLIDE / GLIDE_Z0 keeps k dt / c under 1 on the free plunger against the joint's
    implicit damping (FLOW_C; also at the least a 25 G bore allows, 500 N s/m), and k dt^2 / m under 1 with a 0.2 kg hand on it."""
    k = T.F_GLIDE / T.GLIDE_Z0
    assert k * DT / T.FLOW_C < 0.05 and k * DT / 500.0 < 0.05 and k * DT * DT / 0.2 < 0.5
    for v_eps in (0.0005, 0.001):
        assert T.F_GLIDE / v_eps * DT / 0.002 > 1000 and T.F_GLIDE / v_eps * DT / 1.0 > 8
    assert 0.002 * (0.1 * T.ML_M) / DT < 0.001


def _free_stopper(c_eff, noise, v0, seconds=2.0, seed=0):
    """A free 2 g plunger on the drive's implicit damping c_eff after the breakaway, no thumb, the glide read from a
    jittered position (noise m per step, as the GPU's slide jitter): (start, end, largest travel) in m."""
    rng = np.random.default_rng(seed)
    m, x, v, g = 0.002, 0.005, v0, T.Glide()
    prev, top = x, x
    for _ in range(round(seconds / DT)):
        meas = x + noise * rng.standard_normal()
        f = g.step(meas - prev, DT, True)
        prev = meas
        v = (m * v + DT * f) / (m + c_eff * DT)
        x += v * DT
        top = max(top, x)
    return 0.005, x, top


@pytest.mark.parametrize("c_eff", [500.0, 790.0, 1250.0])
@pytest.mark.parametrize("noise", [0.0, 2e-5])
def test_glide_cannot_drive_a_free_stopper(c_eff, noise):
    """Spike 10: the old velocity friction drove the stopper on with no thumb (0.8 to 0 mL in 1.6 s). Free, at rest or
    let go moving at 10 mm/s, with 0.02 mm of position jitter per step, the elasto-plastic glide only stops it: it never
    runs on beyond where the motion alone (no friction) would coast it, nor creeps on at rest."""
    x0, x1, top = _free_stopper(c_eff, noise, 0.0)
    assert top - x0 < T.GLIDE_Z0 + 1e-4 and abs(x1 - x0) < T.GLIDE_Z0 + 1e-4
    coast = 0.002 * 0.010 / c_eff  # no friction: the damping alone stops it in this far
    x0, x1, top = _free_stopper(c_eff, noise, 0.010)
    assert top - x0 <= coast + 1e-4 and x1 - x0 <= coast + 1e-4


def _arm_push(k, c_eff, c_arm, M, dropout=False):
    """Toy model at 240 Hz: the left hand (mass M, on a k N/m drive from PushCtl's command with damping c_arm, both
    implicit as PhysX's drives) pushing the thumb rest only (a contact that pushes, never pulls); the 2 g plunger on the
    GPU's implicit drive damping c_eff; J_stick until 3 N of contact load (0.2 mm play), the 0 line's stop until 10 N;
    the glide applied explicitly from the slide's displacement per step (T.Glide), off before the breakaway and after the
    retraction. Returns the 0.2 mL lines' crossing times, the plunger's velocity sign flips (0.5 mm/s each side) per
    second from the breakaway to the retraction, the glide's mean magnitude while sliding, and whether it retracted."""
    import demo as D

    m, dt = 0.002, DT
    xp = vp = vh = 0.0
    xh = -D.APPROACH_BACK  # the hand starts 3 mm behind the thumb rest (the demo's approach)
    stuck, through, retracted = True, False, False
    ctl = D.PushCtl(0.0, ZERO_TRAVEL=T.ZERO_TRAVEL, TRAVEL_MAX=T.TRAVEL_MAX, DETENT_F=T.DETENT_F, BREAKAWAY_F=T.BREAKAWAY_F,
                    PUSH_F=T.F_GLIDE + T.FLOW_C * D.PUSH_V)
    g = T.Glide()
    t, N, thumb, prev, t_rel, t_ret, sign, flips = 0.0, 0.0, 0.0, 0.0, None, None, 0, 0
    press_max, x_rel, surge, off, off_max, x_S = 0.0, None, 0.0, 0.0, 0.0, None
    lines = [T.ZERO_TRAVEL - v * T.ML_M for v in (0.8, 0.6, 0.4, 0.2, 0.0)]
    cross = []
    Me = M + dt * c_arm + dt * dt * k  # the arm's implicit effective mass
    mc = m + c_eff * dt
    for step in range(round(40.0 / 0.05)):
        read = 0.0 if (dropout and step % 2) else thumb  # (dropout: the contact view reads 0 every other control step)
        cmd = ctl.step(xp, read, not stuck)
        thumb = 0.0
        for _ in range(12):
            t += dt
            if stuck and N >= T.BREAKAWAY_F:
                stuck, t_rel = False, t
            if not stuck and not through and xp >= T.ZERO_TRAVEL - T.ZERO_TOL and N >= T.DETENT_F:
                through = True
            f = g.step(xp - prev, dt, not stuck and not retracted)
            prev = xp
            vh_f = (M * vh + dt * k * (cmd - xh)) / Me
            vp_f = (m * vp + dt * f) / mc
            touching = xh + vh_f * dt >= xp + vp_f * dt - 1e-12
            vh_n, vp_n = (((Me * vh_f + mc * vp_f) / (Me + mc),) * 2) if touching else (vh_f, vp_f)
            lim = T.TRAVEL_MAX if through else (T.STICK_PLAY if stuck else T.ZERO_TRAVEL)
            at_stop = xp + vp_n * dt > lim
            if at_stop:
                vp_n = max(0.0, (lim - xp) / dt)
                vh_n = vp_n if touching else vh_n
            N = (Me * (vh_f - vh_n) if at_stop else mc * (vp_n - vp_f)) / dt if touching else 0.0
            xp, vp = xp + vp_n * dt, vp_n
            xh, vh = (xp if touching else xh + vh_n * dt), vh_n
            thumb = max(thumb, N)
            if stuck:
                press_max = max(press_max, N)
            elif x_rel is None:
                x_rel = xp
            elif t - t_rel <= 0.25:
                surge = max(surge, xp - x_rel - D.PUSH_V * (t - t_rel))
            if t_rel is not None and not retracted and abs(vp) > 0.0005:
                s = 1 if vp > 0 else -1
                flips += int(sign != 0 and s != sign)
                sign = s
            for x in lines:
                if xp >= x - T.LINE_TOL and x not in [c[0] for c in cross]:
                    cross.append((x, t))
            if x_S is None and T.volume(xp) <= T.s_line(1.0) + 1e-9:
                x_S = xp  # S: the 0.9 graduation (one below the start's reading, 1.0 mL)
            if x_S is not None and len(cross) < 5:  # S to the 0 line: the thumb-on clause's reading
                off = off + dt if N < T.THUMB_ON_N else 0.0
                off_max = max(off_max, off)
            if xp >= T.RETRACT_AT and not retracted:
                retracted, t_ret = True, t
        if retracted:
            break
    span = (t_ret if retracted else t) - (t_rel if t_rel is not None else 0.0)
    _arm_push.last = (press_max, surge, off_max)
    return cross, flips / max(span, 1e-9), g.mean_sliding(), retracted


@pytest.mark.parametrize("k", [165.0, 300.0, 500.0, 700.0, 1500.0, 3000.0])
@pytest.mark.parametrize("c_eff", [500.0, 790.0, 1250.0])
@pytest.mark.parametrize("c_arm", [30.0, 100.0, 300.0])
@pytest.mark.parametrize("M", [0.2, 0.5, 1.0])
@pytest.mark.parametrize("dropout", [False, True])
def test_push_is_steady_against_the_applied_glide(k, c_eff, c_arm, M, dropout):
    """The demo's PushCtl, as it is, on a springy arm pushing the thumb rest only, against the 3 N stiction, the applied
    2 N glide, the flow's damping and the 10 N detent (toy model, `_arm_push`: arms of 165-3,000 N/m, the flow at the
    least, nominal and most a 25 G bore gives, 500 / 790 / 1,250 N s/m, with and without a contact reading that drops out
    every other step): each 0.2 mL line inside the rule's band and, on the measured arms, 1.6-2.4 s after the previous
    one; the glide at about 2 N while sliding; the thumb never off; the retraction reached; no chatter."""
    cross, flips_s, glide, retracted = _arm_push(k, c_eff, c_arm, M, dropout)
    assert len(cross) == 5 and retracted, cross
    press_max, surge, off_max = _arm_push.last  # E2: a 12 N impact and a 10 mm surge; gentle contact and the release keep both down
    assert press_max <= T.BREAKAWAY_F + 1.5 and surge <= 0.001, (press_max, surge)
    assert off_max < T.THUMB_GAP_S, off_max  # the thumb on the thumb rest from S to the 0 line (the rule's thumb-on clause)
    gaps = [b[1] - a[1] for a, b in zip(cross, cross[1:])]
    assert all(T.STEP_S[0] <= g <= T.STEP_S[1] for g in gaps), gaps  # the rule's band, on every arm (no runaway)
    if k <= 700.0 and not dropout and c_eff == T.FLOW_C:  # the demo's own margin, on the measured arms at the nominal bore
        assert all(1.6 <= g <= 2.4 for g in gaps), gaps
        assert 1.7 <= glide <= 2.05, glide
    assert flips_s < 1.0, flips_s


def _engaged_trial():
    tr = T.Trial()
    tr.cap_on = False
    tr.engaged, tr.cur, tr.locked, tr.stuck = True, 0, False, False
    entry = np.array(T.GEOM.site)
    n_out = T.GEOM.normal(entry[1], entry[2])
    tr.punctures.append(T.Puncture(entry=entry, n_out=n_out, D=0.037, L=0.0, engaged=True))
    a = tuple(-n_out)
    feed = lambda travel, gap=0.0: tr.feed(T.Sample(DT, hub=tuple(entry + n_out * gap), axis=a, travel=travel))  # noqa: E731
    return tr, feed


def test_pushed_volume_counts_forward_progress_not_jitter():
    """Spike 10 summed 7.2 mL from a reading that jittered about +-0.1 mm per physics step at the 0 line's stop: the
    tally counts only travel beyond the furthest reached while in muscle."""
    tr, feed = _engaged_trial()
    feed(0.0)
    for x in np.linspace(0.0, 0.010, 50):
        feed(float(x))
    for _ in range(500):  # jitter at 10 mm
        feed(0.0101)
        feed(0.0099)
    assert tr.pushed_in_muscle == pytest.approx(0.0101, abs=1e-6)


def test_pushed_volume_credits_only_travel_in_muscle_and_never_the_detent():
    """Scoring review: travel pushed while engaged but shallow (gap > 5 mm) was credited once the needle dipped back in,
    and the retraction's detent travel past the 0 line counted as dose (spike 11: 1.118 mL)."""
    tr, feed = _engaged_trial()
    feed(0.0)
    feed(0.002)  # in muscle: 2 mm
    for x in np.linspace(0.002, 0.008, 20):  # pulled out to an 8 mm gap and pushed 6 mm there
        feed(float(x), gap=0.008)
    assert not tr.in_muscle_now
    feed(0.008)  # back in: nothing credited for the shallow push
    assert tr.pushed_in_muscle == pytest.approx(0.002, abs=1e-9)
    for x in np.linspace(0.008, T.TRAVEL_MAX, 40):  # on through the 0 line and the detent, in muscle
        feed(float(x))
    assert tr.pushed_in_muscle == pytest.approx(0.002 + T.ZERO_TRAVEL - 0.008, abs=1e-9)
    assert tr.pushed_in_muscle / T.ML_M <= 1.0 + 1e-9


def test_bare_syringe_set_back_on_its_rest_is_a_zero():
    """Scoring review: the text zeros a bare-needle syringe set down anywhere outside the container; the sensing no longer
    exempts its rest."""
    src = Path(T.__file__).read_text()
    i = src.index('SENSE["loose"] = SENSE.get("loose", 0) + 1 if (')
    line = src[i:src.index("\n", i)]
    assert "on_rest" not in line and "not inside" in line
    r = Run().uncap()
    r.tr.status(T.Status(dropped=True))
    assert r.tr.zero and "set down" in r.tr.zero


def test_bare_needle_touch_reads_every_arm_link_and_stow_counts_the_plunger():
    src = Path(T.__file__).read_text()
    assert '_forces(f"needle_{side}_arm", "needle", _arm_globs(side, ARM_LINKS), dt)' in src
    assert "shape_corners(PLUNGER, plunger_state)" in src and 'stow_check(sy, yb.body_state(env, "plunger"), cap)' in src
    assert set(T.ARM_LINKS) >= {"link1", "link2", "link3", "link4", "link5", "gripper", "tip_left", "tip_right"}


def test_stow_checks_use_the_bodies_own_boxes():
    """Spike 10: the syringe lay inside the sharps container and the cap inside the cup, but the body-wide boxes
    (the flange's 25 mm height along the whole barrel; the turned cap's box) reached past the floor and the wall."""
    sh = T.yb.state_from_pose
    tang = (0.0, 0.0, -0.3826834, 0.9238795)  # the cap turned 45 deg about z, lying tangentially in the cup
    c = sh(T.CAP_BODY, (T.CUP_AT[0] + 0.018, T.CUP_AT[1] + 0.018, T.TOP + 0.007), tang)
    assert T.in_cup(None, None, T.shape_corners(T.CAP_BODY, c)) and not T.in_cup(c.low, c.high)
    far = sh(T.CAP_BODY, (T.CUP_AT[0] + 0.035, T.CUP_AT[1], T.TOP + 0.007), (0.0, 0.0, 0.0, 1.0))
    assert not T.in_cup(None, None, T.shape_corners(T.CAP_BODY, far))
    lo, hi = T.SHARPS_INSIDE
    th = math.atan((0.0125 - T.BARREL_HALF) / 0.0605)  # resting on its nose and its flange's lower edge, hub end down
    q = (0.0, math.sin(th / 2), 0.0, math.cos(th / 2))
    mid = (lo + hi) / 2
    s0 = sh(T.SYRINGE, (mid[0] + 0.03, mid[1], 0.0), q)
    z = lo[2] + 0.0005 - float(T.shape_corners(T.SYRINGE, s0)[:, 2].min())
    lying = sh(T.SYRINGE, (mid[0] + 0.03, mid[1], z), q)
    p = T.shape_corners(T.SYRINGE, lying)
    assert T.in_container(p.min(0), p.max(0)) and not T.in_container(lying.low, lying.high)


def test_cup_check_is_the_built_decagon_not_its_apothem_circle():
    """Spike 11: the cap rested in the cup against its wall near a corner of the 10-sided wall (corners 43.4 mm out,
    inside the 43 mm inner faces), and the old single-radius rule (r <= 43 mm) read it as outside."""
    sh = T.yb.state_from_pose
    a = math.pi / 10  # toward a vertex of the decagon
    z = T.TOP + 0.003 + T.CAP_HALF
    yaw = a + math.pi / 2  # lying tangentially, against the wall
    q = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
    at = lambda r: sh(T.CAP_BODY, (T.CUP_AT[0] + r * math.cos(a), T.CUP_AT[1] + r * math.sin(a), z), q)  # noqa: E731
    inside = T.shape_corners(T.CAP_BODY, at(0.034))
    assert np.linalg.norm(inside[:, :2] - np.asarray(T.CUP_AT), axis=1).max() > 0.043  # the old rule failed this
    assert T.in_cup(None, None, inside) and T.cup_excess(None, None, inside) <= T.CONTACT_TOL
    through = T.shape_corners(T.CAP_BODY, at(0.038))  # about 4 mm into the wall: not inside
    assert not T.in_cup(None, None, through) and T.cup_excess(None, None, through) > T.CONTACT_TOL
    rim = T.shape_corners(T.CAP_BODY, sh(T.CAP_BODY, (T.CUP_AT[0] + 0.045, T.CUP_AT[1], T.TOP + T.CUP_H), (0.0, 0.0, 0.0, 1.0)))
    assert not T.in_cup(None, None, rim)  # on the rim, half out
    assert T.CUP_INNER == pytest.approx(0.043) and T.CUP_SIDES == 10


def test_cup_constants_match_the_built_ring():
    walls = [s for s in T.CUP.shapes if abs(s.size[2] - T.CUP_H) < 1e-9]
    assert len(walls) == T.CUP_SIDES
    for s in walls:
        r = math.hypot(s.centre[0], s.centre[1])
        assert r - s.size[0] / 2 == pytest.approx(T.CUP_INNER, abs=1e-9)


def test_container_excess_names_the_face():
    lo, hi = T.SHARPS_INSIDE
    p = np.array([(lo + hi) / 2, (lo + hi) / 2 + [0.0, 0.0, (hi[2] - lo[2]) / 2 + 0.004]])  # 4 mm through the lid
    ex = T.container_excess(p)
    assert ex["lid"] == pytest.approx(4.0, abs=0.05) and max(v for k, v in ex.items() if k != "lid") < 0


@pytest.mark.parametrize("kw", [{"in_container": False}, {"cap_in_cup": False}, {"clear": 0.03}])
def test_not_stowed_or_gripper_still_holding_is_075(kw):
    """Not in sharps, the cap not in the cup, or a gripper still within 5 cm (still holding): 0.75."""
    assert Run().full().end(**kw) == 0.75


def test_hidden_more_than_1_s_loses_needle_still():
    r = Run().uncap().insert().step(30).push(0.3)
    for _ in range(21):  # 1.05 s of hidden frames inside the window
        r.tr.status(T.Status(hidden=True))
    r.push(0.7).retract()
    assert r.tr.hidden_s > 1.0 and r.end() == 0.75
    r2 = Run().uncap().insert().step(30).push(0.3)
    for _ in range(19):
        r2.tr.status(T.Status(hidden=True))
    r2.push(0.7).retract()
    assert r2.end() == 1.0


def test_hidden_frames_count_only_from_s_to_the_retraction():
    r = Run().uncap().insert().step(30)
    for _ in range(40):
        r.tr.status(T.Status(hidden=True))  # before S
    r.push(1.0).retract()
    for _ in range(40):
        r.tr.status(T.Status(hidden=True))  # after the retraction
    assert r.tr.hidden_s == 0.0 and r.end() == 1.0


def _off(deg):
    r = math.radians(deg)
    return (math.cos(r), math.sin(r), 0.0)


def test_bent_needle_is_an_outcome_judged_on_the_needle_after_uncapping():
    """Scoring review: the bend is judged as a person can, on the needle itself after uncapping (its axis more than
    10 deg off the barrel's, seen in the side camera's frame), not from the hand's path during a 3 mm pull."""
    r = Run(cap_on=True)
    r.needle_axis = _off(20.0)
    r.step()
    assert r.tr.zero is None  # capped: not judged
    r.uncap()
    assert r.tr.zero and "bent" in r.tr.zero
    r2 = Run().uncap()
    r2.needle_axis = _off(8.0)
    r2.step(5)
    assert r2.tr.zero is None
    r2.needle_axis = _off(11.0)
    r2.step()
    assert "bent" in r2.tr.zero
    assert "cap_pull" not in Path(T.__file__).read_text() and not hasattr(T, "CAP_BEND_DEG")


def test_bare_needle_touching_anything_else_is_a_permanent_zero():
    r = Run().uncap()
    r.touch = "the right gripper"
    r.step()
    r.touch = None
    r.full()
    assert r.end() == 0.0
    r2 = Run().full()
    r2.tr.status(T.Status(touch="the tray"))
    assert r2.tr.zero is None  # retracted: nothing to touch
    r3 = Run(cap_on=True)
    r3.touch = "the tray"
    r3.step()
    assert r3.tr.zero is None  # capped


def test_dropped_outside_the_container_is_a_permanent_zero():
    """The bare, unretracted needle dropped: 0 whatever follows. A retracted one dropped is only not stowed."""
    r = Run().uncap().insert().step(30)
    r.tr.status(T.Status(dropped=True))
    r.push(1.0).retract()
    assert r.end() == 0.0
    r2 = Run().full()
    r2.tr.status(T.Status(dropped=True))
    assert r2.tr.zero is None


def test_tiers_are_monotonic():
    base = {"zero": False, "punctures": 1, "D": 0.0375, "L": 0.0, "A": 5.0, "full_depth": True, "continuous": True, "still": True, "full_dose": True, "steady": True,
                "stowed": True, "held": True, "pushed_ml": 1.0, "any_025": True}
    assert T.tier(**base) == 1.0
    for k, v, want in (("held", False, 0.75), ("stowed", False, 0.75), ("steady", False, 0.75), ("still", False, 0.75), ("full_depth", False, 0.75),
                       ("A", 12.0, 0.75), ("A", 17.0, 0.5), ("continuous", False, 0.5), ("full_dose", False, 0.5), ("D", 0.022, 0.5),
                       ("L", 0.012, 0.5), ("punctures", 2, 0.25), ("zero", True, 0.0)):
        assert T.tier(**(base | {k: v})) == want, (k, v)
    assert T.tier(**(base | {"continuous": False, "pushed_ml": 0.49})) == 0.25
    assert T.tier(**(base | {"continuous": False, "pushed_ml": 0.49, "any_025": False})) == 0.0


def test_volume_arithmetic_matches_the_plan():
    assert T.ML_M * 0.1 == pytest.approx(0.00169)
    assert 0.1e-6 / (math.pi * 0.00433 ** 2) == pytest.approx(0.0017, abs=0.00002)
    assert -T.EDGE0 == pytest.approx(0.0269) and -T.BAND == pytest.approx(0.0438)
    assert T.ZERO_TRAVEL == pytest.approx(0.0169) and T.RETRACT_AT == pytest.approx(0.0189)


# -- layout -------------------------------------------------------------------------------------------------------------


def test_instruction_matches_task_json_and_run_script():
    j = json.loads((HERE / "task.json").read_text())
    assert j["instruction"] == T.INSTRUCTION
    assert j["instruction"] in (HERE / "tools" / "run_injection.sh").read_text()
    assert "3,600" in j["run_limit"]
    for k in ("title", "category", "instruction", "setup", "scoring", "run_limit", "fidelity", "scores", "media"):
        assert j.get(k), k


@pytest.mark.parametrize("key,limit", [("title", 120), ("instruction", 1000), ("context", 1000), ("setup", 4000), ("scoring", 4000),
                                       ("reset_notes", 1500), ("fidelity", 500), ("scores", 300)])
def test_task_json_fields_within_intake_limits(key, limit):
    assert len(json.loads((HERE / "task.json").read_text())[key]) <= limit


def test_run_script_names_no_user():
    text = (HERE / "tools" / "run_injection.sh").read_text()
    assert "ARENA_DIR" in text and "YAM_EVAL_DIR" in text and re.search(r"^ME=", text, re.MULTILINE) is None
    assert re.search(r"/workspace/(?!\$USER/)", text) is None  # no user's name in a path


def test_task_loads():
    assert TASK.scene == "station"
    assert {o.name for o in T.OBJECTS} >= {"syringe", "needle", "plunger", "cap", "trainer", "stand", "sharps", "cup", "rest", "tray"}
    assert TASK.success_hold_s == 0.0 and TASK.score is not None
    assert "def score(" in (HERE / "im_injection.py").read_text()[:12000]


def test_footprints_do_not_overlap():
    f = T.FOOTPRINTS
    names = list(f)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            (ax0, ax1, ay0, ay1), (bx0, bx1, by0, by1) = f[a], f[b]
            assert ax1 <= bx0 + 1e-9 or bx1 <= ax0 + 1e-9 or ay1 <= by0 + 1e-9 or by1 <= ay0 + 1e-9, (a, b)


@pytest.mark.parametrize("t", [0.050, 0.043])
@pytest.mark.parametrize("h", [0, 1, 2])
def test_trainer_matches_the_setup(h, t):
    """Skin x = 30.285 - 0.1 - t (25.2-25.9 cm); the acromion's edge at 25.0 / 26.83 / 28.66 cm; the trainer 17.5 x 8.5 cm
    from the skin back to the pipe's face (less the sheet)."""
    g = T.Geometry(h=T.HEIGHTS[h], t=t)
    assert 0.2518 <= g.skin_x <= 0.2589
    assert g.z_edge - T.TOP == pytest.approx([0.250, 0.2683, 0.2866][h])
    assert g.z_top - g.z_bottom == pytest.approx(0.175)
    assert g.site[2] == pytest.approx(g.z_edge - 0.0375)
    assert g.normal(0.0, g.site[2]) @ np.array([-1.0, 0.0, 0.0]) == pytest.approx(1.0)
    lo = np.min([np.asarray(s.centre) - s.half_extent() for s in T._trainer_shapes(g) if s.collide], axis=0)
    hi = np.max([np.asarray(s.centre) + s.half_extent() for s in T._trainer_shapes(g) if s.collide], axis=0)
    assert lo[0] == pytest.approx(g.skin_x, abs=0.0002) and hi[0] == pytest.approx(T.PIPE_FACE, abs=0.0002)
    assert lo[2] == pytest.approx(g.z_bottom) and hi[2] == pytest.approx(g.z_top)
    assert hi[1] <= 0.0425 + 0.004 and lo[1] >= -0.0425 - 0.004


def test_skin_normal_of_the_sim_mesh():
    """The sim mesh's local normal: -x on the flat lateral face (|y| <= 25 mm, below the edge), turned 30 deg on the sides."""
    g = T.GEOM
    assert np.allclose(g.normal(0.010, g.site[2]), (-1, 0, 0))
    n = g.normal(0.035, g.site[2])
    assert math.degrees(math.acos(-n[0])) == pytest.approx(30.0) and n[1] > 0
    assert g.surface_x(0.0, g.z_edge + 0.01) == pytest.approx(g.skin_x + T.SHOULDER_BACK)


def test_the_uncapping_place_is_in_clear_space():
    """The uncap (hub at 16, -12, 22 cm; the cap pulled 4 cm on along +x) stays 2 cm or more from the trainer and every
    fixture, and its lowest point is 5 cm or more above the sharps container's top."""
    import demo as D

    P = D.places(T)
    hub = P["uncap"]
    lo = hub + [T.FLANGE[0] - 0.026, -0.013, -0.013]
    hi = hub + [T.CAP[1] + D.PULL_L, 0.013, 0.013]
    for name, blo, bhi in T.static_boxes():
        if name == "camera_mounts":
            continue
        gap = np.maximum(np.maximum(blo - hi, lo - bhi), 0.0)
        assert np.linalg.norm(gap) >= 0.02, name
    assert lo[2] - (T.TOP + T.SH_H) >= 0.05


def test_syringe_rests_on_its_blocks():
    """The flange's lower tab stands in its slot on the slot floor; the cap sits on its notch's floor between the cheeks."""
    rest = {i: (np.asarray(T.REST.pos) + np.asarray(s.centre), s.half_extent()) for i, s in enumerate(T.REST.shapes)}
    floor_f = rest[1][0][2] + rest[1][1][2]
    assert floor_f == pytest.approx(T.SYR_AT[2] - 0.0125, abs=1e-6)
    floor_c = rest[4][0][2] + rest[4][1][2]
    assert floor_c == pytest.approx(T.SYR_AT[2] - T.CAP_HALF, abs=1e-6)
    walls = sorted(rest[k][0][0] for k in (2, 3))
    assert walls[0] + 0.0025 <= T.SYR_AT[0] + T.FLANGE[0] and T.SYR_AT[0] + T.FLANGE[1] <= walls[1] - 0.0025


def test_scoring_cameras_see_the_site_past_the_shoulder():
    """With no arm in the way: both views show the barrel line, the hub and the band at the site, at every height."""
    for h in range(3):
        g = T.Geometry(h=T.HEIGHTS[h])
        boxes = [(lo, hi) for _, lo, hi in T.static_boxes(("stand",))] + [
            (np.asarray(s.centre) - s.half_extent(), np.asarray(s.centre) + s.half_extent()) for s in T._trainer_shapes(g) if s.collide]
        hidden, v = T.hidden_frame(g.site, np.eye(3), None, boxes)
        assert not hidden and v["side"]["hub"] and v["over"]["hub"] and v["over"]["band"], (h, v)


def test_hub_y_from_band():
    for yaw in (0.0, 3.0, -4.0):
        R = np.column_stack([_axis(2.0, yaw), np.cross([0, 0, 1], _axis(2.0, yaw)), [0, 0, 1]])
        assert abs(T.hub_y_from_band(T.GEOM.site, R) - T.GEOM.site[1]) < 0.001


# -- the demo offline: reach, clearance, views, and its score -------------------------------------------------------------

_A = os.path.dirname(YAM_URDF)
_LINK_PTS: dict = {}


def _link_points():
    if not _LINK_PTS:
        import trimesh

        for link in ET.parse(YAM_URDF).getroot().findall("link"):
            v = link.find("visual")
            if v is None or link.get("name") == "base":
                continue
            m = trimesh.load(os.path.join(_A, v.find("geometry/mesh").get("filename")))
            o = v.find("origin")
            M = np.eye(4)
            M[:3, :3] = rpy_matrix(np.array([float(x) for x in o.get("rpy").split()]))
            M[:3, 3] = [float(x) for x in o.get("xyz").split()]
            p = np.asarray(trimesh.sample.sample_surface_even(m, 600, seed=0)[0])
            _LINK_PTS[link.get("name")] = (M[:3, :3] @ p.T).T + M[:3, 3]
    return _LINK_PTS


def arm_points(q, finger, base) -> dict:
    fk = forward_kinematics(YAM_URDF, {**{f"joint{i + 1}": float(v) for i, v in enumerate(q)}, "joint7": finger, "joint8": finger})
    return {n: (fk[n][:3, :3] @ p.T).T + fk[n][:3, 3] + np.asarray(base) for n, p in _link_points().items()}


def _finger(arm, holding) -> float:
    """The finger joint as the simulator has it: a close stops on what it holds (barrel 10.8 mm, cap 8 mm)."""
    f = float(arm.cmd[6]) * FINGER_OPEN
    if holding == "syringe":
        return min(f, -0.0054)
    if holding == "cap":
        return min(f, -0.0040)
    return f


def _dist(points, lo, hi) -> float:
    return float(np.linalg.norm(np.maximum(np.maximum(lo - points, points - hi), 0.0), axis=1).min())


class _OfflineCtx:
    """The demo's context offline: arms that reach every command exactly; the syringe carried by the right hand once it
    closes on it, the cap by the left once it closes on it, the plunger pushed by the left's fingertips; what is let go
    over the sharps container or the cup falls in. Runs the script's own code, and feeds the rule as the sensing would."""

    def __init__(self, mode="full"):
        import yam_demokit as dk

        self.dk, self.mode, self.t = dk, mode, 0.0
        self.emb = type("Emb", (), {"task": TASK, "_env": None})()
        self.left, self.right = dk.Arm(self, "left"), dk.Arm(self, "right")
        self.syr = (np.asarray(T.SYR_AT, float), np.eye(3))
        self.cap = None  # (pos, R) once off the syringe
        self.travel = 0.0
        self.thumb = 0.0  # N the left's fingertips press the thumb rest with (offline: a firm 15 N press while they are in it)
        self.plunger_clear: dict = {}  # m: the holding hand's least distance to the plunger's rod and thumb rest
        self.held = {}  # side -> (what, local pos, local R)
        self.misses = []
        for arm in (self.left, self.right):
            arm.servo = lambda *a, **k: (yield from iter([None] * 5)) or True
            close, move = arm.close, arm.move

            def closed(*a, _arm=arm, _close=close, **k):
                yield from _close(*a, **k)
                self._sync()
                p, R = ik.tcp_pose(_arm.q, _arm.base)
                what = "syringe" if _arm.side == "right" else "cap"
                op, oR = self._pose(what)
                sp, sR = self.syr
                point = sp + sR @ np.array([T.GRASP_R if what == "syringe" else T.GRASP_CAP, 0.0, 0.0])
                if np.linalg.norm(point - p) < 0.015:
                    self.held[_arm.side] = (what, R.T @ (op - p), R.T @ oR)

            arm.close = closed

            def moved(pos, grip=None, finger=None, t=None, label="move", _arm=arm, _move=move):
                g = _arm.grip if grip is None else grip
                if finger is not None and finger <= -0.010 and _arm.side in self.held:
                    self._sync()
                    self._release(_arm.side)
                yield from _move(pos, grip, finger, t, label)
                p, R = ik.tcp_pose(_arm.q, _arm.base)
                mm = 1000 * float(np.linalg.norm(p - (np.asarray(pos, float) + _arm.trim)))
                deg = math.degrees(float(np.linalg.norm(ik._rot_error(g.rot, R))))
                margin = float(np.min(np.minimum(_arm.q - ik.LOWER, ik.UPPER - _arm.q)))
                self.misses.append((_arm.side, label, round(mm, 1), round(deg, 1), round(math.degrees(margin), 1)))

            arm.move = moved

    def _pose(self, what):
        if what == "syringe":
            return self.syr
        if self.cap is not None:
            return self.cap
        p, R = self.syr
        return p + R @ np.array([sum(T.CAP) / 2, 0.0, 0.0]), R

    def _release(self, side):
        what = self.held.pop(side)[0]
        if what == "syringe":
            p, R = self.syr
            lo, hi = self._bounds("syringe")
            if T.SHARPS_INSIDE[0][0] <= lo[0] and hi[0] <= T.SHARPS_INSIDE[1][0] and T.SHARPS_INSIDE[0][1] <= lo[1] and hi[1] <= T.SHARPS_INSIDE[1][1]:
                self.syr = (np.array([p[0], p[1], T.TOP + T.SH_W + 0.0126]), R)  # falls in, onto its flange's lower tab
        elif what == "cap" and self.cap is not None:
            p, R = self.cap
            if np.linalg.norm(p[:2] - np.asarray(T.CUP_AT)) < T.CUP_R - 0.022:
                self.cap = (np.array([T.CUP_AT[0], T.CUP_AT[1], T.TOP + 0.003 + T.CAP_HALF]), np.eye(3))

    def _sync(self):
        for side, (what, o, oR) in self.held.items():
            arm = self.left if side == "left" else self.right
            p, R = ik.tcp_pose(arm.q, arm.base)
            pose = (p + R @ o, R @ oR)
            if what == "syringe":
                self.syr = pose
            elif self.cap is not None or self._cap_pulled(pose[0]):
                self.cap = pose
        arm = self.left
        self.thumb = 0.0
        if "left" not in self.held and float(arm.cmd[6]) < 0.05:  # the push: the left's closed fingertips on the thumb rest
            p, R = ik.tcp_pose(arm.q, arm.base)
            tip = p + R[:, 2] * 0.0128
            sp, sR = self.syr
            back = sp + sR @ np.array([T.EDGE0 - T.ROD_BACK + self.travel, 0.0, 0.0])
            loc = sR.T @ (tip - back)
            if abs(loc[1]) < 0.012 and abs(loc[2]) < 0.012 and -0.0002 < loc[0] < 0.010:  # on it (0.2 mm) or into it
                self.thumb = 15.0
                # stiction (J_stick) and the stops hold it until the rule has seen the press there
                self.travel = float(max(self.travel, min(self.travel + max(loc[0], 0.0), T.TRIAL.limit())))

    def _cap_pulled(self, pos):
        p, R = self.syr
        return float((R.T @ (pos - p))[0]) - sum(T.CAP) / 2 >= T.CAP_POP

    def _bounds(self, what):
        spec = T.SYRINGE if what == "syringe" else T.CAP_BODY
        p, R = self._pose(what)
        s = T.yb.state_from_pose(spec, p, T.yb.quat_from_matrix(R))
        return s.low, s.high

    def observed(self, side):
        a = self.left if side == "left" else self.right
        return np.r_[a.q, 1.0]

    def state(self, name):
        self._sync()
        p, R = self.syr
        if name == "syringe":
            pos = p
        elif name == "cap":
            pos, R = self._pose("cap")
        elif name == "plunger":
            pos = p + R @ np.array([T.EDGE0 + self.travel, 0.0, 0.0])
        else:
            spec = next(o for o in T.OBJECTS if o.name == name)
            return T.yb.state_from_pose(spec, spec.pos, (0, 0, 0, 1))
        spec = next(o for o in T.OBJECTS if o.name == name)
        return T.yb.state_from_pose(spec, pos, T.yb.quat_from_matrix(R))

    def feed(self, tr):
        """One control step of the rule, as the sensing would read it."""
        self._sync()
        p, R = self.syr
        a = R[:, 0]
        tip = p + T.NEEDLE_L * a
        force = 0.0
        if not tr.engaged and not tr.cap_on and not tr.retracted and tr.geom.on_skin(np.r_[tr.geom.surface_x(tip[1], tip[2]), tip[1:]], 0.001):
            force = float(np.clip(T.K_SENSE * (tip[0] - tr.geom.surface_x(tip[1], tip[2])), 0.0, 2.0))
        cap_slide = 0.0
        if self.cap is not None:
            cap_slide = T.CAP_POP
        tr.feed(T.Sample(0.05, hub=tuple(p), axis=tuple(a), tip=tuple(tip), tip_force=force, travel=self.travel, thumb_n=self.thumb, cap_on=True, cap_slide=cap_slide))

    together = __import__("yam_demokit").Ctx.together
    until = __import__("yam_demokit").Ctx.until


def _statics():
    return T.static_boxes()


def _run(mode="full", sweep_every=0, views=False):
    T.TRIAL.__init__()
    T.TRIAL.geom = T.GEOM
    ctx = _OfflineCtx(mode)
    steps, worst, view_log = 0, {}, []
    statics = _statics()
    trainer = [(lo, hi) for n, lo, hi in statics if n in ("trainer", "stand")]
    for _ in T.demo_script(ctx):
        steps += 1
        ctx.feed(T.TRIAL)
        pts = None
        if T.TRIAL.window and views:
            pts = {a.side: arm_points(a.q, _finger(a, ctx.held.get(a.side, (None,))[0]), a.base) for a in (ctx.left, ctx.right)}
            occ = np.vstack([P for side in pts.values() for n, P in side.items() if n != "base"])
            hidden, v = T.hidden_frame(ctx.syr[0], ctx.syr[1], occ, trainer)
            edge = ctx.syr[0] + ctx.syr[1] @ np.array([T.EDGE0 + ctx.travel, T.BARREL_HALF, 0.0])
            view_log.append((hidden, v, not T._segment_blocked(T.SIDE_CAM, edge, occ, [])))
        else:
            hidden = False
        sy_lo, sy_hi = ctx._bounds("syringe")
        cap_lo, cap_hi = ctx._bounds("cap")
        clear = 1.0
        for arm in (ctx.left, ctx.right):
            P = arm_points(arm.q, _finger(arm, ctx.held.get(arm.side, (None,))[0]), arm.base) if pts is None else pts[arm.side]
            F = np.vstack([P["tip_left"], P["tip_right"]])
            clear = min(clear, _dist(F, sy_lo, sy_hi), _dist(F, cap_lo, cap_hi))
        T.TRIAL.status(T.Status(in_container=T.in_container(sy_lo, sy_hi), cap_in_cup=T.in_cup(cap_lo, cap_hi), clear=clear, hidden=hidden))
        if sweep_every and steps % sweep_every == 0:
            sp, sR = ctx.syr
            bare = ctx.cap is not None and not T.TRIAL.retracted
            needle = sp + np.outer(np.linspace(0.0, T.NEEDLE_L, 12), sR[:, 0])
            P2 = {}
            for arm in (ctx.left, ctx.right):
                holding = ctx.held.get(arm.side, (None,))[0]
                P2[arm.side] = arm_points(arm.q, _finger(arm, holding), arm.base)
                for link, P in P2[arm.side].items():
                    for name, lo, hi in statics:
                        key = (arm.side, link, name)
                        worst[key] = min(worst.get(key, 1.0), _dist(P, lo, hi))
                    key = (arm.side, link, "table")
                    worst[key] = min(worst.get(key, 1.0), float(P[:, 2].min()) - T.TOP)
                    if link not in ("tip_left", "tip_right", "gripper"):
                        key = (arm.side, link, "syringe")
                        worst[key] = min(worst.get(key, 1.0), _dist(P, sy_lo - 0.0, sy_hi))
                    if bare and link in ("tip_left", "tip_right", "gripper"):
                        key = (arm.side, link, "bare needle")
                        worst[key] = min(worst.get(key, 1.0), float(np.min(np.linalg.norm(P[:, None, :] - needle[None], axis=2))))
            if "right" in ctx.held:  # the holding hand against the plunger's rod and thumb rest (spike 4: it touched them)
                loc = {n: (P2["right"][n] - sp) @ sR for n in ("tip_left", "tip_right", "gripper")}
                for i, sh in enumerate(T.PLUNGER.shapes):
                    if sh.collide:
                        c, h = np.asarray(sh.centre) + [T.EDGE0 + ctx.travel, 0.0, 0.0], np.asarray(sh.size) / 2
                        d = min(float(np.linalg.norm(np.maximum(np.abs(L - c) - h, 0.0), axis=1).min()) for L in loc.values())
                        key = "rod" if sh.size[0] > 0.01 else "thumb rest"
                        ctx.plunger_clear[key] = min(ctx.plunger_clear.get(key, 1.0), d)
            for kind, links, lim in (("hands", ("gripper", "link5", "link4"), 0.02), ("fingers", ("tip_left", "tip_right", "gripper", "link5"), 0.005)):
                both = min(float(np.min(np.linalg.norm(P2["left"][a][:, None, :] - P2["right"][b][None, :, :], axis=2))) for a in links for b in links)
                key = ("both", kind, lim)
                worst[key] = min(worst.get(key, 1.0), both)
    return ctx, steps, worst, view_log


@pytest.fixture(scope="module")
def full_run():
    return _run("full", sweep_every=3, views=True)


def test_demo_scores_full_offline(full_run):
    """The demo, stepped offline and fed to the same rule: 1.0 held 1 s, within the run limit."""
    _ctx, steps, _, _ = full_run
    print(steps, T.TRIAL.report())
    assert T.TRIAL.full and T.TRIAL.score == 1.0, T.TRIAL.report()
    assert steps < 3600 / 2, steps


def test_demo_script_reaches_every_waypoint(full_run):
    """Every move ends within 3 mm and 2 degrees of its target, at least 2 degrees inside every joint limit."""
    ctx, *_ = full_run
    bad = [m for m in ctx.misses if m[2] >= 3.0 or m[3] >= 2.0 or m[4] < 2.0]
    assert not bad, bad


def test_demo_arms_clear_every_fixture(full_run):
    """The arms' real meshes: no link within 5 mm of the stand (base, flange, pipe), the trainer, the tray, the rest, the
    sharps container, the cup, the camera mounts or the table; no arm link but the fingers within 5 mm of the syringe; no
    finger within 3 mm of the bare needle; the hands (gripper bodies, wrists) 2 cm apart, their fingers 5 mm."""
    _, steps, worst, _ = full_run
    print(f"{steps} steps; closest: " + ", ".join(f"{k}: {1000 * v:.1f} mm" for k, v in sorted(worst.items(), key=lambda kv: kv[1])[:10]))
    bad = {k: round(1000 * v, 1) for k, v in worst.items()
           if (k[2] == "bare needle" and v < 0.003) or (k[0] == "both" and v < k[2]) or (k[0] != "both" and k[2] != "bare needle" and v < 0.005)}
    assert not bad, bad


def test_right_hand_clears_the_plunger(full_run):
    """Spike 4: the holding hand's fingers met the thumb rest near the end of its travel (1.2 mm clear on the meshes with
    the TCP 44 mm behind the hub). At 42 mm they keep 2.5 mm from the thumb rest; the rod is inside the barrel, behind
    its wall from the pads (1.5 mm or more)."""
    clear = full_run[0].plunger_clear
    assert clear["thumb rest"] >= 0.0025 and clear["rod"] >= 0.0015, clear


def test_scoring_views_in_the_demo(full_run):
    """From S to the retraction: both views show the barrel line and the hub or a band in at least 95% of the frames,
    and the side view sees the stopper's edge in every frame."""
    _, _, _, log = full_run
    assert len(log) > 100
    ok = sum(not h for h, _, _ in log) / len(log)
    edge = sum(e for *_, e in log) / len(log)
    print(f"views ok {ok:.3f}, edge seen {edge:.3f}, side {sum(v['side']['ok'] for _, v, _ in log)}/{len(log)}, overhead {sum(v['over']['ok'] for _, v, _ in log)}/{len(log)}")
    assert ok >= 0.95 and edge == 1.0


@pytest.mark.parametrize("h,t", [(0, 0.043), (1, 0.050), (1, 0.043), (2, 0.050), (2, 0.043)])
def test_demo_reaches_at_every_height_and_thickness(h, t, monkeypatch):
    """The same demo at the other spacer heights and trainer thicknesses (skin x 25.2 and 25.9 cm): every waypoint reached,
    and it scores 1.0 offline."""
    global T, TASK
    monkeypatch.setenv("INJ_HEIGHT", str(h))
    monkeypatch.setenv("INJ_TRAINER_T", str(t))
    saved = (T, TASK)
    try:
        TASK = load(HERE / "im_injection.py")
        T = TASK.module
        ctx, _steps, worst, _ = _run("full", sweep_every=6)
        bad = [m for m in ctx.misses if m[2] >= 3.0 or m[3] >= 2.0 or m[4] < 2.0]
        assert not bad, bad
        near = {k: round(1000 * v, 1) for k, v in worst.items() if (k[0] != "both" and k[2] != "bare needle" and v < 0.005) or (k[0] == "both" and v < k[2])}
        assert not near, near
        assert T.TRIAL.full, T.TRIAL.report()
    finally:
        T, TASK = saved


def test_spike_and_steep_modes_run_offline():
    ctx, steps, _, _ = _run("spike")
    assert T.TRIAL.full and steps < 3600 / 2
    bad = [m for m in ctx.misses if m[2] >= 3.0 or m[3] >= 2.0 or m[4] < 2.0]
    assert not bad, bad
    ctx, steps, _, _ = _run("steep")
    bad = [m for m in ctx.misses if m[2] >= 3.0 or m[3] >= 2.0 or m[4] < 2.0]
    assert not bad, bad
    assert not any(p.engaged for p in T.TRIAL.punctures)


def test_push_torques_on_the_wrists():
    """The push's extra joint torques (J^T F) at the injection pose: 13 N along the barrel at the left's fingertips (2 N
    friction + 1.3 N damping + the 10 N detent) and its reaction at the right's TCP. Printed for the README."""
    import demo as D

    site = T.GEOM.site
    for side, p, rot, F in (("left", site + [T.EDGE0 - T.ROD_BACK + T.ZERO_TRAVEL, 0, -0.004] - D.TIP_AHEAD * D.A_LP, D.G_LPUSH.rot, -13.0 * X),
                            ("right", site + [T.GRASP_R, 0, 0], D.G_R.rot, 13.0 * X)):
        base = station.BASES[side]
        q, pe, _ = ik.solve(p, ik.matrix_to_quat_xyzw(rot), np.array(station_ready(), float), base_pos=base, iters=200)
        J = np.zeros((3, 6))
        for i in range(6):
            dq = np.zeros(6)
            dq[i] = 1e-5
            J[:, i] = (ik.tcp_pose(q + dq, base)[0] - ik.tcp_pose(q - dq, base)[0]) / 2e-5
        tau = J.T @ F
        print(side, np.round(tau, 2).tolist())
        # J1 (base yaw, no gravity load) takes the push's moment about the base: under 5 N m; the rest under 2 N m on top
        # of their gravity load (at most 6.9 N m), so every joint stays under the 9 N m budget
        assert pe < 0.002 and abs(tau[0]) < 5.0 and np.abs(tau[1:]).max() < 2.0


def station_ready():
    from yam_arena.constants import READY

    return READY


# -- fail-safes ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kw", [{"travel": float("nan")}, {"hub": (0.2, float("inf"), 1.0)}, {"thumb_n": float("nan")},
                                {"tip_force": float("-inf")}, {"needle_axis": (1.0, float("nan"), 0.0)}])
def test_non_finite_reading_voids_the_trial(kw):
    r = Run().full()
    r.tr.feed(T.Sample(DT, **kw))
    assert r.tr.voided and r.tr.zero.startswith("void") and r.end() == 0.0


@pytest.mark.parametrize("dt", [0.0, -DT, float("nan")])
def test_time_not_running_forward_voids_the_trial(dt):
    r = Run().uncap()
    t0 = r.tr.t
    r.tr.feed(T.Sample(dt))
    assert r.tr.voided and r.tr.t == t0 and r.end() == 0.0


@pytest.mark.parametrize("q", [(0.0, 0.0, 0.0, 1.01), (0.0, 0.0, 0.0, 0.0), (0.1, 0.2, 0.3, float("nan")), (0.0, 0.0, 1.0)])
def test_non_unit_rotation_voids_the_trial(q):
    r = Run().uncap()
    r.tr.feed(T.Sample(DT, quats=((0.0, 0.0, 0.0, 1.0), q)))
    assert r.tr.voided and "quaternion" in r.tr.voided


def test_unit_rotations_and_axis_pass():
    assert T.sample_fault(T.Sample(DT, quats=((0.0, 0.0, 0.0, 1.0), (0.0, 0.0, math.sin(0.3), math.cos(0.3))))) is None
    assert T.sample_fault(T.Sample(DT, axis=(1.01, 0.0, 0.0))) == "the syringe's axis is not unit length"


def test_void_is_permanent():
    r = Run().full()
    r.tr.void("the needle is missing from the scene")
    r.step(24)
    assert r.end() == 0.0 and not r.tr.full and r.tr.voided == "the needle is missing from the scene"


def test_rotation_normalises():
    R = T.yb.rotation((0.0, 0.0, 2.0 * math.sin(0.25), 2.0 * math.cos(0.25)))
    assert R @ R.T == pytest.approx(np.eye(3))


def test_layout_matches_the_constants():
    assert T.layout_errors() == []


def test_layout_check_catches_a_moved_trainer_and_a_wrong_height():
    assert "the trainer's skin is not at its stated x" in T.layout_errors(geom=T.Geometry(t=T.TRAINER_T - 0.003))
    assert "the acromion's edge is not at its stated height" in T.layout_errors(geom=T.Geometry(h=T.HEIGHTS[(T.HEIGHT + 1) % 3]))


def test_layout_check_catches_the_sharps_opening_and_the_cup():
    import dataclasses

    objs = [o for o in T.OBJECTS]
    sh = next(o for o in objs if o.name == "sharps")
    lid_less = dataclasses.replace(sh, shapes=tuple(s for s in sh.shapes if s.centre[2] < T.SH_H - T.SH_W))
    errs = T.layout_errors([lid_less if o is sh else o for o in objs])
    assert "the sharps container's opening is not its stated size" in errs
    cup = next(o for o in objs if o.name == "cup")
    moved = dataclasses.replace(cup, pos=(cup.pos[0] + 0.01, cup.pos[1], cup.pos[2]))
    assert "the waste cup is not where or the size stated" in T.layout_errors([moved if o is cup else o for o in objs])
    assert T.layout_errors([o for o in objs if o.name != "trainer"]) == ["trainer missing"]


# -- the bare needle's shaft --------------------------------------------------------------------------------------------


def _shaft(tip, axis=X):
    return np.asarray(tip, float) - np.outer(np.linspace(0.0, 1.0, 9) * T.NEEDLE_L, np.asarray(axis, float))


def test_shaft_in_free_air_and_on_the_skin_touches_nothing():
    g = T.GEOM
    assert T.shaft_touch(_shaft(g.site - [0.02, 0.0, 0.0]), False) is None
    assert T.shaft_touch(_shaft(g.site + [0.001, 0.0, 0.0]), False) is None  # the tip pressed on the skin


def test_shaft_in_the_trainer_is_fine_only_while_engaged():
    g = T.GEOM
    tip = g.site + [0.020, 0.0, 0.0]
    assert T.shaft_touch(_shaft(tip), True) is None
    assert T.shaft_touch(_shaft(tip), False) == "the trainer away from its skin"


def test_shaft_on_the_sharps_container_outside_vs_inside():
    a, b = T.SHARPS_INSIDE
    mid = (a + b) / 2
    out_wall = np.array([T.SHARPS_AT[0] - T.SH_X / 2 - 0.0002, mid[1], T.TOP + 0.08])
    assert T.shaft_touch(_shaft(out_wall + [0.0, 0.0, 0.0], axis=(0.0, 0.0, -1.0)), False) == "the sharps container's outside"
    lid_top = np.array([T.SHARPS_AT[0] + T.OPEN_X / 2 + 0.010, mid[1], T.TOP + T.SH_H + 0.0003])
    assert T.shaft_touch(_shaft(lid_top, axis=(0.0, 0.0, -1.0)), False) == "the sharps container's outside"
    in_wall = np.array([a[0] + 0.0002, mid[1], T.TOP + 0.05])
    assert T.shaft_touch(_shaft(in_wall, axis=(0.0, 0.0, -1.0)), False) is None


def test_shaft_on_the_table_and_a_fixture():
    assert T.shaft_touch(_shaft(np.array([0.05, 0.3, T.TOP + 0.0002]), axis=(1.0, 0.0, 0.0)), False) == "the table"
    rest_top = np.array([0.18, -0.09, T.TOP + 0.0121])
    assert T.shaft_touch(_shaft(rest_top + [0.01, 0.0, 0.0]), False) == "the rest"


# -- punctures, set-downs, the push rate --------------------------------------------------------------------------------


def test_sliding_3_mm_while_pressing_is_a_second_puncture():
    """A steep press (never goes in) dragged 3 mm over the skin: a second puncture, as a real needle would make."""
    g = T.GEOM
    r = Run(site=g.site, axis=_axis(yaw=30.0)).uncap()
    r.hub = g.site + [0.0025, 0.0, 0.0] - T.NEEDLE_L * r.a  # the tip 2.5 mm into the skin: 1.25 N
    r.step(5)
    assert len(r.tr.punctures) == 1 and not r.tr.engaged and r.tr.pressing
    r.move([0.0, 0.0, -0.0025], v=0.005)
    assert len(r.tr.punctures) == 1
    r.move([0.0, 0.0, -0.001], v=0.005)
    assert len(r.tr.punctures) == 2


def test_setting_down_the_capped_syringe_is_not_a_zero():
    r = Run(cap_on=True)
    r.tr.status(T.Status(dropped=True))
    assert r.tr.zero is None
    r2 = Run().uncap()
    r2.tr.status(T.Status(dropped=True))
    assert r2.tr.zero is not None


@pytest.mark.parametrize("s_per_02,want", [(0.5, 0.75), (0.9, 0.75), (1.2, 1.0), (3.8, 1.0), (4.5, 0.75)])
def test_push_rate_tiers(s_per_02, want):
    """0.2 mL in under 1 s (too fast) or over 4 s is not a steady push: 0.75."""
    assert Run().full(s_per_02).end() == want


def test_plunger_trace_runs_offline_and_names_every_joint():
    T.SENSE.update(travel=0.0, travel_vel=0.0, thumb_win=1.5)
    line = T.plunger_trace()
    assert line.startswith("[plunger] t=") and all(n in line for n in ("J_stick", "J_zero", "J_plunger", "J_lock", "thumb 1.50 N", "holds True"))
    assert T.SENSE["thumb_win"] == 0.0


def test_bare_needle_back_in_the_cap_after_uncapping_is_a_touch():
    """Review note: the bare needle touching the dropped cap (or recapped) after uncapping is a 0; the cap's inside is
    exempt only while the needle is first leaving it. The cap is filtered from the needle, so it is read from geometry."""
    sh = T.yb.state_from_pose
    hub = np.array(T.SYR_AT)
    seg = hub + np.outer(np.linspace(0.0, 1.0, 9) * T.NEEDLE_L, [1.0, 0.0, 0.0])
    capped = sh(T.CAP_BODY, tuple(T.CAP_AT), (0.0, 0.0, 0.0, 1.0))
    assert T.needle_in_cap(seg, capped)
    away = sh(T.CAP_BODY, tuple(np.asarray(T.CAP_AT) + [0.10, 0.0, 0.0]), (0.0, 0.0, 0.0, 1.0))
    assert not T.needle_in_cap(seg, away)
    lying = sh(T.CAP_BODY, tuple(hub + [T.NEEDLE_L - 0.005, 0.0, -0.002]), (0.0, 0.0, 0.7071068, 0.7071068))  # across the tip
    assert T.needle_in_cap(seg, lying)
    src = Path(T.__file__).read_text()
    assert 'if _LAST.get("cap_cleared"):' in src and '_LAST["cap_cleared"] = True' in src
    r = Run().uncap()
    r.tr.status(T.Status(touch="the cap after uncapping"))
    assert r.tr.zero and "cap" in r.tr.zero


def test_letting_go_with_the_needle_in_the_trainer_is_a_zero():
    r = Run().uncap()
    r.tr.engaged = True
    r.tr.status(T.Status(let_go=True))
    assert r.tr.zero and "let go" in r.tr.zero
    r2 = Run().uncap()
    r2.tr.status(T.Status(let_go=True))  # not in the trainer: only the set-down rule applies
    assert r2.tr.zero is None
    src = Path(T.__file__).read_text()
    assert "let_go=SENSE[\"let_go_n\"] * CONTROL_DT >= LET_GO_S - 1e-9" in src and T.LET_GO_S == 0.5


def _covered_by(body, p, names=None, top=None):
    """Whether world point p lies inside one of `body`'s collider boxes (optionally only boxes whose top is `top`)."""
    for sh in body.shapes:
        if not sh.collide:
            continue
        lo_z = body.pos[2] + sh.centre[2] - sh.size[2] / 2
        if top is not None and abs(lo_z + sh.size[2] - top) > 1e-6:
            continue
        d = (np.asarray(p, float) - np.asarray(body.pos) - np.asarray(sh.centre)) @ T.yb.rotation(sh.quat_xyzw)
        if (np.abs(d) <= np.asarray(sh.size) / 2 + 1e-9).all():
            return True
    return False


def test_cup_floor_fills_the_whole_inside():
    """Review: the floor was 6.3-6.8 cm across inside the 8.6 cm decagon, leaving an 11.6 mm bare ring; a cap tipped into
    it read about 3 mm below the floor. The floor now covers the decagon of the wall's inner faces to its corners."""
    z = T.TOP + 0.0015
    for deg in range(0, 360, 3):
        a = math.radians(deg)
        r_max = T.CUP_INNER / math.cos(math.pi / T.CUP_SIDES - abs(((a % (2 * math.pi / T.CUP_SIDES)) - math.pi / T.CUP_SIDES)))
        for r in (0.0, 0.020, 0.035, 0.042, min(0.0429, r_max - 0.0002)):
            p = (T.CUP_AT[0] + r * math.cos(a), T.CUP_AT[1] + r * math.sin(a), z)
            assert _covered_by(T.CUP, p, top=T.TOP + 0.003), (deg, r)
    # a cap lying tangentially against the wall, on the floor (its corners out to the old bare ring): inside
    a = math.radians(18.0)  # toward a corner of the decagon
    yaw = a + math.pi / 2
    c = T.yb.state_from_pose(T.CAP_BODY, (T.CUP_AT[0] + 0.033 * math.cos(a), T.CUP_AT[1] + 0.033 * math.sin(a), T.TOP + 0.003 + T.CAP_HALF),
                             (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)))
    pts = T.shape_corners(T.CAP_BODY, c)
    assert T.in_cup(None, None, pts)
    for q in pts[pts[:, 2] < T.TOP + 0.004]:  # every lower corner rests on the floor
        assert _covered_by(T.CUP, (q[0], q[1], T.TOP + 0.0015), top=T.TOP + 0.003)


def test_flange_is_19_cm_across_and_the_pipe_wall_6_mm():
    fl = [sh for sh in T.STAND.shapes if sh.collide and abs(sh.size[2] - T.FLANGE_PLATE) < 1e-9]
    lo = np.min([np.asarray(sh.centre) - sh.half_extent() for sh in fl], axis=0)
    hi = np.max([np.asarray(sh.centre) + sh.half_extent() for sh in fl], axis=0)
    assert hi[0] - lo[0] == pytest.approx(T.FLANGE_D, abs=0.002) and hi[1] - lo[1] == pytest.approx(T.FLANGE_D, abs=0.002)
    for deg in range(0, 360, 5):  # a disc: every point within 92 mm of the axis is under it
        p = (T.PIPE_X + 0.092 * math.cos(math.radians(deg)), 0.092 * math.sin(math.radians(deg)), T.TOP + T.BASE_TOP + T.FLANGE_PLATE / 2)
        assert _covered_by(T.STAND, p)
    assert T.PIPE_WALL == pytest.approx(0.00602)
    pipe = [sh for sh in T.STAND.shapes if sh.collide and abs(sh.size[2] - T.PIPE_L) < 1e-9]
    assert len(pipe) == 12 and all(sh.size[0] == pytest.approx(T.PIPE_WALL) for sh in pipe)
    back = min(pipe, key=lambda sh: sh.centre[0])  # the wall facing -x (the trainer's side)
    assert back.centre[0] - back.size[0] / 2 == pytest.approx(T.PIPE_FACE, abs=1e-6)  # its outer face is unchanged
    assert all(math.hypot(sh.centre[0] - T.PIPE_X, sh.centre[1]) + sh.size[0] / 2 == pytest.approx(T.PIPE_R) for sh in pipe)  # OD
    assert "the flange is not its stated diameter" not in T.layout_errors()
    assert T.STAND_BOXES["flange"][1][0] - T.STAND_BOXES["flange"][0][0] == pytest.approx(T.FLANGE_D)


def test_steady_push_text_matches_the_rule():
    """task.json's scoring, the README and the constants state the same steady-push clause; the demo's copy of the
    thumb-on floor is the rule's; the instruction asks for the slow, steady push the rule grades."""
    import demo as D

    task = json.loads((HERE / "task.json").read_text())
    sc, readme = task["scoring"], (HERE / "README.md").read_text()
    assert T.STEP_S == (1.0, 4.0) and T.STOP_S == 3.0 and T.GRAD_ML == 0.1 and T.LINE_TOL == 0.0
    assert T.THUMB_ON_N == D.THUMB_ON_N == 0.2 and T.THUMB_GAP_S == 0.1
    for frag in ("1-4 s per 0.2 mL prorated by its volume", "each later line is reached 1-4 s after the previous one",
                 "reached: on or past it", "no stop (3 s in which the edge reaches no new printed 0.1 mL graduation)",
                 "graduation 0.1 mL below its reading (to the nearest graduation) at the first in-muscle frame", "the hub touches the skin (side view)",
                 "gap of 2 mm or more between the pushing finger and the thumb rest", "3 consecutive", "hidden", "any wait at the 0 line are not timed"):
        assert frag in sc, frag
    for frag in ("under 0.2 N", "for 0.1 s or more", "Timing ends on the frame the edge reaches the 0 line", "3 N / 2 N",
                 "gap 1 mm or less", "on or past it", "every 2 s"):
        assert frag in readme, frag
    import re  # the debugging history (spike runs, failed fixes, rounds) lives outside the task folder
    assert not re.search(r"[Ss]pikes? \d|GPU evidence|round \d|probe \d", readme), re.findall(r".{20}[Ss]pikes? \d.{20}", readme)
    assert "slowly and steadily" in task["instruction"] and "steady push" in task["context"]
    assert "OPEN" not in readme and "slide on by itself" not in readme + task["fidelity"]


def test_tissue_reads_the_tip_velocity_from_its_position():
    """Spikes 10-19: the tissue's forces fed by the solver's velocity of the gripped barrel (+-80-140 mm/s with it still)
    kicked the barrel every step and the stopper's free slide drifted 5-10 mm/s; 19c, with the tip's velocity from its
    position (TipVel, the default), held it (-0.00 mm in 2 s). A still tip reads 0, a steady 10 mm/s reads 10 mm/s."""
    tv = T.TipVel()
    p = np.array([0.25, 0.0, 0.96])
    for _ in range(100):
        v = tv.step(p, DT)
    assert np.allclose(v, 0.0)
    for k in range(1, 240):
        v = tv.step(p + [0.010 * k * DT, 0.0, 0.0], DT)
    assert v[0] == pytest.approx(0.010, rel=1e-3) and abs(v[1]) < 1e-12
    src = Path(T.__file__).read_text()
    assert "tip_vel = _TIP_V.step(tip, dt)" in src and "INJ_TIP_V" not in src
    for mode in ('"2"', 'ZERO_WRENCH == "3"', 'ZERO_WRENCH == "4"'):
        assert mode in src


def test_the_step_hook_is_registered():
    """Spikes 18-19 and full19d scored 0.0 with every 1.0 condition met but `stowed`: a diagnostics helper had been
    inserted between `@yb.on_step` and `_on_step`, so the hook ran the helper and the rule's status (stow, touch, views,
    the score) never ran. The hook is `_on_step`, the only function registered for the control step."""
    assert T._on_step in T.yb._STEP and T._body_props not in T.yb._STEP
    # (the test session imports the task file more than once, each import registering its hooks again)
    assert all(f.__name__ == "_on_step" for f in T.yb._STEP if getattr(f, "__module__", "") == T.__name__)


def test_stow_check_is_one_function_for_rule_and_demo():
    """The rule's stow and the demo's end log both call `stow_check` (the syringe with its rod at its travel, wholly in
    the container; the cap in the cup), so they cannot disagree. A syringe lying in the container with its rod pulled
    back 11 mm stays inside; with the thumb rest over the rim it is not."""
    src_demo = (HERE / "demo.py").read_text()
    src = Path(T.__file__).read_text()
    assert "T.stow_check(s, st(\"plunger\"), st(\"cap\"))" in src_demo and "stow = stow_check(sy, yb.body_state(env, \"plunger\"), cap)" in src
    sh = T.yb.state_from_pose
    lo, hi = T.SHARPS_INSIDE
    q = (0.0, 0.0, 0.0, 1.0)
    hub = np.array([lo[0] + 0.100, (lo[1] + hi[1]) / 2, lo[2] + 0.0125 + 0.0005])  # flange's lower tab on the floor
    sy = sh(T.SYRINGE, hub, q)
    cap = sh(T.CAP_BODY, (T.CUP_AT[0], T.CUP_AT[1], T.TOP + 0.003 + T.CAP_HALF), q)
    for travel, want in ((0.0082, True),):
        pl = sh(T.PLUNGER, hub + [T.EDGE0 + travel, 0.0, 0.0], q)
        out = T.stow_check(sy, pl, cap)
        assert out["in_container"] is want and out["cap_in_cup"], out
    # the barrel inside but the rod's thumb rest past the x- wall: not stowed
    sy3 = sh(T.SYRINGE, np.array([lo[0] + 0.072, hub[1], hub[2]]), q)
    pl3 = sh(T.PLUNGER, np.array([lo[0] + 0.072, hub[1], hub[2]]) + [T.EDGE0, 0.0, 0.0], q)
    out = T.stow_check(sy3, pl3, cap)
    assert out["barrel_excess"]["x-"] <= 2.0 and out["rod_excess"]["x-"] > 2.0 and not out["in_container"], out
