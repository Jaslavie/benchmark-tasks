"""The scripted demo, closed loop on the simulator's poses (which a policy does not get; it gets only the cameras).

Modes (TASK_DEMO): full (1.0); spike (the full demo with the physics-spike checks, printed as [spike] PASS/FAIL lines);
steep (uncap, then press the needle on the trainer 25 deg off square and stop: it must not go in).

Right hand: holds the barrel from the rest to the sharps container in one grip, fingers pointing forward and 30 deg down,
pads on the barrel's +-y faces, its TCP 42 mm behind the hub (the pads 30-52 mm behind it; the flange, long axis
vertical, between the fingers behind them). Left hand: takes the cap with fingers level, pointing -y, pads above and below
it, pulls it straight off along the barrel and drops it in the cup; then pushes the thumb rest's lower half with its
closed fingertips, pointing (0.87, -0.5, 0). Every grasp servos the simulator's own TCP (the gripper body's pose plus the
TCP offset) onto its target and logs it; the syringe's pose is then servoed by trimming the holding hand. Every waypoint's
reach, the arms' clearance from every fixture and the views are checked offline in test_im_injection.py.
"""

from __future__ import annotations

import math
import os

import numpy as np
from yam_arena import ik

import yam_demokit as d

GRIP_N = 40.0  # N a finger on the barrel and on the cap (the YAM driver caps grip at 50 N; spike 1 read 25.5 N at the pads)
# INJ_GRIP_N (diagnostics only): the barrel grip's commanded force (40 N commands about 25 N at each pad, so 14 N about 9 N)
GRIP_BARREL_N = float(os.environ.get("INJ_GRIP_N", str(GRIP_N)))
INSERT_V = 0.010  # m/s: the needle in
PUSH_V = 0.1 * 0.0169  # m/s of stopper travel: 0.1 mL/s (2 s per 0.2 mL), 1.0 mL in 10 s, the taught rate
DETENT_V = 0.002  # m/s through the retraction's detent
DT_CTL = d.DT  # s: one control step
# The left arm is a spring at the thumb rest: 165-350 N/m measured (spike 4), 300-700 N/m in the toy model. Every travel
# cap below is a force over the softest of these, so the arm can always deliver the force the step needs; the force
# itself is limited by the measured thumb load, never by the travel.
ARM_K_MIN = 165.0  # N/m
PRESS_V = 0.004  # m/s the command advances while pressing for the breakaway (0.7-2.8 N/s on a 165-700 N/m arm)
PRESS_OVER = 1.0  # N: the press stops growing this far over BREAKAWAY_F (it must reach BREAKAWAY_F on any arm)
APPROACH_BACK, APPROACH_V, APPROACH_MAX = 0.003, 0.002, 0.004  # m behind the touch pose, m/s, m at most past it
CONTACT_F = 0.3  # N of measured thumb load that marks the touch
LEAD_GAIN = 0.1  # the lead grows by this fraction of the lag each control step
KP_PUSH = 1.0  # the command runs this many times the stopper's lag ahead of the target (proportional; 6 chattered on stiff arms)
PUSH_OVER = 1.0  # N: the lead may hold this much more than the push's nominal force (on the softest arm)
DETENT_OVER = 2.0  # N: the lead stops growing at DETENT_F + 2 N of measured load
THUMB_ON_N = 0.2  # N: the rule's thumb-on floor (im_injection.THUMB_ON_N; checked equal in the tests)
PULL_V, PULL_L = 0.02, 0.045  # the cap off: m/s, m along the barrel
PULL_V0, PULL_SLOW = 0.005, 0.005  # m/s for the first 5 mm (the fit gives at 2.8 mm), then PULL_V
PULL_GAIN, PULL_TRIM = 0.3, 0.0005  # the pull's lateral correction: gain per control step, at most 0.5 mm per step
BARREL_W, CAP_W = 0.0108, 0.008
PRESHAPE_R, PRESHAPE_L = -0.016, -0.015  # finger joints: open 32 and 30 mm across
TIP_AHEAD = 0.011  # m: the closed fingertips' end lies 12.8 mm beyond the TCP: aim 1.8 mm into the thumb rest
TIP_LEN = 0.0128  # m: the closed fingertips' end beyond the TCP
STEEP_DEG = 25.0
X = np.array([1.0, 0.0, 0.0])
Y = np.array([0.0, 1.0, 0.0])
Z = np.array([0.0, 0.0, 1.0])


class PushCtl:
    """The left hand's push on the thumb rest, one control step at a time: returns the hand's command (m along +x from
    the pose where its fingertips just touch the thumb rest with the stopper at `tr0`).

    Approach: from APPROACH_BACK behind the touch pose at APPROACH_V, stopping the moment the measured thumb load passes
    CONTACT_F (an earlier demo met the thumb rest at 12 N within one sample). Press: the command advances at PRESS_V, no
    further once the load reaches BREAKAWAY_F + PRESS_OVER (nor (BREAKAWAY_F + PRESS_OVER) / ARM_K_MIN past the touch),
    until the rule says the stopper has broken loose (`given`: its stuck flag, cleared by the measured thumb load; never
    the stopper's travel, which a stuck stopper also shows as its stop gives under load). Push: the hand keeps pressing
    (the command carries on from where it is: the thumb never leaves the thumb rest) and follows a 0.1 mL/s target from
    the stopper's place at the breakaway: the target, plus a lead that integrates the stopper's lag either way (the arm
    is a spring, so the lead is the force it holds: the glide plus the liquid's resistance at the taught rate), plus
    KP_PUSH times the lag. The liquid's resistance grows with speed, so more force only pushes a little faster: the
    stopper cannot run ahead of the hand. Past the 0 line the lead grows against the retraction's stop until DETENT_F +
    DETENT_OVER of measured load. Toy-model tests in test_im_injection.py (165-3,000 N/m arms)."""

    def __init__(self, tr0: float, ZERO_TRAVEL: float, TRAVEL_MAX: float, DETENT_F: float, BREAKAWAY_F: float, PUSH_F: float = 3.4):
        self.tr0, self.zero, self.end, self.detent_f, self.breakaway_f = tr0, ZERO_TRAVEL, TRAVEL_MAX, DETENT_F, BREAKAWAY_F
        self.lead, self.t, self.base, self.released = 0.0, 0.0, tr0, False
        self.phase, self.cmd = "approach", -APPROACH_BACK
        self.touch = 0.0  # the command at first touch (the fingertips on the thumb rest)
        self.press_max = (BREAKAWAY_F + PRESS_OVER) / ARM_K_MIN  # m past the touch at most while pressing
        self.lead_cap = (PUSH_F + PUSH_OVER) / ARM_K_MIN  # m of lead in the push: the push's force and 1 N on the softest arm
        self.lead_detent = (DETENT_F + DETENT_OVER) / ARM_K_MIN  # m of lead against the 0 line's stop

    def target(self) -> float:
        run = self.base + PUSH_V * self.t
        if run <= self.zero:
            return run
        return min(self.zero + DETENT_V * (self.t - (self.zero - self.base) / PUSH_V), self.end)  # through the detent

    def fingertips(self, travel: float) -> float:
        """The command that just touches the thumb rest with the stopper at `travel`."""
        return self.touch + travel - self.tr0

    def step(self, travel: float, thumb: float, given: bool) -> float:
        self.t += DT_CTL
        if self.phase == "approach":
            if thumb > CONTACT_F:
                self.phase, self.touch = "press", self.cmd
            else:
                self.cmd = min(self.cmd + APPROACH_V * DT_CTL, APPROACH_MAX)
                return self.cmd
        if self.phase == "press":
            if given:
                self.phase, self.released = "push", True
                self.t, self.base = 0.0, travel
                # the hand stays where it is, pressing: what it holds now is the lead the push starts from
                self.lead = float(np.clip(self.cmd - self.fingertips(travel), 0.0, self.lead_cap))
                return self.cmd
            if thumb < self.breakaway_f + PRESS_OVER:  # a steady ramp of the load, capped PRESS_OVER over the breakaway
                self.cmd = min(self.cmd + PRESS_V * DT_CTL, self.touch + self.press_max)
            return self.cmd
        tgt = self.target()
        lag = tgt - travel
        if tgt <= self.zero:
            self.lead = float(np.clip(self.lead + LEAD_GAIN * lag, 0.0, self.lead_cap))
            self.cmd = self.touch + tgt - self.tr0 + self.lead + KP_PUSH * lag
            self.cmd = max(self.cmd, self.fingertips(travel))  # never behind the thumb rest: the fingertips stay on it
        else:  # on the 0 line's stop: press until it gives (no harder than DETENT_F + DETENT_OVER measured)
            if thumb < self.detent_f + DETENT_OVER:
                self.lead = float(np.clip(self.lead + LEAD_GAIN * max(lag, 0.0), 0.0, self.lead_detent))
            self.cmd = min(self.touch + tgt - self.tr0 + self.lead, self.touch + self.end - self.tr0 + self.lead_detent)
        return self.cmd


def _x5_report(T, x5, say):
    """(pushonly) The official steady-push clause read off the finger-alone push: S at the 0.9 mL graduation, each 0.2 mL
    line's time, the longest wait for a new 0.1 mL graduation, and the thumb's contact from S to the 0 line."""
    if not x5:
        say("X5: the stopper never broke loose")
        return
    t, tr, th, lo, off, res = (np.array(c, float) for c in zip(*x5))
    vol = np.array([T.volume(x) for x in tr])

    def first(v):
        i = np.nonzero(vol <= v + 1e-9)[0]
        return None if len(i) == 0 else int(i[0])

    iS, i0 = first(0.9), first(0.0)
    say(f"X5: breakaway at t {t[0]:.2f} s; S (0.9 mL) at {None if iS is None else round(t[iS] - t[0], 2)} s after it; 0 line at "
        f"{None if i0 is None else round(t[i0] - t[0], 2)} s")
    if iS is None or i0 is None:
        say(f"X5 FAIL: the edge reached {vol.min():.3f} mL only (travel {1000 * tr.max():.2f} mm)")
        return
    lines = [(v, first(v)) for v in (0.8, 0.6, 0.4, 0.2, 0.0)]
    times = [round(float(t[i] - t[iS]), 2) for _, i in lines]
    gaps = [times[0]] + [round(b - a, 2) for a, b in zip(times, times[1:])]
    band = [(0.5, 2.0)] + [(1.0, 4.0)] * 4  # the first interval is 0.1 mL: prorated
    rate_ok = all(a - 1e-6 <= g <= b + 1e-6 for g, (a, b) in zip(gaps, band))
    grads = [first(round(0.1 * n, 1)) for n in range(8, -1, -1)]
    waits = np.diff([t[iS]] + [t[i] for i in grads])
    seg = slice(iS, i0 + 1)
    off_s = float(off[i0] if off[iS] <= 1e-9 else off[seg].max())  # the longest run under the floor (physics-step resolution)
    under = th[seg] < T.THUMB_ON_N
    rate = (tr[i0] - tr[iS]) / (t[i0] - t[iS])
    say(f"X5: lines 0.8/0.6/0.4/0.2/0.0 at {times} s after S; intervals {gaps} s (band 0.5-2 then 1-4): rate {'PASS' if rate_ok else 'FAIL'}")
    say(f"X5: longest wait for a new 0.1 mL graduation {waits.max():.2f} s (stop at {T.STOP_S:g} s): {'PASS' if waits.max() < T.STOP_S else 'FAIL'}")
    say(f"X5: thumb load S to the 0 line: mean {th[seg].mean():.2f} N, least in any physics step {lo[seg].min():.2f} N, control steps under "
        f"{T.THUMB_ON_N:g} N {int(under.sum())}/{len(under)}, longest time under it from S {off[seg].max():.3f} s (off at {T.THUMB_GAP_S:g} s): "
        f"{'PASS' if off[seg].max() < T.THUMB_GAP_S else 'FAIL'}")
    say(f"X5: stopper {1000 * rate:.2f} mm/s on average; applied resistance mean {res[seg].mean():.2f} N; speed sd over 0.5 s windows "
        f"{1000 * np.std(np.diff(tr[seg][::10]) / (10 * DT_CTL)):.2f} mm/s")


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def grip(close, approach) -> d.Grip:
    """A TCP rotation: fingers pointing along `approach`, closing along `close`."""
    z = _unit(approach)
    x = _unit(np.asarray(close, float) - z * float(np.dot(close, z)))
    return d._as_grip(np.column_stack([x, np.cross(z, x), z]))


A_R = _unit((0.5, 0.0, -0.866))  # the right hand: fingers forward and 30 deg down
A_LP = _unit((0.866, -0.5, 0.0))  # the left hand's push
G_R = grip((0.0, -1.0, 0.0), A_R)
G_LCAP = grip((0.0, 0.0, 1.0), (0.0, -1.0, 0.0))
G_LPUSH = grip((0.0, 0.0, 1.0), A_LP)
G_LDROP = grip((0.0, 0.0, 1.0), (0.5, -0.5, -0.7))  # over the cup


def places(T) -> dict:
    """Where the hub goes in each phase (the syringe square to the skin, along +x, unless turned)."""
    site = T.GEOM.site
    return {"rest": np.asarray(T.SYR_AT, float), "lift": np.asarray(T.SYR_AT, float) + [0.0, 0.0, 0.05],
            "uncap": np.array([0.140, -0.130, T.TOP + 0.220]),  # clear of the trainer, 5 cm above the sharps container's top
            "front": site - [0.080, 0.0, 0.0], "pre": site - [0.035, 0.0, 0.0], "skin": site + [0.0005, 0.0, 0.0],
            "out": site - [0.050, 0.0, 0.0], "up": np.array([0.220, -0.150, T.TOP + 0.260]),
            "sharps": np.array([0.234 - T.GRASP_R, -0.265, T.TOP + 0.235])}


def demo_script(ctx):
    T = ctx.emb.task.module
    mode = ctx.mode
    pushonly = mode == "pushonly"  # diagnostics (experiment 5): the left finger's push alone, the barrel held on its rest
    if mode not in ("full", "spike", "steep", "pushonly"):
        raise ValueError(f"unknown mode {mode!r}")
    spike = mode == "spike"
    l, r = ctx.left, ctx.right
    st = ctx.state
    P = places(T)
    ready = {a.side: ik.tcp_pose(np.array(d.READY), a.base) for a in (l, r)}
    env = ctx.emb._env
    grasp = np.array([T.GRASP_R, 0.0, 0.0])  # the right TCP's point on the syringe

    def say(msg):
        print(f"[demo] {msg}", flush=True)

    def gate(name, ok, val):
        print(f"[spike] {'PASS' if ok else 'FAIL'} {name}: {val}", flush=True)

    def syr():
        return st("syringe")

    def hub():
        return syr().pos.copy()

    def axis():
        return syr().rotation()[:, 0].copy()

    def where(s, y=0.0, z=0.0):
        sy = syr()
        return sy.pos + sy.rotation() @ np.array([s, y, z])

    def travel():
        sy, pl = syr(), st("plunger")
        return float((sy.rotation().T @ (pl.pos - sy.pos))[0] - T.EDGE0)

    def sense(key, default=0.0):
        return T.SENSE.get(key, default) if env is not None else default

    def pad_middle(arm):
        """The TCP as the simulator has it: the `gripper` body's pose plus the TCP offset (offline, from the joints)."""
        return arm.tcp() if env is None else T.tcp_of(env, arm.side)

    def onto(arm, point, label, body="syringe"):
        """Servo the simulator's TCP onto `point`; log it, the URDF TCP and the object's bounds against the target."""
        point = np.asarray(point, float)
        before = pad_middle(arm) - point
        yield from arm.servo(point, read=lambda: pad_middle(arm), tol=0.001, timeout=2.0, label=label)
        s = st(body)
        say(f"{label}: sim TCP minus target {np.round(1000 * before, 1).tolist()} mm before, {np.round(1000 * (pad_middle(arm) - point), 1).tolist()} mm after; "
            f"URDF TCP minus target {np.round(1000 * (arm.tcp() - point), 1).tolist()} mm; {body} bounds (mm) {np.round(1000 * s.low, 0).tolist()} "
            f"to {np.round(1000 * s.high, 0).tolist()}, target {np.round(1000 * point, 0).tolist()}")

    def grip_log(side, body="syringe"):
        if env is not None:
            say(T.grip_report(env, side, body))

    def poses(label):
        s, c = syr(), st("cap")
        say(f"{label}: hub {np.round(1000 * s.pos, 1).tolist()} mm, axis {np.round(s.rotation()[:, 0], 3).tolist()}, cap {np.round(1000 * c.pos, 1).tolist()}, "
            f"plunger travel {1000 * travel():.2f} mm, report {T.TRIAL.report()}")

    def in_grip():
        """The hub in the right hand's TCP frame (for the slip checks)."""
        p, R = (ik.tcp_pose(r.q, r.base) if env is None else (T.tcp_of(env, "right"), ik.tcp_pose(r.q, r.base)[1]))
        return R.T @ (hub() - p)

    def place_syringe(target, timeout=2.0, tol=0.0006, aim=True, label="hold"):
        """Servo the right hand until the hub is at `target` and (aim) the barrel runs along +x: closes out droop and the
        grasp's offsets. Steps are capped at 2 mm (translation) and 1 deg (turn) per control step."""
        c = ik.tcp_pose(r.q, r.base)[0]
        rot = r.grip.rot.copy()
        good = 0
        for _ in range(round(timeout / d.DT)):
            e = np.asarray(target, float) - hub()
            a = axis()
            tilt = np.cross(a, X) if aim else np.zeros(3)
            if np.linalg.norm(e) < tol and np.degrees(np.linalg.norm(tilt)) < 0.3:
                good += 1
                if good >= 3:
                    break
            else:
                good = 0
            c = c + np.clip(0.3 * e, -0.002, 0.002)
            if aim and np.linalg.norm(tilt) > 1e-6:
                ang = min(float(np.arcsin(min(np.linalg.norm(tilt), 1.0))) * 0.5, math.radians(1.0))
                from yam_arena.urdf import axis_angle_matrix

                Rz = axis_angle_matrix(tilt / np.linalg.norm(tilt), ang)
                rot = Rz @ rot
                c = hub() + Rz @ (c - hub())  # turn about the hub
            r._solve(c, rot)
            yield
        r.grip = d._as_grip(rot)
        r.trim = np.zeros(3)  # the command is the hand's pose now: later moves start from it
        say(f"{label}: hub off its target by {np.round(1000 * (hub() - np.asarray(target, float)), 2).tolist()} mm, axis {np.round(axis(), 4).tolist()}")

    def carry(target_hub, t, label):
        """Move the right hand so the hub goes to `target_hub`, keeping the hand's offset to the hub as it is now."""
        off = ik.tcp_pose(r.q, r.base)[0] - hub()
        yield from r.move(target_hub + off - r.trim, t=t, label=label)

    # ---- 0. (spike) the syringe settles on its rest ----------------------------------------------------------------
    if spike:
        h0 = hub()
        yield from ctx.together(r.wait(2.0), l.wait(2.0))
        gate("syringe settles on the rest (2 s): moves < 0.5 mm", np.linalg.norm(hub() - h0) < 0.0005, f"{1000 * np.linalg.norm(hub() - h0):.2f} mm")
    poses("start")

    if pushonly:
        _thumb = where(T.EDGE0 - T.ROD_BACK, 0.0, -0.004)
        _wait = _thumb - TIP_AHEAD * A_LP - 0.030 * X + 0.03 * Y
        yield from l.move(_wait + [0.0, 0.03, 0.03], grip=G_LPUSH, finger=d.FINGER_CLOSED, t=3.0, label="left: behind the thumb rest, fingers closed")
        yield from l.move(_wait, t=1.0, label="left: level with the thumb rest")
    if not pushonly:
        # ---- 1. right: pick the syringe off the rest, from clear space above and behind -----------------------------------
        def right_pick():
            at = P["rest"] + grasp
            yield from r.move(at - 0.07 * A_R, grip=G_R, finger=PRESHAPE_R, t=3.0, label="right: above and behind the barrel, fingers part-open")
            yield from r.move(at - 0.015 * A_R, t=1.2, label="right: fingers either side of the barrel")
            yield from r.move(at, t=0.8, label="right: pads level with the barrel")
            yield from onto(r, at, "right: barrel pinch")
            yield from r.close(BARREL_W, force=GRIP_BARREL_N, label="right: pinch the barrel")
            grip_log("right")
            if spike:
                worst = 0.0
                for _ in range(round(1.0 / d.DT)):
                    worst = max(worst, abs(travel()))
                    yield from r.wait(d.DT)
                gate("plunger at 1.0 mL 1 s after the right hand's pinch (travel < 0.5 mm; at most 1.69 mm, 0.1 mL, meanwhile)",
                     abs(travel()) < 0.0005 and worst <= 0.1 * T.ML_M, f"{1000 * travel():.2f} mm now, {1000 * worst:.2f} mm at most, thumb {sense('thumb_max'):.2f} N")
            z0 = hub()[2]
            yield from carry(P["lift"], 1.5, "right: lift off the rest")
            rose = hub()[2] - z0
            say(f"lifted: the syringe rose {1000 * rose:.0f} mm")
            if rose < 0.035:
                say(f"SYRINGE NOT HELD (lift): it rose {1000 * rose:.0f} mm")
            grip_log("right")
            yield from carry(P["uncap"], 2.5, "right: to the uncapping place")
            yield from place_syringe(P["uncap"], timeout=1.5, tol=0.001, label="right: at the uncapping place")

        def left_wait():
            yield from l.move(P["uncap"] + [T.GRASP_CAP, 0.10, 0.04], grip=G_LCAP, finger=PRESHAPE_L, t=4.0, label="left: wait beside the uncapping place")

        yield from ctx.together(right_pick(), left_wait())
        poses("syringe held at the uncapping place")

        # ---- 2. (spike) the grip holds 15 N along the barrel ---------------------------------------------------------------
        if spike and env is not None:
            g0 = in_grip()
            worst = 0.0
            for sgn in (1.0, -1.0):
                T.EXTRA_PUSH[:] = sgn * 15.0 * axis()
                for _ in range(round(1.0 / d.DT)):
                    worst = max(worst, abs(travel()))
                    yield from r.wait(d.DT)
            T.EXTRA_PUSH[:] = 0.0
            yield from r.wait(0.5)
            slip = float(np.linalg.norm(in_grip() - g0))
            gate("grip with 15 N along the barrel, both ways: slip < 0.5 mm", slip < 0.0005, f"{1000 * slip:.2f} mm")
            gate("plunger stays at 1.0 mL under the 15 N on the barrel, rod free: travel < 0.5 mm throughout, no retraction",
                 max(worst, abs(travel())) < 0.0005 and not T.TRIAL.retracted,
                 f"{1000 * travel():.2f} mm now, {1000 * worst:.2f} mm at most, retracted {T.TRIAL.retracted}, zero {T.TRIAL.zero}")

        # ---- 3. left: the cap straight off, into the cup ------------------------------------------------------------------
        cap_at = where(T.GRASP_CAP)
        yield from l.move(cap_at + 0.07 * Y, grip=G_LCAP, finger=PRESHAPE_L, t=1.5, label="left: level with the cap, 7 cm to its side")
        yield from l.move(cap_at + 0.002 * Y, t=1.5, label="left: pads above and below the cap")
        yield from onto(l, where(T.GRASP_CAP), "left: cap pinch", body="cap")
        yield from l.close(CAP_W, force=GRIP_N, label="left: pinch the cap")
        yield from l.wait(0.3)  # the pads settle on the cap before the pull (the rule measures the pull from here)
        grip_log("left", "cap")
        g0 = in_grip()
        c0 = ik.tcp_pose(l.q, l.base)[0]
        a0 = axis()  # the barrel's axis as measured now, after both pinches

        def tcp_local():
            """The left TCP (the simulator's) in the syringe's frame."""
            sy = syr()
            return sy.rotation().T @ (pad_middle(l) - sy.pos)

        ref = tcp_local()
        trim = np.zeros(3)
        s_cmd, k = 0.0, 0
        while s_cmd < PULL_L - 1e-9:  # coaxial: along the measured axis, the TCP's drift off that line (in the syringe's
            # frame) trimmed out each step; 5 mm/s through the fit, then 20 mm/s; travel capped at 45 mm
            s_cmd = min(PULL_L, s_cmd + (PULL_V0 if s_cmd < PULL_SLOW else PULL_V) * d.DT)
            off = tcp_local() - ref
            lat = syr().rotation() @ np.array([0.0, off[1], off[2]])
            trim = trim - np.clip(PULL_GAIN * lat, -PULL_TRIM, PULL_TRIM)
            l._solve(c0 + a0 * s_cmd + trim, l.grip.rot)
            k += 1
            if env is not None and k % 5 == 0:
                say(f"cap pull: command {1000 * s_cmd:.1f} mm, TCP off its start in the syringe frame {np.round(1000 * off, 2).tolist()} mm, "
                    f"cap slide {1000 * sense('cap_slide'):.2f} mm, cap_on {T.TRIAL.cap_on}, needle off the barrel's axis "
                    f"{sense('needle_deg'):.2f} deg (max {sense('needle_deg_max'):.2f}), zero {T.TRIAL.zero}")
            yield
        l.trim = np.zeros(3)
        slip_cap = float(np.linalg.norm(in_grip() - g0))
        poses("cap pulled")
        off = st("cap").pos - hub()
        if env is not None and T.TRIAL.cap_on:
            say("CAP NOT OFF")
        if spike:
            gate("cap comes off with a straight pull (6 N at 2.8 mm), needle not bent", not T.TRIAL.cap_on and T.TRIAL.zero is None,
                 f"cap {1000 * float(off @ a0):.1f} mm along the barrel from the hub, zero {T.TRIAL.zero}")
            gate("syringe pinch holds against the cap pull: slip < 0.5 mm", slip_cap < 0.0005, f"{1000 * slip_cap:.2f} mm")
            gate("cap pull with the rod free: plunger travel < 0.5 mm, no retraction", abs(travel()) < 0.0005 and not T.TRIAL.retracted,
                 f"{1000 * travel():.2f} mm, retracted {T.TRIAL.retracted}")

        def left_to_cup():
            yield from l.move(l.at() + [-0.03, 0.05, 0.05], t=1.2, label="left: up and back, away from the needle and the trainer")
            yield from l.move(np.array([T.CUP_AT[0], T.CUP_AT[1], T.TOP + 0.13]), grip=G_LDROP, t=3.0, label="left: over the cup, fingers forward and down")
            yield from l.move(l.at(), finger=PRESHAPE_L, t=0.5, label="left: let the cap fall")
            yield from l.wait(0.6)
            yield from l.move(l.at() + [0.0, 0.0, 0.05], t=0.8, label="left: up")
            c = st("cap")
            cc = T.shape_corners(T.CAP_BODY, c)
            say(f"cap at {np.round(1000 * c.pos, 1).tolist()} mm, axis {np.round(c.rotation()[:, 0], 3).tolist()}, in the cup "
                f"{T.in_cup(None, None, cc)} (worst corner {1000 * T.cup_excess(None, None, cc):+.1f} mm past the inside)")

        if mode == "steep":
            yield from ctx.together(left_to_cup(), carry(P["front"], 3.0, "right: in front of the site"))
            # turn the barrel 25 deg (yaw) and press the tip on the skin 20 mm below the site: it must not go in
            yield from place_syringe(P["front"], timeout=1.5, label="right: in front of the site")
            h0 = hub()
            yield from r.arc(h0, Z, STEEP_DEG, t=1.5, label="right: turn the barrel 25 deg")
            a = axis()
            aim = T.GEOM.site - [0.0, 0.0, 0.020]
            start = aim - a * (T.NEEDLE_L + 0.015)
            yield from carry(start, 2.0, "right: the tip 15 mm off the skin, 25 deg")
            c = ik.tcp_pose(r.q, r.base)[0]
            for k in range(round(3.0 / d.DT)):  # in at 10 mm/s, travel capped at 22 mm
                r._solve(c + a * min(0.022, INSERT_V * d.DT * k), r.grip.rot)
                yield
            gate("a 25 deg entry does not go in", not T.TRIAL.engaged and not any(p.engaged for p in T.TRIAL.punctures),
                 f"tip force {sense('tip_force'):.2f} N, report {T.TRIAL.report()}")
            yield from carry(start, 1.5, "right: back off")
            return

        # ---- 4. right: to the site (the shoulder's edge read from the scene), square; left drops the cap ----------------
        def right_to_site():
            yield from carry(P["front"], 3.0, "right: in front of the site")
            yield from carry(P["pre"], 1.5, "right: the tip 10 mm off the skin at the site")
            yield from place_syringe(P["pre"], timeout=2.5, tol=0.0004, label="right: square at the site")

        yield from ctx.together(left_to_cup(), right_to_site())
        poses("at the site")
        say(f"site from the acromion's edge at z {1000 * (T.GEOM.z_edge - T.TOP):.1f} mm above the table: hub target {np.round(1000 * P['pre'], 1).tolist()}")

        # ---- 5. needle in, square, to the hub; the left hand comes round behind meanwhile -----------------------------------
        def insert():
            c = ik.tcp_pose(r.q, r.base)[0]
            cap_x = float(P["skin"][0] - P["pre"][0]) + 0.001  # never more than 1 mm past the skin
            for k in range(round(6.0 / d.DT)):
                if env is not None and T.TRIAL.gap is not None and T.TRIAL.gap <= 0.0003:
                    break
                if env is None and hub()[0] >= P["skin"][0] - 1e-5:
                    break
                e = P["pre"][1:] - hub()[1:]
                c = c + np.r_[min(INSERT_V * d.DT, max(0.0, P["pre"][0] + cap_x - hub()[0])), np.clip(0.3 * e, -0.0005, 0.0005)]
                r._solve(c, r.grip.rot)
                yield
            yield from r.wait(0.6)
            r.trim = np.zeros(3)
            if env is not None and not T.TRIAL.engaged:
                say("NEEDLE DID NOT GO IN")
            poses("needle in")

        l_wait = None

        def left_behind():
            nonlocal l_wait
            thumb = where(T.EDGE0 - T.ROD_BACK, 0.0, -0.004) + (T.GEOM.site[0] - P["pre"][0]) * X
            l_wait = thumb - TIP_AHEAD * A_LP - 0.030 * X + 0.03 * Y
            yield from l.move(l_wait + [0.0, 0.03, 0.03], grip=G_LPUSH, finger=d.FINGER_CLOSED, t=3.0, label="left: behind the thumb rest, fingers closed")
            yield from l.move(l_wait, t=1.0, label="left: level with the thumb rest")

        yield from ctx.together(insert(), left_behind())
        if spike:
            gap = T.TRIAL.gap if env is not None else 0.0
            gate("needle in at 90 deg, the hub on the skin (gap <= 1 mm)", T.TRIAL.engaged and gap is not None and gap <= 0.001 if env is not None else True, f"gap {gap}")
            gate("no phantom contact before the skin (tip force < 0.2 N off the skin)", sense("phantom") < 0.2, f"{sense('phantom'):.3f} N")

    # ---- 6. left: onto the thumb rest, then the push: 0.1 mL/s, then through the detent (travel capped) -----------------
    thumb = where(T.EDGE0 - T.ROD_BACK + travel(), 0.0, -0.004)
    touch = thumb - TIP_LEN * A_LP  # the TCP with the closed fingertips' end just on the thumb rest's back face
    yield from l.move(touch - 0.012 * X, t=1.5, label="left: 12 mm behind the thumb rest")
    yield from onto(l, touch - 0.012 * X, "left: behind the thumb rest", body="plunger")
    yield from l.move(touch - APPROACH_BACK * X, t=1.0, label="left: 3 mm behind the thumb rest")
    g0, h0 = in_grip(), hub()
    c0, tr0 = touch.copy(), travel()
    # PushCtl approaches at 2 mm/s until the touch (0.3 N), ramps the press until the rule breaks the stopper loose (its
    # stuck flag: the measured thumb load reached BREAKAWAY_F), and keeps pressing: the push follows a 0.1 mL/s target
    # from there against the glide and the liquid's resistance, the fingertips on the thumb rest throughout.
    ctl = PushCtl(tr0, ZERO_TRAVEL=T.ZERO_TRAVEL, TRAVEL_MAX=T.TRAVEL_MAX, DETENT_F=T.DETENT_F, BREAKAWAY_F=T.BREAKAWAY_F, PUSH_F=T.F_GLIDE + T.FLOW_C * PUSH_V)
    touched, press_max, rel_at, surge = False, 0.0, None, 0.0
    push_log = []  # (thumb load N, applied resistance N, travel m) each control step from S to the 0 line
    x5 = []  # (pushonly) (t, travel, thumb, least thumb in the step, longest time off so far, resistance) from the breakaway
    drift = {}

    def drift_hold(label):
        """(spike) The reviewer's drift gate: with the hand backed 5 mm off the thumb rest (nothing pushing), a broken-loose
        stopper moves at most 0.5 mm in 2 s. It runs only on a stopper the rule has released (FAIL otherwise)."""
        name = f"no drift {label}: a free stopper moves at most 0.5 mm in 2 s, nothing pushing"
        if T.TRIAL.stuck:
            gate(name, False, "not measured: the rule never broke the stopper loose (still stuck), so it would measure a held stopper")
            return
        x_rel = travel()
        l._solve(c0 + X * (travel() - tr0 - 0.005), l.grip.rot)
        for j in range(round(0.5 / d.DT)):
            yield
            if j == round(0.25 / d.DT) - 1:
                drift[label + " surge"] = travel() - x_rel
        x0, f_max = travel(), 0.0
        for _ in range(round(2.0 / d.DT)):
            yield
            f_max = max(f_max, sense("thumb"))
        drift[label] = (travel() - x0, f_max)
        gate(name, abs(travel() - x0) <= 0.0005,
             f"{1000 * (travel() - x0):+.2f} mm in 2 s (from {1000 * x0:.2f} mm), thumb load meanwhile at most {f_max:.2f} N; "
             f"the rule's unstick at {T.TRIAL.t_unstick} s")

    for k in range(round(45.0 / d.DT)):
        if (env is not None and T.TRIAL.retracted) or (env is None and travel() >= T.RETRACT_AT + 0.0002):
            break
        given = not T.TRIAL.stuck  # the breakaway is the rule's own (its stuck flag), never the stopper's travel
        tr_now, th_now = travel(), (sense("thumb") if env is not None else ctx.thumb)
        l._solve(c0 + X * ctl.step(tr_now, th_now, given), l.grip.rot)
        if ctl.released and T.TRIAL.t_S is not None and T.TRIAL.t_zero is None:  # S to the 0 line: what the push met
            push_log.append((th_now, abs(T.TRIAL.plunger_n), tr_now))
        if pushonly and ctl.released:
            x5.append((T.TRIAL.t if env is not None else k * d.DT, tr_now, th_now, sense("thumb_lo", th_now), sense("thumb_off_max"), abs(T.TRIAL.plunger_n)))
            if env is not None:
                T.SENSE["thumb_lo"] = 1e9
                if T.volume(tr_now) > 0.9:  # the clause runs from S (the 0.9 line): the time-off record starts there
                    T.SENSE["thumb_off_max"], T.SENSE["thumb_off_run"] = 0.0, 0.0
        if env is not None and os.environ.get("INJ_TRACE"):  # one line per control step of the push, for the logs
            tcp = T.tcp_of(env, "left")
            rest = where(T.EDGE0 - T.ROD_BACK + tr_now, 0.0, -0.004)
            print(f"[push] k={k} t={T.TRIAL.t:.3f} {ctl.phase} cmd {1000 * ctl.cmd:.2f} target {1000 * (ctl.target() - ctl.tr0):.2f} "
                  f"lead {1000 * ctl.lead:.2f} travel {1000 * (tr_now - ctl.tr0):.2f} mm, thumb {th_now:.2f} N, hand {1000 * float((tcp - c0) @ X):.2f} mm, "
                  f"fingertips to thumb rest {1000 * float((rest - tcp - TIP_LEN * A_LP) @ X):.2f} mm, resistance {T.TRIAL.plunger_n:+.2f} N", flush=True)
        if ctl.phase != "approach" and not touched:
            touched = True
            say(f"left on the thumb rest: travel {1000 * travel():.2f} mm, thumb {sense('thumb'):.2f} N")
        if not ctl.released:
            press_max = max(press_max, sense("thumb") if env is not None else ctx.thumb)
        elif rel_at is None:
            rel_at = (k, travel())
            say(f"breakaway: the rule's unstick at {T.TRIAL.t_unstick} s under a thumb load of {T.TRIAL.unstick_n} N "
                f"(BREAKAWAY_F {T.BREAKAWAY_F:g} N); the demo saw it at travel {1000 * travel():.2f} mm, largest thumb load it read before {press_max:.2f} N")
            if spike:  # at once, before S (1 mm past the first in-muscle travel), so the pause is not in the push's timing
                yield from drift_hold("after the breakaway")
                surge = drift.get("after the breakaway surge", 0.0)
                c0, tr0 = c0 + X * (travel() - tr0), travel()  # then the push again from where the stopper is
                ctl = PushCtl(tr0, ZERO_TRAVEL=T.ZERO_TRAVEL, TRAVEL_MAX=T.TRAVEL_MAX, DETENT_F=T.DETENT_F, BREAKAWAY_F=T.BREAKAWAY_F, PUSH_F=T.F_GLIDE + T.FLOW_C * PUSH_V)
                ctl.released = True  # (the stopper is loose: the approach, then straight on to the push)
        elif k - rel_at[0] <= round(0.25 / d.DT) and not spike:
            surge = max(surge, travel() - rel_at[1] - PUSH_V * (k - rel_at[0]) * d.DT)
        yield
    if spike:
        gate("breakaway by the rule: its stuck flag cleared by a measured thumb load of at least BREAKAWAY_F", not T.TRIAL.stuck
             and T.TRIAL.unstick_n is not None and T.TRIAL.unstick_n >= T.BREAKAWAY_F - 1e-6,
             f"unstick at {T.TRIAL.t_unstick} s under {T.TRIAL.unstick_n} N (BREAKAWAY_F {T.BREAKAWAY_F:g} N)")
        gate(f"gentle breakaway: thumb load at most {T.BREAKAWAY_F + 1.5:g} N before it (the press stops growing at "
             f"{T.BREAKAWAY_F + PRESS_OVER:g} N; 0.5 N for one control step's ramp and contact noise); the stopper at most 1 mm "
             "ahead of the 0.1 mL/s target 0.25 s after it (1 mm is S's own travel: a surge under it cannot start the timed push by "
             "itself; spike: the hand backed off at once, so it is the free stopper's own move)",
             env is None or (press_max <= T.BREAKAWAY_F + 1.5 and surge <= 0.001), f"largest thumb load {press_max:.2f} N, surge {1000 * surge:.2f} mm")
    if spike:
        yield from drift_hold("after the retraction")
    else:
        l._solve(c0 + X * (travel() - tr0 + 0.001), l.grip.rot)  # retracted: ease off to 1 mm into the thumb rest
        yield from l.wait(0.5)
    if push_log:
        th, res, trv = (np.array(c) for c in zip(*push_log))
        rate = (trv[-1] - trv[0]) / max(len(trv) - 1, 1) / d.DT
        say(f"push, S to the 0 line: {len(th)} control steps, stopper at {1000 * rate:.2f} mm/s on average; thumb load mean {th.mean():.2f} N "
            f"(least {th.min():.2f}, most {th.max():.2f}); applied glide mean {res.mean():.2f} N; glide + flow expected at this "
            f"rate {T.F_GLIDE + T.FLOW_C * rate:.2f} N")
    if pushonly:
        _x5_report(T, x5, say)
        return
    l.trim = np.zeros(3)
    slip_push = float(np.linalg.norm(in_grip() - g0))
    poses("pushed")
    if spike:
        R = T.TRIAL
        gate("syringe pinch holds against the push: slip < 0.5 mm", slip_push < 0.0005, f"{1000 * slip_push:.2f} mm, hub moved {1000 * np.linalg.norm(hub() - h0):.2f} mm")
        gate("S: the edge on the graduation 0.1 mL below its reading (to the nearest graduation) at the first in-muscle frame", R.t_S is not None,
             f"S {R.t_S}, first in-muscle volume {R.report()['vol_first']}, volume at S {R.vol_S}")
        gate("needle still from S to the retraction (5 deg, 3 mm)", not R.moved, f"largest deviation {R.report()['dev']}")
        gate("continuous in muscle from S to the retraction", R.continuous, f"{R.continuous}")
        rp = R.report()
        gate("steady push: rate (first interval prorated, then 1-4 s per 0.2 mL, to the edge on the 0 line)", R.rate_ok(),
             f"crossings after S {rp['crossings']}, the edge on the 0 line at {rp['zero_line_s']} s")
        gate(f"steady push: no stop ({T.STOP_S:g} s with no new {T.GRAD_ML:g} mL graduation reached) from S to the 0 line",
             R.t_zero is not None and not R.stopped, f"stopped {R.stopped}, 0 line reached {R.t_zero is not None}")
        gate(f"steady push: thumb on from S to the 0 line (contact load under {T.THUMB_ON_N:g} N for {T.THUMB_GAP_S:g} s or more is off)",
             R.t_zero is not None and not R.thumb_lost, f"thumb lost {R.thumb_lost}")
        gate("steady push, all clauses (the rule's 1.0 condition)", R.steady(), f"{R.steady()}")
        gl = R.glide
        gate(f"stopper glide applied while sliding: mean |force| {0.75 * T.F_GLIDE:g}-{1.25 * T.F_GLIDE:g} N (F_GLIDE {T.F_GLIDE:g} N +-25%), "
             "velocity sign flips < 1/s",
             env is None or (gl.sliding_n > 0 and 0.75 * T.F_GLIDE <= gl.mean_sliding() <= 1.25 * T.F_GLIDE and gl.flips < max(gl.t_on, 1.0)),
             f"mean {gl.mean_sliding():.2f} N over {gl.sliding_n} sliding steps, {gl.flips} flips in {gl.t_on:.1f} s")
        if push_log:
            th, res, trv = (np.array(c) for c in zip(*push_log))
            rate = (trv[-1] - trv[0]) / max(len(trv) - 1, 1) / d.DT
            want = T.F_GLIDE + T.FLOW_C * rate
            gate("the push met the modelled resistance: the measured thumb load from S to the 0 line within 25% of glide + FLOW_C x rate",
                 env is None or abs(th.mean() - want) <= 0.25 * want,
                 f"rate {1000 * rate:.2f} mm/s, thumb {th.mean():.2f} N, expected {want:.2f} N (glide applied {res.mean():.2f} N)")
        n = sense("views", 0)
        gate("both views show the barrel line and the hub or a band in >= 95% of frames, S to retraction",
             n > 0 and sense("views_ok", 0) / n >= 0.95, f"{sense('views_ok', 0)}/{n} (side {sense('side_ok', 0)}, overhead {sense('over_ok', 0)})")
        gate("the side view sees the stopper's edge throughout", n > 0 and sense("edge_seen", 0) == n, f"{sense('edge_seen', 0)}/{n}")
        gate("hub y from the band within 1 mm of its true y", sense("band_y_err") <= 0.001, f"{1000 * sense('band_y_err'):.3f} mm")
        edge_s = float((syr().rotation().T @ (st("plunger").pos - syr().pos))[0])
        vol_read = (T.LINE0 - edge_s) / T.ML_M
        gate("stopper-edge reading within 0.05 mL of the rule's tally", abs(vol_read - T.volume(travel())) <= 0.05, f"{vol_read:.3f} vs {T.volume(travel()):.3f} mL")
        car = sense("carrier")
        gate("retraction fired; needle back inside the barrel", R.retracted and car <= -(T.NEEDLE_L - 0.001), f"carrier {1000 * car:.1f} mm")
        gate(f"the 0 line's stop gave only to a thumb load of {T.DETENT_F:g} N or more", sense("thumb_max") >= T.DETENT_F - 0.01, f"largest thumb load {sense('thumb_max'):.1f} N")

    # ---- 7. left back; right withdraws the (retracted) syringe and drops it in the sharps container --------------------
    def left_home():
        yield from l.move(l.at() - 0.03 * X + 0.04 * Y, t=1.0, label="left: back off the thumb rest")
        yield from l.move(ready["left"][0], grip=d._as_grip(ready["left"][1]), finger=d.FINGER_OPEN, t=3.0, label="left: back to ready")

    score_before = []

    def right_to_sharps():
        yield from r.wait(0.4)
        yield from carry(P["out"], 1.5, "right: withdraw along the barrel")
        yield from carry(P["up"], 2.5, "right: up, clear of the stand")
        yield from carry(P["sharps"] + [0.0, 0.0, 0.05], 2.5, "right: over the sharps container's opening")
        yield from carry(P["sharps"], 1.0, "right: down to 8 cm over its floor")
        poses("over the sharps container")
        score_before.append(T.TRIAL.score)  # the rule's score just before the syringe is let go (must be under 1.0)
        yield from r.move(r.at(), finger=PRESHAPE_R, t=0.5, label="right: let the syringe fall")
        yield from r.wait(0.8)
        yield from r.move(r.at() - 0.08 * A_R, t=1.2, label="right: back out along the fingers")
        yield from r.move(ready["right"][0], grip=d._as_grip(ready["right"][1]), finger=d.FINGER_OPEN, t=3.0, label="right: back to ready")

    yield from ctx.together(left_home(), right_to_sharps())
    s = syr()
    poses("end")
    sw = T.stow_check(s, st("plunger"), st("cap"))  # the rule's own stow check
    say(f"stow (the rule's check): syringe with its rod in the container {sw['in_container']} (beyond each face: whole "
        f"{sw['container_excess']}, barrel {sw['barrel_excess']}, rod {sw['rod_excess']} mm), cap in the cup {sw['cap_in_cup']} "
        f"(worst corner {sw['cup_excess_mm']:+.1f} mm past the inside); gripper clearance {1000 * sense('clear', 1.0):.0f} mm (needs 50)")
    yield from ctx.until(lambda: T.TRIAL.full, timeout=4.0, label="1.0 held 1 s")
    say(f"end: {T.TRIAL.report()}")
    R = T.TRIAL
    if not R.stowed:  # each of the rule's stow conditions, on the rule's own points
        cs = T.SENSE.get("clear_sides", {})
        say(f"NOT STOWED: barrel in the container {max(sw['barrel_excess'].values()) <= 2.0} (worst face {sw['barrel_excess']} mm), "
            f"rod in the container {max(sw['rod_excess'].values()) <= 2.0} (worst face {sw['rod_excess']} mm), cap in the cup "
            f"{sw['cap_in_cup']} ({sw['cup_excess_mm']:+.1f} mm), gripper clearance left {1000 * cs.get('left', 1.0):.0f} mm, "
            f"right {1000 * cs.get('right', 1.0):.0f} mm (each needs {1000 * T.GRIP_CLEAR:.0f})")
    ran = env is None or (R.status_n > 0 and T.SENSE.get("layout_checked", False))
    before = score_before[0] if score_before else None
    ok = ran and (env is None or (before is not None and before <= 0.75 + 1e-9 and R.score == 1.0))
    msg = (f"the rule's step checks ran {R.status_n} times (layout check {T.SENSE.get('layout_checked', False)}); its score "
           f"{before} before the syringe was let go over the container, {R.score} at the end (stowed {R.stowed}, held {R.full})")
    if spike:
        gate("the scorer ran: score at most 0.75 before the stow, 1.0 after the stow and the 1 s hold", ok, msg)
    else:
        say(f"scorer check {'PASS' if ok else 'FAIL'} (printed only; never voids): {msg}")
    for side in ("left", "right"):
        tq = T.SENSE.get(f"torque_{side}")
        if tq is not None:
            print(f"[spike] {'PASS' if tq.max() <= 9.0 else 'FAIL'} {side} arm's largest joint torques <= 9 N m: {np.round(tq, 1).tolist()} N m", flush=True)
