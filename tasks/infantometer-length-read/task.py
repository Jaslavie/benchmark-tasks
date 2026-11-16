# SPDX-License-Identifier: MIT
# Copyright (c) 2026 the task authors
"""Measure a newborn training doll's recumbent length on an infant length board (infantometer).

Station table; world frame in metres from the midpoint between the two arm bases, x forward, y to the robot's left,
z up; the table top is at z 0.7475 and the board top at z 0.7655.

- A white length board (680 x 260 x 18 mm) lies on the table from x 0.07 to 0.33 and y -0.24 to 0.44. A fixed white
  headpiece stands at its +y end (face at y 0.425, the zero of the scale), and a yellow tape with black 50 mm ticks
  runs along its far edge (x 0.31) from 0 mm at the headpiece to 600 mm at y -0.175.
- A blue footboard with a beech handle on its back slides along the board on a rail with 1.5 N of friction. It starts
  at the 520 mm mark (face at y -0.095).
- A 45 cm, 0.45 kg newborn training doll lies face up on the board, crown 1 mm from the headpiece, midline on x 0.200.
  Its hips and knees are sprung hinges (newborn flexion tone): at rest the legs are drawn up, hips about 80 deg and
  knees about 95 deg, feet in the air, so the soles reach only the 391 mm mark.

Goal: straighten both sprung legs and hold them flat with the crown on the headpiece while the footboard closes onto
both soles (450 mm), leave the footboard at that reading, then let go.
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import linkkit as lk  # noqa: E402
import numpy as np  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

# -- geometry (world) -------------------------------------------------------------------------------------------
Z_B = TOP + 0.018  # board top
X_MID = 0.200  # doll midline and board centre line
Y_HEADPIECE = 0.425  # headpiece face: the 0 mm mark
L_TRUE = 0.450  # crown to sole with every joint at 0 (the design pose)
TOL = 0.005
HEAD_R = 0.050
Y_CROWN0 = Y_HEADPIECE - 0.001  # crown at reset
LEGS = ("n", "f")  # near leg (x 0.174, toward the robot: the doll's right) and far leg (x 0.226)
LEG_X = {"n": 0.174, "f": 0.226}
HIP_Y = Y_CROWN0 - 0.285  # 0.139
KNEE_Y = Y_CROWN0 - 0.370  # 0.054
SOLE_Y = Y_CROWN0 - L_TRUE  # -0.026
AXIS_Z = Z_B + 0.019  # leg axis height when the legs lie flat
THIGH_LEN, THIGH_R, SHIN_R = HIP_Y - KNEE_Y, 0.019, 0.015
FACE_Y0 = Y_HEADPIECE - L_TRUE  # footboard face at slide q = 0 (reading 450 mm); reading = L_TRUE - q
Q_INIT = -0.070  # footboard starts at the 520 mm mark

WHITE, BLUE, BEECH = (0.95, 0.95, 0.93), (0.15, 0.35, 0.85), (0.80, 0.62, 0.40)
TAPE, INK = (0.95, 0.80, 0.15), (0.05, 0.05, 0.05)
SKIN, NAPPY = (0.93, 0.78, 0.66), (0.97, 0.97, 0.97)
# Contact friction (static, dynamic); PhysX averages the two surfaces in contact. Smooth laminate board; a vinyl doll
# whose lower legs (shins and feet, one link each side) are smooth glossy vinyl: heel on board 0.30 / 0.25, so the
# heels slide when the knees are pressed down.
BOARD_MU, DOLL_MU, LOWER_LEG_MU = (0.35, 0.30), (0.6, 0.5), (0.25, 0.2)

# -- the length board: fixed base (plank, headpiece, tape) and the sliding footboard ------------------------------
_BP = np.array([X_MID, 0.100, TOP])  # board linkage frame
FOOT_O = np.array([X_MID, FACE_Y0, Z_B])  # footboard link frame at q = 0: on the face, at board level


def _b(p):
    return tuple(float(v) for v in np.asarray(p, dtype=float) - _BP)


def _f(p):
    return tuple(float(v) for v in np.asarray(p, dtype=float) - FOOT_O)


_board = lk.Tree("board", pos=tuple(_BP), fixed=True, color=WHITE)
_board.link("base", [
    lk.box((0.26, 0.68, 0.018), _b((X_MID, 0.100, TOP + 0.009))),
    lk.box((0.26, 0.015, 0.16), _b((X_MID, Y_HEADPIECE + 0.0075, Z_B + 0.080))),
    lk.box((0.020, 0.600, 0.0006), _b((0.310, Y_HEADPIECE - 0.300, Z_B + 0.0003)), color=TAPE),
] + [lk.box((0.012, 0.0015, 0.0006), _b((0.310, Y_HEADPIECE - 0.050 * k, Z_B + 0.0009)), color=INK) for k in range(13)],
    mass=2.0, friction=BOARD_MU)
_board.link("foot", [
    lk.box((0.20, 0.012, 0.16), _f((X_MID, FACE_Y0 - 0.006, Z_B + 0.082))),  # plate, 2 mm above the board
    lk.box((0.20, 0.060, 0.008), _f((X_MID, FACE_Y0 - 0.042, Z_B + 0.006))),  # shoe, carried by the rail
    lk.box((0.022, 0.050, 0.022), _f((X_MID, FACE_Y0 - 0.037, Z_B + 0.120)), color=BEECH),  # handle
], mass=0.30, at=_b(FOOT_O), color=BLUE, friction=(0.5, 0.4))
_board.prismatic("slide", "base", "foot", anchor=_b(FOOT_O), axis=(0, 1, 0), limits=(-0.13, 0.15), rest=0.0,
                 init=Q_INIT, stiffness=0.0, damping=2.0, friction=1.5, dynamic_friction=1.2)
BOARD = _board.build()

HEADPIECE_FACE_LOCAL = _b((X_MID, Y_HEADPIECE, Z_B + 0.080))  # on the base link
HANDLE_LOCAL = _f((X_MID, FACE_Y0 - 0.062, Z_B + 0.120))  # handle centre, on the foot link

# -- the doll: rigid head-and-trunk root, two sprung two-joint legs ----------------------------------------------
_DP = np.array([X_MID, 0.279, Z_B])  # doll linkage frame
TRUNK_O = np.array([X_MID, 0.279, Z_B + 0.040])  # trunk link frame = its centre of mass


def _d(p):
    return tuple(float(v) for v in np.asarray(p, dtype=float) - _DP)


def _t(p):
    return tuple(float(v) for v in np.asarray(p, dtype=float) - TRUNK_O)


HEAD_LOCAL = _t((X_MID, Y_CROWN0 - HEAD_R, Z_B + HEAD_R))  # head capsule centre, on the trunk link
_baby = lk.Tree("baby", pos=tuple(_DP), fixed=False, color=SKIN)
_baby.link("trunk", [
    lk.capsule(HEAD_R, 0.010, HEAD_LOCAL, axis="X"),
    lk.box((0.100, 0.150, 0.062), _t((X_MID, Y_CROWN0 - 0.180, Z_B + 0.031))),
    lk.box((0.096, 0.055, 0.050), _t((X_MID, Y_CROWN0 - 0.280, Z_B + 0.025)), color=NAPPY),
    lk.capsule(0.013, 0.090, _t((0.136, 0.249, Z_B + 0.013)), axis="Y"),
    lk.capsule(0.013, 0.090, _t((0.264, 0.249, Z_B + 0.013)), axis="Y"),
], mass=0.33, at=_d(TRUNK_O), com=(0.0, 0.0, 0.0), friction=DOLL_MU)
for _s in LEGS:
    _hip, _knee = (LEG_X[_s], HIP_Y, AXIS_Z), (LEG_X[_s], KNEE_Y, AXIS_Z)
    _baby.link(f"thigh_{_s}", [lk.capsule(THIGH_R, THIGH_LEN, (0.0, -THIGH_LEN / 2, 0.0), axis="Y")], mass=0.035,
               at=_d(_hip), friction=DOLL_MU)
    _baby.link(f"shin_{_s}", [
        lk.capsule(SHIN_R, 0.050, (0.0, -0.030, 0.0), axis="Y"),
        lk.box((0.032, 0.020, 0.070), (0.0, SOLE_Y + 0.010 - KNEE_Y, Z_B + 0.035 - AXIS_Z)),  # foot, toes up
    ], mass=0.025, at=_d(_knee), friction=LOWER_LEG_MU)
    _baby.revolute(f"hip_{_s}", "trunk", f"thigh_{_s}", anchor=_d(_hip), axis=(-1, 0, 0), limits_deg=(-5, 125),
                   rest_deg=95, init_deg=80, stiffness=0.07, damping=0.004, armature=0.0002)
    _baby.revolute(f"knee_{_s}", f"thigh_{_s}", f"shin_{_s}", anchor=_d(_knee), axis=(1, 0, 0),
                   limits_deg=(-3, 150), rest_deg=81, init_deg=95, stiffness=0.05, damping=0.003, armature=0.0002)
BABY = _baby.build()

OBJECTS = [BOARD, BABY]

# The four sole-face corners of each foot in its shin link's frame (origin at the knee).
SOLE_CORNERS = tuple((sx * 0.016, SOLE_Y - KNEE_Y, z - AXIS_Z) for sx in (-1.0, 1.0) for z in (Z_B, Z_B + 0.070))


def _true_length() -> float:
    poses = BABY.link_poses({j.name: 0.0 for j in BABY.joints})
    crown = (poses["trunk"] @ np.r_[HEAD_LOCAL, 1.0])[1] + HEAD_R
    sole = (poses["shin_n"] @ np.r_[SOLE_CORNERS[0], 1.0])[1]
    return float(crown - sole)


assert abs(_true_length() - L_TRUE) < 5e-4, _true_length()

# -- scoring rules ----------------------------------------------------------------------------------------------
HIP_MAX, KNEE_MAX = 12.0, 15.0  # deg: legs straight
GAP_MAX, GAP_MIN = 0.002, -0.003  # m: each sole within 2 mm of the footboard face (not passed by it)
PLATE_X = (X_MID - 0.100, X_MID + 0.100)  # each foot's centre must be in front of the plate
HEAD_MAX = 0.002  # m: crown to headpiece
QD_MAX = 0.005  # m/s: footboard still
TOUCH_S = 0.5  # s the touch must hold to count as a measurement
LIFT_GATE = 0.020  # m: the doll's lowest point above the board
TRUNK_X, TRUNK_Y = (0.08, 0.32), (-0.20, 0.44)  # trunk centre must stay on the board
CLEAR, STILL, RELEASE_S = 0.030, 0.005, 1.0  # let-go gate: pads 30 mm clear of doll and footboard, still 1 s
EPS = 1e-6

H = lk.History()  # whole-run record, cleared only by the simulator reset


def lowest(state: lk.LinkState) -> float:
    """Lowest z (m) of any collider of a linkage state (boxes by corners, capsules by their end spheres)."""
    low = math.inf
    for name, m in state.poses.items():
        for s in state.spec.link(name).shapes:
            if s.kind == "box":
                corners = np.array(np.meshgrid(*[[-1, 1]] * 3)).T.reshape(-1, 3) * s.half_extents()
                pts = (m[:3, :3] @ (np.asarray(s.pos) + corners @ s.rot.T).T).T + m[:3, 3]
                low = min(low, float(pts[:, 2].min()))
            elif s.kind == "capsule":
                half = np.zeros(3)
                half["XYZ".index(s.axis)] = s.dims[1] / 2
                for k in (-1.0, 1.0):
                    p = m[:3, :3] @ (np.asarray(s.pos) + s.rot @ (k * half)) + m[:3, 3]
                    low = min(low, float(p[2]) - s.dims[0])
            else:
                raise ValueError(f"lowest: no rule for {s.kind}")
    return low


def measure(baby: lk.LinkState, board: lk.LinkState, tools: dict | None = None) -> dict:
    """Everything the scorer needs from one control step (m, deg, m/s)."""
    q = float(board.q["slide"])
    face = FACE_Y0 + q
    gaps, feet_x = [], []
    for s in LEGS:
        pts = [baby.point(f"shin_{s}", c) for c in SOLE_CORNERS]
        gaps.append(min(float(p[1]) for p in pts) - face)
        feet_x.append(float(np.mean([p[0] for p in pts])))
    m = {
        "hips": tuple(baby.deg(f"hip_{s}") for s in LEGS),
        "knees": tuple(baby.deg(f"knee_{s}") for s in LEGS),
        "head_gap": Y_HEADPIECE - float(baby.point("trunk", HEAD_LOCAL)[1]) - HEAD_R,
        "gaps": tuple(gaps),
        "feet_x": tuple(feet_x),
        "reading": L_TRUE - q,
        "qd": float(board.qd["slide"]),
        "low": lowest(baby) - Z_B,
        "trunk": (float(baby.pos[0]), float(baby.pos[1])),
        "doll_speed": baby.max_speed(),
    }
    if tools is not None:
        m["clear"] = {k: min(lk.clearance(t, baby), lk.clearance(t, board, ("foot",))) for k, t in tools.items()}
        m["speed"] = {k: float(t.speed) for k, t in tools.items()}
    return m


def record(h: lk.History, m: dict) -> None:
    """Feed one control step into the whole-run record (the kit has already advanced ``h``'s clock)."""
    straight = max(m["hips"]) <= HIP_MAX and max(m["knees"]) <= KNEE_MAX
    head_on = m["head_gap"] <= HEAD_MAX
    soles = all(GAP_MIN <= g <= GAP_MAX for g in m["gaps"]) and all(PLATE_X[0] <= x <= PLATE_X[1] for x in m["feet_x"])
    touch = straight and head_on and soles and abs(m["qd"]) < QD_MAX
    h.ever("lifted", m["low"] > LIFT_GATE)
    tx, ty = m["trunk"]
    h.ever("off_board", not (TRUNK_X[0] <= tx <= TRUNK_X[1] and TRUNK_Y[0] <= ty <= TRUNK_Y[1]))
    h.held("straight", straight and head_on)
    held = h.held("touch", touch)
    h.ever("measured", held >= TOUCH_S - EPS)
    if touch and held >= TOUCH_S - EPS:  # the readings taken: every step of a touch once it has held 0.5 s
        h.span("measured_reading", m["reading"])
    clear, speed = m.get("clear"), m.get("speed")
    h.held("released", bool(clear) and min(clear.values()) >= CLEAR and max(speed.values()) < STILL)
    h.set("reading", m["reading"])
    h.set("last", {"head_gap": m["head_gap"], "gaps": m["gaps"], "hips": m["hips"], "knees": m["knees"],
                   "clear": clear, "speed": speed})
    h.peak("lift_max", m["low"])
    h.peak("head_gap_max", m["head_gap"])
    h.span("reading_span", m["reading"])
    h.peak("trunk_slide", abs(ty - TRUNK_O[1]))
    if touch:
        h.span("touch_reading", m["reading"])
    h.trace("diag", (round(h.t, 2), [round(v, 1) for v in m["hips"] + m["knees"]], round(1000 * m["head_gap"], 1),
                     [round(1000 * g, 1) for g in m["gaps"]], round(1000 * m["reading"], 1), round(1000 * m["low"], 1),
                     round(1000 * m.get("doll_speed", 0.0), 1)), every=0.5)


def ladder(h: lk.History) -> tuple[float, str]:
    """Score and the rung's reason: gates first (any time in the run), then the ladder."""
    if "reading" not in h.values:
        return 0.0, "no steps yet"
    if h.when("lifted") is not None:
        return 0.0, f"doll lifted more than {1000 * LIFT_GATE:.0f} mm off the board (t {h.when('lifted'):.2f} s)"
    if h.when("off_board") is not None:
        return 0.0, f"doll pushed off the board (t {h.when('off_board'):.2f} s)"
    r = h.values["reading"]
    measured = h.when("measured") is not None
    # "leaving the footboard at the reading": within 5 mm of a reading taken during the measured touch
    lo, hi = h.lows.get("measured_reading", math.inf), h.peaks.get("measured_reading", -math.inf)
    on_mark = measured and lo - TOL - EPS <= r <= hi + TOL + EPS
    took = f"read {1000 * lo:.1f} to {1000 * hi:.1f} mm" if measured else ""
    if on_mark and h.runs.get("released", 0.0) >= RELEASE_S - EPS:
        return 1.0, f"measured ({took}), footboard left at {1000 * r:.1f} mm, both grippers released"
    if on_mark:
        return 0.75, f"measured ({took}), footboard at {1000 * r:.1f} mm, not released for 1 s"
    if measured:
        return 0.5, f"measured ({took}), but the footboard ended at {1000 * r:.1f} mm"
    if h.best_runs.get("straight", 0.0) >= 1.0 - EPS:
        return 0.25, "legs held straight with the crown on the headpiece for 1 s, never measured"
    return 0.0, f"no measurement (straight hold {h.best_runs.get('straight', 0.0):.2f} s), footboard at {1000 * r:.1f} mm"


@lk.on_step
def _track(tick):
    st = tick.states
    record(H, measure(st["baby"], st["board"], tick.tools))
    if DIAG:  # development only: what the left pads are nearest to
        tl, baby = tick.tools["left"], st["baby"]
        near = min((lk.clearance(tl, baby, (n,)), n) for n in baby.poses)
        H.trace("left", (round(1000 * near[0], 1), near[1], [round(float(c), 4) for c in tl.tcp],
                         [round(v, 1) for v in (baby.deg("hip_n"), baby.deg("knee_n"))],
                         [round(float(c), 4) for c in baby.point("shin_n", SOLE_CORNERS[0])]), every=0.25)


DIAG = False


def score(objects) -> float:
    return ladder(H)[0]


def success(objects) -> bool:
    return ladder(H)[0] >= 1.0


def distance(objects) -> float:
    return 1.0 - ladder(H)[0]


def demo():  # yam_scripted plays this: both arms hold the ready pose (the closed-loop demo is demo_script)
    return {}


# -- scripted demo (python linkdemo.py task.py OUT [--variant NAME]); reads only camera-observable poses ----------

VARIANTS = ("", "nothing", "footboard_only", "footboard_push", "set_by_tape", "pull_ankles", "one_knee",
            "press_then_fetch", "hold_on")
OPEN_W = 2 * 0.04695
SWEEP_W = 0.046  # left pads 46 mm apart: one finger over each thigh
HANDLE_OPEN, HANDLE_GRIP = 0.050, 0.016  # the handle is 22 mm across
D_START = 0.040  # sweep start: pad tips this far headward of the seen knee centres, at knee height
SWEEP_RATE = 16.0  # deg/s, phase A (swing about the hip axis, fingers against the knees)
A_STOP, A_MAX = 52.0, 60.0  # phase A ends once the seen hips are at 52 deg (or after 60 deg of swing)
R_PRESS = 0.065  # m: phase B presses the thigh tops this far from the hip axis (20 mm short of the knee)
B_RATE, B_LEAD, B_END = 15.0, 8.0, -4.0  # deg/s; the command leads the seen hip angle; ends 4 deg below flat
RELEASE_TO = 40.0  # deg: the left lets the thighs back up to here before opening its fingers
FAST, SLOW = 0.025, 0.005  # m/s footboard closing speeds
ANKLE_LOCAL = (0.0, -0.045, 0.0)  # pull_ankles pinch point on each shin
LOG: dict = {}


def _rot(closing):
    """Top-down TCP rotation with the pads closing along ``closing``."""
    x = np.asarray(closing, dtype=float)
    z = np.array([0.0, 0.0, -1.0])
    return np.column_stack([x, np.cross(z, x), z])


LEFT_ROT, RIGHT_ROT = _rot((-1.0, 0.0, 0.0)), _rot((1.0, 0.0, 0.0))


class _View:
    """What the demo reads from one ctx.observe(): link poses (and angles computed from them) only."""

    def __init__(self, seen):
        self.hip = {s: np.asarray(seen.axis("baby", f"hip_{s}")[0]) for s in LEGS}
        self.knee = {s: np.asarray(seen.axis("baby", f"knee_{s}")[0]) for s in LEGS}
        self.hips = {s: seen.angle_deg("baby", f"hip_{s}") for s in LEGS}
        self.knees = {s: seen.angle_deg("baby", f"knee_{s}") for s in LEGS}
        self.face = float(seen.pose("board", "foot")[1, 3])
        self.gaps = {s: min(float(seen.point("baby", f"shin_{s}", c)[1]) for c in SOLE_CORNERS) - self.face
                     for s in LEGS}
        hp = float(seen.point("board", "base", HEADPIECE_FACE_LOCAL)[1])
        self.head_gap = hp - float(seen.point("baby", "trunk", HEAD_LOCAL)[1]) - HEAD_R
        self.reading = hp - self.face
        self.handle = np.asarray(seen.point("board", "foot", HANDLE_LOCAL))
        self.ankle = {s: np.asarray(seen.point("baby", f"shin_{s}", ANKLE_LOCAL)) for s in LEGS}

    def summary(self):
        return {"hips": [round(self.hips[s], 1) for s in LEGS], "knees": [round(self.knees[s], 1) for s in LEGS],
                "head_gap_mm": round(1000 * self.head_gap, 2), "gaps_mm": [round(1000 * self.gaps[s], 2) for s in LEGS],
                "reading_mm": round(1000 * self.reading, 2)}


def _tip_up():
    from yam_arena.constants import PAD_TIP_PAST_TCP

    return np.array([0.0, 0.0, PAD_TIP_PAST_TCP])


def _wait_for(st, key, seconds=60.0):
    for _ in range(round(seconds * 20)):
        if st.get(key):
            return
        yield




def _thigh_top(hip, phi_deg, r=R_PRESS):
    """World point on the top of a thigh at hip angle ``phi_deg``, ``r`` from the hip axis (the pads' tip target)."""
    f = math.radians(phi_deg)
    return np.array([hip[0], hip[1] - r * math.cos(f) + THIGH_R * math.sin(f),
                     hip[2] + r * math.sin(f) + THIGH_R * math.cos(f)])


def _seen_legs(ctx, legs):
    v = _View(ctx.observe())
    return v, np.mean([v.hip[s] for s in legs], axis=0), float(np.mean([v.hips[s] for s in legs]))


def _left_sweep(ctx, st, legs=LEGS, width=SWEEP_W):
    """Left: lower the fingers headward of the curled knees and swing them about the seen hip axis until the thighs
    are below 52 deg (phase A), then press the thigh tops 50 mm from the hips and follow them down to flat (phase B)."""
    import linkdemo as ld

    left, up = ctx.left, _tip_up()
    v, hip, th0 = _seen_legs(ctx, legs)
    knee = np.mean([v.knee[s] for s in legs], axis=0)
    tip0 = np.array([hip[0], knee[1] + D_START, knee[2]])
    LOG["sweep_start"] = {"hip_deg": round(th0, 1), "tip": [round(float(c), 4) for c in tip0]}
    yield from left.move(tip0 + up + [0.0, 0.0, 0.05], LEFT_ROT, width=width, duration=2.0)
    yield from left.move(tip0 + up, duration=1.0)
    a = 0.0
    while a < A_MAX:  # phase A, closed loop on the seen hip angle
        a += SWEEP_RATE / 20
        left._solve(hip + ld.rot_about((1.0, 0.0, 0.0), a) @ (tip0 - hip) + up, LEFT_ROT, left.width)
        yield
        if _seen_legs(ctx, legs)[2] <= A_STOP:
            break
    v, hip, phi = _seen_legs(ctx, legs)
    LOG["phase_a"] = {"swing_deg": round(a, 1)} | v.summary()
    cmd = phi - B_LEAD
    yield from left.move(_thigh_top(hip, cmd) + up, duration=0.8)
    while cmd > B_END:  # phase B: the tip target rides the thigh tops down to 3 deg below flat
        cmd = max(B_END, cmd - B_RATE / 20)
        left._solve(_thigh_top(hip, cmd) + up, LEFT_ROT, left.width)
        yield
    yield from left.wait(0.3)
    press, extra = left.pos.copy(), 0.0
    for _ in range(4):  # closed loop on the seen angles: 3 mm deeper while not flat (at most 12 mm extra)
        v = _View(ctx.observe())
        if max(v.hips[s] for s in legs) <= 1.5 and max(v.knees[s] for s in legs) <= 3.0 or extra >= 0.012 - 1e-9:
            break
        extra += 0.003
        yield from left.move(press - [0.0, 0.0, extra], duration=0.4)
        yield from left.wait(0.3)
    st["hip"] = hip
    LOG["pressed"] = _View(ctx.observe()).summary() | {"extra_mm": round(1000 * extra, 1)}
    st["pressed"] = True


def _left_release(ctx, st, legs=LEGS):
    """Left: let the thighs follow the fingers back up to 40 deg, open the fingers off them (the legs curl up
    between the open fingers, under the raised hand), then lift clear."""
    left, up = ctx.left, _tip_up()
    hip, cmd = st["hip"], B_END
    yield from left.move(_thigh_top(hip, cmd) + up, duration=0.3)
    while cmd < RELEASE_TO:
        cmd = min(RELEASE_TO, cmd + B_RATE / 20)
        left._solve(_thigh_top(hip, cmd) + up, LEFT_ROT, left.width)
        yield
    yield from left.open(0.4)
    yield from left.wait(0.6)
    yield from left.move(left.pos + [0.0, 0.0, 0.10], duration=1.0)
    yield from left.move(np.array([0.20, 0.17, 0.97]), duration=1.0)


def _left_pull(ctx, st):
    """pull_ankles: pinch both shins just above the feet and drag them 80 mm footward and down onto the board."""
    left = ctx.left
    v = _View(ctx.observe())
    c = np.mean([v.ankle[s] for s in LEGS], axis=0)
    yield from left.move(c + [0.0, 0.0, 0.06], LEFT_ROT, width=OPEN_W, duration=2.0)
    yield from left.move(c, duration=1.2)
    yield from left.grip(0.074, 0.6)
    target = np.array([c[0], c[1] - 0.080, AXIS_Z + 0.004])
    yield from left.move(target, duration=float(np.linalg.norm(target - c)) / 0.020)
    LOG["pulled"] = _View(ctx.observe()).summary()
    st["pressed"] = True


def _right_grasp(ctx):
    right = ctx.right
    g = _View(ctx.observe()).handle + np.array([0.0, -0.010, 0.004])
    yield from right.move(g + [0.0, 0.0, 0.06], RIGHT_ROT, width=HANDLE_OPEN, duration=1.5)
    yield from right.move(g, duration=1.0)
    yield from right.grip(HANDLE_GRIP, 0.6)
    yield from right.wait(0.3)


def _right_close(ctx, st, legs=LEGS, after="pressed", first=False, seat_head=True):
    """Right: close the footboard onto the soles, fast to 6 mm then slowly to contact (closed loop on the seen sole
    gaps, and on the crown reaching the headpiece), 0.5 mm more, hold 1 s."""
    right = ctx.right
    yield from _wait_for(st, after)
    p = right.pos.copy()

    def gap(v):
        return (min if first else max)(v.gaps[s] for s in legs)

    def push(dy):
        p[1] += dy
        right._solve(p, right.rot, right.width)

    for _ in range(20 * 12):
        if gap(_View(ctx.observe())) <= 0.006:
            break
        push(FAST / 20)
        yield
    yield from right.wait(0.3)
    faces = []
    for _ in range(20 * 15):
        v = _View(ctx.observe())
        faces.append(v.face)
        stalled = len(faces) > 6 and faces[-1] - faces[-7] < 0.0002
        if stalled or (gap(v) <= 0.0005 and (not seat_head or v.head_gap <= 0.001)):
            break
        push(SLOW / 20)
        yield
    push(0.0005)
    yield from right.wait(1.0)
    v = _View(ctx.observe())
    LOG["closed"] = v.summary()
    st["face"] = v.face  # the reading at the touch
    p[1] = float(ctx.observe().tools["right"].tcp[1])  # hold the footboard where it is, without preload
    right._solve(p, right.rot, right.width)
    yield


def _right_slide_to(ctx, reading):
    """Set the footboard face at a tape mark by eye (seen footboard pose), two passes."""
    right = ctx.right
    for _ in range(2):
        dy = (Y_HEADPIECE - reading) - _View(ctx.observe()).face
        yield from right.move(right.pos + [0.0, dy, 0.0], duration=max(0.3, abs(dy) / FAST))
        yield from right.wait(0.5)
    LOG["set"] = _View(ctx.observe()).summary()


def _right_release(ctx, st):
    """Right: once the left hand is away, put the footboard back where it was seen at the touch (the re-curling
    feet push it a few mm), then open and lift clear."""
    right = ctx.right
    yield from _wait_for(st, "left_away")
    if "face" in st:
        yield from right.wait(1.0)
        for _ in range(3):
            dy = st["face"] - _View(ctx.observe()).face
            if abs(dy) < 0.0003:
                break
            yield from right.move(right.pos + [0.0, dy, 0.0], duration=max(0.4, abs(dy) / SLOW))
            yield from right.wait(0.4)
        LOG["restored"] = _View(ctx.observe()).summary()
    yield from right.grip(HANDLE_OPEN, 0.5)
    yield from right.move(right.pos + [0.0, 0.0, 0.08], duration=1.0)
    yield from right.move(np.array([X_MID, -0.20, 0.95]), duration=1.0)


def demo_script(ctx):
    name = ctx.variant or ""
    if name not in VARIANTS:
        raise ValueError(f"unknown variant {name!r}; one of {VARIANTS}")
    LOG.clear()
    LOG["variant"] = name
    st: dict = {}
    yield from ctx.wait(1.0)
    LOG["start"] = _View(ctx.observe()).summary()
    if name == "nothing":
        yield from ctx.wait(12.0)
        return
    if name in ("footboard_only", "footboard_push", "set_by_tape"):  # the right arm alone
        st["pressed"] = st["left_away"] = True
        yield from _right_grasp(ctx)
        if name == "footboard_only":
            yield from _right_close(ctx, st, first=True, seat_head=False)
        else:
            yield from _right_slide_to(ctx, 0.380 if name == "footboard_push" else 0.450)
            yield from ctx.right.wait(1.0)
        yield from _right_release(ctx, st)
        yield from ctx.wait(3.0)
        return

    legs = ("f",) if name == "one_knee" else LEGS
    width = 0.0 if name == "one_knee" else SWEEP_W

    def left_gen():
        if name == "pull_ankles":
            yield from _left_pull(ctx, st)
        else:
            yield from _left_sweep(ctx, st, legs, width)
        if name == "press_then_fetch":
            yield from ctx.left.wait(2.0)
            yield from _left_release(ctx, st, legs)
            st["fetch"] = st["left_away"] = True
            return
        yield from _wait_for(st, "closed")
        if name == "hold_on":
            return
        if name == "pull_ankles":
            yield from ctx.left.open(0.6)
            yield from ctx.left.move(ctx.left.pos + [0.0, 0.0, 0.10], duration=1.2)
        else:
            yield from _left_release(ctx, st, legs)
        st["left_away"] = True

    def right_gen():
        yield from _right_grasp(ctx)
        yield from _right_close(ctx, st, legs, after="fetch" if name == "press_then_fetch" else "pressed")
        st["closed"] = True
        if name != "hold_on":
            yield from _right_release(ctx, st)

    yield from ctx.both(left_gen(), right_gen())
    LOG["final"] = _View(ctx.observe()).summary()
    yield from ctx.wait(3.0)


def demo_report() -> dict:
    s, why = ladder(H)
    return {"score": s, "why": why, "measured_at": H.when("measured"), "lifted_at": H.when("lifted"),
            "off_board_at": H.when("off_board"), "straight_best_s": round(H.best_runs.get("straight", 0.0), 2),
            "touch_best_s": round(H.best_runs.get("touch", 0.0), 2), "released_s": round(H.runs.get("released", 0.0), 2),
            "reading_mm": round(1000 * H.values.get("reading", float("nan")), 2),
            "touch_reading_mm": [round(1000 * H.lows.get("touch_reading", float("nan")), 2),
                                 round(1000 * H.peaks.get("touch_reading", float("nan")), 2)],
            "measured_reading_mm": [round(1000 * H.lows.get("measured_reading", float("nan")), 2),
                                    round(1000 * H.peaks.get("measured_reading", float("nan")), 2)],
            "lift_max_mm": round(1000 * H.peaks.get("lift_max", float("nan")), 2),
            "head_gap_max_mm": round(1000 * H.peaks.get("head_gap_max", float("nan")), 2),
            "trunk_slide_mm": round(1000 * H.peaks.get("trunk_slide", float("nan")), 2),
            "last": H.values.get("last"), "log": dict(LOG), "trace": H.traces.get("diag", []),
            "left_trace": H.traces.get("left", [])}
