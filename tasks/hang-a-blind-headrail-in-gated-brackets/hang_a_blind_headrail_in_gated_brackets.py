"""On the station table, at the back: a half-scale window. A light grey reveal wall with a pale blue glass pane in front of it,
and two dark box brackets (a floor, a top lip and an outer cheek each) at the window's left and right ends. Each bracket has
an orange hinged gate across its front, with a white grip tab on its free end; both gates start shut. A white blind headrail,
34 cm long, with dark end caps and two blue grip tabs on top, lies on two dark bearers on the table in front of the window.

Requires -E object_jitter=0: the start pose decides the scored positions.

Run it:  inspect-robots run --embodiment yam_arena -E task=hang_a_blind_headrail_in_gated_brackets.py -E object_jitter=0 \\
           --instruction "Hang the white blind headrail in the two window brackets." --scorer task_score ...
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import closedloop  # noqa: E402, F401
from belimits import tilt_deg  # noqa: E402,F401
from closedloop import grip_rot, placeholder_demo  # noqa: E402
from mechkit import Joint, Mech, box, cyl, on_reset  # noqa: E402

from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 8.0  # the demo grips parts: let it open the grippers and park clear before the episode ends

WHITE, DARK, GREY, GLASS = (0.97, 0.97, 0.95), (0.16, 0.18, 0.24), (0.66, 0.66, 0.68), (0.62, 0.84, 0.96)
ORANGE, BLUE, BEARER = (0.98, 0.45, 0.05), (0.08, 0.25, 0.90), (0.30, 0.30, 0.34)

WALL = 0.36  # the reveal's front face (x); the glass and the brackets end here
FLOOR_Z, LIP_Z = 0.10, 0.145  # bracket floor top and top-lip underside, above the table top
BRACKET_Y = (0.115, 0.185)  # the +y bracket's inner and outer edges; the -y bracket mirrors it
HINGE_X, HINGE_Z = 0.313, 0.117
GATE_X = (0.309, 0.317)  # the shut gate plate
TAB_Y = 0.134  # gate tab centre, |y|, shut


def _e(x0, x1, y0, y1, z0, z1, color):
    return box(((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2), (x1 - x0, y1 - y0, z1 - z0), color)


def _gate(sign):
    """A gate on the bracket at y = sign * 0.15: hinge at its front outer corner, 0 shut, 1.57 rad swung out to -x."""
    s = float(sign)
    shapes = [
        box((0.0, -s * 0.03, 0.0), (0.008, 0.060, 0.028), ORANGE),
        cyl((0.0, -s * (0.185 - TAB_Y), 0.029), 0.009, 0.030, WHITE),  # grip tab 1.8 cm across, 3.0 cm tall
    ]
    low, high = (-1.57, 0.0) if s > 0 else (0.0, 1.57)
    return Joint("gate_l" if s > 0 else "gate_r", "hinge", axis="z", at=(HINGE_X, s * 0.185, HINGE_Z), low=low, high=high,
                 spring=0.0, damping=0.03, mass=0.02, friction=0.4, shapes=shapes)


_SHAPES = [
    _e(WALL, 0.38, -0.23, 0.23, 0.0, 0.16, GREY),  # reveal wall
    _e(0.355, WALL, -0.11, 0.11, 0.0, 0.09, GLASS),  # glass pane
]
for _s in (1, -1):
    y0, y1 = sorted((_s * BRACKET_Y[0], _s * BRACKET_Y[1]))
    cy0, cy1 = sorted((_s * 0.175, _s * 0.185))
    _SHAPES += [
        _e(0.34, WALL, y0, y1, 0.0, 0.09, GREY),  # sill block under the bracket
        _e(0.30, WALL, y0, y1, 0.09, FLOOR_Z, DARK),  # floor
        _e(0.328, WALL, y0, y1, LIP_Z, LIP_Z + 0.010, DARK),  # top lip
        _e(0.322, WALL, cy0, cy1, 0.09, LIP_Z + 0.010, DARK),  # outer cheek
    ]

WINDOW = Mech("window", pos=(0.0, 0.0, TOP), mass=None, friction=0.6, shapes=_SHAPES, joints=[_gate(1), _gate(-1)])

# -- the headrail (free). Origin at the middle of the bar. ------------------------------------------------------------------
RAIL_L, RAIL_W, RAIL_H = 0.34, 0.035, 0.030
TAB_RAIL_Y = 0.08  # grip tabs on top at y +-8 cm
RAIL_SHAPES = [
    box((0, 0, 0), (RAIL_W, RAIL_L - 0.02, RAIL_H), WHITE),
    box((0, -(RAIL_L / 2 - 0.005), 0), (RAIL_W, 0.01, RAIL_H), DARK),
    box((0, RAIL_L / 2 - 0.005, 0), (RAIL_W, 0.01, RAIL_H), DARK),
    box((0, -TAB_RAIL_Y, RAIL_H / 2 + 0.0175), (0.034, 0.018, 0.035), BLUE),
    box((0, TAB_RAIL_Y, RAIL_H / 2 + 0.0175), (0.034, 0.018, 0.035), BLUE),
]
RAIL_START = (0.20, 0.0, TOP + 0.055)
RAIL = Mech("headrail", pos=RAIL_START, mass=0.12, friction=0.5, shapes=RAIL_SHAPES)
BEARERS = [Box(f"bearer_{n}", pos=(0.20, y, TOP + 0.02), size=(0.03, 0.03, 0.04), color=BEARER) for n, y in (("r", -0.10), ("l", 0.10))]

OBJECTS = [WINDOW, RAIL, *BEARERS]

# -- scoring ------------------------------------------------------------------------------------------------------------
HOME_TOL = 0.005  # back face within 5 mm of the wall at both ends
SEAT_TOL = 0.004  # rail underside within 4 mm of the floor tops
MAX_TILT = 2.0  # degrees
END_IN = 0.145  # rail ends at least 3 cm past the floors' inner edges
SHUT_TOL = 0.1  # rad
OPEN_MIN = 1.2  # rad


class _Latch:
    def __init__(self):
        self.clear()

    def clear(self):
        self.opened = False
        self.prev = None


LATCH = _Latch()


@on_reset
def _reset_latch(env=None):
    LATCH.clear()


def _rot(q):
    x, y, z, w = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _world(r, local):
    return np.asarray(r.origin, float) + (_rot(r.quat_xyzw) @ np.asarray(local, float).T).T


def back_x(r):
    """World x of the rail's back face at the two ends (just inside the end caps)."""
    return _world(r, [(RAIL_W / 2, -0.16, 0.0), (RAIL_W / 2, 0.16, 0.0)])[:, 0]


def at_wall(r) -> bool:
    b = back_x(r)
    return bool(np.all(b >= WALL - HOME_TOL) and np.all(b <= WALL + 0.004))


def on_floors(r) -> bool:
    """Rail underside at both ends on the bracket floors (within 4 mm), ends 3 cm or more past the floors' inner edges, not
    tilted. A rail more than 1 cm above the floors (still held, perched on a lip, lying on the bearers) fails."""
    low = _world(r, [(sx * RAIL_W / 2, sy * 0.16, -RAIL_H / 2) for sx in (-1, 1) for sy in (-1, 1)])
    z = low[:, 2] - TOP
    ends = _world(r, [(0.0, -0.165, 0.0), (0.0, 0.165, 0.0)])[:, 1]
    return bool(np.all(np.abs(z - FLOOR_Z) <= SEAT_TOL) and tilt_deg(r) <= MAX_TILT
                and ends[0] <= -END_IN and ends[1] >= END_IN and ends[0] >= -0.170 and ends[1] <= 0.170)


def home(r) -> bool:
    return at_wall(r) and on_floors(r)


def _angles(w):
    return abs(w.joints.get("gate_l", 0.0)), abs(w.joints.get("gate_r", 0.0))


def gates_shut(w) -> bool:
    a, b = _angles(w)
    return bool(a <= SHUT_TOL and b <= SHUT_TOL)


def _update(objects):
    r, w = objects["headrail"], objects["window"]
    d = float(np.linalg.norm(np.asarray(r.origin, float) - np.asarray(RAIL_START, float)))
    if LATCH.prev is not None and LATCH.prev > 0.03 and d <= 0.005 and np.linalg.norm(r.lin_vel) < 0.01:
        LATCH.clear()  # the rail teleported back to its start: a trial reset
    LATCH.prev = d
    a, b = _angles(w)
    if a >= OPEN_MIN and b >= OPEN_MIN:
        LATCH.opened = True
    return r, w


def score(objects):
    """End state with one flag kept over the run (both gates swung open 1.2 rad or more at once).
    1.0: rail home (underside on both bracket floors within 4 mm, back face within 5 mm of the wall at both ends, tilted at
    most 2 degrees, ends 3 cm or more inside the brackets) and both gates within 0.1 rad of shut. 0.5: rail home, a gate still
    open. 0.25: both gates were opened 1.2 rad or more at once. Else 0. A rail whose underside is more than 1 cm above the
    floors (still held, perched) is not home."""
    r, w = _update(objects)
    if home(r):
        return 1.0 if gates_shut(w) else 0.5
    return 0.25 if LATCH.opened else 0.0


def success(objects):
    r = objects["headrail"]
    return score(objects) == 1.0 and r.resting()


# The glass: no part of either arm below 9 cm above the table in front of the pane (plus 4 cm margin).
ARM_KEEP_OUT = [
    {"name": "window glass", "x": (0.26, 0.40), "y": (-0.15, 0.15), "z_top": TOP + 0.09},
]

# -- the scripted demo (both arms, closed loop) -----------------------------------------------------------------------------
demo = placeholder_demo


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


LIFT = 0.05  # clearance above a tab when approaching or leaving
PUSH_DZ = 0.0  # tool point at the tab's centre height when pushing (closed tips reach 1.3 cm lower)
GRIP_DZ = 0.003  # tool point above the tab centre
HALF = FINGER_OPEN * 0.5  # approach and release opening (about 2.3 cm each side)
PARTIAL = FINGER_OPEN * 0.35  # release opening beside the brackets (about 1.6 cm each side)
TAB_Z = HINGE_Z + 0.029  # gate tab centre above the table top
CARRY_Z = 0.122  # rail centre while carried (underside 7 mm over the floors)
SEAT_Z = 0.1157  # rail centre on the floors
RAIL_TAB_DZ = RAIL_H / 2 + 0.0175  # rail tab centre above the rail centre
X_DROP = 0.308  # rail centre where it is set down on the floors
X_HOME = WALL - RAIL_W / 2  # rail centre against the wall (0.3425)


def _tab_xy(sign, phi):
    """Gate tab centre (x, y) when the gate on side ``sign`` is open by ``phi`` rad (0 shut)."""
    rel = np.array([0.0, -sign * (0.185 - TAB_Y)])
    c = Rz(-sign * phi)[:2, :2] @ rel
    return np.array([HINGE_X, sign * 0.185]) + c


def _tab_push(sign, phi):
    """Extra travel along the closing tangent for a command past shut (phi < 0): the gate stops at its end stop."""
    t = np.array([1.0, 0.0])  # a shut gate's tab moves toward +x as it closes
    return np.array([t[0] * -phi * 0.051, 0.0, 0.0])


def demo_script(ctx):
    """Both arms swing the two gates open, lift the rail by its blue tabs, carry it level to the floors, set it down, slide it
    back against the wall, and let go; then each arm swings its gate shut by the white tab, and both park."""
    O, C = FINGER_OPEN, FINGER_CLOSED
    A = {1: ctx.left, -1: ctx.right}
    rot = grip_rot(yaw=0.0)
    live = ctx.states() is not ctx.states()  # an offline replay hands back the same start poses every time

    def tcp(sign, phi, dz=GRIP_DZ):
        xy = _tab_xy(sign, phi)
        return np.array([xy[0], xy[1], TOP + TAB_Z + dz])

    def grab_gate(sign, phi):
        arm = A[sign]
        p = tcp(sign, phi)
        yield from arm.move(p + np.array([0.0, 0.0, LIFT]), 3.0, HALF, rot=rot)
        yield from arm.move(p, 2.0, HALF, rot=rot)
        yield from arm.trim_to_measured(p, tol=0.0015, max_comp=0.03, step_mps=0.03, max_duration=0.8)
        yield from arm.grip(C, 0.8)
        yield from arm.hold(0.3)

    def swing(sign, a, b, dur):
        arm = A[sign]
        n = max(1, round(dur / ctx.dt))
        for i in range(1, n + 1):
            phi = a + closedloop.ik.smoothstep(i / n) * (b - a)
            arm._set_cartesian(tcp(sign, phi), rot, C)
            yield

    def let_go(sign):
        arm = A[sign]
        yield from arm.grip(PARTIAL, 0.6)
        yield from arm.move(arm.pos + np.array([0.0, 0.0, LIFT]), 1.5, PARTIAL, rot=rot)
        yield from arm.grip(O, 0.4)

    def gate_cycle(sign, a, b, dur):
        yield from grab_gate(sign, a)
        yield from swing(sign, a, b, dur)
        yield from A[sign].hold(0.4)
        yield from let_go(sign)

    PHI_OPEN = 1.52

    # 1. both gates open
    yield from ctx.both(gate_cycle(1, 0.0, PHI_OPEN, 4.0), gate_cycle(-1, 0.0, PHI_OPEN, 4.0))

    # 2. lift the rail by its two tabs
    r0 = ctx.obj("headrail")
    rx, ry = float(r0.origin[0]), float(r0.origin[1])
    zs = RAIL_START[2] - TOP
    corr = {"z": 0.0}

    def rail_tcp(sign, x, zc, y=None, dz=GRIP_DZ, roll_corr=0.0):
        yy = (ry if y is None else y) + sign * TAB_RAIL_Y
        return np.array([x, yy, TOP + zc + RAIL_TAB_DZ + dz + sign * roll_corr])


    def to_rail(sign):
        arm = A[sign]
        p = rail_tcp(sign, rx, zs)
        yield from arm.move(p + np.array([0.0, 0.0, LIFT]), 3.0, HALF, rot=rot)
        yield from arm.move(p, 2.0, HALF, rot=rot)
        yield from arm.trim_to_measured(p, tol=0.0015, max_comp=0.03, step_mps=0.03, max_duration=0.8)
        yield from arm.grip(C, 0.8)
        yield from arm.hold(0.3)

    yield from ctx.both(to_rail(1), to_rail(-1))

    plan = {"x": rx, "z": zs}
    roll = {"c": 0.0}

    def go(x, z, dur, y=None):
        """Both hands move the rail's planned centre to (x, z) together; level is kept from the rail's live tilt."""
        a = dict(plan)
        n = max(1, round(dur / ctx.dt))
        for i in range(1, n + 1):
            s = closedloop.ik.smoothstep(i / n)
            px, pz = a["x"] + s * (x - a["x"]), a["z"] + s * (z - a["z"])
            if live:
                st = ctx.obj("headrail")
                r = _rot(st.quat_xyzw)
                tilt = float(np.arcsin(np.clip(r[2, 1], -1.0, 1.0)))  # +y end up is positive
                roll["c"] = float(np.clip(roll["c"] - 0.2 * tilt * 0.08, -0.01, 0.01))
            for sign in (1, -1):
                A[sign]._set_cartesian(rail_tcp(sign, px, pz, y, roll_corr=roll["c"]), rot, C)
            yield
        plan.update(x=x, z=z)

    yield from go(rx, CARRY_Z, 1.6)
    yield from go(X_DROP, CARRY_Z, 4.0)
    yield from go(X_DROP, CARRY_Z, 0.6)
    yield from go(X_DROP, SEAT_Z, 2.0)
    yield from ctx.both(A[1].hold(0.5), A[-1].hold(0.5))

    # 3. slide it back against the wall at 1 cm/s until the live back face is within 2 mm
    x = X_DROP
    while x < X_HOME + 0.006 - 0.0005:
        if live:
            b = back_x(ctx.obj("headrail"))
            if float(b.min()) >= WALL - 0.002:
                break
        x = min(x + 0.005, X_HOME + 0.006)
        yield from go(x, SEAT_Z, 0.5)
    yield from ctx.both(A[1].hold(0.8), A[-1].hold(0.8))

    # 4. let go, rise
    def release(sign):
        arm = A[sign]
        yield from arm.grip(PARTIAL, 0.6)
        yield from arm.move(arm.pos + np.array([-0.05, 0.0, 0.05]), 1.5, PARTIAL, rot=rot)
        yield from arm.grip(O, 0.4)

    yield from ctx.both(release(1), release(-1))

    # 5. push each gate shut: closed fingers behind the tab, driving it along its arc
    PUSH_D = 0.020  # tool point this far behind the tab centre

    def pusher(sign, phi):
        p = _tab_xy(sign, phi)
        q = _tab_xy(sign, phi + 0.02)  # where the tab was a moment earlier
        t = (p - q) / np.linalg.norm(p - q)
        xy = p - t * PUSH_D
        return np.array([xy[0], xy[1], TOP + TAB_Z + PUSH_DZ])

    def close_cycle(sign):
        arm = A[sign]
        p = pusher(sign, PHI_OPEN)
        yield from arm.move(p + np.array([0.0, 0.0, LIFT]), 2.5, C, rot=rot)
        yield from arm.move(p, 1.5, C, rot=rot)
        yield from arm.trim_to_measured(p, tol=0.0015, max_comp=0.03, step_mps=0.03, max_duration=0.8)
        n = round(4.0 / ctx.dt)
        for i in range(1, n + 1):
            phi = PHI_OPEN + closedloop.ik.smoothstep(i / n) * (-0.03 - PHI_OPEN)
            arm._set_cartesian(pusher(sign, max(phi, 0.0)) + (np.array([0.0, 0.0, 0.0]) if phi >= 0.0 else _tab_push(sign, phi)), rot, C)
            yield
        yield from arm.hold(0.6)
        yield from arm.move(arm.pos + np.array([0.0, 0.0, LIFT]), 1.5, C, rot=rot)
        yield from arm.home(3.0)

    yield from ctx.both(close_cycle(1), close_cycle(-1))
    yield from ctx.wait(1.0)
