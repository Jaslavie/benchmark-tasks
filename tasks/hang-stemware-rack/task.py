"""On the station table, to the right of the midline: a single-channel overhead bar glass rack (brass-coloured, the
sourced profile: two lips with a 25 mm channel between them, 89 mm opening, 38 mm tall) runs along y at x 0.44 m,
its lips' top 220 mm above the table, from its open end at y -0.20 m to a closed end at y +0.10 m (300 mm, four
glasses). Two square brass posts, 20 mm across and 254 mm tall, at its far side (x 0.50) hold it up.

Two washed wine glasses stand bowl-down on their rims on a black rubber bar mat (200 x 120 x 6 mm), as they come
out of the glass washer:
glass A at (0.44, -0.30), glass B behind it at (0.53, -0.30). Each glass (one rigid body, 180 g): a 75 mm foot,
3 mm thick; a 9 mm stem, 80 mm long; a 95 mm bowl (74 mm rim, 84 mm at its widest); 178 mm tall.

The goal: both glasses hang upside down in the rack: each foot sits on both lips inside the channel (foot underside
within 3 mm of the lips' top, the stem within 8 mm of the channel's centre line, the foot wholly between the open
end and the closed end), each glass hangs straight (tilt <= 5 deg), nothing moves, both TCPs are at least 3 cm from
each glass (released), and no glass was knocked over (tilted past 45 deg) at any time during the trial.
"""

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import robocurve_ext as rx  # noqa: E402

import yam_arena.demo as _demo  # noqa: E402
from yam_arena.constants import FINGER_CLOSED  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402
from yam_arena.urdf import axis_angle_matrix, quat_xyzw_to_matrix  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SCORE_OVER = "end"

BRASS = (0.72, 0.60, 0.36)
GLASS_COLOR = (0.80, 0.86, 0.88)
MAT_COLOR = (0.12, 0.12, 0.13)

# -- the rack (fixed): channel along y; heights above TOP ---------------------------------------------------------------
XS = 0.44  # channel centre line
YE, YC = -0.20, 0.10  # open end, closed end
H = 0.220  # lips' top
SLOT, OPEN, PROF, LIP, WALL = 0.025, 0.089, 0.038, 0.004, 0.003  # sourced profile (README); lip and wall gauge assumed
RL = YC - YE
RACK_W = OPEN + 2 * WALL
RACK_C = (XS, (YE + YC) / 2, H - LIP + PROF / 2)  # profile's bounding-box centre
POST = 0.020  # square posts at the rack's far side
POST_X = XS + RACK_W / 2 + POST / 2
POST_Y = (-0.17, 0.07)
RACK_TOP = H - LIP + PROF


def _piece(centre, x0, x1, y0, y1, z0, z1) -> rx.Piece:
    """An axis-aligned piece from world x/y bounds and z bounds above TOP, in a fixed part's frame at ``centre``."""
    return rx.Piece(size=(x1 - x0, y1 - y0, z1 - z0),
                    offset=((x0 + x1) / 2 - centre[0], (y0 + y1) / 2 - centre[1], (z0 + z1) / 2 - centre[2]))


def _rack_compound() -> rx.Compound:
    p = lambda *b: _piece(RACK_C, *b)  # noqa: E731
    z0 = H - LIP
    pieces = [
        p(XS + SLOT / 2, XS + OPEN / 2, YE, YC, z0, H),  # lips
        p(XS - OPEN / 2, XS - SLOT / 2, YE, YC, z0, H),
        p(XS + OPEN / 2, XS + RACK_W / 2, YE, YC, z0, RACK_TOP),  # side walls
        p(XS - RACK_W / 2, XS - OPEN / 2, YE, YC, z0, RACK_TOP),
        p(XS - OPEN / 2, XS + OPEN / 2, YE, YC, RACK_TOP - WALL, RACK_TOP),  # top
        p(XS - OPEN / 2, XS + OPEN / 2, YC - WALL, YC, H, RACK_TOP - WALL),  # closed end over the lips
    ]
    return rx.Compound(pieces=tuple(pieces))


def _posts_compound(centre) -> rx.Compound:
    pieces = []
    for y in POST_Y:
        pieces.append(_piece(centre, POST_X - POST / 2, POST_X + POST / 2, y - POST / 2, y + POST / 2, 0.0, RACK_TOP))
    return rx.Compound(pieces=tuple(pieces))


RACK = Box("rack", pos=(RACK_C[0], RACK_C[1], TOP + RACK_C[2]), size=(RACK_W, RL, PROF), color=BRASS)
POSTS_C = (POST_X, sum(POST_Y) / 2, RACK_TOP / 2)
POSTS = Box("posts", pos=(POSTS_C[0], POSTS_C[1], TOP + POSTS_C[2]),
            size=(POST, POST_Y[1] - POST_Y[0] + POST, RACK_TOP), color=BRASS)

# -- the bar mat (fixed) and the glasses (loose) ----------------------------------------------------------------------
MAT_T = 0.006
MAT = Box("mat", pos=(0.485, -0.30, TOP + MAT_T / 2), size=(0.20, 0.12, MAT_T), color=MAT_COLOR)
GL = 0.178  # glass length; glass.obj's origin is its bounding-box centre, foot at +z
GLASS_SIZE = (0.084, 0.084, GL)
FOOT_Z = 0.086  # foot underside in the body frame
FOOT_R = 0.0375
GLASS_MASS = 0.18
SPAWN = {"glass_a": (0.44, -0.30), "glass_b": (0.53, -0.30)}
GLASS_NAMES = tuple(SPAWN)


def _glass_box(name) -> Box:
    x, y = SPAWN[name]
    return Box(name, pos=(float(x), float(y), float(TOP + MAT_T + GL / 2 + 0.0005)), size=GLASS_SIZE,
               mass=GLASS_MASS, color=GLASS_COLOR)


GLASSES = [_glass_box(n) for n in GLASS_NAMES]
rx.install({"rack": _rack_compound(), "posts": _posts_compound(POSTS_C),
            **{n: rx.Mesh(os.path.join(HERE, "glass.obj"), approximation="convexHull") for n in GLASS_NAMES}})

OBJECTS = [RACK, POSTS, MAT, *GLASSES]

# -- measures ---------------------------------------------------------------------------------------------------------
FOOT_DZ_MM = 3.0
CENTRE_MM = 8.0  # the 9 mm stem in the 25 mm channel can be 8 mm off its centre line
TILT_DEG = 5.0
KNOCK_TILT_DEG = 45.0
GRIPPER_CLEAR = 0.030


def _rot(state):
    return quat_xyzw_to_matrix(np.asarray(state.quat_xyzw, float))


def tilt_deg(state) -> float:
    """Angle between the glass's axis and the vertical (0: foot straight up, bowl down, as it stands and hangs)."""
    return math.degrees(math.acos(max(-1.0, min(1.0, float(_rot(state)[2, 2])))))


def foot_centre(state) -> np.ndarray:
    """World position of the centre of the foot's underside."""
    return np.asarray(state.pos, float) + _rot(state) @ np.array([0.0, 0.0, FOOT_Z])


def hung(state) -> bool:
    """Foot on both lips inside the channel, the glass hanging straight."""
    f = foot_centre(state)
    return bool(abs((f[2] - TOP - H) * 1000.0) <= FOOT_DZ_MM and abs(f[0] - XS) * 1000.0 <= CENTRE_MM
                and YE + FOOT_R <= f[1] <= YC + 0.001 - FOOT_R and tilt_deg(state) <= TILT_DEG)


def tcp_points() -> list:
    """World TCP positions of both arms this step, read from the embodiment up the call stack; [] offline."""
    f = sys._getframe(1)
    while f is not None:
        s = f.f_locals.get("self")
        if s is not None and hasattr(s, "_held") and getattr(s, "_base", None) is not None:
            from yam_arena import ik, runtime, station

            scene = s._base.scene
            return [ik.tcp_pose(runtime.arm_q(scene[f"{a}_robot"]), station.BASES[a])[0] for a in ("left", "right")]
        f = f.f_back
    return []


def box_distance(state, size, point) -> float:
    """Distance (m) from ``point`` to the nearest point of a box of ``size`` at ``state``'s pose (0 inside it)."""
    local = _rot(state).T @ (np.asarray(point, float) - np.asarray(state.pos, float))
    half = np.asarray(size, float) / 2
    return float(np.linalg.norm(local - np.clip(local, -half, half)))


def released(state, tcps) -> bool:
    return all(box_distance(state, GLASS_SIZE, p) >= GRIPPER_CLEAR for p in tcps)


def episode_t():
    """The embodiment's episode clock (s) from up the call stack; None offline."""
    f = sys._getframe(1)
    while f is not None:
        s = f.f_locals.get("self")
        if s is not None and hasattr(s, "_held") and getattr(s, "_base", None) is not None:
            return float(s._t)
        f = f.f_back
    return None


_KNOCK = {"t": None, "knocked": False}


def knocked(glasses) -> bool:
    """True once any glass has tipped past 45 deg this episode (a knocked-over wine glass); stays true for the rest of
    the episode, so standing it back up does not undo it. Offline (no clock) it is not latched."""
    t = episode_t()
    if t is not None and _KNOCK["t"] is not None and t < _KNOCK["t"]:
        _KNOCK["knocked"] = False  # the clock went back: a new episode
    over = any(tilt_deg(g) > KNOCK_TILT_DEG for g in glasses)
    if t is None:
        return over
    _KNOCK["t"] = t
    _KNOCK["knocked"] = _KNOCK["knocked"] or over
    return _KNOCK["knocked"]


def _parts(objects):
    """(score, glasses hung, knocked, released, resting)."""
    gs = [objects[n] for n in GLASS_NAMES]
    tcps = tcp_points()
    n_hung = sum(hung(g) for g in gs)
    k = knocked(gs)
    rel = all(released(g, tcps) for g in gs)
    rest = all(g.resting() for g in gs)
    if k or not (rel and rest):
        val = 0.0
    else:
        val = {2: 1.0, 1: 0.5}.get(n_hung, 0.0)
    return val, n_hung, k, rel, rest


_STEP = {"n": 0}


def score(objects) -> float:
    """1.0: both glasses hang in the rack, released, resting, none ever knocked over. 0.5: one does. 0 otherwise."""
    val, n_hung, k, rel, rest = _parts(objects)
    if tcp_points():
        _STEP["n"] += 1
        if _STEP["n"] == 1 or _STEP["n"] % 10 == 0:
            parts = []
            for n in GLASS_NAMES:
                g = objects[n]
                f = foot_centre(g)
                parts.append(f"{n[-1]}=({f[0]:.4f},{f[1]:+.4f},{(f[2] - TOP - H) * 1000:+.1f}mm,{tilt_deg(g):.1f})")
            print(f"GLASS_STEP step={_STEP['n']} {' '.join(parts)} hung={n_hung} knocked={int(k)} rel={int(rel)} "
                  f"rest={int(rest)} score={val:.1f}", flush=True)
    return val


def success(objects) -> bool:
    return _parts(objects)[0] == 1.0


HANG_Y = {"glass_a": 0.050, "glass_b": -0.045}  # the demo's hang spots: A at the closed end, B behind it


def distance(objects) -> float:
    """Metres from the goal: each foot's offset from its demo hang spot on the lips, plus 1 cm per degree of tilt."""
    d = 0.0
    for n in GLASS_NAMES:
        g = objects[n]
        d += float(np.linalg.norm(foot_centre(g) - np.array([XS, HANG_Y[n], TOP + H])) + 0.01 * tilt_deg(g))
    return d


# -- demo (right arm; the left stays at READY) ------------------------------------------------------------------------
class Grip(float):
    """A 3x3 TCP rotation matrix carried in Waypoint.yaw (a float field)."""

    def __new__(cls, m):
        obj = super().__new__(cls, 1.0)
        obj.m = np.asarray(m, float)
        return obj


_orig_grip_rotation = _demo.grip_rotation
_demo.grip_rotation = lambda y=0.0: y.m if hasattr(y, "m") else _orig_grip_rotation(y)

PITCH = -85.0  # fingers almost level, pointing +x; the hand rises only ~34 mm above the TCP (75 deg: 47 mm)
GRIP = Grip(axis_angle_matrix(np.array([0.0, 1.0, 0.0]), math.radians(PITCH)) @ _orig_grip_rotation(0.0))
AXIS = GRIP.m @ np.array([0.0, 0.0, 1.0])  # the fingers' pointing direction
GRIP_D = 0.060  # jaws on the stem 60 mm under the foot: the hand stays ~9 mm under the lips while the foot clears them
FINGER_AJAR = -0.020  # 40 mm between the pads: room around the 9 mm stem, clear of the other glass
APPROACH = 0.08  # come in level along +x
BACK_OUT = 0.10
SAG = 0.017  # the loaded pitched hand rides 15-19 mm low (GPU spike); aim the carry this much higher
CARRY_Z = H - GRIP_D + 0.008 + SAG  # foot ~8 mm over the lips while carried
HANG_TCP_Z = H - GRIP_D - 0.002  # lowered onto the lips (the sag goes once the lips take the glass)
PICK_Z = MAT_T + GL / 2 + FOOT_Z - GRIP_D
ENTRY_Y = YE - 0.040  # foot clear of the open end before sliding in


def _p(x, y, z):
    return tuple(float(v) for v in (x, y, TOP + z))


def _glass_steps(name, first):
    x, y = SPAWN[name]
    pick = np.array(_p(x, y, PICK_Z))
    hang = np.array(_p(XS, HANG_Y[name], HANG_TCP_Z))
    w = lambda d, p, f, lab: Waypoint(d, tuple(float(v) for v in p), f, f"{name}: {lab}", GRIP)  # noqa: E731
    start = [] if first else [w(1.5, _p(0.34, y, PICK_Z + 0.02), FINGER_AJAR, "round the hung glass, out of the rack")]
    return start + [
        w(2.0, pick - APPROACH * AXIS, FINGER_AJAR, "level with the stem"),
        w(1.5, pick, FINGER_AJAR, "in around the stem"),
        w(1.2, pick, FINGER_CLOSED, "close"),
        w(0.5, pick, FINGER_CLOSED, "hold"),
        w(2.0, _p(x, y, CARRY_Z), FINGER_CLOSED, "lift, foot over the lips' height"),
        w(1.5, _p(XS, ENTRY_Y, CARRY_Z), FINGER_CLOSED, "to the open end"),
        w(3.5 if first else 2.5, _p(XS, HANG_Y[name], CARRY_Z), FINGER_CLOSED, "slide along the channel"),
        w(1.0, hang, FINGER_CLOSED, "lower the foot onto the lips"),
        w(1.0, hang, FINGER_AJAR, "let go"),
        w(1.5, hang - BACK_OUT * AXIS, FINGER_AJAR, "back out"),
    ]


def demo():
    """Hang glass A at the closed end, then glass B behind it, each by its stem with the near-level grip."""
    park = _p(0.30, -0.25, 0.20)
    right = _glass_steps("glass_a", True) + _glass_steps("glass_b", False) + [
        Waypoint(2.0, park, FINGER_CLOSED, "park", GRIP),
        Waypoint(1.5, park, FINGER_CLOSED, "hold still", GRIP),
    ]
    return {"right": right}
