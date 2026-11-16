"""A small four-stroke engine (the kind on a generator or pressure washer) sits on a wooden stand at the back of the
bench, its crankcase facing the robot. Low on that face a short drain pipe sticks out toward the arms, closed by a
hex-head drain plug. In front of the engine are a black oil drain pan, off to the robot's left, and a small steel
parts dish. The engine is due for an oil change.

A trained mechanic slides the drain pan under the drain pipe before the plug comes out: once the plug is out the oil
runs at once and can't be stopped. The pan stays under the outlet while it drains, and the plug goes in the parts
dish so it isn't lost.
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import yamkit as kit  # noqa: E402  (spawns the modeled props below; nothing on disk changes)
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0
SCORE_OVER = "end"
INSTRUCTION = "The engine is cold. Drain its oil."

# Engine (fixed), sitting on a stand. The crankcase's front face is at x FACE; the drain pipe sticks out 3 cm from it.
FACE, STAND_FACE, STAND_H = 0.285, 0.295, 0.040
ENG_Y0, ENG_Y1, ENG_X1 = -0.08, 0.14, 0.43
PIPE_Y, PIPE_Z, PIPE_LEN, PIPE_R = 0.06, 0.065, 0.030, 0.0075  # PIPE_Z: axis height above the bench
PIPE_END = FACE - PIPE_LEN  # the outlet, where the oil comes out
BORE, WALL = 0.011, 0.003  # square bore of the pipe's colliders, and their wall thickness

# Drain plug: origin at the center of its hex head; the shank runs 2.5 cm back into the pipe.
HEAD_L, HEAD_FLATS = 0.012, 0.020
HEAD_R = HEAD_FLATS / math.sqrt(3)  # corner radius of the hex
SHANK_L, SHANK = 0.025, 0.008
PLUG_START = (PIPE_END - HEAD_L / 2, PIPE_Y, TOP + PIPE_Z)

# Drain pan: origin at the center of its floor's underside.
PAN_R_IN, PAN_R_OUT, PAN_H = 0.055, 0.059, 0.030
PAN_START = (0.235, 0.200, TOP)
LUG, LUG_DX = (0.022, 0.030, 0.022), PAN_R_OUT + 0.009  # the pan's grab handle, on its side toward the robot
OUTLET = (PIPE_END, PIPE_Y)
UNDER_MARGIN = 0.015  # the point below the outlet must be at least 1.5 cm inside the pan's rim

DISH_R, DISH_H = 0.038, 0.014
DISH_POS = (0.25, -0.08, TOP)
VIOLATION_STEPS = 2
MAX_TILT = math.radians(15)

# -- looks -----------------------------------------------------------------------------------

ALU = kit.Look((0.62, 0.63, 0.64), 0.45, metallic=0.6)
ALU_DARK = kit.Look((0.42, 0.43, 0.45), 0.5, metallic=0.6)
BLACK = kit.Look((0.07, 0.07, 0.08), 0.6)
RED = kit.Look((0.75, 0.08, 0.07), 0.35)
YELLOW = kit.Look((0.95, 0.78, 0.10), 0.5)
STEEL = kit.Look((0.80, 0.81, 0.84), 0.25, metallic=1.0)
ZINC = kit.Look((0.72, 0.72, 0.68), 0.3, metallic=1.0)
WOOD = kit.Look((0.62, 0.45, 0.28), 0.8)
PAN_LOOK = kit.Look((0.10, 0.10, 0.11), 0.55)


def _engine():
    """The engine and its stand are one fixed prop; local frame = world frame shifted to (0, 0, TOP)."""
    m = []
    cx, cy = (FACE + ENG_X1) / 2, (ENG_Y0 + ENG_Y1) / 2
    dx, dy = ENG_X1 - FACE, ENG_Y1 - ENG_Y0
    m.append(kit.box(((STAND_FACE + ENG_X1 + 0.01) / 2, cy, STAND_H / 2), (ENG_X1 + 0.01 - STAND_FACE, dy + 0.04,
                                                                            STAND_H), WOOD))
    crank_top = 0.130
    m.append(kit.box((cx, cy, (STAND_H + crank_top) / 2), (dx, dy, crank_top - STAND_H), ALU))
    m.append(kit.box((cx, cy, STAND_H + 0.004), (dx + 0.012, dy + 0.012, 0.008), ALU_DARK))  # mounting base flange
    # Ribs on the crankcase face, the oil-fill cap and dipstick (yellow) and the drain pipe with its boss.
    for k in range(4):
        m.append(kit.box((FACE - 0.0015, ENG_Y0 + 0.02 + k * 0.05, 0.100), (0.003, 0.006, 0.040), ALU_DARK))
    m.append(kit.cylinder((FACE - 0.010, -0.045, 0.090), 0.011, 0.020, ALU_DARK, axis="x"))
    m.append(kit.lathe([(0.0125, 0.0), (0.0125, 0.010), (0.010, 0.014), (0.0, 0.014)], YELLOW,
                       center=(FACE - 0.020, -0.045, 0.090), axis=(-1, 0, 0)))
    m.append(kit.cylinder((FACE - 0.002, PIPE_Y, PIPE_Z), 0.013, 0.006, ALU_DARK, axis="x"))  # drain boss
    m.append(kit.cylinder((FACE - PIPE_LEN / 2, PIPE_Y, PIPE_Z), PIPE_R, PIPE_LEN, ZINC, axis="x"))
    # Cylinder lying back over the crankcase, with cooling fins; spark plug lead at its end.
    m.append(kit.cylinder((0.395, -0.020, 0.158), 0.026, 0.085, ALU_DARK, axis="x"))
    for k in range(8):
        m.append(kit.box((0.358 + k * 0.010, -0.020, 0.158), (0.003, 0.076, 0.058), ALU))
    m.append(kit.cylinder((0.441, -0.020, 0.165), 0.005, 0.012, BLACK, axis="x"))
    # Fuel tank and cap on the left, air filter in front of the cylinder, muffler at the back right.
    m.append(kit.box((0.395, 0.090, 0.155), (0.085, 0.090, 0.050), RED))
    m.append(kit.cylinder((0.380, 0.090, 0.183), 0.011, 0.008, BLACK))
    m.append(kit.box((0.338, -0.020, 0.150), (0.016, 0.080, 0.040), BLACK))
    m.append(kit.cylinder((0.420, -0.095, 0.105), 0.020, 0.060, BLACK, axis="x"))
    m.append(kit.cylinder((0.448, -0.095, 0.105), 0.008, 0.010, STEEL, axis="x"))
    # Recoil starter on the left side, output shaft on the right.
    m.append(kit.lathe([(0.042, 0.0), (0.042, 0.018), (0.034, 0.026), (0.0, 0.028)], BLACK,
                       center=(0.355, ENG_Y1, 0.088), axis="y"))
    m.append(kit.box((0.355, ENG_Y1 + 0.036, 0.088), (0.036, 0.010, 0.010), BLACK))
    m.append(kit.cylinder((0.355, ENG_Y0 - 0.025, 0.080), 0.009, 0.05, STEEL, axis="y"))

    cols = [
        kit.Collider(((STAND_FACE + ENG_X1 + 0.01) / 2, cy, STAND_H / 2), size=(ENG_X1 + 0.01 - STAND_FACE, dy + 0.04,
                                                                                 STAND_H)),
        kit.Collider((cx, cy, (STAND_H + crank_top) / 2), size=(dx, dy, crank_top - STAND_H)),
        kit.Collider((0.390, cy, 0.160), size=(0.110, dy, 0.060)),
    ]
    # The pipe as a square tube along x around the plug's shank.
    hb, pc = BORE / 2, FACE - PIPE_LEN / 2
    for sy, sz, sizey, sizez in ((0, -1, BORE + 2 * WALL, WALL), (0, 1, BORE + 2 * WALL, WALL),
                                 (-1, 0, WALL, BORE), (1, 0, WALL, BORE)):
        cols.append(kit.Collider((pc, PIPE_Y + sy * (hb + WALL / 2), PIPE_Z + sz * (hb + WALL / 2)),
                                 size=(PIPE_LEN, sizey, sizez)))
    return kit.Prop("engine", pos=(0.0, 0.0, TOP), size=(0.90, 0.60, 0.20), meshes=tuple(m), colliders=tuple(cols))


def _hex_x(r, x0, x1, look):
    """A hex prism along x with two flats facing +-y (corners up and down)."""
    pts2 = [(r * math.cos(math.radians(30 + 60 * k)), r * math.sin(math.radians(30 + 60 * k))) for k in range(6)]
    pts = [(x0, y, z) for y, z in pts2] + [(x1, y, z) for y, z in pts2]
    counts, idx = [6, 6], [5, 4, 3, 2, 1, 0, 6, 7, 8, 9, 10, 11]
    for i in range(6):
        j = (i + 1) % 6
        counts.append(4)
        idx += [i, j, 6 + j, 6 + i]
    return kit._mesh(pts, counts, idx, look)


def _plug():
    """A hex-head drain plug with a copper crush washer. The threads are not modeled: the shank is a plain rod that
    slides out of the pipe's bore."""
    m = (
        _hex_x(HEAD_R, -HEAD_L / 2, HEAD_L / 2, STEEL),
        kit.cylinder((HEAD_L / 2 + 0.001, 0, 0), 0.0105, 0.002, kit.Look((0.70, 0.45, 0.20), 0.4, metallic=1.0),
                     axis="x"),
        kit.cylinder((HEAD_L / 2 + SHANK_L / 2, 0, 0), SHANK / 2 + 0.0005, SHANK_L, STEEL, axis="x", n=20),
    )
    cols = (kit.Collider((0, 0, 0), size=(HEAD_L, HEAD_FLATS, HEAD_FLATS)),
            kit.Collider((HEAD_L / 2 + SHANK_L / 2, 0, 0), size=(SHANK_L, SHANK, SHANK)))
    return kit.Prop("plug", pos=PLUG_START, size=(HEAD_L + SHANK_L, HEAD_FLATS, 2 * HEAD_R), mass=0.03, meshes=m,
                    colliders=cols)


def _pan():
    prof = [(0.0, 0.0), (PAN_R_IN - 0.006, 0.0), (PAN_R_OUT - 0.002, 0.002), (PAN_R_OUT, PAN_H),
            (PAN_R_OUT + 0.004, PAN_H), (PAN_R_OUT + 0.004, PAN_H - 0.003), (PAN_R_IN, PAN_H - 0.003),
            (PAN_R_IN - 0.002, 0.004), (0.0, 0.004)]
    m = [kit.lathe(prof, PAN_LOOK, n=48)]
    m.append(kit.box((-LUG_DX, 0, LUG[2] / 2), LUG, PAN_LOOK))  # molded grab handle on the rim
    m.append(kit.box((-LUG_DX - LUG[0] / 2 + 0.003, 0, LUG[2] - 0.004), (0.006, LUG[1] + 0.004, 0.008), PAN_LOOK))
    cols = [kit.Collider((0, 0, 0.002), radius=PAN_R_OUT, height=0.004),
            kit.Collider((-LUG_DX, 0, LUG[2] / 2), size=LUG)]
    n = 16
    for k in range(n):
        t = 2 * math.pi * k / n
        r = (PAN_R_IN + PAN_R_OUT) / 2 + 0.001
        cols.append(kit.Collider((r * math.cos(t), r * math.sin(t), PAN_H / 2),
                                 size=(0.006, 2 * r * math.tan(math.pi / n) + 0.002, PAN_H)))
    return kit.Prop("pan", pos=PAN_START, size=(2 * PAN_R_OUT + 0.04, 2 * PAN_R_OUT, PAN_H), mass=0.08,
                    meshes=tuple(m), colliders=tuple(cols))


def _dish():
    prof = [(0.0, 0.0), (DISH_R - 0.008, 0.0), (DISH_R, DISH_H), (DISH_R + 0.003, DISH_H),
            (DISH_R - 0.002, 0.002), (0.0, 0.002)]
    m = (kit.lathe(prof, STEEL, n=40), kit.cylinder((0, 0, 0.004), 0.012, 0.004, BLACK, n=24))  # magnet base
    cols = [kit.Collider((0, 0, 0.001), radius=DISH_R, height=0.002)]
    n = 12
    for k in range(n):
        t = 2 * math.pi * k / n
        cols.append(kit.Collider(((DISH_R - 0.002) * math.cos(t), (DISH_R - 0.002) * math.sin(t), DISH_H / 2),
                                 size=(0.004, 2 * DISH_R * math.tan(math.pi / n), DISH_H)))
    return kit.Prop("dish", pos=DISH_POS, size=(2 * DISH_R + 0.006, 2 * DISH_R + 0.006, DISH_H), meshes=m,
                    colliders=tuple(cols))


OBJECTS = [_engine(), _dish(), _pan(), _plug()]
_START = {"plug": np.array(PLUG_START), "pan": np.array(PAN_START)}


def _upright(s):
    qx, qy = s.quat_xyzw[0], s.quat_xyzw[1]
    return bool(1 - 2 * (qx * qx + qy * qy) > math.cos(MAX_TILT))


def _plug_out(s):
    """The plug is out once its shank has left the pipe: its head 2.5 cm or more from where it started."""
    return bool(np.linalg.norm(np.asarray(s.pos) - _START["plug"]) > SHANK_L)


def _pan_under(s):
    """The pan under the outlet: flat on the bench and the point below the outlet at least 1.5 cm inside its rim."""
    return bool(_upright(s) and s.pos[2] < TOP + 0.01
                and math.hypot(s.pos[0] - OUTLET[0], s.pos[1] - OUTLET[1]) < PAN_R_IN - UNDER_MARGIN)


def _in_dish(s):
    return bool(math.hypot(s.pos[0] - DISH_POS[0], s.pos[1] - DISH_POS[1]) < DISH_R - 0.004
                and s.pos[2] < TOP + DISH_H + HEAD_FLATS and s.resting())


# --- What happened during the run. success() and score() are called after every control step, so the run's history is
# kept here. yamkit counts the simulator's scene resets (kit.episode()), so a new trial starts with a clean history.
class _History:
    def __init__(self):
        self.episode = kit.episode()  # which trial this history belongs to
        self.last = None
        self.run = 0
        self.spilled = False  # the plug out with no pan under the outlet: never forgiven


_H = _History()


def _update(objects):
    global _H
    if _H.episode != kit.episode():  # the simulator reset the scene: a new trial, a clean history
        _H = _History()
    now = {n: np.asarray(objects[n].pos, dtype=float) for n in _START}
    _H.last = now
    bad = _plug_out(objects["plug"]) and not _pan_under(objects["pan"])
    _H.run = _H.run + 1 if bad else 0
    if _H.run >= VIOLATION_STEPS:
        _H.spilled = True


def _score(objects):
    plug, pan = objects["plug"], objects["pan"]
    if _H.spilled:
        return 0.0
    if not _plug_out(plug):
        return 0.25 if (_pan_under(pan) and pan.resting()) else 0.0
    if not (_pan_under(pan) and pan.resting()):
        return 0.0
    return 1.0 if _in_dish(plug) else 0.75


def success(objects):
    """The plug out and in the parts dish, the pan under the outlet the whole time the plug has been out. Called
    first each step, so it also records the step's history."""
    _update(objects)
    return _score(objects) == 1.0


def score(objects):
    """0 if the plug was ever out of the pipe for 2 control steps in a row (0.1 s) without the pan under the outlet,
    even if the pan was slid under afterwards. With the plug still in: 0.25 if the pan rests under the outlet, else 0.
    With the plug out and the pan under the outlet throughout: 0.75, or 1.0 if the plug rests in the parts dish.
    Possible scores: 0, 0.25, 0.75, 1.0."""
    return _score(objects)


def _grip(top, below=0.010):  # TCP height that puts the finger pads `below` an object's top
    return top - below + 0.013


def demo():
    """Left arm: grip the drain pan by its handle and slide it along the bench until it is under the drain pipe;
    then grip the plug's hex head, draw it straight out of the pipe, carry it to the parts dish and let it go."""
    travel = TOP + 0.20
    t = DEMO_TUNE
    lug0 = np.array(PAN_START[:2]) - (LUG_DX, 0.0)
    lug1 = np.array(OUTLET) + t["pan_goal"] - (LUG_DX, 0.0)
    z_lug = _grip(TOP + LUG[2], 0.010)
    head_z = TOP + PIPE_Z
    hx = PLUG_START[0] + t["plug_dx"]
    hy = PIPE_Y + t["plug_dy"]
    zg = head_z + t["plug_dz"]
    left = [
        Waypoint(1.5, (*lug0, travel), FINGER_OPEN, "over the pan's handle"),
        Waypoint(1.5, (*lug0, z_lug), FINGER_OPEN, "down"),
        Waypoint(0.8, (*lug0, z_lug), FINGER_CLOSED, "grip the handle"),
        Waypoint(3.5, (*lug1, z_lug + 0.002), FINGER_CLOSED, "slide the pan under the pipe"),
        Waypoint(0.6, (*lug1, z_lug + 0.002), FINGER_OPEN, "let go"),
        Waypoint(1.0, (*lug1, travel - 0.08), FINGER_OPEN, "up"),
        Waypoint(1.5, (hx - 0.03, hy, zg + 0.06), FINGER_OPEN, "to the plug"),
        Waypoint(1.2, (hx, hy, zg + 0.06), FINGER_OPEN, "over the head"),
        Waypoint(1.2, (hx, hy, zg), FINGER_OPEN, "down onto the head"),
        Waypoint(0.8, (hx, hy, zg), FINGER_CLOSED, "grip"),
        Waypoint(2.0, (hx - 0.045, hy, zg), FINGER_CLOSED, "draw it out"),
        Waypoint(1.0, (hx - 0.05, hy, zg + 0.06), FINGER_CLOSED, "lift"),
        Waypoint(2.0, (DISH_POS[0] + t["dish"][0], DISH_POS[1] + t["dish"][1], TOP + 0.08), FINGER_CLOSED,
                 "to the dish"),
        Waypoint(1.0, (DISH_POS[0] + t["dish"][0], DISH_POS[1] + t["dish"][1], TOP + 0.045), FINGER_CLOSED, "lower"),
        Waypoint(0.6, (DISH_POS[0] + t["dish"][0], DISH_POS[1] + t["dish"][1], TOP + 0.045), FINGER_OPEN, "let go"),
        Waypoint(1.0, (DISH_POS[0], DISH_POS[1], travel), FINGER_OPEN, "up"),
    ]
    park = (0.16, 0.26, TOP + 0.22)
    left += [Waypoint(1.5, park, FINGER_OPEN, "park"), Waypoint(2.0, park, FINGER_OPEN, "hold")]
    return {"left": left}


DEMO_TUNE = {"pan_goal": np.array([-0.030, 0.0]), "plug_dx": 0.0, "plug_dy": 0.0, "plug_dz": 0.0,
             "dish": (0.0, 0.0)}
