"""On a gantry assembly bench a technician fits a linear-guide carriage onto the open end of its rail and slides it
home. A spring plunger in the rail marks the seated position: the carriage rides up its ramp, drops past it, and the
plunger pops up behind it so the carriage cannot drift back. A carriage left short of the plunger is not locked.
Here: a grey rail lies along x, its open end toward the arms and an orange end stop at the far end, with an orange
spring plunger 4 cm before the stop. A blue carriage with a grip post starts on a mat in front of the right arm.
"""

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import os  # noqa: E402
from partkit import Slider, env, on_reset, on_step, part  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
TOP = TABLE_TOP[SCENE]
INSTRUCTION = ("Pick up the blue carriage by its post, fit it onto the open end of the grey rail, and push it along "
               "the rail past the orange plunger until it touches the end stop. Then let go and keep both fingers at least 3 cm from the post.")

GREY, BLUE, ORANGE = (0.62, 0.64, 0.66), (0.10, 0.30, 0.90), (1.00, 0.50, 0.00)
MAT_Z, RAIL_XY = 0.002, (0.3175, -0.15)   # mat thickness (m); rail centre
CARRIAGE_START = (0.26, 0.02)
HALF_RAIL, HALF_CAR, STOP_T = 97.5, 18.0, 20.0   # mm: rail half length, carriage half length, end stop thickness
PLUNGER_BACK, SEATED_Z = 57.5, 3.0               # mm: plunger's rear face from rail centre; carriage origin height


def mm(x0, x1, y0, y1, z0, z1, color):
    return (((x0 + x1) / 2000, (y0 + y1) / 2000, (z0 + z1) / 2000), ((x1 - x0) / 1000, (y1 - y0) / 1000,
                                                                      (z1 - z0) / 1000), color)


_dx, _dz = 14.0, 2.5
_L = math.hypot(_dx, _dz)
_n = (-_dz / _L, _dx / _L)
_mid = (-HALF_RAIL - 10 + _dx / 2, 11.5 + _dz / 2)
_pitch = math.atan2(-_dz, _dx)
LEAD = (_mid[0] - _n[0], _mid[1] - _n[1], _L, (math.cos(_pitch / 2), 0.0, math.sin(_pitch / 2), 0.0))
# Rail frame: origin on the rail centre at its base, x along the rail. Neck 8 mm, head 16 mm (top at 14 mm). The open end
# has a lead-in chamfer: a 2 mm plank rising from 11.5 mm, 10 mm short of the end, to the head's top 4 mm inside it.
RAIL = part("rail", pos=(RAIL_XY[0], RAIL_XY[1], TOP + MAT_Z), mass=6.0, friction=1.2, color=GREY, boxes=[
    mm(-HALF_RAIL, HALF_RAIL, -4, 4, 0, 8, GREY),
    ((LEAD[0] / 1000, 0.0, LEAD[1] / 1000), (LEAD[2] / 1000, 0.016, 0.002), GREY, LEAD[3]),
    mm(-HALF_RAIL, HALF_RAIL, -8, 8, 8, 14, GREY),
    mm(HALF_RAIL, HALF_RAIL + STOP_T, -14, 14, 0, 30, ORANGE)], sliders=[Slider(
        "plunger", rest=(PLUNGER_BACK / 1000, 0.0, 0.0), travel=0.004, rate=200.0, damping=3.0, mass=0.002,
        boxes=(mm(PLUNGER_BACK - 2, PLUNGER_BACK, -5, 5, 10, 15, ORANGE),
               # ramp: a 2 mm plank whose top runs from the rail head (x 45.5) up to 15 mm at the plunger's back
               ((0.05158, 0.0, 0.01350), (0.01204, 0.010, 0.002), ORANGE,
                (math.cos(-math.atan2(1, 12) / 2), 0.0, math.sin(-math.atan2(1, 12) / 2), 0.0))))])
# Carriage frame: origin on the centre of its footprint at the lips' underside. Plate 26 x 36 mm over the head with a
# 1 mm gap, walls 1 mm clear of the head, lips 1.5 mm thick under the head's shoulders; 30 mm post, 60 mm tall.
CARRIAGE = part("carriage", pos=(*CARRIAGE_START, TOP + MAT_Z), mass=0.08, friction=1.2, color=BLUE, boxes=[
    mm(-HALF_CAR, HALF_CAR, -13, 13, 11, 15, BLUE), mm(-HALF_CAR, HALF_CAR, 9, 13, 0, 11, BLUE),
    mm(-HALF_CAR, HALF_CAR, -13, -9, 0, 11, BLUE), mm(-HALF_CAR, HALF_CAR, 5, 9, 0, 1.5, BLUE),
    mm(-HALF_CAR, HALF_CAR, -9, -5, 0, 1.5, BLUE)], cylinders=[((0.0, 0.0, 0.045), 0.015, 0.060, BLUE)])
OBJECTS = [
    Box("start_mat", pos=(CARRIAGE_START[0], CARRIAGE_START[1], TOP + MAT_Z / 2), size=(0.10, 0.10, MAT_Z),
        color=(0.85, 0.80, 0.45)),
    Box("rail_mat", pos=(RAIL_XY[0], RAIL_XY[1], TOP + MAT_Z / 2), size=(0.27, 0.07, MAT_Z), color=(0.20, 0.25, 0.32)),
    RAIL, CARRIAGE]

_STATE: dict = {}


@on_reset
def _reset(environment=None):
    _STATE.update(recent=[], step=0, scored=None, last=0.0)


@on_step
def _count(environment=None):
    _STATE["step"] += 1


_reset()


def _released(car) -> bool:
    """Both arms: both fingers at least 90% open and both pad tips at least 3 cm from the post's axis (read from the arms)."""
    e = env()
    if e is None:
        return False
    scene = e.unwrapped.scene
    a0, a1 = car.to_world((0, 0, 0.015)), car.to_world((0, 0, 0.075))
    for name in ("left_robot", "right_robot"):
        arm = scene[name]
        q = np.asarray(getattr(arm.data.joint_pos, "torch", arm.data.joint_pos)[0].detach().cpu(), float)
        fingers = q[arm.find_joints(["joint7", "joint8"], preserve_order=True)[0]]
        tips = getattr(arm.data.body_pos_w, "torch", arm.data.body_pos_w)[0].detach().cpu().numpy().astype(float)
        tips = tips[arm.find_bodies(["tip_left", "tip_right"], preserve_order=True)[0]] - np.asarray(
            getattr(scene.env_origins, "torch", scene.env_origins)[0].detach().cpu(), float)
        if fingers.mean() / FINGER_OPEN < 0.9:
            return False
        for t in tips:
            u = np.clip(np.dot(t - a0, a1 - a0) / np.dot(a1 - a0, a1 - a0), 0, 1)
            if np.linalg.norm(t - (a0 + u * (a1 - a0))) < 0.03:
                return False
    return True


def _evaluate(objects) -> float:
    """End-state score from the carriage's pose in the rail frame (mm) only; nothing is remembered but the last second."""
    if _STATE["scored"] == _STATE["step"]:
        return _STATE["last"]
    _STATE["scored"] = _STATE["step"]
    rail, car = objects["rail"], objects["carriage"]
    c = (car.pos - rail.pos) * 1000
    R = car.rotation()
    level = bool(R[2, 2] > math.cos(math.radians(3)) and abs(R[1, 0]) < math.sin(math.radians(3)))
    on_rail = bool(level and abs(c[1]) <= 3.0 and 1.0 <= c[2] <= 5.0 and abs(c[0]) <= HALF_RAIL)
    recent = _STATE["recent"]
    recent.append((c.copy(), _released(car)))
    del recent[:-20]
    still = len(recent) == 20 and all(abs(p - c).max() < 1.0 and r for p, r in recent)
    past = c[0] - HALF_CAR >= PLUNGER_BACK + 1.0 and rail.sliders.get("plunger", -1.0) > -0.0005
    home = HALF_RAIL - 3.0 <= c[0] + HALF_CAR <= HALF_RAIL + 1.0
    _STATE["last"] = 1.0 if on_rail and past and home and still else 0.5 if on_rail else 0.0
    return _STATE["last"]


def score(objects) -> float:
    """1.0: carriage on the rail (level within 3 degrees, lateral offset <= 3 mm, guide lips under the head), wholly
    past the plunger with the plunger back up, front face within 3 mm of the end stop, released (both fingers open and 3 cm from the post) and still for the last 1 s. 0.5: on the rail
    otherwise. 0: not on the rail at the end. Computed from the last second of poses only."""
    return _evaluate(objects)


# -- scripted demo (right arm): pick by the post, enter the open end, slide to the stop, release, park -----------------
# RAIL_VARIANT picks a wrong attempt for the shortcut checks: nothing, half, hold, drop_on_top, elsewhere, liftoff.
def demo():
    variant = os.environ.get("RAIL_VARIANT", "")
    sx, sy = CARRIAGE_START
    ry, fx = RAIL_XY[1], RAIL_XY[0] + (HALF_RAIL - HALF_CAR) / 1000 - 0.0005
    carry, grab = TOP + 0.16, TOP + MAT_Z + 0.052
    seat = TOP + MAT_Z + 0.0075 + 0.050
    entry = RAIL_XY[0] - (HALF_RAIL + HALF_CAR + 12) / 1000
    park = Waypoint(2.0, (0.12, -0.38, TOP + 0.20), FINGER_OPEN, "park")
    if variant == "nothing":
        return {"right": [Waypoint(2.0, (0.12, -0.38, TOP + 0.20), FINGER_OPEN, "park")]}
    pick = [Waypoint(1.5, (sx, sy, carry), FINGER_OPEN, "to the post"), Waypoint(1.5, (sx, sy, grab), FINGER_OPEN, "descend"),
            Waypoint(0.6, (sx, sy, grab), FINGER_OPEN, "settle"), Waypoint(0.8, (sx, sy, grab), FINGER_CLOSED, "grip"),
            Waypoint(1.2, (sx, sy, carry), FINGER_CLOSED, "lift")]
    if variant in ("drop_on_top", "elsewhere"):
        px, py, pz = (fx, ry, TOP + 0.075) if variant == "drop_on_top" else (0.30, 0.0, grab)
        return {"right": pick + [Waypoint(2.0, (px, py, carry), FINGER_CLOSED, "carry"), Waypoint(1.2, (px, py, pz), FINGER_CLOSED, "lower"),
                                 Waypoint(0.8, (px, py, pz), FINGER_OPEN, "let go"), Waypoint(1.2, (px, py, carry), FINGER_OPEN, "up"), park]}
    stop_x = RAIL_XY[0] - 0.025 if variant == "half" else fx
    w = pick + [Waypoint(2.0, (entry, ry, carry), FINGER_CLOSED, "carry"), Waypoint(1.2, (entry, ry, seat + 0.004), FINGER_CLOSED, "lower"),
                Waypoint(0.8, (entry, ry, seat), FINGER_CLOSED, "settle"),
                Waypoint(2.5, (RAIL_XY[0] - 0.05, ry, seat), FINGER_CLOSED, "fit onto the rail"),
                Waypoint(3.0, (stop_x, ry, seat), FINGER_CLOSED, "push to the stop"), Waypoint(0.8, (stop_x, ry, seat), FINGER_CLOSED, "settle")]
    if variant == "hold":
        return {"right": w + [Waypoint(3.0, (stop_x, ry, seat), FINGER_CLOSED, "hold")]}
    w += [Waypoint(0.8, (stop_x, ry, seat), FINGER_OPEN, "let go"), Waypoint(1.2, (stop_x - 0.02, ry, carry), FINGER_OPEN, "up")]
    if variant == "liftoff":
        lift = TOP + 0.12
        w += [Waypoint(1.2, (stop_x, ry, carry), FINGER_OPEN, "back over the post"), Waypoint(1.0, (stop_x, ry, seat), FINGER_OPEN, "down"),
              Waypoint(0.8, (stop_x, ry, seat), FINGER_CLOSED, "regrip"), Waypoint(2.0, (stop_x, ry, lift), FINGER_CLOSED, "lift it off"),
              Waypoint(2.0, (stop_x, ry, lift), FINGER_CLOSED, "hold it up")]
        return {"right": w}
    return {"right": w + [park, Waypoint(2.0, (0.12, -0.38, TOP + 0.20), FINGER_OPEN, "hold")]}
