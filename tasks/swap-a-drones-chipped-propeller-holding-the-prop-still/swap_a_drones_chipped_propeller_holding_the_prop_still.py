"""A quadcopter drone stands on its skids on the bench, its rear motor nearest the arms. That motor's orange propeller
has a chipped blade. The propeller sits on a 23 mm centring boss on the motor's bell and is clamped by a red prop nut on
the motor's threaded shaft. The motor spins freely (it is switched off), so turning the nut just turns the nut, the
propeller and the motor together, unless a hand holds the propeller still by a blade. A new propeller of the same kind,
its printed side up, lies on a stand to the right; a blue bin for worn parts stands to the left; a small steel parts
dish to the right.

Task: hold the chipped propeller still by a blade with one hand while the other unscrews the nut (anticlockwise) and
lifts it off; put the chipped propeller in the blue bin; set the new propeller on the motor, printed side up, hold it
still by a blade and screw the nut back on (clockwise) until it is tight; then take both arms back to where they
started.

Run it:  inspect-robots run --embodiment yam_arena -E task=swap_a_drones_chipped_propeller_holding_the_prop_still.py \\
           --instruction "..." --scorer task_score ...
"""

from __future__ import annotations

import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from bodykit import (Body, BoxShape, CylShape, SphereShape, allow_matrix_grips, as_numpy, hinge, segment,  # noqa: E402
                     on_reset, on_step, rotation, set_slide_target, set_thread_on, thread)
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, TCP_OFFSET  # noqa: E402
from yam_arena.demo import Waypoint, grip_rotation  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

allow_matrix_grips()

SCENE = "station"
TOP = TABLE_TOP[SCENE]

# -- the drone ----------------------------------------------------------------------------------------------------------
M = np.array([0.30, -0.02])  # the rear motor's axis (the motor being serviced)
ARM_L = 0.24  # centre to each motor ("+" layout: rear motor nearest the arms)
C = M + np.array([ARM_L, 0.0])  # the drone's centre
ARM_Z, ARM_T = TOP + 0.050, 0.016  # the arm tubes' centre height and thickness
MOUNT_TOP = TOP + 0.063  # the motor mount plates' top (the bell turns just above it)
Z_BELL = TOP + 0.087  # the bell's top face: the propeller's hub sits on it
BELL_R, BELL_H = 0.016, Z_BELL - MOUNT_TOP - 0.001
SPIGOT_R, BOSS_H, CHAMFER = 0.0115, 0.011, 0.0025  # the round 23 mm centring boss on the bell's top, its top edge
# chamfered 2.5 mm (over its top 3 mm) as a lead-in
SHAFT_R, PILOT_R = 0.005, 0.0035  # the threaded shaft and the plain pilot above it

# -- the propellers (10 x 4.5 inch class, 30 cm across) ---------------------------------------------------------------
HUB_H = 0.016
HUB_TOP = Z_BELL + HUB_H
HOLE = 0.024  # the hub's 24 mm centre hole: 0.5 mm of play on the 23 mm boss (the nut's clamp, not a key, makes the
# propeller turn the motor)
HUB_OUT = 0.036
PROP_R = 0.150
BLADE_SEGS = ((0.018, 0.060, 0.026, 0.008), (0.060, 0.110, 0.024, 0.007), (0.110, PROP_R, 0.020, 0.007))
BLADE_TOP = HUB_H / 2 + 0.0035  # a blade's top above the hub's bottom
PROP_MASS = 0.035
STAND_C = np.array([0.27, -0.30])  # the new propeller's hub on its stand
STAND_TOP = TOP + 0.074

# -- the nut and its thread -------------------------------------------------------------------------------------------
NUT_F = 0.019  # across the hex flats
NUT_HEX = (0.003, 0.033)  # the hex from 3 to 33 mm above the nut's bottom (a washer flange below it): a tall
# bullet nut
NUT_TOP = 0.038  # with its dome
NUT_MASS = 0.012
PITCH = 0.0135  # m per turn (a coarse quick thread)
LEN_ENG = 0.008  # the thread lets go once the nut's bottom is this far above the hub's top
PRELOAD = 0.0010  # the tight nut is pulled this far below its seat (presses the hub with K_S * PRELOAD, capped)
PRELOAD_MAX = 0.0015
K_S, F_MAX = 8000.0, 4.0  # the thread's pull along the shaft (N/m, at most N)
TURN_DRAG = 0.15  # N m s/rad: thread friction: the nut turning on the shaft drags the free motor round with it
MOTOR_DRAG = 1.0e-4  # N m per deg/s: the motor's bearings (it spins freely)
CATCH_R, CATCH_TILT = 0.008, math.radians(10.0)  # a nut this close to the shaft's axis and upright catches the thread
CATCH_S = LEN_ENG - 0.001  # once its bottom is this low above the hub's top (below where it lets go)
SEATED_S = 0.0008
TIGHT_DEG = 16.0  # tight: seated, then turned on at least this far past where it first seated
TIGHT = TIGHT_DEG / 360.0 * PITCH  # ... which the thread turns into a pull this far into its seat (0.6 mm)

# -- the bin and the dish ---------------------------------------------------------------------------------------------
BIN_C, BIN_IN, BIN_WALL = np.array([0.16, 0.0]), (0.120, 0.360), 0.035
DISH_C, DISH_IN, DISH_WALL = np.array([0.20, -0.24]), 0.060, 0.012
DISH_FLOOR = TOP + 0.004

ORANGE, ORANGE_D = (0.95, 0.45, 0.08), (0.75, 0.32, 0.05)
CARBON, BODY, MOTOR, STEEL = (0.08, 0.08, 0.09), (0.20, 0.21, 0.23), (0.14, 0.14, 0.16), (0.72, 0.73, 0.75)
RED, WHITE, GREEN, BLUE = (0.78, 0.10, 0.10), (0.92, 0.92, 0.92), (0.15, 0.65, 0.25), (0.12, 0.28, 0.70)
YELLOW, DARK = (0.92, 0.78, 0.15), (0.05, 0.05, 0.05)

# -- scoring ----------------------------------------------------------------------------------------------------------
SEAT_XY, SEAT_Z, SEAT_TILT = 0.0015, 0.004, 8.0  # the new propeller on the boss: hub this close, up within 8 degrees
HOME_WITHIN = 0.05
SUCCESS_HOLD_S = 1.0


def _rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _blades(z_mid: float, yaw: float, color, collide: bool, chipped: bool = False) -> list:
    """Two blades along +-y (turned by ``yaw``), their middle plane at ``z_mid`` (body frame of a hub on the axis)."""
    out = []
    for sign in (1.0, -1.0):
        for k, (r0, r1, w, t) in enumerate(BLADE_SEGS):
            if chipped and sign < 0 and k == 2:
                r1 = 0.128  # the chipped tip
            mid = sign * (r0 + r1) / 2
            p = _rz(yaw) @ np.array([0.0, mid, 0.0])
            out.append(BoxShape((p[0], p[1], z_mid), (w, r1 - r0, t), color, yaw=yaw, collide=collide))
            # the leading edge, a darker stripe along the blade's top
            q = _rz(yaw) @ np.array([sign * 0.3 * w, mid, 0.0])
            out.append(BoxShape((q[0], q[1], z_mid + t / 2 + 0.0002), (0.25 * w, r1 - r0, 0.0004), ORANGE_D,
                                yaw=yaw, collide=False))
        tip_r = 0.128 if (chipped and sign < 0) else PROP_R
        p = _rz(yaw) @ np.array([0.0, sign * (tip_r - 0.004), 0.0])
        if not (chipped and sign < 0):
            out.append(CylShape((p[0], p[1], z_mid), 0.010, 0.006, color, collide=False))
    return out


def _drone() -> Body:
    """The airframe (body, arms, skids, the three motors not being serviced with their propellers) and the rear motor's
    mount plate; origin at the drone's centre on the table. One fixed body."""
    s = []
    # the body: a dark shell with a battery strapped on top and a camera gimbal under its front
    s += [BoxShape((0.0, 0.0, 0.058), (0.170, 0.110, 0.040), BODY),
          BoxShape((0.0, 0.0, 0.086), (0.130, 0.075, 0.016), (0.10, 0.11, 0.12)),
          BoxShape((0.0, 0.0, 0.095), (0.020, 0.080, 0.002), DARK, collide=False),
          BoxShape((-0.050, 0.0, 0.0795), (0.012, 0.090, 0.003), YELLOW, collide=False),
          BoxShape((0.095, 0.0, 0.040), (0.030, 0.040, 0.030), DARK),
          CylShape((0.112, 0.0, 0.040), 0.010, 0.006, (0.15, 0.25, 0.40), collide=False)]
    # the skids
    for sy in (-1.0, 1.0):
        s += [BoxShape((0.0, sy * 0.075, 0.004), (0.220, 0.010, 0.008), DARK),
              BoxShape((0.050, sy * 0.065, 0.020), (0.008, 0.008, 0.032), DARK),
              BoxShape((-0.050, sy * 0.065, 0.020), (0.008, 0.008, 0.032), DARK)]
    # the four arms and motors
    for name, d in (("rear", (-1.0, 0.0)), ("front", (1.0, 0.0)), ("left", (0.0, 1.0)), ("right", (0.0, -1.0))):
        d = np.array(d)
        r0 = 0.055 if d[0] == 0 else 0.085
        mid = d * (r0 + ARM_L) / 2
        yaw = 0.0 if d[0] != 0 else math.pi / 2
        s.append(BoxShape((mid[0], mid[1], ARM_Z - TOP), (ARM_L - r0, ARM_T, ARM_T), CARBON, yaw=yaw))
        m = d * ARM_L
        s.append(CylShape((m[0], m[1], MOUNT_TOP - TOP - 0.0025), 0.023, 0.005, DARK))
        led = d * (ARM_L - 0.03)
        s.append(BoxShape((led[0], led[1], ARM_Z - TOP + ARM_T / 2 + 0.002), (0.010, 0.010, 0.004),
                          GREEN if name in ("rear", "left") else RED, collide=False))
        if name == "rear":
            continue
        zb = MOUNT_TOP - TOP + BELL_H / 2
        s += [CylShape((m[0], m[1], zb), BELL_R, BELL_H, MOTOR),
              CylShape((m[0], m[1], zb + BELL_H / 2 - 0.004), BELL_R + 0.0005, 0.002, STEEL, collide=False),
              CylShape((m[0], m[1], Z_BELL - TOP + HUB_H / 2), 0.0185, HUB_H, ORANGE),
              CylShape((m[0], m[1], HUB_TOP - TOP + 0.019), 0.0095, 0.038, RED)]
        blade_yaw = {"front": math.radians(35.0), "left": math.radians(-25.0), "right": math.radians(60.0)}[name]
        for shape in _blades(Z_BELL - TOP + HUB_H / 2, blade_yaw, ORANGE, True):
            p = np.array(shape.centre) + np.array([m[0], m[1], 0.0])
            if isinstance(shape, BoxShape):
                s.append(BoxShape(tuple(p.tolist()), shape.size, shape.color, yaw=shape.yaw, collide=shape.collide))
            else:
                s.append(CylShape(tuple(p.tolist()), shape.radius, shape.height, shape.color, collide=shape.collide))
    return Body.build("drone", pos=(C[0], C[1], TOP), mass=2.0, kinematic=True, shapes=s, color=BODY)


def _bell() -> Body:
    """The rear motor's bell (it turns on the motor's bearings), with the centring boss and the threaded shaft;
    origin on the axis at the bell's top face."""
    s = [CylShape((0.0, 0.0, -BELL_H / 2), BELL_R, BELL_H, MOTOR),
         CylShape((0.0, 0.0, -BELL_H + 0.004), BELL_R + 0.0005, 0.002, STEEL, collide=False),
         CylShape((0.0, 0.0, (BOSS_H - 0.003) / 2), SPIGOT_R, BOSS_H - 0.003, STEEL),
         CylShape((0.0, 0.0, BOSS_H - 0.0015), SPIGOT_R - CHAMFER - 0.001, 0.003, STEEL),
         CylShape((0.0, 0.0, (HUB_H + LEN_ENG) / 2), SHAFT_R, HUB_H + LEN_ENG, STEEL),
         CylShape((0.0, 0.0, HUB_H + LEN_ENG + 0.004), PILOT_R, 0.008, STEEL)]
    for k in range(16):  # the boss's chamfer: bars sloping from its full radius to CHAMFER less over its top 3 mm
        u = (math.cos(2 * math.pi * k / 16), math.sin(2 * math.pi * k / 16))
        r0, r1 = SPIGOT_R - 0.001, SPIGOT_R - CHAMFER - 0.001  # bar centres: their outer faces 1 mm further out
        s.append(segment((r0 * u[0], r0 * u[1], BOSS_H - 0.003), (r1 * u[0], r1 * u[1], BOSS_H - 0.0003), 0.002, STEEL))
    for k in range(4):  # the thread's grooves
        s.append(CylShape((0.0, 0.0, HUB_H + 0.0015 + k * 0.002), SHAFT_R + 0.0003, 0.0006, (0.45, 0.46, 0.48),
                          collide=False))
    return Body.build("bell", pos=(M[0], M[1], Z_BELL), mass=0.08, shapes=s, color=MOTOR, com=(0.0, 0.0, -0.008),
                      friction=(0.8, 0.7))


def _prop(name: str, hub_xy, hub_z: float, chipped: bool, yaw: float = 0.0) -> Body:
    """A propeller: a hub with a square hole and two blades along +-y; origin on its axis at the hub's bottom."""
    h, o = HOLE / 2, HUB_OUT / 2
    t = o - h
    s = [BoxShape((0.0, h + t / 2, HUB_H / 2), (2 * o, t, HUB_H), ORANGE, collide=True),
         BoxShape((0.0, -h - t / 2, HUB_H / 2), (2 * o, t, HUB_H), ORANGE),
         BoxShape((h + t / 2, 0.0, HUB_H / 2), (t, 2 * h, HUB_H), ORANGE),
         BoxShape((-h - t / 2, 0.0, HUB_H / 2), (t, 2 * h, HUB_H), ORANGE),
         CylShape((0.0, 0.0, HUB_H / 2), o + 0.0005, HUB_H - 0.001, ORANGE, collide=False)]
    s += _blades(HUB_H / 2, 0.0, ORANGE, True, chipped=chipped)
    if chipped:  # the broken end and a crack running in from it
        s += [BoxShape((0.004, -0.126, HUB_H / 2), (0.012, 0.006, 0.0072), ORANGE_D, yaw=0.5, collide=False),
              BoxShape((0.0, -0.110, BLADE_TOP + 0.0003), (0.0015, 0.030, 0.0005), DARK, yaw=0.15, collide=False)]
    else:  # the printed size on the top of each blade
        for sign in (1.0, -1.0):
            s.append(BoxShape((0.0, sign * 0.045, HUB_H / 2 + 0.0042), (0.010, 0.026, 0.0004), WHITE, collide=False))
    return Body.build(name, pos=(float(hub_xy[0]), float(hub_xy[1]), hub_z), mass=PROP_MASS, shapes=s, color=ORANGE,
                      com=(0.0, 0.0, HUB_H / 2), friction=(1.2, 1.0), yaw=yaw)


def _nut() -> Body:
    """The red prop nut: a washer flange, a hex and a dome; origin on its axis at its bottom."""
    hex_h = NUT_HEX[1] - NUT_HEX[0]
    side = NUT_F / math.sqrt(3.0)
    s = [CylShape((0.0, 0.0, NUT_HEX[0] / 2), 0.0145, NUT_HEX[0], RED)]
    for k in range(3):  # three bars make the hex; flats facing 30, 90 and 150 degrees
        s.append(BoxShape((0.0, 0.0, NUT_HEX[0] + hex_h / 2), (NUT_F, side, hex_h), RED,
                          yaw=math.pi / 6 + k * math.pi / 3))
    s += [CylShape((0.0, 0.0, NUT_HEX[1] + (NUT_TOP - NUT_HEX[1]) / 2), 0.0065, NUT_TOP - NUT_HEX[1], RED),
          SphereShape((0.0, 0.0, NUT_TOP), 0.0065, RED, collide=False)]
    return Body.build("nut", pos=(M[0], M[1], HUB_TOP), mass=NUT_MASS, shapes=s, color=RED, friction=(1.2, 1.0),
                      ignore=("bell",))


def _bin() -> Body:
    """The blue worn-parts bin; origin at its floor's top middle."""
    w, (lx, ly), hgt = 0.004, BIN_IN, BIN_WALL
    s = [BoxShape((0.0, 0.0, -w / 2), (lx + 2 * w, ly + 2 * w, w), BLUE)]
    for sx in (-1.0, 1.0):
        s.append(BoxShape((sx * (lx / 2 + w / 2), 0.0, hgt / 2), (w, ly + 2 * w, hgt), BLUE))
    for sy in (-1.0, 1.0):
        s.append(BoxShape((0.0, sy * (ly / 2 + w / 2), hgt / 2), (lx, w, hgt), BLUE))
    s.append(BoxShape((-lx / 2 - w - 0.0003, 0.0, hgt * 0.6), (0.0006, 0.10, 0.012), YELLOW, collide=False))
    return Body.build("bin", pos=(BIN_C[0], BIN_C[1], TOP + w), mass=0.5, kinematic=True, shapes=s, color=BLUE)


def _stand() -> Body:
    """The stand the new propeller lies on: a foot, a post and a square peg in its hub; origin under the post."""
    peg = 0.014
    s = [BoxShape((0.0, 0.0, 0.003), (0.070, 0.070, 0.006), (0.30, 0.31, 0.33)),
         CylShape((0.0, 0.0, (STAND_TOP - TOP) / 2 + 0.003), 0.010, STAND_TOP - TOP - 0.006, (0.30, 0.31, 0.33)),
         CylShape((0.0, 0.0, STAND_TOP - TOP - 0.002), 0.016, 0.004, (0.30, 0.31, 0.33)),
         BoxShape((0.0, 0.0, STAND_TOP - TOP + 0.004), (peg, peg, 0.008), STEEL)]
    return Body.build("stand", pos=(STAND_C[0], STAND_C[1], TOP), mass=0.5, kinematic=True, shapes=s,
                      color=(0.30, 0.31, 0.33))


def _dish() -> Body:
    """The small steel parts dish; origin at its floor's top middle."""
    w, a, hgt = 0.003, DISH_IN, DISH_WALL
    s = [BoxShape((0.0, 0.0, -w / 2), (a + 2 * w, a + 2 * w, w), STEEL)]
    for sx in (-1.0, 1.0):
        s.append(BoxShape((sx * (a / 2 + w / 2), 0.0, hgt / 2), (w, a + 2 * w, hgt), STEEL))
        s.append(BoxShape((0.0, sx * (a / 2 + w / 2), hgt / 2), (a, w, hgt), STEEL))
    return Body.build("dish", pos=(DISH_C[0], DISH_C[1], DISH_FLOOR), mass=0.2, kinematic=True, shapes=s,
                      color=STEEL, friction=(0.8, 0.7))


hinge("motor", "drone", "bell", anchor=(float(M[0]), float(M[1]), Z_BELL), axis="Z", damping=MOTOR_DRAG)
thread("nut_thread", "bell", "nut", anchor0=(0.0, 0.0, HUB_H), axis0=(0.0, 0.0, 1.0), anchor1=(0.0, 0.0, 0.0),
       axis1=(0.0, 0.0, 1.0), slide_limits=(-0.004, 0.014), turn_damping=TURN_DRAG, slide_stiffness=K_S,
       slide_damping=20.0, slide_max_force=F_MAX, always_on=True)

OBJECTS = [_drone(), _bell(), _prop("old_prop", M, Z_BELL, chipped=True), _prop("new_prop", STAND_C, STAND_TOP,
                                                                                  chipped=False),
           _nut(), _bin(), _stand(), _dish()]


# -- followed after every control step --------------------------------------------------------------------------------

def _fresh() -> dict:
    return {"on": True, "x0": -PRELOAD, "turns": 0.0, "target": -PRELOAD, "phi": None, "came_off": False, "armed": False,
            "s": 0.0, "radial": 0.0, "tilt": 0.0, "tcps": None, "seen": False}


_EP = _fresh()
_DEBUG = {"on": bool(os.environ.get("PROP_DEBUG")), "steps": 0}


@on_reset
def _new_episode() -> None:
    _EP.clear()
    _EP.update(_fresh())
    _DEBUG["steps"] = 0
    try:
        set_slide_target("nut_thread", -PRELOAD)
        set_thread_on("nut_thread", True)
    except Exception:  # before the scene exists
        pass


def _tcps(base, origin) -> list[np.ndarray]:
    out = []
    for arm in ("left_robot", "right_robot"):
        robot = base.scene[arm]
        i = next(j for j, n in enumerate(robot.body_names) if n == "gripper" or n.endswith("/gripper"))
        g = as_numpy(robot.data.body_link_pose_w)[0][i]
        out.append(g[:3] - origin + rotation(g[3:7]) @ np.array(TCP_OFFSET))
    return out


def _nut_geometry(b_pos, b_rot, n_pos, n_rot) -> dict:
    """The nut against the shaft: its bottom's height above the hub's seat (s), how far off the axis, its tilt, and
    its turn against the bell about the axis (anticlockwise from above positive)."""
    up = b_rot @ np.array([0.0, 0.0, 1.0])
    seat = np.asarray(b_pos, float) + b_rot @ np.array([0.0, 0.0, HUB_H])
    v = np.asarray(n_pos, float) - seat
    s = float(v @ up)
    u = b_rot @ np.array([1.0, 0.0, 0.0])
    w = n_rot @ np.array([1.0, 0.0, 0.0])
    return {"s": s, "radial": float(np.linalg.norm(v - s * up)),
            "tilt": math.acos(max(-1.0, min(1.0, float((n_rot @ np.array([0.0, 0.0, 1.0])) @ up)))),
            "phi": math.atan2(float(np.cross(u, w) @ up), float(u @ w))}


@on_step
def _follow(env) -> None:
    base = env.unwrapped
    origin = as_numpy(base.scene.env_origins[0])
    bp = as_numpy(base.scene["bell"].data.root_link_pose_w)[0]
    npz = as_numpy(base.scene["nut"].data.root_link_pose_w)[0]
    g = _nut_geometry(bp[:3] - origin, rotation(bp[3:7]), npz[:3] - origin, rotation(npz[3:7]))
    _EP.update(s=g["s"], radial=g["radial"], tilt=g["tilt"], seen=True)
    _EP["tcps"] = _tcps(base, origin)
    loosen = 0.0  # the nut's turn against the motor since the last step, anticlockwise from above
    if _EP["phi"] is not None:
        loosen = (g["phi"] - _EP["phi"] + math.pi) % (2 * math.pi) - math.pi
    _EP["phi"] = g["phi"]
    if _EP["on"]:
        _EP["turns"] += loosen / (2 * math.pi)
        target = max(_EP["x0"] + PITCH * _EP["turns"], -PRELOAD_MAX)
        if target >= LEN_ENG:  # unscrewed off the end of the thread
            set_thread_on("nut_thread", False)
            _EP.update(on=False, came_off=True, turns=0.0, armed=False)
        else:
            if target <= -PRELOAD_MAX:  # turned on past tight: the thread does not wind further
                _EP["turns"] = (-PRELOAD_MAX - _EP["x0"]) / PITCH
            _EP["target"] = target
            set_slide_target("nut_thread", target)
    elif not _EP["armed"]:  # a nut just unscrewed must leave the shaft before it can catch the thread again
        _EP["armed"] = g["s"] > CATCH_S + 0.010 or g["radial"] > 0.010
    elif (g["radial"] <= CATCH_R and g["tilt"] <= CATCH_TILT and -0.002 <= g["s"] <= CATCH_S):
        set_slide_target("nut_thread", g["s"])
        set_thread_on("nut_thread", True)
        _EP.update(on=True, x0=g["s"], turns=0.0, target=g["s"])
    if _DEBUG["on"]:
        _DEBUG["steps"] += 1
        if _DEBUG["steps"] % 10 == 0:
            tl, tr = _EP["tcps"]
            nut = npz[:3] - origin
            print(f"[prop] t={_DEBUG['steps'] / 20:5.1f}s on={_EP['on']} s={g['s'] * 1000:5.1f}mm "
                  f"target={_EP['target'] * 1000:5.1f}mm turns={_EP['turns']:5.2f} off={_EP['came_off']} "
                  f"radial={g['radial'] * 1000:4.1f} tilt={math.degrees(g['tilt']):4.1f} "
                  f"bell_yaw={math.degrees(math.atan2(rotation(bp[3:7])[1, 0], rotation(bp[3:7])[0, 0])):6.1f} "
                  f"nut={np.round(nut * 1000).astype(int).tolist()} "
                  f"L={np.round(tl * 1000).astype(int).tolist()} R={np.round(tr * 1000).astype(int).tolist()} "
                  f"R-nut={np.round((tr - nut) * 1000).astype(int).tolist()}", file=sys.stderr, flush=True)
            parts = []
            for name in ("old_prop", "new_prop"):
                q = as_numpy(base.scene[name].data.root_link_pose_w)[0]
                rq = rotation(q[3:7])
                parts.append(f"{name}={np.round((q[:3] - origin) * 1000).astype(int).tolist()}"
                             f"/yaw{math.degrees(math.atan2(rq[1, 0], rq[0, 0])):.0f}"
                             f"/tilt{math.degrees(math.acos(max(-1.0, min(1.0, float(rq[2, 2]))))):.0f}")
            print("[prop]   " + " ".join(parts), file=sys.stderr, flush=True)


# -- scoring --------------------------------------------------------------------------------------------------------

def _tilt_deg(state) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(rotation(state.quat_xyzw)[2, 2])))))


def _status(objects) -> dict:
    old, new = objects["old_prop"], objects["new_prop"]
    o = np.asarray(old.pos, float)
    in_bin = (abs(o[0] - BIN_C[0]) < BIN_IN[0] / 2 and abs(o[1] - BIN_C[1]) < BIN_IN[1] / 2
              and o[2] < TOP + BIN_WALL + 0.02)
    n = np.asarray(new.pos, float)
    seated = (float(np.linalg.norm(n[:2] - M)) < SEAT_XY and abs(n[2] - Z_BELL) < SEAT_Z
              and _tilt_deg(new) < SEAT_TILT)
    nut_seated = _EP["seen"] and _EP["on"] and _EP["radial"] < 0.002 and _EP["s"] < SEATED_S
    return {"in_bin": in_bin, "seated": seated, "caught": bool(_EP["on"] and _EP["seen"]),
            "tight": bool(nut_seated and _EP["target"] <= -TIGHT), "came_off": _EP["came_off"],
            "home": _arms_home()}


def _points(st) -> float:
    if st["came_off"] and st["in_bin"] and st["seated"] and st["tight"] and st["home"]:
        return 1.0
    if st["came_off"] and st["in_bin"] and st["seated"] and st["caught"]:
        return 0.5
    return 0.25 if st["came_off"] else 0.0


def score(objects) -> float:
    """Judged at the end, except "came off", which counts at any time. Came off: the nut was unscrewed off the end of
    the motor's thread at some point. 1.0: the nut came off, the chipped propeller lies in the blue bin, the new
    propeller sits on the motor's boss (its hub within 1.5 mm of the motor's axis and 4 mm of the bell's top, printed
    side up within 8 degrees), the nut is screwed down on it and tight (seated within 0.8 mm, then turned on at least
    16 degrees past where it first seated), and both arms are back where they started (each gripper's fingertip
    centre within 5 cm of its start); 0.5: the nut came off, the chipped propeller is in the bin, the new propeller sits
    on the boss and the nut has caught the thread; 0.25: the nut came off at some point; else 0.0."""
    return _points(_status(objects))


_HOME: dict = {}


def _arms_home() -> bool:
    from yam_arena import ik, station
    from yam_arena.constants import READY

    tcps = _EP.get("tcps")
    if tcps is None:
        return False
    if "tcps" not in _HOME:
        _HOME["tcps"] = [np.asarray(ik.tcp_pose(np.array(READY, float), station.BASES[a])[0], float)
                         for a in ("left", "right")]
    return all(float(np.linalg.norm(t - h)) < HOME_WITHIN for t, h in zip(tcps, _HOME["tcps"]))


def success(objects) -> bool:
    """Full score with the new propeller and the nut still."""
    return score(objects) >= 1.0 and objects["new_prop"].resting() and objects["nut"].resting()


# -- scripted demo ----------------------------------------------------------------------------------------------------
# The left hand comes in from its side along the chipped propeller's left blade, tilted a little down, and pinches the
# blade between its fingers. The right hand comes straight down on the nut, fingers across its flats, and unscrews it in
# three strokes of a third of a turn anticlockwise (the wrist turns only so far: turn, let go, turn back open, take the
# nut again, following it up), lifts it off and sets it in the steel dish. The left lets go and draws back. The right
# takes the chipped propeller by its hub, lifts it off the shaft and drops it into the blue bin, then takes the new
# propeller off its stand by its hub, sets it on the motor and lets go; the left pinches its
# left blade. The right takes the nut from the dish, sets it on the shaft and screws it down in three strokes clockwise
# until it is tight. Both hands let go and go back to where they started.

SIDE_DOWN = math.radians(20.0)  # the left's side grip: pointing -y (in along the blade), tilted this far down
L_SIDE = _rx(SIDE_DOWN) @ np.column_stack([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])  # fingers close
# vertically, pinching the blade's faces
R_SIDE = 0.135  # the side grip's TCP this far out along the blade
SIDE_IN = 0.06  # it comes in this far along its pointing direction
PRE_SIDE = -(0.0035 + 0.006)
R_NUT = grip_rotation(math.pi / 2)  # the right's fingers closing along x (turned by the stroke's yaw)
R_HUB_X = grip_rotation(math.pi / 2)  # across a hub's x faces (blades along y)
NUT_IN_HAND = 0.022  # the right's TCP this far above the nut's bottom
HUB_IN_HAND = -0.005  # the right's TCP this far above a hub's top (below: the fingers cover the hub's sides)
YAW_A, STROKE, STROKES = math.radians(-90.0), math.radians(120.0), 3  # strokes between yaw -90 and +30
SLIP = 0.85  # the nut turns about this much of each stroke
PRE_NUT = -(NUT_F / 2 + 0.006)
PRE_HUB = -(HUB_OUT / 2 + 0.006)
L_AIM = np.array([0.0, 0.0, 0.0])  # aim corrections (measured)
R_AIM_NUT = np.array([0.0, 0.0, 0.0])
R_AIM_HUB = np.array([0.0, 0.0, 0.0])
PLACE_AIM = np.array([-0.010, 0.0, 0.0])  # a hub rides ~13 mm +x in the right's grip (along its closing axis)
TURN_S, RELEASE_S, BACK_S, REGRIP_S = 1.6, 0.4, 0.8, 0.4
LIFT_PROP = 0.045
LEAD_S, LEAD_STEPS = 3.0, 15
_IK_CACHE: dict = {}


def _good_ik(arm: str, pos, rot) -> np.ndarray:
    import itertools

    from yam_arena import ik, station

    key = (arm, tuple(np.round(np.asarray(pos, float), 5)), tuple(np.round(np.asarray(rot, float), 5).ravel()))
    if key in _IK_CACHE:
        return _IK_CACHE[key]
    best = None
    for seed in itertools.product((-0.5, 0.0, 0.5), (0.8, 1.3), (0.4, 1.0), (-0.8, 0.0, 0.8), (-1.4, 0.0, 1.4), (0.0,)):
        q, pe, re = ik.solve(np.asarray(pos, float), ik.matrix_to_quat_xyzw(rot), np.array(seed),
                             base_pos=station.BASES[arm], iters=200)
        if pe < 0.001 and re < math.radians(0.5):
            room = float(np.minimum(q - ik.LOWER, ik.UPPER - q).min())
            if best is None or room > best[1]:
                best = (q, room)
    if best is None:
        raise ValueError(f"no IK solution for the {arm} grip at {pos}")
    _IK_CACHE[key] = best[0]
    return best[0]


def _sweep(arm: str, q0, q1, label: str, finger) -> list[Waypoint]:
    from yam_arena import ik, station

    out = []
    for k in range(1, LEAD_STEPS + 1):
        p, r = ik.tcp_pose(q0 + (q1 - q0) * k / LEAD_STEPS, station.BASES[arm])
        out.append(Waypoint(LEAD_S / LEAD_STEPS, tuple(float(v) for v in p), finger, label, r))
    return out


def _lead_in(arm: str, pos, rot, label: str, finger) -> list[Waypoint]:
    from yam_arena.constants import READY

    return _sweep(arm, np.array(READY, float), _good_ik(arm, pos, rot), label, finger)


def _go_home(arm: str, pos, rot, finger) -> list[Waypoint]:
    from yam_arena.constants import READY

    return _sweep(arm, _good_ik(arm, pos, rot), np.array(READY, float), "back to where it started", finger)


def _w(t, pos, finger, label, rot) -> Waypoint:
    return Waypoint(t, tuple(float(v) for v in pos), finger, label, rot)


def _dur(ws) -> float:
    return sum(w.duration for w in ws)


def _side_tcp(hub_xy, hub_z: float) -> np.ndarray:
    """The left's side-grip TCP on the left blade, in the blade's middle plane."""
    return np.array([hub_xy[0], hub_xy[1] + R_SIDE, hub_z + HUB_H / 2]) + L_AIM


def _hub_tcp(hub_xy, hub_z: float) -> np.ndarray:
    return np.array([hub_xy[0], hub_xy[1], hub_z + HUB_H + HUB_IN_HAND]) + R_AIM_HUB


def _nut_tcp(s: float) -> np.ndarray:
    return np.array([M[0], M[1], HUB_TOP + s + NUT_IN_HAND]) + R_AIM_NUT


def _strokes(direction: float, s_start: float, label: str) -> list[Waypoint]:
    """Three turning strokes (direction +1: anticlockwise, unscrewing; -1: clockwise), following the nut along the
    shaft. Starts gripped at the stroke's first yaw."""
    out = []
    y0 = YAW_A if direction > 0 else YAW_A + STROKE
    turned = 0.0

    def s_after(a):
        rel = SLIP * a / (2 * math.pi)
        return min(max(s_start + direction * PITCH * rel, 0.0), LEN_ENG + 0.004)

    for k in range(STROKES):
        for j in range(1, 5):
            a = turned + STROKE * j / 4
            out.append(_w(TURN_S / 4, _nut_tcp(s_after(a)), FINGER_CLOSED, f"{label} ({k + 1})",
                          _rz(y0 + direction * STROKE * j / 4) @ R_NUT))
        turned += STROKE
        at = _nut_tcp(s_after(turned))
        if k < STROKES - 1:
            out.append(_w(RELEASE_S, at, FINGER_OPEN, "let go", _rz(y0 + direction * STROKE) @ R_NUT))
            for j in range(1, 5):  # fingers wide open, clear of the nut (half-open ones drag it round)
                out.append(_w(BACK_S / 4, at, FINGER_OPEN, "turn back, open",
                              _rz(y0 + direction * STROKE * (1 - j / 4)) @ R_NUT))
            out.append(_w(REGRIP_S, at, FINGER_CLOSED, "take the nut again", _rz(y0) @ R_NUT))
    return out


def _right(t_left_holds: float, t_left_off: float, t_left_holds_new: float) -> tuple[list[Waypoint], dict]:
    ev = {}
    y_lo, y_hi = _rz(YAW_A) @ R_NUT, _rz(YAW_A + STROKE) @ R_NUT
    n0 = _nut_tcp(0.0)
    out = _lead_in("right", n0 + [0.0, 0.0, 0.04], y_lo, "over the nut", PRE_NUT)
    out += [_w(1.0, n0, PRE_NUT, "down round the nut", y_lo),
            _w(0.6, n0, FINGER_CLOSED, "grip it", y_lo)]
    if t_left_holds + 0.5 > _dur(out):
        out.append(_w(t_left_holds + 0.5 - _dur(out), n0, FINGER_CLOSED, "wait for the propeller to be held", y_lo))
    out += _strokes(+1.0, -PRELOAD, "unscrew the nut")
    up = np.array(out[-1].pos) + [0.0, 0.0, 0.025]
    dish_tcp = np.array([DISH_C[0], DISH_C[1], DISH_FLOOR + 0.004 + NUT_IN_HAND])
    high = TOP + 0.145  # carried this high, clear of the propeller's blade
    out += [_w(0.8, up, FINGER_CLOSED, "lift it off", y_hi),
            _w(1.4, np.array([DISH_C[0], DISH_C[1], high]), FINGER_CLOSED, "carry it to the dish", y_hi),
            _w(0.6, dish_tcp + [0.0, 0.0, 0.05], FINGER_CLOSED, "down over the dish", y_hi)]
    ev["nut_away"] = _dur(out)
    out += [_w(0.8, dish_tcp, FINGER_CLOSED, "lower it into the dish", y_hi),
            _w(0.4, dish_tcp, FINGER_OPEN, "let go", y_hi),
            _w(0.8, np.array([DISH_C[0], DISH_C[1], high]), FINGER_OPEN, "rise", y_hi)]
    # the chipped propeller off, by its hub, into the bin
    h_old = _hub_tcp(M, Z_BELL)
    out += [_w(1.4, np.array([M[0], M[1], high]), PRE_HUB, "over the chipped propeller's hub", R_HUB_X),
            _w(0.6, h_old + [0.0, 0.0, 0.03], PRE_HUB, "down to just above it", R_HUB_X)]
    if t_left_off > _dur(out):
        out.append(_w(t_left_off - _dur(out), out[-1].pos, PRE_HUB, "wait for the left to let go", R_HUB_X))
    drop = _hub_tcp(BIN_C, TOP + 0.040)
    out += [_w(1.0, h_old, PRE_HUB, "down round the hub", R_HUB_X),
            _w(0.6, h_old, FINGER_CLOSED, "take it", R_HUB_X),
            _w(1.2, h_old + [0.0, 0.0, LIFT_PROP], FINGER_CLOSED, "lift it off the shaft", R_HUB_X),
            _w(1.8, drop + [0.0, 0.0, 0.02], FINGER_CLOSED, "carry it over the bin", R_HUB_X),
            _w(0.6, drop, FINGER_CLOSED, "lower it", R_HUB_X),
            _w(0.4, drop, FINGER_OPEN, "drop it in the bin", R_HUB_X),
            _w(0.6, drop + [0.0, 0.0, 0.04], FINGER_OPEN, "rise", R_HUB_X)]
    # the new propeller, by its hub, onto the motor
    h_s = _hub_tcp(STAND_C, STAND_TOP)
    h_m = _hub_tcp(M, Z_BELL) + PLACE_AIM
    g = R_HUB_X
    out += [_w(1.8, h_s + [0.0, 0.0, 0.04], PRE_HUB, "over the new propeller's hub", g),
            _w(1.0, h_s, PRE_HUB, "down round the hub", g),
            _w(0.6, h_s, FINGER_CLOSED, "take it", g),
            _w(1.0, h_s + [0.0, 0.0, 0.03], FINGER_CLOSED, "lift it off the stand", g),
            _w(2.2, h_m + [0.0, 0.0, LIFT_PROP], FINGER_CLOSED, "carry it over the motor", g),
            _w(1.4, h_m + [0.0, 0.0, 0.003], FINGER_CLOSED, "set it on the motor's boss", g),
            _w(0.4, h_m + [0.0, 0.0, 0.003], FINGER_OPEN, "let go", g),
            _w(0.6, h_m + [0.0, 0.0, 0.05], FINGER_OPEN, "rise", g)]
    ev["new_set"] = _dur(out)
    # the nut again, from the dish
    out += [_w(1.6, dish_tcp + [0.0, 0.0, 0.05], FINGER_OPEN, "over the nut in the dish", y_hi),
            _w(0.8, dish_tcp - [0.0, 0.0, 0.004], PRE_NUT, "down round it", y_hi),
            _w(0.6, dish_tcp - [0.0, 0.0, 0.004], FINGER_CLOSED, "take it", y_hi),
            _w(0.8, dish_tcp + [0.0, 0.0, 0.05], FINGER_CLOSED, "lift it", y_hi)]
    s_set = CATCH_S - 0.0015
    out.append(_w(1.6, _nut_tcp(s_set) + [0.0, 0.0, 0.03], FINGER_CLOSED, "over the shaft", y_hi))
    if t_left_holds_new > _dur(out):
        out.append(_w(t_left_holds_new - _dur(out), out[-1].pos, FINGER_CLOSED,
                      "wait for the new propeller to be held", y_hi))
    out += [_w(1.2, _nut_tcp(s_set), FINGER_CLOSED, "set it on the shaft", y_hi),
            _w(0.4, _nut_tcp(s_set), FINGER_CLOSED, "settle", y_hi)]
    out += _strokes(-1.0, s_set, "screw it down")
    end = np.array(out[-1].pos)
    out += [_w(0.5, end, FINGER_OPEN, "let go", y_lo),
            _w(0.8, end + [0.0, 0.0, 0.03], FINGER_OPEN, "rise", y_lo)]
    ev["done"] = _dur(out)
    out += _go_home("right", end + [0.0, 0.0, 0.03], y_lo, FINGER_OPEN)
    return out, ev


def _left(ev: dict | None, total: float) -> tuple[list[Waypoint], dict]:
    from yam_arena.constants import READY

    times = {}
    side = _side_tcp(M, Z_BELL)
    out_pt = side - SIDE_IN * L_SIDE[:, 2]
    q_out = _good_ik("left", out_pt, L_SIDE)
    out = _sweep("left", np.array(READY, float), q_out, "beside the chipped propeller's left blade", PRE_SIDE)
    out += [_w(1.2, side, PRE_SIDE, "slide the fingers round the blade", L_SIDE),
            _w(0.6, side, FINGER_CLOSED, "pinch it", L_SIDE)]
    times["holds"] = _dur(out)
    if ev is None:
        return out, times
    out.append(_w(max(ev["nut_away"] - _dur(out), 0.3), side, FINGER_CLOSED,
                  "hold the propeller still while the nut comes off", L_SIDE))
    out += [_w(0.4, side, PRE_SIDE, "let go", L_SIDE),
            _w(0.8, out_pt, PRE_SIDE, "draw back", L_SIDE)]
    times["off"] = _dur(out)
    if ev.get("new_set", 0.0) > _dur(out):
        out.append(_w(ev["new_set"] - _dur(out), out_pt, PRE_SIDE, "wait for the new propeller", L_SIDE))
    out += [_w(1.2, side, PRE_SIDE, "slide the fingers round its left blade", L_SIDE),
            _w(0.6, side, FINGER_CLOSED, "pinch it", L_SIDE)]
    times["holds_new"] = _dur(out)
    out.append(_w(max(ev.get("done", 0.0) - _dur(out), 0.3), side, FINGER_CLOSED,
                  "hold the propeller still while the nut goes on", L_SIDE))
    out += [_w(0.5, side, PRE_SIDE, "let go", L_SIDE),
            _w(0.8, out_pt, PRE_SIDE, "draw back", L_SIDE)]
    out += _sweep("left", q_out, np.array(READY, float), "back to where it started", PRE_SIDE)
    if total > _dur(out):
        out.append(_w(total - _dur(out), out[-1].pos, PRE_SIDE,
                      "hold still: the trial ends once the full score has held 1 s", out[-1].yaw))
    return out, times


def demo():
    _, t0 = _left(None, 0.0)
    _, ev0 = _right(t0["holds"], 0.0, 0.0)
    _, t1 = _left(ev0, 0.0)
    right, ev = _right(t0["holds"], t1["off"], t1["holds_new"])
    _, t2 = _left(ev, 0.0)
    right, ev = _right(t0["holds"], t2["off"], t2["holds_new"])
    left, _ = _left(ev, 0.0)
    total = max(_dur(right), _dur(left)) + 3.0
    right.append(_w(total - _dur(right), right[-1].pos, FINGER_OPEN,
                    "hold still: the trial ends once the full score has held 1 s", right[-1].yaw))
    left, _ = _left(ev, total)
    return {"left": left, "right": right}
