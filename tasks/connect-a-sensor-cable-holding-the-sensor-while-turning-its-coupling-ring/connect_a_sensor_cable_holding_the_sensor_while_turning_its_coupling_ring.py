"""On the station table stands a robot cell's pick station: an aluminium fixture plate with a part in its nest, a robot
arm parked behind it, and, aimed down at the part from an angled steel bracket, a new inductive proximity sensor (an
M30 steel barrel) that has just been fitted. Its two jam nuts are not tightened yet, so it turns freely in its bracket.
A red protective cap covers the M12 connector at the sensor's back end. The robot's sensor cable comes out of the cable
duct and ends in an M12 connector with a knurled coupling ring, parked on a parking receptacle next to the sensor.

Task: connect the cable to the sensor. Pull the red cap off the sensor's connector and drop it in the blue parts tray;
take the cable's connector off its parking receptacle and push it onto the sensor's connector; then hold the sensor
still with one hand while the other turns the coupling ring until it is tight and the sensor's green light comes on
(the ring's thread drags the loose sensor round with it unless it is held). Then let go of both and take both arms back
to where they started.

Run it:  inspect-robots run --embodiment yam_arena \\
           -E task=connect_a_sensor_cable_holding_the_sensor_while_turning_its_coupling_ring.py \\
           --instruction "..." --scorer task_score ...
"""

from __future__ import annotations

import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from bodykit import (Body, BoxShape, CylShape, SphereShape, allow_matrix_grips, as_numpy, ball, hinge,  # noqa: E402
                     on_reset, on_step, rotation, segment, set_hinge_drive, set_thread_on, thread)
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, TCP_OFFSET  # noqa: E402
from yam_arena.demo import Waypoint, grip_rotation  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

allow_matrix_grips()

SCENE = "station"
TOP = TABLE_TOP[SCENE]

# -- the sensor's axis: aimed 30 degrees down, toward +y (at the part). U points along it from the connector end into
# the sensor. The sensor, its bracket, its cap, the cable's connector and the parking receptacle are built in a frame
# turned by Q (60 degrees about x), whose z axis is -U (toward the connector end) ---------------------------------------
TAU = math.radians(30.0)
U = np.array([0.0, math.cos(TAU), -math.sin(TAU)])
Q = np.array([[1.0, 0.0, 0.0], [0.0, 0.5, -math.sqrt(3) / 2], [0.0, math.sqrt(3) / 2, 0.5]])
Q_XYZW = (0.5, 0.0, 0.0, math.sqrt(3) / 2)
E = np.array([0.300, -0.030, TOP + 0.105])  # the face of the sensor's back end, on its axis
E_PARK = np.array([0.235, -0.030, TOP + 0.105])  # the face of the parking receptacle, on its axis
PLATE_Z = TOP + 0.010  # the fixture plate's top

# -- the sensor (frame Q, origin at E): back-end face at z = 0, the barrel along -z -------------------------------------
STUB_R, STUB_L = 0.0062, 0.010  # the M12 male thread sticking out of the back end
CAP_END_R = 0.0125
BARREL_R, BARREL_Z = 0.015, (-0.082, -0.012)
FACE_Z = (-0.092, -0.082)  # the orange sensing face
BRACKET_Z = (-0.045, -0.039)  # the bracket's plate, perpendicular to the axis ...
BRACKET_B = (-0.036, 0.017)  # ... reaching this far up (local y) from it: kept low, clear of the left hand
NUT_F, NUT_S, NUT_T = 0.034, 0.0196, 0.006  # jam nuts: across flats, side, thickness
NUT_Z = (-0.036, -0.048)
PIN_Z = -0.058  # where the left hand's fingertips hold the barrel (below the bracket)
LED_Z = -0.004  # the green power light: a ring round the back end
SENSOR_MASS = 0.13
SENSOR_DRAG = 0.00002  # N m per degree/s: the loose jam nuts
# -- the cable's M12 connector (frame Q, origin at its cable exit): its open front at z = -SOCK_L ------------------------
SOCK_L = 0.050
RING_R, RING_IN, RING_L = 0.0105, 0.0080, 0.016  # knurled coupling ring: outer radius, cavity apothem, length
BODY_R = 0.0080
RING_C = -SOCK_L + RING_L / 2  # the ring's middle (z)
SOCK_MASS = 0.028
# -- the protective cap (frame Q, origin at E) ----------------------------------------------------------------------------
CAP_R, CAP_IN, CAP_L = 0.0095, 0.0068, 0.014
CAP_MASS = 0.008
# -- the cable: three stiff-ish segments on ball joints, from the duct's mouth to the connector --------------------------
DUCT_MOUTH = np.array([0.330, -0.150, TOP + 0.030])
CABLE_LEN = 0.19
CABLE_T = 0.0055
CABLE_SEG_MASS = 0.006
# -- the thread's grip on the sensor: once the connector is on, turning its ring drags the free sensor round (a spring
# toward the ring's turn, at most LAG degrees behind), and only turning the ring against a sensor held still screws it
# on. Tight after TIGHT turns ---------------------------------------------------------------------------------------
DRAG_K = 0.006  # N m per degree
LAG = 5.0  # degrees
TIGHT = 1.0  # turns
CATCH_S, CATCH_R, CATCH_TILT = 0.004, 0.003, math.radians(15.0)
LOOSE = -0.6  # turns: unscrewed this far, the thread lets go
# -- the tray, the part, the robot ---------------------------------------------------------------------------------------
TRAY_C, TRAY_IN, TRAY_WALL = np.array([0.240, -0.125]), 0.060, 0.018
PART_C = np.array([0.300, 0.078, PLATE_Z + 0.015])
HIDDEN = np.array([0.45, 0.30, TOP - 0.10])
# -- scoring ----------------------------------------------------------------------------------------------------------
LET_GO = 0.040
HOME_WITHIN = 0.05
SUCCESS_HOLD_S = 1.0

ALU, ALU_DARK, STEEL, DARK_STEEL, NICKEL = (0.70, 0.72, 0.74), (0.52, 0.54, 0.56), (0.74, 0.75, 0.77), \
    (0.36, 0.37, 0.39), (0.82, 0.83, 0.84)
BLACK, DARK, GREY, ORANGE, RED, BLUE = (0.06, 0.06, 0.07), (0.18, 0.18, 0.20), (0.45, 0.46, 0.48), \
    (0.95, 0.50, 0.10), (0.80, 0.08, 0.07), (0.16, 0.36, 0.78)
BRASS, GREEN, DIM_GREEN, YELLOW, CABLE_C = (0.80, 0.64, 0.28), (0.15, 1.00, 0.30), (0.10, 0.12, 0.11), \
    (0.95, 0.80, 0.10), (0.93, 0.48, 0.12)


def _rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _hex(z: float, color) -> list:
    """A hex nut round the axis at local height z: three boxes turned 0, 60 and 120 degrees."""
    return [BoxShape((0.0, 0.0, z), (NUT_F, NUT_S, NUT_T), color, yaw=math.radians(a)) for a in (0.0, 60.0, 120.0)]


def _tube(z0: float, z1: float, r_in: float, r_out: float, color, n: int = 8) -> list:
    """A hollow n-sided tube round the local z axis (collision walls), from z0 to z1."""
    rc, t = (r_in + r_out) / 2, r_out - r_in
    w = 2 * r_out * math.tan(math.pi / n) + 0.0004
    return [BoxShape((rc * math.cos(a), rc * math.sin(a), (z0 + z1) / 2), (t, w, z1 - z0), color, yaw=a)
            for a in (2 * math.pi * k / n for k in range(n))]


def _sensor() -> Body:
    s = [CylShape((0.0, 0.0, (STUB_L - 0.003) / 2), STUB_R, STUB_L - 0.003, BRASS),  # the M12 male thread ...
         CylShape((0.0, 0.0, STUB_L - 0.0015), STUB_R - 0.0017, 0.003, BRASS),  # ... with a lead-in at its tip
         CylShape((0.0, 0.0, STUB_L + 0.0002), 0.0042, 0.0004, DARK, collide=False),  # its pin insert
         CylShape((0.0, 0.0, -0.006), CAP_END_R, 0.012, DARK),  # the plastic back end
         CylShape((0.0, 0.0, LED_Z), CAP_END_R + 0.0003, 0.003, DIM_GREEN, collide=False),  # the power light, off
         BoxShape((0.0, CAP_END_R + 0.0002, -0.009), (0.004, 0.001, 0.003), YELLOW, collide=False),  # output light
         CylShape((0.0, 0.0, (BARREL_Z[0] + BARREL_Z[1]) / 2), BARREL_R, BARREL_Z[1] - BARREL_Z[0], STEEL),
         CylShape((0.0, 0.0, (FACE_Z[0] + FACE_Z[1]) / 2), BARREL_R - 0.0002, FACE_Z[1] - FACE_Z[0], ORANGE)]
    for z in np.arange(BARREL_Z[0] + 0.003, BARREL_Z[1] - 0.001, 0.0035):  # the barrel's thread
        if not any(abs(z - nz) < 0.005 for nz in NUT_Z) and not BRACKET_Z[0] - 0.002 < z < BRACKET_Z[1] + 0.002:
            s.append(CylShape((0.0, 0.0, float(z)), BARREL_R + 0.0003, 0.0008, DARK_STEEL, collide=False))
    for z in NUT_Z:
        s += _hex(z, NICKEL)
    return Body.build("sensor", pos=tuple(E.tolist()), mass=SENSOR_MASS, shapes=s, start_xyzw=Q_XYZW, color=STEEL,
                      friction=(0.9, 0.8), com=(0.0, 0.0, -0.045), ignore=("bracket",))


def _bracket() -> Body:
    """The angled bracket's plate the sensor passes through (the plate's foot is part of the fixture)."""
    zc = (BRACKET_Z[0] + BRACKET_Z[1]) / 2
    bc, bh = (BRACKET_B[0] + BRACKET_B[1]) / 2, BRACKET_B[1] - BRACKET_B[0]
    s = [BoxShape((0.0, bc, zc), (0.050, bh, BRACKET_Z[1] - BRACKET_Z[0]), ALU_DARK),
         CylShape((0.0, 0.0, BRACKET_Z[1] + 0.0002), BARREL_R + 0.001, 0.0004, DARK, collide=False)]
    return Body.build("bracket", pos=tuple(E.tolist()), mass=0.3, shapes=s, start_xyzw=Q_XYZW, kinematic=True,
                      color=ALU_DARK)


def _holder() -> Body:
    """The parking receptacle: a block with a dummy M12 thread the connector sits on."""
    s = [BoxShape((0.0, 0.0, -0.004), (0.026, 0.026, 0.008), DARK),
         CylShape((0.0, 0.0, STUB_L / 2), STUB_R, STUB_L, GREY),
         BoxShape((0.0, 0.0132, -0.004), (0.020, 0.0006, 0.005), YELLOW, collide=False)]  # a label stripe
    return Body.build("holder", pos=tuple(E_PARK.tolist()), mass=0.2, shapes=s, start_xyzw=Q_XYZW, kinematic=True,
                      color=DARK)


def _cap() -> Body:
    s = [*_tube(0.0, CAP_L - 0.002, CAP_IN, CAP_R, RED),
         CylShape((0.0, 0.0, CAP_L - 0.001), CAP_R, 0.002, RED),
         CylShape((0.0, 0.0, CAP_L / 2), CAP_R + 0.0002, CAP_L, RED, collide=False),
         BoxShape((0.0, CAP_R + 0.003, CAP_L - 0.003), (0.005, 0.006, 0.0015), RED, collide=False)]  # pull tab
    return Body.build("cap", pos=tuple(E.tolist()), mass=CAP_MASS, shapes=s, start_xyzw=Q_XYZW, color=RED,
                      friction=(1.0, 0.9), com=(0.0, 0.0, CAP_L / 2))


def _socket(at) -> Body:
    """The cable's M12 connector, its origin at the cable exit, its front (the coupling ring) at z = -SOCK_L."""
    z0 = -SOCK_L
    s = [*_tube(z0, z0 + 0.012, RING_IN, RING_R, NICKEL),  # the ring's front, hollow for the male thread
         CylShape((0.0, 0.0, z0 + 0.014), RING_R, 0.004, NICKEL),  # its back
         CylShape((0.0, 0.0, RING_C), RING_R + 0.0002, RING_L, NICKEL, collide=False),
         CylShape((0.0, 0.0, z0 + 0.0002), RING_IN, 0.0004, BLACK, collide=False),  # the open front, dark inside
         CylShape((0.0, 0.0, (z0 + RING_L + 0.0) / 2), BODY_R, -(z0 + RING_L) - 0.002, BLACK),  # the moulded body
         CylShape((0.0, 0.0, -0.002), 0.0048, 0.004, BLACK)]
    for k in range(16):  # the knurling
        a = 2 * math.pi * k / 16
        s.append(BoxShape(((RING_R + 0.0003) * math.cos(a), (RING_R + 0.0003) * math.sin(a), RING_C),
                          (0.0012, 0.0012, RING_L - 0.002), GREY, yaw=a, collide=False))
    for k in range(4):  # grip ribs on the body
        a = math.pi / 4 + math.pi / 2 * k
        s.append(BoxShape(((BODY_R + 0.0004) * math.cos(a), (BODY_R + 0.0004) * math.sin(a), -0.022),
                          (0.0012, 0.0016, 0.018), DARK, yaw=a, collide=False))
    return Body.build("socket", pos=tuple(np.asarray(at, float).tolist()), mass=SOCK_MASS, shapes=s,
                      start_xyzw=Q_XYZW, color=NICKEL, friction=(1.0, 0.9), com=(0.0, 0.0, -0.030),
                      damping=(0.05, 0.1))


SOCK_AT = E_PARK - SOCK_L * U  # the connector's cable exit, parked


def _cable_points() -> list[np.ndarray]:
    """The cable's joints at the start: from the duct's mouth to the parked connector's cable exit, sagging so that the
    three segments add up to CABLE_LEN."""
    a, b = DUCT_MOUTH, SOCK_AT
    d = b - a
    sag = np.array([0.0, -0.35, -0.94])
    sag -= (sag @ d) / (d @ d) * d
    sag /= np.linalg.norm(sag)

    def pts(h):
        return [a + d * t + h * math.sin(math.pi * t) * sag for t in (0.0, 1 / 3, 2 / 3, 1.0)]

    lo, hi = 0.0, 0.2
    for _ in range(40):
        h = (lo + hi) / 2
        p = pts(h)
        if sum(float(np.linalg.norm(p[i + 1] - p[i])) for i in range(3)) < CABLE_LEN:
            lo = h
        else:
            hi = h
    return pts((lo + hi) / 2)


CABLE_P = _cable_points()


def _cable(i: int) -> Body:
    a, b = CABLE_P[i], CABLE_P[i + 1]
    s = [segment((0.0, 0.0, 0.0), tuple((b - a).tolist()), CABLE_T, CABLE_C),
         SphereShape((0.0, 0.0, 0.0), CABLE_T * 0.62, CABLE_C, collide=False)]
    return Body.build(f"cable{i}", pos=tuple(a.tolist()), mass=CABLE_SEG_MASS, shapes=s, color=CABLE_C,
                      friction=(0.6, 0.5), damping=(0.4, 0.4))


def _fixture() -> Body:
    """The fixture plate with the bracket's foot, the parking receptacle's post, the part's nest stops and the cable
    duct; origin on the table."""
    def at(x, y, z):
        return (x, y, z - TOP)

    foot = E + Q @ np.array([0.0, BRACKET_B[0], (BRACKET_Z[0] + BRACKET_Z[1]) / 2])  # the bracket plate's lower edge
    post = E_PARK + Q @ np.array([0.0, -0.013, -0.004])
    s = [BoxShape(at(0.370, -0.040, TOP + 0.005), (0.360, 0.320, 0.010), ALU),
         BoxShape(at(0.300, float(foot[1]), (PLATE_Z + foot[2]) / 2 + 0.002), (0.050, 0.006, float(foot[2] - PLATE_Z)),
                  ALU_DARK),
         BoxShape(at(0.300, float(foot[1]) + 0.012, PLATE_Z + 0.003), (0.050, 0.030, 0.006), ALU_DARK),
         BoxShape(at(0.235, float(post[1]), (PLATE_Z + post[2]) / 2), (0.016, 0.016, float(post[2] - PLATE_Z)), DARK),
         BoxShape(at(0.235, float(post[1]), PLATE_Z + 0.003), (0.034, 0.034, 0.006), DARK)]
    for dx in (-0.032, 0.032):  # the part's nest stops
        s.append(BoxShape(at(float(PART_C[0]) + dx, float(PART_C[1]), PLATE_Z + 0.008), (0.010, 0.050, 0.016), BLUE))
    mx, my, mz = DUCT_MOUTH
    s += [BoxShape(at(0.440, my, PLATE_Z + 0.017), (0.220, 0.034, 0.034), GREY),  # the cable duct ...
          BoxShape(at(0.440, my, PLATE_Z + 0.035), (0.220, 0.038, 0.004), GREY),  # ... its lid
          BoxShape(at(mx + 0.001, my, mz), (0.002, 0.012, 0.012), BLACK, collide=False)]  # its mouth
    return Body.build("fixture", pos=(0.0, 0.0, TOP), mass=5.0, shapes=s, kinematic=True, color=ALU)


def _robot() -> Body:
    """The cell's robot arm, parked behind the station (fixed); origin on the table under its base."""
    bx, by = 0.660, -0.150
    zs = TOP + 0.300

    def at(x, y, z):
        return (x - bx, y - by, z - TOP)

    w = (0.520, 0.030, TOP + 0.240)  # its wrist
    s = [CylShape(at(bx, by, TOP + 0.020), 0.075, 0.040, DARK),
         segment(at(bx, by, TOP + 0.040), at(bx, by, zs), 0.075, ORANGE),
         CylShape(at(bx, by, zs), 0.050, 0.090, GREY, axis="Y"),
         segment(at(bx, by, zs), at(0.550, -0.040, zs), 0.065, ORANGE),
         SphereShape(at(0.550, -0.040, zs), 0.045, GREY),
         segment(at(0.550, -0.040, zs), at(*w), 0.058, ORANGE),
         SphereShape(at(*w), 0.040, GREY),
         segment(at(*w), at(w[0], w[1], w[2] - 0.070), 0.050, GREY),
         CylShape(at(w[0], w[1], w[2] - 0.078), 0.034, 0.016, DARK),
         BoxShape(at(w[0], w[1] - 0.016, w[2] - 0.106), (0.016, 0.008, 0.040), DARK),
         BoxShape(at(w[0], w[1] + 0.016, w[2] - 0.106), (0.016, 0.008, 0.040), DARK),
         BoxShape(at(bx - 0.060, by, TOP + 0.020), (0.050, 0.040, 0.030), DARK)]  # its I/O box (the duct ends here)
    return Body.build("robot", pos=(bx, by, TOP), mass=20.0, shapes=s, kinematic=True, color=ORANGE)


def _tray() -> Body:
    a, w, h = TRAY_IN, 0.003, TRAY_WALL
    s = [BoxShape((0.0, 0.0, w / 2), (a + 2 * w, a + 2 * w, w), BLUE)]
    for sx in (-1.0, 1.0):
        s.append(BoxShape((sx * (a / 2 + w / 2), 0.0, h / 2), (w, a + 2 * w, h), BLUE))
        s.append(BoxShape((0.0, sx * (a / 2 + w / 2), h / 2), (a, w, h), BLUE))
    return Body.build("tray", pos=(float(TRAY_C[0]), float(TRAY_C[1]), PLATE_Z), mass=0.2, kinematic=True, shapes=s,
                      color=BLUE, friction=(0.8, 0.7))


def _part() -> Body:
    s = [BoxShape((0.0, 0.0, 0.0), (0.050, 0.040, 0.030), ALU),
         CylShape((0.0, 0.0, 0.0152), 0.008, 0.0006, DARK_STEEL, collide=False)]
    return Body.build("part", pos=tuple(PART_C.tolist()), mass=0.15, shapes=s, color=ALU, friction=(0.7, 0.6))


def _light() -> Body:
    return Body.build("power_light", pos=tuple(HIDDEN.tolist()), mass=0.001, kinematic=True, color=GREEN,
                      start_xyzw=Q_XYZW,
                      shapes=[CylShape((0.0, 0.0, 0.0), CAP_END_R + 0.0006, 0.0032, GREEN, collide=False)])


OBJECTS = [_fixture(), _robot(), _bracket(), _holder(), _tray(), _part(), _sensor(), _cap(),
           *[_cable(i) for i in range(3)], _socket(SOCK_AT), _light()]
# the sensor turns in its bracket (its jam nuts loose); the task file drives it round with the ring (see _follow)
hinge("sensor_turn", "bracket", "sensor", anchor=tuple(E.tolist()), axis="Z", spring=0.0, damping=SENSOR_DRAG)
ball("cable_mouth", "fixture", "cable0", anchor=tuple(CABLE_P[0].tolist()), axis=tuple(
    ((CABLE_P[1] - CABLE_P[0]) / np.linalg.norm(CABLE_P[1] - CABLE_P[0])).tolist()), swing_damping=0.0008,
     twist_damping=0.0004)
for _i in (1, 2):
    ball(f"cable_{_i}", f"cable{_i - 1}", f"cable{_i}", anchor=tuple(CABLE_P[_i].tolist()), axis=tuple(
        ((CABLE_P[_i + 1] - CABLE_P[_i]) / np.linalg.norm(CABLE_P[_i + 1] - CABLE_P[_i])).tolist()),
         swing_damping=0.0008, twist_damping=0.0004)
ball("cable_exit", "cable2", "socket", anchor=tuple(CABLE_P[3].tolist()), axis=tuple((-U).tolist()),
     swing_damping=0.0008, twist_damping=0.0004)
# the coupling ring's thread: switched on once the connector is on the sensor's thread (holds it on, coaxial)
thread("m12", "sensor", "socket", anchor0=(0.0, 0.0, 0.0), axis0=(0.0, 0.0, 1.0), anchor1=(0.0, 0.0, -SOCK_L),
       axis1=(0.0, 0.0, 1.0), slide_limits=(-0.001, CATCH_S), turn_damping=0.0, slide_stiffness=0.0,
       slide_damping=0.0, slide_max_force=0.0)


# -- followed after every control step --------------------------------------------------------------------------------

def _fresh() -> dict:
    return {"on": False, "turns": 0.0, "tight": False, "a_sock": None, "a_sens": None, "target": 0.0,
            "light": False, "tcps": None, "s": 0.0, "radial": 0.0, "tilt": 0.0, "seen": False, "sens_deg": 0.0}


_EP = _fresh()
_DEBUG = {"on": bool(os.environ.get("M12_DEBUG")), "steps": 0}


@on_reset
def _new_episode() -> None:
    _EP.clear()
    _EP.update(_fresh())
    _DEBUG["steps"] = 0


def _tcps(base, origin) -> list[np.ndarray]:
    out = []
    for arm in ("left_robot", "right_robot"):
        robot = base.scene[arm]
        i = next(j for j, n in enumerate(robot.body_names) if n == "gripper" or n.endswith("/gripper"))
        g = as_numpy(robot.data.body_link_pose_w)[0][i]
        out.append(g[:3] - origin + rotation(g[3:7]) @ np.array(TCP_OFFSET))
    return out


def _place(base, origin, name, at, xyzw=Q_XYZW) -> None:
    import torch

    asset = base.scene[name]
    p = np.asarray(at, float) + origin
    asset.write_root_pose_to_sim(torch.tensor([[*p.tolist(), *xyzw]], dtype=torch.float32, device=asset.device))


def _angle(rot) -> float:
    """A body's turn about the sensor's axis (rad, from its start; about Q's z)."""
    v = rot @ Q[:, 0]
    return math.atan2(float(v @ Q[:, 1]), float(v @ Q[:, 0]))


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _geometry(sock_pos, sock_rot) -> tuple[float, float, float]:
    """The connector's front against the sensor's back end: how far out along the axis (m), off it (m), tilted (rad)."""
    front = np.asarray(sock_pos, float) + sock_rot @ np.array([0.0, 0.0, -SOCK_L])
    v = front - E
    s = float(v @ Q[:, 2])
    radial = float(np.linalg.norm(v - s * Q[:, 2]))
    _EP['rxy'] = (float(v @ Q[:, 0]), float(v @ Q[:, 1]))
    tilt = math.acos(max(-1.0, min(1.0, float((sock_rot @ np.array([0.0, 0.0, 1.0])) @ Q[:, 2]))))
    return s, radial, tilt


@on_step
def _follow(env) -> None:
    base = env.unwrapped
    origin = as_numpy(base.scene.env_origins[0])
    sp = as_numpy(base.scene["socket"].data.root_link_pose_w)[0]
    np_ = as_numpy(base.scene["sensor"].data.root_link_pose_w)[0]
    s_pos, s_rot = sp[:3] - origin, rotation(sp[3:7])
    n_rot = rotation(np_[3:7])
    s, radial, tilt = _geometry(s_pos, s_rot)
    a_sock, a_sens = _angle(s_rot), _angle(n_rot)
    d_sock = 0.0 if _EP["a_sock"] is None else _wrap(a_sock - _EP["a_sock"])
    d_sens = 0.0 if _EP["a_sens"] is None else _wrap(a_sens - _EP["a_sens"])
    _EP.update(a_sock=a_sock, a_sens=a_sens, s=s, radial=radial, tilt=tilt, seen=True)
    if _EP["on"]:
        _EP["turns"] -= (d_sock - d_sens) / (2 * math.pi)  # tightening: the ring turned clockwise seen from behind
        if _EP["turns"] < LOOSE:  # unscrewed: the thread lets go
            set_thread_on("m12", False)
            set_hinge_drive("sensor_turn", spring=0.0)
            _EP.update(on=False, turns=0.0, tight=False)
        elif not _EP["tight"]:
            # the ring drags the sensor round with it: a spring toward where the ring's turn would take it
            sens_deg = math.degrees(a_sens)
            lag = max(-LAG, min(LAG, _EP["target"] - _EP["sens_deg"] + math.degrees(d_sock)))
            _EP["target"] = sens_deg + lag
            _EP["sens_deg"] = sens_deg
            set_hinge_drive("sensor_turn", target_deg=_EP["target"], spring=DRAG_K)
            if _EP["turns"] >= TIGHT:
                _EP["tight"] = True
                set_hinge_drive("sensor_turn", target_deg=sens_deg, spring=0.0)
    elif s <= CATCH_S and s >= -0.002 and radial <= CATCH_R and tilt <= CATCH_TILT:
        set_thread_on("m12", True)
        _EP.update(on=True, turns=0.0, tight=False, target=math.degrees(a_sens), sens_deg=math.degrees(a_sens))
    lit = bool(_EP["on"] and _EP["tight"])
    if lit != _EP["light"]:
        _place(base, origin, "power_light", E + Q @ np.array([0.0, 0.0, LED_Z]) if lit else HIDDEN)
        _EP["light"] = lit
    _EP["tcps"] = _tcps(base, origin)
    if _DEBUG["on"]:
        _DEBUG["steps"] += 1
        if _DEBUG["steps"] % 10 == 0:
            tl, tr = _EP["tcps"]
            cp = as_numpy(base.scene["cap"].data.root_link_pose_w)[0][:3] - origin
            ring = s_pos + s_rot @ np.array([0.0, 0.0, RING_C])
            pin = E + PIN_Z * Q[:, 2]
            print(f"[m12] t={_DEBUG['steps'] / 20:5.1f}s on={_EP['on']} s={s * 1000:5.1f}mm r={radial * 1000:4.1f}mm "
                  f"rxy={round(_EP['rxy'][0] * 1000, 1)},{round(_EP['rxy'][1] * 1000, 1)} tilt={math.degrees(tilt):4.1f} turns={_EP['turns']:5.2f} tight={_EP['tight']} "
                  f"sock={math.degrees(a_sock):6.1f} sens={math.degrees(a_sens):6.1f} tgt={_EP['target']:6.1f} "
                  f"cap={np.round(cp * 1000).astype(int).tolist()} "
                  f"R-ring={np.round((tr - ring) * 1000).astype(int).tolist()} "
                  f"L-pin={np.round((tl - pin) * 1000).astype(int).tolist()} score={_points(_status_raw())}",
                  file=sys.stderr, flush=True)


# -- scoring --------------------------------------------------------------------------------------------------------

_LAST: dict = {}


def _status_raw() -> dict:
    return {"caught": bool(_EP["on"]), "tight": bool(_EP["on"] and _EP["tight"]),
            "cap_in": bool(_LAST.get("cap_in")), "clear": bool(_LAST.get("clear")), "home": _arms_home()}


def _status(objects) -> dict:
    cap = np.asarray(objects["cap"].pos, float)
    cap_in = (abs(cap[0] - TRAY_C[0]) < TRAY_IN / 2 and abs(cap[1] - TRAY_C[1]) < TRAY_IN / 2
              and cap[2] < PLATE_Z + TRAY_WALL + 0.02)
    sock = objects["socket"]
    ring = np.asarray(sock.pos, float) + rotation(sock.quat_xyzw) @ np.array([0.0, 0.0, RING_C])
    pin = E + PIN_Z * Q[:, 2]
    tcps = _EP.get("tcps")
    clear = tcps is not None and all(float(np.linalg.norm(t - q)) >= LET_GO for t in tcps for q in (ring, pin))
    _LAST.update(cap_in=cap_in, clear=clear)
    return _status_raw()


def _points(st) -> float:
    if st["tight"] and st["cap_in"] and st["clear"] and st["home"]:
        return 1.0
    if st["tight"]:
        return 0.5
    return 0.25 if st["caught"] else 0.0


def score(objects) -> float:
    """Judged at the end. 1.0: the red cap lies in the blue parts tray, the cable's connector is on the sensor's
    connector with its coupling ring screwed on until tight (turned one full turn against the sensor: the green light
    is on), both hands have let go (each gripper's fingertip centre at least 4 cm from the coupling ring and from the
    sensor's barrel where it is held) and both arms are back where they started (each within 5 cm); 0.5: the ring is
    screwed on tight; 0.25: the connector is pushed onto the sensor's connector and its thread has caught; else 0.0."""
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
    """Full score with the connector, the sensor and the cap still."""
    return (score(objects) >= 1.0 and objects["socket"].resting() and objects["sensor"].resting()
            and objects["cap"].resting())


# -- scripted demo ----------------------------------------------------------------------------------------------------
# The right hand, pointing along the sensor's axis, fingers closing across it, slides its open fingers round the red cap,
# takes it, pulls it off the sensor's connector and drops it into the parts tray. It comes down round the parked
# connector's coupling ring, takes it, pulls it off the parking receptacle, carries it across and pushes it onto the
# sensor's connector. The left hand, pointing in along the barrel from its sensing end, tilted a little down, slides
# its open fingers round the barrel below the bracket and holds it still (gripped this way, the ring's drag turns the
# wrist's strong roll axis; a grip from above lets the barrel twist and creep out of the fingers). The right turns the
# ring clockwise (seen from behind) in four strokes of 135 degrees, letting go and turning back open between strokes:
# it is tight after the third and the green light comes on. Both let go and go back to where they started.

R0 = np.column_stack([[1.0, 0.0, 0.0], np.cross(U, [1.0, 0.0, 0.0]), U])  # pointing along U, fingers closing along x
L_TILT = math.radians(15.0)  # the left's side grip: pointing -y (in along the barrel from its sensing end), tilted
_LZ = np.array([0.0, -math.cos(L_TILT), -math.sin(L_TILT)])  # this far down, fingers closing along x
PIN_GRIP = np.column_stack([[1.0, 0.0, 0.0], np.cross(_LZ, [1.0, 0.0, 0.0]), _LZ]) @ np.diag([-1.0, -1.0, 1.0])
SLIDE_IN = 0.045  # the left's fingers slide in this far along their pointing direction
PHI0, STROKE, STROKES = math.radians(-30.0), math.radians(135.0), 4
PRE_CAP = -(CAP_R + 0.007)
HOLD_CAP = -(CAP_R - 0.0025)  # a light squeeze: the light rubber cap must drop when let go
PRE_RING = -(RING_R + 0.007)
PRE_PIN = -(BARREL_R + 0.008)
R_AIM = np.array([0.0, 0.0, 0.0])  # aim corrections (measured)
L_AIM = np.array([0.0, 0.0, 0.0])
PUSH = 0.001  # the connector aimed this far onto the sensor
SLIP = 0.010  # pulled off its parking receptacle, the connector slides this far forward in the hand (measured)
PUSH_AIM = -0.0012 * Q[:, 0] + 0.0025 * Q[:, 1]  # the carried connector sags this far off the axis (measured)
TURN_S, RELEASE_S, BACK_S, REGRIP_S = 1.6, 0.4, 0.8, 0.4
LEAD_S, LEAD_STEPS = 3.0, 15
_IK_CACHE: dict = {}


def _side(phi: float) -> np.ndarray:
    return R0 @ _rz(phi)


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


def _right(t_held: float) -> tuple[list[Waypoint], dict]:
    ev = {}
    g = _side(PHI0)
    cap_c = E - CAP_L / 2 * U + R_AIM
    out = _lead_in("right", cap_c - 0.05 * U, g, "behind the red cap, pointing along the sensor", PRE_CAP)
    tray_drop = np.array([TRAY_C[0], TRAY_C[1], PLATE_Z + 0.050])
    out += [_w(1.2, cap_c, PRE_CAP, "slide the open fingers round the cap", g),
            _w(0.6, cap_c, HOLD_CAP, "take it", g),
            _w(1.0, cap_c - 0.04 * U, HOLD_CAP, "pull it off the sensor's connector", g),
            _w(1.6, tray_drop, HOLD_CAP, "carry it over the parts tray", g),
            _w(0.6, tray_drop, FINGER_OPEN, "drop it in the tray", g)]
    ring_p = E_PARK - RING_L / 2 * U + R_AIM
    ring_e = E - RING_L / 2 * U + R_AIM + PUSH_AIM
    up = np.array([0.0, 0.0, 0.035])
    out += [_w(1.6, ring_p + up, PRE_RING, "over the parked connector's coupling ring", g),
            _w(1.0, ring_p, PRE_RING, "down round the ring", g),
            _w(0.6, ring_p, FINGER_CLOSED, "take it", g),
            _w(1.0, ring_p - 0.030 * U, FINGER_CLOSED, "pull it off the parking receptacle", g),
            _w(0.6, ring_p - 0.030 * U + [0.0, 0.0, 0.015], FINGER_CLOSED, "lift it clear", g),
            _w(1.4, ring_e - (0.030 + SLIP) * U + [0.0, 0.0, 0.015], FINGER_CLOSED, "carry it to the sensor", g),
            _w(0.6, ring_e - (0.025 + SLIP) * U, FINGER_CLOSED, "line it up with the sensor's connector", g),
            _w(1.2, ring_e + (PUSH - SLIP) * U, FINGER_CLOSED, "push it onto the sensor's connector", g)]
    ev["seated"] = _dur(out)
    if t_held + 0.3 > _dur(out):
        out.append(_w(t_held + 0.3 - _dur(out), ring_e + (PUSH - SLIP) * U, FINGER_CLOSED,
                      "wait for the sensor to be held", g))
    at = ring_e + (PUSH - SLIP) * U
    for k in range(STROKES):
        for j in range(1, 5):
            out.append(_w(TURN_S / 4, at, FINGER_CLOSED, f"turn the coupling ring clockwise ({k + 1})",
                          _side(PHI0 + STROKE * j / 4)))
        if k < STROKES - 1:
            out.append(_w(RELEASE_S, at, PRE_RING, "let go", _side(PHI0 + STROKE)))
            for j in range(1, 5):
                out.append(_w(BACK_S / 4, at, PRE_RING, "turn back, open", _side(PHI0 + STROKE * (1 - j / 4))))
            out.append(_w(REGRIP_S, at, FINGER_CLOSED, "take the ring again", g))
    ev["tight"] = _dur(out)
    end = _side(PHI0 + STROKE)
    out += [_w(0.5, at, PRE_RING, "let go: it is tight", end),
            _w(0.8, at - 0.040 * U, PRE_RING, "draw back", end)]
    out += _go_home("right", at - 0.040 * U, end, PRE_RING)
    return out, ev


def _left(ev: dict | None, total: float) -> tuple[list[Waypoint], dict]:
    times = {}
    pin = E + PIN_Z * Q[:, 2] + L_AIM
    over = pin - SLIDE_IN * _LZ
    out = _lead_in("left", over, PIN_GRIP, "beside the sensor's barrel, below the bracket", PRE_PIN)
    if ev is not None and ev["seated"] - 1.6 > _dur(out):
        out.append(_w(ev["seated"] - 1.6 - _dur(out), over, PRE_PIN, "wait for the connector", PIN_GRIP))
    out += [_w(1.2, pin, PRE_PIN, "slide the open fingers round the barrel", PIN_GRIP),
            _w(0.6, pin, FINGER_CLOSED, "hold the sensor still", PIN_GRIP)]
    times["held"] = _dur(out)
    if ev is None:
        return out, times
    out.append(_w(max(ev["tight"] + 0.3 - _dur(out), 0.3), pin, FINGER_CLOSED,
                  "keep it still while the ring is turned", PIN_GRIP))
    out += [_w(0.5, pin, PRE_PIN, "let go", PIN_GRIP),
            _w(0.8, over, PRE_PIN, "draw back", PIN_GRIP)]
    out += _go_home("left", over, PIN_GRIP, PRE_PIN)
    if total > _dur(out):
        out.append(_w(total - _dur(out), out[-1].pos, PRE_PIN,
                      "hold still: the trial ends once the full score has held 1 s", out[-1].yaw))
    return out, times


def demo():
    _, ev0 = _right(0.0)
    _, t0 = _left(ev0, 0.0)
    right, ev = _right(t0["held"])
    _, t1 = _left(ev, 0.0)
    right, ev = _right(t1["held"])
    left, _ = _left(ev, 0.0)
    total = max(_dur(right), _dur(left)) + 3.0
    right.append(_w(total - _dur(right), right[-1].pos, PRE_RING,
                    "hold still: the trial ends once the full score has held 1 s", right[-1].yaw))
    left, _ = _left(ev, total)
    return {"left": left, "right": right}
