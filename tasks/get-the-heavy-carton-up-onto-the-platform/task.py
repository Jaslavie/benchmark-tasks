"""On the station table, straight ahead of the robot: a low grey platform, 2 cm high, its near edge running side to
side. In front of it, on the table, a heavy brown carton with a black handle bar along the top of its far edge (the
edge nearest the platform) and a dark T-shaped mark on its top. Where the carton starts changes between trials.
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import parts  # noqa: E402
import lots  # noqa: E402
from yam_arena.demo import FINGER_CLOSED, Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
EPS = 1e-6

PLATE_FRICTION = (0.30, 0.25)  # painted steel plate (static, dynamic)
BOARD_FRICTION = (0.45, 0.40)  # corrugated board
BROWN, BLACK, DARK, PLAT_GREY = (0.62, 0.45, 0.26), (0.06, 0.06, 0.07), (0.25, 0.17, 0.10), (0.58, 0.60, 0.62)

# The platform (fixed): world frame, x away from the arm bases, y to the robot's left, z up (m).
STEP_X = 0.38  # its near edge
PLAT = (0.25, 0.40, 0.020)
PLAT_TOP = TOP + PLAT[2]
PLATFORM = parts.Part("platform", pos=(STEP_X + PLAT[0] / 2, 0.0, TOP),
                      shapes=[parts.box(PLAT, at=(0.0, 0.0, PLAT[2] / 2), color=PLAT_GREY)], friction=PLATE_FRICTION)

# The carton (loose). Part frame: origin at the centre of its underside, +x toward the platform (its front), z up.
BODY = (0.16, 0.24, 0.06)
HANDLE_R, HANDLE_UP, HANDLE_X = 0.006, 0.016, BODY[0] / 2 - 0.012  # round bar, its axis 16 mm above the top, 12 mm in
POST_Y = 0.115
CARTON_MASS = 3.0
_ht = BODY[2] + HANDLE_UP
CARTON_SHAPES = [
    parts.box(BODY, at=(0.0, 0.0, BODY[2] / 2), color=BROWN),
    parts.cylinder(HANDLE_R, 2 * POST_Y + 0.010, at=(HANDLE_X, 0.0, _ht), color=BLACK, axis="y"),
    *[parts.box((0.010, 0.010, HANDLE_UP), at=(HANDLE_X, s * POST_Y, BODY[2] + HANDLE_UP / 2), color=BLACK)
      for s in (-1, 1)],
    parts.box((0.060, 0.012, 0.001), at=(-0.020, 0.0, BODY[2] + 0.0005), color=DARK),  # arrow shaft
    parts.box((0.012, 0.036, 0.001), at=(0.016, 0.0, BODY[2] + 0.0005), color=DARK),  # arrow head (a bar)
]
GAP = 0.040  # the carton's front face starts 4 cm short of the step
START = (STEP_X - GAP - BODY[0] / 2, 0.0, TOP)
CARTON = parts.Part("carton", pos=START, mass=CARTON_MASS, shapes=CARTON_SHAPES, friction=BOARD_FRICTION)

OBJECTS = [PLATFORM, CARTON]

# Lot A (the screenshot): the carton centred on the midline, its front 4 cm short of the step. Lot B: 5 cm to the
# robot's left and 3 cm further from the step.
LOTS = lots.declare(
    OBJECTS,
    lots.Lot("A"),
    lots.Lot("B", place={"carton": (START[0] - 0.03, 0.05, TOP)}),
)


def _rot(q) -> np.ndarray:
    return parts._quat_to_matrix(q)


_LOCAL_CORNERS = np.array([[sx * BODY[0] / 2, sy * BODY[1] / 2, z] for sx in (-1, 1) for sy in (-1, 1)
                           for z in (0.0, BODY[2])])


def corners(st) -> np.ndarray:
    """World corners of the carton's body (bottom four first in each pair)."""
    return np.asarray(st.pos, float) + _LOCAL_CORNERS @ _rot(st.quat_xyzw).T


def tilt_deg(st) -> float:
    return math.degrees(math.acos(float(np.clip(_rot(st.quat_xyzw)[2, 2], -1.0, 1.0))))


MAX_TILT = 20.0  # the carton's top tilted more than 20 degrees at any time: 0 (contents marked this way up)
FLAT_TOL = 0.001  # bottom within 1 mm of the surface it rests on
UPRIGHT = 2.0  # "flat": top within 2 degrees of level


def on_platform(st) -> bool:
    """Flat on the platform: all four bottom corners within 1 mm of its top and inside its outline."""
    c = corners(st)[0::2]  # bottom corners (z = 0 in the part frame)
    x0, x1, y0, y1 = STEP_X, STEP_X + PLAT[0], -PLAT[1] / 2, PLAT[1] / 2
    inside = np.all((c[:, 0] >= x0 - EPS) & (c[:, 0] <= x1 + EPS) & (c[:, 1] >= y0 - EPS) & (c[:, 1] <= y1 + EPS))
    return bool(inside and np.all(np.abs(c[:, 2] - PLAT_TOP) <= FLAT_TOL + EPS) and tilt_deg(st) <= UPRIGHT + EPS)


def level_overhanging(st) -> bool:
    """Tipped level onto the platform but not yet pushed fully on: all four bottom corners within 1 mm of its top
    height, top within 2 degrees of level, the underside centre past the step's edge, both back bottom corners
    short of it (overhanging the near edge), the front corners inside the far edge and all four inside sideways."""
    c = corners(st)[0::2]
    r = _rot(st.quat_xyzw)
    rel = (c - np.asarray(st.pos)) @ r[:, 0]
    front, rear = c[rel > 0], c[rel < 0]
    return bool(len(front) == 2 and len(rear) == 2
                and np.all(np.abs(c[:, 2] - PLAT_TOP) <= FLAT_TOL + EPS) and tilt_deg(st) <= UPRIGHT + EPS
                and float(st.pos[0]) > STEP_X + EPS and np.all(rear[:, 0] < STEP_X - EPS)
                and np.all(front[:, 0] <= STEP_X + PLAT[0] + EPS) and np.all(np.abs(c[:, 1]) <= PLAT[1] / 2 + EPS))


def front_on_step(st) -> bool:
    """Resting across the step: both front bottom corners past the step's edge (inside the outline sideways) and no
    lower than 1 mm under the platform's top, so its underside bears on the step; both rear bottom corners on the
    table (within 1 mm)."""
    c = corners(st)[0::2]
    r = _rot(st.quat_xyzw)
    front = c[(c - np.asarray(st.pos)) @ r[:, 0] > 0]
    rear = c[(c - np.asarray(st.pos)) @ r[:, 0] < 0]
    return bool(len(front) == 2 and len(rear) == 2
                and np.all(front[:, 2] >= PLAT_TOP - FLAT_TOL - EPS) and np.all(front[:, 0] >= STEP_X - EPS)
                and np.all(np.abs(front[:, 1]) <= PLAT[1] / 2 + EPS)
                and np.all(np.abs(rear[:, 2] - TOP) <= FLAT_TOL + EPS) and np.all(rear[:, 0] < STEP_X - EPS))


# -- trial record: tipped past 20 degrees at any time -------------------------------------------------------------
_H = {"tipped": False, "prev": None}


def _update(objects) -> None:
    st = objects["carton"]
    pos = np.asarray(st.pos, float)
    prev, start = _H["prev"], np.asarray(lots.current().pos("carton"), float)
    # A scene reset (offline): back within 0.5 mm of its start, still, after a jump of over 20 mm in one step.
    if prev is not None and np.linalg.norm(pos - prev) > 0.020 and np.linalg.norm(pos - start) < 0.0005 \
            and st.resting():
        _H["tipped"] = False
    _H["prev"] = pos
    if tilt_deg(st) > MAX_TILT + EPS:
        _H["tipped"] = True


# -- released: from the robot's finger and gripper collision hulls -------------------------------------------------
CLEAR = 0.005
_LINKS = ("gripper", "tip_left", "tip_right")
_GRIP = {"clear": True}


def _surface_local(step=0.006) -> np.ndarray:
    pts = []
    for s in CARTON_SHAPES[:2]:  # the body and the handle bar
        h = s.half
        grids = [np.linspace(-v, v, max(2, int(np.ceil(2 * v / step)) + 1)) for v in h]
        for axis in range(3):
            a, b = [k for k in range(3) if k != axis]
            ga, gb = np.meshgrid(grids[a], grids[b], indexing="ij")
            for sign in (-1.0, 1.0):
                p = np.zeros((ga.size, 3))
                p[:, a], p[:, b], p[:, axis] = ga.ravel(), gb.ravel(), sign * h[axis]
                pts.append(p + np.asarray(s.at))
    return np.concatenate(pts)


_SURFACE = _surface_local()


def _robot_hulls(sim):
    from yam_arena import analysis

    base = sim.env.unwrapped
    origin = parts._np(base.scene.env_origins[0])
    out = []
    for arm in ("left_robot", "right_robot"):
        asset = base.scene[arm]
        poses = parts._np(asset.data.body_link_pose_w)[0]
        link_poses = {}
        for k, n in enumerate(asset.body_names):
            if n in _LINKS:
                t = np.eye(4)
                t[:3, :3], t[:3, 3] = _rot(poses[k, 3:]), poses[k, :3] - origin
                link_poses[n] = t
        out += analysis.world_hulls(link_poses)
    return out


def _clearance(hulls, pts) -> float:
    lo, hi = pts.min(0) - 0.02, pts.max(0) + 0.02
    best = np.inf
    for _, h in hulls:
        if np.any(h.vertices.max(0) < lo) or np.any(h.vertices.min(0) > hi):
            continue
        d = (pts @ h.planes[:, :3].T - h.planes[:, 3]).max(axis=1)
        best = min(best, float(d.min()))
    return best


@parts.on_step
def _check_grip(sim):
    st = sim.states()["carton"]
    pts = np.asarray(st.pos, float) + _SURFACE @ _rot(st.quat_xyzw).T
    _GRIP["clear"] = _clearance(_robot_hulls(sim), pts) >= CLEAR - EPS


@parts.on_reset
def _new_episode(sim):
    _H.update(tipped=False, prev=None)


def _settled(objects) -> bool:
    return bool(not _H["tipped"] and objects["carton"].resting() and _GRIP["clear"])


def success(objects):
    """The carton flat on the platform, wholly inside its outline, let go and still; never tipped past 20 degrees."""
    _update(objects)
    return bool(_settled(objects) and on_platform(objects["carton"]))


def score(objects):
    """1.0 as success; 0.75 flat and level on the platform with its centre past the step's edge but its back
    overhanging the near edge; 0.5 resting across the step (front bottom corners past the step's edge and inside the
    platform's outline sideways, back on the table). Below 1.0, let go, still and never tipped past 20 degrees;
    otherwise 0."""
    if success(objects):
        return 1.0
    if not _settled(objects):
        return 0.0
    st = objects["carton"]
    if level_overhanging(st):
        return 0.75
    return 0.5 if front_on_step(st) else 0.0


def distance(objects):
    p = np.asarray(objects["carton"].pos, float)
    return float(np.linalg.norm(p[:2] - np.array([STEP_X + PLAT[0] / 2, 0.0])))


# -- scripted demo: both arms; fixed waypoints from the carton's place in the trial's lot --------------------------
TIP = 0.013
GRIP_Y = 0.095  # the two grips on the handle, either side of the carton's centre line (the arms' elbows need ~19 cm)
LIFT = 0.050  # commanded 50 mm: the loaded arms lag; traced, the front bottom edge rises 27 mm, over the 20 mm step
REACH = 0.075  # then carried 75 mm toward the platform: the front edge 35 mm past the step's edge
OPEN = -0.020
PUSH_LEAD = 0.040  # pushing, the TCP is sent 40 mm past where the carton's back should end: the loaded arms give way


def demo():
    """Both arms grip the handle, raise the carton's front edge, carry it forward over the step, set it down on the
    platform's top and let go; then push the back until the carton tips level onto the platform, and push it on."""
    cx, cy, _ = lots.current().pos("carton")
    hx, hz = cx + HANDLE_X, TOP + _ht
    tcp = hz - HANDLE_R - 0.002 + TIP  # finger tips 2 mm below the bar's underside
    rear = cx - BODY[0] / 2
    tip_rear = STEP_X + 0.015 - BODY[0] / 2  # its back when its centre is 15 mm past the step's edge: tipped level
    end_rear = STEP_X + 0.025  # pushed on until its back is 25 mm past the step's edge
    plan = {}
    for arm, sy, yaw in (("left", 1.0, -math.pi / 2), ("right", -1.0, math.pi / 2)):
        y = cy + sy * GRIP_Y
        set_z = tcp + PLAT[2] + 0.004  # front edge set 4 mm above the platform top, then let go
        rear_now = rear + REACH
        plan[arm] = [
            Waypoint(1.5, (hx, y, tcp + 0.06), OPEN, "over the handle", yaw),
            Waypoint(1.2, (hx, y, tcp), OPEN, "round the handle", yaw),
            Waypoint(0.8, (hx, y, tcp), FINGER_CLOSED, "grip", yaw),
            Waypoint(2.0, (hx, y, tcp + LIFT), FINGER_CLOSED, "raise the front edge", yaw),
            Waypoint(1.0, (hx, y, tcp + LIFT), FINGER_CLOSED, "hold it up", yaw),
            Waypoint(2.5, (hx + REACH, y, tcp + LIFT), FINGER_CLOSED, "carry the front over the step", yaw),
            Waypoint(1.5, (hx + REACH, y, set_z), FINGER_CLOSED, "set the front edge on the platform", yaw),
            Waypoint(0.8, (hx + REACH, y, set_z), OPEN, "let go", yaw),
            Waypoint(1.2, (hx + REACH - 0.01, y, TOP + 0.13), OPEN, "straight up, clear of the handle", yaw),
            Waypoint(2.0, (rear_now - 0.030, y, TOP + 0.13), FINGER_CLOSED, "back over the carton", 0.0),
            Waypoint(1.2, (rear_now - 0.030, y, TOP + 0.045), FINGER_CLOSED, "down behind it", 0.0),
            Waypoint(1.0, (rear_now - 0.012, y, TOP + 0.030), FINGER_CLOSED, "fingers against its back", 0.0),
            Waypoint(3.0, (tip_rear + PUSH_LEAD, y, TOP + 0.030), FINGER_CLOSED,
                     "push until it tips level onto the platform", 0.0),
            Waypoint(1.0, (tip_rear - 0.015, y, TOP + 0.050), FINGER_CLOSED, "back and up a little", 0.0),
            Waypoint(2.5, (end_rear + PUSH_LEAD / 2, y, TOP + 0.050), FINGER_CLOSED, "push it on", 0.0),
            Waypoint(1.0, (end_rear - 0.030, y, TOP + 0.100), OPEN, "away", 0.0),
            Waypoint(1.5, (end_rear - 0.080, y + sy * 0.05, TOP + 0.160), OPEN, "clear", 0.0),
        ]
    return plan
