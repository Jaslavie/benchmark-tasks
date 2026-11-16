"""On the station table, in front of the left arm: a white store shelf unit with a lower shelf, a low orange fence
along the lower shelf's front edge, side walls, a back wall and a grey upper shelf above whose front edge is set back
from the fence. A tall blue square canister with a flat black carry handle along the front edge of its top stands
upright on the table in front of the shelf. Where the canister starts changes between trials.
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import parts  # noqa: E402  (before lots.py, so its reset hooks run first)
import lots  # noqa: E402
import tilt  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
EPS = 1e-6

WHITE, ORANGE, GREY, BLUE, BLACK = (0.92, 0.92, 0.90), (0.95, 0.45, 0.08), (0.55, 0.57, 0.60), (0.15, 0.32, 0.75), \
    (0.08, 0.08, 0.09)

# ---- shelf unit (fixed). World frame: x away from the arm bases, y to the robot's left, z up (m). -------------
BY = 0.22  # the unit's centre line
DECK_T = 0.010
DZ = TOP + DECK_T  # lower shelf's top
FX0 = 0.340  # fence's front face, the lower shelf's front edge
FENCE_T, FENCE_H = 0.006, 0.035
FI = FX0 + FENCE_T  # fence's inner face
BACK = FI + 0.080  # back wall's front face
ROOM = 0.170  # lower shelf top to the upper shelf's underside
UPPER_T = 0.015
UX = FI + 0.040  # upper shelf's front edge
HALF_W = 0.100  # inner half width between the side walls
WALL_T = 0.010
_x1 = BACK + WALL_T


def _fixed(name, lo, hi, color):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return parts.Part(name, pos=tuple(float(v) for v in (lo + hi) / 2), shapes=[
        parts.box(tuple(float(v) for v in hi - lo), color=color)])


SHELF = [
    _fixed("lower_shelf", (FX0, BY - HALF_W - WALL_T, TOP), (_x1, BY + HALF_W + WALL_T, DZ), WHITE),
    _fixed("fence", (FX0, BY - HALF_W, DZ), (FI, BY + HALF_W, DZ + FENCE_H), ORANGE),
    _fixed("back_wall", (BACK, BY - HALF_W, DZ), (_x1, BY + HALF_W, DZ + ROOM), WHITE),
    _fixed("upper_shelf", (UX, BY - HALF_W - WALL_T, DZ + ROOM), (_x1, BY + HALF_W + WALL_T, DZ + ROOM + UPPER_T),
           GREY),
    _fixed("left_wall", (FX0, BY + HALF_W, DZ), (_x1, BY + HALF_W + WALL_T, DZ + ROOM), WHITE),
    _fixed("right_wall", (FX0, BY - HALF_W - WALL_T, DZ), (_x1, BY - HALF_W, DZ + ROOM), WHITE),
]

# ---- canister (loose). Part frame: origin at the centre of its base, x toward its tab side's far face. ---------
SIDE, H = 0.050, 0.150  # square body, height
TAB = (0.012, 0.040, 0.018)  # carry handle (a flat tab): thickness (x), width (y), height (z)
TAB_X = -SIDE / 2  # the tab's centre lies on the front face's plane (half of it overhangs)
MASS = 0.30
START_A = (0.24, 0.16)
START_B = (0.20, 0.05)

CANISTER = parts.Part("canister", pos=(*START_A, TOP), mass=MASS, friction=(1.0, 0.8), shapes=[
    parts.box((SIDE, SIDE, H), at=(0.0, 0.0, H / 2), color=BLUE),
    parts.box(TAB, at=(TAB_X, 0.0, H + TAB[2] / 2), color=BLACK),
])

OBJECTS = [*SHELF, CANISTER]

# Lot A (the screenshot): the canister in front of the shelf's left half. Lot B: nearer the robot and to the
# right of the midline. Lot A has a zero move so both lots get the same settle and first frame.
LOTS = lots.declare(
    OBJECTS,
    lots.Lot("A", place={"canister": (*START_A, TOP)}),
    lots.Lot("B", place={"canister": (*START_B, TOP)}),
)

# ---- scoring -----------------------------------------------------------------------------------------------------
UPRIGHT = 5.0  # degrees
LEAN_MAX = 30.0  # degrees: the 0.5 tier (standing on its base behind the fence, leaning back on it)
ON_DECK = 0.002  # the base's lowest corner within 2 mm of the lower shelf's top
CLEAR = 0.005  # released: no part of either gripper within 5 mm of the canister
SUCCESS_HOLD_S = 3.0  # long enough that the demo's gripper is seen well clear before the trial ends


def _rot(q) -> np.ndarray:
    return parts._quat_to_matrix(q)


def tilt_deg(state) -> float:
    return float(math.degrees(math.acos(np.clip(_rot(state.quat_xyzw)[2, 2], -1.0, 1.0))))


def base_corners(state) -> np.ndarray:
    r = _rot(state.quat_xyzw)
    h = SIDE / 2
    local = np.array([[sx * h, sy * h, 0.0] for sx in (-1, 1) for sy in (-1, 1)])
    return np.asarray(state.pos, float) + local @ r.T


def _on_base_behind_fence(state) -> bool:
    """Every base corner behind the fence's inner face and between the side walls, the lowest on the shelf, and
    no corner below it (not lying on a side)."""
    c = base_corners(state)
    return bool(
        c[:, 0].min() >= FI - EPS
        and c[:, 0].max() <= BACK + EPS
        and np.abs(c[:, 1] - BY).max() <= HALF_W + EPS
        and abs(c[:, 2].min() - DZ) <= ON_DECK + EPS
    )


_grip = {"clear": True}  # set every step by the on_step callback; offline there is no robot, so it stays True


@parts.on_step
def _measure_grip(sim) -> None:
    from yam_arena import analysis

    base = sim.env.unwrapped
    origin = parts._np(base.scene.env_origins[0])
    st = sim.states()["canister"]
    boxes = CANISTER.world_boxes({"base": (np.asarray(st.pos), _rot(st.quat_xyzw))})
    hulls = []
    for arm in ("left_robot", "right_robot"):
        asset = base.scene[arm]
        poses = parts._np(asset.data.body_link_pose_w)[0]
        for i, n in enumerate(asset.body_names):
            if n in ("tip_left", "tip_right", "gripper"):
                m = np.eye(4)
                m[:3, :3] = _rot(poses[i, 3:])
                m[:3, 3] = poses[i, :3] - origin
                hulls += analysis.world_hulls({n: m})
    _grip["clear"] = all(analysis.hulls_to_box(hulls, c, r, s)[0] > CLEAR for _, c, r, s in boxes)


def _settled(state) -> bool:
    return bool(state.resting() and _grip["clear"])


def stood(objects) -> bool:
    c = objects["canister"]
    return bool(_on_base_behind_fence(c) and tilt_deg(c) <= UPRIGHT + EPS and _settled(c))


def leans_back(state) -> bool:
    """The canister's axis leans toward the robot (back onto the fence): its top's horizontal offset points within
    45 degrees of straight back toward the arms."""
    ax = _rot(state.quat_xyzw)[:, 2]
    return bool(-ax[0] >= abs(ax[1]) - EPS and ax[0] < 0.0)


def leaning_in(objects) -> bool:
    c = objects["canister"]
    return bool(_on_base_behind_fence(c) and UPRIGHT + EPS < tilt_deg(c) <= LEAN_MAX + EPS and leans_back(c)
                and _settled(c))


def success(objects):
    """Standing upright on the lower shelf with its whole base behind the fence, let go and still."""
    return stood(objects)


def distance(objects):
    p = np.asarray(objects["canister"].pos, float)
    return float(np.linalg.norm(p - np.array((FI + 0.030, BY, DZ))))


def score(objects):
    """1.0 on success; 0.5 standing on its base behind the fence but leaning back; otherwise 0."""
    if stood(objects):
        return 1.0
    return 0.5 if leaning_in(objects) else 0.0


# ---- demo: left arm only; fixed waypoints from the lot's start spot (what the cameras show) --------------------
TIP = 0.013  # pad tips below the TCP
TIP_UP = 0.006  # pad tips this far above the tab's underside
GRIP_DZ = H + TIP_UP + TIP  # TCP above the canister's base, upright
HOLD = -0.003  # pads 6 mm apart: 3 mm past each face of the 12 mm tab
WIDE = -0.020  # pads 40 mm apart
OPEN = -0.010  # pads 20 mm apart: let go under the upper shelf without the far finger reaching its edge
LEAN = 15.0  # degrees, base leading toward the shelf: the lean the canister takes
LEAN_CMD = 18.0  # the grip lean asked for: under the canister's weight the wrist gives about 3 degrees
# The loaded arm lags its target (trace, 0.3 kg held at the tab): about 15 mm short in x and 12 mm low while it
# carries the canister, less once the base bears on the shelf. Each waypoint is aimed past its goal by this much.
CARRY_COMP = (0.015, 0.012)
YAW = -math.pi / 2  # pads close along x, across the tab's thickness, so the lean cannot turn the canister in them
PASS_Z = FENCE_H + 0.012  # the base's near bottom edge this far above the shelf top while it passes the fence
SET_X = FI + FENCE_H * math.tan(math.radians(LEAN)) + 0.005  # near bottom edge set down here, clear of the fence
SET_Z = 0.001


def _tcp(px, pz, lean_deg, y):
    """TCP for the canister leaning ``lean_deg`` (top toward the robot) with its near bottom edge at (px, pz)."""
    t = math.radians(lean_deg)
    dx, dz = TAB_X + SIDE / 2, GRIP_DZ  # TCP from the near bottom edge, canister frame
    return (px + dx * math.cos(t) - dz * math.sin(t), y, DZ + pz + dx * math.sin(t) + dz * math.cos(t))


def demo():
    """Left arm: grip the tab across its thickness, lift, lean the canister base-first, pass the base over the fence
    under the upper shelf, set the near bottom edge down behind the fence, stand it up about that edge while still
    holding the tab, let go and rise."""
    sx, sy, _ = lots.current().pos("canister")
    gx = sx + TAB_X
    W = tilt.Waypoint

    def lean(d):
        return dict(yaw=YAW, tilt=d, heading=0.0)

    up = dict(yaw=YAW)
    side = dict(yaw=YAW, tilt=15.0, heading=-math.pi / 2)  # leaned sideways: reaches higher here than straight down
    lift_z = TOP + 0.025 + GRIP_DZ  # base 25 mm up; it rises over the fence while leaning
    def aim(p, share):
        return (p[0] + share * CARRY_COMP[0], p[1], p[2] + share * CARRY_COMP[1])

    approach = aim(_tcp(FX0 - 0.030, PASS_Z, LEAN, BY), 1.0)
    over = aim(_tcp(SET_X, PASS_Z, LEAN, BY), 1.0)
    down = aim(_tcp(SET_X, SET_Z, LEAN, BY), 0.6)
    arc = [aim(_tcp(SET_X, SET_Z, a, BY), k) for a, k in ((10.0, 0.4), (5.0, 0.2), (0.0, 0.0))]
    w = [
        Waypoint(1.5, (gx, sy, TOP + GRIP_DZ + 0.040), WIDE, "over the tab", **up),
        Waypoint(1.2, (gx, sy, TOP + GRIP_DZ), WIDE, "down around the tab", **up),
        Waypoint(0.8, (gx, sy, TOP + GRIP_DZ), HOLD, "grip the tab", **up),
        Waypoint(1.2, (gx, sy, lift_z + CARRY_COMP[1]), HOLD, "lift", **up),
        W(2.0, approach, HOLD, "lean, base leading, in front of the fence", **lean(LEAN_CMD)),
        W(1.5, over, HOLD, "base over the fence", **lean(LEAN_CMD)),
        W(1.2, down, HOLD, "near bottom edge down behind the fence", **lean(LEAN_CMD)),
        W(1.0, arc[0], HOLD, "stand it up about that edge", **lean(12.0)),
        W(1.0, arc[1], HOLD, "stand it up", **lean(6.0)),
        Waypoint(1.0, arc[2], HOLD, "upright", **up),
        Waypoint(0.6, arc[2], OPEN, "let go", **up),
        W(1.0, (arc[2][0], BY, arc[2][2] + 0.030), OPEN, "up off the tab", **side),
        W(1.0, (arc[2][0] - 0.080, BY, arc[2][2] + 0.030), OPEN, "back away", **side),
        W(1.5, (arc[2][0] - 0.160, BY - 0.040, arc[2][2] + 0.060), OPEN, "well clear of the canister", **side),
        W(2.0, (arc[2][0] - 0.160, BY - 0.040, arc[2][2] + 0.060), OPEN, "wait", **side),
    ]
    return {"left": w}
