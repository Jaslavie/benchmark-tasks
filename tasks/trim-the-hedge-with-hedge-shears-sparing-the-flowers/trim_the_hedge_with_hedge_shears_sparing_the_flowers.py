"""On the station table, a cedar planter trough holds a small clipped box hedge. Three overgrown green shoots stick up
out of the front of the hedge; behind them, in the back row, grow two flowers, a pink one and an orange one. On a
rubber mat in front of the planter lie a pair of hedge shears, closed: two long steel blades on a pivot bolt, each
blade forged in one piece with a wooden handle, the two handles splaying back toward the arms.

Task: trim the three shoots off just above the hedge with the shears, without cutting the flowers, then leave the
shears lying closed on the mat and take both arms home. The pivot turns freely, so the blades open and close only when
the two handles are moved against each other: one hand on one handle cannot snip. A real trial judges severing by
observation; here a stem counts as severed when the blades close on it as the scoring describes.

Run it:  inspect-robots run --embodiment yam_arena -E task=trim_the_hedge_with_hedge_shears_sparing_the_flowers.py \\
           --instruction "..." --scorer task_score ...
"""

from __future__ import annotations

import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import bodykit  # noqa: E402,F401
from bodykit import (Body, BoxShape, CylShape, SphereShape, allow_matrix_grips, as_numpy, glue, hinge,  # noqa: E402
                     on_reset, on_step, plate, rotation, set_glue_on)
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, TCP_OFFSET  # noqa: E402,F401
from yam_arena.demo import Waypoint, grip_rotation  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

allow_matrix_grips()

SCENE = "station"
TOP = TABLE_TOP[SCENE]

# -- the mat --------------------------------------------------------------------------------------------------------
MX0, MX1, MY0, MY1, MAT_T = 0.110, 0.440, -0.160, 0.160, 0.003
MAT = TOP + MAT_T
# -- the shears: each half's frame has its origin on the pivot axis, between the two blades; closed, both blades
# point along +x. The left hand's half carries the upper blade and the handle on the left (+y), the right's the lower
# blade and the handle on the right. Opening turns the left half clockwise and the right half anticlockwise. ---------
BLADE_T = 0.004
BLADE_SEGS = ((0.010, 0.050, 0.034), (0.050, 0.090, 0.030), (0.090, 0.130, 0.025), (0.130, 0.163, 0.019))
BLADE_TIP = 0.170
BETA = math.radians(28.0)  # each handle's angle off the blade line (the closed handles splay 56 degrees)
H_IN, H_OUT, H_W, H_H = 0.088, 0.250, 0.019, 0.024  # wooden handle: from/to (m from the pivot), width, height
H_Z = 0.008  # the handles' middle this high above the pivot plane (their undersides level with the lower blade's)
GRIP_R = 0.190  # the hands hold the handles this far from the pivot
COM_R = 0.160  # each half's centre of mass, on its handle
HALF_MASS = 0.22
REST = np.array([0.360, 0.000, MAT + BLADE_T])  # the pivot plane on the mat (the lower blade lying on it)
# -- the planter, the hedge, the shoots and the flowers ------------------------------------------------------------
PX0, PX1, PY0, PY1, WALL, P_H = 0.555, 0.705, -0.175, 0.175, 0.012, 0.050
SOIL = TOP + 0.042
HX0, HX1, HY0, HY1 = 0.570, 0.690, -0.158, 0.158
HEDGE_TOP = TOP + 0.095  # the clipped top (its leafy bumps reach 4 mm higher)
SHOOT_X, SHOOT_Y = 0.585, (0.075, 0.000, -0.075)
CUT_Z = TOP + 0.115  # where each shoot comes apart once it is cut (2 cm above the hedge)
SHOOT_TIP = TOP + 0.215
FLOWER_X, FLOWER_Y = 0.665, (0.0375, -0.0375)
FLOWER_CUT_Z = TOP + 0.122
FLOWER_HEAD = TOP + 0.175
SHOOTS, FLOWERS = len(SHOOT_Y), len(FLOWER_Y)
PIN = (0.45, 0.0, 0.30)  # a hidden fixed body under the table that the uncut tops are held to
# -- the cutting rule ------------------------------------------------------------------------------------------------
OPEN_DEG = 20.0  # blades at least this far open around a stem ...
SHUT_DEG = 6.0  # ... then closed to at most this: the stem is cut
R_MIN, R_MAX = 0.025, 0.150  # the cutting part of the blades (m from the pivot)
Z_TOL = 0.018  # the blades within this of a shoot's cut height
ANG_MARGIN = 3.0
LEAVE_DEG = 30.0

STEEL, EDGE, DARK_STEEL, BRASS = (0.74, 0.76, 0.79), (0.93, 0.94, 0.96), (0.24, 0.25, 0.27), (0.78, 0.62, 0.25)
ASH, ASH_DARK, RUBBER = (0.80, 0.62, 0.40), (0.58, 0.41, 0.24), (0.07, 0.07, 0.07)
CEDAR, CEDAR_DARK, SOIL_C = (0.64, 0.42, 0.26), (0.47, 0.29, 0.16), (0.27, 0.18, 0.11)
BOX_G1, BOX_G2, BOX_G3 = (0.12, 0.34, 0.11), (0.17, 0.43, 0.14), (0.10, 0.28, 0.09)
SHOOT_G, LEAF_G, SHOOT_LEAF = (0.58, 0.84, 0.30), (0.40, 0.66, 0.22), (0.50, 0.80, 0.26)
STEM_G, PINK, ORANGE, YELLOW = (0.22, 0.48, 0.17), (0.94, 0.42, 0.66), (0.98, 0.52, 0.14), (0.98, 0.85, 0.20)
MAT_C, MAT_RIB = (0.20, 0.23, 0.22), (0.16, 0.19, 0.18)

HOME_WITHIN = 0.05
SUCCESS_HOLD_S = 1.0


def _yawq(yaw: float, roll: float = 0.0) -> tuple[float, float, float, float]:
    """(x, y, z, w) of a turn ``yaw`` about z after a turn ``roll`` about x."""
    cy, sy, cr, sr = math.cos(yaw / 2), math.sin(yaw / 2), math.cos(roll / 2), math.sin(roll / 2)
    return (cy * sr, sy * sr, sy * cr, cy * cr)


def _mat() -> Body:
    shapes = [BoxShape((0.0, 0.0, MAT_T / 2), (MX1 - MX0, MY1 - MY0, MAT_T), MAT_C)]
    for k in range(7):  # moulded ribs across the rubber
        shapes.append(BoxShape((-0.135 + 0.045 * k, 0.0, MAT_T + 0.0002), (0.006, MY1 - MY0 - 0.02, 0.0004), MAT_RIB,
                               collide=False))
    return Body.build("mat", pos=((MX0 + MX1) / 2, (MY0 + MY1) / 2, TOP), mass=1.0, kinematic=True, color=MAT_C,
                      shapes=shapes, friction=(0.9, 0.8))


def _planter() -> Body:
    """A cedar trough, soil, and a clipped box hedge (a block with leafy bumps over its top and front)."""
    cx, cy = (PX0 + PX1) / 2, (PY0 + PY1) / 2
    lx, ly = PX1 - PX0, PY1 - PY0
    shapes = [BoxShape((0.0, 0.0, 0.004), (lx, ly, 0.008), CEDAR_DARK)]
    for s in (-1.0, 1.0):
        shapes.append(BoxShape((s * (lx - WALL) / 2, 0.0, P_H / 2), (WALL, ly, P_H), CEDAR))
        shapes.append(BoxShape((0.0, s * (ly - WALL) / 2, P_H / 2), (lx - 2 * WALL, WALL, P_H), CEDAR))
        for z in (0.017, 0.034):  # plank seams
            shapes.append(BoxShape((s * (lx / 2 + 0.0003), 0.0, z), (0.0008, ly - 0.004, 0.0012), CEDAR_DARK,
                                   collide=False))
            shapes.append(BoxShape((0.0, s * (ly / 2 + 0.0003), z), (lx - 0.004, 0.0008, 0.0012), CEDAR_DARK,
                                   collide=False))
        shapes.append(BoxShape((s * (lx - WALL) / 2, 0.0, P_H + 0.002), (WALL + 0.004, ly + 0.004, 0.004), CEDAR_DARK))
        shapes.append(BoxShape((0.0, s * (ly - WALL) / 2, P_H + 0.002), (lx, WALL + 0.004, 0.004), CEDAR_DARK))
    shapes.append(BoxShape((0.0, 0.0, (SOIL - TOP) / 2), (lx - 2 * WALL, ly - 2 * WALL, SOIL - TOP), SOIL_C))
    hx, hy = (HX0 + HX1) / 2 - cx, (HY0 + HY1) / 2 - cy
    z0, z1 = SOIL - TOP - 0.004, HEDGE_TOP - TOP - 0.006
    shapes.append(BoxShape((hx, hy, (z0 + z1) / 2), (HX1 - HX0, HY1 - HY0, z1 - z0), BOX_G1))
    rng = np.random.default_rng(3)
    greens = (BOX_G1, BOX_G2, BOX_G3, (0.15, 0.39, 0.12))
    for _ in range(120):  # leafy bumps of every size over the top ...
        r = float(rng.uniform(0.010, 0.019))
        x, y = float(rng.uniform(HX0 + 0.006, HX1 - 0.006)) - cx, float(rng.uniform(HY0 + 0.006, HY1 - 0.006)) - cy
        z = HEDGE_TOP - TOP - r + float(rng.uniform(-0.002, 0.005))
        shapes.append(SphereShape((x, y, z), r, greens[int(rng.integers(4))], collide=False))
    for _ in range(90):  # ... and down its sides
        r = float(rng.uniform(0.009, 0.015))
        side = int(rng.integers(4))
        z = float(rng.uniform(SOIL - TOP + 0.010, HEDGE_TOP - TOP - 0.012))
        if side < 2:
            x, y = (HX0, HX1)[side] - cx, float(rng.uniform(HY0, HY1)) - cy
        else:
            x, y = float(rng.uniform(HX0, HX1)) - cx, (HY0, HY1)[side - 2] - cy
        shapes.append(SphereShape((x, y, z), r, greens[int(rng.integers(4))], collide=False))
    return Body.build("planter", pos=(cx, cy, TOP), mass=5.0, kinematic=True, color=CEDAR, shapes=shapes,
                      friction=(0.8, 0.7))


def _pin() -> Body:
    return Body.build("pin", pos=PIN, mass=1.0, kinematic=True,
                      shapes=[BoxShape((0.0, 0.0, 0.0), (0.01, 0.01, 0.01), None, collide=False)])


def _half(name: str, s: float) -> Body:
    """One half of the shears: a tapered steel blade, its tang, a steel ferrule and a wooden handle. ``s`` = +1: the
    left hand's half (upper blade, handle to the left); -1: the right's (lower blade, handle to the right)."""
    zc = s * BLADE_T / 2  # the blade's layer
    phi = math.pi - s * BETA  # the handle's direction
    u = np.array([math.cos(phi), math.sin(phi), 0.0])
    shapes = []
    for x0, x1, w in BLADE_SEGS:
        shapes.append(BoxShape(((x0 + x1) / 2, 0.0, zc), (x1 - x0, w, BLADE_T), STEEL))
        # the ground edge, on the side that meets the other blade
        shapes.append(BoxShape(((x0 + x1) / 2, s * (w / 2 - 0.0035), zc + BLADE_T / 2 + 0.0002),
                               (x1 - x0, 0.006, 0.0004), EDGE, collide=False))
    shapes.append(CylShape((BLADE_SEGS[-1][1], 0.0, zc), BLADE_SEGS[-1][2] / 2, BLADE_T, STEEL))  # the rounded tip
    shapes.append(CylShape((0.0, 0.0, zc), 0.021, BLADE_T, STEEL))  # the heel round the pivot
    if s > 0:  # the pivot bolt's head and its brass nut sit on top
        shapes.append(CylShape((0.0, 0.0, BLADE_T + 0.0015), 0.010, 0.003, DARK_STEEL))
        shapes.append(CylShape((0.0, 0.0, BLADE_T + 0.0040), 0.006, 0.002, BRASS, collide=False))
    tang0, tang1 = 0.012, 0.072
    shapes.append(BoxShape(tuple((u * (tang0 + tang1) / 2 + [0.0, 0.0, zc]).tolist()), (tang1 - tang0, 0.014, BLADE_T),
                           DARK_STEEL, yaw=phi))
    # the rubber bumper on the tang's inner side: the two bumpers meet when the blades are closed
    inner = np.array([-math.sin(phi), math.cos(phi), 0.0]) * (-s)
    bp = u * 0.036 + inner * 0.010 + [0.0, 0.0, zc + 0.002]
    shapes.append(BoxShape(tuple(bp.tolist()), (0.012, 0.012, 0.010), RUBBER, yaw=phi, collide=False))
    shapes.append(BoxShape(tuple((u * 0.079 + [0.0, 0.0, H_Z]).tolist()), (0.022, 0.022, H_H), DARK_STEEL, yaw=phi))
    mid = (H_IN + H_OUT) / 2
    shapes.append(BoxShape(tuple((u * mid + [0.0, 0.0, H_Z]).tolist()), (H_OUT - H_IN, H_W, H_H), ASH, yaw=phi))
    shapes.append(BoxShape(tuple((u * mid + [0.0, 0.0, H_Z]).tolist()), (H_OUT - H_IN - 0.004, 0.017, 0.017), ASH,
                           collide=False, quat_xyzw=_yawq(phi, math.pi / 4)))  # bevels: an octagonal handle
    for r in (0.120, 0.215):  # darker grain bands
        shapes.append(BoxShape(tuple((u * r + [0.0, 0.0, H_Z]).tolist()), (0.004, H_W + 0.0006, H_H + 0.0006),
                               ASH_DARK, yaw=phi, collide=False))
    shapes.append(BoxShape(tuple((u * (H_OUT + 0.003) + [0.0, 0.0, H_Z]).tolist()), (0.006, 0.021, H_H), ASH_DARK,
                           yaw=phi))
    com = u * COM_R + [0.0, 0.0, 0.005]
    return Body.build(name, pos=tuple(REST.tolist()), mass=HALF_MASS, shapes=shapes, friction=(1.2, 1.1), color=STEEL,
                      com=tuple(com.tolist()), damping=(0.0, 0.1))


def _stubs() -> Body:
    """What stays rooted: the lower stems of the shoots and the flowers (drawn only; nothing collides with them)."""
    cx = (PX0 + PX1) / 2
    shapes = []
    for y in SHOOT_Y:
        z0, z1 = HEDGE_TOP - 0.025, CUT_Z
        shapes.append(BoxShape((SHOOT_X - cx, y, (z0 + z1) / 2 - TOP), (0.007, 0.007, z1 - z0), SHOOT_G, collide=False))
    for y in FLOWER_Y:
        z0, z1 = HEDGE_TOP - 0.025, FLOWER_CUT_Z
        shapes.append(BoxShape((FLOWER_X - cx, y, (z0 + z1) / 2 - TOP), (0.004, 0.004, z1 - z0), STEM_G,
                               collide=False))
        p = np.array([FLOWER_X - cx, y, HEDGE_TOP + 0.010 - TOP])
        for side in (-1.0, 1.0):
            d = np.array([0.0, side * 0.7, 0.7])
            shapes.append(plate(p, p + 0.022 * d, 0.010, 0.0012, (1.0, 0.0, 0.0), LEAF_G, collide=False))
    return Body.build("stubs", pos=(cx, 0.0, TOP), mass=0.1, kinematic=True, color=SHOOT_G, shapes=shapes)


def _shoot(k: int) -> Body:
    """An overgrown shoot's top, above its cut height: a pale green stem with young leaves."""
    h = SHOOT_TIP - CUT_Z
    shapes = [BoxShape((0.0, 0.0, h / 2), (0.007, 0.007, h), SHOOT_G)]
    for i, (z, side) in enumerate(((0.026, 1.0), (0.044, -1.0), (0.060, 1.0), (0.074, -1.0), (0.088, 1.0))):
        yaw = 0.6 * k + 1.4 * i
        d = np.array([side * math.cos(yaw) * 0.75, side * math.sin(yaw) * 0.75, 0.66])
        p = np.array([0.0, 0.0, z])
        n = np.cross(d, [0.0, 0.0, 1.0]) if abs(d[2]) < 0.99 else np.array([1.0, 0.0, 0.0])
        n = np.cross(n, d)
        shapes.append(plate(p, p + 0.027 * d, 0.015, 0.0014, n, SHOOT_LEAF, collide=False))
    shapes.append(SphereShape((0.0, 0.0, h), 0.005, SHOOT_LEAF, collide=False))
    return Body.build(f"shoot{k}", pos=(SHOOT_X, SHOOT_Y[k], CUT_Z), mass=0.008, shapes=shapes, color=SHOOT_G,
                      friction=(0.8, 0.7), ignore=("shears_l", "shears_r"), damping=(0.2, 0.5))


def _flower(k: int) -> Body:
    """A flower's top, above where it would come apart if cut: its stem and the bloom."""
    h = FLOWER_HEAD - FLOWER_CUT_Z
    petal = (PINK, ORANGE)[k]
    shapes = [BoxShape((0.0, 0.0, h / 2), (0.004, 0.004, h), STEM_G),
              CylShape((0.0, 0.0, h + 0.003), 0.019, 0.005, petal),
              CylShape((0.0, 0.0, h + 0.0065), 0.007, 0.003, YELLOW, collide=False)]
    for i in range(6):  # petal tips round the rim
        a = i * math.pi / 3 + 0.3 * k
        shapes.append(SphereShape((0.019 * math.cos(a), 0.019 * math.sin(a), h + 0.003), 0.0065, petal, collide=False))
    return Body.build(f"flower{k}", pos=(FLOWER_X, FLOWER_Y[k], FLOWER_CUT_Z), mass=0.010, shapes=shapes,
                      color=petal, friction=(0.8, 0.7), ignore=("shears_l", "shears_r"), damping=(0.2, 0.5))


PIVOT = hinge("pivot", "shears_l", "shears_r", anchor=tuple(REST.tolist()), axis="Z", limits_deg=(-2.0, 50.0),
              damping=2e-4)
for _k in range(SHOOTS):
    glue(f"hold_shoot{_k}", "pin", f"shoot{_k}", anchor=(SHOOT_X, SHOOT_Y[_k], CUT_Z))
for _k in range(FLOWERS):
    glue(f"hold_flower{_k}", "pin", f"flower{_k}", anchor=(FLOWER_X, FLOWER_Y[_k], FLOWER_CUT_Z))

OBJECTS = [_mat(), _planter(), _pin(), _half("shears_l", 1.0), _half("shears_r", -1.0), _stubs()] + \
    [_shoot(k) for k in range(SHOOTS)] + [_flower(k) for k in range(FLOWERS)]


# -- followed after every control step --------------------------------------------------------------------------------

def _fresh() -> dict:
    return {"cut": [False] * SHOOTS, "flower_cut": [False] * FLOWERS, "armed": [False] * (SHOOTS + FLOWERS),
            "kicks": [], "tcps": None, "s": {}}


# which way each cut top tips over as it comes free (sideways, so the overhead camera sees it lying on the hedge)
FALL = {"shoot0": (-0.25, 1.0), "shoot1": (-0.25, 1.0), "shoot2": (-0.25, -1.0), "flower0": (-1.0, 0.3),
        "flower1": (-1.0, -0.3)}


def _kick(base, name) -> None:
    import torch

    d = np.array(FALL[name]) / np.linalg.norm(FALL[name])
    v = [0.12 * d[0], 0.12 * d[1], 0.05, -5.0 * d[1], 5.0 * d[0], 0.0]  # a little push, tipping it toward d
    asset = base.scene[name]
    asset.write_root_velocity_to_sim(torch.tensor([v], dtype=torch.float32, device=asset.device))


_EP = _fresh()
_DEBUG = {"on": bool(os.environ.get("SHEARS_DEBUG")), "steps": 0}
# each stem: (x, y, the lowest and highest height the blades may cross it to cut it)
_STEMS = [(SHOOT_X, y, CUT_Z - Z_TOL, CUT_Z + Z_TOL) for y in SHOOT_Y] + \
    [(FLOWER_X, y, HEDGE_TOP + 0.004, FLOWER_HEAD - 0.006) for y in FLOWER_Y]


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


def _pose(base, origin, name):
    p = as_numpy(base.scene[name].data.root_link_pose_w)[0]
    return p[:3] - origin, rotation(p[3:7])


def opening_deg(rl, rr) -> float:
    """How far the blades are open: the right half's turn against the left half about the pivot, degrees."""
    rel = rl.T @ rr
    return math.degrees(math.atan2(rel[1, 0], rel[0, 0]))


def _stem_in_blades(pl, rl, opening, stem):
    """Where a stem crosses the blade plane, in the left half's frame: (in the cutting zone?, distance from the pivot,
    angle from the left blade (the right blade is at ``opening``), the crossing height)."""
    x, y, zlo, zhi = stem
    n = rl[:, 2]
    if abs(n[2]) < 0.5:
        return False, 0.0, 0.0, 0.0
    z = pl[2] - (n[0] * (x - pl[0]) + n[1] * (y - pl[1])) / n[2]
    v = rl.T @ (np.array([x, y, z]) - pl)
    r, ang = math.hypot(v[0], v[1]), math.degrees(math.atan2(v[1], v[0]))
    return (zlo <= z <= zhi and R_MIN <= r <= R_MAX), r, ang, z


def _top_info(base, origin, name) -> str:
    p, r = _pose(base, origin, name)
    return f"{(p[0]) * 1000:.0f},{p[1] * 1000:.0f},{(p[2] - TOP) * 1000:.0f}/{math.degrees(math.acos(min(1.0, r[2, 2]))):.0f}"


@on_step
def _follow(env) -> None:
    base = env.unwrapped
    origin = as_numpy(base.scene.env_origins[0])
    pl, rl = _pose(base, origin, "shears_l")
    _, rr = _pose(base, origin, "shears_r")
    opening = opening_deg(rl, rr)
    for kick in list(_EP["kicks"]):  # two steps after a top comes free, push it over
        kick[1] -= 1
        if kick[1] <= 0:
            _kick(base, kick[0])
            _EP["kicks"].remove(kick)
    info = []
    for i, stem in enumerate(_STEMS):
        done = _EP["cut"] if i < SHOOTS else _EP["flower_cut"]
        j = i if i < SHOOTS else i - SHOOTS
        if done[j]:
            continue
        zone, r, ang, z = _stem_in_blades(pl, rl, opening, stem)
        if zone and opening >= OPEN_DEG and -ANG_MARGIN <= ang <= opening + ANG_MARGIN:
            _EP["armed"][i] = True  # the stem is between the open blades
        elif not zone or abs(ang - opening / 2) > LEAVE_DEG:
            _EP["armed"][i] = False
        if _EP["armed"][i] and opening <= SHUT_DEG:
            done[j] = True
            set_glue_on(f"hold_shoot{j}" if i < SHOOTS else f"hold_flower{j}", False)
            _EP["kicks"].append([f"shoot{j}" if i < SHOOTS else f"flower{j}", 2])
        if zone:
            info.append(f"{'s' if i < SHOOTS else 'f'}{j}:r{r * 1000:.0f} a{ang:.0f} z{(z - TOP) * 1000:.0f}"
                        f"{'*' if _EP['armed'][i] else ''}")
    level = math.degrees(math.acos(max(-1.0, min(1.0, abs(float(rl[2, 2]))))))
    on_mat = (MX0 + 0.02 <= pl[0] <= MX1 - 0.02 and MY0 + 0.02 <= pl[1] <= MY1 - 0.02
              and pl[2] <= REST[2] + 0.012 and level <= 10.0)
    _EP["tcps"] = _tcps(base, origin)
    _EP["s"] = {"on_mat": on_mat, "closed": opening <= SHUT_DEG}
    if _DEBUG["on"]:
        _DEBUG["steps"] += 1
        if _DEBUG["steps"] % 5 == 0:
            tl, trc = _EP["tcps"]
            print(f"[shears] t={_DEBUG['steps'] / 20:5.1f}s pivot={np.round((pl - [0, 0, TOP]) * 1000).astype(int).tolist()}"
                  f" open={opening:5.1f} yaw={math.degrees(math.atan2(rl[1, 0], rl[0, 0])):6.1f} tilt={level:4.1f}"
                  f" cut={_EP['cut']} fl={_EP['flower_cut']} {' '.join(info)} on_mat={on_mat}"
                  f" L={np.round((tl - [0, 0, TOP]) * 1000).astype(int).tolist()}"
                  f" R={np.round((trc - [0, 0, TOP]) * 1000).astype(int).tolist()} score={score(None)}"
                  f" tops={[_top_info(base, origin, f'shoot{k}') for k in range(SHOOTS)]}",
                  file=sys.stderr, flush=True)


# -- scoring --------------------------------------------------------------------------------------------------------

def score(objects) -> float:
    """0.0 once a flower has been severed. Otherwise 1.0 with all three shoots severed at their cut height, the shears
    lying closed on the mat and both arms back where they started, at the end; 0.5 with all three shoots severed; 0.25
    with at least one; else 0.0. (Severed stays severed; here a stem is severed when the blades, opened 20 degrees round
    it, close to 6 degrees on it; a real trial observes severing directly.)"""
    if any(_EP["flower_cut"]):
        return 0.0
    n = sum(_EP["cut"])
    if n == SHOOTS:
        s = _EP["s"]
        if s.get("on_mat") and s.get("closed") and _arms_home():
            return 1.0
        return 0.5
    return 0.25 if n >= 1 else 0.0


_HOME: dict = {}


def _arms_home() -> bool:
    """Both arms back where they started (each TCP within HOME_WITHIN of its start)."""
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
    """Full score with both halves of the shears still on the mat: the trial ends there."""
    return score(objects) >= 1.0 and objects["shears_l"].resting() and objects["shears_r"].resting()


# -- scripted demo ----------------------------------------------------------------------------------------------------
# Both hands come down on the handles and take them, lift the shears off the mat and carry them up to the hedge's top,
# opening the blades on the way. For each shoot, left to right: the open blades go in round it at its cut height and
# close (it is cut off and falls), open again and come back out. Then the hands close the shears, lay them back on the
# mat, let go and go back to where they started.

HANDLE_FINGER = -0.017
GRIP_DZ = 0.004  # the TCP this far above the handle's middle (the fingertips clear of the mat)
OVER = 0.060
LEAD_S, LEAD_STEPS = 3.0, 15
OPEN = 40.0  # degrees: how far the blades open
SHUT = -5.0  # aimed past closed: the bumpers stop the blades at -2
CUT_PIVOT_X = SHOOT_X - 0.120  # each shoot 12 cm out along the blades
BACK_X = 0.400  # out of the hedge: the open blades' tips 2.5 cm short of the shoots
AIM_DZ = 0.005  # aimed this much higher: the loaded arms sag about 6 mm


def _grips(pivot, opening_deg_: float) -> dict:
    """Each hand's TCP pose with the pivot at ``pivot`` and the blades ``opening_deg_`` open, opened evenly."""
    half = math.radians(opening_deg_) / 2
    out = {}
    for arm, s in (("left", 1.0), ("right", -1.0)):
        a = BETA + half
        phi = math.pi - s * a
        p = np.asarray(pivot, float) + GRIP_R * np.array([math.cos(phi), math.sin(phi), 0.0]) + [0.0, 0.0, H_Z + GRIP_DZ]
        out[arm] = (p, grip_rotation(-s * a))
    return out


def _good_ik(arm: str, pos, rot) -> np.ndarray:
    import itertools

    from yam_arena import ik, station

    best = None
    for seed in itertools.product((-0.7, 0.0, 0.7), (1.3, 1.9), (1.0, 1.7), (-1.3, -0.8), (-0.4, 0.0, 0.4), (-0.4, 0.4)):
        q, pe, re = ik.solve(np.asarray(pos, float), ik.matrix_to_quat_xyzw(rot), np.array(seed),
                             base_pos=station.BASES[arm], iters=200)
        if pe < 0.001 and re < math.radians(0.5):
            room = float(np.minimum(q - ik.LOWER, ik.UPPER - q).min())
            if best is None or room > best[1]:
                best = (q, room)
    if best is None:
        raise ValueError(f"no IK solution for the {arm} grip at {pos}")
    return best[0]


def _joint_path(arm: str, q0, q1, label: str, finger) -> list[Waypoint]:
    from yam_arena import ik, station

    out = []
    for k in range(1, LEAD_STEPS + 1):
        p, r = ik.tcp_pose(q0 + (q1 - q0) * k / LEAD_STEPS, station.BASES[arm])
        out.append(Waypoint(LEAD_S / LEAD_STEPS, tuple(float(v) for v in p), finger, label, r))
    return out


def _w(t, pos, finger, label, rot) -> Waypoint:
    return Waypoint(t, tuple(float(v) for v in pos), finger, label, rot)


def demo():
    from yam_arena.constants import READY

    plan = {"left": [], "right": []}
    state = {"pivot": REST.copy(), "open": 0.0}

    def both(t, pivot, opening, finger, label, dz=0.0):
        for arm, (p, r) in _grips(pivot, opening).items():
            plan[arm].append(_w(t, p + [0.0, 0.0, dz], finger, label, r))

    def move(t, pivot, opening, label, n=4, finger=FINGER_CLOSED):
        """Both hands together, the pivot and the opening changing evenly (in n steps)."""
        p0, o0 = state["pivot"], state["open"]
        p1 = np.asarray(pivot, float)
        for j in range(1, n + 1):
            f = j / n
            both(t / n, p0 + (p1 - p0) * f, o0 + (opening - o0) * f, finger, label)
        state["pivot"], state["open"] = p1, opening

    for arm in ("left", "right"):
        p, r = _grips(REST, 0.0)[arm]
        plan[arm] += _joint_path(arm, np.array(READY, float), _good_ik(arm, p + [0.0, 0.0, OVER], r),
                                 "over the shears' handle", HANDLE_FINGER)
    both(0.5, REST, 0.0, HANDLE_FINGER, "over the shears' handle", OVER)
    both(1.2, REST, 0.0, HANDLE_FINGER, "down round the handle")
    both(0.6, REST, 0.0, FINGER_CLOSED, "take the shears by both handles")
    lift = REST + [0.0, 0.0, 0.05]
    move(1.0, lift, 0.0, "lift the shears off the mat", n=2)
    level = CUT_Z + AIM_DZ
    move(2.4, (BACK_X, SHOOT_Y[0], level), OPEN, "carry them up to the hedge's top, opening the blades", n=8)
    for k, y in enumerate(SHOOT_Y):
        if k:
            move(1.2, (BACK_X, y, level), OPEN, "along to the next shoot", n=4)
        move(1.4, (CUT_PIVOT_X, y, level), OPEN, "the open blades in round the shoot", n=5)
        move(1.0, (CUT_PIVOT_X, y, level), SHUT, "close the blades: snip", n=5)
        move(0.4, (CUT_PIVOT_X, y, level), SHUT, "close the blades: snip", n=1)
        move(0.8, (CUT_PIVOT_X, y, level), OPEN, "open them again", n=4)
        move(1.0, (BACK_X, y, level), OPEN, "back out of the hedge", n=4)
    move(1.0, (BACK_X, SHOOT_Y[-1], level), SHUT, "close the shears", n=4)
    move(2.2, lift, SHUT, "carry them back over the mat", n=6)
    move(1.2, REST - [0.0, 0.0, 0.002], SHUT, "lay them down on the mat, closed", n=3)
    both(0.6, REST - [0.0, 0.0, 0.002], SHUT, HANDLE_FINGER, "let go")
    both(1.0, REST, SHUT, HANDLE_FINGER, "rise clear of the handles", OVER)
    for arm in ("left", "right"):
        p, r = _grips(REST, SHUT)[arm]
        plan[arm] += _joint_path(arm, _good_ik(arm, p + [0.0, 0.0, OVER], r), np.array(READY, float),
                                 "back to where it started", HANDLE_FINGER)
        last = plan[arm][-1]
        plan[arm].append(_w(3.0, last.pos, HANDLE_FINGER,
                            "hold still: the trial ends once the full score has held 1 s with both arms home",
                            last.yaw))
    return plan
