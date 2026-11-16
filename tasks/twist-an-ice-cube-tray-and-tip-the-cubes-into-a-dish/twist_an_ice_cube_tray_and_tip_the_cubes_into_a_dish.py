"""A plastic ice-cube tray of eight frozen cubes rests across the top of a glass dish, hanging by the tabs at its two
ends on the dish's end walls.

Task: get the cubes into the dish. Holding the tray by both end tabs, twist it, one end one way and the other the
other way, so the cubes crack loose from their compartments; lift the tray off the dish's rim, turn it over the dish so
the cubes fall in, turn it back, set it back across the dish, and take both arms back to where they started. Frozen
cubes do not come out however far the tray is turned until it has been twisted; twisting needs a hand on each end (one
hand alone just turns the whole tray).

Run it:  inspect-robots run --embodiment yam_arena -E task=twist_an_ice_cube_tray_and_tip_the_cubes_into_a_dish.py \\
           --instruction "..." --scorer task_score ...
"""

from __future__ import annotations

import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from bodykit import (Body, BoxShape, allow_matrix_grips, as_numpy, glue, hinge, on_reset, on_step, rotation,  # noqa: E402
                     set_glue_on)
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, TCP_OFFSET  # noqa: E402
from yam_arena.demo import Waypoint, grip_rotation  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

allow_matrix_grips()

SCENE = "station"
TOP = TABLE_TOP[SCENE]

# -- the glass dish ---------------------------------------------------------------------------------------------------
DISH = np.array([0.300, 0.000])
DISH_IN = (0.150, 0.165, 0.080)  # inside: x, y, height (floor to rim)
DISH_W = 0.004
RIM = TOP + DISH_W + DISH_IN[2]  # the top of its walls
DISH_IN_LOW = 0.050  # a cube in the dish is lower than this above the table (the tray's compartments are higher)

# -- the tray: two halves (+y and -y) joined at the middle by a torsion hinge along y ---------------------------------
CELL = 0.035  # each compartment, square
WALL = 0.003
TRAY_H = 0.032  # the tray's depth
HALF_L = 2 * CELL + 2.5 * WALL  # each half's length along y (the middle seam takes half a wall each side)
TRAY_W = 2 * CELL + 3 * WALL  # across x
SEAM = 0.002  # the gap between the halves
TAB = (0.022, 0.050, 0.016)  # each end tab: x, y (sticking out past the tray's end), z
TAB_Z = RIM + TAB[2] / 2  # the tabs rest on the dish's end walls
TRAY_TOP = RIM + TAB[2]  # the tray's top edge, level with the tabs' tops
AXIS = np.array([DISH[0], 0.0, TAB_Z])  # the torsion hinge's line (along y), through the tabs' middles
END_Y = SEAM / 2 + HALF_L  # each half's outer end (|y|)
TAB_Y = END_Y + TAB[1] / 2  # each tab's middle (|y|)
CUBE = 0.028
CUBE_MASS = 0.02
TWIST_K = 0.010  # N m per degree: the tray's stiffness in twist
FREE_DEG = 12.0  # twisted this far (one end against the other) the cubes crack loose ...
END_DEG = 5.0  # ... with each end turned at least this far from level, the two opposite ways (a twist from both ends:
# the lugs keeping the hanging tray centred cannot hold one end still in place of a hand)
TWIST_MAX = 35.0

TRAY_C, ICE_C, GLASS_C, DARK = (0.55, 0.78, 0.92), (0.88, 0.95, 0.98), (0.80, 0.87, 0.88), (0.25, 0.30, 0.32)

# -- scoring ----------------------------------------------------------------------------------------------------------
BACK_XY, BACK_Z = 0.015, 0.010  # each tab back on the dish's rim
UPRIGHT = 15.0
HOME_WITHIN = 0.05
SUCCESS_HOLD_S = 1.0


def _cells(side: float) -> list[np.ndarray]:
    """The middles of one half's four compartments (world, at the compartments' floor height)."""
    out = []
    for i in range(2):  # along y, from the seam outward
        y = side * (SEAM / 2 + WALL / 2 + CELL / 2 + i * (CELL + WALL))
        for j in (-1.0, 1.0):  # across x
            out.append(np.array([DISH[0] + j * (CELL / 2 + WALL / 2), y, TRAY_TOP - TRAY_H + WALL]))
    return out


def _dish() -> Body:
    lx, ly, h = DISH_IN
    w = DISH_W
    shapes = [BoxShape((0.0, 0.0, w / 2), (lx + 2 * w, ly + 2 * w, w), GLASS_C)]
    for s in (-1.0, 1.0):
        shapes.append(BoxShape((s * (lx / 2 + w / 2), 0.0, w + h / 2), (w, ly + 2 * w, h), GLASS_C))
        shapes.append(BoxShape((0.0, s * (ly / 2 + w / 2), w + h / 2), (lx, w, h), GLASS_C))
        for sx in (-1.0, 1.0):  # lugs on the end walls either side of each tab, keeping the tray centred
            shapes.append(BoxShape((sx * (TAB[0] / 2 + 0.004), s * (ly / 2 + w / 2), w + h + 0.003), (0.004, w, 0.006),
                                   GLASS_C))
    return Body.build("dish", pos=(float(DISH[0]), float(DISH[1]), TOP), mass=0.8, kinematic=True, color=GLASS_C,
                      friction=(0.5, 0.4), shapes=shapes)


def _tab(name: str, side: float) -> Body:
    """An end tab (what the hands hold): origin at its middle. It also carries the hinge and the frozen cubes."""
    shapes = [BoxShape((0.0, 0.0, 0.0), TAB, TRAY_C),
              BoxShape((0.0, side * 0.006, TAB[2] / 2 + 0.0003), (0.014, 0.012, 0.0006), DARK, collide=False)]
    return Body.build(name, pos=(float(DISH[0]), side * TAB_Y, TAB_Z), mass=0.02, shapes=shapes, color=TRAY_C,
                      friction=(1.3, 1.1), com=(0.0, 0.0, 0.0))


def _shell(name: str, side: float, ignore: tuple = ()) -> Body:
    """One half's compartments: a floor and walls, origin at the middle of its top, glued to its tab."""
    yc = side * (SEAM / 2 + HALF_L / 2)
    zf = -TRAY_H + WALL / 2
    shapes = [BoxShape((0.0, 0.0, zf), (TRAY_W, HALF_L, WALL), TRAY_C)]
    for s in (-1.0, 0.0, 1.0):  # walls along y: the two sides and the middle divider
        x = s * (TRAY_W / 2 - WALL / 2)
        shapes.append(BoxShape((x, 0.0, -TRAY_H / 2), (WALL, HALF_L, TRAY_H), TRAY_C))
    for d, t in ((WALL / 4, WALL / 2), (CELL + WALL, WALL), (HALF_L - WALL / 2, WALL)):  # walls across: at the seam
        # (half a wall), between the cells, at the end; ``d`` from the seam end
        shapes.append(BoxShape((0.0, -side * HALF_L / 2 + side * d, -TRAY_H / 2), (TRAY_W, t, TRAY_H), TRAY_C))
    return Body.build(name, pos=(float(DISH[0]), yc, TRAY_TOP), mass=0.04, shapes=shapes, color=TRAY_C,
                      friction=(0.3, 0.25), ignore=ignore)


def _cube(name: str, at) -> Body:
    shapes = [BoxShape((0.0, 0.0, 0.0), (CUBE, CUBE, CUBE), ICE_C),
              BoxShape((0.0, 0.0, CUBE / 2 + 0.0002), (CUBE * 0.6, CUBE * 0.6, 0.0004), (0.97, 0.99, 1.0),
                       collide=False)]
    return Body.build(name, pos=tuple(float(v) for v in at), mass=CUBE_MASS, shapes=shapes, color=ICE_C,
                      friction=(0.15, 0.1), com=(0.0, 0.0, 0.0), damping=(0.05, 0.2))


CUBES = []
_objs = [_dish(), _tab("tab_left", 1.0), _tab("tab_right", -1.0), _shell("tray_left", 1.0),
         _shell("tray_right", -1.0)]
for _side, _half in ((1.0, "left"), (-1.0, "right")):
    for _k, _c in enumerate(_cells(_side)):
        _n = f"cube_{_half}_{_k}"
        CUBES.append((_n, f"tab_{_half}"))
        _objs.append(_cube(_n, _c + np.array([0.0, 0.0, CUBE / 2 + 0.0005])))
OBJECTS = _objs
hinge("twist", "tab_left", "tab_right", anchor=tuple(AXIS.tolist()), axis="Y", limits_deg=(-TWIST_MAX, TWIST_MAX),
      spring=TWIST_K, damping=0.0002, target_deg=0.0)
for _half in ("left", "right"):
    glue(f"shell_{_half}", f"tab_{_half}", f"tray_{_half}", anchor=(float(DISH[0]), (1 if _half == "left" else -1)
                                                                    * END_Y, TAB_Z))
for _n, _t in CUBES:
    glue(f"frozen_{_n}", _t, _n, anchor=tuple(OBJECTS[[o.name for o in OBJECTS].index(_n)].pos))


# -- followed after every control step --------------------------------------------------------------------------------

def _fresh() -> dict:
    return {"tcps": None, "s": {}, "free": False, "max_twist": 0.0}


_EP = _fresh()
_DEBUG = {"on": bool(os.environ.get("ICE_DEBUG")), "steps": 0}


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


def _roll(rot) -> float:
    """A tab's turn about y (deg, + turns its top toward +x)."""
    return math.degrees(math.atan2(float(rot[0, 2]), float(rot[2, 2])))


def _in_dish(pos) -> bool:
    """In the dish, down in it (below 5 cm), not still in the tray hanging in its top."""
    return (abs(float(pos[0]) - DISH[0]) <= DISH_IN[0] / 2 and abs(float(pos[1]) - DISH[1]) <= DISH_IN[1] / 2
            and float(pos[2]) < TOP + DISH_IN_LOW)


def _tab_back(pos, rot, side: float) -> bool:
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(rot[2, 2])))))
    return (float(np.linalg.norm(pos[:2] - np.array([DISH[0], side * TAB_Y]))) <= BACK_XY
            and abs(float(pos[2]) - TAB_Z) <= BACK_Z and tilt <= UPRIGHT)


@on_step
def _follow(env) -> None:
    base = env.unwrapped
    origin = as_numpy(base.scene.env_origins[0])
    st = {}
    for n in ("tab_left", "tab_right", *[c for c, _ in CUBES]):
        a = as_numpy(base.scene[n].data.root_link_pose_w)[0]
        st[n] = (a[:3] - origin, rotation(a[3:7]))
    _EP["tcps"] = _tcps(base, origin)
    rl, rr = _roll(st["tab_left"][1]), _roll(st["tab_right"][1])
    twist = (rl - rr + 180.0) % 360.0 - 180.0
    _EP["max_twist"] = max(_EP["max_twist"], abs(twist))
    both = min(abs(rl), abs(rr)) >= END_DEG and rl * rr < 0.0
    if not _EP["free"] and both and abs(twist) >= FREE_DEG:  # the cubes crack loose
        _EP["free"] = True
        for n, _ in CUBES:
            set_glue_on(f"frozen_{n}", False)
    inside = sum(1 for n, _ in CUBES if _in_dish(st[n][0]))
    _EP["s"] = {"free": _EP["free"], "inside": inside,
                "back": _tab_back(*st["tab_left"], 1.0) and _tab_back(*st["tab_right"], -1.0)}
    if _DEBUG["on"]:
        _DEBUG["steps"] += 1
        if _DEBUG["steps"] % 10 == 0 or _DEBUG["steps"] < 25:
            tl, tr = _EP["tcps"]
            lowest = min(float(st[n][0][2]) for n, _ in CUBES)
            print(f"[ice] t={_DEBUG['steps'] / 20:5.1f}s roll L/R={rl:6.1f}/{rr:6.1f} twist={twist:5.1f} "
                  f"max={_EP['max_twist']:4.1f} free={_EP['free']} in_dish={inside} lowest_cube_z={lowest - TOP:.3f} "
                  f"tabL={np.round(st['tab_left'][0] * 1000).astype(int).tolist()} "
                  f"tabR={np.round(st['tab_right'][0] * 1000).astype(int).tolist()} back={_EP['s']['back']} "
                  f"L={np.round(tl * 1000).astype(int).tolist()} R={np.round(tr * 1000).astype(int).tolist()} "
                  f"score={_points(_status())}", file=sys.stderr, flush=True)


# -- scoring --------------------------------------------------------------------------------------------------------

def _status() -> dict:
    s = _EP["s"]
    return {"free": bool(s.get("free")), "inside": int(s.get("inside", 0)), "back": bool(s.get("back")),
            "home": _arms_home()}


def _points(s) -> float:
    if s["inside"] == len(CUBES) and s["back"] and s["home"]:
        return 1.0
    if s["inside"] == len(CUBES):
        return 0.5
    return 0.25 if s["free"] else 0.0


def score(objects) -> float:
    """1.0 with all eight cubes in the dish, the tray back across the dish (each end tab within 1.5 cm of where it
    started, at its height within 1 cm, upright within 15 degrees) and both arms back where they started (each TCP
    within 5 cm of its start); 0.5 with all eight cubes in the dish; 0.25 with the tray twisted at some point far
    enough to crack the cubes loose (12 degrees, one end against the other, each end turned at least 5 degrees from
    level, the two opposite ways); else 0.0."""
    return _points(_status())


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
    """Full score with the tray and every cube still."""
    names = ["tab_left", "tab_right", "tray_left", "tray_right", *[c for c, _ in CUBES]]
    return score(objects) >= 1.0 and all(objects[n].resting() for n in names)


# -- scripted demo ----------------------------------------------------------------------------------------------------
# Both hands come in level from the sides, pointing along the tray, fingers closing across its end tabs. They twist it,
# the left end's top toward the far side and the right end's toward the arms, and back; lift it 3 cm off the dish's
# rim, turn it together 115 degrees about its length so the cubes fall into the dish, hold it there, turn it back, set
# it back across the dish, let go and go back to where they started.


def _rx(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _ry(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


GRIPS = {"left": _rx(-90.0) @ grip_rotation(-math.pi / 2), "right": _rx(90.0) @ grip_rotation(math.pi / 2)}
SIDE = {"left": 1.0, "right": -1.0}
T_IN = 0.0085  # each TCP this far outward from its tab's middle (the fingertips clear of the dish's wall)
OPEN_T = -0.017  # round the 2.2 cm tab
TWIST = 12.0  # each end turned this far (24 degrees one against the other)
TIP = 115.0
LIFT = 0.030
LEAD_S, LEAD_STEPS = 3.0, 15


def _good_ik(arm: str, pos, rot) -> np.ndarray:
    import itertools

    from yam_arena import ik, station

    best = None
    for seed in itertools.product((-0.5, 0.0, 0.5), (0.8, 1.3), (0.4, 1.0), (-0.8, 0.0, 0.8), (-1.4, 0.0, 1.4),
                                  (-1.5, 0.0, 1.5)):
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


def _hand(arm: str, total: float = 0.0) -> list[Waypoint]:
    from yam_arena.constants import READY

    s, g = SIDE[arm], GRIPS[arm]
    grip = np.array([DISH[0], s * (TAB_Y + T_IN), TAB_Z])
    out_pt = grip + [0.0, 0.0, 0.045]  # above it (the fingers straddle the tab across x)
    out = _joint_path(arm, np.array(READY, float), _good_ik(arm, out_pt, g), "over the tray's end tab", OPEN_T)
    out += [_w(1.0, grip, OPEN_T, "down round the tab", g),
            _w(0.5, grip, FINGER_CLOSED, "take hold of it", g)]
    tw = TWIST if arm == "left" else -TWIST
    up = grip + [0.0, 0.0, LIFT]
    out += [_w(1.0, grip, FINGER_CLOSED, "twist the tray, this end one way and the other end the other", _ry(tw) @ g),
            _w(0.6, grip, FINGER_CLOSED, "hold the twist", _ry(tw) @ g),
            _w(1.0, grip, FINGER_CLOSED, "and back", g),
            _w(0.8, up, FINGER_CLOSED, "lift it off the dish's rim", g)]
    for k in range(1, 6):
        out.append(_w(0.5, grip + [0.0, 0.0, LIFT], FINGER_CLOSED, "turn it over the dish", _ry(TIP * k / 5) @ g))
    out.append(_w(1.5, grip + [0.0, 0.0, LIFT], FINGER_CLOSED, "hold it while the cubes fall into the dish",
                  _ry(TIP) @ g))
    for k in range(1, 6):
        out.append(_w(0.4, grip + [0.0, 0.0, LIFT], FINGER_CLOSED, "turn it back", _ry(TIP * (1 - k / 5)) @ g))
    out += [_w(0.8, grip + [0.0, 0.0, 0.002], FINGER_CLOSED, "set it back across the dish", g),
            _w(0.5, grip + [0.0, 0.0, 0.002], OPEN_T, "let go", g),
            _w(0.8, out_pt, OPEN_T, "up", g)]
    out += _joint_path(arm, _good_ik(arm, out_pt, g), np.array(READY, float), "back to where it started", OPEN_T)
    t = sum(w.duration for w in out)
    out.append(Waypoint(max(total - t, 0.5), out[-1].pos, OPEN_T,
                        "hold still: the trial ends once the full score has held 1 s with both arms home", out[-1].yaw))
    return out


def demo():
    total = sum(w.duration for w in _hand("left")) + 2.0
    return {"left": _hand("left", total), "right": _hand("right", total)}
