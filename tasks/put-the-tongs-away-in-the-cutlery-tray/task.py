"""On a kitchen counter: a pair of stainless spring kitchen tongs with red silicone heads lies flat, sprung open, in
front of the robot. A black pull tab sticks out of the tongs' hinge end. Beyond them stands a white plastic cutlery
drawer tray with a long, narrow compartment along its near side and two shorter compartments behind it.
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import parts  # noqa: E402  (before lots, so the on_reset hook runs before the settle)
import lots  # noqa: E402
from yam_arena.demo import FINGER_OPEN, Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]

STEEL, RED, BLACK, WHITE = (0.74, 0.75, 0.77), (0.82, 0.12, 0.10), (0.06, 0.06, 0.07), (0.93, 0.93, 0.91)

# --- tongs ---------------------------------------------------------------------------------------------------------
# Part frame: origin at the hinge axis, on the counter; x along the tongs from the hinge end to the heads, z up. The
# shapes are written squeezed shut (both hinges at 0); at reset the springs open them to OPEN rad each.
# Each arm is a stainless strip 4 mm thick and 20 mm deep standing on its edge, with a red silicone head at its tip.
ARM_LEN = 0.220  # hinge axis to the end of the heads
ARM_T, ARM_H = 0.004, 0.020
HEAD_L, HEAD_H = 0.025, 0.022
PIVOT_Y = 0.006  # each arm's hinge pin, either side of the centre line
OPEN = math.radians(10.0)  # each arm's open stop
SPRING = 0.8  # N m/rad, each hinge
PRELOAD = 0.20  # rad: the open target lies past the stop, so the open tongs press on their stops
KNUCKLE_R, KNUCKLE_H = 0.010, 0.022
TAB_TRAVEL = 0.010  # the pull tab slides out of the hinge end by up to 10 mm
TAB_SPRING = 40.0  # N/m: the tab springs back in unless the lock has caught
TAB_GRIP = (0.016, 0.014, 0.016)  # the tab's finger grip
TAB_GRIP_X = -0.026  # its centre, part frame, with the tab in
LIFT = 0.001  # the part's origin starts 1 mm above the counter


def _arm(side: int):
    """Arm on the +y side (side 1) or the -y side (side -1), written shut: its head's inner face 0.5 mm off centre."""
    y_arm = side * (0.005 + ARM_T / 2)  # strip from 5 to 9 mm off centre
    head_in, head_out = 0.0005, 0.009
    return [
        parts.box((ARM_LEN - HEAD_L - 0.008, ARM_T, ARM_H),
                  at=(0.008 + (ARM_LEN - HEAD_L - 0.008) / 2, y_arm, ARM_H / 2 + 0.0005), color=STEEL),
        parts.box((HEAD_L, head_out - head_in, HEAD_H),
                  at=(ARM_LEN - HEAD_L / 2, side * (head_in + head_out) / 2, HEAD_H / 2), color=RED),
    ]


TONGS_A = (0.200, -0.100, TOP + LIFT)  # hinge end to the robot's right, heads to the left
YAW_A = math.pi / 2

TONGS = parts.Part(
    "tongs",
    pos=TONGS_A,
    yaw=YAW_A,
    mass=0.015,
    friction=(0.6, 0.5),
    shapes=[parts.cylinder(KNUCKLE_R, KNUCKLE_H, at=(0.0, 0.0, KNUCKLE_H / 2), color=STEEL, axis="z")],
    joints=[
        parts.Hinge("arm_l", shapes=_arm(1), pivot=(0.0, PIVOT_Y, 0.010), axis=(0.0, 0.0, 1.0), lower=0.0, upper=OPEN,
                    mass=0.032, spring=SPRING, damping=0.02),
        parts.Hinge("arm_r", shapes=_arm(-1), pivot=(0.0, -PIVOT_Y, 0.010), axis=(0.0, 0.0, -1.0), lower=0.0,
                    upper=OPEN, mass=0.032, spring=SPRING, damping=0.02),
        parts.Slider("tab", axis=(-1.0, 0.0, 0.0), travel=TAB_TRAVEL, mass=0.006, spring=TAB_SPRING, damping=1.0,
                     shapes=[
                         parts.box((0.012, 0.006, 0.010), at=(-0.012, 0.0, 0.012), color=BLACK),  # stem
                         parts.box(TAB_GRIP, at=(TAB_GRIP_X, 0.0, 0.004 + TAB_GRIP[2] / 2), color=BLACK),
                     ]),
    ],
)
TONGS_END = -0.034 - TAB_TRAVEL  # the tab's end with the tab out (part x)
TONGS_MID = (TONGS_END + ARM_LEN) / 2  # the middle of the locked tongs, part x

# --- cutlery tray (fixed: it sits in its drawer) --------------------------------------------------------------------
WALL, DEPTH, FLOOR = 0.006, 0.050, 0.004
LONG_X0, LONG_W = 0.330, 0.065  # the long compartment's inner near face and its inner width (front to back)
LONG_Y = 0.145  # half its inner length (left to right)
BACK_W = 0.077  # the two short compartments' inner width
X_NEAR = LONG_X0 - WALL
X_DIV = LONG_X0 + LONG_W
X_FAR = X_DIV + WALL + BACK_W
TRAY_Y = LONG_Y + WALL
TRAY_X = (X_NEAR + X_FAR + WALL) / 2
TRAY_LEN_X = X_FAR + WALL - X_NEAR


def _tray_shapes():
    h, zc = DEPTH - FLOOR, FLOOR + (DEPTH - FLOOR) / 2
    rel = lambda x: x - TRAY_X  # noqa: E731
    return [
        parts.box((TRAY_LEN_X, 2 * TRAY_Y, FLOOR), at=(0.0, 0.0, FLOOR / 2), color=WHITE),
        parts.box((WALL, 2 * TRAY_Y, h), at=(rel(X_NEAR + WALL / 2), 0.0, zc), color=WHITE),  # near wall
        parts.box((WALL, 2 * TRAY_Y, h), at=(rel(X_DIV + WALL / 2), 0.0, zc), color=WHITE),  # long divider
        parts.box((WALL, 2 * TRAY_Y, h), at=(rel(X_FAR + WALL / 2), 0.0, zc), color=WHITE),  # far wall
        parts.box((TRAY_LEN_X - 2 * WALL, WALL, h), at=(0.0, LONG_Y + WALL / 2, zc), color=WHITE),  # left end
        parts.box((TRAY_LEN_X - 2 * WALL, WALL, h), at=(0.0, -LONG_Y - WALL / 2, zc), color=WHITE),  # right end
        parts.box((BACK_W, WALL, h), at=(rel(X_DIV + WALL + BACK_W / 2), 0.0, zc), color=WHITE),  # back divider
    ]


TRAY = parts.Part("cutlery_tray", pos=(TRAY_X, 0.0, TOP), friction=(0.5, 0.4), shapes=_tray_shapes())

OBJECTS = [TONGS, TRAY]

# One layout. lots.py is used only for its fresh settle: the tongs are written shut, the on_reset hook points their
# springs open, and a zero move makes the stack settle again (0.5 s) before the first frame, so the tongs start open.
# (Moving the jointed tongs to a second layout per trial did not place them reliably on the GPU, so there is one.)
LOTS = lots.declare(OBJECTS, lots.Lot("A", place={"tongs": TONGS_A}, yaw={"tongs": YAW_A}))

# --- scoring -------------------------------------------------------------------------------------------------------
EPS = 1e-6
SHUT = 0.03  # rad: an arm within this of shut (heads under about 15 mm apart)
TAB_LOCKED = 0.007  # m: the tab pulled out at least 7 mm
TAB_FREE = 0.003  # m: pushed back in to under 3 mm, the lock lets go
IN_WALL = 0.001  # a shape may touch a wall: up to 1 mm into its face (contact)
ON_FLOOR = 0.003  # resting: the tongs' lowest point within 3 mm of the compartment floor
OFF_TABLE = 0.005  # the tongs' lowest point more than 5 mm below the counter: off the table
CLEAR = 0.005  # released: no gripper or finger within 5 mm of any part of the tongs
SUCCESS_HOLD_S = 2.0

FLOOR_Z = TOP + FLOOR
RIM_Z = TOP + DEPTH
LONG_BOX = (LONG_X0, X_DIV, -LONG_Y, LONG_Y)  # inner faces: near, far, right, left


def _rot(q) -> np.ndarray:
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _boxes(t):
    """(centre, rotation, size) of every shape of the tongs: the hinge knuckle, both arms and heads, the tab."""
    poses = {"base": (np.asarray(t.pos, float), _rot(t.quat_xyzw))}
    for name, (p, q) in getattr(t, "links", {}).items():
        poses[name] = (np.asarray(p, float), _rot(q))
    return [(c, r, s) for _, c, r, s in TONGS.world_boxes(poses)]


def _corners(t) -> np.ndarray:
    signs = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], float)
    return np.concatenate([c + (signs * (s / 2)) @ r.T for c, r, s in _boxes(t)])


def locked(t) -> bool:
    """The lock has caught: the tab is pulled out at least 7 mm and both arms are within 0.03 rad of shut."""
    j = t.joints
    return bool(j.get("tab", 0.0) >= TAB_LOCKED - EPS and j.get("arm_l", 1.0) <= SHUT + EPS
                and j.get("arm_r", 1.0) <= SHUT + EPS)


def in_long_compartment(t) -> bool:
    """Every corner of every shape of the tongs inside the long compartment's walls (up to 1 mm into a face), none
    above its rim, and the tongs' lowest point within 3 mm of its floor."""
    p = _corners(t)
    x0, x1, y0, y1 = LONG_BOX
    inside = (p[:, 0].min() >= x0 - IN_WALL - EPS and p[:, 0].max() <= x1 + IN_WALL + EPS
              and p[:, 1].min() >= y0 - IN_WALL - EPS and p[:, 1].max() <= y1 + IN_WALL + EPS)
    low, high = p[:, 2].min(), p[:, 2].max()
    return bool(inside and high <= RIM_Z + EPS and FLOOR_Z - IN_WALL - EPS <= low <= FLOOR_Z + ON_FLOOR + EPS)


# Released: in the simulator an on_step callback measures, every step, how close both arms' fingers and grippers are
# to every shape of the tongs. None means not measured this step. Scoring fails closed: inside the simulator an
# unmeasured step counts as held. Offline (near_misses.py, no simulator loaded) there is no robot, so it counts as clear.
_grip = {"clear": None}


def _released() -> bool:
    if _grip["clear"] is None:
        return "isaaclab" not in sys.modules
    return bool(_grip["clear"])


# The lock, as the real tongs' pull tab works: pulled out while the tongs are squeezed shut, it catches and holds them
# shut (both hinge springs then pull toward shut, the tab stays out); pushed back in, it lets go and they spring open.
# The simulator applies it by moving the springs' targets; scoring reads only the tab and arm positions.
_lock = {"caught": False}


def _targets(sim, caught: bool) -> None:
    sim.set_target("tongs", "arm_l", -0.3 if caught else OPEN + PRELOAD)
    sim.set_target("tongs", "arm_r", -0.3 if caught else OPEN + PRELOAD)
    sim.set_target("tongs", "tab", TAB_TRAVEL + 0.003 if caught else 0.0)


@parts.on_reset
def _reset(sim) -> None:
    _lock["caught"] = False
    _grip["clear"] = None
    _targets(sim, False)


@parts.on_step
def _step(sim) -> None:
    from yam_arena import analysis

    st = sim.states()
    t = st["tongs"]
    j = t.joints
    if not _lock["caught"] and locked(t):
        _lock["caught"] = True
        _targets(sim, True)
    elif _lock["caught"] and j.get("tab", 0.0) <= TAB_FREE + EPS:
        _lock["caught"] = False
        _targets(sim, False)
    _grip["clear"] = None
    try:
        base = sim.env.unwrapped
        origin = parts._np(base.scene.env_origins[0])
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
        if hulls:
            _grip["clear"] = all(analysis.hulls_to_box(hulls, c, r, s)[0] > CLEAR for c, r, s in _boxes(t))
    except Exception:  # noqa: BLE001  unmeasured: stays None, which scores as held
        _grip["clear"] = None


def _settled(t) -> bool:
    return bool(t.resting() and _released() and _corners(t)[:, 2].min() >= TOP - OFF_TABLE - EPS)


def success(objects):
    """The tongs are locked shut and lie in the long compartment, let go and still (held 2 s)."""
    t = objects["tongs"]
    return bool(_settled(t) and locked(t) and in_long_compartment(t))


def distance(objects):
    t = objects["tongs"]
    return float(np.hypot(t.pos[0] - (LONG_X0 + LONG_W / 2), t.pos[1]))


def score(objects):
    """1.0 locked in the long compartment; 0.5 locked, let go and still anywhere else on the counter or tray; 0.25 in
    the long compartment but not locked; otherwise 0. Every credited state is let go and still."""
    t = objects["tongs"]
    if not _settled(t):
        return 0.0
    inside, shut = in_long_compartment(t), locked(t)
    if inside and shut:
        return 1.0
    if shut:
        return 0.5
    if inside:
        return 0.25
    return 0.0


# --- demo ----------------------------------------------------------------------------------------------------------
TIP = 0.013  # finger tips below the TCP
GRIP_AT = 0.160  # squeeze point, part x: 160 mm from the hinge axis toward the heads
SQUEEZE_Z = LIFT + 0.004 + TIP  # TCP above the counter: tips 5 mm above the arms' lower edges
TAB_Z = LIFT + 0.007 + TIP  # tips 3 mm above the tab's underside
SQUEEZE = -0.005  # pads 10 mm apart: 4 mm past each face of the shut 18 mm pair
TAB_PINCH = -0.004  # pads 8 mm apart on the 14 mm tab
TAB_OPEN = -0.018
PULL = 0.013  # commanded pull: 3 mm past the tab's 10 mm travel (the arm gives a little)
RELEASE = -0.012  # pads 24 mm apart: opens inside the 65 mm compartment without touching its walls
TRAVEL = 0.085
YAWS = {"left": -math.pi / 2, "right": math.pi / 2}  # fingers close front to back, across the tongs
PLACE_Z = FLOOR + 0.006 + (SQUEEZE_Z - LIFT)  # TCP above the counter with the tongs 6 mm above the floor


def _w(xy, z):
    return (float(xy[0]), float(xy[1]), TOP + z)


def demo():
    """The arm on the heads' side squeezes the tongs shut at mid-length; the other arm pinches the pull tab and pulls
    it out until the lock catches, lets go and moves away. The first arm lifts the locked tongs, carries them over the
    long compartment, lowers them in and lets go. It plays open-loop from the published start layout."""
    p = np.asarray(TONGS_A, float)  # the published start layout
    yaw = YAW_A
    ax = np.array([math.cos(yaw), math.sin(yaw)])  # along the tongs, hinge to heads
    squeeze = p[:2] + GRIP_AT * ax
    tab = p[:2] + TAB_GRIP_X * ax
    pulled = tab - PULL * ax
    centre = np.array([LONG_X0 + LONG_W / 2, 0.0])
    place = centre + (GRIP_AT - TONGS_MID) * ax
    holder = "left" if ax[1] > 0 else "right"  # the heads' side
    puller = "right" if holder == "left" else "left"
    side = 1.0 if puller == "left" else -1.0
    park = (0.15, side * 0.25)
    hy, py = YAWS[holder], YAWS[puller]
    hold = [
        Waypoint(1.5, _w(squeeze, TRAVEL), FINGER_OPEN, "over the arms' middle", hy),
        Waypoint(1.5, _w(squeeze, SQUEEZE_Z), FINGER_OPEN, "down beside the arms", hy),
        Waypoint(1.0, _w(squeeze, SQUEEZE_Z), SQUEEZE, "squeeze shut", hy),
        Waypoint(3.6, _w(squeeze, SQUEEZE_Z), SQUEEZE, "keep squeezed while the tab is pulled", hy),
        Waypoint(1.2, _w(squeeze, TRAVEL), SQUEEZE, "lift", hy),
        Waypoint(2.0, _w(place, TRAVEL), SQUEEZE, "carry over the long compartment", hy),
        Waypoint(1.5, _w(place, PLACE_Z), SQUEEZE, "lower in", hy),
        Waypoint(0.6, _w(place, PLACE_Z), RELEASE, "let go", hy),
        Waypoint(1.0, _w(place, TRAVEL), RELEASE, "up", hy),
        Waypoint(1.0, _w((0.22, -side * 0.20), TRAVEL), RELEASE, "away", hy),
        Waypoint(3.0, _w((0.22, -side * 0.20), TRAVEL), RELEASE, "wait", hy),
    ]
    pull = [
        Waypoint(1.5, _w(tab, TRAVEL), TAB_OPEN, "over the tab", py),
        Waypoint(1.5, _w(tab, TAB_Z), TAB_OPEN, "down beside the tab", py),
        Waypoint(1.0, _w(tab, TAB_Z), TAB_OPEN, "wait for the squeeze", py),
        Waypoint(0.8, _w(tab, TAB_Z), TAB_PINCH, "pinch the tab", py),
        Waypoint(1.2, _w(pulled, TAB_Z), TAB_PINCH, "pull it out", py),
        Waypoint(0.6, _w(pulled, TAB_Z), TAB_OPEN, "let go", py),
        Waypoint(1.0, _w(pulled, TRAVEL), TAB_OPEN, "up", py),
        Waypoint(1.5, _w(park, TRAVEL), TAB_OPEN, "out of the way", py),
    ]
    return {holder: hold, puller: pull}
