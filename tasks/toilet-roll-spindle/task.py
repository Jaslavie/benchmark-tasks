"""On the station table: a white wall panel stands across the far side of the table. On it, low down, is a chrome
two-post toilet-roll holder (posts 14 cm apart, each with a small square socket on its inner face). A spring-loaded
telescoping spindle sits in the two sockets with an empty cardboard tube hanging on it. A new, full roll of toilet
paper lies on its side on the table in front of the holder, and a small grey tray stands to the front left for the
empty tube. The robot must take the spindle out, slide the empty tube off and put it in the tray, slide the new roll
onto the spindle and fit the spindle back in the holder: one end into its socket, the spindle squeezed shorter until
the other end clears its post, then let it spring out into the second socket.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from rollkit import Hinge, finger_points, on_reset, on_step, part  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
TOP = TABLE_TOP[SCENE]

INSTRUCTION = ("Replace the empty toilet roll with the new one: take out the spindle, put the empty tube in the grey "
               "tray, slide the new roll onto the spindle and fit the spindle back in the holder.")

CHROME = (0.74, 0.75, 0.77)
DARK = (0.30, 0.31, 0.33)
PANEL = (0.93, 0.93, 0.90)
PAPER = (0.97, 0.97, 0.95)
CARD = (0.62, 0.46, 0.30)
SPINDLE_OUT = (0.86, 0.87, 0.88)
SPINDLE_IN = (0.70, 0.72, 0.74)
TRAY = (0.45, 0.50, 0.56)

# -- the holder (fixed to the wall panel), world frame ---------------------------------------------------------------
SX = 0.335                 # socket axis: x (distance in front of the arm bases)
SZ = TOP + 0.100           # socket axis height (10 cm above the table)
HY = 0.0                   # holder centre, midway between the arms
GAP = 0.140                # between the posts' inner faces
DEPTH = 0.005              # socket depth
HOLE = 0.016               # square socket, 16 mm
POST_T = 0.017             # post thickness (inner face to outer face)
POST_H = 0.032             # post height
POST_FRONT = SX - 0.018    # front end of the posts
WALL_X = SX + 0.070        # front face of the holder's back plate: the spindle axis is 7 cm off the wall
CHAMFER = 0.003            # 45-degree chamfer on each socket's front edge


def _b(name, lo, hi, color=CHROME, yaw=0.0):
    lo, hi = np.array(lo, float), np.array(hi, float)
    return Box(name, pos=tuple((lo + hi) / 2), size=tuple(hi - lo), color=color, yaw=yaw)


HOLDER_ORIGIN = np.array([SX, HY, SZ])


def _hb(lo, hi, color=CHROME):
    """A box of the holder, from world corners, in the holder's frame."""
    lo, hi = np.array(lo, float), np.array(hi, float)
    return (tuple(float(v) for v in (lo + hi) / 2 - HOLDER_ORIGIN), tuple(float(v) for v in hi - lo), color)


def _post(s: float):
    """One post (s = +1 left, -1 right): a back slab and a frame round the square socket on its inner face."""
    f = HY + s * GAP / 2                       # inner face
    d = HY + s * (GAP / 2 + DEPTH)             # socket bottom
    o = HY + s * (GAP / 2 + POST_T)            # outer face
    y0, y1 = sorted((f, d))
    yb0, yb1 = sorted((d, o))
    h, z0, z1 = HOLE / 2, SZ - POST_H / 2, SZ + POST_H / 2
    c = CHAMFER
    boxes = [
        _hb((POST_FRONT, yb0, z0), (WALL_X, yb1, z1)),                  # back slab
        _hb((POST_FRONT, y0, z0), (SX - h - c, y1, z1)),                # in front of the socket
        _hb((SX + h, y0, z0), (WALL_X, y1, z1)),                        # behind it
        _hb((SX - h - c, y0, SZ + h), (SX + h, y1, z1)),                # above it
        _hb((SX - h - c, y0, z0), (SX + h, y1, SZ - h)),                # below it
    ]
    # the chamfer: a square bar turned 45 degrees whose face runs from the inner face to the socket's front wall
    a = c * math.sqrt(2)
    mid = np.array([SX - h - c / 2, f + s * c / 2])
    n = np.array([-1.0, s]) / math.sqrt(2)     # from the chamfer face into the post
    cx, cy = mid + n * a / 2
    boxes.append(((cx - SX, cy - HY, 0.0), (a, a, 2 * h), CHROME, ("Z", math.pi / 4)))
    return boxes


HOLDER = part("holder", pos=tuple(float(v) for v in HOLDER_ORIGIN), mass=None, friction=0.3, color=CHROME, boxes=(
    *_post(1.0), *_post(-1.0),
    _hb((WALL_X, HY - 0.095, SZ - 0.030), (WALL_X + 0.008, HY + 0.095, SZ + 0.030)),   # back plate
))


def _holder():
    return [
        HOLDER,
        _b("wall_panel", (WALL_X + 0.008, -0.24, TOP), (WALL_X + 0.026, 0.24, TOP + 0.18), PANEL),
        _b("panel_foot", (WALL_X + 0.026, -0.22, TOP), (WALL_X + 0.10, 0.22, TOP + 0.02), PANEL),
    ]


# -- the spindle: two plastic tubes on a spring, with domed ends -----------------------------------------------------
# Part frame: origin at the middle of the spindle at its relaxed length, axis along y. The body is the outer (right)
# tube; the inner (left) tube slides in it on a prismatic joint whose spring pulls it out to the relaxed length.
L0 = 0.156                 # relaxed length, dome tip to dome tip
TRAVEL = 0.030             # it squeezes up to 30 mm shorter
SPRING = 200.0             # N/m (1.2 N fitted, 6 N fully squeezed)
DOME_R = 0.006             # domed ends 12 mm across
OUT_R, IN_R = 0.011, 0.0095
TIP = L0 / 2
FITTED = GAP + 2 * DEPTH   # 150 mm: ends at the bottoms of both sockets
START_Q = FITTED - L0      # -6 mm

def _prism(y0, y1, r, color, n=12):
    """A solid tube along y from y0 to y1 with a regular n-sided cross-section (n/2 slabs turned about y), and a
    round visual cylinder over it."""
    w = 2 * r * math.tan(math.pi / n)
    slabs = tuple(((0.0, (y0 + y1) / 2, 0.0), (2 * r, y1 - y0, w), color, k * 2 * math.pi / n) for k in range(n // 2))
    return slabs


SPINDLE = part(
    "spindle", pos=(SX, HY + (TIP - (GAP / 2 + DEPTH)), SZ - 0.0015), mass=0.020, friction=0.3, color=SPINDLE_OUT,
    boxes=_prism(-TIP + DOME_R, 0.004, OUT_R, SPINDLE_OUT),
    spheres=(((0.0, -TIP + DOME_R, 0.0), DOME_R, SPINDLE_OUT),),
    hinges=(Hinge("inner", anchor=(0.0, 0.0, 0.0), axis="Y", kind="prismatic", lower=-TRAVEL, upper=0.0,
                  start=START_Q, rest=0.0, rate=SPRING, damping=2.0, mass=0.012,
                  boxes=_prism(-0.040, TIP - DOME_R, IN_R, SPINDLE_IN),
                  spheres=(((0.0, TIP - DOME_R, 0.0), DOME_R, SPINDLE_IN),)),),
)


def _ring(r_in, r_out, length, n, colors):
    """A hollow cylinder along y as n flat staves (boxes turned about y); ``colors`` gives (radius split, colour)
    pairs from the inside out."""
    boxes = []
    w = 2 * r_out * math.tan(math.pi / n)
    for k in range(n):
        th = 2 * math.pi * k / n
        lo = r_in
        for hi, col in colors:
            rc = (lo + hi) / 2
            boxes.append(((-rc * math.sin(th), 0.0, -rc * math.cos(th)), (w, length, hi - lo), col, th))
            lo = hi
    return tuple(boxes)


# -- the empty tube and the new roll ---------------------------------------------------------------------------------
TUBE_R_IN, TUBE_R_OUT, TUBE_LEN = 0.0205, 0.0225, 0.100
ROLL_R_IN, ROLL_R_OUT, ROLL_W = 0.0215, 0.0575, 0.100
TUBE_START = (SX, HY, SZ - 0.0015 - (TUBE_R_IN - OUT_R) - 0.0005)
ROLL_X, ROLL_Y = 0.300, -0.150
ROLL_START = (ROLL_X, ROLL_Y, TOP + ROLL_R_OUT + 0.0005)

OLD_TUBE = part("old_tube", pos=TUBE_START, mass=0.004, friction=0.7, color=CARD,
                boxes=_ring(TUBE_R_IN, TUBE_R_OUT, TUBE_LEN, 24, ((TUBE_R_OUT, CARD),)))
ROLL = part("roll", pos=ROLL_START, mass=0.130, friction=0.7, color=PAPER,
            boxes=_ring(ROLL_R_IN, ROLL_R_OUT, ROLL_W, 24, ((ROLL_R_IN + 0.0012, CARD), (ROLL_R_OUT, PAPER))))

# -- the tray for the empty tube (fixed) -----------------------------------------------------------------------------
TRAY_X, TRAY_Y = 0.190, 0.060
TRAY_LX, TRAY_LY = 0.160, 0.220          # inside, front to back and left to right
TRAY_WALL, TRAY_H, TRAY_FLOOR = 0.006, 0.030, 0.004


def _tray():
    x, y, a, b, w, h = TRAY_X, TRAY_Y, TRAY_LX / 2, TRAY_LY / 2, TRAY_WALL, TRAY_H
    return [
        _b("tray_floor", (x - a - w, y - b - w, TOP), (x + a + w, y + b + w, TOP + TRAY_FLOOR), TRAY),
        _b("tray_near", (x - a - w, y - b - w, TOP), (x - a, y + b + w, TOP + h), TRAY),
        _b("tray_far", (x + a, y - b - w, TOP), (x + a + w, y + b + w, TOP + h), TRAY),
        _b("tray_left", (x - a, y + b, TOP), (x + a, y + b + w, TOP + h), TRAY),
        _b("tray_right", (x - a, y - b - w, TOP), (x + a, y - b, TOP + h), TRAY),
    ]


OBJECTS = [*_holder(), *_tray(), SPINDLE, OLD_TUBE, ROLL]

# -- geometry for scoring --------------------------------------------------------------------------------------------
SEAT_MIN = 0.003           # an end is seated when its tip is at least 3 mm inside its socket (past the post's face) ...
SEAT_OFF = 0.007           # ... and within the 16 mm hole: at most 7 mm off its centre, front to back and up and down
ON_SPINDLE = 0.016         # a roll or tube is on the spindle when its axis is within 16 mm of the spindle's axis ...
ON_ANGLE = math.radians(15)  # ... and within 15 degrees of it, and lies between the spindle's ends
HANDS_OFF = 0.03           # no finger pad within 3 cm of the spindle or the roll


def spindle_ends(sp):
    """World positions of the left (inner tube) and right dome tips, and the spindle's unit axis (right to left)."""
    q = sp.joints.get("inner", 0.0)
    left = sp.to_world((0.0, TIP + q, 0.0))
    right = sp.to_world((0.0, -TIP, 0.0))
    axis = (left - right) / max(np.linalg.norm(left - right), 1e-9)
    return left, right, axis


def seated(tip, s: float) -> bool:
    """The tip is in the socket on side s (+1 left, -1 right)."""
    inside = s * (tip[1] - HY) - GAP / 2
    off = max(abs(tip[0] - SX), abs(tip[2] - SZ))
    return bool(inside >= SEAT_MIN and off <= SEAT_OFF)


def on_spindle(obj, sp, length: float) -> bool:
    """The roll (or tube) of ``length`` is on the spindle: its axis near the spindle's and between its ends."""
    left, right, axis = spindle_ends(sp)
    c = obj.pos
    rel = c - right
    along = float(rel @ axis)
    radial = float(np.linalg.norm(rel - along * axis))
    obj_axis = obj.rotation()[:, 1]
    angle = math.acos(min(1.0, abs(float(obj_axis @ axis))))
    span = float(np.linalg.norm(left - right))
    return bool(radial <= ON_SPINDLE and angle <= ON_ANGLE and length / 2 - 0.005 <= along <= span - length / 2 + 0.005)


def tube_in_tray(tube) -> bool:
    inside = abs(tube.pos[0] - TRAY_X) < TRAY_LX / 2 and abs(tube.pos[1] - TRAY_Y) < TRAY_LY / 2
    return bool(inside and tube.pos[2] < TOP + TRAY_H + 0.01 and tube.resting())


_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(fingers=[], step=0, scored=None, last=(0.0, False))


@on_step
def _track(environment) -> None:
    _STATE["step"] += 1
    _STATE["fingers"] = [np.asarray(p, float) for p in finger_points(environment)]


_reset_state()


def hands_off(sp, roll) -> bool:
    """No finger pad within 3 cm of the spindle's axis segment or of the roll's outer surface."""
    left, right, _ = spindle_ends(sp)
    seg = left - right
    for p in _STATE["fingers"]:
        t = min(1.0, max(0.0, float((p - right) @ seg / (seg @ seg))))
        if np.linalg.norm(p - (right + t * seg)) < HANDS_OFF + OUT_R:
            return False
        rel = p - roll.pos
        ax = roll.rotation()[:, 1]
        along = float(rel @ ax)
        radial = float(np.linalg.norm(rel - along * ax))
        if abs(along) < ROLL_W / 2 + HANDS_OFF and radial < ROLL_R_OUT + HANDS_OFF:
            return False
    return True


def _update(objects) -> tuple[float, bool]:
    if _STATE["scored"] == _STATE["step"]:
        return _STATE["last"]
    _STATE["scored"] = _STATE["step"]
    sp, roll, tube = objects["spindle"], objects["roll"], objects["old_tube"]
    left, right, _ = spindle_ends(sp)
    n_seated = int(seated(left, 1.0)) + int(seated(right, -1.0))
    roll_on = on_spindle(roll, sp, ROLL_W) and roll.low[2] > TOP + 0.01
    tube_off = not on_spindle(tube, sp, TUBE_LEN)
    free = hands_off(sp, roll)
    still = sp.resting() and roll.resting(max_spin=1.0)
    if not (roll_on and tube_off):
        result = (0.0, False)
    elif n_seated == 2 and free and still and tube_in_tray(tube):
        result = (1.0, True)
    elif n_seated == 2 and free:
        result = (0.75, False)
    elif n_seated >= 1 and free:
        result = (0.5, False)
    else:
        result = (0.25, False)
    _STATE["last"] = result
    return result


def success(objects) -> bool:
    """The new roll hangs on the spindle, both spindle ends in their sockets, the empty tube in the tray, hands off,
    everything still, for 1 s."""
    return _update(objects)[1]


def score(objects) -> float:
    """0.25 new roll on the spindle (empty tube off it); 0.5 also one end seated; 0.75 both seated; 1.0 also the
    empty tube in the tray, still; seated credit only with both grippers clear."""
    return _update(objects)[0]


def demo():
    """Not a demo: no waypoints, so --policy yam_scripted holds both arms still (a do-nothing run through the official
    CLI). The full-score demo is demo.py."""
    return {}
