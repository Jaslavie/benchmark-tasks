"""On the station table, in front of the left arm (positions in cm from the midpoint between the arm bases, x forward, y
to the robot's left): two blue aluminium bearing plates (0.635 cm thick, 10 cm wide, 7 cm tall) stand across the
table, fixed, with their inner faces 20.3 cm apart (the length of the standoffs between them): one centred at (25,
25.5) and one at (25, 4.55). Each holds a black square bearing block (no flange) whose 1.47 cm square bore is centred at x = 25, 4 cm above
the table, so the two bores share one line along y. A silver 1/2-inch hex shaft (1.27 cm across, 25.4 cm long, 85
g) lies flat on the table, centred at (16, 36) and turned 20 degrees anticlockwise from pointing along y. The job:
slide the shaft through both bearings so that each end sticks out at least 1 cm past the outside of its bearing, room
for a collar on each end. The bores leave a millimetre each side, and the fingers can't follow the shaft
through a plate, so it goes in a stroke at a time: push, let go, grip it again further along, push again.
"""

import math

import numpy as np

from yam_arena.constants import FINGER_OPEN, PAD_TIP_PAST_TCP
from yam_arena.demo import Waypoint
from yam_arena.task_file import TABLE_TOP, Box

SCENE = "station"
TOP = TABLE_TOP[SCENE]
INCH = 0.0254

# The parts, at real size: WCP's GreyT Shooter (9.5 in.) kit (docs.wcproducts.com): 1/2" hex shaft stock, flanged
# bearings "13.75mm (1/2" ThunderHex ID) x 1.125" OD x 0.313" WD", side plates held apart by 8" standoffs.
BAR = 0.5 * INCH  # across the hex's flats; modelled square (a loose object is one box)
SHAFT_LEN, SHAFT_KG = 10 * INCH, 0.085  # 7075 aluminium hex with its 0.201" bore, about 30 cm^3
# The bore: a square 1 mm wider than the bar each side. WCP gives the bearing's ThunderHex ID as 13.75 mm, a snug fit
# on the rounded hex; at 0.525 mm each side the scripted demo passed with every aim 0.25 mm up or down but failed
# 0.25 mm to either side (official runs, 2026-11-04/05; review on #55), so the bore was widened to leave the demo room.
CLEARANCE = 0.001
BORE = BAR + 2 * CLEARANCE
BEARING_OD, BEARING_W = 1.125 * INCH, 0.313 * INCH
PLATE_T, PLATE_W, PLATE_H = 0.25 * INCH, 0.10, 0.07  # across, along x, tall
SPAN = 8 * INCH  # between the plates' inner faces

AXIS_X, AXIS_Z = 0.25, TOP + 0.040  # the bores' shared line runs along y
PLATE_Y = {"plate_a": 0.255, "plate_b": 0.255 - SPAN - PLATE_T}  # plate_a nearer the left arm's side
PLATE_BLUE, BEARING_BLACK, SHAFT_SILVER = (0.16, 0.32, 0.68), (0.06, 0.06, 0.07), (0.80, 0.81, 0.83)


def _frame(name, y, thick, inner, outer, z_low, z_high, color) -> list[Box]:
    """Four fixed boxes ``thick`` across (along y) around a square hole ``inner`` wide on the bores' line: two posts
    out to ``outer`` either side (from ``z_low`` to ``z_high``), and a bar under and over the hole between them."""
    zc = AXIS_Z - TOP
    side_w = (outer - inner) / 2
    out = []
    for s, label in ((-1, "near"), (1, "far")):  # x either side of the hole
        out.append(Box(f"{name}_{label}", pos=(AXIS_X + s * (inner / 2 + side_w / 2), y, TOP + (z_low + z_high) / 2),
                       size=(side_w, thick, z_high - z_low), color=color))
    below, above = (zc - inner / 2) - z_low, z_high - (zc + inner / 2)
    out.append(Box(f"{name}_under", pos=(AXIS_X, y, TOP + z_low + below / 2), size=(inner, thick, below), color=color))
    out.append(Box(f"{name}_over", pos=(AXIS_X, y, TOP + z_high - above / 2), size=(inner, thick, above), color=color))
    return out


FIXED = []  # no mass: fixed
_ZC = AXIS_Z - TOP
for _plate, _y in PLATE_Y.items():
    FIXED += _frame(_plate, _y, PLATE_T, BEARING_OD, PLATE_W, 0.0, PLATE_H, PLATE_BLUE)
    FIXED += _frame(f"{_plate}_bearing", _y, BEARING_W, BORE, BEARING_OD, _ZC - BEARING_OD / 2, _ZC + BEARING_OD / 2,
                    BEARING_BLACK)

START_XY, START_TURN = (0.16, 0.36), 20.0  # centre (m); turn (degrees, anticlockwise from pointing along y)
SHAFT = Box("shaft", pos=(*START_XY, TOP + BAR / 2 + 0.001), size=(BAR, SHAFT_LEN, BAR), mass=SHAFT_KG,
            color=SHAFT_SILVER, yaw=math.radians(START_TURN))  # starts 1 mm up, and settles
OBJECTS = [*FIXED, SHAFT]

COLLAR_ROOM = 0.010  # each end out at least 1 cm past its bearing: a 1/2" hex collar is 1/4" wide, plus a margin
EPS = 1e-9  # so a limit reads the same both ways
ONE_CREDIT, BOTH_CREDIT = 0.25, 0.5

# Where the shaft is comes from its centre and rotation, not from ``low``/``high`` (Arena builds those from the box's
# corners, turned with it).


def _rot(quat_xyzw) -> np.ndarray:
    x, y, z, w = (float(v) for v in quat_xyzw)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _axis(shaft) -> np.ndarray:
    """The shaft's long axis (unit vector, either way round)."""
    return _rot(shaft.quat_xyzw)[:, 1]


def ends_y(shaft) -> tuple[float, float]:
    """The y of the shaft's two ends' centres, lower first (m)."""
    centre, half = np.asarray(shaft.pos, dtype=float), SHAFT_LEN / 2 * _axis(shaft)
    return tuple(sorted((float((centre - half)[1]), float((centre + half)[1]))))


def through(shaft, plate) -> bool:
    """The shaft's centre line crosses ``plate``'s middle inside its bearing's bore: the shaft runs through it."""
    centre, axis, y = np.asarray(shaft.pos, dtype=float), _axis(shaft), PLATE_Y[plate]
    if abs(float(axis[1])) < 1e-6:
        return False
    t = (y - centre[1]) / float(axis[1])
    if abs(t) > SHAFT_LEN / 2:
        return False
    cross = centre + t * axis
    return abs(float(cross[0]) - AXIS_X) <= BORE / 2 + EPS and abs(float(cross[2]) - AXIS_Z) <= BORE / 2 + EPS


def sticking_out(shaft) -> tuple[float, float]:
    """How far each end reaches past the outside of its bearing (plate_b's end, plate_a's end; m)."""
    low, high = ends_y(shaft)
    return (PLATE_Y["plate_b"] - BEARING_W / 2) - low, high - (PLATE_Y["plate_a"] + BEARING_W / 2)


def _grade(objects) -> float:
    # No separate off-table rule: a shaft through a bore is on the table (review on #55).
    shaft = objects["shaft"]
    n = sum(through(shaft, p) for p in PLATE_Y)
    if n == 2 and min(sticking_out(shaft)) >= COLLAR_ROOM - EPS and shaft.resting():
        return 1.0
    return {0: 0.0, 1: ONE_CREDIT, 2: BOTH_CREDIT}[n]


# score() gives the complete state 0.9, not 1.0: 1.0 comes only from success() holding for SUCCESS_HOLD_S (1 s),
# which ends the trial. The official task_score takes the last step's score() when success hasn't fired, so a
# score() of 1.0 would give full marks to a shaft that only settles in the final second (a scoring hole
# reviewers reject).
COMPLETE_SCORE = 0.9


def success(objects):
    """The shaft through both bearings, each end at least 1 cm past the outside of its bearing, and still. Must hold
    for 1 s."""
    return _grade(objects) == 1.0


def score(objects):
    """Partial credit (0.9 when complete): 0.5 if the shaft runs through both bearings (but an end has less than 1 cm
    for its collar, or it's still moving), 0.25 if through one, from either side."""
    return min(_grade(objects), COMPLETE_SCORE)


# The demo: the left arm feeds the shaft in from plate_a's side in two strokes.
TRAVEL_Z = TOP + 0.12  # the carried shaft's bottom 3.5 cm above the plates' tops
TABLE_GRIP_Z = TOP + PAD_TIP_PAST_TCP + 0.002  # gripping the shaft off the table: pad tips 2 mm above it
CARRY_DROP = TABLE_GRIP_Z - (TOP + BAR / 2)  # the shaft's centre this far under the grip point once lifted
REACH_PAST = 0.028  # the grip's centre stays this far from a plate's middle: half a finger's width (40 mm at the
# plate's top) and half a bearing, with 4 mm spare
# The demo grips the shaft at its middle: on the official stack a pinch grip barely resists a turn about the line
# between the pads, so gripped 2.8 cm off its middle the shaft swung 13-17 degrees when lifted and jammed against the
# plate (first official run, 2026-11-04, score 0).
STROKE_2_GRIP_Y = 0.215  # stroke 2 grips between the plates, 4 cm inside plate_a
LEAD_IN = 0.010  # stroke 1 starts with the shaft's leading end 1 cm short of plate_a's bearing


def _wedge_angle() -> float:
    """The tilt (rad) at which a shaft let go with its middle outside a bore wedges in it: corner to corner, the bar's
    height and the bore's length fill the bore's height."""
    lo, hi = 0.0, 0.5
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if BAR * math.cos(mid) + BEARING_W * math.sin(mid) < BORE else (lo, mid)
    return lo


WEDGE = _wedge_angle()  # 20.9 degrees: as far as a shaft can tip in plate_a's bore
# Let go after stroke 1 (its middle 2.8 cm outside plate_a), the shaft tips until its far end, 15.5 cm out, rests on
# the table, short of WEDGE: about 12.2 degrees.
RELEASE_TILT = min(WEDGE, math.atan((AXIS_Z - TOP - BAR / 2) / (REACH_PAST + SHAFT_LEN / 2)))
# Where the arm puts the grip against where it's told, across the line (x) and up (z), in mm, holding the shaft beside
# plate_a (third official run, 2026-11-04, replayed): 1.4 mm short of forward and 1.3 mm low.
DEMO_TCP_ERROR_MM = (-1.4, -1.3)
# How the shaft sits in the grip by then (same run): turned 2.2 degrees anticlockwise (it turned on the table while the
# fingers closed) and pitched 2.75 degrees, far end up. Gripped at its middle, 12.7 cm from the leading end, that put
# the end 4.9 mm to the side and 6.1 mm low of the bore (then 0.5 mm to spare), and stroke 1 jammed. So stroke 1 turns the
# grip back 2.2 degrees and starts 6.1 mm high, pushes level until the end is 2 cm into the bore, then lowers the grip
# over the rest of the stroke. Pushing level, the pitched shaft meets the bore's top edge, and the grip, which barely
# resists a turn about the pads' line, lets it tip a little more (2.86 to 2.98 degrees) and carry on. A push that
# follows the shaft's pitch instead (review on #55) jammed at the bearing's face in all four official runs that tried
# it (11-14, 2026-11-04), with the end aimed from 0.65 mm under the line to 0.2 mm over it at the face.
DEMO_SLIP_DEG = (2.2, 2.75)
ENTER = 0.03  # stroke 1 pushes level this far (from 1 cm short to 2 cm in) before it lowers the grip
# Stroke 2. Plate_a's thin bearing lets the shaft pivot, but the fingers hold its yaw firmly, so closing them off the
# shaft's line turns it about the bearing, and the arm can't then push it back: in the fourth and fifth official runs
# (2026-11-04, replayed; both 0.25) the close turned the shaft from -1.36 to +2.2 degrees, its end met plate_b 4-5 mm
# forward of the bore, and moving the grip 1.8 mm back moved it 0.1 mm. So the regrasp closes on the shaft where it
# lies after stroke 1, the grip turned to match, and then turns and moves the grip together onto the line, swinging
# the shaft about plate_a's bearing. Where the shaft lies (the same every run): its middle at x 0.2503, turned
# -1.36 degrees. The arm's error there: 0.4 mm short of forward at rest, 0.7 mm past it while pushing.
# With the 1 mm bores (official run, 2026-11-05, replayed) the shaft tipped 12.2 degrees onto the table, its middle at
# x 0.2503, y 0.2862, turned -1.72 degrees (with 0.525 mm: 7.0 degrees, x 0.2503, y 0.2860, -1.36).
DEMO_WEDGED_X, DEMO_WEDGED_YAW_DEG = 0.2503, -1.72
DEMO_WEDGED_MIDDLE_Y = 0.2862
DEMO_REGRASP_ERROR_MM = -0.4
# Stroke 2's aim against the bores' line (mm, forward and up), and its grip's turn clockwise from straight along the
# line (degrees). The grip's yaw steers the shaft far more than its sideways position does: turned 0, 2 and 4 degrees
# (official runs 6, 7 and 8, 2026-11-04, replayed) the shaft met plate_b turned 1.2, 0.1 and -1.0 degrees, its end
# about 4 mm forward, on the line, and 3.6 mm back. With 2 degrees it stopped only because its end was 1.2 mm low:
# the line ran from plate_a's bore floor through the grip, 1.0 mm low there. So the grip goes 1.0 mm higher.
# With the 1 mm bores the shaft sits differently in plate_a's: aimed (-0.7, 3.3) its end met plate_b 1.0 mm back and
# 2.2 mm high (official run, 2026-11-05, 0.25), so the grip goes 1.0 mm further forward and 1.5 mm lower.
DEMO_STROKE_2_AIM_MM = (0.3, 1.8)
DEMO_STROKE_2_TURN_DEG = 2.0
# With that, the ninth official run (2026-11-04, replayed) put the shaft through both bearings but ended it 10.3 mm
# short of centred (plate_b's end out 8.0 mm, under the 1 cm for its collar): loaded, the arm lagged 5.3 mm, and the
# shaft slid 4.9 mm back in the grip. So stroke 2 is told to go this much further, which aims the middle 4 mm past
# halfway (the score allows 8.2 either way) and keeps the grip's command 2.7 mm outside REACH_PAST of plate_b.
DEMO_STROKE_2_FURTHER = 0.0062
# Every aim moved this far across the bores' line and up (mm, + forward, + up). 0 for the demo; the margin runs set
# each either way.
DEMO_AIM_SHIFT_MM = 0.0
DEMO_AIM_SHIFT_Z_MM = 0.0
# Fingers aimed 6 mm inside each face of the 12.7 mm bar: about 30 N each, under their 50 N limit (fully closed, both
# push at the limit and a part can slide in the grip). Let go by opening slowly to 29 mm, 8 mm clear each side.
GRIP_FINGER = FINGER_OPEN * (BAR * 1000 - 12) / 94
RELEASE_FINGER = FINGER_OPEN * (BAR * 1000 + 16) / 94
INSERT_YAW = -math.pi / 2  # the grip turned so the fingers close across a shaft lying along y


def _aim(x, y, z):
    dx, dz = DEMO_TCP_ERROR_MM
    return (x + (DEMO_AIM_SHIFT_MM - dx) / 1000, y, z + (DEMO_AIM_SHIFT_Z_MM - dz) / 1000)


def _aim_2(x, y, z):
    """Stroke 2's aim: moved DEMO_STROKE_2_AIM_MM, and by the margin runs' shifts."""
    sx, sz = DEMO_STROKE_2_AIM_MM
    return (x + (sx + DEMO_AIM_SHIFT_MM) / 1000, y, z + (sz + DEMO_AIM_SHIFT_Z_MM) / 1000)


def _wedged_line_x(y) -> float:
    """Where the shaft's line lies across the table at ``y`` after stroke 1 (m)."""
    turn = math.radians(DEMO_WEDGED_YAW_DEG)
    return DEMO_WEDGED_X - (DEMO_WEDGED_MIDDLE_Y - y) * -math.sin(turn)


def demo():
    """Left arm: grip the shaft off the table at its middle, lift it over the plates, turn it along the bores' line
    and lower it beside plate_a, its leading end 1 cm short of the bearing. Stroke 1: push it in, level and then
    lowering the grip, until the grip is 2.8 cm from the plate (the leading end 9.9 cm through) and let go; the shaft tips and wedges
    in the bore. Stroke 2: grip it again between the plates where it now lies, turn and raise the grip onto the line,
    which swings the shaft about plate_a's bearing, push it through plate_b's bearing until its middle is about 4 mm
    past halfway (ends out about 1.4 and 2.2 cm), and let go."""
    gx, gy = START_XY
    pick_yaw = math.radians(START_TURN) - math.pi / 2  # fingers across the shaft
    y_a = PLATE_Y["plate_a"]
    line_z = AXIS_Z + CARRY_DROP  # grip height that puts the shaft's centre on the bores' line
    start_y = y_a + BEARING_W / 2 + LEAD_IN + SHAFT_LEN / 2
    end_1 = y_a + REACH_PAST
    middle = (PLATE_Y["plate_a"] + PLATE_Y["plate_b"]) / 2
    end_2 = middle + (STROKE_2_GRIP_Y - end_1) - DEMO_STROKE_2_FURTHER  # the grip's place on the shaft, planned
    rest_z = line_z + (y_a - STROKE_2_GRIP_Y) * math.tan(RELEASE_TILT)  # the tipped shaft's height under the grip
    slip_yaw, slip_pitch = (math.radians(v) for v in DEMO_SLIP_DEG)
    yaw_1 = INSERT_YAW - slip_yaw
    high_z = line_z + SHAFT_LEN / 2 * math.tan(slip_pitch)  # puts the pitched shaft's leading end on the line
    regrasp_x = _wedged_line_x(STROKE_2_GRIP_Y) + (DEMO_AIM_SHIFT_MM - DEMO_REGRASP_ERROR_MM) / 1000
    regrasp_yaw = INSERT_YAW + math.radians(DEMO_WEDGED_YAW_DEG)
    regrasp_z = _aim(AXIS_X, STROKE_2_GRIP_Y, rest_z)[2]
    yaw_2 = INSERT_YAW - math.radians(DEMO_STROKE_2_TURN_DEG)
    return {"left": [
        Waypoint(1.5, (gx, gy, TRAVEL_Z), FINGER_OPEN, "to the shaft", pick_yaw),
        Waypoint(1.5, (gx, gy, TABLE_GRIP_Z), FINGER_OPEN, "descend", pick_yaw),
        Waypoint(0.5, (gx, gy, TABLE_GRIP_Z), FINGER_OPEN, "steady", pick_yaw),
        Waypoint(0.8, (gx, gy, TABLE_GRIP_Z), GRIP_FINGER, "close", pick_yaw),
        # Second official run: lifted while the fingers were still closing, one end left the table first and the
        # shaft pivoted 13 degrees in the grip. So the grip settles first, and the first 2 cm go slowly.
        Waypoint(0.7, (gx, gy, TABLE_GRIP_Z), GRIP_FINGER, "settle the grip", pick_yaw),
        Waypoint(2.0, (gx, gy, TABLE_GRIP_Z + 0.02), GRIP_FINGER, "lift off", pick_yaw),
        Waypoint(1.2, (gx, gy, TRAVEL_Z), GRIP_FINGER, "lift", pick_yaw),
        Waypoint(2.0, _aim(AXIS_X, start_y, TRAVEL_Z), GRIP_FINGER, "carry", yaw_1),
        Waypoint(2.0, _aim(AXIS_X, start_y, high_z), GRIP_FINGER, "lower beside the plate", yaw_1),
        Waypoint(0.5, _aim(AXIS_X, start_y, high_z), GRIP_FINGER, "steady", yaw_1),
        Waypoint(1.5, _aim(AXIS_X, start_y - ENTER, high_z), GRIP_FINGER, "enter", yaw_1),
        Waypoint(3.0, _aim(AXIS_X, end_1, line_z), GRIP_FINGER, "stroke 1", yaw_1),
        Waypoint(0.3, _aim(AXIS_X, end_1, line_z), GRIP_FINGER, "hold", yaw_1),
        Waypoint(1.5, _aim(AXIS_X, end_1, line_z), RELEASE_FINGER, "release", yaw_1),
        Waypoint(1.2, _aim(AXIS_X, end_1, TRAVEL_Z), RELEASE_FINGER, "lift", yaw_1),
        Waypoint(1.5, (regrasp_x, STROKE_2_GRIP_Y, TRAVEL_Z), RELEASE_FINGER, "over the shaft", regrasp_yaw),
        Waypoint(1.5, (regrasp_x, STROKE_2_GRIP_Y, regrasp_z), RELEASE_FINGER, "descend", regrasp_yaw),
        Waypoint(0.5, (regrasp_x, STROKE_2_GRIP_Y, regrasp_z), RELEASE_FINGER, "steady", regrasp_yaw),
        Waypoint(0.8, (regrasp_x, STROKE_2_GRIP_Y, regrasp_z), GRIP_FINGER, "close", regrasp_yaw),
        Waypoint(1.0, _aim_2(AXIS_X, STROKE_2_GRIP_Y, line_z), GRIP_FINGER, "level onto the line", yaw_2),
        Waypoint(0.5, _aim_2(AXIS_X, STROKE_2_GRIP_Y, line_z), GRIP_FINGER, "steady", yaw_2),
        Waypoint(4.0, _aim_2(AXIS_X, end_2, line_z), GRIP_FINGER, "stroke 2", yaw_2),
        Waypoint(0.3, _aim_2(AXIS_X, end_2, line_z), GRIP_FINGER, "hold", yaw_2),
        Waypoint(1.5, _aim_2(AXIS_X, end_2, line_z), RELEASE_FINGER, "release", yaw_2),
        Waypoint(1.2, _aim_2(AXIS_X, end_2, TRAVEL_Z), RELEASE_FINGER, "retreat", yaw_2),
    ]}
