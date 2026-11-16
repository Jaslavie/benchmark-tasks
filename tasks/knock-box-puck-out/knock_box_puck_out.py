"""On the station table, in front of the right arm, an espresso portafilter sits basket-up on a fixed riser holder (13
cm tall): a steel basket cup (64 mm across, 25 mm deep) with a 120 mm handle running back toward the arms, the
handle's end sticking out past the holder. A spent coffee puck (a 54 mm disc, 14 mm thick, dark brown) is packed on
the basket's floor. To the right of the holder is a knock box: an open-topped bin, 13 cm square inside and 15 cm
tall, with a black rubber-sleeved bar across its opening near its front wall. The job is to empty the basket into
the knock box and put the portafilter back on its stand.

The puck is stuck in the basket (a breakable bond, 1.5 N). Its own weight with the basket turned over (0.34 N), the
carry and the turn, or setting the basket down slowly on the bar cannot tear it loose (both tried in sim): only a
sharp knock that stops the basket in a few milliseconds (about 0.35 m/s in the demo) pulls the 35 g puck off with
its own momentum. The basket's rim is 3 mm proud of the handle, so a basket-down knock lands on the steel rim, not
the wooden handle. There is no room for the finger pads to get at the puck from the side (2 mm all round), and
anything poked into the basket (digging the puck out, which bends real baskets) scores 0 for the trial. A knock
anywhere but over the knock box (on the table, the holder), or letting go of the portafilter when the puck comes
loose, also scores 0 for the trial.

Run it:  inspect-robots run --embodiment yam_arena -E task=tasks/knock-box-puck-out/knock_box_puck_out.py \\
           -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \\
           --instruction "<the instruction in task.json>" --scorer task_score ...
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mechkit as mk  # noqa: E402  (per-episode memory, gripper TCPs, the puck's breakable bond)

from yam_arena import demo as _demo  # noqa: E402
from yam_arena.constants import FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]

# Frame: x forward from the arm bases, y left, z up, metres. Right arm base (0, -0.305, 0.75).
# -- the portafilter (kit part; frame origin = the basket's centre at the underside of its floor, handle along -x) ---
PF_KG, PUCK_KG = 0.180, 0.035
A_OUT, A_IN = 0.032, 0.029  # basket octagon: outer / inner apothem (64 / 58 mm across)
FLOOR_T = 0.003
WALL_H = 0.025  # rim at z 0.028
RIM_Z = FLOOR_T + WALL_H
SPOUT = (0.024, 0.024, 0.008)  # spout block under the floor, z -0.008 .. 0
HANDLE = (0.120, 0.022, 0.022)  # x -0.032 .. -0.152, z 0.003 .. 0.025: 3 mm below the rim
HANDLE_X0 = -A_OUT
HANDLE_ZC = 0.014
PUCK_A = 0.027  # puck octagon apothem (54 mm across): 2 mm clear of the basket wall all round
PUCK_T = 0.014  # z 0.003 .. 0.017 on the floor
PUCK_BREAK_N = 1.5  # the packed puck's hold in the basket

# -- the stand (start and finish) ----------------------------------------------------------------------------------
STAND_XY = (0.420, -0.050)  # the basket's centre on the stand
PF_Z = TOP + 0.130  # the floor's underside on the stand (a riser holder: the handle at the arm's level)
PED = (0.050, 0.050, PF_Z - SPOUT[2] - TOP)  # pedestal under the spout
REST_X = (-0.100, -0.075)  # handle rest under the handle (local x), top = the handle's underside
REST_W = 0.030

# -- the knock box -------------------------------------------------------------------------------------------------
BOX_XY = (0.455, -0.220)
BOX_IN = 0.130  # inner width (square)
BOX_WALL = 0.008
BOX_H = 0.150
BAR = (0.022, BOX_IN + 2 * BOX_WALL, 0.022)  # the knock bar, along y across the opening
BAR_X = BOX_XY[0] - 0.040
BAR_TOP = TOP + 0.162
BOX_FLOOR_T = 0.004

BLACK, STEEL, BROWN, GREY = (0.07, 0.07, 0.08), (0.62, 0.64, 0.67), (0.24, 0.15, 0.08), (0.35, 0.36, 0.38)
WOOD = (0.45, 0.30, 0.18)


def _octagon(apothem, thick, z, color=None):
    """An octagonal slab (flats on x and y) as the union of four rectangles."""
    s = 2.0 * apothem * math.tan(math.pi / 8)
    return [mk.box((2 * apothem, s, thick), at=(0.0, 0.0, z), color=color, yaw=k * math.pi / 4) for k in range(4)]


def _walls():
    side = 2.0 * A_OUT * math.tan(math.pi / 8) + 0.001
    r = (A_IN + A_OUT) / 2
    out = []
    for k in range(8):
        t = k * math.pi / 4
        out.append(mk.box((A_OUT - A_IN, side, WALL_H), at=(r * math.cos(t), r * math.sin(t), FLOOR_T + WALL_H / 2),
                          yaw=t))
    return out


kit = mk.Kit()
kit.part("portafilter", (*STAND_XY, PF_Z), [
    *_octagon(A_OUT, FLOOR_T, FLOOR_T / 2),
    *_walls(),
    mk.box(SPOUT, at=(0.0, 0.0, -SPOUT[2] / 2), color=BLACK),
    mk.box(HANDLE, at=(HANDLE_X0 - HANDLE[0] / 2, 0.0, HANDLE_ZC), color=WOOD),
], mass=PF_KG, color=STEEL)
kit.part("puck", (*STAND_XY, PF_Z + FLOOR_T + PUCK_T / 2), _octagon(PUCK_A, PUCK_T, 0.0), mass=PUCK_KG, color=BROWN)
# the packed puck: a 1.5 N pull (at the puck's centre) tears it off the basket floor
kit.bond("puck_stuck", "portafilter", "puck", break_force=PUCK_BREAK_N)

_bx, _by = BOX_XY
_wz = TOP + BOX_H / 2
FIXTURES = [
    Box("stand_pedestal", pos=(*STAND_XY, TOP + PED[2] / 2), size=PED, color=GREY),
    Box("stand_handle_rest", pos=(STAND_XY[0] + sum(REST_X) / 2, STAND_XY[1], TOP + (PF_Z + HANDLE_ZC - HANDLE[2] / 2
                                                                                       - TOP) / 2),
        size=(REST_X[1] - REST_X[0], REST_W, PF_Z + HANDLE_ZC - HANDLE[2] / 2 - TOP), color=GREY),
    Box("box_floor", pos=(_bx, _by, TOP + BOX_FLOOR_T / 2), size=(BOX_IN, BOX_IN, BOX_FLOOR_T), color=BLACK),
    Box("box_front", pos=(_bx - BOX_IN / 2 - BOX_WALL / 2, _by, _wz), size=(BOX_WALL, BOX_IN + 2 * BOX_WALL, BOX_H),
        color=BLACK),
    Box("box_back", pos=(_bx + BOX_IN / 2 + BOX_WALL / 2, _by, _wz), size=(BOX_WALL, BOX_IN + 2 * BOX_WALL, BOX_H),
        color=BLACK),
    Box("box_left", pos=(_bx, _by + BOX_IN / 2 + BOX_WALL / 2, _wz), size=(BOX_IN, BOX_WALL, BOX_H), color=BLACK),
    Box("box_right", pos=(_bx, _by - BOX_IN / 2 - BOX_WALL / 2, _wz), size=(BOX_IN, BOX_WALL, BOX_H), color=BLACK),
    Box("knock_bar", pos=(BAR_X, _by, BAR_TOP - BAR[2] / 2), size=BAR, color=BLACK),
]
OBJECTS = [*kit.objects(), *FIXTURES]

SUCCESS_HOLD_S = 1.0
SCORE_OVER = "end"

# -- scoring ---------------------------------------------------------------------------------------------------
DT = 0.05  # control step at 20 Hz
OVER_BOX_MARGIN = 0.010  # the knock counts as over the box with the basket's centre inside its walls + 1 cm
HELD_MARGIN = 0.030  # a TCP within 3 cm of the portafilter when the puck came loose: it was held, not thrown
DIG_R, DIG_Z = 0.025, (FLOOR_T, RIM_Z + 0.013)  # a TCP this close to the basket's axis, at these heights: poking in
STAND_TOL = 0.015
STILL_MM, STILL_DEG = 1.0, 1.0  # pose change per control step that still counts as at rest
DEBUG = False
_TEST_TCPS = None  # offline self-test: the TCPs to use instead of the simulator's
M = "knock"


def _mem() -> dict:
    return mk.MEMORY.setdefault(M, {"freed": False, "bad_knock": False, "dug": False, "last": None, "still": False,
                                    "v": 0.0, "n": 0})


def _tcps():
    return mk.tcps() if _TEST_TCPS is None else [np.asarray(t, float) for t in _TEST_TCPS]


def _near(st, margin) -> bool:
    lo, hi = np.asarray(st.low) - margin, np.asarray(st.high) + margin
    return any(bool(np.all(t >= lo) and np.all(t <= hi)) for t in _tcps())


def released(st, margin=0.025) -> bool:
    """Neither gripper's TCP within ``margin`` (2.5 cm) of the object's bounding box."""
    return not _near(st, margin)


def basket_centre(objects) -> np.ndarray:
    return kit.point(objects, "portafilter", (0.0, 0.0, FLOOR_T))


def basket_up(objects) -> np.ndarray:
    return kit.point(objects, "portafilter", (0.0, 0.0, 1.0)) - kit.point(objects, "portafilter", (0.0, 0.0, 0.0))


def _in_box(p, margin=0.0) -> bool:
    return bool(abs(p[0] - BOX_XY[0]) < BOX_IN / 2 + margin and abs(p[1] - BOX_XY[1]) < BOX_IN / 2 + margin)


def puck_in_box(objects) -> bool:
    """The puck's centre inside the knock box's walls and below the bar's top."""
    pk = objects["puck"]
    return _in_box(pk.pos) and float(pk.pos[2]) < BAR_TOP


def on_stand(objects) -> bool:
    """The portafilter back on its stand: the basket's centre within 1.5 cm of the stand's spot, its floor down on
    the pedestal (within 5 mm), basket up (tilt under 10 deg)."""
    o = kit.point(objects, "portafilter", (0.0, 0.0, 0.0))
    up = basket_up(objects)
    return bool(math.hypot(o[0] - STAND_XY[0], o[1] - STAND_XY[1]) < STAND_TOL and abs(o[2] - PF_Z) < 0.005
                and up[2] > math.cos(math.radians(10.0)))


def _pose_key(objects, name):
    pos, q = kit.pose(objects, name)
    return np.asarray(pos, float), np.asarray(q, float)


def _moved(a, b) -> bool:
    (p0, q0), (p1, q1) = a, b
    ang = 2 * math.degrees(math.acos(min(1.0, abs(float(np.dot(q0, q1))))))
    return bool(np.linalg.norm(p1 - p0) * 1000 > STILL_MM or ang > STILL_DEG)


def _update(objects) -> None:
    """One control step of the trial's history (every step in sim; from score() offline)."""
    mem = _mem()
    if mem.get("frame") is objects:  # offline, success() and score() see the same frame: count it once
        return
    mem["frame"] = objects
    mem["n"] += 1
    pose = {n: _pose_key(objects, n) for n in ("portafilter", "puck")}
    speed = 0.0
    if mem["last"] is not None:
        speed = float(np.linalg.norm(pose["portafilter"][0] - mem["last"]["portafilter"][0])) / DT
        mem["still"] = not any(_moved(mem["last"][n], pose[n]) for n in pose)
    # poking into the basket: a TCP near the basket's axis, between its floor and just above its rim
    pf_t = kit._pose_T(objects, "portafilter")
    for t in _tcps():
        loc = pf_t[:3, :3].T @ (np.asarray(t, float) - pf_t[:3, 3])
        if math.hypot(loc[0], loc[1]) < DIG_R and DIG_Z[0] < loc[2] < DIG_Z[1]:
            mem["dug"] = True
    if not mem["freed"] and kit.broken(objects, "puck_stuck"):
        mem["freed"] = True
        c = basket_centre(objects)
        over = _in_box(c, OVER_BOX_MARGIN)
        held = _near(objects["portafilter"], HELD_MARGIN)
        mem["bad_knock"] = not (over and held)
        mem["freed_at"] = (mem["n"], [round(float(v), 4) for v in c], over, held)
        if DEBUG:
            print(f"[knock] puck freed at step {mem['n']}: basket centre {np.round(c, 3)} over box {over} held {held}",
                  flush=True)
    if DEBUG and (mem["v"] > 0.08 or speed > 0.08):
        low = float(objects["portafilter"].low[2]) - TOP
        print(f"[knock] step {mem['n']} v {mem['v']:.3f}->{speed:.3f} low_z {low:.4f} freed {mem['freed']} "
              f"tcps {[np.round(t, 3).tolist() for t in _tcps()]}", flush=True)
    mem["v"] = speed
    mem["last"] = pose


kit.on_step(_update)


def score(objects) -> float:
    """0 for the rest of the trial once anything has been poked into the basket, or once the puck came loose with
    the basket's centre outside the knock box (+1 cm) or with neither gripper within 3 cm of the portafilter.
    Otherwise: 0.5 with the puck's centre inside the knock box below the bar's top; 1.0 when, on top of that, the
    portafilter is back on its stand basket-up, both are let go (grippers 2.5 cm clear) and nothing has moved over
    the last control step. 0 while the puck is still in the basket or anywhere else."""
    if mk._EP.get("env") is None:  # offline; in sim the per-step rule has already run
        _update(objects)
    mem = _mem()
    if mem["dug"] or mem["bad_knock"] or not mem["freed"]:
        return 0.0
    if not puck_in_box(objects):
        return 0.0
    if (on_stand(objects) and mem["still"] and released(objects["portafilter"]) and released(objects["puck"])):
        return 1.0
    return 0.5


def success(objects) -> bool:
    return score(objects) >= 1.0 - 1e-9


def distance(objects) -> float:
    """Metres to go: the puck's centre to the knock box's centre (horizontal), plus the basket's centre to the
    stand's spot once the puck is out."""
    pk = objects["puck"]
    d = math.hypot(float(pk.pos[0]) - BOX_XY[0], float(pk.pos[1]) - BOX_XY[1])
    o = kit.point(objects, "portafilter", (0.0, 0.0, 0.0))
    return float(d + math.hypot(o[0] - STAND_XY[0], o[1] - STAND_XY[1]) + abs(o[2] - PF_Z))


# -- tilted grips for the scripted demo ----------------------------------------------------------------------------
# yam_arena's Waypoint carries only a yaw about the straight-down grip. The demo needs the gripper pointing forward
# and rolled about its own axis, so a Waypoint's yaw may be a Tilt: a float (0.0) that also carries the full TCP
# rotation, which ArmPlan uses instead (an in-memory patch of yam_arena.demo.grip_rotation; plain yaws unchanged).


class Tilt(float):
    def __new__(cls, rot):
        self = super().__new__(cls, 0.0)
        self.rot = np.asarray(rot, dtype=float)
        return self


def _patch_grip_rotation() -> None:
    if getattr(_demo, "_tilt_patch", False):
        return
    base = _demo.grip_rotation

    def grip_rotation(yaw=0.0):
        rot = getattr(yaw, "rot", None)
        return rot.copy() if rot is not None else base(yaw)

    _demo.grip_rotation = grip_rotation
    _demo._tilt_patch = True


_patch_grip_rotation()


def forward_grip(roll_deg: float, pitch_deg: float = 0.0) -> Tilt:
    """The gripper pointing forward (+x), pitched ``pitch_deg`` down, rolled ``roll_deg`` about its own axis; at
    roll 0 the fingers close along world y, at roll -90 / +90 they close vertically."""
    p = math.radians(pitch_deg)
    a = np.array([math.cos(p), 0.0, -math.sin(p)])
    c = mk.axis_angle(a, math.radians(roll_deg)) @ np.array([0.0, -1.0, 0.0])
    return Tilt(np.column_stack([c, np.cross(a, c), a]))


# -- the scripted demo ------------------------------------------------------------------------------------------
# Right arm only, open loop from the scene's geometry. The gripper points forward and holds the handle's end with
# the fingers closing vertically (roll -90 deg). It lifts the portafilter off the stand, carries it over the knock
# box, rolls it basket-down (roll +90), knocks the basket's rim down onto the bar, rolls it back and sets it on the
# stand.

TCP_X = -0.123  # the TCP on the handle (local x): pad tips 13 mm farther, at x -0.110; pads back to the handle's end
TCP_LOCAL = np.array([TCP_X, 0.0, HANDLE_ZC])
SQUEEZE = 0.008
GRIP = -(HANDLE[2] - SQUEEZE) / 2
HANG = 0.004  # a held part rides a few mm low (measured per task in sim)
TRAVEL = TOP + 0.250
UP, DOWN = -90.0, 90.0  # roll with the basket up / turned over
WIND = 0.060  # the rim this far above the bar before the knock
DRIVE = 0.020  # commanded this far past the bar's top: the knock is mid-stroke, at speed
STRIKE_X = BAR_X + (A_IN + A_OUT) / 2  # the basket's near wall over the bar's centre line


def _phases():
    """(duration, label, right TCP, finger, roll)."""
    o, g = FINGER_OPEN, GRIP
    sx, sy = STAND_XY
    tz = PF_Z + HANDLE_ZC  # TCP height with the portafilter on the stand
    gx = sx + TCP_X
    kx, ky = STRIKE_X + TCP_X, BOX_XY[1]
    # basket down: the rim is 14 mm below the handle's centre line (the TCP)
    kz = BAR_TOP + (RIM_Z - HANDLE_ZC)
    return [
        (2.0, "behind the handle's end", (gx - 0.060, sy, tz + 0.010), o, UP),
        (1.5, "slide in round the handle", (gx, sy, tz), o, UP),
        (0.5, "settle", (gx, sy, tz), o, UP),
        (1.0, "grip the handle", (gx, sy, tz), g, UP),
        (0.5, "settle closed", (gx, sy, tz), g, UP),
        (1.2, "lift off the stand", (gx, sy, tz + 0.030), g, UP),
        (1.3, "up", (gx, sy, TRAVEL), g, UP),
        (2.0, "over the knock box", (kx, ky, TRAVEL), g, UP),
        (1.5, "roll the basket on its side", (kx, ky, TRAVEL), g, 0.0),
        (1.5, "basket down", (kx, ky, TRAVEL), g, DOWN),
        (1.2, "down to just above the bar", (kx, ky, kz + WIND + HANG), g, DOWN),
        (0.6, "steady", (kx, ky, kz + WIND + HANG), g, DOWN),
        (0.30, "knock the rim on the bar", (kx, ky, kz - DRIVE), g, DOWN),
        (0.4, "hold", (kx, ky, kz - DRIVE), g, DOWN),
        (0.8, "up off the bar", (kx, ky, kz + WIND), g, DOWN),
        (0.6, "steady", (kx, ky, kz + WIND), g, DOWN),
        (1.3, "up", (kx, ky, TRAVEL), g, DOWN),
        (1.5, "roll back", (kx, ky, TRAVEL), g, 0.0),
        (1.5, "basket up", (kx, ky, TRAVEL), g, UP),
        (2.0, "back over the stand", (gx, sy, tz + 0.040 + HANG), g, UP),
        (1.5, "set it down", (gx, sy, tz + 0.002 + HANG), g, UP),
        (0.5, "settle", (gx, sy, tz + 0.002 + HANG), g, UP),
        (1.0, "let go", (gx, sy, tz + 0.002 + HANG), o, UP),
        (1.5, "back out", (gx - 0.070, sy, tz + 0.002 + HANG), o, UP),
        (1.0, "up and clear", (gx - 0.090, sy, tz + 0.080), o, UP),
    ]


def demo():
    """Right arm: grip the handle's end, lift the portafilter off its stand, carry it over the knock box, turn the
    basket down, knock its rim once on the bar (the puck drops into the box), turn it back and set it on the
    stand."""
    return {"right": [Waypoint(d, tuple(float(v) for v in p), f, label, forward_grip(r))
                      for d, label, p, f, r in _phases()]}


# -- offline self-test ------------------------------------------------------------------------------------------


def self_test():
    """The scoring rules on hand-built sequences of states (the kit's forward kinematics poses the parts)."""
    global _TEST_TCPS
    flip = mk.mat_to_quat(mk.axis_angle((1, 0, 0), math.pi))  # basket down (rolled about the handle)

    def pf_at(x, y, z, quat=(0, 0, 0, 1)):
        return (np.array([x, y, z]), np.asarray(quat, float))

    def frame(pf=None, puck=None):
        moved = {}
        if pf is not None:
            moved["portafilter"] = pf
        if puck is not None:
            moved["puck"] = puck
        return kit.fake_states(moved=moved)

    kx = STRIKE_X
    over_box_down = pf_at(kx, BOX_XY[1], BAR_TOP + RIM_Z, flip)  # basket down, rim on the bar
    stand = pf_at(*STAND_XY, PF_Z)
    puck_on_floor = (np.array([BOX_XY[0], BOX_XY[1], TOP + BOX_FLOOR_T + PUCK_T / 2]), np.array([0, 0, 0, 1.0]))

    def tcp_on(pf):  # the TCP where the demo holds the handle
        t = mk._T(mk.quat_to_mat(pf[1]), pf[0])
        return (t[:3, :3] @ TCP_LOCAL + t[:3, 3]).tolist()

    s0 = frame()
    held_down = frame(over_box_down)
    knocked = frame(over_box_down, puck_on_floor)
    lifted_pf = pf_at(kx, BOX_XY[1], TRAVEL - HANDLE_ZC)
    lifted_back = frame(lifted_pf, puck_on_floor)
    home = frame(stand, puck_on_floor)
    home_tilted = frame(pf_at(*STAND_XY, PF_Z, mk.mat_to_quat(mk.axis_angle((0, 1, 0), math.radians(20)))),
                        puck_on_floor)
    # knocked on the table next to the stand (basket down, puck on the table)
    table_down = pf_at(0.300, -0.050, TOP + RIM_Z, flip)
    puck_on_table = (np.array([0.300 + 0.0, -0.050, TOP + PUCK_T / 2]), np.array([0, 0, 0, 1.0]))
    # puck tipped out over the stand / knocked on the stand's pedestal
    stand_down = pf_at(*STAND_XY, PF_Z + 0.040, flip)
    puck_on_stand = (np.array([*STAND_XY, PF_Z + 0.010]), np.array([0, 0, 0, 1.0]))
    # dropped the portafilter into the box (thrown): it lands in the box with the puck
    pf_in_box = pf_at(BOX_XY[0], BOX_XY[1], TOP + 0.030, flip)
    puck_in_box_low = (np.array([BOX_XY[0] + 0.01, BOX_XY[1], TOP + 0.012]), np.array([0, 0, 0, 1.0]))
    # puck still in the basket but the portafilter set on the stand: the bond never broke
    away = (0.20, -0.30, TOP + 0.30)  # a TCP well away from everything
    G = tcp_on

    cases = {
        "start": ([(s0, [away])], 0.0, False),
        "carried over the box and turned basket down, puck stuck": ([(s0, [G(stand)]), (held_down, [G(over_box_down)])],
                                                                     0.0, False),
        "knocked on the bar: puck in the box, still held": (
            [(s0, [G(stand)]), (held_down, [G(over_box_down)]), (knocked, [G(over_box_down)])], 0.5, False),
        "knocked, lifted back up": ([(held_down, [G(over_box_down)]), (knocked, [G(over_box_down)]),
                                     (lifted_back, [G(lifted_pf)])],
                                    0.5, False),
        "full demo: knocked, back on the stand, let go, still": (
            [(s0, [G(stand)]), (held_down, [G(over_box_down)]), (knocked, [G(over_box_down)]), (home, [G(stand)]),
             (home, [away]), (home, [away])], 1.0, True),
        "back on the stand but still gripped": (
            [(held_down, [G(over_box_down)]), (knocked, [G(over_box_down)]), (home, [G(stand)]), (home, [G(stand)])],
            0.5, False),
        "back on the stand but leaning 20 deg": (
            [(held_down, [G(over_box_down)]), (knocked, [G(over_box_down)]), (home_tilted, [away]),
             (home_tilted, [away])], 0.5, False),
        "let go on the stand, still moving": (
            [(held_down, [G(over_box_down)]), (knocked, [G(over_box_down)]), (lifted_back, [away]), (home, [away])],
            0.5, False),
        "naive: set straight back on the stand (puck never out)": ([(s0, [G(stand)]), (s0, [away]), (s0, [away])], 0.0,
                                                                   False),
        "knocked on the table: latched 0 even if the puck is then put in the box": (
            [(frame(table_down), [G(table_down)]), (frame(table_down, puck_on_table), [G(table_down)]),
             (frame(stand, puck_on_floor), [away]), (frame(stand, puck_on_floor), [away])], 0.0, False),
        "knocked on the stand itself: latched 0": (
            [(frame(stand_down), [G(stand_down)]), (frame(stand_down, puck_on_stand), [G(stand_down)]),
             (home, [away]), (home, [away])], 0.0, False),
        "portafilter let go over the box (thrown in): latched 0": (
            [(held_down, [G(over_box_down)]), (frame(pf_in_box, puck_in_box_low), [away]),
             (home, [away]), (home, [away])], 0.0, False),
        "digging the puck out (a TCP inside the basket): latched 0": (
            [(s0, [kit.point(s0, "portafilter", (0.0, 0.0, 0.025)).tolist()]), (held_down, [G(over_box_down)]),
             (knocked, [G(over_box_down)]), (home, [away]), (home, [away])], 0.0, False),
        "a TCP above the rim (not in the basket) is fine": (
            [(s0, [kit.point(s0, "portafilter", (0.0, 0.0, 0.060)).tolist()]), (held_down, [G(over_box_down)]),
             (knocked, [G(over_box_down)]), (home, [away]), (home, [away])], 1.0, True),
        "puck out but outside the box (bounced over the wall)": (
            [(held_down, [G(over_box_down)]),
             (frame(over_box_down, (np.array([BOX_XY[0], BOX_XY[1] + 0.12, TOP + 0.007]), np.array([0, 0, 0, 1.0]))),
              [G(over_box_down)]), (frame(stand, (np.array([BOX_XY[0], BOX_XY[1] + 0.12, TOP + 0.007]),
                                                  np.array([0, 0, 0, 1.0]))), [away])], 0.0, False),
        "a new trial after a table knock starts clean": (
            [(frame(table_down, puck_on_table), [G(table_down)]), None, (held_down, [G(over_box_down)]),
             (knocked, [G(over_box_down)]), (home, [away]), (home, [away])], 1.0, True),
    }
    n = 0
    try:
        for name, (frames, want, want_success) in cases.items():
            mk.new_episode()
            got = got_success = None
            for f in frames:
                if f is None:
                    mk.new_episode()
                    continue
                states, tcps = dict(f[0]), f[1]  # a fresh dict per step, as the simulator gives
                _TEST_TCPS = tcps
                got_success, got = success(states), score(states)
            assert abs(got - want) < 1e-9 and got_success == want_success, (name, got, got_success, want, want_success,
                                                                           _mem())
            n += 1
    finally:
        _TEST_TCPS = None
        mk.new_episode()
    return n


if __name__ == "__main__":
    # Offline checks, no GPU: the scoring rules on hand-built states, then the arm's plan replayed through the
    # kinematics at 20 Hz, every joint inside the URDF limits with an IK error under 0.5 mm.
    from yam_arena import ik, station
    from yam_arena.demo import ArmPlan
    from yam_arena.task_file import load

    print(f"scoring: {self_test()} cases pass")
    task = load(__file__)
    hz = 20.0
    for arm, waypoints in task.demo().items():
        plan = ArmPlan(list(waypoints), station.BASES[arm])
        q = np.array([plan.action7((k + 1) / hz)[0] for k in range(round(plan.duration * hz) + 5)])
        assert plan.worst_ik_mm < 0.5 and plan.worst_ik_deg < 0.1, (arm, plan.worst_ik_mm, plan.worst_ik_deg)
        assert np.all(q[:, :6] >= ik.LOWER) and np.all(q[:, :6] <= ik.UPPER), arm
        assert np.all((q[:, 6] >= 0.0) & (q[:, 6] <= 1.0)), arm
        jump = float(np.max(np.abs(np.diff(q[:, :6], axis=0))))
        assert jump < 0.25, (arm, jump)
        print(f"{arm}: {plan.duration:.1f} s plan, worst IK {plan.worst_ik_mm:.3f} mm / {plan.worst_ik_deg:.3f} deg, "
              f"largest joint step {math.degrees(jump):.1f} deg")
