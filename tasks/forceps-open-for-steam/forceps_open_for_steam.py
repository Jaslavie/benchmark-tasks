"""On the station table are two grey silicone prep mats, one in front of each arm, and between them a perforated
stainless sterilizer basket. On each mat lies a pair of clean training-scale surgical forceps (like 20 cm
Rochester-Pean forceps), closed with the ratchet engaged: jaws pointing away from the robot, finger rings toward
it. Each pair is two steel shanks joined at the box lock by a pivot with friction, so the pair stays at whatever
opening it is left at, up to a 25 deg stop. The job is to put both pairs in the basket, ready for the steam cycle.

The steam-sterilization rule (AAMI ST79, instrument makers' reprocessing instructions) is that hinged instruments go
into the cycle open, ratchet disengaged, so steam reaches the box lock and the jaw faces. Scored from the shanks'
poses at the end: each pair in the basket, lying flat on its floor (not stacked), and open at least 15 deg.

Mechanism parts come from mechkit.py (in this folder), which adds hinges to the stock runtime in memory.

Run it:  inspect-robots run --embodiment yam_arena -E task=tasks/forceps-open-for-steam/forceps_open_for_steam.py \\
           -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \\
           --instruction "<the instruction in task.json>" --scorer task_score ...
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mechkit as mk  # noqa: E402  (hinge and per-episode hooks, installed in memory)

from yam_arena.constants import FINGER_CLOSED  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]

# Frame: x forward from the arm bases, y left, z up, metres. Arm bases at (0, +-0.305, 0.75).
MAT = (0.240, 0.110, 0.003)  # grey silicone prep mat
MAT_TOP = TOP + MAT[2]
PIVOT = {"L": (0.300, 0.215), "R": (0.300, -0.215)}  # each pair's box-lock pivot on its mat, jaws toward +x

# One shank in its own frame: origin at the pivot on the underside, x toward the jaws, y across. ``side`` +1: its
# finger ring is on +y (its jaw on -y: the shanks cross at the box lock); -1 the mirror image.
SHANK_KG = 0.035
H_BAR, H_JAW, H_RING = 0.008, 0.006, 0.006  # heights
HANDLE_Y = 0.018  # handle bar centre line off the axis (8 mm bar: 28 mm between the two bars when closed)
RING_C = (-0.133, 0.022)  # finger ring centre
OPEN_STOP = 25.0  # deg, the box lock's stop
FRICTION = 0.03  # N*m at the pivot: stands in for the ratchet and box-lock drag; the pair stays where it is left
STEEL = (0.74, 0.75, 0.77)
DARK_STEEL = (0.55, 0.56, 0.58)


def _shank_boxes(side):
    s = side
    neck_d = np.array([-0.032, s * 0.014])
    ring = [mk.box((0.026, 0.004, H_RING), at=(RING_C[0], s * RING_C[1] + dy, H_RING / 2)) for dy in (-0.011, 0.011)]
    ring += [mk.box((0.004, 0.018, H_RING), at=(RING_C[0] + dx, s * RING_C[1], H_RING / 2)) for dx in (-0.011, 0.011)]
    return [
        mk.box((0.050, 0.004, H_JAW), at=(0.031, -s * 0.002, H_JAW / 2)),  # jaw
        mk.box((0.016, 0.016, H_BAR), at=(0.0, 0.0, H_BAR / 2), color=DARK_STEEL),  # box lock
        mk.box((float(np.linalg.norm(neck_d)) + 0.004, 0.006, H_BAR), at=(-0.024, s * 0.011, H_BAR / 2),
               yaw=math.atan2(neck_d[1], neck_d[0])),  # neck from the box lock out to the handle
        mk.box((0.080, 0.008, H_BAR), at=(-0.080, s * HANDLE_Y, H_BAR / 2)),  # handle bar
        mk.box((0.010, 0.014, 0.004), at=(-0.112, s * 0.007, 0.002), color=DARK_STEEL),  # ratchet tab
        *ring,
    ]


kit = mk.Kit()
PAIRS = ("L", "R")
for _p in PAIRS:
    _x, _y = PIVOT[_p]
    kit.part(f"forceps{_p}_a", (_x, _y, MAT_TOP), _shank_boxes(-1), mass=SHANK_KG, color=STEEL)
    kit.part(f"forceps{_p}_b", (_x, _y, MAT_TOP), _shank_boxes(+1), mass=SHANK_KG, color=STEEL)
    # axis -z: a positive angle swings shank b's ring away from shank a's (and its jaw the other way): opening
    kit.hinge(f"lock{_p}", f"forceps{_p}_a", f"forceps{_p}_b", anchor=(_x, _y, MAT_TOP), axis=(0, 0, -1),
              lower_deg=0.0, upper_deg=OPEN_STOP, start_deg=0.0, friction=FRICTION)

BASKET_XY = (0.330, 0.000)
BASKET_IN = (0.240, 0.320, 0.030)  # inner length (x), width (y), wall height
BASKET_T, BASKET_FLOOR = 0.003, 0.002
FLOOR_TOP = TOP + BASKET_FLOOR


def _basket():
    (bx, by), (lx, ly, h), t = BASKET_XY, BASKET_IN, BASKET_T
    z = TOP + h / 2
    return [
        Box("basket_floor", pos=(bx, by, TOP + BASKET_FLOOR / 2), size=(lx + 2 * t, ly + 2 * t, BASKET_FLOOR),
            color=STEEL),
        Box("basket_wall_near", pos=(bx - lx / 2 - t / 2, by, z), size=(t, ly + 2 * t, h), color=STEEL),
        Box("basket_wall_far", pos=(bx + lx / 2 + t / 2, by, z), size=(t, ly + 2 * t, h), color=STEEL),
        Box("basket_wall_left", pos=(bx, by + ly / 2 + t / 2, z), size=(lx, t, h), color=STEEL),
        Box("basket_wall_right", pos=(bx, by - ly / 2 - t / 2, z), size=(lx, t, h), color=STEEL),
    ]


FIXTURES = [
    Box("mat_left", pos=(PIVOT["L"][0] - 0.045, PIVOT["L"][1], TOP + MAT[2] / 2), size=MAT, color=(0.42, 0.44, 0.47)),
    Box("mat_right", pos=(PIVOT["R"][0] - 0.045, PIVOT["R"][1], TOP + MAT[2] / 2), size=MAT, color=(0.42, 0.44, 0.47)),
    *_basket(),
]
OBJECTS = [*kit.objects(), *FIXTURES]

SUCCESS_HOLD_S = 1.0
SCORE_OVER = "end"

# -- scoring ---------------------------------------------------------------------------------------------------

OPEN_DEG = 15.0  # open: the shanks at least 15 deg apart (a jaw-tip gap of about 15 mm on a photo)
FLAT_DEG = 15.0
STACK_M = 0.004  # lying on the floor: lowest point no more than 4 mm above it (a shank is 6-8 mm tall)
_TEST_TCPS = None  # self-test stand-in for the two grippers' TCPs


def _released(st, margin=0.025) -> bool:
    tcps = mk.tcps() if _TEST_TCPS is None else _TEST_TCPS
    lo, hi = np.asarray(st.low) - margin, np.asarray(st.high) + margin
    return not any(bool(np.all(np.asarray(t) >= lo) and np.all(np.asarray(t) <= hi)) for t in tcps)


def _tilt(st) -> float:
    x, y, _, _ = (float(v) for v in st.quat_xyzw)
    return math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))))


def _shanks(objects, p):
    return objects[f"forceps{p}_a"], objects[f"forceps{p}_b"]


def _in_basket(objects, p) -> bool:
    """Both shanks wholly inside the basket's walls, flat on its floor (not on top of anything), let go and still."""
    (bx, by), (lx, ly, _) = BASKET_XY, BASKET_IN
    for st in _shanks(objects, p):
        lo, hi = np.asarray(st.low, float), np.asarray(st.high, float)
        if not (lo[0] >= bx - lx / 2 and hi[0] <= bx + lx / 2 and lo[1] >= by - ly / 2 and hi[1] <= by + ly / 2):
            return False
        if lo[2] > FLOOR_TOP + STACK_M or _tilt(st) > FLAT_DEG or not st.resting() or not _released(st):
            return False
    return True


def _open(objects, p) -> bool:
    return kit.q_deg(objects, f"lock{p}") >= OPEN_DEG


def score(objects) -> float:
    """Per pair: in the basket (both shanks inside its walls, flat within 15 deg, lowest point within 4 mm of its
    floor, let go and still) and open (at least 15 deg). 1.0 both in and open; 0.6 one in and open, the other in
    but closed; 0.4 only one in and open; 0.2 both in but closed; 0.1 one in, closed; else 0."""
    inside = {p: _in_basket(objects, p) for p in PAIRS}
    good = sum(inside[p] and _open(objects, p) for p in PAIRS)
    closed_in = sum(inside[p] and not _open(objects, p) for p in PAIRS)
    if good == 2:
        return 1.0
    if good == 1:
        return 0.6 if closed_in else 0.4
    return {2: 0.2, 1: 0.1}.get(closed_in, 0.0)


def success(objects) -> bool:
    """Both pairs open in the basket, let go and still; holds for 1 s."""
    return score(objects) >= 1.0 - 1e-9


def distance(objects) -> float:
    """Metres from each pair's box lock to its spot in the basket, plus the missing opening as arc at the jaw tips."""
    out = 0.0
    for p in PAIRS:
        pos, _ = kit.pose(objects, f"forceps{p}_a")
        out += math.dist([float(pos[0]), float(pos[1])], BASKET_SPOT[p])
        out += max(0.0, math.radians(OPEN_DEG - kit.q_deg(objects, f"lock{p}"))) * 0.056
    return float(out)


# -- the scripted demo ------------------------------------------------------------------------------------------
# Each arm, open loop from the scene's fixed geometry: put the closed fingers down between the pair's two handle
# bars (28 mm apart), open them so the pads push the handles apart until the box lock reaches its stop, keep
# pushing a little so the open pair is held from inside, lift it, carry it flat to its spot in the basket, set it on
# the floor and close the fingers to let go. The left arm loads first; the right arm opens its pair at the same time
# and waits with it lifted until the left arm is clear.

TIP = 0.013
PAD_BACK = 0.070  # the fingers go in 7 cm behind the pivot (the pads span 5.1 to 8.9 cm back)
PAD_HALF_W = 0.019
GRIP_Z = MAT_TOP + 0.0015 + TIP  # pad tips 1.5 mm above the mat
SQUEEZE = 0.002  # push this far past the handles' faces at the stop, so the pads hold the pair from inside
PAD_FACE = 0.0012  # the pushing (outer) face of each finger sits ~1.2 mm outside the finger's opening (measured in
# sim: fingers opened 18.7 mm each way spread the handles to 13.4 deg, faces 19.9 mm off the axis)
TRAVEL = TOP + 0.130
BASKET_SPOT = {"L": (0.371, 0.078), "R": (0.371, -0.078)}  # where each pair's pivot goes in the basket
X_TRIM = 0.0015  # light parts land ~1.5 mm short in x


def _handle_face_y(open_deg: float) -> float:
    """Distance off the axis of the handle bars' inner faces at the pads' front edge, with the pair opened
    ``open_deg`` symmetrically (each shank half of it)."""
    h = math.radians(open_deg / 2)
    x_edge = -(PAD_BACK - PAD_HALF_W)  # the pads' edge nearest the pivot
    inner = HANDLE_Y - 0.004
    d = (inner * math.sin(h) - x_edge) / math.cos(h)  # distance back along the bar to the pads' edge
    return d * math.sin(h) + inner * math.cos(h)


SPREAD = -(_handle_face_y(OPEN_STOP) - PAD_FACE + SQUEEZE)  # finger target: pads' outer faces past the handles


def _arm(p, yaw=0.0, wait_s=0.0):
    (px, py), (sx, sy) = PIVOT[p], BASKET_SPOT[p]
    gx, tx = px - PAD_BACK, sx - PAD_BACK + X_TRIM
    c, o = FINGER_CLOSED, SPREAD
    put_z = FLOOR_TOP + 0.002 + 0.0015 + TIP  # the pair's underside 2 mm above the basket floor
    plan = [
        Waypoint(2.0, (gx, py, TRAVEL), c, "over the handles", yaw),
        Waypoint(1.5, (gx, py, GRIP_Z + 0.025), c, "descend", yaw),
        Waypoint(1.0, (gx, py, GRIP_Z), c, "closed fingers down between the handles", yaw),
        Waypoint(0.5, (gx, py, GRIP_Z), c, "settle", yaw),
        Waypoint(2.5, (gx, py, GRIP_Z), o, "open the fingers: spread the handles to the stop", yaw),
        Waypoint(0.5, (gx, py, GRIP_Z), o, "settle", yaw),
        Waypoint(1.5, (gx, py, GRIP_Z + 0.030), o, "lift the open pair", yaw),
        Waypoint(1.0, (gx, py, TRAVEL), o, "lift", yaw),
    ]
    if wait_s:
        plan.append(Waypoint(wait_s, (gx, py, TRAVEL), o, "wait for the other arm", yaw))
    return plan + [
        Waypoint(2.0, (tx, sy, TRAVEL), o, "carry flat to the basket", yaw),
        Waypoint(0.6, (tx, sy, TRAVEL), o, "settle over", yaw),
        Waypoint(1.5, (tx, sy, put_z + 0.025), o, "lower", yaw),
        Waypoint(1.0, (tx, sy, put_z), o, "set down slowly", yaw),
        Waypoint(0.5, (tx, sy, put_z), o, "settle", yaw),
        Waypoint(1.2, (tx, sy, put_z), c, "close the fingers: let go", yaw),
        Waypoint(1.2, (tx, sy, TRAVEL), c, "up out of the basket", yaw),
        Waypoint(1.5, (tx - 0.120, sy * 2.5, TRAVEL), c, "clear", yaw),
    ]


def demo():
    """Left arm: open, lift and load the left pair. Right arm: open and lift the right pair at the same time, wait
    until the left arm is clear of the basket, then load it. About 30 s."""
    left = _arm("L")
    t_left_done = sum(w.duration for w in left[:-1])  # the left arm is up out of the basket
    right_pre = sum(w.duration for w in _arm("R")[:8])
    return {"left": left, "right": _arm("R", wait_s=max(0.5, t_left_done - right_pre - 2.0))}


# -- offline self-test ------------------------------------------------------------------------------------------


def _pair_at(p, pivot_xy, open_deg, z=None, yaw_deg=0.0, tilt_deg=0.0):
    """Kit states with pair ``p``'s pivot at ``pivot_xy`` (underside at ``z``), opened ``open_deg`` symmetrically
    and turned ``yaw_deg``; shank a tilted ``tilt_deg`` about x (for a 'not flat' case)."""
    z = MAT_TOP if z is None else z
    ya = math.radians(yaw_deg + open_deg / 2)  # shank a turns anticlockwise by half the opening ...
    r = mk.axis_angle((0, 0, 1), ya) @ mk.axis_angle((1, 0, 0), math.radians(tilt_deg))
    moved = {f"forceps{p}_a": (np.array([pivot_xy[0], pivot_xy[1], z]), mk.mat_to_quat(r))}
    # ... and shank b (the child) follows through the hinge, the other half the other way
    return kit.fake_states(q={f"lock{p}": open_deg}, deg=True, moved=moved)


def _geometry_check() -> None:
    """The closed fingers fit between the handles with 3 mm or more each side; at the stop the pads' outer faces
    reach past the handles; both pairs, open to the stop, fit in the basket side by side with 10 mm or more to
    spare at every wall and between them."""
    assert HANDLE_Y - 0.004 - 0.010 >= 0.003  # even for 10 mm thick closed fingers
    assert -SPREAD <= 0.04695 and -SPREAD > _handle_face_y(OPEN_STOP) - PAD_FACE
    st = _both_in(OPEN_STOP, OPEN_STOP)
    (bx, by), (lx, ly, _) = BASKET_XY, BASKET_IN
    lo = {p: np.min([st[f"forceps{p}_{k}"].low for k in "ab"], axis=0) for p in PAIRS}
    hi = {p: np.max([st[f"forceps{p}_{k}"].high for k in "ab"], axis=0) for p in PAIRS}
    for p in PAIRS:
        assert lo[p][0] >= bx - lx / 2 + 0.010 and hi[p][0] <= bx + lx / 2 - 0.010, (p, lo[p], hi[p])
        assert lo[p][1] >= by - ly / 2 + 0.010 and hi[p][1] <= by + ly / 2 - 0.010, (p, lo[p], hi[p])
    assert lo["L"][1] > hi["R"][1] + 0.010, (lo["L"], hi["R"])
    print(f"open pair footprint {hi['L'][0] - lo['L'][0]:.3f} x {hi['L'][1] - lo['L'][1]:.3f} m; "
          f"fingers spread to {-SPREAD * 2 * 1000:.1f} mm between the pads", file=sys.stderr)


def _both_in(deg_l, deg_r):
    st = _pair_at("L", BASKET_SPOT["L"], deg_l, FLOOR_TOP)
    pr = _pair_at("R", BASKET_SPOT["R"], deg_r, FLOOR_TOP)
    return {**st, **{k: v for k, v in pr.items() if k.startswith("forcepsR")}}


def self_test():
    """The scoring rules on hand-built poses (each pair posed through its hinge by the kit)."""
    global _TEST_TCPS
    import dataclasses

    _geometry_check()

    def mix(base, p, other):
        return {**base, **{k: v for k, v in other.items() if k.startswith(f"forceps{p}")}}

    start = kit.fake_states()
    both_open = _both_in(25.0, 25.0)
    both_closed = _both_in(0.0, 0.0)
    l_open_r_closed = _both_in(25.0, 0.0)
    l_open_r_mat = mix(_both_in(25.0, 0.0), "R", start)
    l_closed_r_mat = mix(_both_in(0.0, 0.0), "R", start)
    l_open_r_open_mat = mix(_both_in(25.0, 0.0), "R", _pair_at("R", PIVOT["R"], 25.0))
    # the right pair laid on top of the left one (stacked)
    stacked = mix(_both_in(25.0, 25.0), "R", _pair_at("R", (BASKET_SPOT["L"][0], BASKET_SPOT["L"][1] + 0.01), 25.0,
                                                         FLOOR_TOP + H_BAR))
    slight = _both_in(25.0, 12.0)
    over_wall = mix(_both_in(25.0, 25.0), "R", _pair_at("R", (BASKET_SPOT["R"][0], -0.140), 25.0, FLOOR_TOP))
    tilted = mix(_both_in(25.0, 25.0), "R", _pair_at("R", BASKET_SPOT["R"], 25.0, FLOOR_TOP + 0.01, tilt_deg=25.0))
    moving = dict(both_open)
    moving["forcepsR_b"] = dataclasses.replace(moving["forcepsR_b"], lin_vel=np.array([0.0, 0.0, -0.05]))
    tcp_r = (BASKET_SPOT["R"][0] - PAD_BACK, BASKET_SPOT["R"][1], FLOOR_TOP + 0.02)
    cases = {  # name: (state, gripper TCPs, score)
        "start: both closed on the mats": (start, [], 0.0),
        "both open in the basket": (both_open, [], 1.0),
        "naive: both dropped in closed": (both_closed, [], 0.2),
        "left open, right in but closed": (l_open_r_closed, [], 0.6),
        "left open in, right still on its mat": (l_open_r_mat, [], 0.4),
        "left open in, right opened but left on its mat": (l_open_r_open_mat, [], 0.4),
        "left in closed, right on its mat": (l_closed_r_mat, [], 0.1),
        "right pair stacked on the left one": (stacked, [], 0.4),
        "right pair open only 12 deg": (slight, [], 0.6),
        "right pair across the basket wall": (over_wall, [], 0.4),
        "right pair tilted 25 deg (propped on the wall)": (tilted, [], 0.4),
        "right pair still falling": (moving, [], 0.4),
        "right pair still held by the fingers": (both_open, [tcp_r], 0.4),
    }
    try:
        for name, (st, tcps, want) in cases.items():
            mk.new_episode()
            _TEST_TCPS = tcps
            got, got_success = score(st), success(st)
            assert abs(got - want) < 1e-9 and got_success == (want >= 1.0), (name, got, want)
        # the hinge readout follows the poses
        assert abs(kit.q_deg(both_open, "lockL") - 25.0) < 1e-6 and abs(kit.q_deg(start, "lockR")) < 1e-6
    finally:
        mk.new_episode()
        _TEST_TCPS = None
    return len(cases)


if __name__ == "__main__":
    # Offline checks, no GPU: geometry and the scoring rules on hand-built states, then both arms' plans replayed
    # through the kinematics at 20 Hz, every joint inside the URDF limits with an IK error under 0.5 mm.
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
        print(f"{arm}: {plan.duration:.1f} s plan, worst IK {plan.worst_ik_mm:.3f} mm / {plan.worst_ik_deg:.3f} deg")
