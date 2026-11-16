"""A bar back counter at the start of service. A washed rocks-style tumbler (7.5 cm across, 10.5 cm tall, twelve
flat facets, 0.18 kg) stands rim-down on the counter at the robot's right. In front of the robot is a push-down
glass rinser: a stainless drain bowl with a spring-loaded rinse plate in it. Pressing a rim-down glass down on the
plate opens the valve and sprays the inside of the glass; the glass's own weight is not enough. To the left lies a
ribbed rubber drip mat.

Bar practice: a glass is rinsed on the rinser just before use, pressed rim-down over the plate so the spray stays
inside the glass, held down for a second or two, then lifted off to drain rim-down on the drip mat. Pushing the
plate down with nothing over it, or with the glass off centre, sprays the bar and the bartender.

Scored from the poses and the plate's travel: rinse time (plate pressed past the valve point with the glass
rim-down and centred over it); the glass rim-down wholly on the drip mat, let go and still at the end. Spraying
uncovered or knocking the glass over (more than 45 deg off rim-down) at any moment ends in 0.

Mechanism parts come from mechkit.py (in this folder): the rinse plate is a free body on a sprung prismatic joint.

Run it:  inspect-robots run --embodiment yam_arena -E task=tasks/bar-glass-rinser/bar_glass_rinser.py \\
           -E object_jitter=0 -E control_hz=20 -E cam_width=640 -E cam_height=352 \\
           --instruction "<the instruction in task.json>" --scorer task_score ...
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mechkit as mk  # noqa: E402  (multi-box parts, the sprung plate and per-episode hooks, installed in memory)

from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]  # the bar back counter top

# Frame: x forward from the arm bases, y left, z up, metres. Arm bases at (0, +-0.305, 0.75).
GLASS_START = (0.360, -0.250)
RINSER = (0.320, -0.090)  # centre of the rinse plate
MAT = (0.220, -0.240)  # centre of the drip mat

# Tumbler: a regular 12-sided prism, walls 3 mm, a solid 12 mm foot. Part frame: origin at the centre of the rim
# plane, +z toward the foot. Spawned with the frame upright, so the glass stands rim-down.
N_FACETS = 12
APOTHEM = 0.036  # 72 mm across the flats
RIM_R = APOTHEM / math.cos(math.pi / N_FACETS)  # 37.3 mm to a corner
WALL = 0.003
GLASS_H = 0.105
FOOT_H = 0.012
GLASS_KG = 0.180
GLASS_COLOR = (0.80, 0.87, 0.90)


def _glass_boxes():
    w = 2 * APOTHEM * math.tan(math.pi / N_FACETS) + 0.0004
    out = []
    for k in range(N_FACETS):
        th = 2 * math.pi * k / N_FACETS
        r = APOTHEM - WALL / 2
        out.append(mk.box((WALL, w, GLASS_H), at=(r * math.cos(th), r * math.sin(th), GLASS_H / 2), yaw=th))
    for yaw in (0.0, math.pi / 4):  # the foot: two 50 mm squares make a solid octagon inside the walls
        out.append(mk.box((0.050, 0.050, FOOT_H), at=(0.0, 0.0, GLASS_H - FOOT_H / 2), yaw=yaw))
    return out


# Rinser: a fixed stainless drain bowl (inside 14 x 14 cm) and a 9.5 cm square rinse plate on a vertical spring.
STEEL = (0.70, 0.71, 0.73)
DARK_STEEL = (0.35, 0.36, 0.38)
BOWL_IN = 0.140
BOWL_WALL = 0.004
BOWL_FLOOR_TOP = TOP + 0.015
BOWL_RIM_TOP = TOP + 0.040
PLATE = (0.095, 0.095, 0.006)
PLATE_KG = 0.060
PLATE_TOP = TOP + 0.032  # at rest, against its upper stop
PLATE_TRAVEL = 0.009  # down to its lower stop (2 mm above the bowl floor)
VALVE_K = 300.0  # N/m
VALVE_PRELOAD_M = 0.012  # spring preload at the upper stop: 3.6 N, above the glass's 1.8 N weight
VALVE_OPEN_M = 0.004  # the valve opens once the plate is 4 mm down: about 4.8 N on the plate

kit = mk.Kit()
kit.part("tumbler", (*GLASS_START, TOP + 0.0005), _glass_boxes(), mass=GLASS_KG, color=GLASS_COLOR)
kit.part("rinse_plate", (*RINSER, PLATE_TOP - PLATE[2] / 2), [mk.box(PLATE)], mass=PLATE_KG, color=DARK_STEEL)
kit.slider("rinse_valve", None, "rinse_plate", axis=(0, 0, 1), lower=-PLATE_TRAVEL, upper=0.0, start=0.0,
           stiffness=VALVE_K, rest=VALVE_PRELOAD_M, damping=3.0)

_bh = BOWL_RIM_TOP - TOP
_bo = BOWL_IN / 2 + BOWL_WALL / 2
DRIP_MAT = (0.130, 0.130, 0.006)
RUBBER = (0.12, 0.12, 0.13)
FIXTURES = [
    Box("rinser_floor", pos=(*RINSER, (TOP + BOWL_FLOOR_TOP) / 2), size=(BOWL_IN, BOWL_IN, BOWL_FLOOR_TOP - TOP),
        color=STEEL),
    Box("rinser_wall_near", pos=(RINSER[0] - _bo, RINSER[1], TOP + _bh / 2),
        size=(BOWL_WALL, BOWL_IN + 2 * BOWL_WALL, _bh), color=STEEL),
    Box("rinser_wall_far", pos=(RINSER[0] + _bo, RINSER[1], TOP + _bh / 2),
        size=(BOWL_WALL, BOWL_IN + 2 * BOWL_WALL, _bh), color=STEEL),
    Box("rinser_wall_left", pos=(RINSER[0], RINSER[1] + _bo, TOP + _bh / 2), size=(BOWL_IN, BOWL_WALL, _bh),
        color=STEEL),
    Box("rinser_wall_right", pos=(RINSER[0], RINSER[1] - _bo, TOP + _bh / 2), size=(BOWL_IN, BOWL_WALL, _bh),
        color=STEEL),
    Box("drip_mat", pos=(*MAT, TOP + DRIP_MAT[2] / 2), size=DRIP_MAT, color=RUBBER),
]
OBJECTS = [*kit.objects(), *FIXTURES]

SUCCESS_HOLD_S = 1.0
SCORE_OVER = "end"

# -- scoring ---------------------------------------------------------------------------------------------------

RINSE_S = 1.5  # full rinse: 1.5 s of spray into the glass
STEP_S = 0.05  # one control step at 20 Hz
CENTRE_TOL = 0.012  # rim centre within 12 mm of the plate centre: the spray goes into the glass
SEAT_GAP = 0.004  # rim plane within 4 mm of the plate's top face
FLAT_DEG = 8.0
COVER_DEG = 10.0
TIP_OVER_DEG = 45.0
RIM_GAP = 0.003
_TEST_TCPS = None  # self-test stand-in for the two grippers' TCPs


def _released(st, margin=0.025) -> bool:
    tcps = mk.tcps() if _TEST_TCPS is None else _TEST_TCPS
    lo, hi = np.asarray(st.low) - margin, np.asarray(st.high) + margin
    return not any(bool(np.all(np.asarray(t) >= lo) and np.all(np.asarray(t) <= hi)) for t in tcps)


def _tilt(quat_xyzw) -> float:
    """Angle of the part's +z from world up (deg). For the glass: 0 is rim-down, 180 is upright."""
    x, y, _, _ = (float(v) for v in quat_xyzw)
    return math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))))


def _plate_down(objects) -> float:
    """How far the rinse plate is pushed down (m, from the plate's pose)."""
    return -kit.q(objects, "rinse_valve")


def _spraying(objects) -> bool:
    return _plate_down(objects) >= VALVE_OPEN_M


def _covered(objects) -> bool:
    """The glass rim-down, centred over the plate and seated on it: the spray goes inside the glass."""
    gpos, gq = kit.pose(objects, "tumbler")
    ppos, _ = kit.pose(objects, "rinse_plate")
    plate_top = float(ppos[2]) + PLATE[2] / 2
    return bool(_tilt(gq) <= COVER_DEG
                and math.dist((float(gpos[0]), float(gpos[1])), (float(ppos[0]), float(ppos[1]))) <= CENTRE_TOL
                and abs(float(gpos[2]) - plate_top) <= SEAT_GAP)


def _rim_points(objects, n=24):
    return np.array([kit.point(objects, "tumbler",
                               (RIM_R * math.cos(2 * math.pi * k / n), RIM_R * math.sin(2 * math.pi * k / n), 0.0))
                     for k in range(n)])


def _on_mat(objects) -> bool:
    """Rim-down, every rim point on the drip mat and within 3 mm of its top face, let go and still."""
    gst = objects["tumbler"]
    _, gq = kit.pose(objects, "tumbler")
    if _tilt(gq) > FLAT_DEG or not gst.resting() or not _released(gst):
        return False
    pts = _rim_points(objects)
    half = DRIP_MAT[0] / 2
    inside = np.all(np.abs(pts[:, 0] - MAT[0]) <= half) and np.all(np.abs(pts[:, 1] - MAT[1]) <= half)
    on_top = np.all(np.abs(pts[:, 2] - (TOP + DRIP_MAT[2])) <= RIM_GAP)
    return bool(inside and on_top)


def _observe(objects) -> None:
    """Per control step: rinse time; latches for spraying uncovered and for the glass knocked over."""
    m = mk.MEMORY
    _, gq = kit.pose(objects, "tumbler")
    if _tilt(gq) > TIP_OVER_DEG:
        m["tipped"] = True
    if _spraying(objects):
        if _covered(objects):
            m["rinse_s"] = m.get("rinse_s", 0.0) + STEP_S
        else:
            m["sprayed_uncovered"] = True


kit.on_step(_observe)


def score(objects) -> float:
    """0.6 x rinse fraction (rinse time / 1.5 s, capped at 1) + 0.4 if the glass ends rim-down on the drip mat, let
    go and still. Spraying with the plate uncovered, or the glass knocked over, at any moment: 0."""
    m = mk.MEMORY
    if m.get("tipped") or m.get("sprayed_uncovered"):
        return 0.0
    rinse = min(1.0, m.get("rinse_s", 0.0) / RINSE_S)
    return round(0.6 * rinse + (0.4 if _on_mat(objects) else 0.0), 4)


def success(objects) -> bool:
    """Rinsed for 1.5 s, nothing sprayed uncovered, the glass rim-down on the drip mat, let go and still; 1 s."""
    return score(objects) >= 1.0 - 1e-9


def distance(objects) -> float:
    """Rinse time still missing (as 0.1 m per second) plus metres from the glass's rim centre to the mat."""
    gpos, _ = kit.pose(objects, "tumbler")
    left = max(0.0, RINSE_S - mk.MEMORY.get("rinse_s", 0.0))
    return float(0.1 * left + math.dist((float(gpos[0]), float(gpos[1]), float(gpos[2])),
                                        (*MAT, TOP + DRIP_MAT[2])))


# -- the scripted demo ------------------------------------------------------------------------------------------
# Open loop from the fixed layout, right arm only: open the fingers round the glass's upper body (two flats face
# +-y), close, lift, carry it over the rinse plate, set the rim on the plate, press down past the valve point and
# hold, lift off, carry to the drip mat, set it down rim-down, open, rise.

TIP = 0.013  # TCP above the pad tips
TRAVEL = TOP + 0.160
GRIP_DROP = 0.035  # pad tips this far below the glass's foot
LIFT = 0.050  # carry height of the rim above the counter
PLACE_GAP = 0.002
PRESS_M = 0.014  # command the rim this far below the plate's rest top (the plate stops at 9 mm)
Y_TRIM = 0.005  # the grasp leaves the glass ~5 mm off the TCP toward -y (measured in sim)


def _tcp_z(rim_z):
    return rim_z + GLASS_H - GRIP_DROP + TIP


def _right():
    gx, gy = GLASS_START
    rx, ry = RINSER[0], RINSER[1] + Y_TRIM
    mx, my = MAT[0], MAT[1] + Y_TRIM
    o, c = FINGER_OPEN, FINGER_CLOSED
    grip = _tcp_z(TOP)
    carry = _tcp_z(BOWL_RIM_TOP + 0.025)
    return [
        Waypoint(2.0, (gx, gy, TRAVEL), o, "over the glass"),
        Waypoint(1.5, (gx, gy, grip + 0.020), o, "descend round the glass"),
        Waypoint(1.0, (gx, gy, grip), o, "fingers round the upper body"),
        Waypoint(1.2, (gx, gy, grip), c, "close on two flats"),
        Waypoint(0.5, (gx, gy, grip), c, "settle"),
        Waypoint(1.5, (gx, gy, carry), c, "lift rim-down"),
        Waypoint(2.5, (rx, ry, carry), c, "carry over the rinser"),
        Waypoint(0.6, (rx, ry, carry), c, "settle over"),
        Waypoint(1.5, (rx, ry, _tcp_z(PLATE_TOP + PLACE_GAP)), c, "rim onto the plate"),
        Waypoint(1.0, (rx, ry, _tcp_z(PLATE_TOP - PRESS_M)), c, "press down"),
        Waypoint(2.5, (rx, ry, _tcp_z(PLATE_TOP - PRESS_M)), c, "hold: rinse"),
        Waypoint(1.0, (rx, ry, carry), c, "lift off"),
        Waypoint(2.5, (mx, my, carry), c, "carry over the drip mat"),
        Waypoint(0.6, (mx, my, carry), c, "settle over"),
        Waypoint(1.5, (mx, my, _tcp_z(TOP + DRIP_MAT[2] + PLACE_GAP)), c, "set it on the mat"),
        Waypoint(0.5, (mx, my, _tcp_z(TOP + DRIP_MAT[2] + PLACE_GAP)), c, "settle"),
        Waypoint(1.2, (mx, my, _tcp_z(TOP + DRIP_MAT[2] + PLACE_GAP)), o, "open: let go"),
        Waypoint(1.5, (mx, my, TOP + GLASS_H + 0.060), o, "rise straight up"),
        Waypoint(1.5, (0.250, -0.200, TRAVEL), o, "clear"),
    ]



def demo():
    """Right arm: pick the glass, rinse it for 2.5 s on the rinser, set it rim-down on the drip mat. About 27 s."""
    return {"right": _right()}


# -- offline self-test ------------------------------------------------------------------------------------------


def _states(glass_xy=None, glass_z=None, tilt_deg=0.0, plate_down=0.0, moving=False):
    import dataclasses

    gxy = GLASS_START if glass_xy is None else glass_xy
    gz = TOP if glass_z is None else glass_z
    r = mk.axis_angle((1, 0, 0), math.radians(tilt_deg))
    st = kit.fake_states(q={"rinse_valve": -plate_down},
                         moved={"tumbler": (np.array([gxy[0], gxy[1], gz]), mk.mat_to_quat(r))})
    if moving:
        st["tumbler"] = dataclasses.replace(st["tumbler"], lin_vel=np.array([0.0, 0.05, 0.0]))
    return st


def self_test():
    """The scoring rules on hand-built poses, including the rinse timer and the two latches."""
    global _TEST_TCPS
    mat_top = TOP + DRIP_MAT[2]
    seated = PLATE_TOP - 0.007
    rinse = _states(RINSER, seated, plate_down=0.007)
    done = _states(MAT, mat_top)
    n_full = int(round(RINSE_S / STEP_S))
    cases = {  # name: ([states in time order], gripper TCPs at the end, score)
        "start: nothing done": ([_states()], [], 0.0),
        "rinsed 1.5 s, then on the mat": ([rinse] * n_full + [done], [], 1.0),
        "rinsed 0.75 s, then on the mat": ([rinse] * (n_full // 2) + [done], [], 0.7),
        "shortcut: straight to the mat, no rinse": ([done], [], 0.4),
        "glass resting on the plate, not pressed": ([_states(RINSER, PLATE_TOP)] * 40 + [done], [], 0.4),
        "rinsed, glass left on the rinser": ([rinse] * n_full + [_states(RINSER, PLATE_TOP)], [], 0.6),
        "rinsed, still held over the mat": ([rinse] * n_full + [done], [(MAT[0], MAT[1], TOP + 0.1)], 0.6),
        "plate pushed with nothing on it": ([_states(plate_down=0.006)] + [rinse] * n_full + [done], [], 0.0),
        "glass 2 cm off centre while pressed": ([_states((RINSER[0], RINSER[1] + 0.02), seated, plate_down=0.007)]
                                                + [rinse] * n_full + [done], [], 0.0),
        "plate pressed only 3 mm": ([_states(RINSER, PLATE_TOP - 0.003, plate_down=0.003)] * 40 + [done], [], 0.4),
        "glass knocked over": ([rinse] * n_full + [_states(tilt_deg=90.0), done], [], 0.0),
        "glass upright on the mat": ([rinse] * n_full + [_states(MAT, mat_top + GLASS_H, tilt_deg=180.0)], [], 0.0),
        "glass overhanging the mat edge": ([rinse] * n_full + [_states((MAT[0], MAT[1] + 0.06), mat_top)], [], 0.6),
        "glass still sliding on the mat": ([rinse] * n_full + [_states(MAT, mat_top, moving=True)], [], 0.6),
    }
    try:
        for name, (seq, tcps, want) in cases.items():
            mk.new_episode()
            _TEST_TCPS = []
            for st in seq:
                _observe(st)
            _TEST_TCPS = tcps
            got, got_success = score(seq[-1]), success(seq[-1])
            assert abs(got - want) < 1e-6 and got_success == (want >= 1.0), (name, got, want)
    finally:
        mk.new_episode()
        _TEST_TCPS = None
    # geometry: the glass fits the open fingers; its rim covers the plate's middle and fits the bowl and the mat;
    # the glass's weight alone does not open the valve, a few newtons more does
    assert FINGER_OPEN < -(APOTHEM + 0.008) and 2 * RIM_R < BOWL_IN - 0.050 and 2 * RIM_R < DRIP_MAT[0] - 0.050
    assert VALVE_K * VALVE_PRELOAD_M > 1.5 * GLASS_KG * 9.81
    assert VALVE_K * (VALVE_PRELOAD_M + VALVE_OPEN_M) - GLASS_KG * 9.81 < 4.0
    assert PLATE_TOP - PLATE[2] - PLATE_TRAVEL > BOWL_FLOOR_TOP
    return len(cases)


if __name__ == "__main__":
    # Offline checks, no GPU: the scoring rules on hand-built states, then the demo plan replayed through the
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
        print(f"{arm}: {plan.duration:.1f} s plan, worst IK {plan.worst_ik_mm:.3f} mm / {plan.worst_ik_deg:.3f} deg")
