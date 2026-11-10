"""On the station table: three open labeled bins (M3, M4, M5) hold three metric Phillips pan-head screws each.
A fixed manikin forearm and open palm wait on the robot's left with a request card that reads M4 x 20 mm. The
policy must take one screw from the M4 bin and place it alone in the palm, then let go.

Scoring Criteria (tiers do not stack; single best tier, else 0):
  - 0: still holding any screw (gripper within 3 cm), or any M3/M5 on the palm
  - 1.0: one M4 alone on palm, released, still 1 s; other eight still in start bins
  - 0.5: M4 on palm with another screw on or next to the palm
  - 0.25: lifted an M4 by 3 cm without a clean handoff

SimReady assets:
  1. M4 fastener: factory_bolt_m04_loose (also factory_bolt_m04_tight)
  2. Generic screws: hardware_screw_95_a01, hardware_screw_99_physics_a01, screw_high_res
  3. Optional clutter: factory_nut_m4_loose / factory_nut_m4_tight
  4. Bins / trays: bin_waste_a01–a04, bin_waste_b01, or airSystem_air_filter_box_a01
  5. USD path pattern: <asset>/simready_usd/sm_<asset>_01.usd

Run it:  inspect-robots run --embodiment yam_arena -E task=tasks/hand-correct-screw/task.py \\
           -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \\
           --instruction "The person asks for an M4 x 20 mm Phillips pan-head machine screw. Take one screw from the bin whose front label reads M4 and place it in their open palm; then let go." \\
           --scorer task_score --max-steps 2400
"""

from __future__ import annotations

import math

import numpy as np

from yam_arena.constants import FINGER_OPEN
from yam_arena.demo import Waypoint
from yam_arena.task_file import TABLE_TOP, Box, Cylinder

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

INSTRUCTION = (
    "The person asks for an M4 x 20 mm Phillips pan-head machine screw. Take one screw from the bin whose "
    "front label reads M4 and place it in their open palm; then let go."
)

# Fixed open bins.
BIN_OUT = (0.080, 0.060, 0.040)  # outside length width height in metres
WALL_T = 0.004
BIN_IN = (BIN_OUT[0] - 2 * WALL_T, BIN_OUT[1] - 2 * WALL_T)  # inner floor 7.2 by 5.2 cm
BIN_X = 0.28
# Left to right from behind the arms: M3 then M4 then M5.
BIN_Y = {"M3": -0.12, "M4": 0.00, "M5": 0.12}
BIN_COLOR = (0.72, 0.72, 0.74)
# Coloured plates stand in for printed M3 M4 M5 text.
LABEL_COLOR = {"M3": (0.95, 0.85, 0.20), "M4": (0.97, 0.97, 0.97), "M5": (0.35, 0.75, 0.40)}
REQUESTED = "M4"


def _bin(size_label: str) -> list[Box]:
    """Open-top bin with a front (robot-facing, -x) label plate. Fixed (no mass)."""
    cx, cy = BIN_X, BIN_Y[size_label]
    n = size_label.lower()
    lx, ly, h = BIN_OUT
    wz = TOP + h / 2
    o_x, o_y = lx / 2 - WALL_T / 2, ly / 2 - WALL_T / 2
    parts = [
        Box(f"bin_{n}_xn", pos=(cx - o_x, cy, wz), size=(WALL_T, ly, h), color=BIN_COLOR),
        Box(f"bin_{n}_xp", pos=(cx + o_x, cy, wz), size=(WALL_T, ly, h), color=BIN_COLOR),
        Box(f"bin_{n}_yn", pos=(cx, cy - o_y, wz), size=(lx - 2 * WALL_T, WALL_T, h), color=BIN_COLOR),
        Box(f"bin_{n}_yp", pos=(cx, cy + o_y, wz), size=(lx - 2 * WALL_T, WALL_T, h), color=BIN_COLOR),
        # Label plate on the near wall facing the robot.
        Box(
            f"label_{n}",
            pos=(cx - lx / 2 - 0.001, cy, TOP + 0.025),
            size=(0.002, 0.030, 0.015),
            color=LABEL_COLOR[size_label],
        ),
    ]
    return parts


BINS = [p for label in ("M3", "M4", "M5") for p in _bin(label)]

# Screws are boxes not cylinders so they can lie flat.
# Box size is length along x then width then height.
# Width and height use the pan head diameter so the gripper can pinch the head.
# Shank diameter is kept in SCREW_SPEC for scoring and real world checks.
STEEL = (0.62, 0.63, 0.66)
SCREW_SPEC = {
    "M3": {"shank_d": 0.003, "head_d": 0.0055, "head_h": 0.002, "shank_l": 0.016, "mass": 0.0016},
    "M4": {"shank_d": 0.004, "head_d": 0.0070, "head_h": 0.0025, "shank_l": 0.020, "mass": 0.0031},
    "M5": {"shank_d": 0.005, "head_d": 0.0085, "head_h": 0.003, "shank_l": 0.025, "mass": 0.0054},
}


def _screw_size(label: str) -> tuple[float, float, float]:
    s = SCREW_SPEC[label]
    length = s["head_h"] + s["shank_l"]
    d = s["head_d"]
    return (length, d, d)


def _screw_starts(label: str) -> list[tuple[float, float]]:
    """Three screws in a row inside the bin, centres on the inner floor, shanks along +x (head toward +x)."""
    cy = BIN_Y[label]
    length = _screw_size(label)[0]
    # Keep centres inside the inner floor with a small margin.
    ys = (cy - 0.016, cy, cy + 0.016)
    # Place so the long axis and head stay inside the bin.
    xs = BIN_X - (BIN_IN[0] / 2 - length / 2 - 0.004)
    return [(xs, y) for y in ys]


SCREWS: list[Box] = []
SCREW_HOME: dict[str, str] = {}  # screw name to size label
SCREW_START_XY: dict[str, tuple[float, float]] = {}
for label in ("M3", "M4", "M5"):
    size = _screw_size(label)
    mass = SCREW_SPEC[label]["mass"]
    for i, (sx, sy) in enumerate(_screw_starts(label)):
        name = f"screw_{label.lower()}_{i}"
        SCREWS.append(
            Box(name, pos=(sx, sy, TOP + size[2] / 2 + 0.0005), size=size, mass=mass, color=STEEL)
        )
        SCREW_HOME[name] = label
        SCREW_START_XY[name] = (sx, sy)

M4_NAMES = [n for n, lab in SCREW_HOME.items() if lab == "M4"]
ALL_SCREW_NAMES = list(SCREW_HOME)

# Fixed forearm palm and request card.
SKIN = (0.93, 0.74, 0.62)
PALM_C = (0.18, 0.22)
PALM_R = 0.045
PALM_H = 0.015
# Palm disc sits 2 cm above the table so its top face is 3.5 cm up.
PALM_TOP = TOP + 0.020 + PALM_H
PALM = Cylinder(
    "palm",
    pos=(PALM_C[0], PALM_C[1], TOP + 0.020 + PALM_H / 2),
    radius=PALM_R,
    height=PALM_H,
    color=SKIN,
)
PALM_MARK = Cylinder(
    "palm_mark",
    pos=(PALM_C[0], PALM_C[1], PALM_TOP + 0.0005),
    radius=0.010,
    height=0.001,
    color=(0.05, 0.05, 0.05),
)
FOREARM = Box(
    "forearm",
    pos=(PALM_C[0], PALM_C[1] + 0.090 + 0.020, TOP + 0.020 + 0.020),
    size=(0.040, 0.180, 0.040),
    color=SKIN,
)
CARD = Box(
    "request_card",
    pos=(0.12, 0.22, TOP + 0.030),
    size=(0.002, 0.060, 0.040),
    color=(0.97, 0.97, 0.97),
)
CARD_STAND = Box(
    "request_stand",
    pos=(0.12, 0.22, TOP + 0.010),
    size=(0.020, 0.020, 0.020),
    color=(0.45, 0.45, 0.48),
)

OBJECTS = [*BINS, *SCREWS, PALM, PALM_MARK, FOREARM, CARD, CARD_STAND]

# Scoring thresholds.
RELEASE_DIST = 0.03
LIFT_M = 0.03  # 3 cm above the table counts as a lift for 0.25
ON_PALM_Z = 0.004  # lowest point within 4 mm of the palm top
BIN_MARGIN = 0.0
REST_SPEED = 0.01
REST_OMEGA = 0.5

_lifted_m4 = False
_ENV: dict = {"env": None}


def _install_env_hook() -> None:
    """Cache the sim env for gripper release checks in score."""
    try:
        from yam_arena import inspect_embodiment as ie
    except Exception:  # pragma: no cover
        return
    cls = ie.YamArenaEmbodiment
    if getattr(cls._task_step, "_screw_hooked", False):
        return
    base = cls._task_step

    def _task_step(self, *a, **k):
        _ENV["env"] = self._env
        return base(self, *a, **k)

    _task_step._screw_hooked = True
    cls._task_step = _task_step


_install_env_hook()


def grippers() -> list[np.ndarray]:
    """World positions of both grippers' TCP (between the finger pads), [] if unknown."""
    env = _ENV["env"]
    if env is None:
        return []
    base = env.unwrapped
    origin = base.scene.env_origins[0]
    out = []
    for key in ("left_robot", "right_robot"):
        art = base.scene[key]
        names = list(art.body_names)
        pos = (getattr(art.data.body_pos_w, "torch", art.data.body_pos_w)[0] - origin).detach().cpu().numpy()
        g = pos[names.index("gripper")].astype(np.float64)
        mid = (pos[names.index("tip_left")] + pos[names.index("tip_right")]).astype(np.float64) / 2
        ax = (mid - g) / (np.linalg.norm(mid - g) + 1e-12)
        out.append(g + ax * 0.134)
    return out


def _track_lift(objects) -> None:
    """Track whether any M4 was lifted 3 cm above the table."""
    global _lifted_m4
    # Reset on a fresh episode when every M4 is near start
    if all(float(objects[n].pos[2] - SCREW_SPEC["M4"]["head_d"] / 2) < TOP + 0.005 for n in M4_NAMES):
        if all(
            math.hypot(float(objects[n].pos[0]) - SCREW_START_XY[n][0], float(objects[n].pos[1]) - SCREW_START_XY[n][1])
            < 0.01
            for n in M4_NAMES
        ):
            _lifted_m4 = False
    for n in M4_NAMES:
        if float(objects[n].low[2]) >= TOP + LIFT_M:
            _lifted_m4 = True


def _resting(obj) -> bool:
    if hasattr(obj, "resting"):
        return bool(obj.resting())
    return float(np.linalg.norm(obj.lin_vel)) < REST_SPEED and float(np.linalg.norm(obj.ang_vel)) < REST_OMEGA


def _on_palm(obj) -> bool:
    """Screw centre inside the 9 cm palm disc (top view), lowest point within 4 mm of the palm top, at rest."""
    dx = float(obj.pos[0]) - PALM_C[0]
    dy = float(obj.pos[1]) - PALM_C[1]
    if dx * dx + dy * dy > PALM_R * PALM_R:
        return False
    if float(obj.low[2]) > PALM_TOP + ON_PALM_Z:
        return False
    if float(obj.low[2]) < PALM_TOP - 0.006:
        return False
    return _resting(obj)


def _in_home_bin(name: str, obj) -> bool:
    """Screw centre still inside its starting bin's inner floor outline."""
    label = SCREW_HOME[name]
    cx, cy = BIN_X, BIN_Y[label]
    hx, hy = BIN_IN[0] / 2 + BIN_MARGIN, BIN_IN[1] / 2 + BIN_MARGIN
    return abs(float(obj.pos[0]) - cx) <= hx and abs(float(obj.pos[1]) - cy) <= hy


def _released(objects) -> bool:
    gs = grippers()
    if not gs:
        # Treat resting as released when env is unavailable
        return True
    return all(
        float(np.linalg.norm(np.asarray(objects[n].pos[:3], float) - g)) >= RELEASE_DIST
        for g in gs
        for n in ALL_SCREW_NAMES
    )


def _on_palm_names(objects) -> list[str]:
    return [n for n in ALL_SCREW_NAMES if _on_palm(objects[n])]


def score(objects) -> float:
    """End-of-trial score; tiers do not stack.

    - 0: still holding a screw, or wrong size on palm
    - 1.0: one M4 alone on palm, others in home bins
    - 0.5: M4 on palm with another screw nearby
    - 0.25: lifted an M4 only
    """
    _track_lift(objects)
    if not _released(objects):
        return 0.0
    on_palm = _on_palm_names(objects)
    wrong = [n for n in on_palm if SCREW_HOME[n] != REQUESTED]
    if wrong:
        return 0.0
    m4_on = [n for n in on_palm if SCREW_HOME[n] == REQUESTED]
    if len(m4_on) == 1:
        others_home = all(_in_home_bin(n, objects[n]) for n in ALL_SCREW_NAMES if n != m4_on[0])
        # Block full score if another screw is near the palm
        near_palm = [
            n
            for n in ALL_SCREW_NAMES
            if n != m4_on[0]
            and math.hypot(float(objects[n].pos[0]) - PALM_C[0], float(objects[n].pos[1]) - PALM_C[1]) <= PALM_R + 0.03
            and float(objects[n].low[2]) <= PALM_TOP + 0.02
        ]
        if near_palm:
            return 0.5
        if others_home:
            return 1.0
        return 0.5
    if len(m4_on) > 1:
        return 0.5
    if _lifted_m4:
        return 0.25
    return 0.0


def success(objects) -> bool:
    """True when score is full success (1.0)."""
    return score(objects) >= 1.0 - 1e-9


def distance(objects) -> float:
    """Nearest M4 centre to palm centre, xy plus height above palm top, metres."""
    best = math.inf
    for n in M4_NAMES:
        p = objects[n]
        dxy = math.hypot(float(p.pos[0]) - PALM_C[0], float(p.pos[1]) - PALM_C[1])
        dz = max(0.0, float(p.low[2]) - PALM_TOP)
        best = min(best, math.hypot(dxy, dz))
    return float(best)


# Scripted demo.
TRAVEL_Z = TOP + 0.14
# Grip across the 7 mm M4 head.
# Close 2 mm narrower than the part so the fingers centre it.
M4_WIDTH = SCREW_SPEC["M4"]["head_d"]
SQUEEZE = 0.002
GRIP = -(M4_WIDTH - SQUEEZE) / 2  # finger joint target in metres


def _pick_place(pick_xyz, place_xyz, grip=GRIP, travel_z=TRAVEL_Z):
    px, py, pz = pick_xyz
    qx, qy, qz = place_xyz
    return [
        Waypoint(1.5, (px, py, travel_z), FINGER_OPEN, "above pick"),
        Waypoint(1.2, (px, py, pz), FINGER_OPEN, "down to pick"),
        Waypoint(0.6, (px, py, pz), FINGER_OPEN, "settle"),
        Waypoint(1.2, (px, py, pz), grip, "close"),
        Waypoint(1.0, (px, py, travel_z), grip, "lift"),
        Waypoint(2.0, (qx, qy, travel_z), grip, "carry"),
        Waypoint(0.6, (qx, qy, travel_z), grip, "settle over place"),
        Waypoint(1.5, (qx, qy, qz), grip, "lower"),
        Waypoint(0.8, (qx, qy, qz), grip, "settle"),
        Waypoint(1.0, (qx, qy, qz), FINGER_OPEN, "release"),
        Waypoint(1.0, (qx, qy, travel_z), FINGER_OPEN, "retreat"),
    ]


def demo():
    """Left arm places screw_m4_1 on the palm centre then retreats; right stays clear. Needs object_jitter=0."""
    name = "screw_m4_1"
    sx, sy = SCREW_START_XY[name]
    pick_z = TOP + SCREW_SPEC["M4"]["head_d"] / 2 + 0.002
    place_z = PALM_TOP + SCREW_SPEC["M4"]["head_d"] / 2 + 0.003
    left = _pick_place((sx, sy, pick_z), (PALM_C[0], PALM_C[1], place_z))
    left.append(Waypoint(1.5, (0.20, 0.28, TRAVEL_Z), FINGER_OPEN, "clear the palm"))
    right = [
        Waypoint(1.0, (0.22, -0.22, TRAVEL_Z), FINGER_OPEN, "stay clear"),
        Waypoint(sum(w.duration for w in left), (0.22, -0.22, TRAVEL_Z), FINGER_OPEN, "wait out the handoff"),
    ]
    return {"left": left, "right": right}


def self_test(ObjectState=None) -> dict:
    """Scoring rules on hand-built states; no GPU. Returns a score report per case."""
    if ObjectState is None:
        from yam_arena.task_file import ObjectState

    global _lifted_m4
    half = {n: np.asarray(next(s.size for s in SCREWS if s.name == n)) / 2 for n in ALL_SCREW_NAMES}

    def screw(name, x, y, bottom=TOP, vel=(0.0, 0.0, 0.0)):
        h = half[name]
        pos = np.array([x, y, bottom + h[2]])
        return ObjectState(pos, np.array([0.0, 0.0, 0.0, 1.0]), np.asarray(vel, float), np.zeros(3), pos - h, pos + h)

    def start_scene(**moved):
        base = {n: screw(n, *SCREW_START_XY[n]) for n in ALL_SCREW_NAMES}
        base.update(moved)
        return base

    report = {}

    # Score 0 at start
    _lifted_m4 = False
    st = start_scene()
    got, ok = score(st), success(st)
    assert got == 0.0 and not ok
    report["start"] = {"score": got, "success": ok}

    # Score 1.0 for one M4 alone on palm
    good = start_scene(screw_m4_1=screw("screw_m4_1", PALM_C[0], PALM_C[1], bottom=PALM_TOP))
    _lifted_m4 = False
    got, ok = score(good), success(good)
    assert abs(got - 1.0) < 1e-9 and ok
    report["one_m4_alone"] = {"score": got, "success": ok}

    # Score 0 for wrong size on palm
    bad = start_scene(screw_m3_0=screw("screw_m3_0", PALM_C[0], PALM_C[1], bottom=PALM_TOP))
    got, ok = score(bad), success(bad)
    assert got == 0.0 and not ok
    report["wrong_size_on_palm"] = {"score": got, "success": ok}

    # Score 0.5 for M4 on palm with another nearby
    two = start_scene(
        screw_m4_1=screw("screw_m4_1", PALM_C[0], PALM_C[1], bottom=PALM_TOP),
        screw_m4_0=screw("screw_m4_0", PALM_C[0] + 0.02, PALM_C[1], bottom=PALM_TOP),
    )
    got, ok = score(two), success(two)
    assert abs(got - 0.5) < 1e-9 and not ok
    report["m4_with_nearby"] = {"score": got, "success": ok}

    # Score 0.25 for lift without place
    _lifted_m4 = True
    lifted = start_scene()
    got, ok = score(lifted), success(lifted)
    assert abs(got - 0.25) < 1e-9 and not ok
    report["lifted_only"] = {"score": got, "success": ok}
    _lifted_m4 = False
    return report


if __name__ == "__main__":
    # Offline report like dual-latch / din-rail / cabledemo result.json, without GPU
    import json

    from yam_arena import ik, station
    from yam_arena.demo import ArmPlan
    from yam_arena.task_file import load

    scoring = self_test()
    task = load(__file__)
    hz = 20.0
    duration_s = {}
    worst_ik_mm = {}
    worst_ik_deg = {}
    for arm, waypoints in task.demo().items():
        plan = ArmPlan(list(waypoints), station.BASES[arm])
        q = np.array([plan.action7((k + 1) / hz)[0] for k in range(round(plan.duration * hz) + 5)])
        assert plan.worst_ik_mm < 0.5 and plan.worst_ik_deg < 0.1, (arm, plan.worst_ik_mm, plan.worst_ik_deg)
        assert np.all(q[:, :6] >= ik.LOWER) and np.all(q[:, :6] <= ik.UPPER), arm
        duration_s[arm] = round(plan.duration, 1)
        worst_ik_mm[arm] = round(plan.worst_ik_mm, 3)
        worst_ik_deg[arm] = round(plan.worst_ik_deg, 3)

    result = {
        "success": True,
        "scoring_cases": len(scoring),
        "scoring": scoring,
        "duration_s": duration_s,
        "worst_ik_mm": worst_ik_mm,
        "worst_ik_deg": worst_ik_deg,
    }
    print(json.dumps(result, indent=2), flush=True)
