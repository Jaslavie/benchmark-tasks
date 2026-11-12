"""Three colored bins hold metric Phillips screws. A request card names one size. The policy must
match that request, take one screw of the correct size, and place it alone in the handoff tray.

Scoring Criteria (tiers do not stack; single best tier, else 0):
  - 0: gripper within 3 cm of any screw; more than one screw in the tray; any wrong-size screw in the tray
  - 1.0: exactly one requested-size screw alone in the tray, released, still 1 s; other eight in start bins
  - 0.5: exactly one requested-size screw in the tray, but another screw touches or sits next to the tray
  - 0.25: lifted a requested-size screw by 3 cm without a clean handoff

Whole-bin colours and front labels identify size; scorer keys off start bin:
  - yellow + M3 label = M3
  - white + M4 label = M4
  - green + M5 label = M5

SimReady meshes (NVIDIA, CC-BY-4.0; under assets/simready/) for fixed bin mates and tray:
  - M3: hardware_screw_m3_a02
  - M4: factory_bolt_m04_loose
  - M5: screw_high_res
  - Handoff tray: dishware_plastic_tray_a01
Free pick is a metal blank sized to the M4 head envelope (parallel jaws cannot retain thin convex SimReady bolts).

Goal: match the request card, hand one correct screw alone into the tray, leave the other eight in their bins

Run it:  inspect-robots run --embodiment yam_arena -E task=tasks/hand-requested-screw/hand_requested_screw.py \\
           -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \\
           --instruction "The person asks for an M4 x 20 mm Phillips pan-head machine screw. Choose the matching size from the three bins, place one screw alone in the tray, then let go." \\
           --scorer task_score --max-steps 2400
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from yam_arena.constants import FINGER_OPEN
from yam_arena.demo import Waypoint
from yam_arena.task_file import TABLE_TOP, Asset, Box

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

INSTRUCTION = (
    "The person asks for an M4 x 20 mm Phillips pan-head machine screw. Choose the matching size "
    "from the three bins, place one screw alone in the tray, then let go."
)

# ========================================
# Local NVIDIA SimReady USD registration
# ========================================

_ASSETS = Path(__file__).resolve().parent / "assets" / "simready"
SIMREADY_PHYSX_VARIANTS: dict[str, str] = {"PhysX": "Enabled"}
SIMREADY_ENTRIES: dict[str, Path] = {
    "simready_factory_bolt_m04_loose": _ASSETS
    / "factory_bolt_m04_loose"
    / "simready_usd"
    / "sm_factory_bolt_m04_loose_01.usda",
    "simready_hardware_screw_m3_a02": _ASSETS
    / "hardware_screw_m3_a02"
    / "simready_usd"
    / "sm_hardware_screw_m3_a02_01.usda",
    "simready_screw_high_res": _ASSETS / "screw_high_res" / "simready_usd" / "sm_screw_high_res_01.usda",
    "simready_dishware_plastic_tray_a01": _ASSETS
    / "dishware_plastic_tray_a01"
    / "simready_usd"
    / "sm_dishware_plastic_tray_a01_01.usd",
}
SCREW_ASSET = {
    "M3": "simready_hardware_screw_m3_a02",
    "M4": "simready_factory_bolt_m04_loose",
    "M5": "simready_screw_high_res",
}
TRAY_ASSET = "simready_dishware_plastic_tray_a01"


def missing_assets() -> list[str]:
    """Registry names whose entry USD file is absent."""
    return [name for name, path in SIMREADY_ENTRIES.items() if not path.is_file()]


def register_simready() -> None:
    """Register local SimReady USDs and enable PhysX for yam_arena Asset spawns."""
    missing = missing_assets()
    if missing:
        raise FileNotFoundError(
            "SimReady assets missing: " + ", ".join(missing) + f". Place packages under {_ASSETS}/"
        )

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.assets.registries import AssetRegistry

    registry = AssetRegistry()
    for name, path in SIMREADY_ENTRIES.items():
        if registry.is_registered(name, ensure_loaded=False):
            continue
        usd = str(path)

        class _SimReadyProp(Object):
            """Local SimReady prop with class-level usd_path for yam_arena Asset."""

        _SimReadyProp.__name__ = name
        _SimReadyProp.name = name
        _SimReadyProp.tags = ["object", "sim-ready"]
        _SimReadyProp.usd_path = usd
        _SimReadyProp.object_type = ObjectType.RIGID
        registry.register(_SimReadyProp, name)

    import yam_arena.scene as scene

    if getattr(scene._library_item, "_simready_variants", False):
        return
    base = scene._library_item

    def _library_item(name, item):
        obj = base(name, item)
        if str(item.asset).startswith("simready_"):
            addon = dict(obj.spawn_cfg_addon or {})
            variants = dict(addon.get("variants") or {})
            variants.update(SIMREADY_PHYSX_VARIANTS)
            addon["variants"] = variants
            obj.spawn_cfg_addon = addon
            obj.object_cfg = obj._init_object_cfg()
            obj._pose_event_cfg = obj._build_reset_event()
        return obj

    _library_item._simready_variants = True
    scene._library_item = _library_item


# Bin outer size in metres; walls sit on the table floor
BIN_OUT = (0.28, 0.24, 0.045)
WALL_T = 0.008
BIN_IN = (BIN_OUT[0] - 2 * WALL_T, BIN_OUT[1] - 2 * WALL_T)
BIN_X = 0.26
BIN_Y = {"M3": -0.30, "M4": 0.00, "M5": 0.30}
BIN_COLOR = {"M3": (0.95, 0.85, 0.20), "M4": (0.97, 0.97, 0.97), "M5": (0.35, 0.75, 0.40)}
REQUESTED = "M4"

# Free M4 blank matches the working cube-demo envelope; fixtures use extent only
SCREW_SPEC = {
    "M3": {"shank_d": 0.015, "head_d": 0.030, "head_h": 0.010, "shank_l": 0.055, "mass": 0.0016, "extent": 0.030},
    "M4": {"shank_d": 0.020, "head_d": 0.040, "head_h": 0.014, "shank_l": 0.040, "mass": 0.05, "extent": 0.040},
    "M5": {"shank_d": 0.025, "head_d": 0.042, "head_h": 0.014, "shank_l": 0.080, "mass": 0.0054, "extent": 0.042},
}


def _screw_size(label: str) -> tuple[float, float, float]:
    """Upright AABB used by offline self_test ObjectState half-extents."""
    s = SCREW_SPEC[label]
    return (s["head_d"], s["head_d"], s["head_h"] + s["shank_l"])


def _screw_starts(label: str) -> list[tuple[float, float]]:
    """Three screws in each bin; M4 free pick sits forward so open jaws clear neighbors."""
    cy = BIN_Y[label]
    if label == "M4":
        return [(0.30, cy - 0.090), (0.22, cy), (0.30, cy + 0.090)]
    return [(BIN_X, cy + dy) for dy in (-0.090, 0.0, 0.090)]


def _bin(size_label: str) -> list[Box]:
    """Open-top coloured bin walls plus a front size label plaque."""
    cx, cy = BIN_X, BIN_Y[size_label]
    n = size_label.lower()
    lx, ly, h = BIN_OUT
    color = BIN_COLOR[size_label]
    wz = TOP + h / 2
    o_x, o_y = lx / 2 - WALL_T / 2, ly / 2 - WALL_T / 2
    label = Box(
        f"bin_{n}_label",
        pos=(cx - lx / 2 - 0.001, cy, TOP + 0.028),
        size=(0.002, 0.050, 0.022),
        color=(0.08, 0.08, 0.08),
    )
    return [
        Box(f"bin_{n}_xn", pos=(cx - o_x, cy, wz), size=(WALL_T, ly, h), color=color),
        Box(f"bin_{n}_xp", pos=(cx + o_x, cy, wz), size=(WALL_T, ly, h), color=color),
        Box(f"bin_{n}_yn", pos=(cx, cy - o_y, wz), size=(lx - 2 * WALL_T, WALL_T, h), color=color),
        Box(f"bin_{n}_yp", pos=(cx, cy + o_y, wz), size=(lx - 2 * WALL_T, WALL_T, h), color=color),
        label,
    ]


BINS = [p for label in ("M3", "M4", "M5") for p in _bin(label)]

# Free pick is a metal blank; other eight are fixed SimReady fixtures
PICK_SCREW = "screw_m4_1"
SCREWS: list = []
SCREW_HOME: dict[str, str] = {}
SCREW_START_XY: dict[str, tuple[float, float]] = {}
for label in ("M3", "M4", "M5"):
    for i, (sx, sy) in enumerate(_screw_starts(label)):
        name = f"screw_{label.lower()}_{i}"
        free = name == PICK_SCREW
        if free:
            spec = SCREW_SPEC[label]
            length = spec["head_h"] + spec["shank_l"]
            SCREWS.append(
                Box(
                    name,
                    pos=(sx, sy, TOP + length / 2),
                    size=(spec["head_d"], spec["head_d"], length),
                    color=(0.72, 0.72, 0.76),
                    mass=spec["mass"],
                )
            )
        else:
            SCREWS.append(
                Asset(
                    name,
                    SCREW_ASSET[label],
                    xy=(sx, sy),
                    extent=SCREW_SPEC[label]["extent"],
                    mass=None,
                    convex=True,
                )
            )
        SCREW_HOME[name] = label
        SCREW_START_XY[name] = (sx, sy)

M4_NAMES = [n for n, lab in SCREW_HOME.items() if lab == "M4"]
ALL_SCREW_NAMES = list(SCREW_HOME)

TRAY_C = (0.05, 0.00)
TRAY_EXTENT = 0.12
TRAY_IN = (TRAY_EXTENT * 0.70, TRAY_EXTENT * 0.55)
PALM_C = TRAY_C
PALM_TOP = TOP
HAND_TRAY = Asset("hand_tray", TRAY_ASSET, xy=TRAY_C, extent=TRAY_EXTENT)
CARD = Box(
    "request_card",
    pos=(-0.02, 0.00, TOP + 0.030),
    size=(0.002, 0.060, 0.040),
    color=(0.97, 0.97, 0.97),
)
CARD_STAND = Box(
    "request_stand",
    pos=(-0.02, 0.00, TOP + 0.010),
    size=(0.020, 0.020, 0.020),
    color=(0.45, 0.45, 0.48),
)

OBJECTS = [*BINS, *SCREWS, HAND_TRAY, CARD, CARD_STAND]

RELEASE_DIST = 0.03
LIFT_M = 0.03
ON_PALM_Z = 0.004
ON_PALM_Z_BELOW = 0.004
BIN_MARGIN = 0.0
REST_SPEED = 0.01
REST_OMEGA = 0.5

_lifted_m4 = False
_ENV: dict = {"env": None}


def _install_env_hook() -> None:
    """Register SimReady after Kit boots; cache env for gripper release checks."""
    try:
        from yam_arena import inspect_embodiment as ie
    except Exception:  # pragma: no cover
        return

    cls = ie.YamArenaEmbodiment

    if not getattr(cls._ensure_sim, "_simready_hooked", False):
        base_ensure = cls._ensure_sim

        def _ensure_sim(self, *a, **k):
            from isaaclab.app import AppLauncher

            if not getattr(AppLauncher.__init__, "_simready_patched", False):
                orig_init = AppLauncher.__init__

                def __init__(app_self, *aa, **kk):
                    orig_init(app_self, *aa, **kk)
                    register_simready()

                __init__._simready_patched = True
                AppLauncher.__init__ = __init__

            if ie._APP is not None:
                register_simready()
            return base_ensure(self, *a, **k)

        _ensure_sim._simready_hooked = True
        cls._ensure_sim = _ensure_sim

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
    """Track whether the free pick screw was lifted 3 cm above the table."""
    global _lifted_m4
    n = PICK_SCREW
    low_z = float(objects[n].low[2])
    near_home = (
        math.hypot(float(objects[n].pos[0]) - SCREW_START_XY[n][0], float(objects[n].pos[1]) - SCREW_START_XY[n][1])
        < 0.01
    )
    if low_z < TOP + 0.005 and near_home:
        _lifted_m4 = False
    if low_z >= TOP + LIFT_M:
        _lifted_m4 = True


def _resting(obj) -> bool:
    if hasattr(obj, "resting"):
        return bool(obj.resting())
    return float(np.linalg.norm(obj.lin_vel)) < REST_SPEED and float(np.linalg.norm(obj.ang_vel)) < REST_OMEGA


def _on_palm(obj) -> bool:
    """Screw centre inside the tray floor outline, within 4 mm of the tray floor, at rest."""
    hx, hy = TRAY_IN[0] / 2, TRAY_IN[1] / 2
    if abs(float(obj.pos[0]) - TRAY_C[0]) > hx or abs(float(obj.pos[1]) - TRAY_C[1]) > hy:
        return False
    low_z = float(obj.low[2])
    if low_z > PALM_TOP + ON_PALM_Z:
        return False
    if low_z < PALM_TOP - ON_PALM_Z_BELOW:
        return False
    return _resting(obj)


def _in_home_bin(name: str, obj) -> bool:
    """Screw centre still inside its starting bin's scoring outline."""
    label = SCREW_HOME[name]
    cx, cy = BIN_X, BIN_Y[label]
    hx, hy = BIN_IN[0] / 2 + BIN_MARGIN, BIN_IN[1] / 2 + BIN_MARGIN
    return abs(float(obj.pos[0]) - cx) <= hx and abs(float(obj.pos[1]) - cy) <= hy


def _released(objects) -> bool:
    gs = grippers()
    if not gs:
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

    - 0: still holding, wrong size in tray, or more than one screw in the tray
    - 1.0: one requested screw alone in the tray, others in home bins
    - 0.5: one requested screw in the tray with another screw next to the tray
    - 0.25: lifted a requested screw only
    """
    _track_lift(objects)
    if not _released(objects):
        return 0.0
    on_palm = _on_palm_names(objects)
    if len(on_palm) > 1:
        return 0.0
    wrong = [n for n in on_palm if SCREW_HOME[n] != REQUESTED]
    if wrong:
        return 0.0
    m4_on = [n for n in on_palm if SCREW_HOME[n] == REQUESTED]
    if len(m4_on) == 1:
        others_home = all(_in_home_bin(n, objects[n]) for n in ALL_SCREW_NAMES if n != m4_on[0])
        near_palm = [
            n
            for n in ALL_SCREW_NAMES
            if n != m4_on[0]
            and abs(float(objects[n].pos[0]) - TRAY_C[0]) <= TRAY_EXTENT / 2 + 0.03
            and abs(float(objects[n].pos[1]) - TRAY_C[1]) <= TRAY_EXTENT / 2 + 0.03
            and float(objects[n].low[2]) <= PALM_TOP + 0.03
        ]
        if near_palm:
            return 0.5
        if others_home:
            return 1.0
        return 0.5
    if _lifted_m4:
        return 0.25
    return 0.0


def success(objects) -> bool:
    """True when score is full success (1.0)."""
    return score(objects) >= 1.0 - 1e-9


def distance(objects) -> float:
    """Nearest requested screw centre to tray centre, xy plus height above tray floor."""
    best = math.inf
    for n in M4_NAMES:
        p = objects[n]
        dxy = math.hypot(float(p.pos[0]) - TRAY_C[0], float(p.pos[1]) - TRAY_C[1])
        dz = max(0.0, float(p.low[2]) - PALM_TOP)
        best = min(best, math.hypot(dxy, dz))
    return float(best)


TRAVEL_Z = TOP + 0.16
FINGER_APPROACH = -0.025


def demo():
    """Left arm picks the free M4 blank into the tray; right stays clear."""
    from yam_arena.constants import FINGER_CLOSED, PAD_TIP_PAST_TCP
    from yam_arena.demo import pick_place_waypoints

    name = PICK_SCREW
    sx, sy = SCREW_START_XY[name]
    mid = 0.5 * (SCREW_SPEC["M4"]["head_h"] + SCREW_SPEC["M4"]["shank_l"])
    pick_z = TOP + PAD_TIP_PAST_TCP + mid
    place_z = TOP + PAD_TIP_PAST_TCP + SCREW_SPEC["M4"]["head_d"] / 2 + 0.005
    qx, qy = TRAY_C
    left = pick_place_waypoints(
        (sx, sy, pick_z),
        (qx, qy, place_z),
        travel_z=TRAVEL_Z,
        speed=0.7,
    )
    left = [
        Waypoint(2.0, (sx, sy, TRAVEL_Z), FINGER_APPROACH, "above pick"),
        Waypoint(1.5, (sx, sy, pick_z), FINGER_APPROACH, "descend approach"),
        *left[2:],
        Waypoint(3.0, (0.22, 0.28, TRAVEL_Z), FINGER_OPEN, "hold clear for score"),
    ]
    right = [
        Waypoint(1.0, (0.22, -0.28, TRAVEL_Z), FINGER_OPEN, "stay clear"),
        Waypoint(sum(w.duration for w in left), (0.22, -0.28, TRAVEL_Z), FINGER_OPEN, "wait out the handoff"),
    ]
    return {"left": left, "right": right}


def self_test(ObjectState=None) -> dict:
    """Scoring rules on hand-built states; no GPU."""
    if ObjectState is None:
        from yam_arena.task_file import ObjectState

    global _lifted_m4
    half = {n: np.asarray(_screw_size(SCREW_HOME[n])) / 2 for n in ALL_SCREW_NAMES}

    def screw(name, x, y, bottom=TOP, vel=(0.0, 0.0, 0.0)):
        h = half[name]
        pos = np.array([x, y, bottom + h[2]])
        return ObjectState(pos, np.array([0.0, 0.0, 0.0, 1.0]), np.asarray(vel, float), np.zeros(3), pos - h, pos + h)

    def start_scene(**moved):
        base = {n: screw(n, *SCREW_START_XY[n]) for n in ALL_SCREW_NAMES}
        base.update(moved)
        return base

    report = {}

    _lifted_m4 = False
    st = start_scene()
    got, ok = score(st), success(st)
    assert got == 0.0 and not ok
    report["start"] = {"score": got, "success": ok}

    good = start_scene(screw_m4_1=screw("screw_m4_1", PALM_C[0], PALM_C[1], bottom=PALM_TOP))
    _lifted_m4 = False
    got, ok = score(good), success(good)
    assert abs(got - 1.0) < 1e-9 and ok
    report["one_m4_alone"] = {"score": got, "success": ok}

    bad = start_scene(screw_m3_0=screw("screw_m3_0", PALM_C[0], PALM_C[1], bottom=PALM_TOP))
    got, ok = score(bad), success(bad)
    assert got == 0.0 and not ok
    report["wrong_size_on_palm"] = {"score": got, "success": ok}

    two = start_scene(
        screw_m4_1=screw("screw_m4_1", PALM_C[0], PALM_C[1], bottom=PALM_TOP),
        screw_m4_0=screw("screw_m4_0", PALM_C[0] + 0.02, PALM_C[1], bottom=PALM_TOP),
    )
    got, ok = score(two), success(two)
    assert got == 0.0 and not ok
    report["two_m4_in_tray"] = {"score": got, "success": ok}

    beside = start_scene(
        screw_m4_1=screw("screw_m4_1", PALM_C[0], PALM_C[1], bottom=PALM_TOP),
        screw_m4_0=screw("screw_m4_0", PALM_C[0] + TRAY_EXTENT / 2 + 0.01, PALM_C[1], bottom=PALM_TOP),
    )
    got, ok = score(beside), success(beside)
    assert abs(got - 0.5) < 1e-9 and not ok
    report["m4_with_nearby"] = {"score": got, "success": ok}

    sx, sy = SCREW_START_XY["screw_m4_1"]
    lifted = start_scene(screw_m4_1=screw("screw_m4_1", sx, sy, bottom=TOP + LIFT_M + 0.01))
    _lifted_m4 = False
    got, ok = score(lifted), success(lifted)
    assert abs(got - 0.25) < 1e-9 and not ok
    report["lifted_only"] = {"score": got, "success": ok}
    _lifted_m4 = False
    return report


if __name__ == "__main__":
    from yam_arena import ik, station
    from yam_arena.demo import ArmPlan
    from yam_arena.task_file import load

    if missing_assets():
        print("warning: SimReady USD files missing under assets/simready/; scoring/IK still checked.")

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
        "simready_assets": {k: str(v) for k, v in SIMREADY_ENTRIES.items()},
        "missing_simready": missing_assets(),
    }
    print(json.dumps(result, indent=2), flush=True)
