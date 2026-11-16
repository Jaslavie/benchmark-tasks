"""On the station table, an A4 desktop pouch laminator lies along x in the middle of the table, warmed up (its ready lamp
is green), its entry slot facing the right arm. In line with the slot, to the robot's right, a box of card-size
laminating pouches lies on the desk, and on it a loaded pouch: a picture flash card in a jumbo card pouch, sealed along
one short edge. The pouch lies backwards: its open end toward the slot, overhanging the box's edge, and its sealed end
toward the robot's right.

The laminator is fixed. Once any part of the pouch reaches the rollers' nip line inside the slot (the bite), a helper
here draws it through at 10 mm/s if its sealed edge leads, or jams it if an open edge leads or a gripper still holds it.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # lets this file import helpers beside it

import assets as a  # noqa: E402
import steptime  # noqa: E402
from scenekit import UsdBody, on_reset, on_step  # noqa: E402
from yam_arena.constants import FINGER_OPEN, PAD_TIP_PAST_TCP  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

a.build()

SCENE = "station"
TOP = TABLE_TOP[SCENE]
BOX_TOP = a.BOX[2]  # heights below are above the table top
POUCH_XY = (0.35, a.BOX_XY[1] + a.BOX[1] / 2 + 0.045 - a.POUCH[1] / 2)  # open end overhangs the box by 4.5 cm
BACKWARDS = (0.0, 0.0, 1.0, 0.0)  # yaw 180: the pouch's +y (its sealed edge) points at the robot's right
OBJECTS = [
    UsdBody("laminator", usd="assets/laminator.usda", pos=(0.0, 0.0, TOP), mass=None, size=(0.34, 0.11, a.LAM_H),
            friction=(0.4, 0.35)),
    UsdBody("pouch_box", usd="assets/pouch_box.usda", pos=(a.BOX_XY[0], a.BOX_XY[1], TOP + a.BOX[2] / 2 + 0.0002),
            mass=a.BOX_MASS, size=a.BOX, friction=(0.5, 0.45), solver_iters=32),
    UsdBody("pouch", usd="assets/pouch.usda", pos=(POUCH_XY[0], POUCH_XY[1], TOP + BOX_TOP + a.POUCH[2] / 2 + 0.0003),
            rot_xyzw=BACKWARDS, mass=a.POUCH_MASS, size=a.POUCH, friction=(0.45, 0.4), solver_iters=32,
            torsional_patch=0.008),
]

# -- the laminator's feed and the scoring (see spec.md, "How it is scored") ---------------------------------------
HZ = steptime.CONTROL_HZ
SPEED = 0.010  # m/s: 0.6 m/min
JAM_DRAW = 0.02  # an open-edge-first pouch is drawn this far, then stops
HOLD_CHECK = 0.02  # from this much draw on, a gripper closed on the pouch stops it (a jam)
SQUARE, SKEWED = 10.0, 30.0  # degrees between the sealed edge's direction and the feed direction
CLOSED_GAP = 0.085  # m: fingertips nearer than this, round the pouch, are closed on it
FEED_DIR = np.array([0.0, a.IN])  # straight into the slot, square to the entry face (toward the robot's left)


def _rot(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def corners(pos, quat) -> np.ndarray:
    """The pouch's eight corners (station frame, absolute heights)."""
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    return np.asarray(pos, float) + (signs * np.array(a.POUCH) / 2) @ _rot(quat).T


def heading(quat) -> float:
    """Seen from above, degrees between the feed direction and the direction from the pouch's centre to its sealed edge."""
    d = _rot(quat)[:2, 1]
    n = float(np.linalg.norm(d))
    if n < 1e-6:  # standing on edge: no sealed edge leads
        return 180.0
    return math.degrees(math.acos(float(np.clip(d @ FEED_DIR / n, -1.0, 1.0))))


def bite_class(h: float) -> str:
    return "square" if h <= SQUARE else "skewed" if h <= SKEWED else "open"


def at_nip(pos, quat) -> bool:
    """Any part of the pouch has reached the nip line inside the slot."""
    c = corners(pos, quat)
    z = c[:, 2] - TOP
    inside = (c[:, 0] >= a.SLOT_X[0]) & (c[:, 0] <= a.SLOT_X[1]) & (z >= a.SLOT_Z[0] - 0.003) & (z <= a.SLOT_Z[1] + 0.003)
    return bool(np.any(inside & (a.IN * (c[:, 1] - a.NIP_Y) >= 0.0)))


def is_out(pos, quat) -> bool:
    return bool(np.all(a.IN * (corners(pos, quat)[:, 1] - a.BACK_Y) > 0.0))


def closed_on(tcp, gap: float, pos, quat) -> bool:
    """One gripper closed on the pouch: fingertips less than CLOSED_GAP apart, their tips below the pouch's top
    surface, and the point between them over the pouch, seen from above."""
    r = _rot(quat)
    local = r.T @ (np.asarray(tcp, float) - np.asarray(pos, float))
    over = abs(local[0]) <= a.POUCH[0] / 2 and abs(local[1]) <= a.POUCH[1] / 2
    top = float(corners(pos, quat)[:, 2].max())
    return bool(gap < CLOSED_GAP and float(tcp[2]) - PAD_TIP_PAST_TCP < top and over)


_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(step=0, lift=False, phase="free", cls=None, head=None, drawn=0.0, jam=None, pose0=None, closed_now=False,
                  bite_t=None, out_t=None)


_reset_state()


def advance(pos, quat, closed: bool) -> tuple | None:
    """One control step of the laminator: the bite, the draw, a jam or a hold. Returns the pose (pos, quat_xyzw) the
    pouch must be held at this step, or None when it is free."""
    s = _STATE
    s["step"] += 1
    s["closed_now"] = closed
    if s["phase"] == "free":
        if not at_nip(pos, quat):
            return None
        h = heading(quat)
        r = _rot(quat)
        yaw = math.atan2(r[1, 0], r[0, 0])
        s.update(phase="drawing", head=round(h, 2), cls=bite_class(h), bite_t=s["step"] / HZ,
                 pose0=(float(pos[0]), float(pos[1]), yaw))
        if s["cls"] == "open":
            s["jam"] = "open edge first"
    if s["phase"] in ("drawing", "stuck"):
        if s["phase"] == "drawing":
            if s["jam"] and s["drawn"] >= JAM_DRAW:
                s["phase"] = "stuck"
            elif s["drawn"] >= HOLD_CHECK and closed:
                s["jam"] = s["jam"] or "held"
                s["phase"] = "stuck"
            else:
                s["drawn"] += SPEED / HZ
        x0, y0, yaw = s["pose0"]
        p = (x0, y0 + a.IN * s["drawn"], TOP + a.SLOT_Z[0] + a.POUCH[2] / 2 + 0.0002)
        q = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
        if s["phase"] == "drawing" and not s["jam"] and is_out(p, q):
            s["phase"], s["out_t"] = "out", s["step"] / HZ
            return None
        return p, q
    return None


def outcome() -> float:
    s = _STATE
    if s["jam"] or s["cls"] not in ("square", "skewed"):
        return 0.0
    if s["phase"] == "out" or (s["phase"] == "drawing" and not s["closed_now"]):
        return 1.0 if s["cls"] == "square" else 0.5
    return 0.0


def _trace(record: dict) -> None:
    folder = os.environ.get("TASK_SCORE_DIR")
    if not folder:
        return
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "trace.jsonl"), "a") as f:
        f.write(json.dumps(record) + "\n")


def _np(x) -> np.ndarray:
    return getattr(x, "torch", x).detach().cpu().numpy().astype(np.float64)


@on_step
def _laminator(environment) -> None:
    import torch

    from scenekit import arm_joints
    from yam_arena.runtime import tcp_world
    from yam_arena.scene import object_states

    base = environment.unwrapped
    origin = _np(base.scene.env_origins)[0]
    st = object_states(environment, ["pouch"])["pouch"]
    pos, quat = np.asarray(st.pos, float), np.asarray(st.quat_xyzw, float)
    tcps, gaps = {}, {}
    for side in ("left", "right"):
        tcps[side] = tcp_world(base.scene[f"{side}_robot"]) - origin
        fingers = arm_joints(environment, side, ["joint7", "joint8"])
        gaps[side] = float(abs(fingers[0]) + abs(fingers[1]))
    closed = any(closed_on(tcps[s], gaps[s], pos, quat) for s in tcps)
    hold = advance(pos, quat, closed)
    asset = base.scene["pouch"]
    lift = hold is not None
    if lift != _STATE.get("lift", False):  # while the laminator holds the pouch, its weight is carried (no tipping between poses)
        up = torch.tensor([[[0.0, 0.0, a.POUCH_MASS * 9.81 if lift else 0.0]]], dtype=torch.float32, device=asset.device)
        asset.permanent_wrench_composer.set_forces_and_torques_index(forces=up, torques=torch.zeros_like(up), is_global=True)
        _STATE["lift"] = lift
    if hold is not None:
        p, q = hold
        pose = torch.tensor([[p[0] + origin[0], p[1] + origin[1], p[2] + origin[2], *q]], dtype=torch.float32,
                            device=asset.device)
        asset.write_root_pose_to_sim_index(root_pose=pose)
        asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), dtype=torch.float32, device=asset.device))
    s = _STATE
    rec = {"t": round(s["step"] / HZ, 2), "phase": s["phase"], "cls": s["cls"], "head": s["head"], "jam": s["jam"],
           "drawn": round(s["drawn"], 4), "closed": closed, "score": outcome(),
           "pouch": [round(float(v), 4) for v in pos] + [round(heading(quat), 1),
                                                         round(math.degrees(math.acos(float(np.clip(_rot(quat)[2, 2], -1, 1)))), 1)],
           "tcp_r": [round(float(v), 4) for v in tcps["right"]], "gap_r": round(gaps["right"], 4)}
    if os.environ.get("TASK_SCORE_DIR"):
        rec["q_r"] = [round(float(v), 3) for v in arm_joints(environment, "right", [f"joint{i}" for i in range(1, 7)])]
        box = object_states(environment, ["pouch_box"])["pouch_box"]
        rec["box"] = [round(float(v), 4) for v in box.pos]
    _trace(rec)


def success(objects) -> bool:
    """The pouch was bitten square, sealed edge first, and has come out of the back; the harness ends the trial at 1.0
    after 1 s of this."""
    steptime.tick()  # the harness calls success() once per control step
    s = _STATE
    return bool(not s["jam"] and s["cls"] == "square" and s["phase"] == "out")


def score(objects) -> float:
    """0 after a jam (an open edge first, or a gripper closed on the pouch once 2 cm is drawn). 1.0 for a square bite and
    the pouch out of the back, 0.5 for a skewed bite (10 to 30 degrees) and out; a pouch still being drawn at the run
    limit, with no gripper closed on it, counts as out. Otherwise 0."""
    return outcome()


# -- scripted demo (open loop: it reads no simulation state) -----------------------------------------------------
LEFT, RIGHT = -math.pi / 2, math.pi / 2  # straight-down grips with the fingers closing along x
GRIP_FROM_END = 0.018  # the grip's centre from the pouch's open end
TIP_BELOW = 0.004  # pad tips this far below the pouch's underside
HOLD = -(a.POUCH[0] - 0.008) / 2  # pads commanded 8 mm narrower than the pouch: a firm pinch
TRAVEL = TOP + 0.10
OPEN_Y = POUCH_XY[1] + a.POUCH[1] / 2  # the open end at the start
GRIP = (POUCH_XY[0], OPEN_Y - GRIP_FROM_END, TOP + BOX_TOP - TIP_BELOW + PAD_TIP_PAST_TCP)
FEED_Z = TOP + a.SLOT_Z[0] + 0.0005 - TIP_BELOW + PAD_TIP_PAST_TCP  # the held pouch 0.5 mm above the tunnel floor
LEAD = a.POUCH[1] - GRIP_FROM_END  # grip to the far (leading) end after the turn
PARK = (0.34, -0.18, TOP + 0.20)  # its wrist camera then looks along the laminator's slot
PARK_YAW = -math.pi / 3
LOOK = (POUCH_XY[0], POUCH_XY[1], TOP + 0.15)  # over the pouch's centre: the wrist camera sees all of it


def _grip_open_end(yaw: float, label: str = "") -> list[Waypoint]:
    x, y, z = GRIP
    return [
        Waypoint(2.0, LOOK, FINGER_OPEN, f"look at the pouch{label}", yaw),
        Waypoint(1.0, LOOK, FINGER_OPEN, "look", yaw),
        Waypoint(1.2, (x, y, TRAVEL), FINGER_OPEN, "over the open end", yaw),
        Waypoint(1.0, (x, y, z + 0.02), FINGER_OPEN, "near", yaw),
        Waypoint(0.5, (x, y, z + 0.02), FINGER_OPEN, "settle", yaw),
        Waypoint(0.8, (x, y, z), FINGER_OPEN, "down", yaw),
        Waypoint(0.4, (x, y, z), FINGER_OPEN, "settle", yaw),
        Waypoint(0.6, (x, y, z), HOLD, "pinch", yaw),
        Waypoint(1.2, (x, y, TRAVEL), HOLD, "lift", yaw),
    ]


def _feed(yaw: float, lead: float, release: bool = True, skew_deg: float = 0.0) -> list[Waypoint]:
    """From TRAVEL over the slot line, lower the held pouch to the tunnel and push it in until its far end reaches the
    nip (lead: grip to the far end), then let go and rise."""
    x = a.SLOT_CX
    y_start = a.FRONT_Y - a.IN * (0.025 + lead)  # the far end 2.5 cm short of the entry face
    y_bite = a.NIP_Y - a.IN * (lead * math.cos(math.radians(skew_deg)) - 0.002)
    out = [
        Waypoint(1.2, (x, y_start, TRAVEL), HOLD, "over the slot line", yaw),
        Waypoint(1.2, (x, y_start, FEED_Z + 0.015), HOLD, "lower", yaw),
        Waypoint(0.5, (x, y_start, FEED_Z + 0.015), HOLD, "settle", yaw),
        Waypoint(0.8, (x, y_start, FEED_Z), HOLD, "to the slot's height", yaw),
        Waypoint(0.4, (x, y_start, FEED_Z), HOLD, "settle", yaw),
        Waypoint(2.5, (x, y_bite, FEED_Z), HOLD, "feed", yaw),
    ]
    if release:
        out += [Waypoint(0.5, (x, y_bite, FEED_Z), FINGER_OPEN, "let go", yaw),
                Waypoint(1.0, (x, y_bite, TRAVEL), FINGER_OPEN, "rise", yaw),
                Waypoint(2.5, PARK, FINGER_OPEN, "park", PARK_YAW),
                Waypoint(26.0, PARK, FINGER_OPEN, "wait while the rollers draw it through", PARK_YAW)]
    else:
        out += [Waypoint(10.0, (x, y_bite, FEED_Z), HOLD, "keep holding", yaw)]
    return out


def demo():
    """The right arm alone; the left stays at READY. It pinches the pouch across its width where its open end overhangs
    the box, lifts it, turns it 180 degrees in the grasp (yaw -90 to +90 through 0) so that the sealed edge points at the
    slot, lowers it to the slot's height and pushes it in until the sealed edge reaches the nip, then opens and rises; the
    rollers draw it through. Variants (TASK_DEMO), to check the scorer: failure feeds it open end first (it is set down
    with its sealed end over the box's other edge, pinched there and carried to the slot without a turn); held keeps
    holding; skewed feeds it 20 degrees off square; turned turns it and sets it back on the box without feeding;
    nothing does nothing."""
    mode = os.environ.get("TASK_DEMO", "success")
    x, y, z = GRIP
    if mode == "nothing":
        return {"right": [Waypoint(4.0, PARK, FINGER_OPEN, "wait", RIGHT)]}
    turn = [Waypoint(1.2, (x, y, TRAVEL), HOLD, "turn", 0.0), Waypoint(1.2, (x, y, TRAVEL), HOLD, "turn", RIGHT)]
    if mode == "failure":
        # set the pouch down unturned with its sealed end over the box's far (-y) edge, then pinch that end (the same
        # grip at yaw +90, which keeps the wrist in range) and feed it as it lies
        sealed_over = a.BOX_XY[1] - a.BOX[1] / 2 - 0.045
        y_set = sealed_over + LEAD  # the grip (near the open end) when the pouch lies there
        z_set = TOP + BOX_TOP + 0.005 - TIP_BELOW + PAD_TIP_PAST_TCP  # 5 mm above the box: it drops onto it
        y_g2 = sealed_over + GRIP_FROM_END
        plan = _grip_open_end(LEFT) + [
            Waypoint(1.5, (x, y_set, TRAVEL), HOLD, "carry over the box", LEFT),
            Waypoint(1.2, (x, y_set, z_set), HOLD, "lower", LEFT),
            Waypoint(0.4, (x, y_set, z_set), HOLD, "settle", LEFT),
            Waypoint(0.5, (x, y_set, z_set), FINGER_OPEN, "drop it on the box", LEFT),
            Waypoint(0.8, (x, y_set, TRAVEL), FINGER_OPEN, "rise", LEFT),
            Waypoint(3.0, (x, y_g2, TRAVEL), FINGER_OPEN, "over the sealed end", RIGHT),
            Waypoint(1.0, (x, y_g2, z + 0.02), FINGER_OPEN, "near", RIGHT),
            Waypoint(0.5, (x, y_g2, z + 0.02), FINGER_OPEN, "settle", RIGHT),
            Waypoint(0.8, (x, y_g2, z), FINGER_OPEN, "down", RIGHT),
            Waypoint(0.4, (x, y_g2, z), FINGER_OPEN, "settle", RIGHT),
            Waypoint(0.6, (x, y_g2, z), HOLD, "pinch", RIGHT),
            Waypoint(1.2, (x, y_g2, TRAVEL), HOLD, "lift", RIGHT),
        ] + _feed(RIGHT, LEAD)
        return {"right": plan}
    if mode == "turned":
        # turned about the grip, the pouch reaches 8.7 cm toward the slot: set it back on the box, sealed end toward the
        # slot, its open end now overhanging the box's far (-y) edge by 4.5 cm, and leave it there
        y_set = a.BOX_XY[1] - a.BOX[1] / 2 - 0.045 + GRIP_FROM_END
        plan = _grip_open_end(LEFT) + turn + [
            Waypoint(1.5, (x, y_set, TRAVEL), HOLD, "carry", RIGHT),
            Waypoint(1.2, (x, y_set, z + 0.0005), HOLD, "lower", RIGHT),
            Waypoint(0.5, (x, y_set, z + 0.0005), FINGER_OPEN, "let go", RIGHT),
            Waypoint(0.8, (x, y_set, TRAVEL), FINGER_OPEN, "rise", RIGHT),
            Waypoint(1.5, PARK, FINGER_OPEN, "park", RIGHT),
            Waypoint(5.0, PARK, FINGER_OPEN, "wait", RIGHT)]
        return {"right": plan}
    if mode == "skewed":
        yaw = RIGHT + math.radians(20.0)
        plan = _grip_open_end(LEFT) + [Waypoint(1.2, (x, y, TRAVEL), HOLD, "turn", 0.0),
                                       Waypoint(1.8, (x, y, TRAVEL), HOLD, "turn", yaw)] + _feed(yaw, LEAD, skew_deg=20.0)
        return {"right": plan}
    plan = _grip_open_end(LEFT) + turn + _feed(RIGHT, LEAD, release=mode != "held")
    return {"right": plan}
