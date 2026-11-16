"""On a felt mat lies a picture frame, face down: an oak frame 30 x 40 cm with a new picture in it. Its back (a hardboard
backboard on the mount and foam behind it) springs up 12 mm proud of the frame until it is pressed down. Four brass turn
buttons (swivel clips), two on each long side of the frame, lie turned along the frame, open.

Task: close the back. One hand presses the backboard down flush; the other turns the clips on its side a quarter turn,
over the board's edge. A clip cannot pass over a board that is not pressed down (it stands proud, in the clip's way).
Then the hands swap: the other hand presses the board down while the first turns the clips on the other side. Then both
hands let go, the board staying down, held by its clips, and both arms go back to where they started.

Run it:  inspect-robots run --embodiment yam_arena -E task=close_a_picture_frame_back_and_turn_its_four_clips.py \\
           --instruction "..." --scorer task_score ...
"""

from __future__ import annotations

import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import bodykit  # noqa: E402
from bodykit import (Body, BoxShape, CylShape, allow_matrix_grips, as_numpy, hinge, on_reset, on_step,  # noqa: E402
                     rotation, set_hinge_drive, set_slide_target, thread)
from yam_arena.constants import FINGER_CLOSED, TCP_OFFSET  # noqa: E402
from yam_arena.demo import Waypoint, grip_rotation  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

allow_matrix_grips()

SCENE = "station"
TOP = TABLE_TOP[SCENE]

# -- the frame, face down: its origin is the middle of its underside (on the mat); x toward the far side, y left -------
FX, FY = 0.30, 0.0
MAT_T = 0.004
FZ = TOP + MAT_T
OUT = (0.30, 0.40)  # outside: along x, along y
MOULD_W, MOULD_H = 0.030, 0.025  # the moulding's width and its height (the frame's thickness), face down
INNER = (OUT[0] - 2 * MOULD_W, OUT[1] - 2 * MOULD_W)
FRAME_MASS = 0.8

# -- the back: the backboard on the mount and foam behind it, one slab; seated, its top is just under the frame's back --
BACK = (INNER[0] - 0.004, INNER[1] - 0.004, 0.012)
SEAT_TOP = MOULD_H - 0.001
PROUD = 0.012  # sprung up this far until pressed down
BACK_K, BACK_D = 250.0, 8.0  # N/m, toward ...
BACK_PUSH = 0.035  # ... 3.5 cm up: it stands 12 mm proud, and takes about 9 N to press flush
TILT_K, TILT_D = 0.25, 0.01  # N m per degree (and per degree/s): it tilts back level, like the foam behind it
BACK_MASS = 0.25

# -- the clips: brass turn buttons on the frame's back, two on each long side; each swivels about its screw -----------
CLIP_L, CLIP_W, CLIP_T = 0.032, 0.012, 0.003
KNOB = (0.008, 0.008, 0.009)  # a raised knob near its end, to push it round by
KNOB_AT = 0.024  # from the screw
SIDE_Y = OUT[1] / 2 - MOULD_W / 2  # the screws on the moulding's middle line
CLIP_X = 0.08  # either side of the middle (x)
# name: (x off the middle, side (+1 left, -1 right), which way it lies open (+1: toward +x), the quarter turn's sign)
CLIPS = {"clip_rn": (-CLIP_X, -1, -1, -1), "clip_rf": (CLIP_X, -1, 1, 1),
         "clip_ln": (-CLIP_X, 1, -1, 1), "clip_lf": (CLIP_X, 1, 1, -1)}
CLIP_HOLD = 0.002  # N m per degree: it stays where it is left ...
SLIP = 3.0  # ... pushed more than this past its hold, it gives
CLIP_DRAG = 0.0002
CLIP_MASS = 0.02
OVER = 80.0  # a clip turned at least this far lies over the board's edge

OAK, OAK_D, BOARD_C, MOUNT_C, BRASS, FELT = (0.62, 0.45, 0.28), (0.45, 0.31, 0.18), (0.52, 0.36, 0.22), \
    (0.95, 0.94, 0.90), (0.80, 0.65, 0.30), (0.18, 0.30, 0.22)

# -- scoring ----------------------------------------------------------------------------------------------------------
FLUSH = 0.003  # the board's middle within this of seated ...
LEVEL = 1.5  # ... and tilted at most this many degrees
CLEAR = 0.05  # each TCP at least this far above the board, or off the frame
HOME_WITHIN = 0.05
SUCCESS_HOLD_S = 1.0


def _clip_pivot(name: str) -> np.ndarray:
    dx, side, _, _ = CLIPS[name]
    return np.array([FX + dx, FY + side * SIDE_Y, FZ + MOULD_H])


def _mat() -> Body:
    return Body.build("mat", pos=(FX, FY, TOP), mass=1.0, kinematic=True, color=FELT, friction=(0.8, 0.7),
                      shapes=[BoxShape((0.0, 0.0, MAT_T / 2), (OUT[0] + 0.08, OUT[1] + 0.08, MAT_T), FELT)])


def _frame() -> Body:
    hx, hy = OUT[0] / 2, OUT[1] / 2
    shapes = [BoxShape((s * (hx - MOULD_W / 2), 0.0, MOULD_H / 2), (MOULD_W, OUT[1], MOULD_H), OAK) for s in (-1, 1)]
    shapes += [BoxShape((0.0, s * (hy - MOULD_W / 2), MOULD_H / 2), (OUT[0] - 2 * MOULD_W, MOULD_W, MOULD_H), OAK)
               for s in (-1, 1)]
    shapes += [BoxShape((s * (hx - MOULD_W / 2), 0.0, MOULD_H + 0.0003), (0.004, OUT[1] - 0.01, 0.0006), OAK_D,
                        collide=False) for s in (-1, 1)]  # a darker groove along the long sides (wood look)
    return Body.build("frame", pos=(FX, FY, FZ), mass=FRAME_MASS, shapes=shapes, friction=(0.7, 0.6), color=OAK,
                      com=(0.0, 0.0, MOULD_H / 2))


def _back() -> Body:
    """The backboard (hardboard) on the mount and foam, one slab; origin at the middle of its top."""
    shapes = [BoxShape((0.0, 0.0, -0.0015), (BACK[0], BACK[1], 0.003), BOARD_C),
              BoxShape((0.0, 0.0, -0.003 - (BACK[2] - 0.003) / 2), (BACK[0], BACK[1], BACK[2] - 0.003), MOUNT_C),
              BoxShape((0.0, 0.0, 0.0002), (0.06, 0.03, 0.0005), (0.90, 0.88, 0.80), collide=False)]  # a label
    return Body.build("back", pos=(FX, FY, FZ + SEAT_TOP + PROUD), mass=BACK_MASS, shapes=shapes, friction=(0.8, 0.7),
                      color=BOARD_C)


def _clip(name: str) -> Body:
    _, _, lie, _ = CLIPS[name]
    shapes = [BoxShape((lie * (CLIP_L / 2 - 0.005), 0.0, CLIP_T / 2), (CLIP_L, CLIP_W, CLIP_T), BRASS),
              BoxShape((lie * KNOB_AT, 0.0, CLIP_T + KNOB[2] / 2), KNOB, BRASS),
              CylShape((0.0, 0.0, CLIP_T + 0.0008), 0.004, 0.0016, (0.55, 0.45, 0.22), collide=False)]  # the screw
    return Body.build(name, pos=tuple(_clip_pivot(name).tolist()), mass=CLIP_MASS, shapes=shapes, friction=(0.6, 0.5),
                      color=BRASS, com=(lie * 0.012, 0.0, CLIP_T / 2))


OBJECTS = [_mat(), _frame(), _back(), *[_clip(n) for n in CLIPS]]
# the back slides up and down in the frame on its spring, and tilts a little (a corner held by a clip, the rest springs up)
thread("back_spring", "frame", "back", anchor0=(0.0, 0.0, SEAT_TOP), axis0=(0.0, 0.0, 1.0), anchor1=(0.0, 0.0, 0.0),
       axis1=(0.0, 0.0, 1.0), slide_limits=(0.0, PROUD), turn_damping=50.0, slide_stiffness=BACK_K,
       slide_damping=BACK_D, slide_max_force=100.0, play=0.001, tilt_deg=6.0, always_on=True)
_author_threads = bodykit._author_threads


def _author_threads_with_tilt(stage, prim_path: str, spec) -> None:
    """bodykit's threads, and for the back a spring on its tilt (the foam under it pushes it back level)."""
    _author_threads(stage, prim_path, spec)
    if spec.name != "back":
        return
    from pxr import UsdPhysics

    prim = stage.GetPrimAtPath(f"{prim_path}/back_spring")
    for dof in ("rotY", "rotZ"):
        drive = UsdPhysics.DriveAPI.Apply(prim, dof)
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(TILT_K)
        drive.CreateDampingAttr(TILT_D)
        drive.CreateTargetPositionAttr(0.0)
        drive.CreateMaxForceAttr(1.0e6)


bodykit._author_threads = _author_threads_with_tilt
for _n, (_dx, _side, _lie, _sign) in CLIPS.items():
    hinge(f"{_n}_screw", "frame", _n, anchor=tuple(_clip_pivot(_n).tolist()), axis="Z",
          limits_deg=(-95.0, 3.0) if _sign < 0 else (-3.0, 95.0), spring=CLIP_HOLD, damping=CLIP_DRAG, target_deg=0.0)


# -- followed after every control step --------------------------------------------------------------------------------

def _fresh() -> dict:
    return {"hold": {n: 0.0 for n in CLIPS}, "turned": {n: 0.0 for n in CLIPS}, "tcps": None, "s": {}}


_EP = _fresh()
_DEBUG = {"on": bool(os.environ.get("FRAME_DEBUG")), "steps": 0}


@on_reset
def _new_episode() -> None:
    _EP.clear()
    _EP.update(_fresh())
    _DEBUG["steps"] = 0
    set_slide_target("back_spring", BACK_PUSH)


def _tcps(base, origin) -> list[np.ndarray]:
    out = []
    for arm in ("left_robot", "right_robot"):
        robot = base.scene[arm]
        i = next(j for j, n in enumerate(robot.body_names) if n == "gripper" or n.endswith("/gripper"))
        g = as_numpy(robot.data.body_link_pose_w)[0][i]
        out.append(g[:3] - origin + rotation(g[3:7]) @ np.array(TCP_OFFSET))
    return out


def _yaw(rot) -> float:
    return math.degrees(math.atan2(float(rot[1, 0]), float(rot[0, 0])))


@on_step
def _follow(env) -> None:
    base = env.unwrapped
    origin = as_numpy(base.scene.env_origins[0])
    fr = as_numpy(base.scene["frame"].data.root_link_pose_w)[0]
    f_pos, f_rot = fr[:3] - origin, rotation(fr[3:7])
    for n in CLIPS:
        c = as_numpy(base.scene[n].data.root_link_pose_w)[0]
        a = (_yaw(rotation(c[3:7])) - _yaw(f_rot) + 180.0) % 360.0 - 180.0
        _EP["turned"][n] = a
        if abs(a - _EP["hold"][n]) > SLIP:  # pushed past its hold: it gives, and stays where it is left
            _EP["hold"][n] = a
            set_hinge_drive(f"{n}_screw", target_deg=a)
    bk = as_numpy(base.scene["back"].data.root_link_pose_w)[0]
    b_pos, b_rot = bk[:3] - origin, rotation(bk[3:7])
    rel = f_rot.T @ (b_pos - f_pos)
    proud = float(rel[2]) - SEAT_TOP
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, float((f_rot.T @ b_rot)[2, 2])))))
    _EP["tcps"] = _tcps(base, origin)
    clear = True
    for t in _EP["tcps"]:
        v = f_rot.T @ (t - f_pos)
        on = abs(float(v[0])) <= OUT[0] / 2 + 0.02 and abs(float(v[1])) <= OUT[1] / 2 + 0.02
        clear &= (not on) or float(v[2]) - (SEAT_TOP + proud) >= CLEAR
    _EP["s"] = {"proud": proud, "tilt": tilt, "clear": clear}
    if _DEBUG["on"]:
        _DEBUG["steps"] += 1
        if _DEBUG["steps"] % 10 == 0:
            tl, tr = _EP["tcps"]
            turned = " ".join(f"{n[5:]}={_EP['turned'][n]:5.1f}" for n in CLIPS)
            print(f"[frame] t={_DEBUG['steps'] / 20:5.1f}s proud={proud * 1000:5.1f}mm tilt={tilt:4.1f} {turned} "
                  f"frame={np.round(f_pos * 1000).astype(int).tolist()} yaw={_yaw(f_rot):4.1f} clear={clear} "
                  f"L={np.round(tl * 1000).astype(int).tolist()} R={np.round(tr * 1000).astype(int).tolist()} "
                  f"score={_points(_status())}", file=sys.stderr, flush=True)


# -- scoring --------------------------------------------------------------------------------------------------------

def _status() -> dict:
    s = _EP["s"]
    over = sum(1 for n, (_, _, _, sign) in CLIPS.items() if sign * _EP["turned"][n] >= OVER)
    flush = s.get("proud", 1.0) <= FLUSH and s.get("tilt", 90.0) <= LEVEL
    return {"over": over, "flush": flush, "clear": bool(s.get("clear")), "home": _arms_home()}


def _points(s) -> float:
    if s["over"] == 4 and s["flush"] and s["clear"] and s["home"]:
        return 1.0
    if s["over"] == 4 and s["flush"]:
        return 0.5
    return 0.25 if s["over"] >= 2 else 0.0


def score(objects) -> float:
    """1.0 with all four clips turned over the back (each at least 80 degrees from where it lay), the back flush (its
    middle within 3 mm of seated, tilted at most 1.5 degrees) and both hands off it (each TCP at least 5 cm above the
    back, or off the frame) and both arms back where they started (each TCP within 5 cm of its start); 0.5 with all four
    over and the back flush otherwise; 0.25 with at least two clips turned
    over; else 0.0."""
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
    """Full score with the frame, its back and its clips still."""
    still = all(objects[n].resting() for n in ("frame", "back", *CLIPS))
    return score(objects) >= 1.0 and still


# -- scripted demo ----------------------------------------------------------------------------------------------------
# The left hand comes down on the middle of the back, fingers closed, and presses it flush. The right hand turns the
# two clips on the right side a quarter turn each, over the back's edge: fingertips down beside each clip's knob, then
# round. The left lets go (the left side of the back springs up) and moves over to the left side; the right presses
# the back flush in the middle and the left turns the two clips on its side. Both let go and rise; the back stays down,
# and both arms go back to where they started.

PRESS_GRIP = grip_rotation(0.0)  # straight down, fingers closed
TIP = 0.013  # closed fingertips below the TCP
PRESS = 0.015  # aimed this far below flush
PUSH_Z = CLIP_T + KNOB[2] / 2 + 0.0015  # fingertips at the knob's middle, above the clip's plate
BEHIND = 0.012  # the fingertips this far behind the knob's middle, along the way it turns
OVERTURN = 8.0  # push it round this far past a quarter turn (the arm gives)
LEAD_S, LEAD_STEPS = 3.0, 15
STEPS = 8


def _good_ik(arm: str, pos, rot) -> np.ndarray:
    import itertools

    from yam_arena import ik, station

    best = None
    for seed in itertools.product((-0.5, 0.0, 0.5), (0.8, 1.3), (0.4, 1.0), (-0.8, 0.0, 0.8), (-1.4, 0.0, 1.4), (0.0,)):
        q, pe, re = ik.solve(np.asarray(pos, float), ik.matrix_to_quat_xyzw(rot), np.array(seed),
                             base_pos=station.BASES[arm], iters=200)
        if pe < 0.001 and re < math.radians(0.5):
            room = float(np.minimum(q - ik.LOWER, ik.UPPER - q).min())
            if best is None or room > best[1]:
                best = (q, room)
    if best is None:
        raise ValueError(f"no IK solution for the {arm} grip at {pos}")
    return best[0]


def _joint_path(arm: str, q0, q1, label: str) -> list[Waypoint]:
    from yam_arena import ik, station

    out = []
    for k in range(1, LEAD_STEPS + 1):
        p, r = ik.tcp_pose(q0 + (q1 - q0) * k / LEAD_STEPS, station.BASES[arm])
        out.append(Waypoint(LEAD_S / LEAD_STEPS, tuple(float(v) for v in p), FINGER_CLOSED, label, r))
    return out


def _lead_in(arm: str, pos, label: str) -> list[Waypoint]:
    from yam_arena.constants import READY

    return _joint_path(arm, np.array(READY, float), _good_ik(arm, pos, PRESS_GRIP), label)


def _go_home(arm: str, pos) -> list[Waypoint]:
    from yam_arena.constants import READY

    return _joint_path(arm, _good_ik(arm, pos, PRESS_GRIP), np.array(READY, float), "back to where it started")


def _w(t, pos, label) -> Waypoint:
    return Waypoint(t, tuple(float(v) for v in pos), FINGER_CLOSED, label, PRESS_GRIP)


def _press_at(y: float) -> np.ndarray:
    return np.array([FX, y, FZ + SEAT_TOP + TIP - PRESS])


def _push_point(name: str, deg: float) -> np.ndarray:
    """The TCP pushing clip ``name`` round, the clip turned ``deg`` (signed, its own way) from where it lay."""
    _, _, lie, sign = CLIPS[name]
    p = _clip_pivot(name)
    a = math.radians(deg)
    arm = np.array([lie * math.cos(a), lie * math.sin(a), 0.0])  # along the clip, from its screw
    way = sign * np.array([-arm[1], arm[0], 0.0])  # the way the knob moves as it turns
    return p + KNOB_AT * arm - BEHIND * way + [0.0, 0.0, PUSH_Z + TIP]


def _turn(name: str) -> list[Waypoint]:
    _, _, _, sign = CLIPS[name]
    start = _push_point(name, 0.0)
    out = [_w(1.0, start + [0.0, 0.0, 0.04], "over the clip"),
           _w(0.8, start, "fingertips down beside its knob")]
    for k in range(1, STEPS + 1):
        out.append(_w(0.35, _push_point(name, sign * (90.0 + OVERTURN) * k / STEPS),
                      "turn it a quarter turn, over the back's edge"))
    out.append(_w(0.6, out[-1].pos + np.array([0.0, 0.0, 0.04]), "lift off it"))
    return out


def _plan():
    """Both arms' waypoints, timed against each other."""
    # left presses the middle; right turns its two clips
    left = _lead_in("left", _press_at(FY) + [0.0, 0.0, 0.05], "over the middle of the back")
    left += [_w(1.0, _press_at(FY), "press the back flush")]
    pressed = sum(w.duration for w in left)
    right = _lead_in("right", _push_point("clip_rn", 0.0) + [0.0, 0.0, 0.04], "over the near clip on the right")
    right.append(_w(max(pressed + 0.3 - LEAD_S, 0.3), right[-1].pos, "wait for the back to be pressed down"))
    right += _turn("clip_rn")[1:] + _turn("clip_rf")
    t_right = sum(w.duration for w in right)
    left.append(_w(t_right - pressed + 0.3, _press_at(FY), "hold it down while the clips turn"))
    # swap: left lets go and moves over its clips; right presses the middle
    left += [_w(0.8, _press_at(FY) + [0.0, 0.0, 0.06], "let go: the left side springs up"),
             _w(1.2, _push_point("clip_ln", 0.0) + [0.0, 0.0, 0.04], "over the near clip on the left")]
    t_left = sum(w.duration for w in left)
    right += [_w(max(t_left - 1.0 - t_right, 0.3), _press_at(FY) + [0.0, 0.0, 0.05], "over the middle of the back"),
              _w(1.0, _press_at(FY), "press the back flush")]
    pressed = sum(w.duration for w in right)
    left.append(_w(max(pressed + 0.3 - t_left, 0.3), left[-1].pos, "wait for the back to be pressed down"))
    left += _turn("clip_ln")[1:] + _turn("clip_lf")
    t_left = sum(w.duration for w in left)
    right.append(_w(t_left - pressed + 0.3, _press_at(FY), "hold it down while the clips turn"))
    up_r = _press_at(FY) + [0.0, 0.0, 0.06]
    right.append(_w(0.8, up_r, "let go: the back stays down, held by its clips"))
    right += _go_home("right", up_r)
    left += _go_home("left", left[-1].pos)
    total = max(sum(w.duration for w in right), sum(w.duration for w in left)) + 3.0
    for plan in (left, right):
        plan.append(Waypoint(total - sum(w.duration for w in plan), plan[-1].pos, FINGER_CLOSED,
                             "hold still: the trial ends once the full score has held 1 s with both arms home",
                             plan[-1].yaw))
    return left, right


def demo():
    left, right = _plan()
    return {"left": left, "right": right}
