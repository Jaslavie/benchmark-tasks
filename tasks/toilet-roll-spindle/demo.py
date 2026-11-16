"""Scripted full-score demo for the toilet-roll task, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 3000] [--no-record]

It is closed loop. It plans from the fixed start layout (where the holder's sockets and the tray are) and, while it
runs, reads object poses a camera can see (the spindle's two ends, the empty tube and the roll) and the arms' own
joint angles. It never reads forces, contact flags or physical parameters.

1. The left arm grips the empty tube with a light squeeze and pulls it toward the robot: the tube drags the spindle
   forward, the spindle's domed ends ride out over the sockets' chamfered front edges and it comes out. The left arm
   lays the tube, with the spindle in it, in the tray and moves away.
2. The right arm grips the spindle's right end where it sticks out of the tube, lifts it to the middle of the bore
   and draws it out sideways; the tube stays in the tray. It lays the spindle on the table and takes it again 12 mm
   in from its tip.
3. It pushes the spindle through the core of the new roll lying on the table, lets go, takes the right end again from
   the front (fingers closing up and down, 20 mm in from the tip) and lifts the roll on the spindle, correcting the
   droop it sees in the spindle's pose.
4. It turns the spindle 20 degrees (right end toward the robot), brings the left end in between the posts and into
   the left socket, pushes the spindle 13 mm shorter against that socket and swings the right end round until the
   spindle lies across the holder, its right tip 2 mm inside the right post.
5. It lets go: the spindle springs out into the right socket. The hand backs out and both arms move clear.

Variants (DEMO_VARIANT=...): nothing (both arms stay still), early_release (let go as soon as the left end is in its
socket, without squeezing: the naive attempt), no_squeeze (swing the right end in without squeezing the spindle
shorter), lift_only (stop with the roll held on the spindle in the air).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

CONTROL_HZ = 20
YAW = math.radians(20)        # the spindle is held turned this far (right end toward the robot) for the first end
SQUEEZE = 0.013               # how much shorter the right arm pushes the spindle against the left socket
GRIP_END = 0.012              # TCP at most this far in from the right tip when holding the spindle
GRIP_FIT = 0.0205             # TCP this far in from the right tip for the fitting grip (fingers 4 mm short of it)
FRONT_PITCH = math.radians(35)  # the fitting grip comes from the front, pointing this far down
PAD_HALF = 0.0165             # half the width of a finger pad along the spindle where it holds it
REGRIP_X, REGRIP_Y = 0.175, -0.215  # where the right arm lays the spindle down to take a better hold
TUBE_GRIP = 0.40              # finger command for the cardboard tube (about 38 mm apart: a light squeeze)


def _rz(a) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class Arm:
    """One arm: smooth Cartesian moves (IK on the URDF), each starting where the last command left it."""

    def __init__(self, name: str, base_pos):
        from yam_arena import ik
        from yam_arena.constants import READY

        self.name, self.base = name, np.asarray(base_pos, float)
        self.q = np.array(READY, float)
        self.grip = 1.0
        self.pos, self.rot = ik.tcp_pose(self.q, self.base)
        self.cmd = np.r_[self.q, self.grip]
        self.worst_mm = 0.0
        self.worst_at = ""
        self.bias = np.zeros(3)     # added to commanded positions so the real TCP lands on the target
        self.track = True           # update the bias from the arm's measured joints (free motion only)
        self.actual = self.pos.copy()
        self.actual_rot = self.rot.copy()

    def measured(self, q_arm) -> None:
        """Feed the arm's measured joint angles: the TCP they put the hand at (forward kinematics)."""
        from yam_arena import ik

        self.actual, self.actual_rot = ik.tcp_pose(np.asarray(q_arm, float), self.base)
        if self.track and self.grip > 0.5:   # frozen while the fingers are closed on something
            self.bias = np.clip(self.bias + 0.1 * (self.pos - self.actual), -0.015, 0.015)

    def _solve(self, pos, rot, label=""):
        from yam_arena import ik

        q, pe, re = ik.solve(np.asarray(pos) + self.bias, ik.matrix_to_quat_xyzw(rot), self.q, base_pos=self.base,
                             iters=60)
        self.q, self.pos, self.rot = q, np.asarray(pos, float), rot
        err = pe * 1000 + 20 * re       # mm, plus 1 mm per 0.05 rad (about 3 degrees) of wrist error
        if err > self.worst_mm:
            self.worst_mm, self.worst_at = err, label
        self.cmd = np.r_[q, self.grip]

    def move(self, pos, rot, duration: float, grip: float | None = None, label: str = ""):
        """Generator: go to ``pos``/``rot`` (world TCP); ``grip`` 1 open .. 0 closed (None keeps it)."""
        from yam_arena import ik

        p0, r0, g0 = self.pos.copy(), self.rot.copy(), self.grip
        g1 = g0 if grip is None else grip
        n = max(1, round(duration * CONTROL_HZ))
        for k in range(1, n + 1):
            s = ik.smoothstep(k / n)
            self.grip = g0 + s * (g1 - g0)
            self._solve(p0 + s * (np.asarray(pos) - p0), ik.slerp_matrix(r0, rot, s), label)
            yield

    def set_grip(self, grip: float, duration: float = 0.8):
        yield from self.move(self.pos, self.rot, duration, grip)

    def aim(self, pos, rot, label=""):
        self._solve(pos, rot, label)


def wait(seconds: float):
    for _ in range(max(1, round(seconds * CONTROL_HZ))):
        yield


def wait_still(ctx, name, speed=0.01, at_most=2.0):
    """Wait until an object has stopped moving (as seen in successive poses), at most ``at_most`` seconds."""
    for _ in range(round(at_most * CONTROL_HZ)):
        if np.linalg.norm(ctx.states()[name].lin_vel) < speed:
            return
        yield


def wait_for(flags, key):
    while not flags.get(key):
        yield


def both(*gens):
    gens = [iter(g) for g in gens]
    while gens:
        for g in list(gens):
            try:
                next(g)
            except StopIteration:
                gens.remove(g)
        if gens:
            yield


def script(task, ctx):
    """The demo as a generator: one yield per control step."""
    from yam_arena.demo import grip_rotation

    T = task
    variant = os.environ.get("DEMO_VARIANT", "")
    flags: dict = {}
    L, R = ctx.left, ctx.right

    def grip_across(axis_yaw: float, arm=None):
        """Straight-down grip whose fingers close across a horizontal bar turned ``axis_yaw`` from the y axis (the
        left arm reaches it with the hand turned the other way round)."""
        return grip_rotation((-math.pi / 2 if arm is L else math.pi / 2) - axis_yaw)

    def front_grip(axis_yaw: float):
        """Grip from the front, hand pointing forward and FRONT_PITCH down, fingers closing (nearly) up and down
        across a horizontal bar turned ``axis_yaw`` from the y axis."""
        base = np.column_stack([(0.0, 0.0, 1.0), (0.0, -1.0, 0.0), (1.0, 0.0, 0.0)])
        c, s = math.cos(FRONT_PITCH), math.sin(FRONT_PITCH)
        pitch = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        return _rz(-axis_yaw) @ pitch @ base

    def ends():
        s = ctx.states()["spindle"]
        return T.spindle_ends(s)

    def left_arm():
        # 1. grip the empty tube gently (a cardboard tube is not squeezed flat) and pull it, with the spindle, toward
        #    the robot: the spindle's domed ends ride out over the sockets' chamfers
        tube = ctx.states()["old_tube"].pos.copy()
        rot = grip_across(0.0, L)
        yield from L.move(tube + [0, 0.010, 0.07], rot, 2.5, grip=0.85, label="L tube above")
        yield from wait(0.5)
        yield from L.move(tube + [0, 0.010, 0.0], rot, 3.0, label="L tube")
        yield from wait(0.5)
        yield from L.set_grip(TUBE_GRIP, 0.8)
        yield from wait(0.3)
        yield from L.move(tube + [-0.045, 0.010, 0.0], rot, 2.5, label="L pull")
        yield from L.move(tube + [-0.060, 0.010, 0.040], rot, 1.5, label="L lift out")
        # 2. lay the tube, with the spindle in it, in the tray
        put = np.array([T.TRAY_X, T.TRAY_Y, T.TOP + T.TRAY_FLOOR + T.TUBE_R_OUT + 0.004])
        yield from L.move(put + [0, 0, 0.06], rot, 2.5, label="L over tray")
        yield from L.move(put, rot, 1.5, label="L into tray")
        yield from L.set_grip(0.85, 0.6)
        yield from L.move(put + [0, 0, 0.08], rot, 1.0, label="L up from tray")
        yield from L.move(np.array([0.18, 0.25, T.TOP + 0.18]), grip_rotation(-math.pi / 2), 2.0, label="L park")
        flags["in_tray"] = True

    def right_arm():
        yield from wait_for(flags, "in_tray")
        yield from R.move(np.array([0.22, -0.12, T.TOP + 0.20]), grip_across(0.0), 2.0, label="R via")
        # 3. grip the spindle's right end where it sticks out of the tube (look again once above it)
        for dz, t in ((0.06, 2.0), (0.03, 1.0)):
            yield from wait_still(ctx, "spindle")
            left, right, axis = ends()
            tube = ctx.states()["old_tube"]
            free = float((tube.pos - right) @ axis) - T.TUBE_LEN / 2          # how far the end sticks out
            grip_at = min(GRIP_END, free - PAD_HALF - 0.003)
            yaw = math.atan2(axis[0], axis[1])
            rr = grip_across(yaw)
            hold = right + axis * grip_at
            yield from R.move(hold + [0, 0, dz], rr, t, grip=1.0, label="R above right end")
        yield from R.move(hold, rr, 1.0, label="R right end")
        yield from R.set_grip(0.0, 0.8)
        yield from wait(0.3)
        # 4. lift it to the middle of the tube's bore and draw it out sideways; the tube stays in the tray
        lift = (T.TUBE_R_IN - T.OUT_R) + 0.001
        yield from R.move(R.pos + [0, 0, lift], rr, 1.0, label="R centre in bore")
        yield from R.move(R.pos - axis * (T.L0 + 0.02), rr, 3.0, label="R draw out")
        # lay it on the table beside the tray and take it again 12 mm in from its right tip
        yield from R.move(R.pos + [0, 0, 0.03], rr, 1.0, label="R up")
        spot = np.array([REGRIP_X, REGRIP_Y, T.TOP + T.OUT_R + 0.004])
        yield from R.move(spot + [0, 0, 0.03], grip_across(0.0), 2.0, label="R over table")
        yield from R.move(spot, grip_across(0.0), 1.0, label="R lay down")
        yield from R.set_grip(1.0, 0.6)
        yield from R.move(R.pos + [0, 0, 0.04], grip_across(0.0), 0.8, label="R up from spindle")
        for dz, t in ((0.04, 0.8), (0.02, 0.6)):
            yield from wait_still(ctx, "spindle")
            left, right, axis = ends()
            yaw = math.atan2(axis[0], axis[1])
            rr = grip_across(yaw)
            hold = right + axis * GRIP_END
            hold[2] = max(hold[2], T.TOP + 0.0135)                 # fingertips just clear of the table
            yield from R.move(hold + [0, 0, dz], rr, t, label="R above regrip")
        yield from R.move(hold, rr, 0.8, label="R regrip")
        yield from R.set_grip(0.0, 0.8)
        yield from wait(0.3)
        yield from R.move(R.pos + [0, 0, 0.05], grip_across(0.0), 1.2, label="R lift spindle")
        # where the hand holds the spindle: measured TCP to the right tip, along the spindle
        yield from wait(0.5)
        left, right, axis = ends()
        grip_at = float((R.actual - right) @ axis)
        over = PAD_HALF - grip_at                    # how far the fingers reach past the right tip
        print(f"grip: TCP {grip_at * 1000:.1f} mm in from the right tip, fingers {over * 1000:.1f} mm past it",
              flush=True)
        # 5. line the spindle up with the roll's core (coming from the side, clear of the roll) and push it through
        #    until the roll nearly touches the fingers
        rr = grip_across(0.0)
        reach = T.L0 - grip_at                                  # TCP to the left tip, relaxed
        roll = ctx.states()["roll"].pos.copy()
        tcp_y0 = roll[1] - T.ROLL_W / 2 - 0.020 - reach
        yield from R.move(np.array([R.pos[0], tcp_y0, roll[2] + 0.03]), rr, 2.0, label="R aside of roll")
        yield from R.move(np.array([roll[0], tcp_y0, roll[2] + 0.03]), rr, 1.5, label="R before roll")
        yield from wait_still(ctx, "roll")
        roll = ctx.states()["roll"].pos.copy()
        tcp_y0 = roll[1] - T.ROLL_W / 2 - 0.020 - reach
        yield from R.move(np.array([roll[0], tcp_y0, roll[2]]), rr, 1.2, label="R core height")
        tcp_y1 = roll[1] - T.ROLL_W / 2 - PAD_HALF - 0.007     # fingers stop 7 mm short of the roll's right face
        yield from R.move(np.array([roll[0], tcp_y1, roll[2]]), rr, 2.5, label="R through roll")
        yield from wait(0.5)
        # 6. let go (the spindle rests in the roll's core) and take the right end again from the front, fingers
        #    closing up and down: held that way, the roll's weight cannot twist it in the fingers
        yield from R.set_grip(1.0, 0.6)
        yield from R.move(R.pos + [0, 0, 0.05], rr, 1.0, label="R up from roll")
        yield from wait_still(ctx, "spindle")
        left, right, axis = ends()
        roll = ctx.states()["roll"].pos.copy()
        face = float((roll - right) @ axis) - T.ROLL_W / 2            # right tip to the roll's right face
        grip_at = min(GRIP_FIT, face - PAD_HALF - 0.003)
        yaw = math.atan2(axis[0], axis[1])
        fr = front_grip(yaw)
        hold = right + axis * grip_at
        back = fr[:, 2] * 0.07                                         # along the approach direction
        yield from R.move(hold - back + [0, 0, 0.02], fr, 2.0, label="R in front of right end")
        yield from R.move(hold - back, fr, 0.8, label="R level with right end")
        yield from R.move(hold, fr, 1.2, label="R onto right end")
        yield from R.set_grip(0.0, 0.8)
        yield from wait(0.3)
        print(f"front grip: TCP {grip_at * 1000:.1f} mm in from the right tip", flush=True)
        reach = T.L0 - grip_at
        # from here on the hand corrects, from the spindle's seen pose, the droop the roll's weight gives it: it turns
        # about the horizontal line across the spindle to keep it level, and moves up or down to keep its left tip on
        # the planned height
        servo_state = {"turn": 0.0, "dz": 0.0}

        def servo_aim(tcp, rot, label, update=True):
            left, right, axis = ends()
            tip_plan = tcp - rot[:, 1] * reach
            if update:
                pitch = math.asin(max(-1.0, min(1.0, float(axis[2]))))       # positive: left end higher
                servo_state["turn"] = max(-0.4, min(0.4, servo_state["turn"] - 0.25 * pitch))
                servo_state["dz"] = max(-0.04, min(0.04, servo_state["dz"] + 0.25 * (tip_plan[2] - left[2])))
            h = np.array([axis[0], axis[1], 0.0])
            h /= max(np.linalg.norm(h), 1e-9)
            nvec = np.array([h[1], -h[0], 0.0])                                # horizontal, across the spindle
            k = np.array([[0, -nvec[2], nvec[1]], [nvec[2], 0, -nvec[0]], [-nvec[1], nvec[0], 0]])
            a = servo_state["turn"]
            turn = np.eye(3) + math.sin(a) * k + (1 - math.cos(a)) * k @ k
            R.aim(np.asarray(tcp) + [0, 0, servo_state["dz"]], turn @ rot, label)
            R.pos, R.rot = np.asarray(tcp, float), rot                           # the plan, without the correction

        def servo_move(tcp, rot, duration, label, update=True):
            from yam_arena import ik
            p0, r0 = R.pos.copy(), R.rot.copy()
            n = max(1, round(duration * CONTROL_HZ))
            for k in range(1, n + 1):
                s = ik.smoothstep(k / n)
                servo_aim(p0 + s * (np.asarray(tcp) - p0), ik.slerp_matrix(r0, rot, s), label, update)
                yield

        yield from servo_move(R.pos + [0, 0, 0.05], fr, 1.5, "R lift roll")
        if variant == "lift_only":
            return
        # 7. carry it in front of the holder, the spindle turned YAW (right end toward the robot); bring the left
        #    end in between the posts from the front, then along the spindle into the left socket
        sock_l = np.array([T.SX, T.HY + T.GAP / 2 + T.DEPTH, T.SZ])
        u = np.array([math.sin(YAW), math.cos(YAW), 0.0])       # right end -> left end
        ry = front_grip(YAW)
        p2 = sock_l - u * 0.008                                  # left tip 8 mm short of the socket bottom
        p1 = np.array([0.25, p2[1], T.SZ])                       # ... and first well in front of the holder
        yield from servo_move(R.pos + [-0.09, 0, 0], fr, 1.5, "R back from the posts")
        yield from servo_move(p1 - u * reach + [-0.03, 0, 0.02], ry, 2.5, "R in front of holder")
        yield from servo_move(p1 - u * reach + [0, 0, 0.02], ry, 1.0, "R closer")
        yield from servo_move(p1 - u * reach, ry, 1.0, "R at socket height")
        yield from servo_move(p1 - u * reach, ry, 1.0, "R settle")
        yield from servo_move(p2 - u * reach, ry, 2.0, "R between the posts")
        yield from servo_move(sock_l + u * 0.001 - u * reach, ry, 1.0, "R left end in")
        yield from wait(0.3)
        if variant == "early_release":
            yield from R.set_grip(1.0, 0.6)
            yield from R.move(R.pos - ry[:, 2] * 0.08, ry, 1.5, label="R away")
            yield from R.move(R.pos + [0, 0, 0.08], ry, 1.0, label="R up")
            return
        # squeeze it so short that its right tip ends 2 mm, and the fingers at least 2.5 mm, inside the right post
        tcp_in = max(0.002 + grip_at, 0.0025 + PAD_HALF)        # TCP inside the right post's face
        fitted_reach = T.GAP + T.DEPTH - tcp_in                 # TCP to the left socket bottom
        squeeze = 0.0 if variant == "no_squeeze" else reach - fitted_reach
        print(f"squeeze {squeeze * 1000:.1f} mm", flush=True)
        yield from servo_move(sock_l + u * 0.001 - u * (reach - squeeze), ry, 1.5, "R squeeze", update=False)
        # swing the right end round the left socket until the spindle lies along y
        n = round(2.5 * CONTROL_HZ)
        pivot = sock_l + np.array([0.0, 0.001, 0.0])
        for k in range(1, n + 1):
            s = 0.5 - 0.5 * math.cos(math.pi * k / n)
            a = YAW * (1 - s)
            ua = np.array([math.sin(a), math.cos(a), 0.0])
            servo_aim(pivot - ua * (reach - squeeze), front_grip(a), "R swing", update=False)
            yield
        yield from wait(0.5)
        # 8. let go: the spindle springs out into the right socket; back the hand out the way it came
        yield from R.set_grip(1.0, 0.8)
        yield from wait(0.5)
        yield from R.move(R.pos - R.rot[:, 2] * 0.09, R.rot, 1.5, label="R back out")
        yield from R.move(R.pos + [0, 0, 0.06], R.rot, 1.0, label="R up")
        yield from R.move(np.array([0.18, -0.20, T.TOP + 0.16]), grip_rotation(math.pi / 2), 2.0, label="R park")

    if variant == "nothing":
        for _ in range(round(60 * CONTROL_HZ)):
            yield
    else:
        yield from both(left_arm(), right_arm())


class Ctx:
    def __init__(self, emb):
        from yam_arena import station

        self.emb = emb
        self.left = Arm("left", station.BASES["left"])
        self.right = Arm("right", station.BASES["right"])
        self.steps = 0

    def states(self):
        from yam_arena.scene import object_states

        return object_states(self.emb._env, self.emb.task.names)

    def measure(self) -> None:
        """The robot's own joint angles (what a policy observes) -> each arm's measured TCP."""
        for arm, name in ((self.left, "left_robot"), (self.right, "right_robot")):
            r = self.emb._env.unwrapped.scene[name]
            names = list(r.joint_names)
            jp = r.data.joint_pos
            jp = getattr(jp, "torch", jp)[0].tolist()
            arm.measured([jp[names.index(f"joint{i}")] for i in range(1, 7)])

    def action(self):
        from inspect_robots.types import Action
        from inspect_robots_yam.packing import pack

        return Action(data=pack(self.left.cmd, self.right.cmd))


def _write_mp4(frames, path: Path) -> None:
    import imageio

    imageio.mimsave(path, [np.asarray(f) for f in frames], fps=CONTROL_HZ, codec="libx264",
                    output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])


def _report(task, s) -> str:
    left, right, _ = task.spindle_ends(s["spindle"])
    return (f"spL {np.round(left, 4).tolist()} spR {np.round(right, 4).tolist()} "
            f"q {s['spindle'].joints.get('inner', 0):.4f} tube {np.round(s['old_tube'].pos, 3).tolist()} "
            f"roll {np.round(s['roll'].pos, 3).tolist()}")


def _fingers(emb) -> str:
    """Debug: each gripper's finger opening (mm between pads)."""
    out = []
    for arm in ("left_robot", "right_robot"):
        r = emb._env.unwrapped.scene[arm]
        names = list(r.joint_names)
        jp = r.data.joint_pos
        jp = getattr(jp, "torch", jp)[0].tolist()
        out.append(f"{arm[0]}gap {1000 * sum(abs(jp[names.index(j)]) for j in ('joint7', 'joint8')):.0f}")
    return " ".join(out)


def run(task_path: str, out_dir: str, max_steps: int = 3000, record: bool = True, cam_width: int = 320,
        cam_height: int = 176, hold_s: float = 3.0) -> dict:
    from types import SimpleNamespace

    from inspect_robots import Scene
    from yam_arena.inspect_embodiment import yam_arena_embodiment
    from yam_arena.scoring import task_score

    task_path = str(Path(task_path).resolve())
    sys.path.insert(0, str(Path(task_path).parent))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    emb = yam_arena_embodiment(task=task_path, object_jitter=0, control_hz=CONTROL_HZ, cam_width=cam_width,
                               cam_height=cam_height)
    obs = emb.reset(Scene(id="demo", instruction="demo"))
    task = emb.task.module
    ctx = Ctx(emb)
    print("start", _report(task, ctx.states()), flush=True)
    gen = script(task, ctx)
    frames: dict[str, list] = {}

    def snap(o):
        if record:
            for cam, img in o.images.items():
                frames.setdefault(cam, []).append(np.asarray(img))

    snap(obs)
    steps, ended, res = 0, None, None
    debug = bool(os.environ.get("DEMO_DEBUG"))
    with (out / "actions.jsonl").open("w") as log:
        while steps < max_steps:
            try:
                next(gen)
            except StopIteration:
                ended = steps if ended is None else ended
            act = ctx.action()
            res = emb.step(act)
            steps = ctx.steps = steps + 1
            ctx.measure()
            log.write(json.dumps({"step": steps, "action": np.asarray(act.data, float).round(6).tolist()}) + "\n")
            snap(res.observation)
            if steps % 50 == 0 or (debug and steps % 5 == 0):
                print(f"step {steps} score {res.info.get('score')} {_report(task, ctx.states())} {_fingers(emb)} "
                      f"tcpL {np.round(ctx.left.pos, 3).tolist()} actL {np.round(ctx.left.actual, 3).tolist()} "
                      f"tcpR {np.round(ctx.right.pos, 3).tolist()} actR {np.round(ctx.right.actual, 3).tolist()}"
                      + (f" Rtilt {math.degrees(math.asin(float(ctx.right.actual_rot[2, 1]))):.1f}"
                         f" Rcmdtilt {math.degrees(math.asin(float(ctx.right.rot[2, 1]))):.1f}" if debug else ""),
                      flush=True)
            if res.terminated:
                break
            if ended is not None and steps - ended >= hold_s * CONTROL_HZ:
                break
    scored = task_score()(SimpleNamespace(steps=[SimpleNamespace(result=res)]), None)
    print("end", _report(task, ctx.states()), flush=True)
    print(f"task_score: {scored.value:g} ({scored.explanation}) after {steps} steps", flush=True)
    result = {"task_score": scored.value, "explanation": scored.explanation, "steps": steps,
              "variant": os.environ.get("DEMO_VARIANT", ""),
              "worst_ik_mm": {"left": [round(ctx.left.worst_mm, 2), ctx.left.worst_at],
                              "right": [round(ctx.right.worst_mm, 2), ctx.right.worst_at]}}
    if record and frames:
        import imageio

        cams = sorted(frames, key=lambda c: ("left" not in c, "top" not in c))   # left, top, right
        _write_mp4([np.hstack([frames[c][i] for c in cams]) for i in range(len(frames[cams[0]]))], out / "demo.mp4")
        imageio.imwrite(out / "setup.png", np.hstack([frames[c][0] for c in cams]))
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("out")
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--cam-width", type=int, default=320)
    ap.add_argument("--cam-height", type=int, default=176)
    a = ap.parse_args()
    run(a.task, a.out, a.max_steps, record=not a.no_record, cam_width=a.cam_width, cam_height=a.cam_height)
