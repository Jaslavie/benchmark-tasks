"""A chalk line, hooked to a nail at one mark on a board. One arm pulls the reel to the other mark and holds the line taut; the other pinches the line at its middle, lifts it straight up and lets go. The chalk print appears from the release, as a real snap leaves it.

Debug:  python demokit.py task.py OUT --fast --mode full
Final:  python demokit.py task.py OUT
Model:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 --scorer task_score --max-steps 1200 --instruction "<INSTRUCTION below>"
"""

from __future__ import annotations

import dataclasses
import math
import os
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import demokit as d  # noqa: E402
import mechkit as m  # noqa: E402

SCENE = "station"
INSTRUCTION = "Snap a chalk line between the two marks. The hook is on the far nail. Pull the reel to the near mark and hold the line taut across both marks, then lift the line straight up at its middle and let it go. Put the reel on its rest."
SUCCESS_HOLD_S = 1.0
SCORE_OVER = "end"
HOLD_STEPS = 20  # SUCCESS_HOLD_S at 20 Hz: the harness ends the trial with 1.0 once success() has held this many control steps
TOP = d.TABLE_TOP

LOOKS = {
    "board": m.Look(m.rgb(0.13, 0.16, 0.15), roughness=0.9),
    "mark": m.Look(m.rgb(0.95, 0.85, 0.20), roughness=0.6),
    "nail": m.Look(m.rgb(0.55, 0.57, 0.60), metallic=0.9, roughness=0.3),
    "line": m.Look(m.rgb(1.0, 0.40, 0.0), emissive=(0.6, 0.2, 0.0), roughness=0.6),
    "print": m.Look(m.rgb(0.93, 0.93, 0.90), roughness=1.0),
    "reel": m.Look(m.rgb(0.10, 0.30, 0.75), roughness=0.4),
    "crank": m.Look(m.rgb(0.75, 0.76, 0.78), metallic=0.7, roughness=0.3),
    "rest": m.Look(m.rgb(0.35, 0.37, 0.40), roughness=0.6),
}

BOARD_Z = 0.012
LINE_X = 0.33
NAIL_Y, NEAR_Y = 0.30, -0.30
HOOK_Z = BOARD_Z + 0.010  # the line stands this low at both ends, so that a snap reaches the board along its length
REEL_R, REEL_H = 0.036, 0.040
NOSE_Y = REEL_R + 0.012  # the line leaves the reel here, towards the nail
NOSE_Z = 0.008
LINE_L = 0.595  # taut at this length: the line lock is set for the marks
REST_AT = (0.20, -0.30)
REST_H = 0.03
PINCH_NEAR = 0.02
PINCH_SHUT = 0.012  # half the fingers' opening, closed on the line
RISE_MIN = 0.04
BOW_MAX = 0.01
GOOD_MISS, FAIR_MISS = 0.005, 0.015
CLEAR = 0.03
_NAIL = np.array([LINE_X, NAIL_Y, TOP + HOOK_Z])
_NEAR = np.array([LINE_X, NEAR_Y, TOP + BOARD_Z])


def _chalk_dust() -> m.Mesh:
    """A chalk line along a unit length, stretched when placed: a soft thin core with flecks of dust scattered about it, and small gaps."""
    rng = np.random.default_rng(7)
    pieces = []
    for x in np.arange(0.0, 1.0, 0.012):
        if rng.random() > 0.12:
            pieces.append(m.box((0.012 * rng.uniform(0.6, 1.0), rng.uniform(0.005, 0.009), 0.0006), (x + 0.006, rng.normal(0.0, 0.0008), 0.0)))
    for _ in range(170):
        pieces.append(m.box((rng.uniform(0.002, 0.006), rng.uniform(0.0012, 0.003), 0.0006), (rng.uniform(0.0, 1.0), rng.normal(0.0, 0.0065), 0.0)))
    return m.join(*pieces)


def _frame(a, b) -> np.ndarray:
    """The 4 x 4 matrix, in USD's row order, that stretches a unit box along x from a to b."""
    x = np.asarray(b, float) - np.asarray(a, float)
    n = float(np.linalg.norm(x))
    u = x / n if n > 1e-6 else np.array([0.0, 1.0, 0.0])
    y = np.cross([0.0, 0.0, 1.0], u)
    y = y / max(float(np.linalg.norm(y)), 1e-6)
    return np.array([[*(u * max(n, 1e-6)), 0.0], [*y, 0.0], [*np.cross(u, y), 0.0], [*a, 1.0]])


def _board() -> m.Part:
    w, b, t = 0.04, 0.006, 0.0006
    parts = [("board", m.box((0.16, 0.76, BOARD_Z), (LINE_X, 0, BOARD_Z / 2)), "board")]
    for name, y in (("far", NAIL_Y), ("near", NEAR_Y)):
        parts.append((f"mark_{name}_a", m.box((w, b, t), (LINE_X, y, BOARD_Z + t / 2)), "mark"))
        parts.append((f"mark_{name}_b", m.box((b, w, t), (LINE_X, y, BOARD_Z + t / 2)), "mark"))
    parts += [
        ("nail", m.cylinder(0.003, HOOK_Z - BOARD_Z + 0.008, (LINE_X, NAIL_Y, (HOOK_Z + BOARD_Z + 0.008) / 2)), "nail"),
        ("nail_head", m.cylinder(0.007, 0.002, (LINE_X, NAIL_Y, HOOK_Z + 0.009)), "nail"),
        ("hook", m.box((0.012, 0.016, 0.003), (LINE_X, NAIL_Y - 0.006, HOOK_Z)), "crank"),
        ("rest", m.box((0.09, 0.09, REST_H), (*REST_AT, REST_H / 2)), "rest"),
        ("seg_a", m.box((1.0, 0.007, 0.007), (0.5, 0, 0)), "line"),
        ("seg_b", m.box((1.0, 0.007, 0.007), (0.5, 0, 0)), "line"),
        ("print", _chalk_dust(), "print", True),
    ]
    cols = [m.col_box("board_c", (0.16, 0.76, BOARD_Z), (LINE_X, 0, BOARD_Z / 2)), m.col_box("rest_c", (0.09, 0.09, REST_H), (*REST_AT, REST_H / 2))]
    part = m.fixed("line_board", (0.0, 0.0, TOP), parts, cols, LOOKS)
    # The line stands in its start pose from the first frame: slack, from the nail, down to the board, to the reel on its rest.
    nose = np.array([REST_AT[0], REST_AT[1] + NOSE_Y, REST_H + 0.001 + NOSE_Z])
    nail = _NAIL - [0.0, 0.0, TOP]
    mid = np.array([*(0.5 * (nail + nose))[:2], BOARD_Z + 0.004])
    start = {"seg_a": _frame(nail, mid), "seg_b": _frame(mid, nose), "print": _frame(nail, nail)}
    build = part.build

    def posed() -> str:
        text = build()
        for name, f in start.items():
            rows = ", ".join("(" + ", ".join(f"{v:.6f}" for v in row) + ")" for row in f)
            text = re.sub(rf'(def Mesh "{name}" \(\s*prepend apiSchemas = \["MaterialBindingAPI"\]\s*\)\s*\{{)', rf'\1\n            matrix4d xformOp:transform = ({rows})\n            uniform token[] xformOpOrder = ["xformOp:transform"]', text)
        return text

    return dataclasses.replace(part, build=posed)


def _reel() -> m.Part:
    parts = [
        ("case", m.cylinder(REEL_R, REEL_H, (0, 0, REEL_H / 2), segments=32), "reel"),
        ("nose", m.box((0.018, 0.024, 0.016), (0, REEL_R + 0.002, NOSE_Z)), "reel"),
        ("crank", m.box((0.010, 0.030, 0.006), (0, -0.012, REEL_H + 0.003)), "crank"),
        ("knob", m.cylinder(0.006, 0.012, (0, -0.024, REEL_H + 0.012)), "crank"),
    ]
    cols = [m.col_cylinder("case_c", REEL_R, REEL_H, (0, 0, REEL_H / 2))]
    return m.free("reel", (*REST_AT, TOP + REST_H + 0.001), parts, cols, LOOKS, mass=0.2, friction=(1.0, 0.9))


OBJECTS = [_board(), _reel()]


def _place(path: str, a, b) -> None:
    """Stretch a unit box along x so that it runs from a to b, world points, in the part's frame."""
    import isaaclab.sim as sim_utils
    from pxr import Gf, UsdGeom

    prim = sim_utils.get_current_stage().GetPrimAtPath(path)
    xf = UsdGeom.Xformable(prim)
    ops = xf.GetOrderedXformOps()
    op = ops[0] if ops else xf.AddTransformOp()
    top = np.array([0.0, 0.0, TOP])
    op.Set(Gf.Matrix4d(*[float(v) for v in _frame(np.asarray(a, float) - top, np.asarray(b, float) - top).reshape(-1)]))


def judge(snaps: int, miss: float, bowed: bool, parked: bool, clear: bool, still: bool, held_steps: int = HOLD_STEPS) -> tuple[bool, float]:
    """The score now: (full, score). `miss`: how far the print passes from the near mark, in metres.
    full: every 1.0 condition holds at this step. held_steps: for how many control steps they have held, this one included.
    The score is 1.0 once they have held for SUCCESS_HOLD_S, and the lower score of the text until then. Run on hand-built states by check_score.py."""
    if snaps != 1 or bowed or miss > FAIR_MISS:
        return False, 0.0
    if miss > GOOD_MISS:
        return False, 0.5
    full = bool(parked and clear and still)
    return full, 1.0 if full and held_steps >= HOLD_STEPS else 0.75


def _on_reel(fingers: np.ndarray, reel_pos: np.ndarray) -> bool:
    """Whether these fingertips are closed on the reel's case: centred on it, at its height, and at about its width."""
    tcp = fingers.mean(axis=0)
    half_gap = 0.5 * float(np.linalg.norm(fingers[0] - fingers[6]))
    return bool(np.hypot(*(tcp - reel_pos)[:2]) <= 0.025 and -0.005 <= tcp[2] - reel_pos[2] <= REEL_H + 0.015 and REEL_R <= half_gap <= REEL_R + 0.012)


def release_outcome(taut: bool, rise: float, side: float) -> str:
    """What letting go of the pinched line does: "none", when lowered too near its path to snap; "snap"; or "bowed", when slack or lifted sideways, which leaves a bowed line that stays."""
    if rise < RISE_MIN:
        return "none"
    return "snap" if taut and side <= BOW_MAX else "bowed"


class Trial:
    """What the scorer holds across a trial: the line's state, and the snaps."""

    def reset(self, env=None) -> None:
        self.steps = 0
        self.held = False
        self.last_pinch = None
        self.snaps = 0
        self.miss = 1.0
        self.bowed = False
        self.taut = False
        self.full, self.score = False, 0.0
        self.full_steps = 0  # control steps the full state has lasted
        if env is not None:
            m.set_visible(f"{m.ENV_PATH}/line_board/visual/print", False)
            self.step(env)

    def step(self, env) -> None:
        self.steps += 1
        if os.environ.get("TASK_STEP_FILE"):  # the simulated time, for anything following the run from outside
            with open(os.environ["TASK_STEP_FILE"], "w") as f:
                f.write(f"{self.steps * d.DT:.2f}")
        reel = m.state(env, "reel")
        nose = reel.pos + reel.rotation() @ np.array([0.0, NOSE_Y, NOSE_Z])
        span = float(np.linalg.norm(_NAIL - nose))
        all_tips = m.fingertips(env)
        points = {"left": all_tips[:12], "right": all_tips[12:]}
        # Taut takes a gripper closed on the reel, at the length the line is locked to.
        on_reel = any(_on_reel(f, reel.pos) for f in points.values())
        self.taut = bool(span >= LINE_L and on_reel)
        straight = _NAIL + 0.5 * (nose - _NAIL)
        pinch = None
        for fingers in points.values():
            tcp = fingers.mean(axis=0)
            opening = 0.5 * float(np.linalg.norm(fingers[0] - fingers[6]))  # the two fingers' first points: half the gap, with their thickness
            if self.held:
                if opening <= PINCH_SHUT + 0.006:
                    pinch = tcp
            elif opening <= PINCH_SHUT and np.linalg.norm(tcp - straight) <= PINCH_NEAR and self.taut:
                pinch = tcp
                print(f"[chalk] line pinched, {np.round(tcp - straight, 3)} from its middle", flush=True)
            if pinch is not None:
                break
        if pinch is None and self.held:  # just released
            rise = float(self.last_pinch[2] - straight[2])
            along = (nose - _NAIL)[:2] / max(float(np.linalg.norm((nose - _NAIL)[:2])), 1e-6)
            off = (self.last_pinch - straight)[:2]
            side = abs(float(off[0] * along[1] - off[1] * along[0]))  # across the line only
            print(f"[chalk] released: {rise * 100:.1f} cm above the line, {side * 100:.1f} cm to the side, line {'taut' if self.taut else 'slack'}", flush=True)
            outcome = release_outcome(self.taut, rise, side)
            if outcome != "none":
                self.snaps += 1
                self.bowed = self.bowed or outcome == "bowed"
                if self.snaps == 1:
                    u = (nose - _NAIL) / span
                    rel = _NEAR - _NAIL
                    self.miss = float(np.hypot(*(rel - (rel @ u) * u)[:2]))
                    _place(f"{m.ENV_PATH}/line_board/visual/print", [_NAIL[0], _NAIL[1], TOP + BOARD_Z + 0.0005], [nose[0], nose[1], TOP + BOARD_Z + 0.0005])
                    m.set_visible(f"{m.ENV_PATH}/line_board/visual/print", True)
                print(f"[chalk] {outcome} {self.snaps}: the print misses the near mark by {self.miss * 1000:.1f} mm, bowed {self.bowed}", flush=True)
        self.held = pinch is not None
        if pinch is not None:
            self.last_pinch = pinch
            mid = pinch
        elif self.taut:
            mid = straight
        else:
            mid = np.array([straight[0], straight[1], TOP + BOARD_Z + 0.004])
        _place(f"{m.ENV_PATH}/line_board/visual/seg_a", _NAIL, mid)
        _place(f"{m.ENV_PATH}/line_board/visual/seg_b", mid, nose)
        rest = np.array([*REST_AT, TOP + REST_H])
        parked = bool(np.linalg.norm(reel.pos - rest) <= 0.03 and reel.tilt_deg() <= 10.0)
        near = np.linalg.norm(all_tips - (reel.pos + [0.0, 0.0, REEL_H / 2]), axis=1).min() - REEL_R
        self.full, self.score = judge(self.snaps, self.miss, self.bowed, parked, bool(near >= CLEAR), reel.resting(), self.full_steps + 1)
        self.full_steps = self.full_steps + 1 if self.full else 0
        if self.full_steps == HOLD_STEPS:
            print(f"[chalk] full state held {HOLD_STEPS * d.DT:.2f} s: score {self.score}", flush=True)


TRIAL = Trial()
TRIAL.reset()
m.on_step(TRIAL.step)
m.on_reset(TRIAL.reset)


def success(objects) -> bool:
    """One snap, within 5 mm of the near mark, not bowed; the reel on its rest and the fingers clear."""
    return TRIAL.full


def score(objects) -> float:
    """1.0 once the full state has held for SUCCESS_HOLD_S; otherwise the lower score of the scoring text for the state."""
    return TRIAL.score


# -- the demo: reads the reel's pose, as a camera could see it ----------------------------------------------------------------------------------------


def demo_script(ctx):
    """Modes: full. none: the arms wait. Other score lines are checked offline: see check_score.py."""
    mode = ctx.mode
    if mode not in ("full", "none"):
        raise ValueError(f"unknown mode {mode!r}")
    l, r = ctx.left, ctx.right
    if mode == "none":
        yield from l.wait(3.0)
        return
    grip_up = REEL_H / 2 + 0.005
    reel_at = ctx.pos("reel")
    grip_r = d.down(90.0)  # fingers across the case, along x
    over = reel_at + [0.0, 0.0, grip_up + 0.06]
    yield from r.move(over, grip=grip_r, finger=d.FINGER_OPEN, label="over the reel")
    yield from r.move(reel_at + [0.0, 0.0, grip_up], t=0.8, label="on the reel")
    yield from r.servo(ctx.pos("reel") + [0.0, 0.0, grip_up], timeout=0.5, label="on the reel")
    yield from r.close(2 * REEL_R, force=25.0, label="grip the reel")
    yield from r.move(r.at() + [0.0, 0.0, 0.05], t=0.7, label="lift the reel")

    def nose_of() -> np.ndarray:
        st = ctx.state("reel")
        return st.pos + st.rotation() @ np.array([0.0, NOSE_Y, NOSE_Z])

    # The nose over the near mark, the line level with the hook, the reel just off the board.
    goal = np.array([LINE_X, NEAR_Y + 0.003, TOP + HOOK_Z])
    held = r.at() - ctx.pos("reel")
    yield from r.move(goal - np.array([0.0, NOSE_Y, NOSE_Z]) + held, t=2.0, label="to the near mark")
    yield from r.servo(goal, read=nose_of, timeout=1.5, tol=0.002, label="the nose on the near mark")
    print(f"[chalk] reel at the mark: nose {np.round(nose_of() - goal, 4)}, span {np.linalg.norm(_NAIL - nose_of()):.3f}, taut {TRIAL.taut}", flush=True)
    # The left arm pinches the line at its middle, lifts it straight up and opens.
    mid = 0.5 * (_NAIL + nose_of())
    grip_l = d.down(-90.0)  # fingers across the line, along x
    yield from l.move(mid + [0.0, 0.0, 0.06], grip=grip_l, finger=d.FINGER_OPEN, label="over the middle")
    at = mid + [0.0, 0.0, 0.010]  # the fingertips just over the board
    yield from l.move(at, t=0.8, label="round the line")
    yield from l.servo(at, timeout=0.5, label="round the line")
    yield from l.close(0.006, force=10.0, label="pinch the line")
    yield from l.wait(0.2)
    yield from l.move(mid + [0.0, 0.0, 0.06], t=1.0, label="lift the line")
    yield from l.servo(mid + [0.0, 0.0, 0.06], timeout=0.5, label="lift the line")
    yield from l.open(t=0.15, label="let go")
    yield from l.wait(0.3)
    yield from l.move(l.at() + [0.0, 0.0, 0.05], t=0.5, label="up")
    yield from l.move((0.20, 0.30, TOP + 0.20), t=1.2, label="away")
    print(f"[chalk] snaps {TRIAL.snaps}, miss {TRIAL.miss * 1000:.1f} mm", flush=True)
    # The reel back on its rest.
    rest_top = np.array([*REST_AT, TOP + REST_H + 0.002])
    yield from r.move(rest_top + held + [0.0, 0.0, 0.05], t=1.5, label="over the rest")
    yield from r.move(rest_top + held, t=0.8, label="on the rest")
    yield from r.open(label="let go of the reel")
    yield from r.move(r.at() + [0.0, 0.0, 0.08], t=0.8, label="up")
    print(f"[chalk] reel at {np.round(ctx.pos('reel') - rest_top, 3)}", flush=True)
    yield from r.wait(1.5)
