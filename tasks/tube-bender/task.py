"""On the station table: a lever tube bender standing free on the bench, with a copper tube in it and a protractor card on its base plate. The form handle points left and the lever points right, in one line with the tube. Pulling the lever towards the arms bends the tube round the form wheel. Copper springs back when the lever is relaxed, and it cannot be unbent."""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import demokit as d  # noqa: E402
import mechkit as m  # noqa: E402

SCENE = "station"
INSTRUCTION = "Bend the tube in the bender to the 90 degree mark on the card, judged with the lever fully relaxed, then take both grippers away."
SUCCESS_HOLD_S = 1.0
SCORE_OVER = "end"
TOP = d.TABLE_TOP

PIVOT = (0.33, 0.0)
LEVER_L, FORM_L = 0.17, 0.17
BAR_Z, TUBE_Z = 0.05, 0.03
TUBE_OFF = 0.035  # the tube lies against the form wheel's edge, beside the handles, where it can be seen
GRIP_AT = 0.15  # along each handle, where a gripper holds it
SPRING, DAMPER = 0.6, 0.3  # N m/rad and N m s/rad of the lever: bending load, and its return
SPRINGBACK = {"as drawn": 12.0, "annealed": 4.0}
TARGET, TOL, OVER, PARTIAL_FROM = 90.0, 3.0, 96.0, 70.0
RELAXED = 5.0  # the lever within this of its rest
CLEAR = 0.03

LOOKS = {"plate": m.Look(m.rgb(0.28, 0.3, 0.32), metallic=0.6, roughness=0.45), "steel": m.Look(m.rgb(0.62, 0.64, 0.66), metallic=0.85, roughness=0.35), "rubber": m.Look(m.rgb(0.75, 0.1, 0.07), roughness=0.75), "copper": m.Look(m.rgb(0.8, 0.45, 0.25), metallic=0.8, roughness=0.3),
         "card": m.Look(m.rgb(0.95, 0.95, 0.93)), "zero": m.Look(m.rgb(0.1, 0.55, 0.2)), "band": m.Look(m.rgb(0.85, 0.08, 0.06)), "ink": m.Look(m.rgb(0.05, 0.05, 0.06))}


def _wedge(a0: float, a1: float, r0: float, r1: float, z0: float, z1: float, n: int = 8) -> m.Mesh:
    """A flat sector of a ring, between these angles in degrees from +x."""
    outer = [(r1 * math.cos(math.radians(a)), r1 * math.sin(math.radians(a))) for a in np.linspace(a0, a1, n)]
    inner = [(r0 * math.cos(math.radians(a)), r0 * math.sin(math.radians(a))) for a in np.linspace(a1, a0, n)]
    return m.prism(np.array(outer + inner), z0, z1)


def _dir(bend: float) -> np.ndarray:
    """The lever's direction at this bend: -y at 0, turning towards -x."""
    a = math.radians(-90.0 - bend)
    return np.array([math.cos(a), math.sin(a), 0.0])


def _bender() -> m.Part:
    plate = (0.20, 0.34, 0.012, 0.05)  # x, y, z sizes, and its centre's y
    frame_parts = [("plate", m.box(plate[:3], (0, plate[3], plate[2] / 2)), "plate"),
                   ("card", m.cylinder(0.125, 0.001, (0, 0, plate[2] + 0.0005), segments=48), "card"),
                   ("zero", m.box((0.006, 0.03, 0.0006), (TUBE_OFF, -0.105, plate[2] + 0.0012)), "zero"),
                   ("band", _wedge(180.0 - TOL, 180.0 + TOL, 0.04, 0.09, plate[2] + 0.0009, plate[2] + 0.0015).moved((0, -TUBE_OFF, 0)), "band"),
                   ("tick70", _wedge(180.0 - 20.5, 180.0 - 19.5, 0.07, 0.11, plate[2] + 0.0009, plate[2] + 0.0015).moved((0, -TUBE_OFF, 0)), "ink"),
                   ("tick80", _wedge(180.0 - 10.5, 180.0 - 9.5, 0.07, 0.11, plate[2] + 0.0009, plate[2] + 0.0015).moved((0, -TUBE_OFF, 0)), "ink"),
                   ("tick96", _wedge(180.0 + 5.5, 180.0 + 6.5, 0.07, 0.11, plate[2] + 0.0009, plate[2] + 0.0015).moved((0, -TUBE_OFF, 0)), "ink"),
                   ("form", m.cylinder(0.035, TUBE_Z + 0.012 - plate[2], (0, 0, (TUBE_Z + 0.012 + plate[2]) / 2), segments=40), "steel"),
                   ("form_bar", m.box((0.02, FORM_L - 0.03, 0.014), (0, 0.03 + (FORM_L - 0.03) / 2, BAR_Z)), "steel"),
                   ("form_post", m.box((0.03, 0.03, BAR_Z - plate[2]), (0, 0.05, (BAR_Z + plate[2]) / 2)), "steel"),
                   ("form_grip", m.box((0.024, 0.07, 0.018), (0, GRIP_AT, BAR_Z)), "rubber"),
                   ("tube_fixed", m.tube([(TUBE_OFF, 0.0, TUBE_Z), (TUBE_OFF, 0.16, TUBE_Z)], 0.005), "copper")]
    frame_colliders = [m.col_box("plate_c", plate[:3], (0, plate[3], plate[2] / 2)), m.col_cylinder("form_c", 0.035, TUBE_Z + 0.012 - plate[2], (0, 0, (TUBE_Z + 0.012 + plate[2]) / 2)),
                       m.col_box("form_bar_c", (0.02, FORM_L - 0.03, 0.014), (0, 0.03 + (FORM_L - 0.03) / 2, BAR_Z)), m.col_box("form_post_c", (0.03, 0.03, BAR_Z - plate[2]), (0, 0.05, (BAR_Z + plate[2]) / 2))]
    bar = m.box((0.02, LEVER_L - 0.03, 0.014), (0, -(0.03 + (LEVER_L - 0.03) / 2), BAR_Z))
    head = m.cylinder(0.03, 0.012, (0, 0, BAR_Z), segments=32)
    grip = m.box((0.024, 0.07, 0.018), (0, -GRIP_AT, BAR_Z))
    lever = m.hinge("lever", "Z", (0, 0, BAR_Z), -125.0, 0.0, [("bar", bar, "steel"), ("head", head, "steel"), ("grip", grip, "rubber")], [m.col_box("bar_c", (0.02, LEVER_L - 0.03, 0.014), (0, -(0.03 + (LEVER_L - 0.03) / 2), BAR_Z)), m.col_cylinder("head_c", 0.03, 0.012, (0, 0, BAR_Z))],
                    mass=0.15, spring=SPRING, damping=DAMPER, friction=0.0, effort=100.0, armature=0.002)
    tube = m.hinge("tube", "Z", (0, 0, TUBE_Z), -125.0, 0.0, [("tube", m.tube([(TUBE_OFF, 0.0, TUBE_Z), (TUBE_OFF, -0.13, TUBE_Z)], 0.005), "copper")], [], mass=0.02, spring=20.0, damping=1.0, friction=0.0, effort=100.0, armature=0.002)
    return m.mechanism("bender", (*PIVOT, TOP + 0.001), [lever, tube], {k: LOOKS[k] for k in ("plate", "card", "zero", "band", "ink", "steel", "rubber", "copper")}, frame_parts=frame_parts, frame_colliders=frame_colliders, free_mass=0.4, friction=(0.4, 0.3))


OBJECTS = [_bender()]


def springback_for(seed: int | None) -> float:
    return SPRINGBACK["annealed"] if seed is not None and seed % 2 == 1 else SPRINGBACK["as drawn"]


def segment_gap(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """The least distance from any of these points to the segment a-b."""
    ab = b - a
    s = np.clip((points - a) @ ab / (ab @ ab), 0.0, 1.0)
    return float(np.linalg.norm(points - (a + s[:, None] * ab), axis=1).min())


class Trial:
    """The trial-long record: how far the tube has been set."""

    def __init__(self) -> None:
        self.reset()

    def reset(self, env=None) -> None:
        self.live = env is not None
        self.steps = 0
        self.back = springback_for(m.run_seed())
        self.set_angle = 0.0
        self.too_far = False
        self.off_bench = False
        self.moved = None
        self.recent = []
        self.tube = 0.0
        self.settled_s = 0.0  # how long the lever has stayed relaxed and still, with both grippers clear
        self.full, self.score = False, 0.0

    def step(self, env) -> None:
        self.steps += 1
        st = m.state(env, "bender")
        bend = -st.deg("lever")
        local = st.to_local(m.fingertips(env))
        dt = env.unwrapped.step_dt
        self.update(bend, st.pos, st.tilt_deg(), local, self.steps * dt, dt)
        m.set_target(env, "bender", "tube", -math.radians(self.tube))

    def update(self, bend: float, pos: np.ndarray, tilt: float, local_tips: np.ndarray, t: float, dt: float) -> None:
        """The scorer, as a plain function of the lever's angle, the bender's pose and the fingertips in the bender's frame, called once a control step of dt seconds. `check_scorer.py` calls it on hand-built states.

        Every score line needs the lever relaxed and still, with both grippers clear, for SUCCESS_HOLD_S without a break; the tube's set angle then picks the line."""
        if bend - self.back > self.set_angle:
            self.set_angle = bend - self.back
        self.tube = max(self.set_angle, bend)
        if self.set_angle > OVER and not self.too_far:
            self.too_far = True
            print(f"[tube] set past {OVER:.0f} degrees at {t:.2f} s", flush=True)
        if (abs(pos[0] - 0.36) > 0.55 or abs(pos[1]) > 0.65 or tilt > 30.0) and not self.off_bench:
            self.off_bench = True
            print(f"[tube] off the bench at {t:.2f} s", flush=True)
        self.recent = (self.recent + [bend])[-6:]
        still = len(self.recent) == 6 and max(self.recent) - min(self.recent) < 0.5
        origin = np.array([0.0, 0.0, BAR_Z])
        gaps = min(segment_gap(local_tips, origin, origin + LEVER_L * _dir(bend)), segment_gap(local_tips, origin, origin + np.array([0.0, FORM_L, 0.0])))
        settled = bend <= RELAXED and still and gaps >= CLEAR
        self.settled_s = self.settled_s + dt if settled else 0.0
        in_band = abs(self.set_angle - TARGET) <= TOL
        self.full = bool(settled and not self.too_far and not self.off_bench and in_band)
        if self.too_far or self.off_bench or self.settled_s < SUCCESS_HOLD_S - 1e-6:
            self.score = 0.0
        elif in_band:
            self.score = 1.0
        elif self.set_angle >= 80.0:
            self.score = 0.5
        elif self.set_angle >= PARTIAL_FROM:
            self.score = 0.25
        else:
            self.score = 0.0


TRIAL = Trial()
m.on_step(TRIAL.step)
m.on_reset(TRIAL.reset)


def success(objects) -> bool:
    """The tube set within 3 degrees of 90, with the lever relaxed and still and both grippers clear, at this step. The harness ends the trial once this has held for SUCCESS_HOLD_S."""
    return TRIAL.full


def score(objects) -> float:
    """Once the lever has been relaxed and still, with both grippers clear, for 1 s: 1.0 with the tube in the band from 87 to 93 degrees; 0.5 from 80 to under 87, or over 93 up to 96; 0.25 from 70 to under 80. 0 before that second, under 70, past 96 degrees, or off the bench."""
    return TRIAL.score


def distance(objects) -> float:
    return abs(TARGET - objects["bender"].deg("tube") * -1.0) / TARGET


# -- the demo: from the lever's and the tube's angles, as they can be seen -----------------------


def demo_script(ctx):
    """Modes: full, and one_arm: the right arm pulls the lever, with nothing holding the bender."""
    if ctx.mode not in ("full", "one_arm"):
        raise ValueError(f"unknown mode {ctx.mode!r}")
    l, r = ctx.left, ctx.right
    lever = lambda: -ctx.deg("bender", "lever")  # noqa: E731
    tube = lambda: -ctx.deg("bender", "tube")  # noqa: E731
    pivot = lambda: ctx.pos("bender") + np.array([0.0, 0.0, BAR_Z - 0.001])  # noqa: E731
    over = np.array([0.0, 0.0, 0.03])
    lever_grip = pivot() + GRIP_AT * _dir(0.0)
    form_grip = pivot() + np.array([0.0, GRIP_AT, 0.0])
    yield from r.move(lever_grip + over, grip=d.down(90), t=2.0, label="over the lever")
    if ctx.mode == "full":
        yield from l.move(form_grip + over, grip=d.down(-90), t=2.0, label="over the form handle")
        yield from l.move(form_grip, t=0.4, label="on the form handle")
        yield from l.close(0.024, force=30.0, t=0.4, label="hold the form handle")
    yield from r.move(lever_grip, t=0.4, label="on the lever")
    yield from r.close(0.024, force=30.0, t=0.4, label="hold the lever")

    def pull_to(bend: float, t: float, label: str):
        """Swing the lever to this bend, by where it is now and where the bender is now."""
        for _ in range(4):
            gap = bend - lever()
            if abs(gap) < 1.0:
                return
            yield from r.arc(pivot(), (0.0, 0.0, 1.0), -gap, t=max(0.4, t * abs(gap) / 80.0), label=label)
            yield from r.wait(0.2)
            t = 0.6

    yield from pull_to(80.0, 2.5, "first pull")
    peak = lever()
    print(f"[tube] first pull reached {peak:.1f} degrees", flush=True)
    if ctx.mode == "one_arm":
        st = ctx.state("bender")
        print(f"[tube] with one arm, the bender moved {100 * np.linalg.norm(st.pos[:2] - np.array(PIVOT)):.1f} cm and turned {np.degrees(2 * np.arctan2(st.quat_xyzw[2], st.quat_xyzw[3])):.0f} degrees; the lever is at {lever():.1f}", flush=True)
        yield from r.open()
        return
    yield from pull_to(max(peak - 25.0, 40.0), 1.2, "relax to read")
    yield from r.wait(0.3)
    read = tube()
    springback = peak - read
    print(f"[tube] relaxed, the tube reads {read:.1f}: it sprang back {springback:.1f} degrees", flush=True)
    yield from pull_to(TARGET + springback, 2.0, "second pull")
    print(f"[tube] second pull reached {lever():.1f} degrees", flush=True)
    yield from pull_to(25.0, 2.0, "relax")
    yield from r.open()
    yield from r.move(r.at() + [0.0, 0.0, 0.05], t=0.5, label="off the lever")
    yield from r.move((0.20, -0.33, TOP + 0.16), grip=d.down(0), t=1.2, label="clear")
    yield from l.open()
    yield from l.move(l.at() + [0.0, 0.0, 0.05], t=0.5, label="off the form handle")
    yield from l.move((0.20, 0.33, TOP + 0.16), grip=d.down(0), t=1.2, label="clear")
    print(f"[tube] the tube reads {tube():.1f} degrees", flush=True)
    yield from r.wait(2.0)
