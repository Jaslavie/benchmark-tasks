"""Inspect Robots embodiment and scorer. Policy observations contain RGB and robot state only."""
import math
import numpy as np

from inspect_robots.embodiment import EmbodimentInfo
from inspect_robots.registry import embodiment, policy, scorer, task
from inspect_robots.scene import Scene
from inspect_robots.scorer import Score
from inspect_robots.spaces import ActionSemantics, Box, CameraSpec, ObservationSpace, StateField, StateSpec
from inspect_robots.task import Task
from inspect_robots.types import Observation, StepResult

from .environment import BottleEnv
from .rendering import Views
from .scoring import INSTRUCTION, MAX_STEPS, Scorer


def action_space():
    labels = tuple(f"{arm}_{key}" for arm in ("left", "right")
                   for key in ("x", "y", "z", "roll", "pitch", "yaw", "jaw_gap"))
    return Box(shape=(14,),
               low=np.tile([-.30, -.40, .81, -math.pi, -math.pi, -math.pi, 0.], 2),
               high=np.tile([.35, .40, 1.40, math.pi, math.pi, math.pi, .09], 2),
               semantics=ActionSemantics(control_mode="eef_abs_pose", rotation_repr="euler_xyz",
                                         gripper="continuous", frame="world", dim_labels=labels,
                                         # Euler targets may cross +/-pi. Native
                                         # control enforces the 0.08 rad bound
                                         # on the shortest SO(3) rotation.
                                         max_step=tuple([.008]*3 + [2*math.pi]*3 + [.01]) * 2))


class DetentBottleEmbodiment:
    def __init__(self, resolution=320):
        resolution = int(resolution)
        if not 64 <= resolution <= 640:
            raise ValueError("resolution must be in 64..640")
        self.resolution = resolution
        self.env = self.views = None
        self.max_steps = MAX_STEPS
        fields = StateSpec(fields=(
            StateField("eef_state", (14,), "left then right XYZ m, XYZ Euler rad, jaw gap m"),
            StateField("joint_pos", (14,), "rad"),
            StateField("joint_vel", (14,), "rad/s"),
            StateField("gripper_width", (2,), "m"),
            StateField("wrist_wrench", (2, 6), "world N and N.m"),
        ))
        self.info = EmbodimentInfo(
            name="detent_bottle", morphology="dual-panda-detent-bottle",
            action_space=action_space(), control_hz=20., is_simulated=True,
            observation_space=ObservationSpace(
                cameras=tuple(CameraSpec(name, resolution, resolution) for name in ("overview", "closeup", "front")),
                state=fields),
            capabilities=frozenset({"resettable", "auto_reset", "privileged_success", "renderable"}),
            environment_id="mujoco-3.3.7-detent-bottle-v1",
            docs="Two Panda arms with parallel grippers on a table at world Z=0.80 m. "
                 "Actions are absolute targets: left XYZ metres, fixed-axis XYZ Euler radians, jaw gap metres; "
                 "then the same seven fields for the right arm. Jaw gap 0 closes, 0.09 opens. "
                 "Tool +Z points from palm to fingertips, local X is the jaw-opening direction; "
                 "roll=pi,pitch=yaw=0 points down, roll=pi/2 points toward world -Y. "
                 "Each control tick is 0.05 s. Translation is limited to 8 mm and rotation to 0.08 rad per tick. "
                 "The other arm retains its target. Only robot motors can be commanded. "
                 "Camera reads do not advance physics. This is one deterministic starting layout. "
                 "Observe the mechanism through RGB and infer its operation through contact. "
                 "Evaluator object measurements are excluded from observations.",
        )

    def bind_task(self, envelope):
        self.max_steps = min(MAX_STEPS, envelope.max_steps)

    def reset(self, scene, *, seed=None):
        self.close()
        self.instruction = scene.instruction
        self.env = BottleEnv(max_steps=self.max_steps)
        self.views = Views(self.env, self.resolution)
        return self.observe()

    def observe(self):
        e = self.env
        measured = []
        for arm in (0, 1):
            rotation = e.Rotation.from_matrix(e.data.site_xmat[e.tips[arm]].reshape(3, 3))
            measured.extend(e.data.site_xpos[e.tips[arm]].tolist())
            measured.extend(rotation.as_euler("xyz").tolist())
            measured.append(float(np.sum(e.data.qpos[e.finger_qids[arm]])))
        now = e.steps / 20
        return Observation(
            images={name: self.views.image(name) for name in ("overview", "closeup", "front")},
            state=dict(eef_state=np.asarray(measured),
                       joint_pos=np.concatenate([e.data.qpos[q].copy() for q in e.qids]),
                       joint_vel=np.concatenate([e.data.qvel[d].copy() for d in e.dids]),
                       gripper_width=np.array([np.sum(e.data.qpos[q]) for q in e.finger_qids]),
                       wrist_wrench=e.wrenches_world()),
            instruction=self.instruction,
            state_time=now, image_times={name: now for name in ("overview", "closeup", "front")})

    def step(self, action):
        if self.env.reason:
            raise RuntimeError("Episode has ended; reset before sending another action")
        values = np.array(action.data, dtype=float, copy=True)
        space = self.info.action_space
        if values.shape != (14,) or not np.isfinite(values).all() or np.any(values < space.low) or np.any(values > space.high):
            raise ValueError("Action must be fourteen finite targets within the declared bounds")
        values[3:6] = np.degrees(values[3:6])
        values[10:13] = np.degrees(values[10:13])
        self.env.action(values)
        reason = self.env.reason
        return StepResult(observation=self.observe(),
                          terminated=reason is not None and reason != "time_limit",
                          truncated=reason == "time_limit", termination_reason=reason,
                          info={"measurements": self.env.measurements()})

    def close(self):
        if self.views:
            self.views.close()
            self.views = None
        if self.env:
            self.env.close()
            self.env = None


class BottleScorer:
    name = "detent_bottle_score"

    def __call__(self, record, target):
        evaluator = Scorer()
        for tick in record.steps:
            evaluator.update(tick.result.info["measurements"])
        return Score(value=evaluator.score, metadata=evaluator.result())


@embodiment("detent_bottle")
def make_embodiment(resolution=320):
    return DetentBottleEmbodiment(resolution=resolution)


@scorer("detent_bottle_score")
def make_scorer():
    return BottleScorer()


@task("detent-latch-bottle")
def make_task():
    return Task(name="detent-latch-bottle",
                scenes=[Scene(id="fixed-42", instruction=INSTRUCTION, init_seed=42)],
                scorer=BottleScorer(), max_steps=MAX_STEPS)


@policy("detent_bottle_demo")
def make_demo_policy():
    """An offline integration check; this policy is never a model-run submission."""
    from inspect_robots.policy import PolicyConfig, PolicyInfo
    from inspect_robots.types import Action, ActionChunk
    from .reference import opening_sequence

    class DemoPolicy:
        info = PolicyInfo(name="detent_bottle_demo", action_space=action_space(),
                          observation_space=ObservationSpace())
        config = PolicyConfig(action_horizon=1)

        def reset(self, scene):
            current = np.array([-.12, -.18, 1.12, math.pi, 0, 0, .09,
                                .13, .16, 1.12, math.pi, 0, 0, .09])
            actions = []
            for _, arm, position, rpy, gap, ticks in opening_sequence():
                current[7*arm:7*arm+7] = np.r_[position, np.radians(rpy), gap]
                actions.extend(current.copy() for _ in range(ticks))
            self.actions = iter(actions)
            self.last = current.copy()

        def act(self, observation):
            self.last = next(self.actions, self.last)
            return ActionChunk(actions=[Action(data=self.last.copy())], control_hz=20)

    return DemoPolicy()


if __name__ == "__main__":
    from inspect_robots.cli import main
    main()
