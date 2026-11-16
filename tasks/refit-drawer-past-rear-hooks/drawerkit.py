"""drawerkit: rigid parts built from boxes, free or fixed in place, for yam_arena task files.

yam_arena's task-file objects are single primitives. A ``Part`` is one rigid body made of several box colliders,
each with its own colour and friction. ``mass=None`` fixes the part where it is put (a kinematic body).

Importing this module hooks ``yam_arena.scene`` and the embodiment's step and reset without modifying yam_arena:
``success()/score()`` receive the body origin's pose of each part. ``on_step``/``on_reset`` register callbacks run
after every control step / reset, and ``env()`` returns the Isaac Lab environment (used by the task to see where the
grippers are). Pure Python at import time; everything that needs Isaac Sim is imported lazily.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import math
import sys
from dataclasses import dataclass

import numpy as np
from yam_arena import task_file
from yam_arena.task_file import Box, ObjectState

__all__ = ["Part", "part", "rot", "on_reset", "on_step", "env", "gripper_points", "install"]

# A box: (centre, size, colour, friction) in the part frame (m); friction None uses the part's default.
BoxSpec = tuple


@dataclass(frozen=True)
class Part(Box):
    """A rigid part. It subclasses Box only so yam_arena's task-file checks accept it; ``size`` is its bounding
    box and ``pos`` the body origin. ``mass`` None: fixed in place."""

    boxes: tuple[BoxSpec, ...] = ()
    com: tuple[float, float, float] | None = None
    friction: float = 0.5


def part(name, pos, boxes, mass=None, com=None, color=(0.5, 0.5, 0.5), friction=0.5) -> Part:
    pts = [np.array(b[0]) + s * np.array(b[1]) / 2 for b in boxes for s in (-1, 1)]
    lo, hi = np.min(pts, axis=0), np.max(pts, axis=0)
    return Part(name=name, pos=tuple(float(v) for v in pos), size=tuple((hi - lo).tolist()),
                mass=None if mass is None else float(mass), color=color, boxes=tuple(boxes),
                com=None if com is None else tuple(float(c) for c in com), friction=friction)


def rot(quat_xyzw) -> np.ndarray:
    x, y, z, w = quat_xyzw
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


# -- spawning -----------------------------------------------------------------------------------------------------

_SPECS: dict[str, Part] = {}


def _spawn_part(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material, bind_visual_material
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    from yam_arena.station import linear_rgb

    spec: Part = _SPECS[cfg.part_name]
    stage = get_current_stage()
    root = UsdGeom.Xform.Define(stage, prim_path)
    root.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*(translation or (0.0, 0.0, 0.0))))
    x, y, z, w = orientation if orientation is not None else (0.0, 0.0, 0.0, 1.0)
    root.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(w, x, y, z))

    mats: dict[float, str] = {}
    looks: dict[tuple, str] = {}

    def material(mu: float) -> str:
        if mu not in mats:
            path = f"{prim_path}/Looks/phys{len(mats)}"
            pm = sim_utils.RigidBodyMaterialCfg(static_friction=mu, dynamic_friction=mu)
            pm.func(path, pm)
            mats[mu] = path
        return mats[mu]

    def look(color) -> str:
        if color not in looks:
            path = f"{prim_path}/Looks/c{len(looks)}"
            vis = sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(color), roughness=0.5)
            vis.func(path, vis)
            looks[color] = path
        return looks[color]

    main = f"{prim_path}/body"
    xform = UsdGeom.Xform.Define(stage, main)
    xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0.0, 0.0, 0.0))
    xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1, 0, 0, 0))
    prim = xform.GetPrim()
    rb = UsdPhysics.RigidBodyAPI.Apply(prim)
    if spec.mass is None:
        rb.CreateKinematicEnabledAttr(True)
    else:
        mapi = UsdPhysics.MassAPI.Apply(prim)
        mapi.CreateMassAttr(spec.mass)
        if spec.com is not None:
            mapi.CreateCenterOfMassAttr(Gf.Vec3f(*spec.com))
    prb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    prb.CreateMaxDepenetrationVelocityAttr(1.0)
    prb.CreateSolverPositionIterationCountAttr(32)
    prb.CreateSolverVelocityIterationCountAttr(4)

    for i, (centre, size, color, *mu) in enumerate(spec.boxes):
        cube = UsdGeom.Cube.Define(stage, f"{main}/box_{i}")
        cube.CreateSizeAttr(1.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(*centre))
        cube.AddScaleOp().Set(Gf.Vec3f(*size))
        p = cube.GetPrim()
        UsdPhysics.CollisionAPI.Apply(p)
        cp = PhysxSchema.PhysxCollisionAPI.Apply(p)
        cp.CreateContactOffsetAttr(0.002)
        cp.CreateRestOffsetAttr(0.0)
        friction = mu[0] if mu and mu[0] is not None else spec.friction
        bind_physics_material(str(p.GetPath()), material(float(friction)), stage=stage)
        bind_visual_material(str(p.GetPath()), look(tuple(color)), stage=stage)
    return root.GetPrim()


def _arena_object(spec: Part):
    from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    @configclass
    class PartSpawnerCfg(SpawnerCfg):
        func: object = clone(_spawn_part)
        part_name: str = ""

    return Object(name=spec.name, object_type=ObjectType.RIGID, spawner_cfg=PartSpawnerCfg(part_name=spec.name),
                  initial_pose=Pose(position_xyz=tuple(spec.pos), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))


def _np(x) -> np.ndarray:
    return getattr(x, "torch", x).detach().cpu().numpy().astype(np.float64)


def _part_state(environment, spec: Part) -> ObjectState:
    base = environment.unwrapped
    asset = base.scene[spec.name]
    origin = _np(base.scene.env_origins[0])
    pose = _np(asset.data.root_link_pose_w)[0]
    vel = _np(asset.data.root_vel_w)[0] if hasattr(asset.data, "root_vel_w") else np.zeros(6)
    R = rot(pose[3:7])
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * np.array(spec.size) / 2
    pts = pose[:3] - origin + corners @ R.T
    return ObjectState(pos=pose[:3] - origin, quat_xyzw=pose[3:7], lin_vel=vel[:3], ang_vel=vel[3:6],
                       low=pts.min(0), high=pts.max(0))


def gripper_points(environment) -> list[np.ndarray]:
    """World points (env frame) on both grippers: the palm, the centre between the finger pads and each pad tip."""
    base = environment.unwrapped
    origin = _np(base.scene.env_origins[0])
    out = []
    for key in ("left_robot", "right_robot"):
        robot = base.scene[key]
        names = list(robot.body_names)
        g = _np(robot.data.body_link_pose_w)[0][names.index("gripper")]
        R, p = rot(g[3:7]), g[:3] - origin
        jn = list(robot.joint_names)
        half = abs(float(_np(robot.data.joint_pos)[0][jn.index("joint7")])) + 0.008
        for local in ((0, 0, 0.06), (0, 0, 0.134), (half, 0, 0.147), (-half, 0, 0.147), (half, 0, 0.12),
                      (-half, 0, 0.12)):
            out.append(p + R @ np.array(local, float))
    return out


def joint_positions(environment) -> dict[str, np.ndarray]:
    base = environment.unwrapped
    return {k: _np(base.scene[k].data.joint_pos)[0] for k in ("left_robot", "right_robot")}


# -- hooking yam_arena -------------------------------------------------------------------------------------------

_STEP: list = []
_RESET: list = []
_ENV: dict = {"env": None}


def on_step(fn):
    _STEP.append(fn)
    return fn


def on_reset(fn):
    _RESET.append(fn)
    return fn


def env():
    return _ENV["env"]


def _patch_scene(scene) -> None:
    if getattr(scene, "_drawerkit_patched", False):
        return
    orig_task_object, orig_states = scene.task_object, scene.object_states

    def task_object(spec):
        if isinstance(spec, Part):
            _SPECS[spec.name] = spec
            return _arena_object(spec)
        return orig_task_object(spec)

    def object_states(environment, names):
        _ENV["env"] = environment
        plain = [n for n in names if n not in _SPECS]
        out = orig_states(environment, plain) if plain else {}
        for n in names:
            if n in _SPECS:
                out[n] = _part_state(environment, _SPECS[n])
        return {n: out[n] for n in names}

    scene.task_object = task_object
    scene.object_states = object_states
    scene._drawerkit_patched = True


def _patch_embodiment(ie) -> None:
    cls = ie.YamArenaEmbodiment
    if getattr(cls, "_drawerkit_patched", False):
        return
    base_task_step, base_reset = cls._task_step, cls.reset

    def _task_step(self, *a, **k):
        _ENV["env"] = self._env
        for fn in _STEP:
            fn(self._env)
        return base_task_step(self, *a, **k)

    def reset(self, *a, **k):
        obs = base_reset(self, *a, **k)
        _ENV["env"] = self._env
        for fn in _RESET:
            fn(self._env)
        return obs

    cls._task_step, cls.reset = _task_step, reset
    cls._drawerkit_patched = True


class _Finder(importlib.abc.MetaPathFinder):
    targets = {"yam_arena.scene": _patch_scene, "yam_arena.inspect_embodiment": _patch_embodiment}

    def find_spec(self, fullname, path, target=None):
        if fullname not in self.targets:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        loader, patch = spec.loader, self.targets[fullname]
        orig_exec = loader.exec_module

        def exec_module(module):
            orig_exec(module)
            patch(module)

        loader.exec_module = exec_module
        return spec


def install() -> None:
    for name, patch in _Finder.targets.items():
        if name in sys.modules:
            patch(sys.modules[name])
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())


install()
assert task_file and math
