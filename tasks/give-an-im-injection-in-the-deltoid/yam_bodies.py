"""yam_bodies: rigid bodies made of boxes, joints between them that a task switches on and off, contact-force readings
and per-physics-step sampling, for a yam_arena task file. Adapted from the team's bodykit.

A :class:`Body` is one rigid body made of several boxes (some visual only). ``static=True`` makes it a fixed
collider with its own friction (a bench, a tray). Two bodies can be joined by a :func:`fixed_joint`, a
:func:`revolute_joint`, a :func:`prismatic_joint` (optionally with a sprung drive authored once, never changed) or a
:func:`d6_joint` (some axes free, some limited, its two frames anywhere); :func:`set_joint` switches one on or off while
the episode runs (PhysX honours a
``physics:jointEnabled`` change on the GPU; it does not honour runtime drive targets, kinematic targets, or pose writes on
one jointed body without its neighbours, so this module uses none of those). Every reset writes all free bodies back to
their start poses together and re-arms every joint.

:func:`contact_view` reads the force on one body from a list of others (PhysX's contact-force matrix, as Isaac Lab's
ContactSensor reads it); the body needs ``contact_report=True``. :func:`on_substep` runs a function after every physics
step (240 Hz) rather than every control step (20 Hz), by wrapping the scene's update.

Pure Python and NumPy at import; Isaac Sim is imported lazily, so the task file loads (and its scorer is tested) without
the simulator.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import math
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
from yam_arena.task_file import Box, ObjectState

ENV = "/World/envs/env_0"
Color = tuple[float, float, float]


@dataclass(frozen=True)
class BoxShape:
    """A box in the body frame: ``centre``, ``size`` (m), rotated by ``quat_xyzw``. ``color=None``: collider only."""

    centre: tuple[float, float, float]
    size: tuple[float, float, float]
    color: Color | None = (0.6, 0.6, 0.6)
    collide: bool = True
    quat_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

    def half_extent(self) -> np.ndarray:
        return np.abs(rotation(self.quat_xyzw)) @ (np.array(self.size) / 2)


def rotation(q) -> np.ndarray:
    """The rotation matrix of quaternion ``q`` (xyzw), normalised first (a task's fail-safe checks the raw norm)."""
    q = np.asarray(q, dtype=np.float64)
    n = float(np.linalg.norm(q))
    x, y, z, w = (q / n) if n > 0.0 and math.isfinite(n) else q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


@dataclass(frozen=True)
class Body(Box):
    """A rigid body of ``shapes``; ``pos`` is its origin (world, m). It subclasses Box so yam_arena's checks accept it.
    ``static``: a fixed collider (mass ignored). ``com``: centre of mass in the body frame. ``ignore``: names of bodies
    (earlier in OBJECTS) it never collides with. ``contact_report``: PhysX reports its contacts (for contact_view)."""

    shapes: tuple = ()
    static: bool = False
    friction: tuple[float, float] = (0.6, 0.5)
    com: tuple[float, float, float] | None = None
    inertia: tuple[float, float, float] | None = None
    ignore: tuple[str, ...] = ()
    contact_report: bool = False
    contact_offset: float = 0.002
    solver_iterations: int = 16
    damping: tuple[float, float] = (0.0, 0.05)
    max_depenetration: float = 1.0
    torsional_patch: float | None = None  # m: a contact patch this wide resists twisting in a pinch (as mechkit's free parts)
    sleep_threshold: float | None = None  # PhysX sleep threshold (None: PhysX's default); 0 keeps the body awake (probe)
    velocity_iterations: int | None = None  # solver velocity iterations (None: PhysX's default)
    gyroscopic: bool | None = None  # PhysX gyroscopic forces (None: the default)
    stabilization: float | None = None  # PhysX stabilization threshold (None: the default)
    articulation_root: bool = False  # the floating root link of a reduced-coordinate articulation (its `articulated` joints)

    @staticmethod
    def build(name: str, pos, shapes, mass: float | None = None, **kw) -> Body:
        lo = np.min([np.array(s.centre) - s.half_extent() for s in shapes], axis=0)
        hi = np.max([np.array(s.centre) + s.half_extent() for s in shapes], axis=0)
        return Body(name=name, pos=tuple(float(v) for v in pos), size=tuple((hi - lo).tolist()), mass=mass, shapes=tuple(shapes), **kw)

    def local_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lo = np.min([np.array(s.centre) - s.half_extent() for s in self.shapes], axis=0)
        hi = np.max([np.array(s.centre) + s.half_extent() for s in self.shapes], axis=0)
        return lo, hi

    def _check(self) -> None:
        super()._check()
        if not self.shapes:
            raise ValueError(f"{self.name}: a Body needs at least one shape")
        if self.static == (self.mass is not None):
            raise ValueError(f"{self.name}: a static Body has no mass; a free one needs one")


@dataclass(frozen=True)
class BodyState(ObjectState):
    """``pos``/``quat_xyzw``: the body origin's pose (world)."""

    def rotation(self) -> np.ndarray:
        return rotation(self.quat_xyzw)

    def to_local(self, p) -> np.ndarray:
        return (np.atleast_2d(p) - self.pos) @ self.rotation()

    def to_world(self, p) -> np.ndarray:
        return np.atleast_2d(p) @ self.rotation().T + self.pos


def state_from_pose(spec: Body, pos, quat, lin=(0, 0, 0), ang=(0, 0, 0)) -> BodyState:
    lo, hi = spec.local_bounds()
    corners = np.array([[a, b, c] for a in (lo[0], hi[0]) for b in (lo[1], hi[1]) for c in (lo[2], hi[2])])
    pts = np.asarray(pos, float) + corners @ rotation(quat).T
    return BodyState(pos=np.asarray(pos, float), quat_xyzw=np.asarray(quat, float), lin_vel=np.asarray(lin, float),
                     ang_vel=np.asarray(ang, float), low=pts.min(0), high=pts.max(0))


# -- joints --------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Joint:
    """``kind`` "fixed", "revolute", "prismatic" or "d6" between ``body0`` and ``body1`` (both unrotated at the start;
    body0 earlier in OBJECTS; body0 may be static), at world point ``anchor`` on body0 and ``anchor1`` (default the
    same point) on body1, both frames turned by ``frame_quat`` (xyzw) from the bodies' frames. A revolute turns about
    ``axis`` within ``limits_deg``; a prismatic slides along it within ``limits`` (m) with an optional linear ``drive``
    (stiffness N/m, damping N s/m, target m); a d6 leaves every axis free except those in ``d6_limits``
    ({"transY": (lo, hi), "rotZ": (lo_deg, hi_deg)...}). ``on`` at every reset. ``break_force`` (N, authored once):
    PhysX breaks the joint for good when its constraint force passes it (a breakaway without a runtime switch; a
    broken joint is re-made at the next reset). ``remake``: never switched off and on at a reset; left as authored at
    the first reset and re-made from scratch at every later one (as a breakable joint is)."""

    name: str
    kind: str
    body0: str
    body1: str
    anchor: tuple[float, float, float]
    axis: str = "X"
    limits_deg: tuple[float, float] | None = None
    on: bool = True
    anchor1: tuple[float, float, float] | None = None
    frame_quat: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    limits: tuple[float, float] | None = None
    drive: tuple[float, float, float] | None = None
    d6_limits: tuple = ()
    break_force: float | None = None
    remake: bool = False
    articulated: bool = False  # part of the articulation rooted at an `articulation_root` body: solved in reduced
    # coordinates (its drive is integrated by the solver exactly); never switched at run time or at a reset

    @property
    def rebuilt(self) -> bool:
        """Re-made from scratch at later resets rather than switched off and on at every reset."""
        return self.remake or self.break_force is not None


_SPECS: dict[str, Body] = {}
_JOINTS: dict[str, Joint] = {}
_JOINT_ATTR: dict[str, Any] = {}


def fixed_joint(name, body0, body1, anchor, on=True) -> Joint:
    j = Joint(name, "fixed", body0, body1, tuple(anchor), on=on)
    _JOINTS[name] = j
    return j


def revolute_joint(name, body0, body1, anchor, axis="X", limits_deg=None, on=True) -> Joint:
    j = Joint(name, "revolute", body0, body1, tuple(anchor), axis, limits_deg, on)
    _JOINTS[name] = j
    return j


def prismatic_joint(name, body0, body1, anchor, axis="Z", limits=None, drive=None, on=True, articulated=False) -> Joint:
    j = Joint(name, "prismatic", body0, body1, tuple(anchor), axis, None, on, limits=limits, drive=drive, articulated=bool(articulated))
    _JOINTS[name] = j
    return j


def d6_joint(name, body0, body1, anchor0, anchor1, frame_quat, limits: dict, on=True, break_force=None, remake=False,
             drive=None) -> Joint:
    """``drive``: (stiffness N/m, damping N s/m, target m) on transX, authored once (never changed at run time)."""
    j = Joint(name, "d6", body0, body1, tuple(anchor0), on=on, anchor1=tuple(anchor1), frame_quat=tuple(frame_quat),
              d6_limits=tuple(sorted(limits.items())), break_force=None if break_force is None else float(break_force),
              remake=bool(remake), drive=None if drive is None else (float(drive[0]), float(drive[1]), float(drive[2])))
    _JOINTS[name] = j
    return j


def _author_joints(stage, prim_path: str, spec: Body) -> None:
    for j in _JOINTS.values():
        if j.body1 == spec.name:
            _author_joint(stage, prim_path, spec, j, f"{prim_path}/{j.name}")


_JOINT_PATH: dict[str, str] = {}  # joint name -> its prim's path, when re-made under a new name (a broken joint)
_GEN = [0, 0]  # re-made joints so far; resets so far


def _joint_path(name: str) -> str:
    j = _JOINTS[name]
    return _JOINT_PATH.get(name, f"{ENV}/{j.body1}/{name}")


def _author_joint(stage, prim_path: str, spec: Body, j: Joint, path: str) -> None:
    from pxr import Gf, UsdPhysics

    parent = prim_path.rsplit("/", 1)[0]
    b0 = _SPECS[j.body0]
    define = {"fixed": UsdPhysics.FixedJoint, "revolute": UsdPhysics.RevoluteJoint, "prismatic": UsdPhysics.PrismaticJoint,
              "d6": UsdPhysics.Joint}[j.kind]
    joint = define.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([f"{parent}/{j.body0}/body"])
    joint.CreateBody1Rel().SetTargets([f"{prim_path}/body"])
    a1 = j.anchor if j.anchor1 is None else j.anchor1
    joint.CreateLocalPos0Attr(Gf.Vec3f(*(np.asarray(j.anchor, float) - np.asarray(b0.pos, float)).tolist()))
    joint.CreateLocalPos1Attr(Gf.Vec3f(*(np.asarray(a1, float) - np.asarray(spec.pos, float)).tolist()))
    qx, qy, qz, qw = (float(v) for v in j.frame_quat)
    joint.CreateLocalRot0Attr(Gf.Quatf(qw, qx, qy, qz))
    joint.CreateLocalRot1Attr(Gf.Quatf(qw, qx, qy, qz))
    if j.kind == "prismatic":
        joint.CreateAxisAttr(j.axis)
        if j.limits is not None:
            joint.CreateLowerLimitAttr(float(j.limits[0]))
            joint.CreateUpperLimitAttr(float(j.limits[1]))
        if j.drive is not None:
            drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
            drive.CreateTypeAttr("force")
            drive.CreateStiffnessAttr(float(j.drive[0]))
            drive.CreateDampingAttr(float(j.drive[1]))
            drive.CreateTargetPositionAttr(float(j.drive[2]))
            drive.CreateTargetVelocityAttr(0.0)
            drive.CreateMaxForceAttr(1.0e4)
    if j.kind == "d6":
        if j.drive is not None:
            drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "transX")
            drive.CreateTypeAttr("force")
            drive.CreateStiffnessAttr(float(j.drive[0]))
            drive.CreateDampingAttr(float(j.drive[1]))
            drive.CreateTargetPositionAttr(float(j.drive[2]))
            drive.CreateMaxForceAttr(1.0e4)
        for dof, (lo, hi) in j.d6_limits:
            lim = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), dof)
            lim.CreateLowAttr(float(lo))
            lim.CreateHighAttr(float(hi))
    if j.kind == "revolute":
        joint.CreateAxisAttr(j.axis)
        if j.limits_deg is not None:
            joint.CreateLowerLimitAttr(float(j.limits_deg[0]))
            joint.CreateUpperLimitAttr(float(j.limits_deg[1]))
    if j.break_force is not None:
        joint.CreateBreakForceAttr(float(j.break_force))
    joint.CreateJointEnabledAttr(bool(j.on))
    joint.CreateCollisionEnabledAttr(False)
    joint.CreateExcludeFromArticulationAttr(not j.articulated)
    _JOINT_ATTR.pop(j.name, None)


def joint_attrs(name: str) -> dict:
    """A joint as the stage has it now (for traces): enabled, break force, and whether its prim exists."""
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import UsdPhysics

    prim = get_current_stage().GetPrimAtPath(_joint_path(name))
    if not prim or not prim.IsValid():
        return {"prim": False}
    js = UsdPhysics.Joint(prim)
    out = {"prim": True, "enabled": bool(js.GetJointEnabledAttr().Get()), "break": js.GetBreakForceAttr().Get()}
    for attr in ("physxJoint:maxJointVelocity", "physxJoint:armature", "physxJoint:jointFriction"):  # authored only if set
        a = prim.GetAttribute(attr)
        if a and a.HasAuthoredValue():
            out[attr.split(":")[1]] = a.Get()
    for inst in ("transX", "linear"):
        if prim.HasAPI(UsdPhysics.DriveAPI, inst):
            d = UsdPhysics.DriveAPI(prim, inst)
            out[f"drive_{inst}"] = {"stiffness": d.GetStiffnessAttr().Get(), "damping": d.GetDampingAttr().Get(),
                                    "target": d.GetTargetPositionAttr().Get(), "target_v": d.GetTargetVelocityAttr().Get(),
                                    "max_force": d.GetMaxForceAttr().Get(), "type": d.GetTypeAttr().Get()}
    return out


def set_joint(name: str, on: bool) -> None:
    """Switch a joint on or off (``physics:jointEnabled``; PhysX rebuilds a joint switched on)."""
    if name not in _JOINT_ATTR:
        from isaaclab.sim.utils.stage import get_current_stage
        from pxr import UsdPhysics

        prim = get_current_stage().GetPrimAtPath(_joint_path(name))
        _JOINT_ATTR[name] = UsdPhysics.Joint(prim).GetJointEnabledAttr()
    _JOINT_ATTR[name].Set(bool(on))


# -- spawning --------------------------------------------------------------------------------------------------------


def _spawn_body(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material, bind_visual_material
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics
    from yam_arena.station import linear_rgb

    spec = _SPECS[cfg.body_name]
    stage = get_current_stage()
    root = UsdGeom.Xform.Define(stage, prim_path)
    root.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*(translation or (0.0, 0.0, 0.0))))
    x, y, z, w = orientation if orientation is not None else (0.0, 0.0, 0.0, 1.0)
    root.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(w, x, y, z))
    phys = f"{prim_path}/Looks/physics"
    material = sim_utils.RigidBodyMaterialCfg(static_friction=spec.friction[0], dynamic_friction=spec.friction[1])
    material.func(phys, material)
    looks: dict = {}

    def look(color) -> str:
        if color not in looks:
            path = f"{prim_path}/Looks/c{len(looks)}"
            surface = sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(color), roughness=0.6)
            surface.func(path, surface)
            looks[color] = path
        return looks[color]

    body_path = f"{prim_path}/body"
    xform = UsdGeom.Xform.Define(stage, body_path)
    xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0.0, 0.0, 0.0))
    xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1, 0, 0, 0))
    prim = xform.GetPrim()
    if not spec.static:
        UsdPhysics.RigidBodyAPI.Apply(prim)
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_api.CreateMassAttr(float(spec.mass))
        if spec.com is not None:
            mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in spec.com]))
        if spec.inertia is not None:
            mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in spec.inertia]))
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        rb.CreateMaxDepenetrationVelocityAttr(float(spec.max_depenetration))
        rb.CreateSolverPositionIterationCountAttr(int(spec.solver_iterations))
        rb.CreateLinearDampingAttr(float(spec.damping[0]))
        rb.CreateAngularDampingAttr(float(spec.damping[1]))
        if spec.sleep_threshold is not None:
            rb.CreateSleepThresholdAttr(float(spec.sleep_threshold))
        if spec.velocity_iterations is not None:
            rb.CreateSolverVelocityIterationCountAttr(int(spec.velocity_iterations))
        if spec.gyroscopic is not None:
            rb.CreateEnableGyroscopicForcesAttr(bool(spec.gyroscopic))
        if spec.stabilization is not None:
            rb.CreateStabilizationThresholdAttr(float(spec.stabilization))
        if spec.articulation_root:  # a floating-base articulation: this body is its root link
            UsdPhysics.ArticulationRootAPI.Apply(prim)
            art = PhysxSchema.PhysxArticulationAPI.Apply(prim)
            art.CreateSolverPositionIterationCountAttr(int(spec.solver_iterations))
            if spec.velocity_iterations is not None:
                art.CreateSolverVelocityIterationCountAttr(int(spec.velocity_iterations))
            art.CreateSleepThresholdAttr(0.0)
            art.CreateEnabledSelfCollisionsAttr(False)
        if spec.contact_report:
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
        if spec.ignore:
            parent = prim_path.rsplit("/", 1)[0]
            pairs = UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel()
            for other in spec.ignore:
                pairs.AddTarget(f"{parent}/{other}/body")
    for i, shape in enumerate(spec.shapes):
        geom = UsdGeom.Cube.Define(stage, f"{body_path}/s{i}")
        geom.CreateSizeAttr(1.0)
        geom.AddTranslateOp().Set(Gf.Vec3d(*shape.centre))
        qx, qy, qz, qw = shape.quat_xyzw
        if (qx, qy, qz) != (0.0, 0.0, 0.0):
            geom.AddOrientOp().Set(Gf.Quatf(qw, qx, qy, qz))
        geom.AddScaleOp().Set(Gf.Vec3f(*shape.size))
        path = str(geom.GetPrim().GetPath())
        if shape.color is None:
            UsdGeom.Imageable(geom.GetPrim()).MakeInvisible()
        else:
            bind_visual_material(path, look(tuple(shape.color)), stage=stage)
        if shape.collide:
            UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
            col = PhysxSchema.PhysxCollisionAPI.Apply(geom.GetPrim())
            col.CreateContactOffsetAttr(float(spec.contact_offset))
            col.CreateRestOffsetAttr(0.0)
            if spec.torsional_patch:
                col.CreateTorsionalPatchRadiusAttr(float(spec.torsional_patch))
                col.CreateMinTorsionalPatchRadiusAttr(0.8 * float(spec.torsional_patch))
            bind_physics_material(path, phys, stage=stage)
    _author_joints(stage, prim_path, spec)
    return root.GetPrim()


def _arena_object(spec: Body):
    from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    @configclass
    class BodySpawnerCfg(SpawnerCfg):
        func: object = clone(_spawn_body)
        body_name: str = ""

    return Object(name=spec.name, object_type=ObjectType.BASE if spec.static else ObjectType.RIGID,
                  spawner_cfg=BodySpawnerCfg(body_name=spec.name),
                  initial_pose=Pose(position_xyz=tuple(spec.pos), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))


def as_numpy(x) -> np.ndarray:
    """An Isaac Lab / warp / torch array as float64 NumPy."""
    x = getattr(x, "torch", x)
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy().astype(np.float64)
    try:
        import warp as wp

        if isinstance(x, wp.array):
            return wp.to_torch(x).detach().cpu().numpy().astype(np.float64)
    except (ImportError, AttributeError, TypeError, RuntimeError):  # not a warp array (or no warp): plain conversion below
        return np.asarray(x, dtype=np.float64)
    return np.asarray(x, dtype=np.float64)


def body_state(environment, name: str) -> BodyState:
    spec = _SPECS[name]
    if spec.static:
        return state_from_pose(spec, spec.pos, (0.0, 0.0, 0.0, 1.0))
    base = environment.unwrapped
    data = base.scene[name].data
    origin = as_numpy(base.scene.env_origins[0])
    pose = as_numpy(data.root_link_pose_w)[0]
    vel = as_numpy(data.root_com_vel_w)[0] if hasattr(data, "root_com_vel_w") else as_numpy(data.root_vel_w)[0]
    return state_from_pose(spec, pose[:3] - origin, pose[3:7], vel[:3], vel[3:6])


def push(environment, name: str, force, torque=(0.0, 0.0, 0.0)) -> None:
    """Hold ``force`` (N, world) on a free body's centre of mass, and ``torque`` (N m, world), until changed; every
    reset clears them."""
    import torch

    root = _art_root(name)
    if root is not None:  # a link of an articulation: its forces go in one call for all its links (_flush_art_forces)
        _ART_FORCES.setdefault(root, {})[name] = (np.asarray(force, np.float64), np.asarray(torque, np.float64))
        return
    asset = environment.unwrapped.scene[name]
    f = torch.tensor([[[float(v) for v in force]]], device=asset.device)
    t = torch.tensor([[[float(v) for v in torque]]], device=asset.device)
    asset.permanent_wrench_composer.set_forces_and_torques_index(forces=f, torques=t, body_ids=[0], is_global=True)
    _PUSHED.add(name)


_PUSHED: set[str] = set()
_ART_VIEWS: dict[str, object] = {}
_ART_FORCES: dict[str, dict] = {}  # articulation root body -> {link body: (force, torque), world}
_ART_LINK: dict[str, dict] = {}  # articulation root body -> {link body: its link index in the articulation view}


def _art_root(name: str) -> str | None:
    """The root body of the articulation ``name`` is a link of (itself, if it is the root), or None for a free body."""
    for j in _JOINTS.values():
        if j.articulated and j.body1 == name:
            return _art_root(j.body0)
    if any(j.articulated and j.body0 == name for j in _JOINTS.values()):
        return name
    return None


def _art_view(name: str):
    from isaaclab_physx.physics import PhysxManager

    if name not in _ART_VIEWS:
        _ART_VIEWS[name] = PhysxManager.get_physics_sim_view().create_articulation_view(f"{ENV}/{name}/body")
    return _ART_VIEWS[name]


def _flush_art_forces(env) -> None:
    """Every articulation's link forces and torques (world, at each link's centre of mass) in ONE call per articulation,
    after each physics step, for the next one. Sent per body through each asset's own rigid-body view, the force on the
    root link was not delivered (rig finding); PhysX clears applied forces after every step, so this runs every step."""
    import warp as wp

    base = env.unwrapped
    dev = str(base.device)
    for root, forces in _ART_FORCES.items():
        view = _art_view(root)
        if root not in _ART_LINK:  # which link is which body: by position, once
            tf = as_numpy(view.get_link_transforms()).reshape(view.count, view.max_links, 7)[0]
            origin = as_numpy(base.scene.env_origins[0])
            idx = {}
            for body in [root] + [j.body1 for j in _JOINTS.values() if j.articulated and _art_root(j.body1) == root]:
                p = body_state(env, body).pos + origin
                d = np.linalg.norm(tf[:, :3] - p, axis=1)
                idx[body] = int(np.argmin(d))
                if float(d.min()) > 0.002:
                    raise RuntimeError(f"articulation '{root}': no link at body '{body}' (nearest {1000 * float(d.min()):.1f} mm away)")
            if len(set(idx.values())) != len(idx):
                raise RuntimeError(f"articulation '{root}': bodies share a link index {idx}")
            _ART_LINK[root] = idx
            print(f"[yam_bodies] articulation '{root}': {view.count} x {view.max_links} links, {view.max_dofs} joint(s), link of each body {idx}", flush=True)
        F = np.zeros((view.count, view.max_links, 3), np.float32)
        Tq = np.zeros((view.count, view.max_links, 3), np.float32)
        for body, (f, tq) in forces.items():
            F[0, _ART_LINK[root][body]] = f
            Tq[0, _ART_LINK[root][body]] = tq
        view.apply_forces_and_torques_at_position(
            force_data=wp.array(F.reshape(-1), dtype=wp.float32, device=dev), torque_data=wp.array(Tq.reshape(-1), dtype=wp.float32, device=dev),
            position_data=None, indices=wp.array(np.arange(view.count, dtype=np.int32), device=dev), is_global=True)


# -- contact views ---------------------------------------------------------------------------------------------------

_VIEWS: dict[str, object] = {}


def contact_view(key: str, body_glob: str, filter_globs: list[str]):
    """A PhysX rigid-contact view of one body (``contact_report=True``) against ``filter_globs``, made once."""
    if key not in _VIEWS:
        from isaaclab_physx.physics import PhysxManager

        sim_view = PhysxManager.get_physics_sim_view()
        _VIEWS[key] = sim_view.create_rigid_contact_view([body_glob], filter_patterns=[list(filter_globs)], max_contact_data_count=256)
    return _VIEWS[key]


def force_matrix(view, dt: float) -> np.ndarray:
    """Forces on the view's body from each filter body: [filters, 3], world, N."""
    return as_numpy(view.get_contact_force_matrix(dt=dt)).reshape(-1, 3)


# -- hooks -------------------------------------------------------------------------------------------------------------

_RESET: list = []
_STEP: list = []
_SUB: list = []


def on_reset(fn):
    """``fn(env)`` after every reset, once the scene has settled."""
    _RESET.append(fn)
    return fn


def on_step(fn):
    """``fn(env)`` after every control step, before the task file is scored."""
    _STEP.append(fn)
    return fn


def on_substep(fn):
    """``fn(env, dt)`` after every physics step (the scene's update)."""
    _SUB.append(fn)
    return fn


def _put_back(env) -> None:
    """Every free body back to its start pose, all in the same write, and every joint re-armed."""
    import torch

    base = env.unwrapped
    for name in list(_PUSHED):
        base.scene[name].permanent_wrench_composer.reset()
    _PUSHED.clear()
    _ART_FORCES.clear()
    for name, spec in _SPECS.items():
        if spec.static:
            continue
        obj = base.scene[name]
        pose = obj.data.default_root_pose.torch.clone()
        pose[:, :3] += base.scene.env_origins
        obj.write_root_pose_to_sim_index(root_pose=pose)
        obj.write_root_velocity_to_sim_index(root_velocity=torch.zeros_like(obj.data.default_root_vel.torch))
    for name, spec in _SPECS.items():  # an articulation's links other than its root cannot be put back by a pose write:
        if spec.articulation_root:  # the root's pose (written above) and its joints at zero place them
            _reset_articulation(base, name)
    for j in _JOINTS.values():
        if j.articulated:
            continue
        if not j.rebuilt:  # breakable and remake joints are re-made at the reset instead
            set_joint(j.name, j.on)


def _reset_articulation(base, name: str) -> None:
    """The articulation rooted at body ``name``: root at its start pose, at rest, every joint at 0 and still."""
    import warp as wp

    view = _art_view(name)
    dev = str(base.device)
    idx = wp.array(np.arange(view.count, dtype=np.int32), device=dev)
    obj = base.scene[name]
    pose = obj.data.default_root_pose.torch.clone()
    pose[:, :3] += base.scene.env_origins
    root = np.ascontiguousarray(pose.detach().cpu().numpy().astype(np.float32).reshape(view.count, 7))
    view.set_root_transforms(wp.array(root.reshape(-1), dtype=wp.float32, device=dev), idx)
    view.set_root_velocities(wp.array(np.zeros(view.count * 6, np.float32), dtype=wp.float32, device=dev), idx)
    zeros = wp.array(np.zeros(view.count * view.max_dofs, np.float32), dtype=wp.float32, device=dev)
    view.set_dof_positions(zeros, idx)
    view.set_dof_velocities(zeros, idx)
    print(f"[yam_bodies] articulation '{name}' reset: {view.count} x {view.max_links} links, {view.max_dofs} joint(s) to 0", flush=True)


def _remake_breakable() -> None:
    """Every joint with a break force removed and authored afresh under a new prim name: a broken PhysX joint stays
    broken (switching it off and on does not mend it), so each episode starts with them whole (as mechkit's bonds)."""
    from isaaclab.sim.utils.stage import get_current_stage

    _GEN[1] += 1
    if _GEN[1] == 1:  # the first episode: as authored
        return
    stage = get_current_stage()
    for j in _JOINTS.values():
        if not j.rebuilt:
            continue
        stage.RemovePrim(_joint_path(j.name))
        _GEN[0] += 1
        path = f"{ENV}/{j.body1}/{j.name}_r{_GEN[0]}"
        _author_joint(stage, f"{ENV}/{j.body1}", _SPECS[j.body1], j, path)
        _JOINT_PATH[j.name] = path
        _JOINT_ATTR.pop(j.name, None)


def _wrap_update(env) -> None:
    scene = env.unwrapped.scene
    if getattr(scene, "_yam_bodies_sub", False) or not _SUB:
        return
    update = scene.update

    def update_and_sample(*a, **k):
        out = update(*a, **k)
        dt = float(k.get("dt", a[0] if a else env.unwrapped.physics_dt))
        for fn in _SUB:
            fn(env, dt)
        _flush_art_forces(env)
        return out

    scene.update = update_and_sample
    scene._yam_bodies_sub = True


def _patch_scene(scene) -> None:
    if getattr(scene, "_yam_bodies", False):
        return
    task_object, object_states = scene.task_object, scene.object_states

    def task_object_hook(spec):
        if isinstance(spec, Body):
            _SPECS[spec.name] = spec
            return _arena_object(spec)
        return task_object(spec)

    def object_states_hook(environment, names):
        plain = [n for n in names if n not in _SPECS]
        out = object_states(environment, plain) if plain else {}
        out.update({n: body_state(environment, n) for n in names if n in _SPECS})
        return {n: out[n] for n in names}

    scene.task_object, scene.object_states = task_object_hook, object_states_hook
    scene._yam_bodies = True


def _patch_embodiment(module) -> None:
    cls = module.YamArenaEmbodiment
    if getattr(cls, "_yam_bodies", False):
        return
    reset, task_step = cls.reset, cls._task_step

    def reset_hook(self, *a, **k):
        if self._env is not None:
            _put_back(self._env)
        obs = reset(self, *a, **k)
        _remake_breakable()
        for j in _JOINTS.values():  # rebuilt whole where the settled bodies are
            if j.articulated:
                continue
            if j.on and not j.rebuilt:
                set_joint(j.name, False)
                set_joint(j.name, True)
        _wrap_update(self._env)
        for fn in _RESET:
            fn(self._env)
        return obs

    def task_step_hook(self, *a, **k):
        for fn in _STEP:
            fn(self._env)
        return task_step(self, *a, **k)

    cls.reset, cls._task_step = reset_hook, task_step_hook
    cls._yam_bodies = True


_TARGETS = {"yam_arena.scene": _patch_scene, "yam_arena.inspect_embodiment": _patch_embodiment}


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname not in _TARGETS:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        run = spec.loader.exec_module

        def exec_module(module):
            run(module)
            _TARGETS[fullname](module)

        spec.loader.exec_module = exec_module
        return spec


def install() -> None:
    for name, patch in _TARGETS.items():
        if name in sys.modules:
            patch(sys.modules[name])
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())


install()


def quat_from_matrix(m: np.ndarray) -> tuple[float, float, float, float]:
    """(x, y, z, w) of a rotation matrix."""
    w = math.sqrt(max(0.0, 1.0 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1.0 + m[0, 0] - m[1, 1] - m[2, 2])) / 2, m[2, 1] - m[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1.0 - m[0, 0] + m[1, 1] - m[2, 2])) / 2, m[0, 2] - m[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1.0 - m[0, 0] - m[1, 1] + m[2, 2])) / 2, m[1, 0] - m[0, 1])
    n = math.sqrt(w * w + x * x + y * y + z * z)
    return (x / n, y / n, z / n, w / n)
