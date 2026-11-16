"""Task-local USD articulation and evaluator hooks; Isaac imports are lazy.

Uses the same task-local scene/embodiment hook pattern as the upstream
dual-latch-lid partkit. Does not modify installed YAM source. No object poses or
joint positions are written after reset, and no software latch is implemented.
"""
from __future__ import annotations

from dataclasses import dataclass,replace
import importlib.abc
import importlib.util
import math
import os
from pathlib import Path
import sys
import json
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from detent_bottle.scoring import Scorer
from detent_bottle.support import table_support
from yam_arena.task_file import Box, ObjectState


@dataclass(frozen=True)
class Bottle(Box):
    """Free, four-link bottle; size is metadata, not a box collider."""


_SPEC = None
_SCORER = None
_TRACE = None
# Both wrists validated with distinct known loads on Brev, 2026-11-07:
# left10N ->9.999999N increment, right20N ->20.000000N, both recover.
# This validates units/channel mapping, not full dynamic sensor fidelity.
FORCE_SENSOR_VALIDATED = True
_FORCE_READY = False


def arr(value):
    return getattr(value, "torch", value).detach().cpu().numpy()


def current_score():
    # Missing live measurements block positive credit even after calibration.
    return _SCORER.score if FORCE_SENSOR_VALIDATED and _FORCE_READY and _SCORER is not None else 0.0


def spawn(prim_path, cfg, translation=None, orientation=None, **kwargs):
    from bottle_usd import spawn_bottle
    return spawn_bottle(prim_path, translation, orientation)


def arena_object(spec):
    from isaaclab.assets import ArticulationCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    @configclass
    class SpawnCfg(SpawnerCfg):
        func: object = clone(spawn)

    class BottleObject(Object):
        def _generate_articulation_cfg(self):
            # Passive joint spring laws. No policy controls these joints.
            actuators = {
                "lid": ImplicitActuatorCfg(joint_names_expr=["lid_hinge"], stiffness=.018, damping=.0045),
                "button": ImplicitActuatorCfg(joint_names_expr=["button_hinge"], stiffness=.10, damping=.0007),
                "loop": ImplicitActuatorCfg(joint_names_expr=["catch_hinge"], stiffness=0.0, damping=.0015),
            }
            cfg = ArticulationCfg(prim_path=self.prim_path, spawn=self.spawner_cfg,
                actuators=actuators, init_state=ArticulationCfg.InitialStateCfg(
                    joint_pos={".*": 0.0}, joint_vel={".*": 0.0}))
            return self._add_initial_pose_to_cfg(cfg)

    return BottleObject(name=spec.name, object_type=ObjectType.ARTICULATION,
        spawner_cfg=SpawnCfg(), initial_pose=Pose(position_xyz=spec.pos,
        rotation_xyzw=(0., 0., math.sin(spec.yaw/2), math.cos(spec.yaw/2))))


def configure_passive(base):
    """Reapply passive targets after scene reset clears them."""
    import torch
    bottle = base.scene["bottle"]
    targets = torch.zeros_like(getattr(bottle.data.joint_pos, "torch", bottle.data.joint_pos))
    targets[:, list(bottle.joint_names).index("lid_hinge")] = 2.1
    bottle.set_joint_position_target(targets)


def physics_callback(base, dt):
    """Effective bistable detent, evaluated on every physics substep.

    Gravity compensation follows the loop COM (0,0,.013), mass .006 kg and
    parent attitude. Dry joint frictionloss is not yet modeled in PhysX.
    """
    import torch
    configure_passive(base)
    bottle = base.scene["bottle"]
    q = getattr(bottle.data.joint_pos, "torch", bottle.data.joint_pos)
    v = getattr(bottle.data.joint_vel, "torch", bottle.data.joint_vel)
    i = list(bottle.joint_names).index("catch_hinge")
    th = q[:, i]
    # Spring restoring torques only, never a target chosen from task progress.
    torque = torch.where(th < math.pi/6, -.018-.010*th-.018*v[:, i],
        torch.where(th > 5*math.pi/6, .018+.010*(math.pi-th)-.018*v[:, i],
                    -.002*v[:, i]))
    pose=getattr(bottle.data.root_link_pose_w,"torch",bottle.data.root_link_pose_w)
    x,y,z,w=pose[:,3],pose[:,4],pose[:,5],pose[:,6]
    gravity_y=-.006*9.81*2*(y*z+w*x)
    gravity_z=-.006*9.81*(1-2*(x*x+y*y))
    torque -= (-.013*torch.sin(th))*gravity_z-(.013*torch.cos(th))*gravity_y
    efforts = torch.zeros_like(q)
    efforts[:, i] = torque
    bottle.set_joint_effort_target(efforts)


def measure(emb,step=None):
    bottle = emb._base.scene["bottle"]
    pose = arr(bottle.data.root_link_pose_w)[0]
    q = arr(bottle.data.joint_pos)[0]
    velocity=arr(bottle.data.joint_vel)[0]
    names = list(bottle.joint_names)
    x,y,z,w = pose[3:7]
    z0 = float(arr(emb._base.scene.env_origins)[0, 2])
    # Keep native scorer's reference height, accounting for station table height.
    z_native = float(pose[2]-z0 + .8000 - .7475)
    forces={}
    finite_robot_state=True
    for side in ("left","right"):
        robot=emb._base.scene[side+"_robot"]
        finite_robot_state=finite_robot_state and bool(np.isfinite(
            np.r_[arr(robot.data.joint_pos).ravel(),arr(robot.data.joint_vel).ravel()]).all())
        key=f"{side}_bottle_wrench"
        if key in emb._base.scene.sensors:
            sensor=emb._base.scene[key]
            force=arr(sensor.data.force)[0,sensor.body_names.index("gripper")]
            forces[side]=float(np.linalg.norm(force))
    support=table_support(pose[:3]-arr(emb._base.scene.env_origins)[0],float(1-2*(x*x+y*y)))
    step=_SCORER.steps+1 if step is None else step
    row = dict(step=step,time_s=step/20,
        bottle_world_xyz=pose[:3].astype(float).tolist(),
        bottle_quat_xyzw=pose[3:7].astype(float).tolist(),
        lid_deg=math.degrees(float(q[names.index("lid_hinge")])),
        loop_deg=math.degrees(float(q[names.index("catch_hinge")])),
        button_deg=math.degrees(float(q[names.index("button_hinge")])),
        upright_dot=float(1-2*(x*x+y*y)),bottle_z_m=z_native,
        wrist_force_n=max(forces.values(),default=0.0),wrist_forces_n=forces,
        wrist_force_measured=len(forces)==2 and all(math.isfinite(f) for f in forces.values()),
        numerical_warning=not bool(np.isfinite(np.r_[pose,q,velocity]).all())
            or not finite_robot_state or not all(math.isfinite(f) for f in forces.values()))
    row.update(support)
    for key in ("lid_deg","loop_deg","button_deg","upright_dot","bottle_z_m"):
        if not math.isfinite(row[key]):
            row[key]=0.0
            row["numerical_warning"]=True
    if not math.isfinite(row["wrist_force_n"]):
        row["wrist_force_n"]=0.0
        row["numerical_warning"]=True
    return row


def patch_scene(scene):
    if getattr(scene,"_bottle_patched",False): return
    old_object,old_states,old_build = scene.task_object,scene.object_states,scene.build_task_environment
    def task_object(spec):
        global _SPEC
        if isinstance(spec,Bottle):
            _SPEC=spec
            return arena_object(spec)
        return old_object(spec)
    def states(env,names):
        if _SPEC is None or _SPEC.name not in names: return old_states(env,names)
        base=env.unwrapped; asset=base.scene[_SPEC.name]
        p=arr(asset.data.root_link_pose_w)[0]; pos=p[:3]-arr(base.scene.env_origins)[0]
        vel=arr(asset.data.root_vel_w)[0]
        out=old_states(env,[n for n in names if n!=_SPEC.name]) if len(names)>1 else {}
        out[_SPEC.name]=ObjectState(pos=pos,quat_xyzw=p[3:7],lin_vel=vel[:3],ang_vel=vel[3:],
            low=pos-np.array([.05,.06,0]),high=pos+np.array([.05,.06,.282]))
        return out
    def build(embodiment,task,control_hz=None):
        env=old_build(embodiment,task,control_hz)
        if any(isinstance(o,Bottle) for o in task.objects):
            previous=env.env_cfg_callback
            def configure(cfg):
                cfg=previous(cfg)
                cfg.sim.dt=.001
                cfg.decimation=50
                cfg.sim.render_interval=50
                from isaaclab.sensors import JointWrenchSensorCfg
                cfg.scene.left_bottle_wrench=JointWrenchSensorCfg(prim_path="{ENV_REGEX_NS}/LeftArm")
                cfg.scene.right_bottle_wrench=JointWrenchSensorCfg(prim_path="{ENV_REGEX_NS}/RightArm")
                return cfg
            env.env_cfg_callback=configure
        return env
    scene.task_object,scene.object_states,scene.build_task_environment=task_object,states,build
    scene._bottle_patched=True


def patch_embodiment(module):
    cls=module.YamArenaEmbodiment
    if getattr(cls,"_bottle_patched",False): return
    old_reset,old_task=cls.reset,cls._task_step
    def active(self):
        return self.task is not None and any(isinstance(o,Bottle) for o in self.task.objects)
    def reset(self,*args,**kwargs):
        global _SCORER,_TRACE,_FORCE_READY
        if not active(self): return old_reset(self,*args,**kwargs)
        if not math.isclose(float(self.info.control_hz),20.0):
            raise ValueError("The bottle scorer requires20Hz: use -E control_hz=20")
        self._ensure_sim(None)
        bottle=self._base.scene["bottle"]
        if not getattr(bottle,"_bottle_callback_installed",False):
            previous_write=bottle.write_data_to_sim
            def write_data(*a,**k):
                physics_callback(self._base,self._base.physics_dt)
                return previous_write(*a,**k)
            bottle.write_data_to_sim=write_data
            bottle._bottle_callback_installed=True
        obs=old_reset(self,*args,**kwargs)
        _SCORER=Scorer()
        _FORCE_READY=False
        configure_passive(self._base)
        trace=os.environ.get("BOTTLE_TRACE")
        if _TRACE is not None: _TRACE.close()
        _TRACE=open(trace,"w",buffering=1) if trace else None
        return obs
    def task_step(self):
        global _FORCE_READY
        if not active(self): return old_task(self)
        row=measure(self)
        row.setdefault("bottle_on_table",False)
        _FORCE_READY=row.get("wrist_force_measured",False)
        if FORCE_SENSOR_VALIDATED and not _FORCE_READY:
            _SCORER.failure=_SCORER.failure or "wrist_force_unavailable"
        _SCORER.update(row)
        if _TRACE is not None: _TRACE.write(json.dumps(row)+"\n")
        result=old_task(self)
        # Evaluator-only joint reactions leave the stock policy observations intact.
        info=dict(result.info,force_validation_pending=not FORCE_SENSOR_VALIDATED,
                  geometry_score_diagnostic=_SCORER.score)
        if _SCORER.failure:
            return replace(result,terminated=True,termination_reason=_SCORER.failure,info=info)
        return replace(result,info=info)
    cls.reset,cls._task_step=reset,task_step
    cls._bottle_patched=True


class Finder(importlib.abc.MetaPathFinder):
    targets={"yam_arena.scene":patch_scene,"yam_arena.inspect_embodiment":patch_embodiment}
    def find_spec(self,fullname,path,target=None):
        if fullname not in self.targets: return None
        sys.meta_path.remove(self)
        try: spec=importlib.util.find_spec(fullname)
        finally: sys.meta_path.insert(0,self)
        if spec is None or spec.loader is None: return None
        execute=spec.loader.exec_module; patch=self.targets[fullname]
        def exec_module(module):
            execute(module);patch(module)
        spec.loader.exec_module=exec_module
        return spec


def install():
    from motion import patch_scripted
    Finder.targets["yam_arena.scripted"]=patch_scripted
    for name,patch in Finder.targets.items():
        if name in sys.modules: patch(sys.modules[name])
    if not any(isinstance(f,Finder) for f in sys.meta_path): sys.meta_path.insert(0,Finder())
