"""Standalone native controller. No console service or object-action interface."""
import math
import numpy as np
from .scene import scene_xml
from .scoring import Scorer

DETENT=(.018,.01,.018,.002)

class BottleEnv:
    def __init__(self, *, max_steps=2400):
        self.max_steps = max_steps
        import mujoco
        from scipy.spatial.transform import Rotation
        self.mj=mujoco;self.Rotation=Rotation
        self.xml=scene_xml();self.model=mujoco.MjModel.from_xml_string(self.xml)
        self.data=mujoco.MjData(self.model);self.ik=mujoco.MjData(self.model)
        self.ids=[[self.model.joint(f'{s}_joint{i}').id for i in range(1,8)] for s in ('left','right')]
        self.qids=[self.model.jnt_qposadr[j] for j in self.ids]
        self.dids=[self.model.jnt_dofadr[j] for j in self.ids]
        self.tips=[self.model.site(s+'_tip').id for s in ('left','right')]
        self.wrists=[self.model.site(s+'_wrist').id for s in ('left','right')]
        self.finger_qids = [[self.model.jnt_qposadr[self.model.joint(f'{s}_finger_{f}').id] for f in ('a', 'b')] for s in ('left', 'right')]
        for ids in self.finger_qids:self.data.qpos[ids]=.045
        catch=self.model.joint('catch_hinge').id;self.catch_q,self.catch_d=self.model.jnt_qposadr[catch],self.model.jnt_dofadr[catch]
        self.targets=np.array([[-.12,-.18,1.12],[.13,.16,1.12]])
        self.rotations=np.array([np.diag([1.,-1.,-1.])]*2)
        self.grip_targets=np.full(2,.045)
        self.qtargets=[]
        self.steps=0;self.success=False;self.reason=None;self.open_time=0.;self.overload_time=0.
        self.history=[];self.renderer=None
        for q in self.qids:self.data.qpos[q]=[0,-.45,0,-2.15,0,1.7,.7854]
        mujoco.mj_forward(self.model,self.data)
        for a in range(2):
            self.data.qpos[self.qids[a]]=self.solve_ik(a,self.targets[a],self.rotations[a],300)
            mujoco.mj_forward(self.model,self.data)
        self.qtargets=[self.data.qpos[q].copy() for q in self.qids]
        for _ in range(10):self.step(check=False)
        self.baseline=self.wrenches_world()
        self.data.time=0.;self.steps=0;self.history=[]
        self.start_bottle=self.data.body('bottle').xpos.copy()
        self.scorer = Scorer()

    def solve_ik(self, arm, target, rotation, iterations=20, q0=None):
        mj=self.mj
        for qi in self.qids+self.finger_qids:self.ik.qpos[qi]=self.data.qpos[qi]
        if q0 is not None:self.ik.qpos[self.qids[arm]]=q0
        qi,di,site=self.qids[arm],self.dids[arm],self.tips[arm]
        jp=np.zeros((3,self.model.nv));jr=jp.copy()
        for _ in range(iterations):
            mj.mj_kinematics(self.model,self.ik);mj.mj_comPos(self.model,self.ik)
            rot=self.ik.site_xmat[site].reshape(3,3)
            err=np.r_[target-self.ik.site_xpos[site], .3*self.Rotation.from_matrix(rotation@rot.T).as_rotvec()]
            if np.linalg.norm(err)<1e-5:break
            mj.mj_jacSite(self.model,self.ik,jp,jr,site)
            J=np.vstack([jp[:,di],.3*jr[:,di]])
            dq=J.T@np.linalg.solve(J@J.T+np.eye(6)*.0001,err)
            self.ik.qpos[qi]+=np.clip(dq,-.08,.08)
            self.ik.qpos[qi]=np.clip(self.ik.qpos[qi],self.model.jnt_range[self.ids[arm],0]+.01,self.model.jnt_range[self.ids[arm],1]-.01)
        return self.ik.qpos[qi].copy()

    def target(self,arm,position,rpy,jaw):
        position=np.asarray(position,float);rpy=np.asarray(rpy,float)
        if position.shape!=(3,) or rpy.shape!=(3,) or not np.isfinite(np.r_[position,rpy,jaw]).all():
            raise ValueError('Expected finite XYZ, roll/pitch/yaw and jaw gap')
        if np.any(position < [-.30,-.40,.81]) or np.any(position > [.35,.40,1.40]):
            raise ValueError('Outside worktable tool workspace: X[-.30,.35], Y[-.40,.40], Z[.81,1.40] m')
        if not 0<=jaw<=.09 or np.any(abs(rpy)>180):raise ValueError('Jaw gap must be 0–0.09 m; angles within ±180 degrees')
        self.targets[arm]=position
        self.rotations[arm]=self.Rotation.from_euler('xyz',rpy,degrees=True).as_matrix()
        self.grip_targets[arm]=jaw/2

    def step(self,check=True):
        if self.reason:return
        mj=self.mj
        # Limit commanded Cartesian/rotational travel per 50 ms control tick.
        # Hold the other gripper at its previous target; no object pose writes.
        for a in range(2):
            current=self.data.site_xpos[self.tips[a]];delta=self.targets[a]-current
            waypoint=current+delta*min(1.,.008/max(np.linalg.norm(delta),1e-9))
            rot=self.data.site_xmat[self.tips[a]].reshape(3,3)
            dr=self.Rotation.from_matrix(self.rotations[a]@rot.T).as_rotvec()
            intermediate=self.Rotation.from_rotvec(dr*min(1.,.08/max(np.linalg.norm(dr),1e-9))).as_matrix()@rot
            self.qtargets[a]=self.solve_ik(a,waypoint,intermediate)
        for i in range(round(.05/self.model.opt.timestep)):
            for a in range(2):
                qi,di=self.qids[a],self.dids[a]
                tau=450*(self.qtargets[a]-self.data.qpos[qi])-40*self.data.qvel[di]+self.data.qfrc_bias[di]
                self.data.ctrl[a*7:(a+1)*7]=np.clip(tau,[-80]*5+[-12]*2,[80]*5+[12]*2)
            self.data.ctrl[14:]=np.repeat(self.grip_targets,2)
            self.data.qfrc_applied[self.catch_d]=self.detent_torque()
            mj.mj_step(self.model,self.data)
        mj.mj_forward(self.model,self.data);mj.mj_rnePostConstraint(self.model,self.data)
        self.steps+=1
        if check:
            row = self.measurements()
            self.scorer.update(row)
            self.success = self.scorer.score == 1.0
            if self.scorer.failure:
                self.reason = self.scorer.failure
            elif self.success:
                self.reason = 'success'
            elif self.steps >= self.max_steps:
                self.reason = 'time_limit'

    def measurements(self):
        lid = math.degrees(float(self.data.joint('lid_hinge').qpos[0]))
        loop = math.degrees(float(self.data.joint('catch_hinge').qpos[0]))
        row = dict(step=self.steps, time_s=self.steps/20,
                    lid_deg=lid, loop_deg=loop,
                    upright_dot=float(self.data.body('bottle').xmat.reshape(3,3)[2,2]),
                    bottle_z_m=float(self.data.body('bottle').xpos[2]),
                    wrist_force_n=max(float(np.linalg.norm(self.data.sensor(s+'_force').data)) for s in ('left','right')),
                    numerical_warning=bool(any(w.number for w in self.data.warning) or not np.isfinite(self.data.qpos).all()))
        # Keep a numerical failure record scoreable even if a sensor/angle is
        # already nonfinite. The warning permanently vetoes credit; a corrupt
        # external trace without this explicit flag is still rejected.
        for key in ('lid_deg', 'loop_deg', 'upright_dot', 'bottle_z_m', 'wrist_force_n'):
            if not math.isfinite(row[key]):
                row[key] = 0.0
                row['numerical_warning'] = True
        return row

    def action(self, values):
        values = np.asarray(values, dtype=float)
        if values.shape != (14,) or not np.isfinite(values).all():
            raise ValueError('Expected fourteen finite targets: left XYZ/RPY-degrees/gap, then right')
        for i in (0, 1):
            v = values[7*i:7*i+7]
            if np.any(v[:3] < [-.30,-.40,.81]) or np.any(v[:3] > [.35,.40,1.40]) or np.any(abs(v[3:6])>180) or not 0<=v[6]<=.09:
                raise ValueError('Target is outside the robot workspace, angle or jaw limits')
        for i in (0, 1):
            v = values[7*i:7*i+7]
            self.target(i, v[:3], v[3:6], v[6])
        self.step()

    def targets_vector(self):
        return np.concatenate([np.r_[self.targets[i],self.Rotation.from_matrix(self.rotations[i]).as_euler('xyz', degrees=True),2*self.grip_targets[i]] for i in (0,1)])

    def detent_torque(self):
        # Bias (gravity) compensated everywhere. Below 30 deg: drive to 0; above 150: drive to 180; between: damped hold.
        # F0 clears the 0.012 N m lid-lip ramp at ~8 deg and the 0.002 N m hinge frictionloss; C sets ~1 rad/s, overdamped.
        F0,K,C,CH=DETENT
        th,v=self.data.qpos[self.catch_q],self.data.qvel[self.catch_d];tau=self.data.qfrc_bias[self.catch_d]
        if th<math.radians(30):return tau-F0-K*th-C*v
        if th>math.radians(150):return tau+F0+K*(math.pi-th)-C*v
        return tau-CH*v

    def mechanism(self):
        return {name+'_deg':round(math.degrees(float(self.data.joint(joint).qpos[0])),2)
                for name,joint in [('lid','lid_hinge'),('loop','catch_hinge'),('button','button_hinge')]}

    def wrenches_world(self):
        result=[]
        for side,site in zip(('left','right'),self.wrists):
            R=self.data.site_xmat[site].reshape(3,3)
            result.append(np.r_[R@self.data.sensor(side+'_force').data,R@self.data.sensor(side+'_torque').data])
        return np.array(result)

    def close(self):
        if self.renderer:self.renderer.close()
