"""Square the vise fixed jaw to the traverse axis and secure its swivel base.

Confirm the alignment with the physical indicator. Leave both grippers clear.
The machine is unpowered. The cutting spindle and workpiece are not part of this trial.
"""
import math
import numpy as np
import design as d
import native_contacts
from scoring import Sample,TrammingScore
from yam_articulated import Articulated
from yam_arena.demo import Waypoint
from yam_arena.constants import FINGER_OPEN as O,FINGER_CLOSED as C,PAD_TIP_PAST_TCP as D,FINGER_JOINTS
from yam_arena.urdf import quat_xyzw_to_matrix

SCENE='station';TOP=d.TOP;SCORE_OVER='end';SUCCESS_HOLD_S=1.
follower_start=d.follower_position(d.CAM_CLOSED)
pad_start=(d.PAD_HOME[0],d.PAD_HOME[1],.0172)
VISE=Articulated('vise',pos=d.ORIGIN,size=(.14,.11,.11),mass=.4,
    parts=d.BASE_PARTS,fixed_base=True,self_collisions=True,
    initial_joints=(('swivel',d.INITIAL_YAW),('cam_axis',d.CAM_CLOSED),
                    ('follower_slide',follower_start[2]-d.FOLLOWER_HOME[2]),
                    ('pad_spring',d.expected_clamp_compression(d.CAM_CLOSED))),
    contact_pairs=(('cam','follower'),('pad','rotor'),('root','rotor')),
    links=(dict(name='bearing',pos=(0,0,d.ROTOR_Z),mass=.01,
                parts=(d.cylinder((0,0,0),.003,.002),)),
           dict(name='rotor',pos=(0,0,d.ROTOR_Z),quat_xyzw=d.quat(2,d.INITIAL_YAW),mass=.60,parts=d.ROTOR_PARTS),
           dict(name='cam',pos=d.CAM_PIVOT,quat_xyzw=d.quat(1,d.CAM_CLOSED),mass=.060,parts=d.CAM_PARTS),
           dict(name='follower',pos=follower_start,mass=.006,parts=d.FOLLOWER_PARTS),
           dict(name='pad',pos=pad_start,mass=.004,parts=d.PAD_PARTS)),
    # Axial float lets gravity load the thrust face while the cam is open.
    joints=(dict(name='bearing_float',body0='root',body1='bearing',type='prismatic',axis='Z',
                 anchor0=(0,0,d.ROTOR_Z),anchor1=(0,0,0),lower=-.0002,upper=.0006,
                 stiffness=0.,damping=0.,max_force=0.),
            dict(name='swivel',body0='bearing',body1='rotor',type='revolute',axis='Z',
                 anchor0=(0,0,0),anchor1=(0,0,0),lower=-8.,upper=8.,
                 stiffness=0.,damping=.00001,max_force=.001),
            dict(name='cam_axis',body0='root',body1='cam',type='revolute',axis='Y',
                 anchor0=d.CAM_PIVOT,anchor1=(0,0,0),lower=-55.,upper=100.,
                 stiffness=0.,damping=.00001,max_force=.001),
            dict(name='follower_slide',body0='root',body1='follower',type='prismatic',axis='Z',
                 anchor0=d.FOLLOWER_HOME,anchor1=(0,0,0),lower=-.006,upper=.001,
                 stiffness=20.,damping=.1,max_force=1.),
            dict(name='pad_spring',body0='follower',body1='pad',type='prismatic',axis='Z',
                 anchor0=(0,0,-.00575),anchor1=(0,0,0),lower=0.,upper=.006,
                 stiffness=d.PAD_K,damping=2.,max_force=30.)))

INDICATOR=Articulated('indicator',pos=d.ORIGIN,size=(.15,.12,.09),mass=.25,
    parts=d.RAIL_PARTS,fixed_base=True,self_collisions=True,
    initial_joints=(('traverse',d.TRAVERSE_START),
                    ('plunger',d.predicted_probe_q(d.TRAVERSE_START,d.INITIAL_YAW))),
    links=(dict(name='carriage',pos=(0,d.TRAVERSE_START,0),mass=.15,parts=d.CARRIAGE_PARTS),
           dict(name='probe',pos=(d.PROBE_HOME[0],d.TRAVERSE_START,d.PROBE_HOME[2]),
                mass=.006,parts=d.PROBE_PARTS)),
    joints=(dict(name='traverse',body0='root',body1='carriage',type='prismatic',axis='Y',
                 anchor0=(0,0,0),anchor1=(0,0,0),lower=-.04,upper=.04,
                 stiffness=0.,damping=.1,max_force=.01),
            dict(name='plunger',body0='carriage',body1='probe',type='prismatic',axis='X',
                 anchor0=d.PROBE_HOME,anchor1=(0,0,0),lower=d.Q_MIN,upper=d.Q_MAX,
                 stiffness=150.,damping=.15,max_force=1.5)))
OBJECTS=[VISE,INDICATOR]
_scorer=TrammingScore();_value=0.;_diagnostics={};_pointer=None


def reset_history():
    global _scorer,_value,_diagnostics,_pointer
    _scorer=TrammingScore();_value=0.;_diagnostics={};_pointer=None
    native_contacts.reset()


def _joint(art,name):
    return float(art.data.joint_pos.torch[0,art.joint_names.index(name)])


def _dial_visual(reading):
    """Only uncollidable display geometry changes. This cannot move a mechanism."""
    global _pointer
    from pxr import Gf,UsdGeom
    from isaaclab.sim.utils.stage import get_current_stage
    if _pointer is None:
        for prim in get_current_stage().Traverse():
            p=prim.GetPath().pathString
            if '/env_0/' in p and 'indicator' in p.split('/') and p.endswith('/carriage/p'+str(d.POINTER_INDEX)):
                ops=UsdGeom.Xformable(prim).GetOrderedXformOps()
                _pointer={o.GetOpName():o for o in ops};break
        if _pointer is None:raise RuntimeError('Physical indicator visual pointer missing')
    angle=d.display_angle(reading)
    centre=d.POINTER_CENTRE
    _pointer['xformOp:translate'].Set(Gf.Vec3d(centre[0]+.007*math.cos(angle),.007*math.sin(angle),centre[2]))
    _pointer['xformOp:orient'].Set(Gf.Quatf(math.cos(angle/2),Gf.Vec3f(0,0,math.sin(angle/2))))


def observe(states,t,base):
    global _value,_diagnostics
    native_contacts.bind(base)
    ready,contacts=native_contacts.status()
    if not ready:_value=0.;return
    vise=base.scene['vise'];indicator=base.scene['indicator']
    swivel_coordinate=_joint(vise,'swivel');compression=_joint(vise,'pad_spring')
    q=_joint(indicator,'plunger');traverse=_joint(indicator,'traverse')
    if not np.isfinite([q,traverse,swivel_coordinate,compression,_joint(vise,'cam_axis')]).all():
        raise RuntimeError('Vise or indicator has a nonfinite native joint position')
    origin=base.scene.env_origins[0].detach().cpu().numpy()
    rotor=vise.data.body_link_pose_w.torch[0,vise.body_names.index('rotor')].detach().cpu().numpy()
    jaw_rotation=quat_xyzw_to_matrix(rotor[3:])
    yaw=math.atan2(jaw_rotation[1,0],jaw_rotation[0,0])
    error=float(np.linalg.norm(rotor[:3]-origin-np.array(d.ORIGIN)-[0,0,d.ROTOR_Z]))
    opened=[]
    for name in ('left_robot','right_robot'):
        robot=base.scene[name];ids=robot.find_joints(FINGER_JOINTS,preserve_order=True)[0]
        opened.append(float(np.min(np.abs(robot.data.joint_pos.torch[0,ids].detach().cpu().numpy())))>.035)
    rotor_velocity=vise.data.body_link_vel_w.torch[0,vise.body_names.index('rotor')].detach().cpu().numpy()
    speed=float(np.linalg.norm(rotor_velocity[:3]))
    # Low centre speed plus bounded yaw stability must both be satisfied.
    speed=max(speed,float(np.linalg.norm(rotor_velocity[3:]))*.08)
    m=Sample(float(t),traverse,q,yaw,error,contacts['tip'],contacts['pad'],compression,
             opened[0] and not contacts['left'],all(opened) and not contacts['robot'],speed)
    _value=_scorer.update(m)
    _dial_visual(-q)
    _diagnostics=dict(score=_value,yaw_deg=math.degrees(yaw),swivel_coordinate_deg=math.degrees(swivel_coordinate),probe_mm=-q*1000,
        traverse_mm=traverse*1000,clamp_compression_mm=compression*1000,
        cam_deg=math.degrees(_joint(vise,'cam_axis')),tip_contact=contacts['tip'],
        pad_contact=contacts['pad'],adjustment_hand_clear=m.adjustment_hand_clear,
        both_open_clear=m.both_open_clear,measured_range_mm=None if _scorer.last_range is None else _scorer.last_range*1000,
        verified=_scorer.verified)


def score(states):return _value
def success(states):return _value==1.
def distance(states):return 1.-_value
def diagnostics():return _diagnostics


def _carriage(y):return np.array(d.ORIGIN)+[-.115,y,.034+D]
def _jaw(q):return np.array(d.ORIGIN)+[0,0,d.ROTOR_Z]+d.rz(q)@np.array([.062,0.,.050])+[0,0,.005]
def _cam(q):return d.cam_knob(q)+[0,0,D]


def demo():
    """Fixed layout actions only. No simulator states or measured dial values are read."""
    right=[];left=[]
    def add(path,dt,pos,finger,yaw=0.):path.append(Waypoint(float(dt),tuple(map(float,pos)),finger,yaw=float(yaw)))
    def wait(path,time,pos,finger,yaw=0.):
        duration=time-sum(p.duration for p in path)
        if duration>1e-6:add(path,duration,pos,finger,yaw)
    start=_carriage(d.TRAVERSE_START);end=_carriage(d.TRAVERSE_END)
    add(right,1.5,start+[0,0,.06],O);add(right,1.,start,O);add(right,.8,start,C)
    add(right,3.,end,C);add(right,.5,end,C);add(right,.8,end,O);add(right,1.2,end+[0,0,.06],O)
    closed=_cam(d.CAM_CLOSED);opened=_cam(d.CAM_OPEN)
    add(right,1.2,closed+[0,0,.05],O);add(right,1.,closed,O);add(right,.8,closed,C)
    jaw=_jaw(d.INITIAL_YAW);jaw_yaw=d.INITIAL_YAW
    left_park=jaw+[0,.13,.12]
    right_park=np.array(d.ORIGIN)+[-.09,-.19,.18]
    add(left,2.,left_park,O,jaw_yaw)
    for angle in (-40,-20,0,20,40,60,80,100):add(right,.6,_cam(math.radians(angle)),C)
    add(right,.8,opened,O);add(right,1.2,opened+[0,0,.05],O)
    add(right,1.4,right_park,O)
    open_end=sum(p.duration for p in right)
    wait(left,open_end,left_park,O,jaw_yaw)
    add(left,1.2,jaw+[0,0,.05],O,jaw_yaw);add(left,1.,jaw,O,jaw_yaw);add(left,.8,jaw,C,jaw_yaw)
    for angle in (3,2,1,0,-1.10):add(left,.8,_jaw(math.radians(angle)),C,math.radians(angle))
    release_yaw=math.radians(-1.10)
    add(left,.8,_jaw(release_yaw),O,release_yaw);add(left,1.2,_jaw(release_yaw)+[0,0,.08],O,release_yaw)
    add(left,1.4,left_park,O,jaw_yaw)
    align_end=sum(p.duration for p in left)
    wait(right,align_end,right_park,O)
    add(right,1.2,opened+[0,0,.05],O)
    add(right,1.,opened,O);add(right,.8,opened,C)
    for angle in (80,60,40,20,0,-20,-40,-55):add(right,.6,_cam(math.radians(angle)),C)
    add(right,.8,closed,O);add(right,1.2,closed+[0,0,.05],O)
    add(right,1.5,end+[0,0,.06],O);add(right,1.,end,O);add(right,.8,end,C)
    add(right,3.,start,C);add(right,.5,start,C);add(right,3.,end,C);add(right,.8,end,O);add(right,1.2,end+[0,0,.07],O)
    add(right,2.,end+[0,0,.07],O)
    return dict(left=left,right=right)
