"""Original small milling vise and mechanical indicator comparator. SI units."""
import math
import numpy as np

TOP=.7475
ORIGIN=(.250,0.,TOP)
INITIAL_YAW=math.radians(4.)
CAM_OPEN=math.radians(100.)
CAM_CLOSED=math.radians(-55.)
CAM_PIVOT=(.032,-.031,.03225)
FOLLOWER_HOME=(.032,-.031,.0235)
PAD_HOME=(.032,-.031,.01775)
ECCENTRIC=(-.0015,0.,-.0015)
CAM_RADIUS=.009
FORK_THICK=.003
PAD_K=3000.
ROTOR_Z=.012
PROBE_HOME=(-.050,0.,.035)
TRAVERSE_START=-.035
TRAVERSE_END=.035
Q_MIN=-.0065
Q_MAX=.001
POINTER_CENTRE=(-.077,0.,.065)


def box(pos,size,color=(.5,.5,.5),friction=.4,**extra):
    return dict(pos=tuple(map(float,pos)),size=tuple(map(float,size)),color=color,
                friction=friction,contact_offset=.00005,**extra)


def cylinder(pos,radius,height,axis='Z',color=(.5,.5,.5),**extra):
    size=[2*radius]*3
    size[{'X':0,'Y':1,'Z':2}[axis]]=height
    return box(pos,size,color,shape='cylinder',radius=radius,height=height,axis=axis,**extra)


def ry(q):
    c,s=math.cos(q),math.sin(q)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])


def rz(q):
    c,s=math.cos(q),math.sin(q)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])


def quat(axis,q):
    v=[0.,0.,0.];v[axis]=math.sin(q/2)
    return (*v,math.cos(q/2))


def cam_bottom(q):
    return CAM_PIVOT[2]+float((ry(q)@ECCENTRIC)[2])-CAM_RADIUS


def cam_knob(q):
    return np.array(ORIGIN)+np.array(CAM_PIVOT)+ry(q)@np.array([0.,0.,.060])


def follower_position(q):
    p=np.array(FOLLOWER_HOME)
    p[2]=cam_bottom(q)-FORK_THICK/2
    return tuple(map(float,p))


def expected_clamp_compression(q):
    follower_z=follower_position(q)[2]
    pad_z=.0172 # 2 mm thick pad seated on the 16.2 mm rotor contact island.
    return max(0.,pad_z-follower_z+.00575)


BASE_PARTS=(box((0,0,.004),(.14,.11,.008),(.35,.38,.42),friction=.25),)
ROTOR_PARTS=(
    box((0,0,0),(.100,.076,.008),(.4,.45,.5),friction=.25),
    # Leave the radial clamping land clear of the raised jaw support.
    box((0,0,.009),(.080,.028,.010),(.4,.45,.5)),
    box((-.025,0,.023),(.008,.080,.026),(.65,.65,.65)),
    box((.020,0,.023),(.008,.080,.026),(.65,.65,.65)),
    cylinder((.043,0,.020),.004,.050,axis='X',color=(.65,.65,.65)),
    # The tommy bar through the vise screw gives the operator an accessible grip.
    cylinder((.062,0,.028),.003,.044,axis='Z',color=(.54,.55,.58)),
    cylinder((.062,0,.050),.006,.009,axis='Z',color=(.28,.29,.31)),
    box((.032,-.031,.0041),(.020,.014,.0002),(.3,.3,.3),friction=.8),
)
CAM_PARTS=(
    cylinder(ECCENTRIC,CAM_RADIUS,.032,axis='Y',color=(.6,.6,.65),friction=.15),
    box((0,0,.032),(.008,.008,.056),(.6,.6,.65)),
    box((0,0,.060),(.016,.018,.014),(.18,.38,.65)),
)
FOLLOWER_PARTS=tuple(box((0,y,0),(.018,.005,FORK_THICK),(.6,.6,.65),friction=.2)
                     for y in (-.012,.012))+(box((.018,0,.002),(.003,.006,.002),(.2,.3,.6)),)
PAD_PARTS=(box((0,0,0),(.018,.014,.002),(.2,.2,.2),friction=.8),
           box((.011,0,0),(.008,.003,.002),(.6,.6,.6)),
           box((.015,0,.008),(.0015,.0015,.016),(.75,.15,.1)))
RAIL_PARTS=(box((-.100,0,.009),(.050,.120,.018),(.35,.38,.42)),)
CARRIAGE_PARTS=[
    box((-.100,0,.022),(.055,.028,.008),(.45,.45,.5)),
    box((-.115,0,.034),(.012,.014,.018),(.18,.38,.65)),
    box((-.077,0,.043),(.012,.010,.040),(.6,.6,.65)),
    cylinder((-.077,0,.060),.021,.006,color=(.8,.8,.8)),
    cylinder((-.077,0,.0632),.0195,.0004,color=(.95,.95,.92),visual_only=True),
    box((-.058,0,.035),(.022,.008,.008),(.65,.65,.65)),
]
for tick in range(61):
    angle=math.radians(-150+tick*5.)
    radius=.0175
    CARRIAGE_PARTS.append(box((-.077+radius*math.cos(angle),radius*math.sin(angle),.0636),
        (.0025 if tick%10==0 else .0012,.0004,.0002),(.05,.05,.05),
        visual_only=True,rot_z=angle))
_digits={'0':'abcedf','1':'bc','2':'abdeg','3':'abcdg','4':'bcfg','5':'acdfg','6':'acdefg'}
_seg={'a':(0,.0012,.0014,.00025),'g':(0,0,.0014,.00025),'d':(0,-.0012,.0014,.00025),
      'b':(.0007,.0006,.00025,.0012),'c':(.0007,-.0006,.00025,.0012),
      'e':(-.0007,-.0006,.00025,.0012),'f':(-.0007,.0006,.00025,.0012)}
for value in range(7):
    angle=math.radians(-150+50*value)
    cx=-.077+.012*math.cos(angle);cy=.012*math.sin(angle)
    for name in _digits[str(value)]:
        x,y,w,h=_seg[name]
        CARRIAGE_PARTS.append(box((cx+x,cy+y,.0638),(w,h,.0002),(.05,.05,.05),visual_only=True))
POINTER_INDEX=len(CARRIAGE_PARTS)
CARRIAGE_PARTS.append(box((-.070,0,.065),(.014,.0007,.0003),(.75,.1,.1),
                            visual_only=True,rot_z=0.))
CARRIAGE_PARTS=tuple(CARRIAGE_PARTS)
PROBE_PARTS=(
    cylinder((.010,0,0),.0015,.020,axis='X',color=(.65,.65,.65),friction=.05),
    box((.022,0,0),(.004,.004,.004),(.85,.85,.85),shape='sphere',radius=.002,friction=.05),
)


def display_angle(reading_m):
    return math.radians(-150+50*reading_m*1000)


def predicted_probe_q(y,yaw):
    # The unrotated jaw face is x=-29 mm. The 2 mm sphere touches
    # when its centre lies one radius short of the rotated face.
    x=(-.029-.002-math.sin(yaw)*y)/math.cos(yaw)
    return x-(-.028)
