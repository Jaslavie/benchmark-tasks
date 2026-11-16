"""Known-layout robot-only contact plan; direct PhysX trial reached full score."""
from dataclasses import dataclass
import math
import numpy as np
from yam_arena import ik
from yam_arena.constants import FINGER_OPEN
from yam_arena.demo import ArmPlan, Waypoint
from yam_arena.urdf import quat_xyzw_to_matrix


@dataclass
class PoseWaypoint(Waypoint):
    rotation: tuple = (0.,0.,0.,1.)


class PoseArmPlan(ArmPlan):
    def target(self,t):
        if not self.waypoints or not isinstance(self.waypoints[0],PoseWaypoint):
            return super().target(t)
        p,R,g=self.start_pos,self.start_rot,self.finger_start
        for w in self.waypoints:
            goal=np.asarray(w.pos);rotation=quat_xyzw_to_matrix(np.asarray(w.rotation))
            if t<w.duration:
                s=ik.smoothstep(t/w.duration)
                return p+s*(goal-p),ik.slerp_matrix(R,rotation,s),g+s*(w.finger-g),w.label
            t-=w.duration;p,R,g=goal,rotation,w.finger
        return p,R,g,"done"


def plan(top=.7475):
    def wp(seconds,pos,gap,roll,label,flip=False):
        # Finger joints are signed half-gaps; do not replace robot geometry.
        rotation=(0.,-math.sin(roll/2),math.cos(roll/2),0.) if flip else (math.sin(roll/2),0.,0.,math.cos(roll/2))
        return PoseWaypoint(seconds,pos,-gap/2,label,rotation=rotation)
    x=.20
    left=[
        wp(2.5,(x,.14,top+.14),.09,math.pi/2,"approach body from left",True),
        wp(2.5,(x,.013,top+.14),.09,math.pi/2,"fingers around body",True),
        wp(1.0,(x,.013,top+.14),.060,math.pi/2,"steady body",True),
        wp(24.,(x,.013,top+.14),.060,math.pi/2,"hold body",True),
    ]
    right=[
        wp(6.,(x,-.10,top+.268),.045,-math.pi/2,"wait above front"),
        wp(2.,(x,-.049,top+.266),.045,-math.pi/2,"approach safety loop"),
        wp(1.,(x,-.049,top+.266),.003,-math.pi/2,"pinch loop crown"),
    ]
    for degrees in (15,30,50,70,90,110,130,150,170):
        t=math.radians(degrees)
        # Place the loop inside the pads, not at their +13mm terminal edge.
        right.append(wp(.5,(x,-.0465-.0275*math.sin(t)-.0025,
                             top+.240+.0275*math.cos(t)-.0015),.003,-math.pi/2,
                        f"lower loop {degrees}"))
    right.extend([
        wp(.8,(x,-.060,top+.212),.05,-math.pi/2,"release loop"),
        wp(1.5,(x,-.10,top+.228),.05,-math.pi/2,"clear loop"),
        wp(1.,(x,-.10,top+.228),0.,-math.pi/2,"close for press"),
        wp(2.,(x,-.067,top+.228),0.,-math.pi/2,"approach lower rocker"),
        wp(1.5,(x,-.054,top+.228),0.,-math.pi/2,"press lower rocker"),
        wp(2.,(x,-.054,top+.228),0.,-math.pi/2,"hold press"),
        wp(1.5,(x,-.10,top+.228),.05,-math.pi/2,"withdraw"),
    ])
    return {"left":left,"right":right}


def patch_scripted(module):
    # Task-local extension for arbitrary tool orientations, keeping the official
    # 14-D joint action and the original behavior of ordinary YAM waypoints.
    module.ArmPlan=PoseArmPlan
