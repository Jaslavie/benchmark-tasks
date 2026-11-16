"""Native fixed action controls. Select CONTROL in a fresh process."""
import os
import task as original
from task import *
CONTROL=os.environ.get('CONTROL','zero_only')


def demo():
    paths=original.demo()
    if CONTROL in ('zero_only','centre_only'):
        # No full stroke. A single valid reading cannot certify parallelism.
        p=original._carriage(0.)
        paths={'right':[Waypoint(1.5,tuple(p+[0,0,.06]),O),Waypoint(1.,tuple(p),O),
                        Waypoint(.8,tuple(p),C),Waypoint(3.,tuple(p),C),
                        Waypoint(.8,tuple(p),O),Waypoint(2.,tuple(p+[0,0,.07]),O)]}
    elif CONTROL=='clamp_misaligned':
        # Stop after the initial full traverse. The start is genuinely clamped but crooked.
        paths={'right':paths['right'][:7]}
    elif CONTROL=='leave_unclamped':
        # Omit reclamping and try the final released sweep. No forged lock flag.
        # Indices are selected by the known fixed action sequence only.
        r=paths['right'];paths['right']=r[:20]+r[-7:]
    elif CONTROL=='hold_vise':
        left=paths['left']
        paths['left']=left[:-2]+[Waypoint(20.,tuple(original._jaw(0.)),C,yaw=math.pi/2)]
    else:raise ValueError('Unknown native control '+CONTROL)
    return paths
