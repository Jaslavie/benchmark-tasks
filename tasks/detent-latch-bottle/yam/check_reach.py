"""Offline URDF IK checks before spending GPU time on a manipulation plan."""
import json
import numpy as np
from yam_arena import ik
from yam_arena.station import BASES

rotations={"down":np.diag([1.,-1.,-1.]),
    "front":np.array([[0,0,1],[1,0,0],[0,1,0.]]),
    "side":np.array([[1,0,0],[0,0,1],[0,-1,0.]])}
for arm in ["right","left"]:
    for name,R in rotations.items():
        target=[.20,-.0465,1.015]
        q,p,r=ik.solve(target,ik.matrix_to_quat_xyzw(R),base_pos=BASES[arm],iters=200)
        print(json.dumps(dict(arm=arm,orientation=name,target=target,q=q.tolist(),
            error_mm=p*1000,error_deg=float(np.degrees(r)))),flush=True)

from motion import plan
from yam_arena.constants import READY
for arm,waypoints in plan().items():
    q=np.array(READY)
    for w in waypoints:
        q,p,r=ik.solve(w.pos,w.rotation,q,base_pos=BASES[arm],iters=200)
        print(json.dumps(dict(arm=arm,label=w.label,error_mm=p*1000,
            error_deg=float(np.degrees(r)))),flush=True)
