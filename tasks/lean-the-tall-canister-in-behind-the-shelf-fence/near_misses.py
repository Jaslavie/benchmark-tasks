"""Near-successes and unintended strategies, scored offline by the factory checker (factory/check.py), in lot A.
Each case: (label, {object: ready state}, expected score, expected success); a state comes from the canister's own
geometry (CANISTER.state(...)). Offline there is no robot, so every case counts as let go; held cases are covered
by the on_step release check in the simulator.
"""

import math

import numpy as np

from task import BACK, BY, CANISTER, DZ, FENCE_H, FI, FX0, H, HALF_W, SIDE, TOP

HS = SIDE / 2


def _quat(axis, deg):
    a = math.radians(deg) / 2
    v = np.asarray(axis, float) * math.sin(a)
    return (float(v[0]), float(v[1]), float(v[2]), math.cos(a))


def lean_back(deg, near_x, y=BY, lift=0.0):
    """Leaning ``deg`` with its top toward the robot, its near bottom edge at x ``near_x`` on the shelf."""
    t = math.radians(deg)
    # Rotation about -y tips the top toward -x; the near (-x) bottom edge is lowest.
    cx = near_x + HS * math.cos(t)
    cz = DZ + HS * math.sin(t) + lift
    return {"canister": CANISTER.state(pos=(cx, y, cz), quat_xyzw=_quat((0, -1, 0), deg))}


def upright(near_x, y=BY, z=DZ, vel=None):
    return {"canister": CANISTER.state(pos=(near_x + HS, y, z), lin_vel=vel)}


def sideways(deg, near_x):
    """Tilted ``deg`` toward the robot's right (about x), its right bottom edge on the shelf."""
    t = math.radians(deg)
    return {"canister": CANISTER.state(pos=(near_x + HS, BY + HS * (1 - math.cos(t)), DZ + HS * math.sin(t)),
                                       quat_xyzw=_quat((1, 0, 0), deg))}


def lying(near_x):
    """On its side behind the fence, its base facing the back wall."""
    return {"canister": CANISTER.state(pos=(near_x + H, BY, DZ + HS), quat_xyzw=_quat((0, 1, 0), 90.0))}


SET = FI + 0.0144  # where the demo sets the near bottom edge

CASES = [
    ("upright behind the fence where the demo sets it", upright(SET), 1.0, True),
    ("upright, base touching the fence's inner face", upright(FI), 1.0, True),
    ("upright, base 1 mm into the fence (on its top edge)", upright(FI - 0.001), 0.0, False),
    ("upright against the back wall", upright(BACK - SIDE), 1.0, True),
    ("upright, base at the left wall", upright(SET, y=BY + HALF_W - HS), 1.0, True),
    ("upright, held 3 mm above the shelf", upright(SET, z=DZ + 0.003), 0.0, False),
    ("upright, base 2 mm above the shelf (settling)", upright(SET, z=DZ + 0.002), 1.0, True),
    ("upright, still sliding", upright(SET, vel=(0.05, 0.0, 0.0)), 0.0, False),
    ("leaning back 5 degrees", lean_back(5.0, SET), 1.0, True),
    ("leaning back 6 degrees", lean_back(6.0, SET), 0.5, False),
    ("leaning back on the fence, 20 degrees", lean_back(20.0, FI + FENCE_H * math.tan(math.radians(20.0))), 0.5,
     False),
    ("leaning back 30 degrees", lean_back(30.0, FI + 0.025), 0.5, False),
    ("leaning back 31 degrees", lean_back(31.0, FI + 0.025), 0.0, False),
    ("leaning, its near edge 1 mm short of the fence's inner face", lean_back(20.0, FI - 0.001), 0.0, False),
    ("leaning, held with its near edge 3 mm up", lean_back(20.0, SET, lift=0.003), 0.0, False),
    ("tilted 4 degrees sideways", sideways(4.0, SET), 1.0, True),
    ("tilted 8 degrees sideways (not back onto the fence)", sideways(8.0, SET), 0.0, False),
    ("leaning 20 degrees forward, top toward the back wall",
     {"canister": CANISTER.state(pos=(SET + HS * math.cos(math.radians(20.0)), BY, DZ + HS * math.sin(math.radians(20.0))),
                                 quat_xyzw=_quat((0, 1, 0), 20.0))}, 0.0, False),
    ("lying on its side behind the fence", lying(FI + 0.005), 0.0, False),
    ("standing on top of the fence", upright(FX0, z=DZ + FENCE_H), 0.0, False),
    ("upright on the table in front of the shelf", upright(FX0 - 0.080, z=TOP), 0.0, False),
    ("lying on the table in front of the shelf",
     {"canister": CANISTER.state(pos=(0.20, BY, TOP + HS), quat_xyzw=_quat((0, 1, 0), 90.0))}, 0.0, False),
    ("at its start", {}, 0.0, False),
]
