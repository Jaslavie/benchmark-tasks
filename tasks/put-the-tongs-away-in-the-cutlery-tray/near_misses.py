"""Near-successes and unintended strategies, scored offline by the factory checker (factory/check.py) (hinge
end to the robot's right). Each case: (label, {object: ready state}, expected score, expected success); a state comes
from the tongs' own geometry (TONGS.state(...)). Offline there is no robot, so every case counts as let go.
"""

import math

from task import (FLOOR_Z, LONG_W, LONG_X0, LONG_Y, OPEN, SHUT, TAB_LOCKED, TAB_TRAVEL, TONGS, TONGS_A, TONGS_MID,
                  TOP, YAW_A)

CX = LONG_X0 + LONG_W / 2  # the long compartment's centre line (front to back)


def quat(yaw=YAW_A, tilt_x=0.0):
    """Yaw about the vertical, then a tilt (rad) about the world x axis (front to back)."""
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    cx, sx = math.cos(tilt_x / 2), math.sin(tilt_x / 2)
    # q = q_tilt * q_yaw (xyzw)
    return (sx * cy, -sx * sy, cx * sy, cx * cy)


def tongs(x=CX, y=None, z=FLOOR_Z, arms=0.0, tab=TAB_TRAVEL, q=None, vel=None):
    """The tongs with their body origin at (x, y, z); by default centred in the long compartment, locked shut."""
    y = -TONGS_MID if y is None else y  # lot A: the tongs point to +y, so the middle is TONGS_MID along +y
    return {"tongs": TONGS.state(joints={"arm_l": arms, "arm_r": arms, "tab": tab}, pos=(x, y, z),
                                 quat_xyzw=quat() if q is None else q, lin_vel=vel)}


def turned(deg):
    """Locked, its middle on the compartment's centre, turned `deg` degrees about the vertical."""
    yaw = YAW_A + math.radians(deg)
    return tongs(x=CX - TONGS_MID * math.cos(yaw), y=-TONGS_MID * math.sin(yaw), q=quat(yaw))


OPEN_AT_START = {"tongs": TONGS.state(joints={"arm_l": OPEN, "arm_r": OPEN}, pos=TONGS_A, quat_xyzw=quat())}
LOCKED_ON_COUNTER = {"tongs": TONGS.state(joints={"arm_l": 0.0, "arm_r": 0.0, "tab": TAB_TRAVEL}, pos=TONGS_A,
                                          quat_xyzw=quat())}
WALL_SPRUNG = 0.105  # rad: each arm open until its head presses on a compartment wall (65 mm inside)

CASES = [
    ("locked, lying in the long compartment", tongs(), 1.0, True),
    ("locked, 3 mm above the floor (settling)", tongs(z=FLOOR_Z + 0.003), 1.0, True),
    ("locked, 4 mm above the floor: not resting there (a held state; offline nothing holds it)",
     tongs(z=FLOOR_Z + 0.004), 0.5, False),
    ("locked, in the compartment, still sliding", tongs(vel=(0.0, 0.02, 0.0)), 0.0, False),
    ("locked, pushed to the left end of the compartment", tongs(y=LONG_Y - 0.220 - 0.0005), 1.0, True),
    ("locked, 2 mm past the left end: lying on the end wall", tongs(y=LONG_Y - 0.220 + 0.002), 0.5, False),
    ("locked, turned 8 degrees about its middle in the compartment", turned(8), 1.0, True),
    ("locked, turned 12 degrees about its middle: an end up against a wall", turned(12), 0.5, False),
    ("locked, one end up on the rim (tilted 12 degrees)", tongs(z=FLOOR_Z + 0.02, q=quat(tilt_x=math.radians(12))),
     0.5, False),
    ("locked, lying across the tray's top", tongs(z=TOP + 0.050), 0.5, False),
    ("tab exactly 7 mm out, arms exactly 0.03 rad from shut: locked", tongs(arms=SHUT, tab=TAB_LOCKED), 1.0, True),
    ("tab 6.9 mm out: not locked, released inside", tongs(tab=TAB_LOCKED - 0.0001), 0.25, False),
    ("arms 0.031 rad open with the tab out: not locked", tongs(arms=SHUT + 0.001), 0.25, False),
    ("squeezed in and let go, sprung open against the walls, tab in", tongs(arms=WALL_SPRUNG, tab=0.0), 0.25, False),
    ("sprung open, tab in, still sliding", tongs(arms=WALL_SPRUNG, tab=0.0, vel=(0.0, 0.0, -0.03)), 0.0, False),
    ("open tongs set on the compartment: heads up on the walls", tongs(z=FLOOR_Z + 0.046, arms=OPEN, tab=0.0), 0.0,
     False),
    ("open tongs at their start on the counter", OPEN_AT_START, 0.0, False),
    ("tongs locked and left on the counter", LOCKED_ON_COUNTER, 0.5, False),
    ("tongs locked but dropped off the counter", tongs(x=0.2, y=-0.6, z=TOP - 0.2), 0.0, False),
    ("open tongs dropped off the counter", tongs(x=0.2, y=-0.6, z=TOP - 0.2, arms=OPEN, tab=0.0), 0.0, False),
    ("tongs on the counter, tab pulled but not shut (it would spring back)",
     {"tongs": TONGS.state(joints={"arm_l": OPEN, "arm_r": OPEN, "tab": TAB_TRAVEL}, pos=TONGS_A, quat_xyzw=quat())},
     0.0, False),
]
