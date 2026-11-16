"""Near-successes and unintended strategies, scored offline by factory/check.py, in lot A. Each case: (label,
{object: ready state}, expected score, expected success). Cases run in order and share the trial record (tipped past
20 degrees), so a fault stays until a scene reset (an empty case after the carton was moved)."""

import math

import numpy as np

from task import BODY, CARTON, PLAT, PLAT_TOP, STEP_X, TOP

L = BODY[0]


def qy(deg):
    """Turned about world y; negative raises the carton's front (+x) end."""
    a = math.radians(deg) / 2
    return (0.0, math.sin(a), 0.0, math.cos(a))


def flat(x, y=0.0, z=PLAT_TOP, vel=None):
    """Upright, its underside centre at (x, y, z)."""
    return {"carton": CARTON.state(pos=(x, y, z), lin_vel=vel)}


def front_up(rear_x, rise, rear_z=TOP, vel=None):
    """Its rear bottom edge at rear_x on height rear_z, its front bottom edge `rise` higher."""
    th = math.asin(rise / L)
    q = qy(-math.degrees(th))
    r = CARTON.state(quat_xyzw=q).quat_xyzw
    from task import _rot
    origin = np.array([rear_x, 0.0, rear_z]) + _rot(r) @ np.array([L / 2, 0.0, 0.0])
    return {"carton": CARTON.state(pos=tuple(origin), quat_xyzw=q, lin_vel=vel)}


def tilted(deg, x=0.26):
    """Tipped deg about y (front up), its rear bottom edge on the table."""
    th = math.radians(deg)
    from task import _rot
    q = qy(-deg)
    origin = np.array([x - L / 2, 0.0, TOP]) + _rot(q) @ np.array([L / 2, 0.0, 0.0])
    return {"carton": CARTON.state(pos=tuple(origin), quat_xyzw=q)}


def tilted_at(x, deg):
    """Its underside centre at (x, 0) on the platform's top height, turned deg about y (front up)."""
    return {"carton": CARTON.state(pos=(x, 0.0, PLAT_TOP), quat_xyzw=qy(-deg))}


def shifted(case, dy):
    """The same state moved dy sideways."""
    st = case["carton"]
    p = st.pos
    return {"carton": CARTON.state(pos=(p[0], p[1] + dy, p[2]), quat_xyzw=st.quat_xyzw)}


ON = STEP_X + PLAT[0] / 2  # the carton centred on the platform
ON_FLAT = flat(ON)
RESET = {}
RISE = PLAT_TOP - TOP  # 20 mm

CASES = [
    ("start: on the table in front of the step", {}, 0.0, False),
    ("flat on the platform, centred", ON_FLAT, 1.0, True),
    ("flat on the platform, back edge 1 mm past the step's edge", flat(STEP_X + L / 2 + 0.001), 1.0, True),
    ("flat, back edge exactly at the step's edge", flat(STEP_X + L / 2), 1.0, True),
    ("back edge 1 mm short of the step's edge (level, overhanging the near edge)", flat(STEP_X + L / 2 - 0.001),
     0.75, False),
    ("tipped level, centre 15 mm past the step's edge (as the demo leaves it before pushing on)",
     flat(STEP_X + 0.015), 0.75, False),
    ("level overhanging the near edge, still sliding", flat(STEP_X + 0.015, vel=(0.05, 0.0, 0.0)), 0.0, False),
    ("level overhanging the near edge, 1.5 mm up (held)", flat(STEP_X + 0.015, z=PLAT_TOP + 0.0015), 0.0, False),
    ("level overhanging the near edge, centre exactly at the step's edge", flat(STEP_X), 0.0, False),
    ("level overhanging the near edge, 3 degrees off level", tilted_at(STEP_X + 0.015, 3.0), 0.0, False),
    ("level overhanging the near edge, side 1 mm past the platform's left edge",
     flat(STEP_X + 0.015, y=PLAT[1] / 2 - BODY[1] / 2 + 0.001), 0.0, False),
    ("front edge 1 mm past the platform's far edge", flat(STEP_X + PLAT[0] - L / 2 + 0.001), 0.0, False),
    ("on the platform, 1 mm up (settling)", flat(ON, z=PLAT_TOP + 0.001), 1.0, True),
    ("held 1.5 mm above the platform", flat(ON, z=PLAT_TOP + 0.0015), 0.0, False),
    ("on the platform, still sliding", flat(ON, vel=(0.05, 0.0, 0.0)), 0.0, False),
    ("front bottom edge on the platform, back on the table", front_up(STEP_X - 0.05, RISE), 0.5, False),
    ("resting across the step's edge, front 4 mm above the platform (as the demo leaves it)",
     front_up(STEP_X - 0.130, 0.0243), 0.5, False),
    ("across the step, front corner 1 mm past the platform's left edge",
     shifted(front_up(STEP_X - 0.05, RISE), PLAT[1] / 2 - BODY[1] / 2 + 0.001), 0.0, False),
    ("front edge just past the step's edge, back on the table", front_up(STEP_X - L * math.cos(math.asin(RISE / L))
                                                                         + 0.002, RISE), 0.5, False),
    ("front edge on the platform, back 2 mm up (held)", front_up(STEP_X - 0.05, RISE - 0.002, rear_z=TOP + 0.002),
     0.0, False),
    ("front edge on the platform, still moving", front_up(STEP_X - 0.05, RISE, vel=(0.05, 0.0, 0.0)), 0.0, False),
    ("pushed flat against the step's face (jammed)", flat(STEP_X - L / 2, z=TOP), 0.0, False),
    ("front edge lifted 10 mm in the air, back on the table", front_up(0.20, 0.010), 0.0, False),
    ("front edge 19 mm up, short of the step's edge", front_up(STEP_X - 0.161, 0.019), 0.0, False),
    ("tipped 19 degrees (no fault)", tilted(19.0), 0.0, False),
    ("then flat on the platform: 1.0", ON_FLAT, 1.0, True),
    ("scene reset", RESET, 0.0, False),
    # Faults from here on; each is followed by a scene reset.
    ("tipped 21 degrees", tilted(21.0), 0.0, False),
    ("then flat on the platform: still 0", ON_FLAT, 0.0, False),
    ("scene reset", RESET, 0.0, False),
    ("after the reset, flat on the platform: 1.0", ON_FLAT, 1.0, True),
    ("scene reset", RESET, 0.0, False),
    ("lying on its front face on the platform",
     {"carton": CARTON.state(pos=(ON, 0.0, PLAT_TOP + L / 2), quat_xyzw=qy(-90.0))}, 0.0, False),
    ("scene reset", RESET, 0.0, False),
    ("upside down on the platform",
     {"carton": CARTON.state(pos=(ON, 0.0, PLAT_TOP + BODY[2]), quat_xyzw=qy(180.0))}, 0.0, False),
]
