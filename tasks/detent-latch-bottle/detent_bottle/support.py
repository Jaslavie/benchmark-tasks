"""Geometry-only YAM table-support test, shared by live and offline evaluators."""
import math


def table_support(position, upright_dot, *, top=.7475, center=(.36,0.),
                  half_size=(.55,.65), base_radius=.036, tolerance=.010):
    """Require the circular base's low edge near the table and its footprint inside.

    The bottom disc is centred at the bottle root. Its lowest edge is
    root_z - radius*sin(tilt). This permits a tilted bottle touching the table,
    while rejecting a bottle held above it, below it, or outside its boundaries.
    This is a geometric proximity criterion, not a contact-force measurement.
    """
    if len(position)!=3 or not all(math.isfinite(v) for v in (*position,upright_dot)):
        return dict(bottle_on_table=False,base_clearance_m=None)
    x,y,z=map(float,position)
    upright_dot=float(upright_dot)
    clearance=z-base_radius*math.sqrt(max(0.,1-min(1.,max(-1.,upright_dot))**2))-top
    inside=abs(x-center[0])<=half_size[0]-base_radius+1e-9 and abs(y-center[1])<=half_size[1]-base_radius+1e-9
    return dict(bottle_on_table=bool(inside and abs(clearance)<=tolerance+1e-9),
                base_clearance_m=clearance)
