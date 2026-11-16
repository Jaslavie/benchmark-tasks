"""USD bodies of the weld fit-up task: two bevelled steel coupons. Import after Isaac Sim starts.

Each coupon is one rigid body: a convex prism (its cross-section a square plate with one edge bevelled 30 degrees
down to a 3.2 mm root face) that collides, in the dark blue-grey of hot-rolled mill scale, plus three visual-only
layers 0.3 mm proud of the bevel, the root face and a 25 mm band of the top face beside the bevel, in bright ground
steel, as a prepared coupon looks once the bevel is ground and the mill scale cleaned off beside the joint. The
welding-table top is a fixed task-file object (weld_fitup_task.py).
"""

from __future__ import annotations

import math
from pathlib import Path

from . import design as d
from .usd_bodies import Body

MILL_SCALE = (0.22, 0.24, 0.28)
GROUND = (0.86, 0.86, 0.88)
FRICTION = (0.6, 0.5)  # dry steel on steel
PROUD = 0.0003
CLEAN_BAND = 0.025  # 1 in


def coupon(name: str) -> Path:
    b = Body(name, mass=d.MASS, friction=FRICTION)
    h, t, f, run = d.SIZE / 2, d.THICK / 2, d.ROOT_FACE, d.BEVEL_RUN
    # Cross-section in x-z: bottom from the back edge to the root face, up the root face, up the bevel, back along
    # the top.
    section = [(-h, -t), (h, -t), (h, -t + f), (h - run, t), (-h, t)]
    b.prism_y(section, -h, h, MILL_SCALE)
    b.box((h + PROUD / 2, 0.0, -t + f / 2), (PROUD, d.SIZE, f), GROUND, collide=False)  # the root face
    # The bevel: a thin slab laid on the sloped face, offset outward along its normal.
    nx, nz = math.cos(math.radians(d.BEVEL_DEG)), math.sin(math.radians(d.BEVEL_DEG))
    lo, hi = (h, -t + f), (h - run, t)
    slab = [lo, hi, (hi[0] + nx * PROUD, hi[1] + nz * PROUD), (lo[0] + nx * PROUD, lo[1] + nz * PROUD)]
    b.prism_y(slab, -h, h, GROUND, collide=False)
    # The top face cleaned to bright metal for CLEAN_BAND back from the bevel, as plates are ground clean of mill scale
    # beside the joint before welding: from above, the prepared edge shows as a bright band, not a 5 mm sliver.
    top_edge = h - run
    b.box((top_edge - CLEAN_BAND / 2, 0.0, t + PROUD / 2), (CLEAN_BAND, d.SIZE, PROUD), GROUND, collide=False)
    return b.save()


BUILDERS = {n: (lambda n=n: coupon(n)) for n in d.COUPONS}
