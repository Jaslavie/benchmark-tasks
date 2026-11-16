"""Geometry, start layout and scoring of the weld fit-up task. Pure NumPy: no Isaac, testable anywhere.

Two 1/2-inch steel coupons, 3 x 3 in, each with one edge prepared for a single-V groove butt weld: a 30-degree
bevel down to a 1/8-inch root face. They lie loose on a fixed steel welding-table top, each turned so its bevel faces
the wrong way. Fit-up: both bevels facing each other across the joint (a 60-degree V), a 1/16 to 1/8 in root opening
all along the joint, and the coupons' ends in line.

Frames: positions in metres, x forward from the line between the arm bases, y to the left, z up (world height,
relative to the env origin). A coupon's own frame has its centre at the origin, its bevelled edge on the +x face,
the bevel's wide side up. Poses are xyz + quaternion xyzw.
"""

from __future__ import annotations

import math

import numpy as np

INCH = 0.0254
TOP = 0.7475  # the station table's top (yam_arena.task_file.TABLE_TOP["station"])

# The welding-table top: a fixed steel plate on the station table.
BENCH_XY = (0.34, 0.205)
BENCH_SIZE = (0.30, 0.32, 0.010)
BENCH_TOP = TOP + BENCH_SIZE[2]

# A coupon (AWS D1.1 prequalified single-V groove, B-U2: root face and root opening 0 to 1/8 in, groove angle 60
# degrees; this fit-up details both at 1/8 in).
SIZE = 3 * INCH  # 76.2 mm square
THICK = 0.5 * INCH  # 12.7 mm
ROOT_FACE = 0.125 * INCH  # 3.2 mm of square edge at the bottom of the bevel
BEVEL_DEG = 30.0  # each coupon's bevel from vertical: 60 degrees across the groove
BEVEL_RUN = (THICK - ROOT_FACE) * math.tan(math.radians(BEVEL_DEG))  # 5.5 mm: how far the top edge sits back
STEEL_KG_M3 = 7850.0
MASS = STEEL_KG_M3 * (SIZE * SIZE * THICK - 0.5 * BEVEL_RUN * (THICK - ROOT_FACE) * SIZE)  # 0.57 kg

# The fit-up: a root opening of 1/16 to 1/8 in (1.6 to 3.2 mm), measured at both ends of the joint, as an AWS
# groove-weld procedure gives it (LANL WPS 3501-11B, "Root Opening: 1/16"-1/8""). The demo aims at the middle.
OPENING_MIN, OPENING_MAX = 0.0625 * INCH, 0.125 * INCH
ROOT_OPENING = (OPENING_MIN + OPENING_MAX) / 2  # 2.4 mm
STAGGER_MAX = 0.005  # the coupons' ends in line within 5 mm along the joint
# Partial credit: the coupons laid side by side along a joint, not yet within the fit-up limits.
ROUGH_OPENING_MAX = 0.010
ROUGH_STAGGER_MAX = 0.020
ROUGH_PARALLEL_DEG = 10.0
OVERLAP_TOL = 0.0005  # PhysX contact: touching coupons can read a fraction of a millimetre "inside" each other

# A coupon lying flat on the welding table.
FLAT_TILT_DEG = 5.0
ON_BENCH_TOL = 0.003  # its centre within 3 mm of its height lying flat on the bench
STILL_MPS, STILL_RADPS = 0.01, 0.5
OFF_TABLE_Z = TOP - 0.01  # a coupon whose centre is below this has fallen off

COUPONS = ("coupon_a", "coupon_b")
# Start: both coupons at the far side of the welding table, coupon_a (left) with its bevel pointing straight away
# from the arms, coupon_b (right) with its bevel pointing right, away from coupon_a. The fit-up the demo makes, with
# the joint running straight out from the arms 22 cm left of centre, needs coupon_a turned 90 degrees and coupon_b
# turned end for end. Everything is on the left arm's side: the arm holds a 0.56 kg coupon to a millimetre near its
# own side, not reaching across the table (with the joint at the centre, each arm landed its coupon 5-12 mm off).
START = {"coupon_a": ((0.40, 0.29), 0.0), "coupon_b": ((0.40, 0.13), -90.0)}  # (x, y), bevel heading in degrees
JOINT_XY = (0.30, 0.22)  # where the demo makes the joint (the scoring doesn't care where)


def yaw_quat(yaw_deg: float) -> np.ndarray:
    h = math.radians(yaw_deg) / 2
    return np.array([0.0, 0.0, math.sin(h), math.cos(h)])


def pose(xy, yaw_deg: float, z: float = BENCH_TOP + THICK / 2) -> np.ndarray:
    return np.r_[float(xy[0]), float(xy[1]), z, yaw_quat(yaw_deg)]


def start_poses() -> dict[str, np.ndarray]:
    return {n: pose(xy, yaw) for n, (xy, yaw) in START.items()}


def rot(quat_xyzw) -> np.ndarray:
    x, y, z, w = (float(v) for v in quat_xyzw)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def tilt_deg(p) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, rot(p[3:])[2, 2]))))


def _flat(v) -> np.ndarray:
    v = np.array([v[0], v[1], 0.0])
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def bevel_dir(p) -> np.ndarray:
    """Which way the coupon's bevel faces, in the table plane (unit)."""
    return _flat(rot(p[3:])[:, 0])


def heading_deg(p) -> float:
    b = bevel_dir(p)
    return math.degrees(math.atan2(b[1], b[0]))


def root_edge(p) -> tuple[np.ndarray, np.ndarray]:
    """The two ends of the coupon's root edge (the bottom of its bevelled side), world xyz."""
    r = rot(p[3:])
    c = np.asarray(p[:3], float)
    mid = c + r @ np.array([SIZE / 2, 0.0, -THICK / 2])
    along = r @ np.array([0.0, SIZE / 2, 0.0])
    return mid - along, mid + along


def flat(p) -> bool:
    return tilt_deg(p) <= FLAT_TILT_DEG and abs(float(p[2]) - (BENCH_TOP + THICK / 2)) <= ON_BENCH_TOL


def off_table(p) -> bool:
    return float(p[2]) < OFF_TABLE_Z


def faces(p, other) -> bool:
    """The coupon's bevel looks at the other coupon: the other's centre lies ahead of the bevelled side, within the
    bevelled edge's length."""
    d = np.asarray(other[:3], float) - np.asarray(p[:3], float)
    along = float(d[:2] @ bevel_dir(p)[:2])
    side = float(d[:2] @ _flat(rot(p[3:])[:, 1])[:2])
    return along > 0 and abs(side) <= SIZE / 2


def joint(a, b) -> dict:
    """Root opening at both ends of the joint and the stagger, for coupons laid with a's bevelled side toward b.

    The opening is measured from each end of one coupon's root edge to the other coupon's bevelled side (its root
    face's plane), both ways, so a joint that is wider at one end reads wider there. The stagger is how far the
    coupons' centres sit apart along the joint."""
    ba, bb = bevel_dir(a), bevel_dir(b)
    face_a = np.asarray(a[:3], float) + ba * SIZE / 2
    face_b = np.asarray(b[:3], float) + bb * SIZE / 2
    gaps = [float((q - face_b)[:2] @ bb[:2]) for q in root_edge(a)]
    gaps += [float((q - face_a)[:2] @ ba[:2]) for q in root_edge(b)]
    along = _flat(rot(a[3:])[:, 1])
    stagger = float((np.asarray(b[:3], float) - np.asarray(a[:3], float))[:2] @ along[:2])
    parallel = math.degrees(math.acos(max(-1.0, min(1.0, float(-ba @ bb)))))
    return {"open_min": min(gaps), "open_max": max(gaps), "stagger": stagger, "parallel_deg": parallel}


def grade(poses: dict, still: dict | None = None) -> tuple[float, str]:
    """The fit-up's credit (1.0 is the complete fit-up; the embodiment turns it into the trial's score) and why.

    1.0  both coupons flat on the table, both bevels facing each other, the root opening 1.6-3.2 mm at both ends of
         the joint, the ends in line within 5 mm, both still
    0.5  both flat and still, both bevels facing each other, side by side (opening 0-10 mm, ends within 2 cm), but
         the opening or the ends outside the fit-up limits
    0.25 both flat and still, side by side and parallel within 10 degrees with only one bevel at the joint (a square
         edge in the joint)
    0    otherwise, including a coupon off the table, on edge or on top of the other, or moving
    """
    a, b = (np.asarray(poses[n], float) for n in COUPONS)
    if not all(np.all(np.isfinite(p)) for p in (a, b)):
        return 0.0, "non-finite state"
    if off_table(a) or off_table(b):
        return 0.0, "a coupon is off the table"
    if not (flat(a) and flat(b)):
        return 0.0, "a coupon isn't lying flat on the table"
    if still is not None and not all(still.get(n, False) for n in COUPONS):
        return 0.0, "moving"
    fa, fb = faces(a, b), faces(b, a)
    if fa and fb:
        j = joint(a, b)
        if j["parallel_deg"] > ROUGH_PARALLEL_DEG:
            return 0.0, "bevels toward each other but not parallel"
        if (OPENING_MIN <= j["open_min"] and j["open_max"] <= OPENING_MAX and abs(j["stagger"]) <= STAGGER_MAX):
            return 1.0, "fit up"
        if j["open_min"] >= -OVERLAP_TOL and j["open_max"] <= ROUGH_OPENING_MAX and abs(j["stagger"]) <= ROUGH_STAGGER_MAX:
            return 0.5, (f"bevels facing, opening {j['open_min'] * 1000:.1f}-{j['open_max'] * 1000:.1f} mm, "
                         f"stagger {j['stagger'] * 1000:+.1f} mm")
        return 0.0, "bevels facing but not side by side"
    if fa != fb:
        x, y = (a, b) if fa else (b, a)  # x faces the joint, y shows it a square edge
        # The square edge y shows the joint: its side toward x. Measure the opening against that face instead.
        to = _flat(np.asarray(x[:3], float) - np.asarray(y[:3], float))
        r = rot(y[3:])
        sides = [_flat(r[:, 0]), _flat(r[:, 1]), -_flat(r[:, 0]), -_flat(r[:, 1])]
        side = max(sides, key=lambda s: float(s @ to))
        heading = math.degrees(math.atan2(side[1], side[0]))
        y_square = np.r_[y[:3], yaw_quat(heading)]  # y as if that square side were its bevelled one
        j = joint(x, y_square)
        if (j["parallel_deg"] <= ROUGH_PARALLEL_DEG and j["open_min"] >= -OVERLAP_TOL
                and j["open_max"] <= ROUGH_OPENING_MAX and abs(j["stagger"]) <= ROUGH_STAGGER_MAX):
            return 0.25, "side by side with a square edge in the joint"
        return 0.0, "only one bevel at the joint, not side by side"
    return 0.0, "neither bevel faces the joint"


def target_poses(opening: float = ROOT_OPENING) -> dict[str, np.ndarray]:
    """Where the demo sets the coupons: the joint along x at JOINT_XY, coupon_a on its left with its bevel facing
    right (heading -90 degrees), coupon_b on its right facing left (+90)."""
    (x, y), off = JOINT_XY, opening / 2 + SIZE / 2
    return {"coupon_a": pose((x, y + off), -90.0), "coupon_b": pose((x, y - off), 90.0)}
