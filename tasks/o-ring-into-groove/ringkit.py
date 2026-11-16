"""ringkit: elastic and split rings on a shaft (circlips, snap rings, O-rings), built on yamkit.

A ring sits on a shaft. Its inner radius is spring-loaded: left alone it closes toward its free (rest) radius, which is
smaller than the shaft, so it grips the shaft wherever it is, or springs into a groove when it is over one. A tool
spreads it (circlip pliers in its lug holes, a stretching cone for an O-ring); spread past the shaft it slides along
the shaft and turns about it with the tool. Spread past its elastic limit it takes a permanent set and is scrap.

    import yamkit as kit
    import ringkit as rk

    CLIP = rk.Ring("clip", end=(0, 0, 0.105), axis="-Z", shaft_d=0.040, free_id=0.0365, thickness=0.00175,
                   width=0.0045, length=0.05, grooves=(rk.Groove(at=0.0381, width=0.00185, diameter=0.0375),),
                   stop=0.039, limit_id=0.041, lug_gap=0.012, lug_out=0.0058, look=kit.Look((0.2, 0.2, 0.25)))
    fixture = kit.Mechanism("fixture", ..., links=(*rk.links(CLIP), ...))      # the shaft is part of the base
    pliers = rk.free_tool(kit.Mechanism("pliers", ..., mass=0.07, links=(jaw,)))  # a loose, articulated tool
    rk.attach("fixture", CLIP, rk.TipPair(("pliers", tip_a), ("pliers/jaw", tip_b)))

    rk.seated(objects, "fixture/clip")       # in the scorer

The model (all lengths in m; `s` is the ring's mid-plane position along the shaft, measured from the shaft's end face
into the shaft, so s = -thickness/2 is a ring lying on the end face):
- Shaft profile: radius shaft_d/2 for 0 < z <= length, a groove's bottom radius within each `Groove`, nothing past the
  end (z <= 0). The ring covers [s - t/2, s + t/2]; the radius it has to clear is the largest radius under it.
- Free: the ring closes to max(free radius + permanent set, the radius under it): gripping the land, closed into a
  groove, or (off the end) at its free radius. It does not move along the shaft or turn.
- Snap: a closed ring whose mid-plane is within half a groove width of the groove's middle sits in the groove (its own
  spring pushes it in); `stop` (optional) is a face the ring's far face cannot pass (the part it retains, a shoulder).
- Held by a tool (a spreader reports a `Grip`): the ring opens to what the tool asks (never smaller than the shaft
  under it allows), turns with the tool (unless cocked) and moves along the shaft toward where the tool puts it, but
  only as far as its radius clears the shaft by `clear` (diametral): a ring not spread far enough stays where it is,
  and the tool's tips slip out of it (the spreader then stops reporting a grip).
- Over-spread: the largest inner diameter ever reached is kept; past `limit_id` the excess becomes a permanent set
  (the free radius grows by it): such a ring may close loosely in its groove.
- Tilt and twist: the tool reports how far it tilts the ring's plane off square to the shaft and how far it twists
  the ring's two ends apart along the shaft. A ring that closes into a groove tilted more than `tilt_limit` degrees
  lands cocked (one side in the groove, the other on the land; not seated, it cannot turn); spreading it again frees
  it, and (FingerTips, this copy) a fingertip that comes down onto its high side and presses it half its thickness
  past the land's edge pushes it into the groove (an elastomer ring rolls by that distance on the way). The largest tilt and twist while held are kept for the scorer (FingerTips: tilt from the tips' heights).
- Seated: in a groove, closed onto the groove bottom (inner radius within `seat_tol` of it), not cocked, not held.

Spreaders (anything with `grip(ring, st, frame, objects) -> Grip | None`): `TipPair` is a pair of pliers tips in a
split ring's two lug holes. A tip engages when it is within `catch` of a lug hole across the shaft axis and its end
is inside the hole (below the ring's near face, at most `depth` past its far face). The ring's lug gap follows the
tips' spacing, so its inner radius grows by (tip spacing - rest lug gap) / (2 pi). Write another spreader for other
tools (an O-ring cone, fingers): it gets the ring spec, its state and the shaft frame, and returns a Grip or None.
`FingerTips` (this task's copy) is a gripper's two fingertips in an elastomer ring: the gripper opening stretches it
(its inner circumference is the convex loop round the fingertips' outer faces and, once both are outside the shaft,
the shaft), it rides down on the fingertips' ends and stays when they are drawn back out. A fingertip pressing on
the ring's face without stretching it pushes it (`Grip(push=True)`): a `Ring(roll=True)` then rolls along the shaft
instead of sliding, its section twisting by distance / (thickness / 2) rad; stretched clear of the shaft again it
springs back untwisted. A stretched ring is drawn along that loop.

Joints: `links(ring)` gives the ring's invisible, collider-less links for the fixture mechanism, all on its base:
"<name>_axial" (prismatic along the shaft: s), "<name>_turn" (revolute about it: the ring's turn, ccw seen from the
end) and "<name>" (prismatic: the opening, inner radius minus free radius). The kit writes them every control step
from its model, so `kit.joint(objects, "fixture/clip")` etc. read the ring too (one step behind the kit's readouts).
The ring is drawn with Isaac Lab visualization markers (segments of a band and two lug ears) at its current radius,
turn, tilt and twist; drawing never affects scoring. New marker prims take about five rendered frames to appear, so a
fourth link, "<name>_shown", carries a mesh of the ring at its start pose for the first SHOW_STEPS steps and is then
moved to s = `hide` (inside the table or fixture). Use a matte look: a metallic dark ring mirrors the bright dome.

Free tools: `free_tool(mech)` marks a loose (massed) Mechanism so that its base body carries the articulation root
(otherwise PhysX roots a floating articulation at its most central body and the joint drives misbehave).
`HandleSpring` is a law for a tool's handle (a preloaded return spring).

Scorer readouts (key "fixture/clip"; all reset with kit.episode()):
  position(objects, key)       s now (m)               groove(objects, key)      index of the groove it sits in, or None
  seated(objects, key)         seated now (see above)  engaged(objects, key)     a tool holds it now
  inner_d(objects, key)        inner diameter now (m)  max_d(objects, key)       largest inner diameter this trial
  overspread(objects, key)     max_d past limit_id     set_mm(objects, key)      permanent set (mm of radius)
  tilt_deg / max_tilt_deg      plane off square         max_twist_deg             largest twist while held
  cocked(objects, key)         landed tilted in a groove
  turned_deg(objects, key)     degrees turned while closed in a groove and not cocked (a seating check)
  twist_deg(objects, key)      an elastomer ring's roll left in it now (deg); max_roll_deg the largest this trial
Offline tests call `update(objects)` once per control step with hand-made ObjectStates (fixture, tool parts).
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field, replace

import numpy as np
import yamkit as kit

# -- specs ------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Groove:
    """A groove: its middle at `at` (m from the shaft's end face, into the shaft), `width` along the shaft and
    `diameter` at its bottom."""

    at: float
    width: float
    diameter: float


@dataclass(frozen=True)
class Ring:
    """An elastic or split ring on a shaft that is part of a mechanism's base. `end` is the shaft's end-face center in
    the base's frame and `axis` the direction into the shaft from there ("-Z" for an upright shaft with its end up;
    "X", "-Y", ...). `free_id` is the inner diameter at rest, `thickness` the ring's size along the shaft, `width` its
    radial section. `length` is how far along the shaft it can go (m from the end face); `start` its starting s
    (default: lying on the end face). `limit_id` is the elastic limit (None: no limit), `clear` the diametral
    clearance it needs to slide. A split ring (circlip) has lug holes `lug_gap` apart at rest, their centers `lug_out`
    beyond the inner radius, its split at `lug_angle` (rad, ccw seen from the end, from the shaft frame's first
    axis); a closed ring (O-ring) has lug_gap 0. See the module docstring for the model."""

    name: str
    end: tuple
    shaft_d: float
    free_id: float
    thickness: float
    width: float
    length: float
    axis: str = "-Z"
    grooves: tuple = ()
    stop: float | None = None
    start: float | None = None
    limit_id: float | None = None
    clear: float = 0.0002
    lug_gap: float = 0.0
    lug_out: float = 0.0
    lug_angle: float = math.pi
    tilt_limit: float = 4.0
    roll: bool = False  # an elastomer ring (O-ring): pushed along the shaft without being spread, it rolls (twists)
    seat_tol: float = 0.00005
    hide: float = 0.15  # m along the shaft from its end face where the start-pose placeholder mesh is put away
    look: kit.Look = kit.Look((0.05, 0.05, 0.06), 0.85)  # matte: a metallic dark ring reflects the dome and vanishes

    @property
    def s0(self) -> float:
        return -self.thickness / 2 if self.start is None else self.start


@dataclass(frozen=True)
class Grip:
    """What a tool holding the ring asks of it this step: `s` where it would put the ring (m along the shaft), `theta`
    the tool's angle about the shaft (rad, ccw from the end; only changes matter), `r` the inner radius it spreads the
    ring to (m; None: it does not spread it), `tilt` (deg) how far it holds the ring's plane off square, `tilt_axis`
    (unit vector, world) the axis of that tilt, `twist` (deg, signed) how far it twists the ring's ends apart."""

    s: float
    theta: float
    r: float | None = None
    tilt: float = 0.0
    tilt_axis: tuple = (1.0, 0.0, 0.0)
    twist: float = 0.0
    push: bool = False  # a fingertip pressing on the ring's face (an elastomer ring then rolls along the shaft)
    press: float = 0.0  # m: a fingertip pressed a cocked ring's high side this far down, over the land into its groove
    path: tuple | None = None  # the stretched ring's inner contour (2D points in the shaft frame's e1, e2), drawing only


@dataclass
class RingState:
    """The ring this trial (see the module docstring); lengths in m, angles in deg unless noted."""

    episode: int
    s: float
    r: float  # inner radius now
    theta: float = 0.0  # rad, ccw from the end
    set: float = 0.0  # permanent set (radius)
    max_r: float = 0.0
    engaged: bool = False
    grip: Grip | None = None
    theta_off: float = 0.0  # ring turn minus tool angle, kept from when the tool took hold
    tilt: float = 0.0
    tilt_axis: tuple = (1.0, 0.0, 0.0)
    twist: float = 0.0
    max_tilt: float = 0.0
    max_twist: float = 0.0
    cocked: bool = False
    groove: int | None = None
    turned: float = 0.0  # deg turned while closed in a groove (not cocked)
    roll: float = 0.0  # deg an elastomer ring's section has rolled (twisted) about its own centerline, now
    max_roll: float = 0.0
    path: tuple | None = None
    grip_by: str | None = None  # which tool part holds it (spreaders that know several)
    steps: int = 0
    above: dict = field(default_factory=dict)  # FingerTips: per fingertip, was its end on the near side of the ring's
    #                                            near face last step (only such a tip can push the ring)


@dataclass
class _Attach:
    mech: str
    ring: Ring
    spreader: object = None


_RINGS: dict[str, _Attach] = {}  # "mechanism/ring" -> attachment
_STATE: dict[str, RingState] = {}
_FREE: dict[str, tuple | None] = {}  # free tools: name -> pinned base center of mass


def attach(mech: str, ring: Ring, spreader=None) -> str:
    """Put `ring` on the shaft of mechanism `mech` (whose links include `links(ring)`), worked by `spreader`. Returns
    the readout key "mech/ring". Call at module level in the task file."""
    key = f"{mech}/{ring.name}"
    _RINGS[key] = _Attach(mech, ring, spreader)
    return key


def free_tool(mech, com=None):
    """Mark a loose (massed) Mechanism so its base carries the articulation root. `com` (base frame, m) pins the
    base's center of mass, e.g. at a tool's grip point: a two-pad grip barely resists turning about the pads' normal,
    so a hand tool whose weight sits forward of the pads pitches nose-down in the gripper (sim pilot, Nov 10: 57
    degrees). Returns the mechanism unchanged."""
    _FREE[mech.name] = com
    return mech


# -- shaft frame and profile -------------------------------------------------------------------------


def _letter(axis: str) -> tuple[str, int]:
    a = axis.strip().upper()
    return a[-1], (-1 if a.startswith("-") else 1)


def _unit(axis: str) -> np.ndarray:
    k, sg = _letter(axis)
    v = np.zeros(3)
    v["XYZ".index(k)] = sg
    return v


def _rotm(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _local_frame(ring: Ring):
    """In the base's frame: u (into the shaft), e1, e2 with (e1, e2, -u) right-handed, so angles run ccw seen from
    the end."""
    u = _unit(ring.axis)
    out = -u
    e1 = np.array([0.0, 0.0, 1.0]) if abs(out[2]) < 0.5 else np.array([1.0, 0.0, 0.0])
    e1 = e1 - np.dot(e1, out) * out
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(out, e1)
    return u, e1, e2


@dataclass(frozen=True)
class Frame:
    """The shaft in world coordinates (relative to the env origin): end-face center `o`, into-shaft `u`, `e1`, `e2`."""

    o: np.ndarray
    u: np.ndarray
    e1: np.ndarray
    e2: np.ndarray

    def s_of(self, p) -> float:
        return float(np.dot(np.asarray(p, float) - self.o, self.u))

    def angle_of(self, p) -> float:
        d = np.asarray(p, float) - self.o
        return math.atan2(float(np.dot(d, self.e2)), float(np.dot(d, self.e1)))

    def radial(self, p) -> float:
        d = np.asarray(p, float) - self.o
        return float(np.linalg.norm(d - np.dot(d, self.u) * self.u))

    def point(self, s: float, ang: float, rad: float) -> np.ndarray:
        return self.o + s * self.u + rad * (math.cos(ang) * self.e1 + math.sin(ang) * self.e2)


def frame(objects, mech: str, ring: Ring) -> Frame:
    b = objects[mech]
    rot = _rotm(b.quat_xyzw)
    u, e1, e2 = _local_frame(ring)
    return Frame(np.asarray(b.pos, float) + rot @ np.asarray(ring.end, float), rot @ u, rot @ e1, rot @ e2)


def shaft_r(ring: Ring, z: float) -> float:
    """The shaft's radius at z (m from the end face): 0 off the end, a groove's bottom inside it."""
    if z <= 1e-6 or z > ring.length + 1e-6:
        return 0.0
    for g in ring.grooves:
        if abs(z - g.at) <= g.width / 2 + 1e-7:
            return g.diameter / 2
    return ring.shaft_d / 2


def span_r(ring: Ring, s: float) -> float:
    """The largest shaft radius under a ring at s (its section covers s +/- thickness/2)."""
    h = ring.thickness / 2 - 1e-6
    zs = np.linspace(s - h, s + h, 9)
    pts = [shaft_r(ring, float(z)) for z in zs]
    for g in ring.grooves:  # groove edges inside the span
        for e in (g.at - g.width / 2 - 1e-7, g.at + g.width / 2 + 1e-7):
            if s - h < e < s + h:
                pts.append(shaft_r(ring, e))
    return max(pts)


def _snap(ring: Ring, s: float):
    """(groove index, s at its middle) if a ring closing at s is pushed into a groove, else (None, s)."""
    for i, g in enumerate(ring.grooves):
        if abs(s - g.at) <= g.width / 2:
            return i, g.at
    return None, s


def _s_max(ring: Ring) -> float:
    hi = ring.length - ring.thickness / 2
    if ring.stop is not None:
        hi = min(hi, ring.stop - ring.thickness / 2)
    return hi


def _travel(ring: Ring, s: float, target: float, r: float) -> float:
    """Move from s toward target in 0.2 mm steps while the ring (inner radius r) clears the shaft by `clear`."""
    target = min(max(target, ring.s0), _s_max(ring))
    n = int(abs(target - s) / 0.0002) + 1
    ok = s
    for k in range(1, n + 1):
        z = s + (target - s) * k / n
        if span_r(ring, z) > r - ring.clear / 2 + 1e-9:
            break
        ok = z
    return ok


# -- per-step model --------------------------------------------------------------------------------


def state(key: str) -> RingState:
    """The ring's state this trial (a fresh one after every scene reset)."""
    st = _STATE.get(key)
    if st is None or st.episode != kit.episode():
        ring = _RINGS[key].ring
        r = max(ring.free_id / 2, span_r(ring, ring.s0))
        st = _STATE[key] = RingState(kit.episode(), ring.s0, r, max_r=r)
        st.groove = _in_groove(ring, st)
    return st


def _in_groove(ring: Ring, st: RingState):
    for i, g in enumerate(ring.grooves):
        if abs(st.s - g.at) <= (g.width - ring.thickness) / 2 + 1e-6:
            return i
    return None


def _step(key: str, grip: Grip | None) -> RingState:
    att = _RINGS[key]
    ring, st = att.ring, state(key)
    free_r = ring.free_id / 2 + st.set
    was_open = st.r > span_r(ring, st.s) + 1e-6  # spread clear of what is under it
    theta0 = st.theta
    if grip is not None and grip.push and grip.press > 0 and st.cocked:
        # a cocked ring's high side pressed down off the land: it rolls over the last crest (by the same law as
        # below) and the ring's own tension pulls it into the groove; it closes there square
        st.cocked = False
        st.tilt, st.twist = 0.0, 0.0
        if ring.roll:
            st.roll += math.degrees(grip.press / (ring.thickness / 2))
        grip = replace(grip, r=None)
    elif grip is not None and grip.push:
        # pressed on its face, not spread: an elastomer ring rolls along the shaft as far as it is pushed (only
        # deeper); every m it rolls over the shaft turns its section by m / (half its thickness) rad
        if ring.roll and st.groove is None:
            target = min(max(grip.s, st.s), _s_max(ring))
            rolled = max(0.0, target - max(st.s, 0.0))
            st.roll += math.degrees(rolled / (ring.thickness / 2))
            st.s = target
        grip = replace(grip, r=None)
    elif grip is not None:
        if not st.engaged or (st.grip is not None and st.grip.push):
            st.theta_off = st.theta - grip.theta
        under = span_r(ring, st.s)
        want = grip.r if grip.r is not None else free_r
        st.r = max(want, free_r, under)
        if st.cocked and st.r > ring.shaft_d / 2 + ring.clear / 2:
            st.cocked = False  # spread again: free of the land
        st.s = _travel(ring, st.s, grip.s, st.r)
        if not st.cocked:
            st.theta = grip.theta + st.theta_off
        st.tilt, st.tilt_axis, st.twist = grip.tilt, tuple(grip.tilt_axis), grip.twist
        st.max_tilt = max(st.max_tilt, abs(grip.tilt))
        st.max_twist = max(st.max_twist, abs(grip.twist))
    st.engaged = grip is not None
    st.grip = grip
    st.path = grip.path if grip is not None and not grip.push else None
    under = span_r(ring, st.s)
    closing_r = max(free_r, under)
    if grip is None or grip.push or (grip.r is not None and grip.r <= closing_r):
        # nothing (or nothing more) spreading it: the ring closes onto what is under it, into a groove if over one
        if under > 0 and not st.cocked:
            g, s_g = _snap(ring, st.s)
            if g is not None and free_r <= ring.grooves[g].diameter / 2 + ring.seat_tol:
                if was_open and abs(st.tilt) > ring.tilt_limit:
                    st.cocked = True  # one side lands on the land
                else:
                    st.s = s_g
            under = span_r(ring, st.s)
        st.r = ring.shaft_d / 2 if st.cocked else max(free_r, under)
    if grip is None and not st.cocked:
        st.tilt, st.twist = 0.0, 0.0  # let go square: nothing tilts it any more
    if ring.roll:
        if grip is not None and not grip.push and st.r >= span_r(ring, st.s) + ring.clear / 2:
            st.roll = 0.0  # spread clear of the shaft: the elastomer springs back untwisted
        st.max_roll = max(st.max_roll, st.roll)
        st.twist, st.max_twist = st.roll, max(st.max_twist, st.roll)
    st.max_r = max(st.max_r, st.r)
    if ring.limit_id is not None and st.r > ring.limit_id / 2:
        st.set = max(st.set, st.r - ring.limit_id / 2)
    st.groove = _in_groove(ring, st)
    if st.groove is not None and not st.cocked and st.r <= ring.grooves[st.groove].diameter / 2 + ring.seat_tol:
        st.turned += abs(math.degrees(_wrap(st.theta - theta0)))
    st.steps += 1
    return st


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def update(objects) -> None:
    """Advance every attached ring by one control step (the sim calls this; offline tests call it themselves)."""
    for key, att in _RINGS.items():
        if att.mech not in objects:
            continue
        st = state(key)
        fr = frame(objects, att.mech, att.ring)
        grip = att.spreader.grip(att.ring, st, fr, objects) if att.spreader is not None else None
        st = _step(key, grip)
        if hasattr(att.spreader, "after"):  # a spreader that tracks its tool between steps
            att.spreader.after(att.ring, st, fr, objects)


# -- spreaders ------------------------------------------------------------------------------------------


def lug_holes(ring: Ring, st: RingState, fr: Frame):
    """World positions of the two lug-hole centers (mid-plane) of a split ring now."""
    rad = st.r + ring.lug_out
    gap = ring.lug_gap + 2 * math.pi * (st.r - ring.free_id / 2)
    half = (gap / 2) / rad
    mid = st.theta + ring.lug_angle
    return fr.point(st.s, mid - half, rad), fr.point(st.s, mid + half, rad)


@dataclass(frozen=True)
class TipPair:
    """Two tool tips (circlip pliers) for a split ring's lug holes. `a` and `b` are (object key, tip end point in that
    object's frame); `down` is the direction the tips point in object a's frame (for tilt). A tip engages within
    `catch` (m) of a lug hole across the shaft axis with its end at least `enter` past the ring's near face and at
    most `depth` past its far face; both must engage. Held, the ring's far face sits at the tips' ends."""

    a: tuple
    b: tuple
    down: tuple = (0.0, 0.0, -1.0)
    catch: float = 0.004
    enter: float = 0.0005
    depth: float = 0.003

    def tips(self, objects):
        out = []
        for key, p in (self.a, self.b):
            o = objects[key]
            out.append(np.asarray(o.pos, float) + _rotm(o.quat_xyzw) @ np.asarray(p, float))
        return out

    def grip(self, ring: Ring, st: RingState, fr: Frame, objects):
        if self.a[0] not in objects or self.b[0] not in objects:
            return None
        ta, tb = self.tips(objects)
        # what the tips ask: the ring's far face at their ends, its lug gap their spacing, its turn their angle
        za, zb = fr.s_of(ta), fr.s_of(tb)
        da = (ta - fr.o) - np.dot(ta - fr.o, fr.u) * fr.u
        db = (tb - fr.o) - np.dot(tb - fr.o, fr.u) * fr.u
        gap = float(np.linalg.norm(da - db))
        r = ring.free_id / 2 + (gap - ring.lug_gap) / (2 * math.pi)
        mid = da + db
        theta = math.atan2(float(np.dot(mid, fr.e2)), float(np.dot(mid, fr.e1))) - ring.lug_angle
        s = (za + zb) / 2 - ring.thickness / 2
        # Where the lug holes are: a free ring's holes are where the ring is now. Once the tips are in, the ring
        # deforms, turns and slides with them as far as the shaft lets it, so check them against where it would be.
        probe = st
        if st.engaged:
            r_new = max(r, ring.free_id / 2 + st.set, span_r(ring, st.s))
            th_new = st.theta if st.cocked else theta + st.theta_off
            probe = replace(st, r=r_new, theta=th_new, s=_travel(ring, st.s, s, r_new))
        ha, hb = lug_holes(ring, probe, fr)

        def across(p, h):
            d = p - h
            return float(np.linalg.norm(d - np.dot(d, fr.u) * fr.u))

        pair = min(((ta, ha), (tb, hb)), ((ta, hb), (tb, ha)), key=lambda c: across(*c[0]) + across(*c[1]))
        near, far = probe.s - ring.thickness / 2, probe.s + ring.thickness / 2
        for p, h in pair:
            z = fr.s_of(p)
            if across(p, h) > self.catch or not (near + self.enter <= z <= far + self.depth):
                return None
        rot_a = _rotm(objects[self.a[0]].quat_xyzw)
        d = rot_a @ np.asarray(self.down, float)
        d /= np.linalg.norm(d)
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(d, fr.u))))))
        ax = np.cross(fr.u, d)
        ax = tuple(ax / np.linalg.norm(ax)) if np.linalg.norm(ax) > 1e-9 else (1.0, 0.0, 0.0)
        twist = math.degrees(math.atan2(zb - za, max(gap, 1e-6)))
        return Grip(s=s, theta=theta, r=r, tilt=tilt, tilt_axis=ax, twist=twist)


def hull(points) -> list:
    """The convex hull of 2D points, counterclockwise (Andrew's monotone chain)."""
    pts = sorted(set((round(float(x), 9), round(float(y), 9)) for x, y in points))
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lo, up = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    return lo[:-1] + up[:-1]


def perimeter(poly) -> float:
    return sum(math.dist(poly[i], poly[i - 1]) for i in range(len(poly))) if len(poly) > 1 else 0.0


@dataclass(frozen=True)
class FingerTips:
    """A parallel gripper's two fingertips stretching an elastomer ring (an O-ring) by hand: they go into the ring's
    hole and the gripper opens. The ring's inner circumference is the loop round the fingertips' outer faces (each
    `2 * half_w` wide, `out` outside its pad face, contact `up` above the tip end) and, once both fingers are outside
    the shaft, round the shaft too (their convex hull); its inner radius is that loop's length / 2 pi. The fingers
    engage when both are inside the ring's hole with the shaft axis between them (the line between the fingertips
    within `axis_r` of it) and their ends between `enter` below the ring's near face and `depth` past its far face;
    held, the ring rides on the fingertips' ends going deeper and stays where it is when they are drawn back, so
    they slide out of it. The ring's tilt is the fingertips' height difference over their spacing, atan(dz / gap);
    when one of them leaves it, the tilt they had at that moment is kept as the tilt it is let go at. Without a hold,
    a fingertip end arriving on the ring's near face (over its section, beside the shaft) pushes it deeper: a
    `Ring(roll=True)` then rolls instead of sliding (it twists). A ring already in its groove is not pushed or rolled
    (the groove's wall holds it), and a fingertip beside or inside its section does not push it."""

    grippers: tuple = ("left_gripper", "right_gripper")
    full_gap: float = 0.0939  # the pads' gap fully open (2 x FINGER_OPEN)
    tip_past: float = 0.0128  # the pad tips past the tool point (PAD_TIP_PAST_TCP)
    out: float = 0.004
    half_w: float = 0.007
    up: float = 0.002
    enter: float = 0.0005
    depth: float = 0.003
    axis_r: float = 0.006
    push_pad: float = 0.002

    def fingers(self, objects, g):
        """(tip-end centers, outer-face contact points per finger, pointing direction) of gripper g, world."""
        o = objects[g]
        rot = _rotm(o.quat_xyzw)
        tcp = np.asarray(o.pos, float)
        gap = kit.opening(objects, g) * self.full_gap
        if not math.isfinite(gap):
            gap = self.full_gap
        ends, faces = [], []
        for sg in (1.0, -1.0):
            ends.append(tcp + rot @ np.array([sg * (gap / 2 + self.out / 2), 0.0, self.tip_past]))
            faces.append([tcp + rot @ np.array([sg * (gap / 2 + self.out), w, self.tip_past - self.up])
                          for w in (-self.half_w, self.half_w)])
        return ends, faces, rot @ np.array([0.0, 0.0, 1.0])

    def loop(self, ring: Ring, fr: Frame, faces):
        """The stretched ring's inner contour (2D, shaft frame) round the fingertips and, if both are outside the
        shaft, the shaft."""
        pts = [(float(np.dot(p - fr.o, fr.e1)), float(np.dot(p - fr.o, fr.e2))) for f in faces for p in f]
        a = ring.shaft_d / 2
        if all(max(math.hypot(*q) for q in pts[2 * i:2 * i + 2]) >= a for i in (0, 1)):
            pts += [(a * math.cos(t), a * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 96, endpoint=False)]
        return hull(pts)

    def _stretch(self, ring, st, fr, objects, g):
        ends, faces, down = self.fingers(objects, g)
        held = st.engaged and st.grip is not None and not st.grip.push and st.grip_by == g
        if not held:  # take hold: both fingertips inside the ring's hole, the shaft axis between them
            for f in faces:
                for p in f:
                    if fr.radial(p) >= st.r:
                        return None
            a2 = np.array([np.dot(ends[0] - fr.o, fr.e1), np.dot(ends[0] - fr.o, fr.e2)])
            b2 = np.array([np.dot(ends[1] - fr.o, fr.e1), np.dot(ends[1] - fr.o, fr.e2)])
            d = b2 - a2
            k = min(1.0, max(0.0, float(np.dot(-a2, d) / max(float(np.dot(d, d)), 1e-12))))
            if np.linalg.norm(a2 + k * d) > self.axis_r:
                return None
        poly = self.loop(ring, fr, faces)
        r = perimeter(poly) / (2 * math.pi)
        s_end = (fr.s_of(ends[0]) + fr.s_of(ends[1])) / 2
        s = max(st.s, s_end - ring.thickness / 2)  # rides on the fingertips' ends going deeper, stays going back
        r_new = max(r, ring.free_id / 2 + st.set, span_r(ring, st.s))
        s_new = _travel(ring, st.s, s, r_new)
        near, far = s_new - ring.thickness / 2, s_new + ring.thickness / 2
        for e in ends:
            if not (near + self.enter <= fr.s_of(e) <= far + self.depth):
                return None
        tilt, ax = self.tip_tilt(fr, ends)
        line = ends[0] - ends[1]
        theta = math.atan2(float(np.dot(line, fr.e2)), float(np.dot(line, fr.e1)))
        path = tuple(poly) if r > ring.free_id / 2 + st.set else None
        return Grip(s=s, theta=theta, r=r, tilt=tilt, tilt_axis=ax, path=path)

    @staticmethod
    def tip_tilt(fr: Frame, ends):
        """(deg, world axis) the ring's plane is tilted off square by the two fingertips that hold it: atan(dz / gap),
        dz their ends' height difference along the shaft, gap their spacing across it. (The gripper's own angle does
        not tilt the ring: tilted about the line between its fingertips, both still hold it at one height.)"""
        line = ends[0] - ends[1]
        dz = float(np.dot(line, fr.u))
        lat = line - dz * fr.u
        gap = float(np.linalg.norm(lat))
        if gap < 1e-6:
            return 0.0, (1.0, 0.0, 0.0)
        ax = np.cross(lat / gap, fr.u)  # rotating the plane about this by +tilt puts tip 0's side deeper
        tilt = math.degrees(math.atan2(dz, gap))
        if tilt < 0:
            tilt, ax = -tilt, -ax
        return tilt, tuple(ax / np.linalg.norm(ax))

    press_arc: float = 60.0  # deg: a press counts within this of the cocked ring's highest point

    @staticmethod
    def high_near(ring: Ring, st: RingState) -> float:
        """s of the near face of a cocked ring's high side: it lies on the land just past the edge of the groove its
        low side went into (a ring cocks only closing within half a groove's width of its middle)."""
        i = st.groove if st.groove is not None else _snap(ring, st.s)[0]
        g = ring.grooves[i if i is not None else 0]
        return g.at - g.width / 2 - ring.thickness

    def _on_high_side(self, ring, st, fr, e) -> bool:
        """A fingertip end over a cocked ring's high side: over its section (radially) and within press_arc of its
        highest point (the side away from the deeper side, -(u x tilt_axis))."""
        if not (ring.shaft_d / 2 - self.push_pad <= fr.radial(e) <= ring.shaft_d / 2 + ring.width + self.push_pad):
            return False
        hi = -np.cross(fr.u, np.asarray(st.tilt_axis, float))
        d = np.asarray(e, float) - fr.o
        d = d - np.dot(d, fr.u) * fr.u
        n = np.linalg.norm(d) * np.linalg.norm(hi)
        return n > 1e-9 and float(np.dot(d, hi)) / n >= math.cos(math.radians(self.press_arc))

    def _press(self, ring, st, fr, objects, g):
        """A fingertip end that came down onto a cocked ring's high side and has pushed it half its section (1.5 mm
        for a 3 mm ring) down past the land's edge: past that, the ring's own tension pulls it into the groove."""
        ends, _, _ = self.fingers(objects, g)
        need = self.high_near(ring, st) + ring.thickness / 2
        for i, e in enumerate(ends):
            if st.above.get((g, i), False) and self._on_high_side(ring, st, fr, e) and fr.s_of(e) >= need:
                return Grip(s=st.s, theta=st.theta, push=True, press=ring.thickness / 2)
        return None

    def _push(self, ring, st, fr, objects, g):
        ends, _, _ = self.fingers(objects, g)
        if st.cocked:  # one side in the groove, the other on the land: only a press on the high side moves it
            return self._press(ring, st, fr, objects, g)
        if st.groove is not None:
            return None  # in its groove: it cannot roll any deeper (the groove's wall holds it), so nothing pushes it
        r = max(ring.free_id / 2 + st.set, span_r(ring, st.s))  # where its section lies once nothing spreads it
        lo = max(r, ring.shaft_d / 2) - self.push_pad
        hi = r + ring.width + self.push_pad
        near = st.s - ring.thickness / 2
        deepest = None
        for i, e in enumerate(ends):
            z = fr.s_of(e)
            # a fingertip end that was on the near side of the ring's near face last step and is now at or past
            # it, over its section: it pushes the ring ahead of it. A tip already beside or inside the section (slid
            # in sideways, or left there) did not arrive from the face and does not push it.
            if lo <= fr.radial(e) <= hi and z >= near - 0.0005 and st.above.get((g, i), False):
                deepest = z if deepest is None else max(deepest, z)
        if deepest is None:
            return None
        return Grip(s=max(st.s, deepest + ring.thickness / 2), theta=st.theta, push=True)

    def grip(self, ring: Ring, st: RingState, fr: Frame, objects):
        return self._grip(ring, st, fr, objects, [g for g in self.grippers if g in objects])

    def after(self, ring: Ring, st: RingState, fr: Frame, objects):
        """After the step: where each fingertip end is against the ring's (moved) near face, for next step's push."""
        if st.cocked:
            # on a cocked ring the press is on its high side: a tip counts once it has come down onto that side's
            # near face from above, and stays counted while it stays over that side
            hn = self.high_near(ring, st)
            st.above = {(g, i): fr.s_of(e) <= hn + 0.0005
                        or (st.above.get((g, i), False) and self._on_high_side(ring, st, fr, e))
                        for g in self.grippers if g in objects for i, e in enumerate(self.fingers(objects, g)[0])}
            return
        near = st.s - ring.thickness / 2
        st.above = {(g, i): fr.s_of(e) <= near + 0.0005
                    for g in self.grippers if g in objects for i, e in enumerate(self.fingers(objects, g)[0])}

    def _grip(self, ring: Ring, st: RingState, fr: Frame, objects, gs):
        if st.engaged and st.grip is not None and not st.grip.push and st.grip_by in gs:  # keep the hand that holds it
            gs = [st.grip_by] + [g for g in gs if g != st.grip_by]
        for g in gs:
            out = self._stretch(ring, st, fr, objects, g)
            if out is not None:
                st.grip_by = g
                return out
        if st.engaged and st.grip is not None and not st.grip.push and st.grip_by in gs:
            # it was held and is not now: one fingertip (or both) left it. The ring is let go at the tilt the two
            # tips had at that moment (the one still in it holds that side where it is), so keep that as its tilt
            # for the landing check; a square draw-out leaves it ~0.
            ends, _, _ = self.fingers(objects, st.grip_by)
            tilt, ax = self.tip_tilt(fr, ends)
            if tilt > abs(st.tilt):
                st.tilt, st.tilt_axis = tilt, ax
                st.max_tilt = max(st.max_tilt, tilt)
        for g in gs:
            out = self._push(ring, st, fr, objects, g)
            if out is not None:
                st.grip_by = g
                return out
        st.grip_by = None
        return None


@dataclass(frozen=True)
class HandleSpring:
    """A yamkit law for a tool handle: a return spring of `k` (N/m or N*m/rad) toward `rest`, which may lie past the
    joint's stop to preload it (rest = stop - preload / k), with implicit damping `damp`."""

    rest: float
    k: float
    damp: float = 1.0

    def drive(self, key, q, qd, J, mem):
        return self.rest, self.k, 0.0, self.damp


# -- the ring's joints ------------------------------------------------------------------------------


SHOW_STEPS = 12  # control steps the placeholder mesh stands in for the markers (they show from about frame 5)


def _band_mesh(ring: Ring, r: float, n: int = 48) -> kit.Mesh:
    """The ring as a mesh at inner radius r in the shaft frame at s = 0 (the link's frame), split at lug_angle."""
    u, e1, e2 = _local_frame(ring)
    rm = r + ring.width / 2
    gap = 0.004 / rm if ring.lug_gap > 0 else 0.0
    angs = np.linspace(ring.lug_angle + gap / 2, ring.lug_angle + 2 * math.pi - gap / 2, n)
    path = [rm * (math.cos(a) * e1 + math.sin(a) * e2) for a in angs]
    if ring.lug_gap == 0:
        path.append(path[0])
    return kit.band(path, ring.thickness, ring.width, ring.look, normal=tuple(-u))


def links(ring: Ring, mass: float = 0.05) -> tuple:
    """The ring's joints for its fixture mechanism (see the module docstring): invisible, collider-less, gravity off;
    the kit writes their positions every step. A fourth link, "<name>_shown", carries a mesh of the ring at its start
    pose for the first SHOW_STEPS control steps (new marker prims take a few rendered frames to appear, and setup
    shots are taken at frame 2); the kit then moves it to s = `hide` (choose a point inside the
    table or the fixture)."""
    k, sg = _letter(ring.axis)
    _, e1, _ = _local_frame(ring)
    k1 = "XYZ"[int(np.argmax(np.abs(e1)))]
    s_hi = _s_max(ring)
    lo, hi = sorted((sg * ring.s0, sg * s_hi))
    common = dict(anchor=tuple(ring.end), mass=mass, damping=0.0, floating=True)
    return (kit.Link(f"{ring.name}_axial", "prismatic", k, lower=lo - 0.001, upper=hi + 0.001, start=sg * ring.s0,
                     **common),
            kit.Link(f"{ring.name}_turn", "revolute", k, lower=-math.pi - 0.01, upper=math.pi + 0.01, start=0.0,
                     **common),
            kit.Link(ring.name, "prismatic", k1, lower=-0.001, upper=0.02, start=0.0, **common),
            kit.Link(f"{ring.name}_shown", "prismatic", k, lower=min(lo, sg * ring.hide) - 0.001,
                     upper=max(hi, sg * ring.hide) + 0.001, start=sg * ring.s0,
                     meshes=(_band_mesh(ring, max(ring.free_id / 2, span_r(ring, ring.s0))),), **common))


def _joint_values(ring: Ring, st: RingState) -> dict:
    _, sg = _letter(ring.axis)
    shown = ring.s0 if st.steps <= SHOW_STEPS else ring.hide
    return {f"{ring.name}_axial": sg * st.s, f"{ring.name}_turn": -sg * _wrap(st.theta),
            ring.name: st.r - ring.free_id / 2, f"{ring.name}_shown": sg * shown}


def _write_joints(base) -> None:
    for key, att in _RINGS.items():
        try:
            art = base.scene[att.mech]
        except KeyError:
            continue
        names = list(art.joint_names)
        for jn, v in _joint_values(att.ring, state(key)).items():
            if f"{jn}_joint" in names:
                j = names.index(f"{jn}_joint")
                kit._put_joint(art, j, v)


# -- USD: inertia for the ring's links, the root on a free tool's base ---------------------------------

_orig_mech_usda = kit._mech_usda


def _mech_usda(mech):
    text = _orig_mech_usda(mech)
    ring_links = {f"{a.ring.name}{x}" for a in _RINGS.values() if a.mech == mech.name for x in ("_axial", "_turn", "", "_shown")}
    for name in ring_links:
        head = f'    def Xform "{name}" ('
        if head not in text:
            continue
        i = text.index(head)
        m = text.index("float physics:mass", i)
        eol = text.index("\n", m) + 1
        text = text[:eol] + "        float3 physics:diagonalInertia = (1e-06, 1e-06, 1e-06)\n" \
                            "        point3f physics:centerOfMass = (0, 0, 0)\n" + text[eol:]
    if mech.name in _FREE and mech.mass is not None:
        root = 'def Xform "Root" (\n    prepend apiSchemas = ["PhysicsArticulationRootAPI", "PhysxArticulationAPI"]\n)\n{\n'
        if root not in text:
            raise RuntimeError("ringkit: unexpected yamkit USD layout for the root")
        text = text.replace(root, 'def Xform "Root"\n{\n', 1)
        old = ('    def Xform "base" (\n        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", '
               '"PhysxRigidBodyAPI"]\n    )\n    {\n')
        if old not in text:
            raise RuntimeError("ringkit: unexpected yamkit USD layout for the base")
        settings = [ln for ln in text.splitlines(True) if ln.startswith("    ") and "physxArticulation:" in ln
                    and not ln.startswith("        ")]
        text = "".join(ln for ln in text.splitlines(True) if ln not in settings or ln.startswith("        "))
        new = ('    def Xform "base" (\n        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", '
               '"PhysxRigidBodyAPI", "PhysicsArticulationRootAPI", "PhysxArticulationAPI"]\n    )\n    {\n'
               + "".join("    " + ln for ln in settings))
        text = text.replace(old, new, 1)
        com = _FREE[mech.name]
        if com is not None:
            i = text.index('    def Xform "base" (')
            m = text.index("float physics:mass", i)
            eol = text.index("\n", m) + 1
            text = text[:eol] + f"        point3f physics:centerOfMass = {kit._fmt(com)}\n" + text[eol:]
    return text


kit._mech_usda = _mech_usda


# -- visuals ------------------------------------------------------------------------------------------

_VIS = {"markers": None, "failed": False}
SEGMENTS = 40


def _note(msg: str) -> None:
    """A visuals problem: to stderr, and to the file named by RINGKIT_LOG if set (run scripts filter stderr)."""
    print(msg, file=sys.stderr, flush=True)
    if os.environ.get("RINGKIT_LOG"):
        with open(os.environ["RINGKIT_LOG"], "a") as f:
            f.write(msg + "\n")


def _markers():
    if _VIS["markers"] is not None or _VIS["failed"]:
        return _VIS["markers"]
    try:
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        protos, index = {}, {}
        for key, att in _RINGS.items():
            lk = att.ring.look
            index[key] = len(protos)
            protos[f"ring{len(protos)}"] = sim_utils.CuboidCfg(
                size=(1.0, 1.0, 1.0), visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=tuple(kit._linear(c) for c in lk.color), roughness=lk.roughness,
                    metallic=lk.metallic))
        _VIS["markers"] = VisualizationMarkers(VisualizationMarkersCfg(prim_path="/World/Visuals/ringkit",
                                                                       markers=protos))
        _VIS["index"] = index
    except Exception as e:  # visuals are optional: scoring never depends on them
        _note(f"ringkit: no ring visuals ({e!r})")
        _VIS["failed"] = True
    return _VIS["markers"]


def _axis_angle(axis, ang) -> np.ndarray:
    a = np.asarray(axis, float)
    a /= np.linalg.norm(a)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(ang) * k + (1 - math.cos(ang)) * k @ k


def _quat_xyzw(R) -> tuple:
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    if w > 1e-6:
        return ((R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w), w)
    i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(max(0.0, 1 + R[i, i] - R[j, j] - R[k, k])) * 2
    q = [0.0, 0.0, 0.0, (R[k, j] - R[j, k]) / s]
    q[i], q[j], q[k] = s / 4, (R[j, i] + R[i, j]) / s, (R[k, i] + R[i, k]) / s
    return tuple(q)


def pieces(key: str, objects) -> list:
    """The drawn ring as (center, rotation matrix, size) boxes in world coordinates (relative to the env origin):
    band segments round the arc and, for a split ring, a lug ear at each end."""
    att = _RINGS[key]
    ring, st = att.ring, state(key)
    fr = frame(objects, att.mech, ring)
    rm = st.r + ring.width / 2
    split = ring.lug_gap > 0
    gap_vis = (0.004 + 2 * math.pi * (st.r - ring.free_id / 2)) if split else 0.0
    half = (gap_vis / 2) / rm
    a0, a1 = st.theta + ring.lug_angle + half, st.theta + ring.lug_angle + 2 * math.pi - half
    center = fr.o + st.s * fr.u
    tilt = _axis_angle(st.tilt_axis, math.radians(st.tilt)) if st.tilt else np.eye(3)
    dz = (math.tan(math.radians(st.twist)) * (ring.lug_gap or 0.01)) if st.twist else 0.0
    out = []
    n = SEGMENTS

    def put(ang, rad, f, size):  # size: (radial, along the arc, along the shaft)
        radial = math.cos(ang) * fr.e1 + math.sin(ang) * fr.e2
        tang = np.cross(-fr.u, radial)
        p = center + rad * radial + (f - 0.5) * dz * fr.u
        R = np.column_stack([radial, tang, -fr.u])  # right-handed: radial x tang = outward
        out.append((center + tilt @ (p - center), tilt @ R, size))

    if st.path is not None and len(st.path) > 2 and not split:  # stretched round fingertips: draw that loop
        poly = np.asarray(st.path, float)
        seg = np.roll(poly, -1, axis=0) - poly
        cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(seg, axis=1))])
        total = cum[-1]
        for k in range(n):
            pts = []
            for f in (k / n, (k + 1) / n):
                d = f * total
                i = min(int(np.searchsorted(cum, d, side="right")) - 1, len(poly) - 1)
                pts.append(poly[i] + seg[i] * ((d - cum[i]) / max(cum[i + 1] - cum[i], 1e-12)))
            t2 = pts[1] - pts[0]
            ln = float(np.linalg.norm(t2))
            if ln < 1e-9:
                continue
            t2 = t2 / ln
            n2 = np.array([t2[1], -t2[0]])  # outward for a counterclockwise loop
            m2 = (pts[0] + pts[1]) / 2 + n2 * ring.width / 2
            radial = n2[0] * fr.e1 + n2[1] * fr.e2
            tang = np.cross(-fr.u, radial)
            p = center + m2[0] * fr.e1 + m2[1] * fr.e2
            out.append((center + tilt @ (p - center), tilt @ np.column_stack([radial, tang, -fr.u]),
                        (ring.width, ln * 1.15, ring.thickness)))
        return out
    for k in range(n):
        f = (k + 0.5) / n
        ang = a0 + f * (a1 - a0)
        put(ang, rm, f, (ring.width, rm * (a1 - a0) / n * 1.15, ring.thickness))
    if split:
        ear = ring.lug_out + 0.0025
        for f, ang in ((0.0, a0 + 0.0025 / rm), (1.0, a1 - 0.0025 / rm)):
            put(ang, st.r + ear / 2, f, (ear, 0.005, ring.thickness))
    return out


def _draw(objects, origin) -> None:
    m = _markers()
    if m is None:
        return
    tr, qs, sc, ids = [], [], [], []
    for key in _RINGS:
        if _RINGS[key].mech not in objects:
            continue
        for c, R, size in pieces(key, objects):
            tr.append(tuple(c + origin)), qs.append(_quat_xyzw(R)), sc.append(size), ids.append(_VIS["index"][key])
    if not tr:
        return
    if os.environ.get("RINGKIT_LOG") and not _VIS.get("noted"):
        _VIS["noted"] = True
        _note(f"ringkit: drawing {len(tr)} pieces, first at {np.round(tr[0], 4).tolist()} size {sc[0]}")
    # this Isaac Lab takes (x, y, z, w) quaternions, as it reports them (pourkit's markers verified it)
    m.visualize(translations=np.array(tr, float), orientations=np.array(qs, float), scales=np.array(sc, float),
                marker_indices=np.array(ids, int))


# -- hook into yam_arena.scene (after yamkit's) ------------------------------------------------------


def _patch(scene) -> None:
    if getattr(scene, "_ringkit", False):
        return
    scene._ringkit = True
    inner = scene.object_states
    last = {"key": None}

    def object_states(env, names):
        out = inner(env, names)
        base = env.unwrapped
        buf = getattr(base, "episode_length_buf", None)
        key = (kit.episode(), int(buf[0].item()) if buf is not None else None)
        if key[1] is None or key != last["key"]:  # once per control step
            last["key"] = key
            update(out)
            _write_joints(base)
            if os.environ.get("RINGKIT_DEBUG"):
                for k in _RINGS:
                    st = state(k)
                    _note(f"ringkit {k} s={st.s * 1000:.2f}mm id={2 * st.r * 1000:.2f}mm "
                          f"th={math.degrees(st.theta):.1f} eng={st.engaged} g={st.groove} cock={st.cocked} "
                          f"tilt={st.tilt:.1f} tw={st.twist:.1f} turned={st.turned:.1f} set={st.set * 1000:.2f}")
            try:
                _draw(out, base.scene.env_origins[0].detach().cpu().numpy())
            except Exception as e:  # noqa: BLE001  (never let the visuals break a run)
                if not _VIS["failed"]:
                    import traceback

                    _note(f"ringkit: visuals off ({e!r})\n{traceback.format_exc()}")
                _VIS["failed"] = True
        return out

    scene.object_states = object_states


class _Hook(kit._Hook):
    def exec_module(self, module):
        self._inner.exec_module(module)
        kit._patch(module)
        _patch(module)


if "yam_arena.scene" in sys.modules:
    _patch(sys.modules["yam_arena.scene"])
else:  # replace yamkit's import hook with one that also adds the ring step after yamkit's patch
    sys.meta_path[:] = [f for f in sys.meta_path if not isinstance(f, kit._Hook)]
    sys.meta_path.insert(0, _Hook())


# -- scorer readouts --------------------------------------------------------------------------------


def position(objects, key: str) -> float:
    """The ring's mid-plane along the shaft, m from the end face (into the shaft)."""
    return state(key).s


def groove(objects, key: str):
    """Index of the groove the ring's section lies in now (closed or not), or None."""
    return state(key).groove


def seated(objects, key: str) -> bool:
    """In a groove, closed onto its bottom, not cocked and not held by a tool."""
    st, ring = state(key), _RINGS[key].ring
    return (st.groove is not None and not st.cocked and not st.engaged
            and st.r <= ring.grooves[st.groove].diameter / 2 + ring.seat_tol)


def engaged(objects, key: str) -> bool:
    """A tool holds the ring now."""
    return state(key).engaged


def inner_d(objects, key: str) -> float:
    return 2 * state(key).r


def max_d(objects, key: str) -> float:
    """The largest inner diameter the ring reached this trial (m)."""
    return 2 * state(key).max_r


def overspread(objects, key: str) -> bool:
    """Spread past its elastic limit at any point this trial (irreversible: it has taken a set)."""
    ring = _RINGS[key].ring
    return ring.limit_id is not None and state(key).max_r > ring.limit_id / 2 + 1e-9


def set_mm(objects, key: str) -> float:
    return 1000 * state(key).set


def tilt_deg(objects, key: str) -> float:
    return abs(state(key).tilt)


def max_tilt_deg(objects, key: str) -> float:
    return state(key).max_tilt


def max_twist_deg(objects, key: str) -> float:
    return state(key).max_twist


def twist_deg(objects, key: str) -> float:
    """An elastomer ring's twist now: how far its section has rolled about its own centerline (deg). Spreading it
    clear of the shaft lets it spring back to 0, so this is the twist it is left with."""
    return state(key).roll


def max_roll_deg(objects, key: str) -> float:
    return state(key).max_roll


def cocked(objects, key: str) -> bool:
    return state(key).cocked


def turned_deg(objects, key: str) -> float:
    """Degrees the ring turned while closed in a groove and not cocked, this trial (a seating check)."""
    return state(key).turned
