"""Give a 1 mL intramuscular injection in a strap-on deltoid trainer with a retractable-needle safety syringe.

A strap-on IM injection trainer is strapped to a vertical 4 in PVC pipe on a plywood stand in front of the robot, as on a
left upper arm; a plywood spacer under the stand sets its height (0, 18.3 or 36.6 mm, by trial). A 3 mL retractable
safety syringe with a fixed 25 G x 1 in needle, holding 1.0 mL, lies capped on a rest. The robot uncaps it, drops the cap
in a paper waste cup, finds the deltoid site 25-50 mm below the shoulder's bony edge, puts the needle in square to the
skin up to the hub, keeps it in the muscle and still while the other hand pushes the 1 mL steadily (about 10 s) and on
past the end so the needle retracts, then drops the syringe in a sharps container.

    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=im_injection.py ...   the scripted demo (demo.py)
    python -m pytest -q test_im_injection.py                                                    scorer, layout, reach; no simulator

Simulation: a simulator cannot puncture or flow, so the needle in the tissue is a rule (`Trial` below) with a D6 joint
between the trainer and the syringe, authored off; `score()` reads the rule's record.
"""

import math
import os
import pathlib
import sys
from collections import deque
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # helpers beside this file

SCENE = "station"
INSTRUCTION = ("Give the 1 mL dose in the syringe as an intramuscular injection in the training arm's deltoid, pushing the plunger slowly and steadily. "
               "Push it all the way so the needle retracts, then put the syringe in the sharps container and the needle cap in the waste cup.")
SUCCESS_HOLD_S = 0.0  # Trial counts its own 1 s hold: score() and success() agree
SCORE_OVER = "end"
CONTROL_DT = 1.0 / 20.0
HOLD_STEPS = 20  # the 1.0 state held 1 s

# -- the scoring table (task.json `scoring`) ----------------------------------------------------------------------------
D_FULL, D_05 = (0.025, 0.050), (0.020, 0.055)  # m below the acromion's lower edge (on the y = 0 line)
L_FULL, L_05 = 0.010, 0.015  # m from the midline plane y = 0
A_FULL, A_075, A_05 = 10.0, 15.0, 20.0  # deg: the worse of pitch and yaw against the skin's local normal
D_HIGH = 0.020  # m: a puncture closer than this below the acromion is a 0
IN_MUSCLE_GAP, FULL_DEPTH_GAP = 0.005, 0.001  # m hub-to-skin gap: in muscle; full depth "the hub touches the skin in the
# side view" (the simulation's reading: within 1 mm)
GRAD_ML = 0.1  # mL: the finest graduation printed on the VanishPoint 3 mL barrel (0.1 mL ticks, longer every 0.5 mL)
# S: the first frame the stopper edge has reached the printed graduation 0.1 mL below its reading at the first
# in-muscle frame, that reading taken to the nearest graduation (a graduation is reached when the edge is on or past it:
# no tolerance). The edge must travel at least half a graduation (0.05 mL, 0.85 mm) to reach it, so S is the push's
# start and never the insertion, however close to a line the fill sits (`s_line`).
STILL_DEG, STILL_M = 5.0, 0.003  # needle still: pitch and yaw within 5 deg, hub within 3 mm in x, y and z of its pose at S
HIDDEN_MAX_S = 1.0  # s of hidden frames (cumulative, S to retraction) above which needle-still is not credited
DOSE_FIRST_ML = 0.9  # mL: the edge at or above this line at the first in-muscle frame (full dose)
PUSHED_05_ML = 0.5  # mL the edge moved while in muscle (0.5 tier)
PRE_PUSH_ML = 0.2  # mL pushed before the first puncture: a 0
LINES = (0.8, 0.6, 0.4, 0.2, 0.0)  # mL lines for the steady push
# Steady push (README, "The steady-push rule"): every number from the taught rate of about 10 s per mL (2 s per 0.2 mL)
STEP_ML, STEP_S = 0.2, (1.0, 4.0)  # 1-4 s per 0.2 mL (half to twice the taught rate); the first interval prorated by volume
LINE_TOL = 0.0  # m: a line is reached when the edge is on or past it (as a person reads it against the graduations);
# the push's timing ends on the frame the edge reaches the 0 line (the detent push and the retraction after it are not timed)
STOP_S = 3.0  # a stop: from S to the 0 line, 3 s or more in which the edge reaches no new printed graduation (GRAD_ML).
# From the rates: at the slowest the band allows (4 s per 0.2 mL) a graduation is reached every 2 s, at the taught rate
# every 1 s, so 3 s (1.5 times the slowest spacing) flags only a stopped plunger; within one 4 s interval it leaves at
# most a 3 s pause unflagged, against the 0.2 mL lines' 4 s
THUMB_ON_N, THUMB_GAP_S = 0.2, 0.1  # thumb on, from S to the 0 line (simulation's reading of "no visible 2 mm gap"):
# the robot's contact load on the plunger under 0.2 N (the sensing floor the phantom-contact gate uses) for 0.1 s or
# more (3 frames of the 30 fps side video: the shortest gap the video shows on consecutive frames) is the thumb off
GRIP_CLEAR = 0.05  # m: both grippers this far from the syringe and the cap for "stowed"
ML_M = 0.0169  # m of stopper travel per mL (8.66 mm bore: 1.69 mm per 0.1 mL)


def score(objects) -> float:
    """The trial's score now (inspect-robots' task_score reads it at the end): see `tier`."""
    return TRIAL.score


def success(objects) -> bool:
    return TRIAL.full


def tier(zero: bool, punctures: int, D: float, L: float, A: float, full_depth: bool, continuous: bool, still: bool,
         full_dose: bool, steady: bool, stowed: bool, held: bool, pushed_ml: float, any_025: bool) -> float:
    """The scoring table. D, L, A: the first puncture's depth below the acromion (m), distance from the midline (m) and
    angle (deg, at its first in-muscle frame); pushed_ml: mL the edge moved while in muscle; any_025: some puncture was in
    muscle within the 0.5 site and A <= 20 deg. The other flags are as task.json defines them."""
    if zero:
        return 0.0
    one = punctures == 1
    site1 = D_FULL[0] - 1e-9 <= D <= D_FULL[1] + 1e-9 and L <= L_FULL + 1e-9
    site05 = D_05[0] - 1e-9 <= D <= D_05[1] + 1e-9 and L <= L_05 + 1e-9
    if one and site1 and A <= A_FULL + 1e-9 and full_depth and continuous and still and full_dose and steady and stowed and held:
        return 1.0
    if one and site1 and A <= A_075 + 1e-9 and continuous and full_dose:
        return 0.75
    if one and site05 and A <= A_05 + 1e-9 and pushed_ml >= PUSHED_05_ML - 1e-9:
        return 0.5
    if any_025:
        return 0.25
    return 0.0


# -- the scene's geometry (world m: x forward, y left, z up; the setup's cm / 100, heights + TOP) ------------------------
TOP = 0.7475  # station.TABLE_TOP
HEIGHTS = (0.0, 0.0183, 0.0366)  # spacer: none, one or two layers of 3/4 in plywood (18.3 mm measured); trial k uses k mod 3
HEIGHT = int(os.environ.get("INJ_HEIGHT", "0")) % 3
TRAINER_T = float(os.environ.get("INJ_TRAINER_T", "0.050"))  # m: trainer thickness at the site (TO MEASURE, 43-50 mm)
PLY_T = 0.0183
PIPE_X, PIPE_R, PIPE_L = 0.360, 0.05715, 0.300  # 4 in Schedule 40 PVC: 114.3 mm OD
PIPE_FACE = PIPE_X - PIPE_R  # 0.30285: the pipe's -x face
SHEET = 0.001  # silicone anti-slip sheet
EDGE_Z0 = 0.250  # m above the table: the acromion's lower edge with no spacer
SITE_D = 0.0375  # m below it: the middle of the 25-50 mm window
TRAINER_H, TRAINER_W, TRAINER_ABOVE = 0.175, 0.085, 0.030  # 17.5 x 8.5 cm; its top 3 cm above the acromion's edge
CENTRE_HALF = 0.025  # m: the flat lateral face, |y| <= 25 mm; beyond it the sides fall back at BEVEL_DEG
BEVEL_DEG = 30.0
SHOULDER_BACK = 0.006  # m: the shoulder cap above the acromion's edge stands this far back from the deltoid's face


@dataclass(frozen=True)
class Geometry:
    """The trainer's skin as the rule sees it, at one spacer height and thickness. `normal` is the sim mesh's local
    outward normal (task.json: A is measured against the trainer's local surface normal at the site, TO MEASURE)."""

    h: float = HEIGHTS[HEIGHT]
    t: float = TRAINER_T

    @property
    def skin_x(self) -> float:
        return PIPE_FACE - SHEET - self.t

    @property
    def z_edge(self) -> float:
        return TOP + EDGE_Z0 + self.h

    @property
    def z_top(self) -> float:
        return self.z_edge + TRAINER_ABOVE

    @property
    def z_bottom(self) -> float:
        return self.z_top - TRAINER_H

    @property
    def site(self) -> np.ndarray:
        return np.array([self.skin_x, 0.0, self.z_edge - SITE_D])

    def surface_x(self, y: float, z: float) -> float:
        if z > self.z_edge:
            return self.skin_x + SHOULDER_BACK
        return self.skin_x + max(0.0, abs(y) - CENTRE_HALF) * math.tan(math.radians(BEVEL_DEG))

    def normal(self, y: float, z: float) -> np.ndarray:
        if abs(y) <= CENTRE_HALF or z > self.z_edge:
            return np.array([-1.0, 0.0, 0.0])
        b = math.radians(BEVEL_DEG)
        return np.array([-math.cos(b), math.copysign(math.sin(b), y), 0.0])

    def on_skin(self, p, tol: float) -> bool:
        x, y, z = (float(v) for v in p)
        return abs(y) <= TRAINER_W / 2 and self.z_bottom <= z <= self.z_top and abs(x - self.surface_x(y, z)) <= tol


GEOM = Geometry()


# -- the syringe (its frame: x along the barrel toward the needle, origin at the hub, the front of the barrel's nose) ----
NEEDLE_L = 0.0254  # 25 G x 1 in
NOSE = (-0.010, 0.0)
BARREL = (-0.0695, -0.010)
BARREL_HALF = 0.0054  # the barrel's collider is 10.8 mm square: the pads grip its flat +-y faces
FLANGE = (-0.0715, -0.0695)  # 25 x 7 mm finger flange, long axis vertical (it passes between the fingers)
LINE0 = -0.010  # the 0 mL line: the stopper's front edge at the nose
EDGE0 = LINE0 - 1.0 * ML_M  # -0.0269: the edge at 1.0 mL
BAND = LINE0 - 2.0 * ML_M  # -0.0438: 3 mm marker bands on the top and +y faces at the 2.0 mL line
ZERO_TRAVEL = 1.0 * ML_M  # m of travel to the 0 line (16.9 mm)
DETENT_L, DETENT_F = 0.002, 10.0  # the retraction: a stop at the 0 line that gives at 10 N of thumb load, then 2 mm on
RETRACT_AT = ZERO_TRAVEL + DETENT_L  # 18.9 mm
TRAVEL_MAX = RETRACT_AT + 0.0005  # the plunger's stop
ROD_BACK = 0.068  # the thumb rest's back face, behind the stopper's front edge
CAP = (-0.008, 0.034)
CAP_HALF = 0.004
GRASP_R = -0.042  # where the right hand's TCP holds the barrel (pads about 30-52 mm behind the hub; spike 4: at -44 mm the
# fingers met the thumb rest near the end of its travel, 1.2 mm clear on the meshes)
GRASP_CAP = 0.016  # where the left hand's TCP takes the cap

# -- the rule's forces (provisional until measured: README, "TO MEASURE") -------------------------------------------
ENGAGE_F, ENGAGE_DEG = 1.0, 20.0  # N the tip presses the skin, within this of its local normal: the needle goes in
K_SENSE = 500.0  # N/m: the needle's sensing spring (1 N at 2 mm)
TISSUE_F, TISSUE_V = 0.8, 0.002  # N stick-slip along the needle, m/s to full friction
K_LAT, C_LAT = 300.0, 2.0  # N/m, N s/m: the tissue holds the needle's line
PLUNGER_JIG = [0.0]  # N along the axis on the plunger: the spike's damping measurement sets it (0 in every scored run)
# The stopper's resistance has two parts, both acting between the plunger and the barrel.
# (1) Glide: F_GLIDE of Coulomb friction along the barrel's axis, applied every physics step to the plunger and, equal and
# opposite, to the barrel, from the stopper's breakaway (J_stick released) on. It is elasto-plastic (a stopper's rubber
# shears GLIDE_Z0 before it slides): F = -F_GLIDE z / GLIDE_Z0, z the slide's displacement since it last slid, held
# within +-GLIDE_Z0 (z += the slide's per-step displacement, from its position, then clipped). Sliding either way it is
# a constant F_GLIDE against the motion, with no velocity feedback; at rest it can only spring the stopper back by at
# most GLIDE_Z0. Its stiffness F_GLIDE / GLIDE_Z0 = 4,000 N/m is explicit, against the joint's implicit FLOW_C:
# k dt / c = 0.02 (stable well under 1). A velocity law F = -F_GLIDE tanh(v / V_EPS) is not stable explicit at 240 Hz.
# (2) Flow: see FLOW_C below (the joint's damping, integrated by the solver).
F_GLIDE = 2.0  # N: the stopper's glide once broken loose (above a plain 1 mL syringe's measured 0.1-0.7 N: README; the breakaway is BREAKAWAY_F)
GLIDE_Z0 = 0.0005  # m of presliding (the stopper's rubber) before the full F_GLIDE
# The liquid through the needle: what makes a real push slow. Hagen-Poiseuille for water (1.0 mPa s at 20 C) through the
# 25 G x 1 in needle (bore 0.26 mm nominal, 25.4 mm long) from a barrel of 59.2 mm2 (1.69 mm per 0.1 mL): 226 kPa per
# mL/s, so 13.4 N per mL/s on the plunger: 1.3 N at the taught 0.1 mL/s, FLOW_C = 790 N s/m of stopper speed (ISO 9626
# bores for 25 G, 0.232-0.292 mm, give 500-1,250 N s/m: the purchased unit's push force is measured before trials).
# It is J_plunger's zero-velocity damping drive. The barrel and the plunger are one reduced-coordinate articulation (the
# barrel its floating root link, J_plunger its only joint), so the solver integrates that damping exactly: measured
# 790-791 N s/m on the GPU with the barrel held by contact. As two free bodies on a maximal-coordinate joint the same
# drive acted at about 63 N s/m whenever the barrel was held by contact (a gripper, its rest), and the glide's force
# pair cancelled; with next to no resistance the dose ran out in about 1 s. A drive cannot be changed at run time, so
# the damping also acts over the detent's 2 mm past the 0 line and on the retracted plunger (where no liquid is left).
FLOW_C = 790.0  # N s/m
GLIDE_TAU = 0.0075  # s: the low-pass on the slide's velocity (from its position) the trace and the spike's sub-gate read
GLIDE_V_SLIDE = 0.0005  # m/s: the filtered slide velocity above which the stopper counts as sliding (the sub-gate's mean)
BREAKAWAY_F = 3.0  # N of thumb load on the thumb rest that breaks the stopper loose (static friction; TO MEASURE)
ZERO_TOL = 0.0003  # m: the stopper counts as on a stop within this
# The stops along the stopper's slide the rule releases only to the robot's measured contact load on the plunger, with
# the stopper on them: (joint, travel at the stop, the load that passes it). Before them J_stick holds it at 1.0 mL until
# BREAKAWAY_F. Spike 4: four more stops (one short of each 0.2 mL line) jammed the slide at the right hand's pinch (seven
# joints on one pair of bodies), so only the 0 line's stop is kept.
STOPS = (("J_zero", ZERO_TRAVEL, DETENT_F),)
CAP_K, CAP_POP = 2140.0, 0.0028  # N/m on the cap's fit: it comes off at 6 N (2.8 mm)
NEEDLE_BENT_DEG = 10.0  # the needle bent: its axis more than this off the barrel's (an outcome, judged after uncapping)
NEEDLE_PUSH, CARRIER_RETURN, RETRACT_PUSH = 0.1, 0.3, 0.3  # N


def _unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _deg(a, b) -> float:
    return math.degrees(math.acos(float(np.clip(np.dot(_unit(a), _unit(b)), -1.0, 1.0))))


def pitch_yaw(axis, n_out) -> tuple[float, float]:
    """The syringe's pitch (side view) and yaw (overhead) against the skin's inward normal -n_out, deg."""
    e1 = -_unit(n_out)
    ez = _unit(np.array([0.0, 0.0, 1.0]) - e1[2] * e1)
    ey = np.cross(ez, e1)
    a = _unit(axis)
    return math.degrees(math.atan2(float(a @ ez), float(a @ e1))), math.degrees(math.atan2(float(a @ ey), float(a @ e1)))


def angle_A(axis, n_out) -> float:
    p, y = pitch_yaw(axis, n_out)
    return max(abs(p), abs(y))


def grad_index(vol: float) -> int:
    """The lowest printed graduation (in GRAD_ML steps) the edge has reached at volume `vol`: on or past it."""
    return int(math.ceil(vol / GRAD_ML - 1e-6))


def grad_below(vol: float) -> float:
    """The first printed graduation below `vol` (strictly; on a graduation, the next one down)."""
    return (math.ceil(vol / GRAD_ML - 1e-6) - 1) * GRAD_ML


def s_line(vol_first: float) -> float:
    """The graduation whose reaching is S: one graduation (GRAD_ML) below the reading at the first in-muscle frame, that
    reading taken to the nearest printed graduation (midway reads up). A 1.0 mL fill that reads 1.0002 or 0.98 mL gives
    the 0.9 line, so the edge must travel at least half a graduation before S: jitter about a line cannot fire it."""
    return (math.floor(vol_first / GRAD_ML + 0.5 + 1e-6) - 1) * GRAD_ML


def volume(travel: float) -> float:
    """mL left in front of the stopper's edge after `travel` m."""
    return 1.0 - travel / ML_M


@dataclass
class Puncture:
    entry: np.ndarray  # world point on the skin
    n_out: np.ndarray  # the skin's outward normal there
    D: float  # m below the acromion's edge
    L: float  # m from y = 0
    engaged: bool = False  # within 20 deg: the needle went in
    A: float | None = None  # deg, at the first in-muscle frame
    travel_first: float | None = None  # m of stopper travel at the first in-muscle frame
    in_muscle_ever: bool = False
    full_depth: bool = False
    travel_hub: float | None = None  # m of stopper travel when the hub first touched the skin (full depth): S's start


@dataclass
class Sample:
    """One physics step's reading (every 1/240 s). Positions world, m."""

    dt: float
    hub: tuple = (0.20, -0.09, TOP + 0.08)
    axis: tuple = (1.0, 0.0, 0.0)  # the syringe's axis, toward the needle
    tip: tuple | None = None  # the needle's tip as it stands (the sensing needle body's tip); None: hub + NEEDLE_L axis
    tip_vel: tuple = (0.0, 0.0, 0.0)
    tip_force: float = 0.0  # N the needle's tip is pushed back along the axis (its sensing spring)
    travel: float = 0.0  # m the stopper's edge has moved forward from 1.0 mL
    travel_vel: float = 0.0
    thumb_n: float = 0.0  # N: the robot's contact load on the plunger along the axis, forward (a thumb's push)
    cap_on: bool = True
    cap_slide: float = 0.0  # m the cap has moved off its seat
    needle_axis: tuple | None = None  # the needle body's own axis (world unit vector), for the bent-needle outcome
    touch: str | None = None  # something other than the skin, the cap or the container's inside on the bare needle
    quats: tuple = ()  # the moving bodies' rotations as read (xyzw), for the fail-safe


SLIDE_NEW = 0.003  # m: the tip slid this far over the skin from a puncture while pressing: a new puncture
QUAT_TOL = 1e-3


def sample_fault(smp: Sample) -> str | None:
    """Why a reading cannot be trusted (the trial is then void: scored 0), or None."""
    if not math.isfinite(float(smp.dt)) or smp.dt <= 0.0:
        return f"time step {smp.dt} s (time must run forward)"
    vals = [*smp.hub, *smp.axis, *(smp.tip or ()), *smp.tip_vel, smp.tip_force, smp.travel, smp.travel_vel, smp.thumb_n, smp.cap_slide,
            *(smp.needle_axis or ())]
    if not all(math.isfinite(float(v)) for v in vals):
        return "a reading is not a finite number"
    if abs(float(np.linalg.norm(np.asarray(smp.axis, float))) - 1.0) > QUAT_TOL:
        return "the syringe's axis is not unit length"
    for q in smp.quats:
        q = np.asarray(q, float)
        if q.shape != (4,) or not np.isfinite(q).all() or abs(float(np.linalg.norm(q)) - 1.0) > QUAT_TOL:
            return "a body's rotation is not a unit quaternion"
    return None


@dataclass
class Status:
    """One control step's end-state checks."""

    in_container: bool = False  # the syringe wholly inside the sharps container
    cap_in_cup: bool = False
    clear: float = 1.0  # m: least distance from either gripper to the syringe or the cap
    dropped: bool = False  # the syringe at rest, out of every gripper, not in the container nor on its rest
    hidden: bool = False  # this frame, a scoring view lacks the barrel line and the hub or a band
    touch: str | None = None  # as Sample.touch, from the contact views at the control step
    let_go: bool = False  # no hand on the syringe for LET_GO_S while its needle is in the trainer


TIP_V_TAU = 0.02  # s: the low-pass on the tip's position-derived velocity (the tissue's stick-slip and lateral damping)


@dataclass
class TipVel:
    """The needle tip's velocity from its position, differentiated per physics step and low-passed over TIP_V_TAU: the
    tissue's forces then respond to the tip's real motion, not to the solver's velocity of the gripped barrel."""

    prev: np.ndarray | None = None
    v: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def step(self, tip, dt: float) -> np.ndarray:
        tip = np.asarray(tip, float)
        if self.prev is not None and dt > 0.0:
            self.v = self.v + dt / (TIP_V_TAU + dt) * ((tip - self.prev) / dt - self.v)
        self.prev = tip.copy()
        return self.v.copy()


_TIP_V = TipVel()


@dataclass
class Glide:
    """The stopper's glide friction (see F_GLIDE): `step` takes the slide's displacement over one physics step (m, from
    its position) and returns the force on the plunger along the axis (N, forward positive; the barrel takes minus it).
    `on` is False before the breakaway and after the retraction (no glide; the presliding is let go). It also keeps the
    sub-gate's record: the force's mean magnitude while sliding, and the filtered velocity's sign flips (0.5 mm/s each
    side) from the breakaway on."""

    z: float = 0.0  # m: the rubber's shear (presliding), within +-GLIDE_Z0
    v: float = 0.0  # m/s: the slide's velocity, low-passed over GLIDE_TAU
    force: float = 0.0
    sliding_n: int = 0
    sliding_sum: float = 0.0
    flips: int = 0
    sign: int = 0
    t_on: float = 0.0

    def step(self, dx: float, dt: float, on: bool) -> float:
        self.v += dt / (GLIDE_TAU + dt) * (dx / dt - self.v)
        if not on:
            self.z, self.force = 0.0, 0.0
            return 0.0
        self.t_on += dt
        self.z = float(np.clip(self.z + dx, -GLIDE_Z0, GLIDE_Z0))
        self.force = -F_GLIDE * self.z / GLIDE_Z0
        if abs(self.v) > GLIDE_V_SLIDE:
            s = 1 if self.v > 0.0 else -1
            self.flips += int(self.sign != 0 and s != self.sign)
            self.sign = s
            self.sliding_n += 1
            self.sliding_sum += abs(self.force)
        return self.force

    def mean_sliding(self) -> float:
        return self.sliding_sum / self.sliding_n if self.sliding_n else 0.0


@dataclass
class Trial:
    """The needle, the plunger, the retraction and the scoring table on plain values. `feed` takes each physics step's
    `Sample` and returns the joints to switch ("lock_off", "lock_on", "tissue_on", "tissue_off", "cap_off", "unstick",
    "detent", "retract"); `syr_force` (world, acting at the needle's tip; `syr_torque` is its moment about the hub),
    `plunger_n` (the stopper's glide friction along the axis on the plunger, minus it on the barrel: see `Glide`), `carrier_n` (along the axis, on the needle's carrier) and `needle_n` are what the sensing applies. `status` takes
    each control step's `Status`."""

    geom: Geometry = field(default_factory=Geometry)
    punctures: list = field(default_factory=list)
    cur: int | None = None
    engaged: bool = False
    pressing: bool = False  # the tip is on the skin since the last puncture record
    locked: bool = True  # the needle's carrier is locked to the barrel
    stuck: bool = True  # the stopper held at 1.0 mL by its static friction (J_stick) until a thumb breaks it loose
    passed: int = 0  # STOPS the thumb has pushed the stopper through, in order
    retracted: bool = False
    cap_on: bool = True
    last_travel: float = 0.0
    pushed_in_muscle: float = 0.0  # m
    edge_hw: float | None = None  # m: the furthest travel reached while in muscle (the first puncture)
    t_S: float | None = None
    pose_S: tuple | None = None
    vol_S: float | None = None
    crossings: list = field(default_factory=list)  # (line, t)
    t_zero: float | None = None  # when the edge reached the 0 line: the push's timing, the stop and thumb-on clauses end
    grad_k: int = 10 ** 6  # the lowest printed graduation (index, GRAD_ML each) the edge has reached since S
    t_grad: float = 0.0  # when the edge last reached a new graduation
    stopped: bool = False  # a stop: STOP_S or more between S and the 0 line with no new graduation reached
    status_n: int = 0  # control-step status() calls (the rule's own check that its scoring ran)
    thumb_off_s: float = 0.0  # s the contact load on the plunger has been under THUMB_ON_N, running (S to the 0 line)
    thumb_lost: bool = False  # the thumb off the thumb rest for THUMB_GAP_S or more between S and the 0 line
    t_unstick: float | None = None  # when the rule broke the stopper loose, and the thumb load that did it
    unstick_n: float | None = None
    window: bool = False
    continuous: bool = True
    moved: bool = False
    dev: dict = field(default_factory=lambda: {"pitch": 0.0, "yaw": 0.0, "x": 0.0, "y": 0.0, "z": 0.0})
    hidden_s: float = 0.0
    t_retract: float | None = None
    in_muscle_now: bool = False
    gap: float | None = None
    syr_force: np.ndarray = field(default_factory=lambda: np.zeros(3))
    syr_torque: np.ndarray = field(default_factory=lambda: np.zeros(3))
    plunger_n: float = 0.0
    glide: Glide = field(default_factory=Glide)
    carrier_n: float = 0.0
    needle_n: float = 0.0
    zero: str | None = None
    voided: str | None = None  # a fail-safe tripped: the trial is void (0)
    stowed: bool = False
    hold: int = 0
    full: bool = False
    score: float = 0.0
    t: float = 0.0

    # -- one physics step -----------------------------------------------------------------------------------------------
    def void(self, why: str) -> None:
        """A fail-safe: the readings or the scene cannot be trusted; the trial scores 0 whatever follows."""
        if self.voided is None:
            self.voided = why
        self.zero = f"void: {self.voided}"
        self.score, self.full, self.hold = 0.0, False, 0

    def feed(self, smp: Sample) -> list[str]:
        why = sample_fault(smp)
        if why is not None:
            self.void(why)
            return []
        self.t += smp.dt
        out: list[str] = []
        g = self.geom
        a = _unit(smp.axis)
        hub = np.asarray(smp.hub, float)
        tip = np.asarray(smp.tip, float) if smp.tip is not None else hub + NEEDLE_L * a
        travel = float(smp.travel)
        self.syr_force, self.syr_torque = np.zeros(3), np.zeros(3)
        # the cap's fit, and a pull off its axis bending the needle
        if self.cap_on:
            if smp.cap_slide >= CAP_POP - 1e-9 or not smp.cap_on:
                self.cap_on = False
                out.append("cap_off")
        bare = not self.cap_on and not self.retracted
        if bare and smp.needle_axis is not None and _deg(smp.needle_axis, a) > NEEDLE_BENT_DEG:
            self._zero(f"the needle was bent (more than {NEEDLE_BENT_DEG:.0f} deg off the barrel's axis)")
        if bare and smp.touch:
            self._zero(f"the bare needle touched {smp.touch}")
        if not self.punctures and travel > PRE_PUSH_ML * ML_M + 1e-9:
            self._zero("more than 0.2 mL pushed before the first puncture")
        # the plunger: held by its static friction until a thumb pushes the thumb rest with BREAKAWAY_F; then the glide
        # friction (an applied force pair, F_GLIDE) and the liquid's flow through the needle (the joint's damping, FLOW_C); a stop at
        # the 0 line that gives only to DETENT_F of thumb load there
        thumb = float(smp.thumb_n)
        if self.stuck and thumb >= BREAKAWAY_F - 1e-9:  # the only breakaway: the measured thumb load, never travel
            self.stuck = False
            self.t_unstick, self.unstick_n = self.t, thumb
            out.append("unstick")
        while not self.stuck and self.passed < len(STOPS):
            name, at, need = STOPS[self.passed]
            if travel < at - ZERO_TOL - 1e-9 or thumb < need - 1e-9:
                break
            self.passed += 1
            out.append("detent" if name == "J_zero" else f"pass_{name}")
        # the glide also holds the retracted plunger (a real VanishPoint's plunger is locked down once retracted; spikes 16,
        # 19c: with no force on it, the free slide crept back 19.6 -> 8.2 mm, pulling the thumb rest out of the barrel)
        self.plunger_n = self.glide.step(float(smp.travel_vel) * smp.dt, smp.dt, not self.stuck)
        # the needle on the skin: a press, a puncture, the needle going in
        if bare and not self.engaged:
            on = g.on_skin(tip, 0.003)
            if on and smp.tip_force >= ENGAGE_F - 1e-9:
                n = g.normal(tip[1], tip[2])
                A_now = angle_A(a, n)
                slid = False
                if self.pressing and self.punctures:  # still pressing, but slid 3 mm or more over the skin: a new puncture
                    off = tip - self.punctures[-1].entry
                    slid = float(np.linalg.norm(off - float(off @ n) * n)) >= SLIDE_NEW - 1e-9
                if not self.pressing or slid:  # a new puncture (a steep one still punctures a real arm; it never goes in here)
                    entry = tip - a * (float(np.dot(tip - np.array([g.surface_x(tip[1], tip[2]), tip[1], tip[2]]), -n)) / max(float(np.dot(a, -n)), 0.2))
                    D = g.z_edge - float(entry[2])
                    self.punctures.append(Puncture(entry=entry, n_out=n, D=D, L=abs(float(entry[1]))))
                    self.pressing = True
                    if D < D_HIGH - 1e-9:
                        self._zero("a puncture less than 20 mm below the acromion")
                if A_now <= ENGAGE_DEG + 1e-9:
                    self.cur = len(self.punctures) - 1
                    self.punctures[self.cur].engaged = True
                    self.engaged, self.locked = True, False
                    out += ["lock_off", "tissue_on"]
            elif self.pressing and not g.on_skin(tip, 0.004):
                self.pressing = False
        self.in_muscle_now = False
        self.gap = None
        if self.engaged and self.cur is not None:
            p = self.punctures[self.cur]
            n_in = -p.n_out
            cos = max(float(np.dot(a, n_in)), 0.2)
            self.gap = float(np.dot(p.entry - hub, n_in)) / cos
            depth = NEEDLE_L - self.gap
            self.in_muscle_now = self.gap <= IN_MUSCLE_GAP + 1e-9
            if self.in_muscle_now:
                p.in_muscle_ever = True
                if p.A is None:
                    p.A, p.travel_first = angle_A(a, p.n_out), travel
                if self.cur == 0:  # the edge's forward progress in muscle: beyond the furthest it has been there
                    # (spike 10 summed 7.2 mL of per-step jitter), never past the 0 line (the retraction's detent
                    # travel is not dose: spike 11 read 1.118 mL)
                    tc = min(travel, ZERO_TRAVEL)
                    if self.edge_hw is None:
                        self.edge_hw = min(self.last_travel, ZERO_TRAVEL)
                    if tc > self.edge_hw:
                        self.pushed_in_muscle += tc - self.edge_hw
                        self.edge_hw = tc
            if self.gap <= FULL_DEPTH_GAP + 1e-9 and self.cur == 0 and not self.window_closed():
                if not p.full_depth:
                    p.travel_hub = travel
                p.full_depth = True
            # the tissue: stick-slip along the needle, and a soft hold on its line
            tv = np.asarray(smp.tip_vel, float)
            f = -TISSUE_F * math.tanh(float(tv @ a) / TISSUE_V) * a
            off = tip - p.entry
            lat = off - float(off @ a) * a
            lat_v = tv - float(tv @ a) * a
            f = f - K_LAT * lat - C_LAT * lat_v
            self.syr_force = f
            self.syr_torque = np.cross(tip - hub, f)
            if depth < -0.001:  # the tip is out of the skin: withdrawn
                self.engaged = False
                self.pressing = False
                out.append("tissue_off")
        if not (self.in_muscle_now and self.cur == 0):  # out of muscle: what is pushed there is never credited later
            self.edge_hw = min(travel, ZERO_TRAVEL)
        # S, and the window from S to the retraction
        p0 = self.punctures[0] if self.punctures else None
        if (self.t_S is None and p0 is not None and p0.travel_first is not None
                and not self.retracted and volume(travel) <= s_line(volume(p0.travel_first)) + 1e-9):
            n0 = p0.n_out
            self.t_S, self.vol_S, self.window = self.t, volume(travel), True
            self.pose_S = (hub.copy(), *pitch_yaw(a, n0))
            self.grad_k, self.t_grad = grad_index(self.vol_S), self.t
        if self.window and p0 is not None and self.pose_S is not None and self.vol_S is not None:
            if not (self.engaged and self.cur == 0 and self.in_muscle_now):
                self.continuous = False
            pit, yaw = pitch_yaw(a, p0.n_out)
            d = hub - self.pose_S[0]
            dev = {"pitch": abs(pit - self.pose_S[1]), "yaw": abs(yaw - self.pose_S[2]), "x": abs(d[0]), "y": abs(d[1]), "z": abs(d[2])}
            self.dev = {k: max(self.dev[k], dev[k]) for k in dev}
            if dev["pitch"] > STILL_DEG or dev["yaw"] > STILL_DEG or max(dev["x"], dev["y"], dev["z"]) > STILL_M:
                self.moved = True
            vol = volume(travel)
            for line in self.lines_due():
                if vol <= line + LINE_TOL / ML_M + 1e-9 and line not in [c[0] for c in self.crossings]:
                    self.crossings.append((line, self.t))
            if self.t_zero is None:  # from S until the edge reaches the 0 line: the stop and the thumb on the thumb rest
                k = grad_index(vol)
                if k < self.grad_k:  # a new printed graduation reached
                    self.grad_k, self.t_grad = k, self.t
                elif self.t - self.t_grad >= STOP_S - 1e-9:
                    self.stopped = True
                self.thumb_off_s = 0.0 if thumb >= THUMB_ON_N - 1e-9 else self.thumb_off_s + smp.dt
                if self.thumb_off_s >= THUMB_GAP_S - 1e-9:
                    self.thumb_lost = True
                if any(c[0] == 0.0 for c in self.crossings):
                    self.t_zero = self.t
        # the retraction: only once a thumb has pushed the stopper through the 0 line's stop
        if not self.retracted and not self.at_stop and travel >= RETRACT_AT - 1e-9:
            self.retracted, self.t_retract = True, self.t
            self.window = False
            if self.engaged:
                self.engaged = False
                out.append("tissue_off")
            out.append("retract")
        # the needle's carrier: forward onto the skin while in, back home when out, into the barrel once retracted
        self.needle_n = NEEDLE_PUSH if self.engaged else 0.0
        if self.retracted:
            self.carrier_n = -RETRACT_PUSH
        elif not self.engaged and not self.locked:
            self.carrier_n = CARRIER_RETURN
        else:
            self.carrier_n = 0.0
        self.last_travel = travel
        return out

    @property
    def at_stop(self) -> bool:
        """The 0 line's stop (J_zero) not yet pushed through."""
        return self.passed < len(STOPS)

    def limit(self) -> float:
        """The stopper's travel can go no further than this now (m): its stiction and the next stop."""
        if self.stuck:
            return 0.0
        return STOPS[self.passed][1] if self.passed < len(STOPS) else TRAVEL_MAX

    def carrier_home(self) -> list[str]:
        """The sensing calls this when the carrier is back at its seat (after a withdrawal): lock it again."""
        if not self.locked and not self.engaged and not self.retracted:
            self.locked = True
            return ["lock_on"]
        return []

    def window_closed(self) -> bool:
        return self.retracted

    def _zero(self, why: str) -> None:
        if self.zero is None:
            self.zero = why

    # -- derived results -------------------------------------------------------------------------------------------------
    def first(self) -> Puncture | None:
        return self.punctures[0] if self.punctures else None

    def lines_due(self) -> list:
        """The 0.2 mL lines the push is timed over: those more than LINE_TOL below the edge at S."""
        return [] if self.vol_S is None else [x for x in LINES if x < self.vol_S - LINE_TOL / ML_M - 1e-9]

    def rate_ok(self) -> bool:
        """Each 0.2 mL in 1-4 s, the first interval (S to the first line reached) prorated by its volume."""
        if self.t_S is None or self.vol_S is None:
            return False
        want = self.lines_due()
        got = dict(self.crossings)
        if not want or any(x not in got for x in want):
            return False
        tol = LINE_TOL / ML_M  # each line is reached LINE_TOL early, so the first interval's volume is to there
        prev_t, prev_v = self.t_S, self.vol_S
        for line in want:
            dt = got[line] - prev_t
            frac = (prev_v - line - (tol if prev_t == self.t_S else 0.0)) / STEP_ML
            if not (STEP_S[0] * frac - 1e-6 <= dt <= STEP_S[1] * frac + 1e-6):
                return False
            prev_t, prev_v = got[line], line
        return True

    def steady(self) -> bool:
        """Steady push: the rate band, no stop and the thumb on, all from S to the edge reaching the 0 line."""
        return self.retracted and self.t_zero is not None and self.rate_ok() and not self.stopped and not self.thumb_lost

    def full_dose(self) -> bool:
        p = self.first()
        return bool(self.retracted and p and p.travel_first is not None and volume(p.travel_first) >= DOSE_FIRST_ML - 1e-9)

    def still(self) -> bool:
        return self.t_S is not None and not self.moved and self.hidden_s <= HIDDEN_MAX_S + 1e-9

    def any_025(self) -> bool:
        return any(p.in_muscle_ever and D_05[0] - 1e-9 <= p.D <= D_05[1] + 1e-9 and p.L <= L_05 + 1e-9 and p.A is not None and p.A <= A_05 + 1e-9
                   for p in self.punctures)

    def args(self) -> dict:
        p = self.first()
        return {"zero": self.zero is not None, "punctures": len(self.punctures), "D": p.D if p else -1.0, "L": p.L if p else 1.0,
                    "A": p.A if p and p.A is not None else 90.0, "full_depth": bool(p and p.full_depth), "continuous": self.t_S is not None and self.continuous and self.retracted,
                    "still": self.still(), "full_dose": self.full_dose(), "steady": self.steady(), "stowed": self.stowed, "pushed_ml": self.pushed_in_muscle / ML_M,
                    "any_025": self.any_025()}

    # -- one control step ---------------------------------------------------------------------------------------------
    def status(self, st: Status) -> float:
        self.status_n += 1  # the control-step checks ran (spikes 18-19: a misplaced decorator meant they never did)
        if self.voided is not None:
            return self.score
        if self.window and st.hidden:
            self.hidden_s += CONTROL_DT
        if st.dropped and not self.cap_on and not self.retracted:  # a capped (or retracted) syringe set down is no hazard
            self._zero("the syringe with its bare needle was dropped or set down outside the sharps container")
        if st.touch and not self.cap_on and not self.retracted:
            self._zero(f"the bare needle touched {st.touch}")
        if st.let_go and self.engaged and not self.retracted:
            self._zero("the syringe was let go with its needle in the trainer")
        self.stowed = bool(st.in_container and st.cap_in_cup and st.clear >= GRIP_CLEAR - 1e-9)
        args = self.args()
        now = tier(held=True, **args)
        self.hold = self.hold + 1 if now == 1.0 else 0
        self.full = self.hold >= HOLD_STEPS
        self.score = tier(held=self.full, **args)
        return self.score

    def report(self) -> dict:
        p = self.first()
        return {"punctures": len(self.punctures), "engaged": self.engaged, "gap_mm": None if self.gap is None else round(1000 * self.gap, 1),
                "D_mm": round(1000 * p.D, 1) if p else None, "L_mm": round(1000 * p.L, 1) if p else None,
                "A_deg": round(p.A, 2) if p and p.A is not None else None, "full_depth": bool(p and p.full_depth),
                "vol_first": round(volume(p.travel_first), 3) if p and p.travel_first is not None else None,
                "S_s": None if self.t_S is None else round(self.t_S, 2), "retract_s": None if self.t_retract is None else round(self.t_retract, 2),
                "pushed_ml": round(self.pushed_in_muscle / ML_M, 3), "continuous": self.continuous, "moved": self.moved,
                "dev": {k: round(v if k in ("pitch", "yaw") else 1000 * v, 2) for k, v in self.dev.items()}, "hidden_s": round(self.hidden_s, 2),
                "crossings": [(c[0], round(c[1] - (self.t_S or 0.0), 2)) for c in self.crossings], "steady": self.steady(),
                "status_n": self.status_n,
                "rate_ok": self.rate_ok(), "stopped": self.stopped, "thumb_lost": self.thumb_lost,
                "zero_line_s": None if self.t_zero is None else round(self.t_zero, 2),
                "unstick_s": None if self.t_unstick is None else round(self.t_unstick, 2),
                "unstick_n": None if self.unstick_n is None else round(self.unstick_n, 2),
                "full_dose": self.full_dose(), "stowed": self.stowed, "cap_on": self.cap_on, "retracted": self.retracted,
                "zero": self.zero, "voided": self.voided, "hold": self.hold, "score": self.score}


TRIAL = Trial()


# =====================================================================================================================
# The scoring cameras' views (pure NumPy: used in the simulator every control step of the window, and offline)
# =====================================================================================================================

SIDE_CAM = np.array([0.235, 0.40, TOP + 0.21])  # clamp stand, looking along -y (in front of the skin, so the hub shows)
OVER_CAM = np.array([0.200, 0.0, TOP + 0.70])  # 1 m boom, looking down; at x 20 cm so the shoulder hides no hub
OCCLUDE_R = 0.005  # m: a sampled arm-mesh point this close to a sight line hides what is behind it


def view_targets(hub, R) -> dict:
    """The points each scoring view reads, world. R: the syringe's rotation (its x along the barrel)."""
    hub = np.asarray(hub, float)

    def at(s, y=0.0, z=0.0):
        return hub + R @ np.array([s, y, z])

    side = {"hub": [at(-0.001, BARREL_HALF * 0.55, 0.0)], "band": [at(BAND, BARREL_HALF, 0.0)],
            "line": [at(s, BARREL_HALF, 0.0) for s in (-0.012, -0.018, -0.024, -0.030)]}
    over = {"hub": [at(-0.001, 0.0, BARREL_HALF * 0.55)], "band": [at(BAND, 0.0, BARREL_HALF)],
            "line": [at(s, 0.0, BARREL_HALF) for s in (-0.012, -0.020, -0.028, -0.036)]}
    return {"side": side, "over": over}


def _segment_blocked(cam, p, occluders, boxes) -> bool:
    d = p - cam
    n = float(np.linalg.norm(d))
    if occluders is not None and len(occluders):
        w = occluders - cam
        t = np.clip((w @ d) / (n * n), 0.0, 1.0)
        near = np.linalg.norm(w - t[:, None] * d, axis=1)
        if bool(((near < OCCLUDE_R) & (t * n < n - 0.004)).any()):
            return True
    for lo, hi in boxes:  # slab test, stopping 1 mm short of the point
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = (lo - cam) / d, (hi - cam) / d
        tmin = np.nanmax(np.minimum(t1, t2))
        tmax = np.nanmin(np.maximum(t1, t2))
        if tmax >= max(tmin, 0.0) and tmin < 1.0 - 0.001 / n:
            return True
    return False


def view_ok(cam, targets: dict, occluders, boxes) -> dict:
    """Which of a view's targets show: 'line' (3 of its 4 barrel points), 'hub', 'band', and 'ok' (line and hub or band)."""
    seen = {k: [not _segment_blocked(cam, np.asarray(p), occluders, boxes) for p in v] for k, v in targets.items()}
    line = sum(seen["line"]) >= 3
    out = {"line": line, "hub": all(seen["hub"]), "band": all(seen["band"])}
    out["ok"] = line and (out["hub"] or out["band"])
    return out


def hidden_frame(hub, R, occluders, boxes) -> tuple[bool, dict]:
    t = view_targets(hub, R)
    s, o = view_ok(SIDE_CAM, t["side"], occluders, boxes), view_ok(OVER_CAM, t["over"], occluders, boxes)
    return not (s["ok"] and o["ok"]), {"side": s, "over": o}


def hub_y_from_band(hub, R) -> float:
    """The overhead's fallback: the band's position plus 43.8 mm along the barrel line as the overhead sees it (x-y)."""
    band = np.asarray(hub, float) + R @ np.array([BAND, 0.0, 0.0])
    ax = (R @ np.array([1.0, 0.0, 0.0]))[:2]
    return float(band[1] + (-BAND) * ax[1] / max(np.linalg.norm(ax), 1e-9))


# =====================================================================================================================
# Scene (positions: world metres, which are the setup's cm from the midpoint between the arm bases; heights above TOP)
# =====================================================================================================================

import yam_bodies as yb

B = yb.BoxShape
GRIP_FRICTION = (1.2, 1.0)  # yam_arena's material for gripped objects (GRIPPY_FRICTION), with a torsional patch
PLY, PVC, SKIN, SKIN_D, BLACK, INK = (0.62, 0.52, 0.38), (0.93, 0.93, 0.92), (0.88, 0.70, 0.60), (0.80, 0.62, 0.52), (0.07, 0.07, 0.08), (0.03, 0.03, 0.04)
CLEAR, WHITE, STEEL, RUBBER, ORANGE, RED, GREY = (0.82, 0.88, 0.92), (0.96, 0.96, 0.96), (0.80, 0.81, 0.84), (0.18, 0.18, 0.20), (0.95, 0.55, 0.12), (0.80, 0.12, 0.10), (0.30, 0.31, 0.33)
SILICONE, PAD, PAPER, POLY = (0.55, 0.60, 0.62), (0.92, 0.92, 0.88), (0.94, 0.92, 0.86), (0.85, 0.87, 0.90)

H = GEOM.h
BASE_TOP = H + PLY_T  # above the table
FLANGE_PLATE, FLANGE_D, COLLAR_D, COLLAR_H = 0.015, 0.190, 0.135, 0.050


def _ring(cx, cy, z0, h, r_out, wall, color, n=8, collide=True) -> list:
    """An n-sided ring of boxes (a cylinder's wall), centred at (cx, cy), from z0 up h."""
    out = []
    side = 2 * r_out * math.tan(math.pi / n) + 0.001
    for k in range(n):
        a = 2 * math.pi * k / n
        c = (r_out - wall / 2)
        out.append(B((cx + c * math.cos(a), cy + c * math.sin(a), z0 + h / 2), (wall, side, h), color, collide=collide,
                     quat_xyzw=(0.0, 0.0, math.sin(a / 2), math.cos(a / 2))))
    return out


def _ngon_floor(cx, cy, z0, h, apothem, color, n=10) -> list:
    """A flat n-gon (a disc): n boxes from the centre to each side, each as wide as that side (together they cover the
    n-gon to its corners, at apothem / cos(pi / n)). The flange plate, and the cup's floor filling its 10-sided wall."""
    side = 2 * apothem * math.tan(math.pi / n)
    out = []
    for k in range(n):
        a = 2 * math.pi * k / n
        out.append(B((cx + apothem / 2 * math.cos(a), cy + apothem / 2 * math.sin(a), z0 + h / 2), (apothem, side, h), color,
                     quat_xyzw=(0.0, 0.0, math.sin(a / 2), math.cos(a / 2))))
    return out


# the stand: spacer (by trial), 3/4 in plywood base 22 x 22 at x 25-47, flange, pipe (4 in Schedule 40: 6.0 mm wall)
PIPE_WALL = 0.00602
_stand = []
if H > 0:
    _stand.append(B((0.36, 0.0, H / 2), (0.22, 0.22, H), PLY))
_stand += [B((0.36, 0.0, H + PLY_T / 2), (0.22, 0.22, PLY_T), PLY),
           B((0.36, 0.1102, H + PLY_T * 0.5), (0.12, 0.0005, 0.010), (0.98, 0.98, 0.98), collide=False),  # the 100 mm scale bar on the +y face
           B((0.36, 0.1105, H + PLY_T * 0.5), (0.10, 0.0005, 0.002), BLACK, collide=False),
           *_ngon_floor(PIPE_X, 0.0, BASE_TOP, FLANGE_PLATE, FLANGE_D / 2, PVC, n=16),  # 19.0 cm across its flats
           *_ring(PIPE_X, 0.0, BASE_TOP + FLANGE_PLATE, COLLAR_H, COLLAR_D / 2, 0.010, PVC),
           *_ring(PIPE_X, 0.0, BASE_TOP + FLANGE_PLATE, PIPE_L, PIPE_R, PIPE_WALL, PVC, n=12)]
STAND = yb.Body.build("stand", (0.0, 0.0, TOP), _stand, static=True, friction=(0.5, 0.5))
STAND_BOXES = {"base": ((0.25, -0.11, TOP), (0.47, 0.11, TOP + BASE_TOP)),
               "flange": ((PIPE_X - FLANGE_D / 2, -FLANGE_D / 2, TOP + BASE_TOP), (PIPE_X + FLANGE_D / 2, FLANGE_D / 2, TOP + BASE_TOP + FLANGE_PLATE + COLLAR_H)),
               "pipe": ((PIPE_FACE, -PIPE_R, TOP + BASE_TOP), (PIPE_X + PIPE_R, PIPE_R, TOP + BASE_TOP + FLANGE_PLATE + PIPE_L))}


def _trainer_shapes(g: Geometry) -> list:
    """The trainer, world-placed: its flat lateral face (the deltoid) at skin_x for |y| <= 25 mm, sides falling back
    30 deg, the shoulder cap above the acromion's edge 6 mm back (the edge shows as a step), a puncture-proof back, the
    silicone sheet behind it and two straps round the pipe. Collision from these convex pieces (within 1 mm at the site),
    drawn invisible; what the cameras see is visual-only geometry within the colliders' envelope or behind the skin's
    plane (x >= skin_x, so it can hide nothing the scoring views look at): the arm segment's rounded contour (the
    deltoid's face 40 mm wide, then facets on to the 30 deg sides), the shoulder cap rounded the same way with its top
    front edge chamfered, and the acromion's bony ridge just above the edge (2 mm proud of the cap, still behind the skin)."""
    sx, back = g.skin_x, PIPE_FACE - SHEET
    zb, ze, zt = g.z_bottom, g.z_edge, g.z_top
    t = back - sx
    b = math.radians(BEVEL_DEG)
    side_w = TRAINER_W / 2 - CENTRE_HALF
    out = [B(((sx + back) / 2, 0.0, (zb + ze) / 2), (t, 2 * CENTRE_HALF, ze - zb), None),  # the deltoid: its flat face
           B(((sx + SHOULDER_BACK + back) / 2, 0.0, (ze + zt) / 2), (t - SHOULDER_BACK, TRAINER_W, zt - ze), None)]  # shoulder cap
    for s in (-1.0, 1.0):
        # the sides: a box whose front face runs back at 30 deg from |y| 25 mm
        L = side_w / math.cos(b)
        cy = s * (CENTRE_HALF + side_w / 2)
        fx = sx + side_w / 2 * math.tan(b)  # the facet's middle; the box lies 6 mm behind it, along its inward normal
        out.append(B((fx + 0.006 * math.cos(b), cy - s * 0.006 * math.sin(b), (zb + ze) / 2), (0.012, L, ze - zb), None,
                     quat_xyzw=(0.0, 0.0, math.sin(-s * b / 2), math.cos(-s * b / 2))))
        out.append(B(((sx + 0.011 + back) / 2, cy, (zb + ze) / 2), (back - sx - 0.011, side_w, ze - zb), SKIN))
    # the shoulder cap: the same contour up to 8 mm below its top, then its top front edge sloped back 45 deg (all inside
    # the cap's collider)
    ch = 0.008
    fc = sx + SHOULDER_BACK
    out += _contour(sx, back, zb, ze, SKIN) + _contour(fc, back, ze, zt - ch, SKIN_D)
    out.append(B((fc + ch / 2 + 0.002 * math.sqrt(0.5), 0.0, zt - ch / 2 - 0.002 * math.sqrt(0.5)), (0.004, 2 * CONTOUR_KNEE, ch * math.sqrt(2)),
                 SKIN_D, collide=False, quat_xyzw=(0.0, math.sin(math.pi / 8), 0.0, math.cos(math.pi / 8))))
    out.append(B(((fc + ch + back) / 2, 0.0, zt - ch / 2), (back - fc - ch, TRAINER_W - 0.004, ch), SKIN_D, collide=False))
    # the acromion: a bony ridge along y just above the edge, its front 4 mm behind the skin's plane
    rs = 0.0057
    out.append(B((sx + 0.004 + rs * math.sqrt(0.5), 0.0, ze + 0.002 + rs * math.sqrt(0.5)), (rs, 0.040, rs), SKIN_D, collide=False,
                 quat_xyzw=(0.0, math.sin(math.pi / 8), 0.0, math.cos(math.pi / 8))))
    out += [B((back + SHEET / 2, 0.0, (zb + zt) / 2), (SHEET, TRAINER_W, zt - zb), SILICONE),  # the anti-slip sheet
            B((back - 0.002, 0.0, (zb + zt) / 2), (0.004, TRAINER_W + 0.002, zt - zb), (0.30, 0.30, 0.32), collide=False)]  # the back plate
    for zc in (zb + 0.025, zb + TRAINER_H - 0.045):  # two straps round the pipe
        out += [B((PIPE_X, sy * (PIPE_R + 0.002), zc), (2 * PIPE_R, 0.003, 0.025), BLACK, collide=False) for sy in (-1.0, 1.0)]
        out.append(B((PIPE_X + PIPE_R + 0.0015, 0.0, zc), (0.003, 2 * PIPE_R, 0.025), BLACK, collide=False))
    return out


CONTOUR_FLAT, CONTOUR_KNEE = 0.020, 0.032  # m: the visible face is flat to |y| 20 mm, then a facet to the 30 deg side at 32


def _contour(fx0, back, z0, z1, color) -> list:
    """Visual-only: a rounded front from x = fx0 (flat for |y| <= CONTOUR_FLAT, then a facet that cuts inside the
    collider's corner at |y| 25 mm to meet its 30 deg side at CONTOUR_KNEE, then that side to |y| 42.5 mm), 12 mm deep."""
    b = math.radians(BEVEL_DEG)
    half = TRAINER_W / 2
    d = 0.012
    xk = fx0 + (CONTOUR_KNEE - CENTRE_HALF) * math.tan(b)  # the side's x at the knee
    xe = fx0 + (half - CENTRE_HALF) * math.tan(b)
    out = [B((fx0 + d / 2, 0.0, (z0 + z1) / 2), (d, 2 * CONTOUR_FLAT, z1 - z0), color, collide=False),
           B(((fx0 + d + back) / 2, 0.0, (z0 + z1) / 2), (back - fx0 - d, 2 * half - 0.004, z1 - z0), color, collide=False)]
    for s in (-1.0, 1.0):
        for (y0, x0), (y1, x1) in (((CONTOUR_FLAT, fx0), (CONTOUR_KNEE, xk)), ((CONTOUR_KNEE, xk), (half, xe))):
            L = math.hypot(y1 - y0, x1 - x0)
            ang = math.atan2(x1 - x0, y1 - y0)  # the facet's slope back from the face
            nx, ny = math.cos(ang), -math.sin(ang)  # its inward normal (into +x), on the +y side
            cx, cy = (x0 + x1) / 2 + nx * d / 2, (y0 + y1) / 2 + ny * d / 2
            out.append(B((cx, s * cy, (z0 + z1) / 2), (d, L, z1 - z0), color, collide=False,
                         quat_xyzw=(0.0, 0.0, math.sin(-s * ang / 2), math.cos(-s * ang / 2))))
    return out


TRAINER = yb.Body.build("trainer", (0.0, 0.0, 0.0), _trainer_shapes(GEOM), static=True, friction=(0.8, 0.7))

# the splash tray and the absorbent strip on the base's front edge (ahead of the flange)
TRAY = yb.Body.build("tray", (0.1975, 0.0, TOP), [
    B((0.0, 0.0, 0.002), (0.095, 0.080, 0.004), POLY), *[B((sx * 0.0465, 0.0, 0.0075), (0.002, 0.080, 0.015), POLY) for sx in (-1, 1)],
    *[B((0.0, sy * 0.039, 0.0075), (0.095, 0.002, 0.015), POLY) for sy in (-1, 1)], B((0.0, 0.0, 0.0045), (0.090, 0.075, 0.001), PAD, collide=False),
    B((0.059, 0.0, BASE_TOP + 0.0005), (0.013, 0.080, 0.001), PAD, collide=False)], static=True, friction=(0.5, 0.5))  # the strip: x 25.0-26.3

# the syringe rest: a 12 x 8 cm plywood base at x 12-24, y -13 to -5; a slotted block under the flange (the flange's
# lower tab stands in its 3 mm slot) and a notch block under the cap. Syringe axis along x, 8 cm up, hub at x 20.
SYR_AT = np.array([0.200, -0.090, TOP + 0.080])  # the hub, on the rest
_fx = SYR_AT[0] + sum(FLANGE) / 2
_cx = SYR_AT[0] + 0.022
_slot_floor = 0.080 - 0.0125
REST = yb.Body.build("rest", (0.18, -0.09, TOP), [
    B((0.0, 0.0, 0.006), (0.12, 0.08, 0.012), PLY),
    B((_fx - 0.18, 0.0, (0.012 + _slot_floor) / 2), (0.012, 0.024, _slot_floor - 0.012), PLY),  # under the flange: the slot floor
    *[B((_fx - 0.18 + sx * 0.0040, 0.0, _slot_floor + 0.0015), (0.005, 0.024, 0.003), PLY) for sx in (-1, 1)],  # the slot's walls (3 mm wide)
    B((_cx - 0.18, 0.0, (0.012 + 0.080 - CAP_HALF) / 2), (0.012, 0.024, 0.080 - CAP_HALF - 0.012), PLY),  # under the cap
    *[B((_cx - 0.18, sy * (CAP_HALF + 0.0035), 0.080 - CAP_HALF + 0.0015), (0.012, 0.005, 0.003), PLY) for sy in (-1, 1)]],  # the notch's cheeks
    static=True, friction=(0.5, 0.5))

# the sharps container: about 2.5 L, 15 (x) x 11 (y) x 15 cm, centred at (24.0, -27.5), a lid with an 11 x 7 cm opening
SHARPS_AT = (0.240, -0.275)
SH_X, SH_Y, SH_H, SH_W = 0.150, 0.110, 0.150, 0.003
OPEN_X, OPEN_Y = 0.110, 0.070
SHARPS = yb.Body.build("sharps", (*SHARPS_AT, TOP), [
    B((0.0, 0.0, SH_W / 2), (SH_X, SH_Y, SH_W), RED),
    *[B((sx * (SH_X - SH_W) / 2, 0.0, SH_H / 2), (SH_W, SH_Y, SH_H), RED) for sx in (-1, 1)],
    *[B((0.0, sy * (SH_Y - SH_W) / 2, SH_H / 2), (SH_X, SH_W, SH_H), RED) for sy in (-1, 1)],
    *[B((sx * (SH_X + OPEN_X) / 4, 0.0, SH_H - SH_W / 2), ((SH_X - OPEN_X) / 2, SH_Y, SH_W), RED) for sx in (-1, 1)],  # lid, either end of the opening
    *[B((0.0, sy * (SH_Y + OPEN_Y) / 4, SH_H - SH_W / 2), (OPEN_X, (SH_Y - OPEN_Y) / 2, SH_W), RED) for sy in (-1, 1)],
    B((0.0, -SH_Y / 2 - 0.0003, SH_H * 0.55), (0.06, 0.0005, 0.03), (0.95, 0.95, 0.95), collide=False)],  # its label
    static=True, friction=(0.5, 0.5))
SHARPS_INSIDE = (np.array([SHARPS_AT[0] - SH_X / 2 + SH_W, SHARPS_AT[1] - SH_Y / 2 + SH_W, TOP + SH_W]),
                 np.array([SHARPS_AT[0] + SH_X / 2 - SH_W, SHARPS_AT[1] + SH_Y / 2 - SH_W, TOP + SH_H - SH_W]))

# the paper waste cup: 9 cm across, 6 cm tall, at (20.0, +18.0), taped down
CUP_AT, CUP_R, CUP_H = (0.200, 0.180), 0.045, 0.060
CUP = yb.Body.build("cup", (*CUP_AT, TOP), [*_ngon_floor(0.0, 0.0, 0.0, 0.003, CUP_R - 0.002, PAPER),  # the floor: the whole inside
                                            *_ring(0.0, 0.0, 0.0, CUP_H, CUP_R, 0.002, PAPER, n=10)],
                    static=True, friction=(0.5, 0.5))

# the scoring cameras' mounts (props; the simulator's own cameras film the trial, these poses are used for visibility)
CAMERAS = yb.Body.build("camera_mounts", (0.0, 0.0, TOP), [
    B((SIDE_CAM[0], SIDE_CAM[1] + 0.02, 0.004), (0.10, 0.08, 0.008), GREY), B((SIDE_CAM[0], SIDE_CAM[1] + 0.03, 0.115), (0.016, 0.016, 0.23), GREY),
    B((SIDE_CAM[0], SIDE_CAM[1] + 0.012, SIDE_CAM[2] - TOP), (0.05, 0.035, 0.04), BLACK),
    B((OVER_CAM[0], 0.0, OVER_CAM[2] - TOP + 0.012), (0.04, 0.05, 0.035), BLACK, collide=False),
    B((OVER_CAM[0] + 0.32, 0.0, OVER_CAM[2] - TOP + 0.04), (0.66, 0.02, 0.02), GREY, collide=False)], static=True)

# -- the syringe and its moving parts (each body's frame world-aligned at the start; x along the barrel) ---------------
_q45x = (math.sin(math.pi / 8), 0.0, 0.0, math.cos(math.pi / 8))


def _sx(s0, s1):
    return ((s0 + s1) / 2, s1 - s0)


_ticks = []
for k in range(31):  # the printed scale on the +y face: every 0.1 mL, longer every 0.5
    s = LINE0 - k * 0.1 * ML_M
    if s > BARREL[0] + 0.002:
        _ticks.append(B((s, BARREL_HALF + 0.0002, 0.0018 if k % 5 else 0.0), (0.0005, 0.0004, 0.0036 if k % 5 else 0.0072), INK, collide=False))
SYRINGE_SHAPES = [
    B((_sx(*BARREL)[0], 0.0, 0.0), (_sx(*BARREL)[1], 2 * BARREL_HALF, 2 * BARREL_HALF), None),  # the barrel (clear: its collider not drawn)
    *[B((_sx(*BARREL)[0], sy * (BARREL_HALF - 0.0004), sz * (BARREL_HALF - 0.0004)), (_sx(*BARREL)[1], 0.0008, 0.0008), CLEAR, collide=False)
      for sy in (-1, 1) for sz in (-1, 1)],  # its four edges
    B((-0.0105, 0.0, 0.0), (0.001, 2 * BARREL_HALF + 0.0004, 2 * BARREL_HALF + 0.0004), CLEAR, collide=False),
    B((BARREL[0] + 0.0005, 0.0, 0.0), (0.001, 2 * BARREL_HALF + 0.0004, 2 * BARREL_HALF + 0.0004), CLEAR, collide=False),
    B((_sx(*NOSE)[0], 0.0, 0.0), (_sx(*NOSE)[1], 0.006, 0.006), CLEAR),  # the nose; its front is the hub
    B((_sx(*FLANGE)[0], 0.0, 0.0), (_sx(*FLANGE)[1], 0.007, 0.025), CLEAR),  # the finger flange, long axis vertical
    B((BAND, 0.0, BARREL_HALF + 0.0002), (0.003, 0.009, 0.0004), BLACK, collide=False),  # marker bands at 2.0 mL: top
    B((BAND, BARREL_HALF + 0.0003, 0.0), (0.003, 0.0004, 0.009), BLACK, collide=False),  # and +y
    *_ticks]
# INJ_PROBE (diagnostics only; unset in every scored run): "1" logs the syringe's bodies every physics step around a test
# push (5.0-5.5 s, 1 N on the plunger) and the right hand's pinch (7.3-8.2 s); "2" also keeps the syringe's five bodies
# awake (PhysX sleep threshold 0)
PROBE = os.environ.get("INJ_PROBE", "")
ZERO_WRENCH = os.environ.get("INJ_ZERO_WRENCH", "")  # diagnostics only: "1" glide pair, "2" nothing, "3" + tissue, "4" + pushes
_AWAKE = {"sleep_threshold": 0.0} if PROBE == "2" else {}
# INJ_SOLVER (diagnostics only; unset in every scored run): the syringe's five bodies with 4 velocity iterations,
# gyroscopic forces, a 0.01 m/s cap on depenetration and stabilization on (spike 15a: the gripped barrel's velocity read
# -80 to -144 mm/s along x with its position still to 0.03 mm, a solver correction that may ratchet the free plunger)
if os.environ.get("INJ_SOLVER"):
    _AWAKE = {**_AWAKE, "velocity_iterations": 4, "gyroscopic": True, "max_depenetration": 0.01, "stabilization": 1e-5}
SYR_MASS = 0.009  # kg with 1 mL of water, less the plunger, cap and needle (TO MEASURE)
SYRINGE = yb.Body.build("syringe", tuple(SYR_AT), SYRINGE_SHAPES, mass=SYR_MASS, com=(-0.040, 0.0, 0.0), friction=GRIP_FRICTION,
                        torsional_patch=0.004, contact_report=True, solver_iterations=32, articulation_root=True, **_AWAKE)
CARRIER = yb.Body.build("carrier", tuple(SYR_AT + [NEEDLE_L / 2, 0.0, 0.0]), [B((0.0, 0.0, 0.0), (0.002, 0.001, 0.001), None, collide=False)],
                        mass=0.001, inertia=(1e-8, 1e-8, 1e-8), ignore=("syringe",), solver_iterations=32, **_AWAKE)
NEEDLE = yb.Body.build("needle", tuple(SYR_AT + [NEEDLE_L / 2, 0.0, 0.0]), [
    B((0.0, 0.0, 0.0), (NEEDLE_L, 0.001, 0.001), None),  # its collider
    B((0.0, 0.0, 0.0), (NEEDLE_L, 0.0005, 0.0005), STEEL, collide=False),
    B((NEEDLE_L / 2 - 0.0008, 0.0, 0.0), (0.0016, 0.0003, 0.0006), (0.9, 0.9, 0.95), collide=False)],  # the bevel's glint
    mass=0.001, inertia=(1e-8, 1e-7, 1e-7), friction=(0.2, 0.2), ignore=("syringe", "carrier"), contact_report=True, solver_iterations=32, **_AWAKE)
PLUNGER_AT = SYR_AT + [EDGE0, 0.0, 0.0]
PLUNGER = yb.Body.build("plunger", tuple(PLUNGER_AT), [
    B((-0.0025, 0.0, 0.0), (0.005, 0.0084, 0.0084), RUBBER, collide=False),  # the stopper; its front edge is the reading
    B((-0.0355, 0.0, 0.0), (0.061, 0.0012, 0.0070), WHITE), B((-0.0355, 0.0, 0.0), (0.061, 0.0070, 0.0012), WHITE),  # the rod (cross)
    B((-ROD_BACK + 0.001, 0.0, 0.0), (0.002, 0.012, 0.017), WHITE), B((-ROD_BACK + 0.001, 0.0, 0.0), (0.002, 0.017, 0.012), WHITE)],  # thumb rest
    mass=float(os.environ.get("INJ_PLUNGER_MASS", "0.002")), friction=GRIP_FRICTION, ignore=("syringe", "carrier", "needle"),
    contact_report=True, solver_iterations=32, **_AWAKE)  # INJ_PLUNGER_MASS: diagnostics only
CAP_AT = SYR_AT + [sum(CAP) / 2, 0.0, 0.0]
CAP_BODY = yb.Body.build("cap", tuple(CAP_AT), [B((0.0, 0.0, 0.0), (CAP[1] - CAP[0], 2 * CAP_HALF, 2 * CAP_HALF), ORANGE),
                                                 B((CAP[1] - CAP[0] - 0.004 - (CAP[1] - CAP[0]) / 2, 0.0, 0.0), (0.004, 2 * CAP_HALF + 0.0006, 2 * CAP_HALF + 0.0006), ORANGE, collide=False)],
                         mass=0.0015, friction=GRIP_FRICTION, torsional_patch=0.003, ignore=("syringe", "carrier", "needle", "plunger"),
                         contact_report=True, solver_iterations=32, **_AWAKE)

# joints: the needle's carrier (locked, or free to slide back into the barrel), the needle's sensing spring, the plunger,
# the cap's fit, and the needle in the tissue (rotation within 20 deg; authored off)
yb.prismatic_joint("J_slide", "syringe", "carrier", tuple(SYR_AT + [NEEDLE_L / 2, 0.0, 0.0]), axis="X", limits=(-(NEEDLE_L + 0.002), 0.0))
yb.fixed_joint("J_lock", "syringe", "carrier", tuple(SYR_AT + [NEEDLE_L / 2, 0.0, 0.0]))
yb.prismatic_joint("J_sense", "carrier", "needle", tuple(SYR_AT + [NEEDLE_L / 2, 0.0, 0.0]), axis="X", limits=(-0.004, 0.0005), drive=(K_SENSE, 0.5, 0.0))
yb.prismatic_joint("J_plunger", "syringe", "plunger", tuple(PLUNGER_AT), axis="X", limits=(0.0, TRAVEL_MAX),
                   drive=(0.0, FLOW_C, 0.0), articulated=True)
# the stopper's stiction (J_stick, below) and the 0 line's stop (STOPS: a D6 limit on the slide only), on from the
# start and switched OFF by the rule when the robot's contact load on the plunger passes each one's load. Spike 3: PhysX break forces broke under
# barrel loads with no thumb (the 15 N grip test, the cap's pull), and a free stopper on its damped slide crept several
# mm/s while the barrel was handled (to its stop during the needle's insertion); spike 2: an unbreakable D6 limit held
# under 15 N; spikes 1-3: switching a joint off at run time reaches the GPU (the carrier's lock, the cap's fit).
# J_stick: the stopper's static friction as a hard stop with STICK_PLAY of play, released by the rule at BREAKAWAY_F of
# measured thumb load: a D6 transX limit of exactly J_zero's form (lower -1 mm, upper > lower, no drive, no break force,
# re-armed at a reset as J_zero is). Probe 1 (INJ_PROBE): the spring it replaces (a transX drive authored at 2e4 N/m)
# acted on the GPU at about 1 kN/m (1 N pushed it 0.97 mm), and as the right tips loaded the barrel the solver drove the
# 2 g plunger 16.9 mm in about 5 physics steps with no contact on it, to J_zero's upper limit, which held there in every
# spike (and J_plunger's at 19.4 mm in spike 6). Before the spring, a (0, 0) transX lock did not hold either (spikes
# 2, 4, 5, 6, 7). So the stiction is now a limit with a range, as the stops that held.
STICK_PLAY = 0.0002  # m: 0.2 mm (0.012 mL) of play before the stop, within ZERO_TOL
yb.d6_joint("J_stick", "syringe", "plunger", tuple(PLUNGER_AT), tuple(PLUNGER_AT), (0.0, 0.0, 0.0, 1.0), {"transX": (-0.001, STICK_PLAY)})
for _name, _at, _ in STOPS:
    yb.d6_joint(_name, "syringe", "plunger", tuple(PLUNGER_AT), tuple(PLUNGER_AT), (0.0, 0.0, 0.0, 1.0), {"transX": (-0.001, _at)})
yb.prismatic_joint("J_cap", "syringe", "cap", tuple(CAP_AT), axis="X", limits=(0.0, 0.003), drive=(CAP_K, 5.0, 0.0))
yb.d6_joint("J_tissue", "trainer", "syringe", tuple(GEOM.site), tuple(SYR_AT + [NEEDLE_L, 0.0, 0.0]), (0.0, 0.0, 0.0, 1.0),
            {"rotY": (-ENGAGE_DEG, ENGAGE_DEG), "rotZ": (-ENGAGE_DEG, ENGAGE_DEG)}, on=False)

# INJ_PUSHONLY (diagnostics only, experiment 5; unset in every scored run): the barrel is held still on its rest by a
# ranged D6 stop (0.1 mm, 0.2 deg), authored on from the start, so the left finger's push can be tested alone on the
# production syringe (dynamic barrel, same plunger, joints and resistance)
PUSHONLY = os.environ.get("INJ_PUSHONLY", "")
if PUSHONLY:
    _hold = tuple(SYR_AT + [-0.040, 0.0, 0.0])
    yb.d6_joint("J_hold_rest", "rest", "syringe", _hold, _hold, (0.0, 0.0, 0.0, 1.0),
                {"transX": (-0.0001, 0.0001), "transY": (-0.0001, 0.0001), "transZ": (-0.0001, 0.0001),
                 "rotX": (-0.2, 0.2), "rotY": (-0.2, 0.2), "rotZ": (-0.2, 0.2)})

OBJECTS = [STAND, TRAINER, TRAY, REST, SHARPS, CUP, CAMERAS, SYRINGE, CARRIER, NEEDLE, PLUNGER, CAP_BODY]
STATIC = ("stand", "trainer", "tray", "rest", "sharps", "cup", "camera_mounts")
JOINT_OF = {"lock_off": [("J_lock", False)], "lock_on": [("J_lock", True)], "tissue_on": [("J_tissue", True)], "tissue_off": [("J_tissue", False)],
            "cap_off": [("J_cap", False)], "unstick": [("J_stick", False)], "detent": [("J_zero", False)], "retract": [("J_lock", False)],
            **{f"pass_{n}": [(n, False)] for n, _, _ in STOPS if n != "J_zero"}}

# footprints on the table (x0, x1, y0, y1), m: none may overlap (tested)
FOOTPRINTS = {"base": (0.25, 0.47, -0.11, 0.11), "tray": (0.15, 0.245, -0.04, 0.04), "rest": (0.12, 0.24, -0.13, -0.05),
              "sharps": (SHARPS_AT[0] - SH_X / 2, SHARPS_AT[0] + SH_X / 2, SHARPS_AT[1] - SH_Y / 2, SHARPS_AT[1] + SH_Y / 2),
              "cup": (CUP_AT[0] - CUP_R, CUP_AT[0] + CUP_R, CUP_AT[1] - CUP_R, CUP_AT[1] + CUP_R),
              "side_camera": (SIDE_CAM[0] - 0.05, SIDE_CAM[0] + 0.05, SIDE_CAM[1] - 0.02, SIDE_CAM[1] + 0.06)}


def static_boxes(names=None) -> list:
    """World AABBs of the static bodies' colliders: (name, lo, hi)."""
    out = []
    for o in OBJECTS:
        if getattr(o, "static", False) and (names is None or o.name in names):
            for s in o.shapes:
                if s.collide:
                    c = np.asarray(o.pos) + np.asarray(s.centre)
                    out.append((o.name, c - s.half_extent(), c + s.half_extent()))
    return out


def _in_shape(points, body, shape, margin: float) -> np.ndarray:
    """Which `points` (world) lie within `margin` of one box of a static body (in the box's own frame)."""
    loc = (np.atleast_2d(points) - (np.asarray(body.pos) + np.asarray(shape.centre))) @ yb.rotation(shape.quat_xyzw)
    return np.all(np.abs(loc) <= np.asarray(shape.size) / 2 + margin, axis=1)


LET_GO_S = 0.5  # s with no fingertip load on the barrel, needle in the trainer: let go


def needle_in_cap(seg, cap_state, margin: float = 0.0005) -> bool:
    """Any point of the needle's shaft (`seg`, world) within the cap's collider. The cap does not collide with the needle
    (filtered, so its fit can slide), so the bare needle back in the cap after uncapping is read from geometry."""
    seg = np.atleast_2d(np.asarray(seg, float))
    local = (seg - np.asarray(cap_state.pos, float)) @ cap_state.rotation()
    for sh in CAP_BODY.shapes:
        if not sh.collide:
            continue
        h = np.asarray(sh.size, float) / 2 + margin
        d = (local - np.asarray(sh.centre, float)) @ yb.rotation(sh.quat_xyzw)
        if bool((np.abs(d) <= h).all(axis=1).any()):
            return True
    return False


def shaft_touch(seg, engaged: bool, margin: float = 0.0005) -> str | None:
    """What the bare needle's shaft (points `seg`, world) touches besides the skin and the sharps container's inside:
    a fixture, the table, the sharps container's outside or lid top, or the trainer away from its skin (its back, sheet
    or the stand behind it); the trainer is not checked while the needle is in it."""
    seg = np.atleast_2d(np.asarray(seg, float))
    for o in OBJECTS:
        if not getattr(o, "static", False) or (o.name == "trainer" and engaged):
            continue
        for sh in o.shapes:
            if not sh.collide:
                continue
            hit = seg[_in_shape(seg, o, sh, margin)]
            if not len(hit):
                continue
            if o.name == "sharps":
                a, b = SHARPS_INSIDE
                inside = (hit[:, 0] >= a[0] - margin) & (hit[:, 0] <= b[0] + margin) & (hit[:, 1] >= a[1] - margin) & (hit[:, 1] <= b[1] + margin) \
                    & (hit[:, 2] <= b[2] + margin)
                if not inside.all():
                    return "the sharps container's outside"
            elif o.name == "trainer":
                if not all(GEOM.on_skin(p, 0.004) for p in hit):
                    return "the trainer away from its skin"
            else:
                return f"the {o.name}"
    if float(seg[:, 2].min()) < TOP + margin:
        return "the table"
    return None


def layout_errors(objects=None, geom: "Geometry | None" = None) -> list[str]:
    """The built static scene against the constants the rule and the setup text use (empty: they match)."""
    g = geom or GEOM
    objs = {o.name: o for o in (objects if objects is not None else OBJECTS)}
    errs = []
    for name in ("trainer", "sharps", "cup", "stand"):
        if name not in objs:
            errs.append(f"{name} missing")
    if errs:
        return errs

    def boxes(name):
        o = objs[name]
        return [(np.asarray(o.pos) + np.asarray(sh.centre) - sh.half_extent(), np.asarray(o.pos) + np.asarray(sh.centre) + sh.half_extent())
                for sh in o.shapes if sh.collide]

    face = [lo for lo, hi in boxes("trainer") if lo[1] >= -CENTRE_HALF - 1e-6 and hi[1] <= CENTRE_HALF + 1e-6]
    if not face or abs(min(float(lo[0]) for lo in face) - g.skin_x) > 0.0005:
        errs.append("the trainer's skin is not at its stated x")
    cap = [(lo, hi) for lo, hi in boxes("trainer") if abs(float(lo[0]) - (g.skin_x + SHOULDER_BACK)) < 1e-6]
    if not cap or abs(min(float(lo[2]) for lo, _ in cap) - g.z_edge) > 0.0005:
        errs.append("the acromion's edge is not at its stated height")
    sb = boxes("sharps")
    lo_all, hi_all = np.min([lo for lo, _ in sb], axis=0), np.max([hi for _, hi in sb], axis=0)
    fp = FOOTPRINTS["sharps"]
    if np.abs(np.r_[lo_all[:2], hi_all[:2]] - np.array([fp[0], fp[2], fp[1], fp[3]])).max() > 0.0005 or abs(float(hi_all[2]) - (TOP + SH_H)) > 0.0005:
        errs.append("the sharps container's outside is not its stated size")
    lid = [(lo, hi) for lo, hi in sb if float(lo[2]) > TOP + SH_H - SH_W - 1e-6]
    c = np.array(SHARPS_AT)
    probe = [c + [dx, dy] for dx in (-OPEN_X / 2 + 0.001, 0.0, OPEN_X / 2 - 0.001) for dy in (-OPEN_Y / 2 + 0.001, 0.0, OPEN_Y / 2 - 0.001)]
    rim = [c + [dx, dy] for dx, dy in ((OPEN_X / 2 + 0.001, 0.0), (-OPEN_X / 2 - 0.001, 0.0), (0.0, OPEN_Y / 2 + 0.001), (0.0, -OPEN_Y / 2 - 0.001))]

    def covered(p):
        return any(lo[0] <= p[0] <= hi[0] and lo[1] <= p[1] <= hi[1] for lo, hi in lid)

    if any(covered(p) for p in probe) or not all(covered(p) for p in rim):
        errs.append("the sharps container's opening is not its stated size")
    fl = [(lo, hi) for lo, hi in boxes("stand") if abs(float(hi[2] - lo[2]) - FLANGE_PLATE) < 1e-6]
    if not fl or abs(float(max(hi[0] for _, hi in fl) - min(lo[0] for lo, _ in fl)) - FLANGE_D) > 0.002:
        errs.append("the flange is not its stated diameter")
    cb = boxes("cup")
    clo, chi = np.min([lo for lo, _ in cb], axis=0), np.max([hi for _, hi in cb], axis=0)
    if np.abs((clo[:2] + chi[:2]) / 2 - np.asarray(CUP_AT)).max() > 0.001 or abs(float(chi[0] - clo[0]) / 2 - CUP_R) > 0.003:
        errs.append("the waste cup is not where or the size stated")
    return errs


def stage_errors(env) -> list[str]:
    """The static bodies as the simulator built them (their prims' world positions) against their specs."""
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Usd, UsdGeom

    stage = get_current_stage()
    origin = yb.as_numpy(env.unwrapped.scene.env_origins[0])
    errs = []
    for o in OBJECTS:
        if not getattr(o, "static", False):
            continue
        prim = stage.GetPrimAtPath(f"{ENVP}/{o.name}")
        if not prim or not prim.IsValid():
            errs.append(f"the {o.name} is missing from the scene")
            continue
        t = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation(), float) - origin
        if float(np.abs(t - np.asarray(o.pos)).max()) > 0.002:
            errs.append(f"the {o.name} is at {np.round(1000 * t, 1).tolist()} mm, not {np.round(1000 * np.asarray(o.pos), 1).tolist()}")
    return errs


def in_container(lo, hi) -> bool:
    a, b = SHARPS_INSIDE
    return bool((np.asarray(lo) >= a - 0.002).all() and (np.asarray(hi) <= b + 0.002).all())


CUP_WALL, CUP_SIDES = 0.002, 10  # as built by _ring for CUP (keep in step with CUP above)
CUP_INNER = CUP_R - CUP_WALL  # the wall's inner faces' distance from the cup's axis (the decagon's apothem), 43 mm
CONTACT_TOL = 0.001  # a resting part's corner may sit this far into a wall (contact compliance), m


def _cup_points(lo, hi, points):
    if points is None:
        lo, hi = np.asarray(lo), np.asarray(hi)
        points = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    return np.atleast_2d(np.asarray(points, float))


def cup_excess(lo, hi, points=None) -> float:
    """How far (m) the worst corner lies outside the cup's inside: the decagon of the wall's inner faces (as built,
    faces at 2 pi k / 10, CUP_INNER from the axis), its floor top and its rim. <= 0 is inside. Spike 10: a single
    radius of 43 mm (the faces' distance, not the corners' 45.2 mm) read a cap resting against the wall near a corner
    of the decagon as outside, though the top camera showed it inside."""
    p = _cup_points(lo, hi, points)
    d = p[:, :2] - np.asarray(CUP_AT)
    ang = 2 * math.pi * np.arange(CUP_SIDES) / CUP_SIDES
    n = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    radial = float((d @ n.T).max()) - CUP_INNER
    floor = float((TOP + 0.003) - p[:, 2].min())  # the cup's floor disc is 3 mm thick
    rim = float(p[:, 2].max() - (TOP + CUP_H))
    return max(radial, floor, rim)


def in_cup(lo, hi, points=None) -> bool:
    """Wholly inside the cup: the corners of the axis-aligned box lo-hi, or (exact, when given) the corners of the
    body's own boxes (`points`, [n, 3], world), within the decagon of the wall's inner faces, above its floor and
    below its rim, each to CONTACT_TOL."""
    return bool(cup_excess(lo, hi, points) <= CONTACT_TOL)


def container_excess(points) -> dict:
    """Per face of the sharps container's inside (SHARPS_INSIDE), how far (mm) the worst corner lies beyond it;
    > 2 fails `in_container`. For the log, so a failed stow check names the face it crosses."""
    p = np.atleast_2d(np.asarray(points, float))
    a, b = SHARPS_INSIDE
    lo, hi = p.min(0), p.max(0)
    names = ("x-", "y-", "floor", "x+", "y+", "lid")
    vals = [*(a - lo), *(hi - b)]
    return {k: round(1000 * float(v), 1) for k, v in zip(names, vals)}


def stow_check(syringe_state, plunger_state, cap_state) -> dict:
    """The stow checks, one function for the rule and the demo's log: the syringe with its plunger rod (at its current
    travel) wholly inside the sharps container, below the rim; the cap wholly inside the cup. Returns the two flags and,
    for the log, how far each part's worst corner lies past each face (mm; container: > 2 fails, cup: > 1 fails)."""
    sy_pts = np.vstack([shape_corners(SYRINGE, syringe_state), shape_corners(PLUNGER, plunger_state)])
    cap_pts = shape_corners(CAP_BODY, cap_state)
    return {"in_container": in_container(sy_pts.min(0), sy_pts.max(0)), "cap_in_cup": in_cup(None, None, cap_pts),
            "container_excess": container_excess(sy_pts), "barrel_excess": container_excess(shape_corners(SYRINGE, syringe_state)),
            "rod_excess": container_excess(shape_corners(PLUNGER, plunger_state)),
            "cup_excess_mm": round(1000 * cup_excess(None, None, cap_pts), 1)}


def shape_corners(spec, state) -> np.ndarray:
    """The world corners of every box a body is made of ([8 n, 3]). Spike 10: the body-wide box (the syringe's 25 mm
    flange height along its whole length; the cap's box turned) reached past the container's floor and the cup's wall
    with both visibly inside, so the stow checks use the boxes themselves (exact for boxes in a box or a cylinder)."""
    pts = []
    for sh in spec.shapes:
        Rs = yb.rotation(sh.quat_xyzw)
        h = np.asarray(sh.size, float) / 2
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    pts.append(np.asarray(sh.centre, float) + Rs @ (h * [sx, sy, sz]))
    return np.asarray(pts) @ state.rotation().T + np.asarray(state.pos, float)


# =====================================================================================================================
# Sensing: from the simulator to `Sample` and `Status` (runs only inside Isaac Sim)
# =====================================================================================================================

ENVP = yb.ENV
FINGER_BODIES = ("tip_left", "tip_right")
ARM_LINKS = ("link1", "link2", "link3", "link4", "link5", "gripper", "tip_left", "tip_right")
EXTRA_PUSH = np.zeros(3)  # N on the syringe: the spike's grip-slip test sets it
SENSE: dict = {"samples": 0}  # the latest readings, for logs and the demo's closed loops (as a wrist force sensor reads)
_LAST: dict = {}


def _link_paths() -> dict:
    if "links" not in _LAST:
        from yam_arena.robot import asset_link_paths

        _LAST["links"] = asset_link_paths()
    return _LAST["links"]


def _arm_globs(side: str, names=("gripper",) + FINGER_BODIES) -> list[str]:
    prim = {"left": "LeftArm", "right": "RightArm"}[side]
    return [f"{ENVP}/{prim}{_link_paths()[n]}" for n in names]


def _forces(key, body, filters, dt) -> np.ndarray:
    """Forces on `body` from each explicit filter (never an empty filter list: on the GPU that reads nothing)."""
    return yb.force_matrix(yb.contact_view(key, f"{ENVP}/{body}/body", filters), dt)


def grip_report(env, side: str, body: str = "syringe") -> str:
    """This arm's finger joints and its fingertips' contact forces on `body`, for the logs."""
    robot = env.unwrapped.scene[f"{side}_robot"]
    q = yb.as_numpy(robot.data.joint_pos)[0]
    j = [float(q[robot.find_joints(n)[0][0]]) for n in ("joint7", "joint8")]
    f = np.linalg.norm(_forces(f"{body}_{side}_tips", body, _arm_globs(side, FINGER_BODIES), env.unwrapped.physics_dt), axis=1)
    return f"{side} fingers at {1000 * j[0]:.1f}, {1000 * j[1]:.1f} mm (open -46.95, closed 0), tip forces on the {body} {np.round(f, 1).tolist()} N"


def tcp_of(env, side: str) -> np.ndarray:
    """The simulator's TCP: the `gripper` body's pose plus the TCP offset."""
    from yam_arena.constants import TCP_OFFSET

    base = env.unwrapped
    robot = base.scene[f"{side}_robot"]
    i = robot.find_bodies("gripper")[0][0]
    pose = yb.as_numpy(robot.data.body_link_pose_w)[0, i]
    return pose[:3] - yb.as_numpy(base.scene.env_origins[0]) + yb.rotation(pose[3:7]) @ np.asarray(TCP_OFFSET)


def gripper_points(env) -> np.ndarray:
    """Points on both arms' fingers (world): tip and 3 cm up, middle and both edges of each finger. [24, 3]: left, right."""
    base = env.unwrapped
    origin = yb.as_numpy(base.scene.env_origins[0])
    out = []
    for side in ("left", "right"):
        robot = base.scene[f"{side}_robot"]
        i = robot.find_bodies("gripper")[0][0]
        pose = yb.as_numpy(robot.data.body_link_pose_w)[0, i]
        R = yb.rotation(pose[3:7])
        opening = abs(float(yb.as_numpy(robot.data.joint_pos)[0, robot.find_joints("joint7")[0][0]]))
        for sx in (-1.0, 1.0):
            for dy in (-0.013, 0.0, 0.013):
                for dz in (0.147, 0.117):
                    out.append(pose[:3] - origin + R @ np.array([sx * (opening + 0.0045), dy, dz]))
    return np.array(out)


_MESH: dict = {}


def link_points_local() -> dict:
    """Points on each arm link's visual mesh, in the link's frame (sampled once; the visibility's occluders)."""
    if not _MESH:
        import xml.etree.ElementTree as ET

        import trimesh
        from yam_arena import YAM_URDF
        from yam_arena.urdf import rpy_matrix

        root = os.path.dirname(YAM_URDF)
        for link in ET.parse(YAM_URDF).getroot().findall("link"):
            v = link.find("visual")
            mesh = None if v is None else v.find("geometry/mesh")
            o = None if v is None else v.find("origin")
            if mesh is None or o is None or link.get("name") not in ARM_LINKS:
                continue
            m = trimesh.load(os.path.join(root, mesh.get("filename", "")))
            M = np.eye(4)
            M[:3, :3] = rpy_matrix(np.array([float(x) for x in o.get("rpy", "0 0 0").split()]))
            M[:3, 3] = [float(x) for x in o.get("xyz", "0 0 0").split()]
            p = np.asarray(trimesh.sample.sample_surface_even(m, 600, seed=0)[0])
            _MESH[link.get("name", "")] = (M[:3, :3] @ p.T).T + M[:3, 3]
    return _MESH


def arm_points_sim(env) -> np.ndarray:
    base = env.unwrapped
    origin = yb.as_numpy(base.scene.env_origins[0])
    pts = []
    for side in ("left", "right"):
        robot = base.scene[f"{side}_robot"]
        poses = yb.as_numpy(robot.data.body_link_pose_w)[0]
        for name, P in link_points_local().items():
            i = robot.find_bodies(name)[0][0]
            pts.append(P @ yb.rotation(poses[i, 3:7]).T + poses[i, :3] - origin)
    return np.vstack(pts)


def box_distance(points, lo, hi) -> np.ndarray:
    p = np.atleast_2d(points)
    return np.linalg.norm(np.maximum(np.maximum(lo - p, p - hi), 0.0), axis=1)


def _rel(S: dict, a: str, b: str, rest) -> float:
    """b's slide along a's x from its rest offset (S: the bodies' states, read once per physics step)."""
    A, Bs = S[a], S[b]
    return float((A.rotation().T @ (Bs.pos - A.pos))[0] - rest)


def syringe_push(trial: "Trial", a) -> np.ndarray:
    """The force the sensing holds on the barrel (world): the tissue's, and the reactions of the carrier's and the
    needle's pushes and of the stopper's glide friction (equal and opposite to `plunger_n`, so the pair conserves
    momentum). Spike 1's reaction fed back into the plunger only through the velocity misread (fixed): the glide is now
    elasto-plastic in the slide's own displacement, so a load on the barrel cannot drive the stopper through it."""
    a = _unit(a)
    return np.asarray(trial.syr_force, float) - trial.carrier_n * a - trial.needle_n * a - trial.plunger_n * a


PUSH_LINKS = (FINGER_BODIES, ("gripper",))  # one contact view each: the fingertips (as the verified grip readings) and the hand


def plunger_contacts(env, side: str) -> np.ndarray:
    """One arm's contact forces on the plunger ([n, 3], world). Spike 3 read 0 for the whole run from a single view
    filtered on all eight arm links; the fingertips' view is built exactly as the grip readings that worked (the tips on
    the syringe, 25 N), and the hand's own body is a second view, so a bad one cannot blank the other."""
    out = []
    for names in PUSH_LINKS:
        try:
            out.append(_forces(f"plunger_{side}_{names[0]}", "plunger", _arm_globs(side, names), env.unwrapped.physics_dt))
        except (RuntimeError, KeyError, ValueError, IndexError) as e:
            SENSE[f"view_error_{side}_{names[0]}"] = type(e).__name__
    return np.vstack(out) if out else np.zeros((0, 3))


def thumb_load(forces, a) -> float:
    """The robot's contact forces on the plunger ([n, 3], world) summed along the axis `a`, forward positive: the thumb."""
    f = np.asarray(forces, float).reshape(-1, 3)
    return float(max(0.0, float(f.sum(axis=0) @ _unit(a))))


def _sample(env, dt: float) -> None:
    S = {}
    for n in ("syringe", "carrier", "needle", "plunger", "cap"):
        try:
            S[n] = yb.body_state(env, n)
        except (KeyError, IndexError, AttributeError) as e:
            TRIAL.void(f"the {n} is missing from the scene ({type(e).__name__})")
            return
    sy = S["syringe"]
    R = sy.rotation()
    a = R[:, 0]
    travel = _rel(S, "syringe", "plunger", EDGE0)
    # the slide's own velocity, from its position (the bodies' centre-of-mass velocities read about 5-10 mm/s, and in
    # spike 4 about 1 m/s, along the axis with the slide still; fed to the glide friction that offset pushed the stopper
    # at 2 N / 770 N s/m = 2.6 mm/s: spike 3's creep at 2.7-3.1 mm/s)
    prev = _LAST.get("travel")
    tv = 0.0 if prev is None else (travel - float(prev)) / dt
    _LAST["travel"] = travel
    cap_slide = _rel(S, "syringe", "cap", sum(CAP) / 2)
    sense = _rel(S, "carrier", "needle", 0.0)  # negative: compressed
    carrier = _rel(S, "syringe", "carrier", NEEDLE_L / 2)
    nd = S["needle"]
    tip = nd.pos + nd.rotation()[:, 0] * NEEDLE_L / 2
    com = sy.pos + R @ np.asarray(SYRINGE.com)
    # the tip's velocity from its position, low-passed (TipVel): the solver's velocity of the gripped barrel carries a
    # phantom +-80-140 mm/s with its position still (spike 15a), and the tissue's stick-slip and lateral damping fed by it
    # kicked the barrel every step, which the stopper's free slide rectified into a 5-10 mm/s drift (spikes 10-18;
    # spike 19a: tissue force alone drifts, 19b: the needle's and carrier's pushes alone do not, 19c: this, all forces
    # on, -0.00 mm in 2 s after the breakaway)
    tip_vel = _TIP_V.step(tip, dt)
    force = max(0.0, -K_SENSE * sense)
    touch = None
    if not TRIAL.cap_on and not TRIAL.retracted and force > 0.3 and not GEOM.on_skin(tip, 0.004) and not in_container(tip, tip):
        touch = "something (its tip pressed off the skin)"
    n_axis = tuple(float(v) for v in nd.rotation()[:, 0])
    SENSE["needle_deg"] = _deg(n_axis, a)
    SENSE["needle_deg_max"] = max(SENSE.get("needle_deg_max", 0.0), SENSE["needle_deg"]) if not TRIAL.cap_on else 0.0
    thumb = 0.0
    if not TRIAL.retracted:
        per = {s: plunger_contacts(env, s) for s in ("left", "right")}
        thumb = thumb_load(np.vstack(list(per.values())), a)
        for s, f in per.items():  # for the trace: each arm's contact force on the plunger, world, the largest since the last line
            v = np.asarray(f, float).reshape(-1, 3).sum(axis=0)
            if float(np.linalg.norm(v)) >= float(np.linalg.norm(SENSE.get(f"plunger_{s}", np.zeros(3)))):
                SENSE[f"plunger_{s}"] = v
    smp = Sample(dt=dt, hub=tuple(sy.pos), axis=tuple(a), tip=tuple(tip), tip_vel=tuple(tip_vel), tip_force=force, travel=travel, travel_vel=tv,
                 thumb_n=thumb, cap_on=TRIAL.cap_on, cap_slide=cap_slide, needle_axis=n_axis, touch=touch,
                 quats=tuple(tuple(float(v) for v in S[n].quat_xyzw) for n in S))
    acts = TRIAL.feed(smp)
    if not TRIAL.locked and not TRIAL.engaged and not TRIAL.retracted and carrier > -0.001:
        acts += TRIAL.carrier_home()
    for act in acts:
        for j, on in JOINT_OF[act]:
            yb.set_joint(j, on)
        print(f"[inj] t={TRIAL.t:.2f} s: {act}: {TRIAL.report()}", flush=True)
    probe_f = _probe(env, S, travel, a) if PROBE else np.zeros(3)
    if ZERO_WRENCH:  # diagnostics: "2" nothing on the syringe's bodies; "1" the glide pair only; "3" the glide pair and the
        # tissue's force and torque on the barrel; "4" the glide pair and the needle's and carrier's pushes
        glide = 0.0 if ZERO_WRENCH == "2" else TRIAL.plunger_n
        tissue = ZERO_WRENCH == "3"
        pushes = ZERO_WRENCH == "4"
        sy_f = -glide * a + (TRIAL.syr_force if tissue else 0.0) - ((TRIAL.carrier_n + TRIAL.needle_n) * a if pushes else 0.0)
        wrench = {"syringe": (sy_f, np.cross(tip - com, TRIAL.syr_force) if tissue else np.zeros(3)), "plunger": (glide * a, np.zeros(3)),
                  "carrier": (TRIAL.carrier_n * a if pushes else np.zeros(3), np.zeros(3)),
                  "needle": (TRIAL.needle_n * a if pushes else np.zeros(3), np.zeros(3))}
    else:
        wrench = {"syringe": (syringe_push(TRIAL, a) + EXTRA_PUSH, np.cross(tip - com, TRIAL.syr_force)),
                  "plunger": ((TRIAL.plunger_n + PLUNGER_JIG[0]) * a + probe_f, np.zeros(3)),
                  "carrier": (TRIAL.carrier_n * a, np.zeros(3)), "needle": (TRIAL.needle_n * a, np.zeros(3))}
    for n, (f, tq) in wrench.items():  # written every physics step, so nothing stays on from an earlier window
        yb.push(env, n, tuple(np.asarray(f, float)), tuple(np.asarray(tq, float)))
    SENSE["wrench_set"] = {n: (np.round(np.asarray(f, float), 3).tolist(), np.round(np.asarray(tq, float), 4).tolist()) for n, (f, tq) in wrench.items()}
    off_skin = not TRIAL.engaged and not GEOM.on_skin(tip, 0.003) and float(tip[0]) < GEOM.skin_x - 0.003
    run = 0.0 if thumb >= THUMB_ON_N - 1e-9 else SENSE.get("thumb_off_run", 0.0) + dt  # (diagnostics: the thumb's longest time off)
    SENSE.update(thumb_off_run=run, thumb_off_max=max(run, SENSE.get("thumb_off_max", 0.0)), thumb_lo=min(thumb, SENSE.get("thumb_lo", 1e9)))
    SENSE.update(travel=travel, travel_vel=tv, tip_force=force, thumb=thumb, thumb_max=max(thumb, SENSE.get("thumb_max", 0.0)),
                 thumb_win=max(thumb, SENSE.get("thumb_win", 0.0)), phantom=max(SENSE.get("phantom", 0.0), force if off_skin and not TRIAL.cap_on and not TRIAL.retracted else 0.0),
                 cap_slide=cap_slide, carrier=carrier, gap=TRIAL.gap, in_muscle=TRIAL.in_muscle_now, samples=SENSE["samples"] + 1)


def _sleeping(path: str):
    """PhysX's sleep flag for a body prim (probe only); None when this build cannot say."""
    try:
        from omni.physx import get_physx_interface

        return bool(get_physx_interface().is_sleeping(path))
    except Exception as e:  # noqa: BLE001  (diagnostics: any failure is reported once, never raised)
        if "sleep_err" not in _LAST:
            _LAST["sleep_err"] = f"{type(e).__name__}: {e}"
            print(f"[probe] is_sleeping unavailable: {_LAST['sleep_err']}", flush=True)
        return None


def _probe(env, S: dict, travel: float, a) -> np.ndarray:
    """INJ_PROBE only: the test push on the plunger (N, world) for this physics step, and one [probe] line per physics
    step in the two windows. Every contact view is made from the first step on, so none is created inside a window."""
    t = TRIAL.t
    statics = [f"{ENVP}/{n}/body" for n in STATIC] + [f"{ENVP}/{n}/body" for n in ("syringe", "carrier", "needle", "cap")]
    dt = env.unwrapped.physics_dt
    fr, fo = {"left_all": np.zeros((1, 3)), "right_all": np.zeros((1, 3))}, np.zeros((1, 3))
    try:
        fr = {f"{s}_all": _forces(f"probe_plunger_{s}_all", "plunger", _arm_globs(s, ARM_LINKS), dt) for s in ("left", "right")}
        fo = _forces("probe_plunger_others", "plunger", statics, dt)
    except Exception as e:  # noqa: BLE001  (diagnostics: reported once, never raised)
        if "view_err" not in _LAST:
            _LAST["view_err"] = f"{type(e).__name__}: {e}"
            print(f"[probe] contact view failed: {_LAST['view_err']}", flush=True)
    push = 1.0 * np.asarray(a, float) if 5.0 <= t < 5.5 else np.zeros(3)
    if TRIAL.stuck:
        _LAST["t_unstick"] = None
    elif _LAST.get("t_unstick") is None:
        _LAST["t_unstick"] = t
    tu = _LAST.get("t_unstick")
    if not (4.9 <= t < 5.8 or 7.3 <= t < 8.2 or (tu is not None and t - tu < 1.0)):  # and 1 s from the breakaway
        return push
    sy, pl = S["syringe"], S["plunger"]
    R = sy.rotation()
    rel = R.T @ (pl.pos - sy.pos) - np.array([EDGE0, 0.0, 0.0])
    ang = math.degrees(math.acos(max(-1.0, min(1.0, float(pl.rotation()[:, 0] @ R[:, 0])))))
    base = env.unwrapped
    robot = base.scene["right_robot"]
    q = yb.as_numpy(robot.data.joint_pos)[0]
    fing = [round(1000 * float(q[robot.find_joints(n)[0][0]]), 2) for n in ("joint7", "joint8")]
    link_v = {}
    for n in ("syringe", "plunger"):
        d = base.scene[n].data
        link_v[n] = np.round(1000 * yb.as_numpy(d.root_link_vel_w)[0][:3], 2).tolist() if hasattr(d, "root_link_vel_w") else None
    # the slide's own velocity as the solver has it (these joints are maximal-coordinate, outside any articulation, so
    # there is no joint-velocity readout): the plunger's link velocity minus the barrel's at the plunger's origin, along
    # the axis; next to it the position-derived one the rule uses
    v_rel = None
    try:
        vs = yb.as_numpy(base.scene["syringe"].data.root_link_vel_w)[0]
        vp = yb.as_numpy(base.scene["plunger"].data.root_link_vel_w)[0]
        v_rel = round(1000 * float((vp[:3] - (vs[:3] + np.cross(vs[3:6], pl.pos - sy.pos))) @ R[:, 0]), 2)
        syr_w = np.round(vs[3:6], 3).tolist()
    except (AttributeError, KeyError, IndexError, TypeError):
        syr_w = None
    sl = {n: _sleeping(f"{ENVP}/{n}/body") for n in ("syringe", "plunger", "carrier")}
    tips = _forces("syringe_right_tips", "syringe", _arm_globs("right", FINGER_BODIES), dt)
    print(f"[probe] t={t:.5f} travel {1000 * travel:.3f} rel_mm {np.round(1000 * rel, 3).tolist()} tilt {ang:.3f} deg "
          f"syr_pos {np.round(1000 * sy.pos, 3).tolist()} pl_pos {np.round(1000 * pl.pos, 3).tolist()} "
          f"syr_comv {np.round(1000 * sy.lin_vel, 2).tolist()} pl_comv {np.round(1000 * pl.lin_vel, 2).tolist()} link_v {link_v} "
          f"pl_angv {np.round(pl.ang_vel, 3).tolist()} fingers {fing} tips_N {np.round(np.linalg.norm(tips, axis=1), 2).tolist()} "
          f"pl_contact L {np.round(fr['left_all'].sum(0), 3).tolist()} R {np.round(fr['right_all'].sum(0), 3).tolist()} "
          f"others {np.round(fo, 3).tolist()} push {np.round(push, 2).tolist()} sleeping {sl} plunger_n {TRIAL.plunger_n:.3f} "
          f"extra {np.round(EXTRA_PUSH, 2).tolist()} stuck {TRIAL.stuck} slide_v solver {v_rel} mm/s, from position "
          f"{1000 * SENSE.get('travel_vel', 0.0):.2f} mm/s, syr_angv {syr_w}", flush=True)
    return push


@yb.on_substep
def _on_substep(env, dt: float) -> None:
    _sample(env, dt)


def _gripper_clear(env, names=("syringe", "cap"), side=None) -> float:
    pts = gripper_points(env)
    if side is not None:  # gripper_points lists the left gripper's 12 points, then the right's
        pts = pts[:12] if side == "left" else pts[12:]
    d = 1.0
    for n in names:
        s = yb.body_state(env, n)
        lo, hi = (SYRINGE if n == "syringe" else CAP_BODY).local_bounds()
        d = min(d, float(box_distance((pts - s.pos) @ s.rotation(), lo, hi).min()))
    return d


def _body_props(name: str) -> dict:
    """A free body's authored PhysX rigid-body settings (diagnostics): damping, velocity caps, iterations, mass."""
    try:
        from isaaclab.sim.utils.stage import get_current_stage

        prim = get_current_stage().GetPrimAtPath(f"{ENVP}/{name}/body")
        out = {}
        for a in ("physxRigidBody:linearDamping", "physxRigidBody:angularDamping", "physxRigidBody:maxLinearVelocity",
                  "physxRigidBody:maxAngularVelocity", "physxRigidBody:maxDepenetrationVelocity",
                  "physxRigidBody:solverPositionIterationCount", "physxRigidBody:solverVelocityIterationCount", "physics:mass"):
            at = prim.GetAttribute(a)
            out[a.split(":")[1]] = at.Get() if at and at.HasAuthoredValue() else "default"
        return out
    except Exception as e:  # noqa: BLE001  (diagnostics: never raised)
        return {"error": type(e).__name__}


@yb.on_step
def _on_step(env) -> None:
    _LAST["env"] = env
    if not SENSE.get("layout_checked"):  # the first step: the scene as built against the constants
        SENSE["layout_checked"] = True
        if os.environ.get("INJ_TRACE"):
            for n in ("syringe", "plunger", "carrier", "needle"):
                print(f"[bodies] {n}: {_body_props(n)}", flush=True)
        errs = layout_errors() + stage_errors(env)
        print(f"[inj] layout check: {'ok' if not errs else errs}", flush=True)
        if errs:
            TRIAL.void("the scene does not match its specification: " + "; ".join(errs))
    if SENSE["samples"] == 0:  # the scene's update was not hooked: one reading per control step
        _sample(env, CONTROL_DT)
    SENSE["samples"] = 0
    base = env.unwrapped
    sy, cap = yb.body_state(env, "syringe"), yb.body_state(env, "cap")
    touch = None
    if not TRIAL.cap_on and not TRIAL.retracted:
        dt = base.physics_dt
        for side in ("left", "right"):
            f = _forces(f"needle_{side}_arm", "needle", _arm_globs(side, ARM_LINKS), dt)  # every link of the arm
            if float(np.linalg.norm(f, axis=1).max()) > 0.05:
                touch = f"the {side} arm"
        nd = yb.body_state(env, "needle")
        seg = nd.pos + np.outer(np.linspace(-0.5, 0.5, 9) * NEEDLE_L, nd.rotation()[:, 0])
        touch = shaft_touch(seg, TRIAL.engaged) or touch
        # the cap: the needle leaves it while uncapping; back in it after that (recapping, or a touch on the dropped cap)
        if needle_in_cap(seg, cap):
            if _LAST.get("cap_cleared"):
                touch = touch or "the cap after uncapping"
        else:
            _LAST["cap_cleared"] = True
    lo, hi = sy.low, sy.high
    clear = _gripper_clear(env)
    SENSE["clear_sides"] = {side: _gripper_clear(env, side=side) for side in ("left", "right")}
    touching = sum(float(np.linalg.norm(_forces(f"syringe_{s}_tips", "syringe", _arm_globs(s, FINGER_BODIES), base.physics_dt), axis=1).sum())
                   for s in ("left", "right"))
    resting = float(np.linalg.norm(sy.lin_vel)) < 0.02 and float(np.linalg.norm(sy.ang_vel)) < 0.3
    # the syringe with its plunger rod (the thumb rest stands about 4.5 mm proud of the barrel's end): wholly inside
    stow = stow_check(sy, yb.body_state(env, "plunger"), cap)
    inside = stow["in_container"]
    SENSE["stow"] = stow
    # set down anywhere but inside the container counts, its rest included (a bare needle is never put back down)
    SENSE["loose"] = SENSE.get("loose", 0) + 1 if (touching < 0.3 and resting and not inside and not TRIAL.engaged) else 0
    SENSE["let_go_n"] = SENSE.get("let_go_n", 0) + 1 if (TRIAL.engaged and touching < 0.3) else 0
    hidden = False
    if TRIAL.window:
        occ = arm_points_sim(env)
        hidden, views = hidden_frame(sy.pos, sy.rotation(), occ, [(lo_, hi_) for _, lo_, hi_ in static_boxes(("trainer", "stand"))])
        SENSE["views"] = SENSE.get("views", 0) + 1
        SENSE["views_ok"] = SENSE.get("views_ok", 0) + (0 if hidden else 1)
        SENSE["side_ok"] = SENSE.get("side_ok", 0) + int(views["side"]["ok"])
        SENSE["over_ok"] = SENSE.get("over_ok", 0) + int(views["over"]["ok"])
        edge = sy.pos + sy.rotation() @ np.array([EDGE0 + SENSE.get("travel", 0.0), BARREL_HALF, 0.0])
        SENSE["edge_seen"] = SENSE.get("edge_seen", 0) + int(not _segment_blocked(SIDE_CAM, edge, occ, []))
        SENSE["band_y_err"] = max(SENSE.get("band_y_err", 0.0), abs(hub_y_from_band(sy.pos, sy.rotation()) - float(sy.pos[1])))
    st = Status(in_container=inside, cap_in_cup=stow["cap_in_cup"], clear=clear, dropped=SENSE["loose"] >= 10, hidden=hidden, touch=touch,
                let_go=SENSE["let_go_n"] * CONTROL_DT >= LET_GO_S - 1e-9)
    TRIAL.status(st)
    SENSE.update(clear=clear, inside=inside, touching=touching)
    for side in ("left", "right"):  # the largest torque each arm's joints have applied (the model's actuators allow +-10 N m)
        try:
            tq = np.abs(yb.as_numpy(base.scene[f"{side}_robot"].data.applied_torque)[0][:6])
            SENSE[f"torque_{side}"] = np.maximum(SENSE.get(f"torque_{side}", np.zeros(6)), tq)
        except (AttributeError, KeyError, IndexError, TypeError):  # no applied-torque buffer in this build: nothing to log
            continue
    if os.environ.get("INJ_TRACE") and round(TRIAL.t / CONTROL_DT) % 5 == 0:
        print(plunger_trace(), flush=True)
    if os.environ.get("INJ_TRACE") and round(TRIAL.t / CONTROL_DT) % 20 == 0:
        print(f"[inj] t={TRIAL.t:.1f}: {TRIAL.report()} travel {1000 * SENSE.get('travel', 0):.2f} mm tip {SENSE.get('tip_force', 0):.2f} N "
              f"cap {1000 * SENSE.get('cap_slide', 0):.2f} mm carrier {1000 * SENSE.get('carrier', 0):.1f} mm clear {1000 * clear:.0f} mm "
              f"syringe {np.round(1000 * sy.pos, 1).tolist()} cap {np.round(1000 * cap.pos, 1).tolist()}", flush=True)


def plunger_trace() -> str:
    """Every 0.25 s with INJ_TRACE: the plunger's joints as the stage has them now (enabled, break force; the GPU's
    broken state is not readable, so `holds` is read from the measured slide: J_stick holds while travel stays within
    its 0.2 mm of play (plus 0.05 mm) of 1.0 mL), the slide's position and velocity, the applied glide force, the largest thumb load and each arm's largest contact force on
    the plunger since the last line, and the rule's record."""
    js = {}
    for n in ("J_stick", *[n for n, _, _ in STOPS], "J_plunger", "J_lock"):
        try:
            js[n] = yb.joint_attrs(n)
        except (KeyError, AttributeError, RuntimeError, ImportError) as e:
            js[n] = {"error": type(e).__name__}
    tr = SENSE.get("travel", 0.0)
    out = (f"[plunger] t={TRIAL.t:.2f}: travel {1000 * tr:.2f} mm, vel {1000 * SENSE.get('travel_vel', 0.0):.2f} mm/s, "
           f"thumb {SENSE.get('thumb_win', 0.0):.2f} N, glide {TRIAL.plunger_n:+.2f} N (shear {1000 * TRIAL.glide.z:+.2f} mm, "
           f"mean sliding {TRIAL.glide.mean_sliding():.2f} N, flips {TRIAL.glide.flips}), contact L {np.round(SENSE.get('plunger_left', np.zeros(3)), 2).tolist()} "
           f"R {np.round(SENSE.get('plunger_right', np.zeros(3)), 2).tolist()} N; J_stick {js['J_stick']} holds {abs(tr) < STICK_PLAY + 0.00005}; "
           + "".join(f"{n} {js[n].get('enabled', js[n])} " for n, _, _ in STOPS)
           + f"; within the rule's limit {tr <= TRIAL.limit() + 0.0002}; J_plunger {js['J_plunger']}; J_lock {js['J_lock']}; "
           f"rule stuck {TRIAL.stuck} stops passed {TRIAL.passed}/{len(STOPS)} limit {1000 * TRIAL.limit():.2f} mm retracted {TRIAL.retracted}; "
           f"wrench set {SENSE.get('wrench_set')}")
    SENSE["thumb_win"] = 0.0
    SENSE.pop("plunger_left", None)
    SENSE.pop("plunger_right", None)
    return out


@yb.on_reset
def _on_reset(env) -> None:
    Trial.__init__(TRIAL)
    SENSE.clear()
    SENSE.update(samples=0)
    _LAST.pop("travel", None)
    EXTRA_PUSH[:] = 0.0
    PLUNGER_JIG[0] = 0.0
    _LAST.pop("cap_cleared", None)
    _TIP_V.__init__()


# =====================================================================================================================
# The scripted demo (demo.py), played by `--policy yam_scripted` closed loop through yam_demokit's bridge
# =====================================================================================================================

import yam_demokit  # noqa: F401  (hooks yam_scripted to run demo_script)
from demo import demo_script  # noqa: F401


def demo():
    """yam_scripted needs a demo(); the bridge in yam_demokit plays demo_script(ctx) instead."""
    return {}
