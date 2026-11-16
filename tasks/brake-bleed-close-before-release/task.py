"""Bleeding a brake caliper by the two-person pedal method, on a bench brake trainer. A brake pedal on a stand works a
master cylinder whose steel line runs to a disc-brake caliper on a bracket. An 11 mm box-end wrench is already on the
caliper's bleeder screw (an 11 mm hex), and a clear hose runs from the bleeder's nipple into a catch jar, its end under the old fluid.
There is air in the caliper (the line was opened for a new caliper), so the pedal is spongy.

One hand presses the pedal and keeps it down; the other opens the bleeder a quarter turn with the wrench, so fluid and
air push out into the jar, and shuts it again while the pedal is still down; only then does the pedal come up.
Repeated until no more air comes out. Letting the pedal up while the bleeder is still open is the classic mistake: the
master cylinder's piston springs back and pulls on the line, and air can be drawn back in at the open bleeder, so the
pedal can stay spongy however many strokes are done.

Kits: yamkit (props, the pedal and the bleeder wrench as hinged links, gripper states), mechkit (the pedal's master
cylinder, mk.Piston: its return spring and its stop on the fluid; the wrench's shut/open detents), livekit (the line,
the jar and the reservoir: the air left in the caliper and the latched "air_in" hazard), guardkit (gk.touching: a
gripper's own shape in contact with the pedal pad or the wrench). livekit is imported last so
its step runs after yamkit's joint update.
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import yamkit as kit  # noqa: E402
import mechkit as mk  # noqa: E402
import guardkit as gk  # noqa: E402
import livekit as lk  # noqa: E402  (LAST: it wraps the other kits' per-step updates)
from yam_arena.constants import FINGER_CLOSED, PAD_TIP_PAST_TCP  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP, ObjectState  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0
SCORE_OVER = "end"
INSTRUCTION = "Bleed the air out of this brake caliper."

# --- Scene, world coordinates (m): x forward from the arm bases, y to the robot's left, z up from the table top.
# The pedal: a steel lever hinged about y at its back end, rubber pad at its front end (toward the robot).
PIVOT = (0.380, 0.160, 0.080)  # the pedal's hinge
LEVER = 0.110  # hinge to pad center
PAD = (0.045, 0.050, 0.012)  # rubber pad: x, y, thickness; its top is 1 cm above the hinge height at rest
PAD_DZ = 0.004  # pad center above the lever's axis
TRAVEL = 0.30  # rad: the pedal's stroke (3.3 cm at the pad), q from 0 (up, on its stop) to -TRAVEL (on the floor)
PEDAL_K, PEDAL_PRELOAD = 1.0, 0.20  # N*m/rad, rad: return spring (0.2 N*m preload: about 2 N at the pad at rest, 4.5 N pressed)
# The master cylinder (mechkit.Piston on the pedal): its piston shuts the compensating port 8% into the pedal's stroke
# and sweeps 4.2 mL from there to the floor stop (a 19 mm bore, which moves about 1.6 cm over the whole stroke). Pressed, the pedal stops where the line
# reaches the hand's push, 6 psi (about 6 N at the pad, at most 9 N with the spring, light for one arm): past
# the port by the fluid it takes to squeeze the air in the caliper. It sinks further only as fluid leaves the line.
PORT = -0.08 * TRAVEL
SWEEP = 4.2  # mL from the port to the floor stop
AREA = SWEEP / (TRAVEL + PORT)  # mL per rad of pedal
PUSH_PSI = 6.0
BLEED_FLOW = 0.6  # mL/s per psi through the bleeder a quarter turn open: 3.6 mL/s at 6 psi, the pedal down in about 1 s
THREADS_FLOW = 0.1  # mL/s per psi of air around its threads, once its seat has lifted (the threads do not seal)
CUP_FLOW = 1.0  # mL/s per psi from the reservoir past the piston's cup seal as the piston comes back
RETURN_PSI = 5.0  # psi of suction the piston's return spring pulls on the line (the piston back in about 1 s)

# The caliper on its bracket, the bleeder screw on its top, the wrench on the bleeder (hinged about z).
CAL = (0.300, -0.120)  # the caliper body's center (x, y)
CAL_SIZE = (0.090, 0.060, 0.060)  # cast-iron body, on a 3 cm steel bracket
CAL_Z = (0.030, 0.090)
BLEED = (0.290, -0.110)  # the bleeder screw's axis
# The bleeder's hex is 11 mm across its flats (12.7 mm across its corners), drawn as a 12 mm cylinder; the 11 mm
# box-end wrench's ring is drawn 13 mm across inside, so it fits over the corners.
HEX_R, RING_R = 0.006, 0.0065
WRENCH_Z = 0.100  # the wrench's mid-plane (on the bleeder's hex)
WRENCH_LEN, WRENCH_W, WRENCH_T = 0.125, 0.016, 0.008  # axis to handle end, width, thickness
SHUT, OPEN = 0.0, math.pi / 2  # wrench angle: 0 points -y (bleeder shut, as the scene starts); +90 deg (CCW from
# above, loosening a right-hand thread) points +x: the bleeder a quarter turn open
CRACK = math.radians(20)  # the bleeder starts to pass fluid 20 degrees past snug; full flow at a quarter turn
SHUT_TOL = math.radians(10)  # within 10 degrees of shut the detent pulls it snug (and the score counts it shut)
GRIP_R = 0.080  # demo: the hand works the handle 8 cm out from the bleeder

JAR = (0.215, -0.045)  # the catch jar (glass, fixed in a wire holder)
JAR_R, JAR_H, JAR_FLUID = 0.035, 0.090, 0.045
HOSE_END = (0.215, -0.040, 0.025)  # the clear hose's end, held 2 cm under the fluid (4.5 cm deep) in the jar

# --- Air and the pressure network (livekit). Volumes in mL, pressures in psi gauge.
AIR0 = 1.2  # mL of air in the caliper at the start (scaled: a few strokes clear it)
FIRM_AIR = 0.2  # mL: at or below this the pedal is firm (scoring)
HALF_AIR = 0.3 * AIR0  # mL: 70% of the air out (three correct cycles reach it, two do not) (scoring)
PART_AIR = 0.7 * AIR0  # mL: at least 30% of the air out = real progress (scoring)
AIR_IN_ML = 0.01  # mL of air drawn back in that the kit logs as the "air_in" hazard (one bubble, 3 mm across)
# The air left, drawn as a white disc on the caliper's top (the clear caliper of the real build): 3.2 cm across at
# 1.2 mL, its area following the air; clear of the bleeder and of the wrench at shut and open.
AIR_DISC = (0.325, -0.132, TOP + CAL_Z[1] + 0.001)

# --- Other scoring thresholds.
STILL = 0.2  # rad/s: a pedal or wrench still moving does not count
# A hand still on the pedal pad or the wrench at the end (gk.touching): a gripper's own shape (palm and both fingers,
# boxes fitted to the YAM's collision meshes, at their present opening) within gk.CONTACT (2 mm: the solver's resting
# gap plus the box fit) of the pad or of the wrench. A hand lifted off either, however close, is not on it.
PAD_SHAPE = gk.Block((-LEVER, 0.0, PAD_DZ), PAD)  # in the pedal's frame: the hinge, turning with the pedal
WRENCH_SHAPE = (gk.Seg((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.011),  # the box end around the hex
                gk.Block((0.0, -(0.010 + WRENCH_LEN) / 2, 0.0), (WRENCH_W, WRENCH_LEN - 0.010, WRENCH_T)))  # handle

STEEL = kit.Look((0.62, 0.63, 0.66), 0.35, metallic=0.8)
BARE = kit.Look((0.80, 0.81, 0.83), 0.25, metallic=0.9)  # the chrome wrench
IRON = kit.Look((0.10, 0.10, 0.11), 0.7)  # black-painted cast iron
RUBBER = kit.Look((0.12, 0.12, 0.12), 0.9)
ALU = kit.Look((0.70, 0.72, 0.74), 0.4, metallic=0.6)
AMBER = kit.Look((0.80, 0.55, 0.12), 0.2)  # brake fluid (new DOT 3 is pale amber; the old fluid in the jar darker)
OLD_FLUID = kit.Look((0.45, 0.28, 0.08), 0.2)
RESERVOIR = kit.Look((0.92, 0.92, 0.88), 0.3, opacity=0.45)
GLASS = kit.Look((0.86, 0.92, 0.95), 0.05, opacity=0.30)
HOSE_LOOK = kit.Look((0.85, 0.90, 0.85), 0.2, opacity=0.55)
YELLOW = kit.Look((0.95, 0.80, 0.10), 0.5)  # the bleeder screw's dust-cap color, to find it on camera


def _displace(dv):
    """The piston pushed `dv` mL into the line (negative: drew it back)."""
    lk.set_liquid("caliper", lk.liquid(None, "caliper") + dv)


def _room():
    """mL the line still takes before it is at the hand's push (6 psi). Below 1 atm (the piston just back) the
    reservoir fills it to 1 atm first, so that part is not room for the pedal."""
    return lk.room(None, "caliper", PUSH_PSI) - max(0.0, lk.room(None, "caliper", 0.0))


def _draw():
    """mL the returning piston can draw back this step: what lets the line down to its spring's 5 psi of suction, plus
    what flows in at that suction past the cup seal, around a loosened bleeder's threads and up the bleeder's hose."""
    inflow = CUP_FLOW + THREADS_FLOW * lk.opening(None, "threads") + BLEED_FLOW * lk.opening(None, "bleeder")
    return max(0.0, -lk.room(None, "caliper", -RETURN_PSI)) + inflow * RETURN_PSI * 0.05


_PUSHED = {"episode": None, "seen": 0.0}


def _pushed():
    """mL the hand's push moved into the line since the last call (the "hand" supply's delivered volume)."""
    if _PUSHED["episode"] != kit.episode():
        _PUSHED.update(episode=kit.episode(), seen=0.0)
    d = lk.delivered(None, "hand")
    out, _PUSHED["seen"] = d - _PUSHED["seen"], d
    return out


def _pressing(objects):
    """The hand holds the pedal on the piston, short of the floor stop: the line is at the hand's push."""
    return bool(kit.law_state("pedal_stand/pedal").get("pressing", False))


def _port_open(objects):
    """The compensating port: open once the piston is all the way back and the pedal within 8% of the top."""
    return pedal_q(objects) >= PORT and kit.law_state("pedal_stand/pedal").get("swept", 0.0) <= 1e-9


PEDAL_LAW = mk.Piston(port=PORT, area=AREA, sweep=SWEEP, displace=_displace, room=_room, draw=_draw, pushed=_pushed,
                      sign=-1, rest=PEDAL_PRELOAD, k=PEDAL_K, damp=0.02, lower=-TRAVEL, upper=0.0, lag=0.3)


def _pedal_stand():
    """The pedal stand (fixed): a steel post under the hinge, the master cylinder behind it with its translucent
    reservoir, and the pedal lever on the hinge with a return spring."""
    px, py, pz = PIVOT
    m, c = [], []
    m.append(kit.box((0, 0, pz / 2 - 0.006), (0.030, 0.040, pz - 0.012), STEEL))  # post
    c.append(kit.Collider((0, 0, pz / 2 - 0.006), size=(0.030, 0.040, pz - 0.012)))
    m.append(kit.box((0.020, 0, 0.004), (0.12, 0.10, 0.008), STEEL))  # foot plate
    c.append(kit.Collider((0.020, 0, 0.004), size=(0.12, 0.10, 0.008)))
    # master cylinder (aluminum, along x behind the hinge) and its reservoir with MIN/MAX lines
    m.append(kit.cylinder((0.060, 0, pz - 0.010), 0.016, 0.070, ALU, axis="x"))
    m.append(kit.box((0.060, 0, pz + 0.025), (0.060, 0.040, 0.045), RESERVOIR))
    m.append(kit.box((0.060, 0, pz + 0.017), (0.056, 0.036, 0.028), AMBER))
    m.append(kit.box((0.060, 0.0205, pz + 0.012), (0.050, 0.001, 0.002), IRON))  # MIN line
    m.append(kit.box((0.060, 0.0205, pz + 0.038), (0.050, 0.001, 0.002), IRON))  # MAX line
    m.append(kit.cylinder((0.060, 0, pz + 0.050), 0.014, 0.008, IRON))  # cap
    c.append(kit.Collider((0.060, 0, pz + 0.010), size=(0.070, 0.040, 0.085)))
    # the pedal lever (link frame at the hinge, lever along -x), steel arm and rubber pad
    arm = kit.box((-LEVER / 2, 0, 0), (LEVER, 0.018, 0.010), STEEL)
    pad = kit.box((-LEVER, 0, PAD_DZ), PAD, RUBBER)
    hub = kit.cylinder((0, 0, 0), 0.010, 0.030, STEEL, axis="y")
    pedal = kit.Link("pedal", "revolute", "Y", (0.0, 0.0, pz), lower=-TRAVEL, upper=0.0, start=0.0, mass=0.15,
                     meshes=(hub, arm, pad),
                     colliders=(kit.Collider((-LEVER, 0, PAD_DZ), size=PAD),
                                kit.Collider((-LEVER / 2 + 0.01, 0, 0), size=(LEVER - 0.04, 0.018, 0.010))),
                     law=PEDAL_LAW)
    return kit.Mechanism("pedal_stand", pos=(px, py, TOP), size=(0.14, 0.10, pz + 0.06), meshes=tuple(m),
                         colliders=tuple(c), links=(pedal,))


def _tube(a, b, r, look):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return kit.cylinder(tuple((a + b) / 2), r, float(np.linalg.norm(b - a)), look, axis=tuple(b - a))


# The clear bleeder hose, from the bleeder's nipple over to the jar and down under the fluid (world points).
HOSE_PATH = [(BLEED[0], BLEED[1], 0.118), (BLEED[0], BLEED[1], 0.138), (0.265, -0.085, 0.150), (0.230, -0.055, 0.140),
             (HOSE_END[0], HOSE_END[1], 0.100), HOSE_END]
# The steel brake line from the master cylinder, along the table, to the caliper's inlet at its back (world points).
LINE_PATH = [(PIVOT[0] + 0.095, PIVOT[1], PIVOT[2] - 0.010), (PIVOT[0] + 0.110, PIVOT[1], 0.006),
             (PIVOT[0] + 0.110, CAL[1], 0.006), (CAL[0] + CAL_SIZE[0] / 2 + 0.03, CAL[1], 0.006),
             (CAL[0] + CAL_SIZE[0] / 2, CAL[1], 0.050)]


def _caliper():
    """The caliper on its bracket with the bleeder, the hose and the brake line (fixed), and the wrench on the
    bleeder's hex (hinged about z, detents at shut and a quarter turn open)."""
    o = np.array([CAL[0], CAL[1], 0.0])

    def at(p):
        return tuple(np.asarray(p, float) - o)

    m, c = [], []
    m.append(kit.box((0, 0, 0.015), (0.050, 0.040, 0.030), STEEL))  # bracket
    c.append(kit.Collider((0, 0, 0.015), size=(0.050, 0.040, 0.030)))
    zc = sum(CAL_Z) / 2
    m.append(kit.box((0, 0, zc), CAL_SIZE, IRON))
    c.append(kit.Collider((0, 0, zc), size=CAL_SIZE))
    bx, by = at((BLEED[0], BLEED[1], 0))[:2]
    m.append(kit.cylinder((bx, by, 0.095), HEX_R, 0.010, BARE))  # the bleeder's 11 mm hex, under the wrench
    m.append(kit.cylinder((bx, by, 0.110), 0.0035, 0.016, BARE))  # its nipple
    m.append(kit.cylinder((bx, by, 0.104), 0.0045, 0.003, YELLOW))  # a ring of yellow paint to find it
    for a, b in zip(HOSE_PATH, HOSE_PATH[1:]):
        m.append(_tube(at(a), at(b), 0.004, HOSE_LOOK))
    for a, b in zip(LINE_PATH, LINE_PATH[1:]):
        m.append(_tube(at(a), at(b), 0.0025, STEEL))
    # the wrench: box end around the hex, handle along -y at q = 0 (link frame at the bleeder's axis, wrench height)
    ring = kit.lathe([(RING_R, -WRENCH_T / 2), (0.011, -WRENCH_T / 2), (0.011, WRENCH_T / 2), (RING_R, WRENCH_T / 2),
                      (RING_R, -WRENCH_T / 2)], BARE)
    handle = kit.box((0, -(0.010 + WRENCH_LEN) / 2, 0), (WRENCH_W, WRENCH_LEN - 0.010, WRENCH_T), BARE)
    wrench = kit.Link("wrench", "revolute", "Z", (bx, by, WRENCH_Z), lower=SHUT - 0.05, upper=OPEN + 0.05, start=SHUT,
                      mass=0.10, meshes=(ring, handle),
                      colliders=(kit.Collider((0, -(0.030 + WRENCH_LEN) / 2, 0),
                                              size=(WRENCH_W, WRENCH_LEN - 0.030, WRENCH_T)),),
                      law=mk.Detents((SHUT, OPEN), ("SHUT", "OPEN"), zone=SHUT_TOL, pull=0.6))
    return kit.Mechanism("caliper", pos=(CAL[0], CAL[1], TOP), size=(0.25, 0.30, 0.16), meshes=tuple(m),
                         colliders=tuple(c), links=(wrench,))


def _jar():
    """The catch jar in its wire holder (fixed), half full of old fluid."""
    m = [kit.lathe([(JAR_R, 0.0), (JAR_R, JAR_H)], GLASS),
         kit.cylinder((0, 0, JAR_FLUID / 2 + 0.002), JAR_R - 0.003, JAR_FLUID, OLD_FLUID),
         kit.box((0, 0, 0.002), (2 * JAR_R + 0.02, 2 * JAR_R + 0.02, 0.004), STEEL),
         # an inverted graduated tube over the hose's end: it collects the air pushed out (read in mL in a real trial)
         kit.cylinder((HOSE_END[0] - JAR[0], HOSE_END[1] - JAR[1], 0.055), 0.005, 0.065, GLASS)]
    c = [kit.Collider((0, 0, JAR_H / 2), radius=JAR_R, height=JAR_H)]
    return kit.Prop("jar", pos=(JAR[0], JAR[1], TOP), size=(2 * JAR_R + 0.02, 2 * JAR_R + 0.02, JAR_H),
                    meshes=tuple(m), colliders=tuple(c))


OBJECTS = [_pedal_stand(), _caliper(), _jar()]


def pedal_q(objects):
    return kit.joint(objects, "pedal_stand/pedal").pos


def wrench_q(objects):
    return kit.joint(objects, "caliper/wrench").pos


lk.setup(
    # the master-cylinder reservoir, open to the air through its cap's vent (topping up is not part of the task)
    lk.Pool("reservoir", head_cm=0.0),
    # the closed side: the master cylinder's chamber, the line and the caliper, 8 mL with 1.2 mL of air in the caliper.
    # The pedal's piston pushes fluid in and draws it back (mechkit.Piston, _displace). Air coming in is the latched
    # "air_in" hazard.
    lk.Tank("caliper", volume=8.0, liquid=8.0 - AIR0, protect="gas", protect_ml=AIR_IN_ML, protect_kind="air_in",
            gas_show=True, gas_full=AIR0, at=AIR_DISC),
    # the hand holding the pedal down on the piston: the line at 6 psi, the piston following what it pushes out
    lk.Supply("hand", psi=PUSH_PSI, medium="liquid", on=_pressing),
    lk.Pipe("piston", "hand", "caliper", flow=2.0, port_b="bottom"),
    lk.Pool("jar", head_cm=2.0, at=(HOSE_END[0], HOSE_END[1], TOP + HOSE_END[2])),  # hose end, 2 cm under the fluid
    # the compensating port: open while the pedal is within 8% of the top, so the line sits at the reservoir's 1 atm
    lk.Pipe("port", "reservoir", "caliper", open=_port_open, flow=1.0),
    # past the piston's cup seal: on the return the chamber pulls fluid from the reservoir around the cup's lip
    lk.Pipe("cup", "reservoir", "caliper", flow=CUP_FLOW, one_way=True),
    # the open bleeder: fluid and the air mixed through the caliper go out to the jar (bubbles at the hose's end), and
    # on a return stroke with it open, the jar's fluid is pulled back up the hose
    lk.Pipe("bleeder", "caliper", "jar", open=lk.joint_fraction("caliper/wrench", CRACK, OPEN), flow=BLEED_FLOW,
            port_a="mixed", show=True, at=(HOSE_END[0], HOSE_END[1], TOP + 0.040)),
    # around the loosened bleeder's threads: air can only come in, and only when the line is below 1 atm (the return
    # stroke). A bleeder seals at its seat, not its threads, so once the seat lifts they can leak however little it is
    # open.
    lk.Pipe("threads", "ambient", "caliper", open=lk.joint_past("caliper/wrench", CRACK),
            flow=THREADS_FLOW, one_way=True),
)
lk.BUBBLE_R = 0.009  # bubbles 1.8 cm across at the jar (about 4 px in the top camera)



def air(objects):
    """mL of air (at 1 atm) left in the caliper."""
    return lk.gas(objects, "caliper")


def pad_points(objects):
    """The pedal pad's top corners and center (world)."""
    q = pedal_q(objects)
    c, s = math.cos(q), math.sin(q)
    pts = []
    for dx in (-PAD[0] / 2, 0.0, PAD[0] / 2):
        for dy in (-PAD[1] / 2, 0.0, PAD[1] / 2):
            lx, lz = -LEVER + dx, PAD_DZ + PAD[2] / 2
            pts.append(np.array([PIVOT[0] + lx * c + lz * s, PIVOT[1] + dy, TOP + PIVOT[2] - lx * s + lz * c]))
    return pts


def wrench_points(objects):
    """Points along the wrench handle from the bleeder to its end (world)."""
    q = wrench_q(objects)
    d = np.array([math.sin(q), -math.cos(q), 0.0])
    piv = np.array([BLEED[0], BLEED[1], TOP + WRENCH_Z])
    return [piv + d * s for s in np.linspace(0.0, WRENCH_LEN, 9)]


def _link_state(pos, quat):
    p = np.asarray(pos, float)
    return ObjectState(pos=p, quat_xyzw=np.asarray(quat, float), lin_vel=np.zeros(3), ang_vel=np.zeros(3),
                       low=p, high=p)


def _held(objects):
    """A hand still on the pedal pad or the wrench: a gripper's own shape touching either (gk.touching, 2 mm)."""
    q, w = pedal_q(objects), wrench_q(objects)
    objs = dict(objects)
    objs["_pedal"] = _link_state((PIVOT[0], PIVOT[1], TOP + PIVOT[2]), (0.0, math.sin(q / 2), 0.0, math.cos(q / 2)))
    objs["_wrench"] = _link_state((BLEED[0], BLEED[1], TOP + WRENCH_Z), (0.0, 0.0, math.sin(w / 2), math.cos(w / 2)))
    return bool(gk.touching(objs, "_pedal", PAD_SHAPE) or gk.touching(objs, "_wrench", WRENCH_SHAPE))


def _job(objects):
    if lk.hazard(objects, "air_in") is not None:
        return 0.0  # the pedal let up with the bleeder open: the returning piston drew air in at its threads (latched)
    if _held(objects):
        return 0.0  # nothing counts while a hand still touches the pedal pad or the wrench (contact, not nearness)
    pedal_v, wrench_v = kit.joint(objects, "pedal_stand/pedal").vel, kit.joint(objects, "caliper/wrench").vel
    if abs(pedal_v) > STILL or abs(wrench_v) > STILL:
        return 0.0  # the pedal or the wrench still moving
    if wrench_q(objects) > SHUT + SHUT_TOL:
        return 0.0  # the bleeder left open: it can leak, and the next pedal stroke can draw air in
    if pedal_q(objects) < PORT or kit.law_state("pedal_stand/pedal").get("swept", 0.0) > 1e-9:
        return 0.0  # the pedal not back up: the master cylinder's piston not back past its port
    a = air(objects)
    if a <= FIRM_AIR:
        return 1.0  # bled: a firm pedal
    if a <= HALF_AIR:
        return 0.5  # most of the air out (70%), the pedal still a little soft
    if a <= PART_AIR:
        return 0.25  # real progress (at least 30% of the air out), still spongy
    return 0.0  # doing nothing, pumping the pedal alone or opening the bleeder alone leave the air in


def _score(objects):
    return _job(objects)


def success(objects):
    """Bled to a firm pedal, the bleeder shut, the pedal up, no air drawn back in, hands off."""
    return _score(objects) == 1.0


def score(objects):
    """0 if more than 0.01 mL of air (one bubble) was drawn into the caliper at any time (latched): the pedal let up
    while the bleeder is open past 20 degrees, so the master cylinder's piston, coming back on its spring, pulls on
    the line and air comes in around the loosened bleeder's threads. 0 if a gripper still touches the pedal pad or
    the wrench (its palm or fingers, at their present opening, within 2 mm: a hand still on it), if either is still
    moving faster than 0.2 rad/s, if the wrench is more than 10 degrees from shut (the bleeder left open), or if the
    pedal is not back up (its piston not back past the compensating port). Otherwise by the air left in the caliper
    (it started with 1.2 mL): 1.0 at 0.2 mL or less (a firm pedal); 0.5 at 0.36 mL or less (70% out); 0.25 at 0.84
    mL or less (at least 30% out, still spongy); 0 above that (doing nothing). Where the arms end up does not count.
    The final state must hold 1 s. Possible scores: 0, 0.25, 0.5, 1.0."""
    return _score(objects)


# --- Demo: the left arm works the pedal, the right arm the wrench, five cycles, then both home.

CYCLES = 5


def _tcp_z(pad_tip_z):
    return TOP + pad_tip_z + PAD_TIP_PAST_TCP


def _press_xy():
    """Where the closed fingers push the pad: its center at about half stroke (the pad swings 5 mm back as it goes
    down)."""
    q = -TRAVEL / 2
    return PIVOT[0] - LEVER * math.cos(q), PIVOT[1]


def on_handle(q, dz=0.0):
    """The right arm's tool point straddling the handle 8 cm out, fingers across it, at wrench angle q: the fingertips
    6 mm below the handle's underside, so the handle cannot ride out under them (it clears the caliper there)."""
    d = np.array([math.sin(q), -math.cos(q)])
    p = np.array(BLEED) + GRIP_R * d
    return (float(p[0]), float(p[1]), _tcp_z(WRENCH_Z - 0.010) + dz)


def handle_yaw(q):
    """Fingers closing across the handle (yaw 0 closes along y; the handle points along (sin q, -cos q))."""
    return q - math.pi / 2


def demo():
    """Left arm: closed fingers on the pedal pad, pushed toward a point 12 mm below the pedal's floor stop and held
    there: the pedal stops partway, on the fluid, and sinks to the floor as the right arm opens the bleeder a quarter
    turn; the right arm holds it 1 s and shuts it; then the left arm lifts and the pedal springs up.
    Right arm: fingers part open, straddling the wrench handle 8 cm out (the handle between the fingers), swept round
    the bleeder's axis to open and back to shut. Five cycles, then both arms go home."""
    straddle = 0.015  # each finger 1.5 cm from center: a 3.0 cm gap around the 1.6 cm handle
    px, py = _press_xy()
    above = _tcp_z(PIVOT[2] + 0.025)  # 1.5 cm above the pad's top at rest
    # 12 mm below where the pad's top sits on its floor stop, so the fingers keep pushing it down onto the fluid and,
    # once the bleeder is open, all the way to the floor
    down = _tcp_z(PIVOT[2] - LEVER * math.sin(TRAVEL) + PAD_DZ + PAD[2] / 2 - 0.012)
    t_press, t_hold0, t_swing, t_open, t_settle, t_up, t_wait = 0.6, 0.4, 1.8, 1.0, 0.4, 0.6, 0.6
    left = [Waypoint(1.6, (px, py, above), FINGER_CLOSED, "over the pedal")]
    right = [Waypoint(1.4, on_handle(SHUT, 0.04), straddle, "over the wrench", handle_yaw(SHUT)),
             Waypoint(0.8, on_handle(SHUT - 0.08), straddle, "fingers either side of the handle",
                      handle_yaw(SHUT - 0.08))]
    pad = sum(w.duration for w in right) - sum(w.duration for w in left)
    left.append(Waypoint(max(pad, 0.05), (px, py, above), FINGER_CLOSED, "wait over the pedal"))
    if pad < 0:
        right.append(Waypoint(-pad, on_handle(SHUT - 0.08), straddle, "wait", handle_yaw(SHUT - 0.08)))
    q_open, q_shut = OPEN + 0.10, SHUT - 0.08  # push 6 and 5 degrees past each detent, against the stops
    for k in range(CYCLES):
        # left: press and hold for the whole bleeder open-and-shut, then let the pedal up
        hold = t_hold0 + 2 * t_swing + t_open + t_settle
        left += [Waypoint(t_press, (px, py, down), FINGER_CLOSED, f"press the pedal {k + 1}"),
                 Waypoint(hold, (px, py, down), FINGER_CLOSED, "hold it down"),
                 Waypoint(t_up, (px, py, above), FINGER_CLOSED, "let the pedal up"),
                 Waypoint(t_wait, (px, py, above), FINGER_CLOSED, "pedal back up")]
        # right: wait for the press, open a quarter turn, hold, shut, wait for the pedal to come up
        right.append(Waypoint(t_press + t_hold0, on_handle(q_shut), straddle, "wait for the pedal",
                              handle_yaw(q_shut)))
        for i in range(1, 7):
            q = q_shut + (q_open - q_shut) * i / 6
            right.append(Waypoint(t_swing / 6, on_handle(q), straddle, f"open the bleeder {k + 1}", handle_yaw(q)))
        right.append(Waypoint(t_open, on_handle(q_open), straddle, "fluid and air out", handle_yaw(q_open)))
        for i in range(1, 7):
            q = q_open + (q_shut - q_open) * i / 6
            right.append(Waypoint(t_swing / 6, on_handle(q), straddle, "shut the bleeder", handle_yaw(q)))
        right.append(Waypoint(t_settle + t_up + t_wait, on_handle(q_shut), straddle, "keep it shut",
                              handle_yaw(q_shut)))
    left += [Waypoint(0.6, (px, py, above + 0.04), FINGER_CLOSED, "lift clear"), kit.Home(1.8),
             kit.Home(1.5, label="stay home")]
    right += [Waypoint(0.6, on_handle(SHUT, 0.05), straddle, "lift off the wrench", handle_yaw(SHUT)),
              kit.Home(1.8), kit.Home(1.5, label="stay home")]
    pad = sum(w.duration for w in right) - sum(w.duration for w in left)
    if pad > 0:
        left.append(kit.Home(pad, label="stay home"))
    elif pad < 0:
        right.append(kit.Home(-pad, label="stay home"))
    return {"left": left, "right": right}
