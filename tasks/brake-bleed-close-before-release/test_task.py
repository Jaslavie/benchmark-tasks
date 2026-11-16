"""Offline checks of the brake-bleeding scorer: step-by-step pedal, wrench and gripper histories run through the livekit
pressure network, near misses included, plus a replay of the scripted demo's timing. No simulator.

    PY=R/venv/bin/python R/yam-kits/chat/offline/runtest.sh <this file>      (prints "all pass")
"""

import importlib.util
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_spec = importlib.util.spec_from_file_location("brake_task", os.path.join(HERE, "task.py"))
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)
kit, lk = m.kit, m.lk
from yam_arena.task_file import ObjectState  # noqa: E402

PARK = {"left_gripper": (0.20, 0.27, m.TOP + 0.20), "right_gripper": (0.20, -0.27, m.TOP + 0.20)}
AWAY = {"left_gripper": (0.25, 0.20, m.TOP + 0.15)}  # done, but the left arm parked away from home
DOWN, UP_Q = -m.TRAVEL, 0.0  # the hand pushing the pedal toward its floor stop / letting it up
SHUT, OPEN = m.SHUT, m.OPEN
KEY = "pedal_stand/pedal"
PEDAL = m.OBJECTS[0].links[0]
RISE = 0.06  # rad per step: the pedal comes up behind the lifting hand (about 1.2 rad/s)
fails = 0
SIM = {"q": 0.0, "peak_psi": 0.0, "above_stop": 0}


DOWN_Q = (1.0, 0.0, 0.0, 0.0)  # a gripper pointing straight down, its fingers closing along x
TIP = 0.0128  # m: the fingertips' reach below the tool point, pointing down (guardkit's tip box)
OPENING = {"left_gripper": 0.0, "right_gripper": 0.32}  # the demo's closed left hand and straddling right hand


def st(pos, q=(0, 0, 0, 1.0)):
    p = np.array(pos, float)
    return ObjectState(pos=p, quat_xyzw=np.array(q, float), lin_vel=np.zeros(3), ang_vel=np.zeros(3),
                       low=p - 0.01, high=p + 0.01)


def check(cond, what):
    global fails
    print(("ok  " if cond else "FAIL") + " " + what)
    fails += not cond


def new_trial():
    kit._EPISODE["count"] += 1
    SIM.update(q=0.0, peak_psi=0.0, above_stop=0)


def mem():
    return kit._law_mem(KEY, PEDAL)


def step(pedal, wrench, grippers=None, pvel=None, wvel=0.0, opening=None):
    """One control step as the sim runs it. `pedal` is where the hand puts the pedal: pressing, it goes there or to the
    fluid stop the law set last step (the joint's lower limit), whichever is higher; let up, it rises at most RISE per
    step. Then the pedal's law (mechkit.Piston) runs on the new angle, then the livekit network, then the scorer."""
    lo = (mem().get("limits") or (-m.TRAVEL, 0.0))[0]
    q0 = SIM["q"]
    q = min(max(pedal, lo, -m.TRAVEL), q0 + RISE, 0.0)
    SIM["q"] = q
    v = (q - q0) / 0.05 if pvel is None else pvel
    kit._JOINTS[KEY] = kit.JointState(q, v, False)
    kit._JOINTS["caliper/wrench"] = kit.JointState(wrench, wvel, False)
    PEDAL.law.drive(KEY, q, v, {"pedal": q}, mem())
    SIM["above_stop"] += mem()["stop"] > q + 1e-9  # a limit above the joint would kick it back up
    g = dict(PARK, **(grippers or {}))
    objs = {"pedal_stand": st((*m.PIVOT[:2], m.TOP)), "caliper": st((*m.CAL, m.TOP)), "jar": st((*m.JAR, m.TOP)),
            **{k: st(p, DOWN_Q) for k, p in g.items()}}
    for k in PARK:
        kit._OPENING[k] = dict(OPENING, **(opening or {}))[k]
    lk.update(objs)
    SIM["peak_psi"] = max(SIM["peak_psi"], lk.pressure(objs, "caliper"))
    ok = m.success(objs)
    return m.score(objs), ok


def run(phases):
    """Phases of (steps, pedal, wrench, grippers); returns (score, success) at the last step."""
    out = None
    for n, p, w, grp in phases:
        for _ in range(n):
            out = step(p, w, grp)
    return out


def cycle(open_deg=90.0, release_open=False, hold_s=1.0, grp=None):
    """One pedal-method cycle: press onto the fluid, open the bleeder (ramped over 1.8 s) so the pedal sinks, hold,
    then either shut it and let the pedal up (right) or let the pedal up first and shut it after (the mistake). The
    piston gets 1.5 s to come back before the next press, as in the demo."""
    ph = [(10, DOWN, SHUT, grp)]
    q = math.radians(open_deg)
    ph += [(4, DOWN, q * i / 9, grp) for i in range(1, 10)]
    ph += [(int(hold_s * 20), DOWN, q, grp)]
    if release_open:
        ph += [(40, UP_Q, q, grp), (10, UP_Q, SHUT, grp)]
    else:
        ph += [(4, DOWN, q * i / 9, grp) for i in range(8, -1, -1)] + [(30, UP_Q, SHUT, grp)]
    return ph


START = [(10, UP_Q, SHUT, None)]


def trial(phases, end=(30, UP_Q, SHUT, None)):
    new_trial()
    return run(START + phases + ([end] if end else []))


# --- the bleeder and its wrench fit (review, Nov 12: an 8 mm wrench cannot go on a 12 mm hex)
corner = 0.011 / math.sqrt(3)  # an 11 mm hex's corner radius (12.7 mm across its corners)
check(m.HEX_R < corner < m.RING_R < m.HEX_R + 0.001,
      f"the 11 mm wrench's ring ({2 * m.RING_R * 1000:.1f} mm) clears the 11 mm hex's corners ({2 * corner * 1000:.1f} mm)"
      f" and the drawn hex ({2 * m.HEX_R * 1000:.0f} mm) is between its flats and corners")
_txt = open(os.path.join(HERE, "task.json")).read() + open(os.path.join(HERE, "README.md")).read()
check("11 mm hex" in _txt and "11 mm chrome box-end wrench" in _txt and "8 mm box-end" not in _txt
      and "8 mm chrome" not in _txt and "1.2 cm hex" not in _txt, "task text: an 11 mm wrench on an 11 mm hex everywhere")

# --- the master cylinder (mechkit.Piston) and the line, offline: the pedal moves only as the fluid lets it
def pct(q):
    return -q / m.TRAVEL * 100


new_trial()
run(START)
check(abs(m.air(None) - m.AIR0) < 1e-6, f"start: {m.air(None):.2f} mL of air in the caliper")
run([(60, DOWN, SHUT, None)])
spongy = SIM["q"]
check(0.15 < pct(spongy) < 0.30 * 100 and abs(lk.pressure(None, "caliper") - m.PUSH_PSI) < 0.5,
      f"bleeder shut, pedal pushed for 3 s: it stops on the fluid at {pct(spongy):.0f}% of its stroke (the air "
      f"squeezed), the line at {lk.pressure(None, 'caliper'):.1f} psi; it never reaches the floor")
run([(20, DOWN, SHUT, None)])
check(abs(SIM["q"] - spongy) < 1e-6 and abs(m.air(None) - m.AIR0) < 1e-6,
      "pushed on with the bleeder shut, the pedal stays put and no air leaves")
sink = []
for i in range(1, 10):
    run([(4, DOWN, OPEN * i / 9, None)])
    sink.append(round(pct(SIM["q"])))
run([(20, DOWN, OPEN, None)])
check(all(b >= a for a, b in zip(sink, sink[1:])) and sink[-1] > 40 and pct(SIM["q"]) > 99,
      f"bleeder opened with the pedal held: the pedal sinks as fluid leaves ({sink} %) and reaches the floor")
check(m.air(None) < 0.8 * m.AIR0, f"... and air goes out with it ({m.air(None):.2f} mL left)")
check(SIM["peak_psi"] < m.PUSH_PSI + 0.5, f"the line never goes past the hand's push ({SIM['peak_psi']:.1f} psi)")
check(SIM["above_stop"] == 0, "the fluid stop is never set above the pedal (no kick)")
run([(4, DOWN, OPEN * i / 9, None) for i in range(8, -1, -1)] + [(5, DOWN, SHUT, None)])
check(mem()["swept"] > 0.95 * m.SWEEP, "bleeder shut with the pedal still on the floor")
run([(4, UP_Q, SHUT, None)])
check(mem()["returning"] and lk.opening(None, "port") == 0, "let up: the piston comes back behind the pedal, its port "
      "still shut")
run([(40, UP_Q, SHUT, None)])
check(mem()["swept"] == 0 and lk.opening(None, "port") == 1 and abs(lk.pressure(None, "caliper")) < 0.1
      and lk.inflow(None, "caliper", "gas") == 0, "the piston back, the port open, the line at 1 atm, no air in")


def sink_rate(deg):
    """How far (% of stroke) the pedal sinks in 1 s with the bleeder held at `deg` from the start of the press."""
    new_trial()
    run(START + [(30, DOWN, SHUT, None), (1, DOWN, math.radians(deg), None)])
    q0 = SIM["q"]
    run([(20, DOWN, math.radians(deg), None)])
    return pct(SIM["q"]) - pct(q0)


rates = [(d, round(sink_rate(d), 1)) for d in (15, 25, 40, 90)]
check(rates[0][1] == 0 and all(b[1] > a[1] for a, b in zip(rates, rates[1:])),
      f"the pedal sinks as fast as the bleeder passes fluid: % of stroke in 1 s by bleeder angle {rates}")

new_trial()
run(START + [(40, UP_Q, OPEN, None), (40, 0.5 * m.PORT, OPEN, None), (10, UP_Q, SHUT, None)])
check(abs(m.air(None) - m.AIR0) < 0.01 and lk.hazard(None) is None,
      "bleeder open, pedal up or barely touched (above its port): nothing moves, no air out or in")

new_trial()
run(START)
airs, stops = [m.air(None)], []
for _ in range(5):
    run([(30, DOWN, SHUT, None)])
    stops.append(round(pct(SIM["q"])))
    run(cycle()[1:])
    airs.append(m.air(None))
check(all(b < a for a, b in zip(airs, airs[1:])), "every right-order cycle takes air out: "
      + ", ".join(f"{a:.2f}" for a in airs))
check(all(b < a for a, b in zip(stops, stops[1:])), f"the pedal firms up cycle by cycle: it stops on the fluid at "
      f"{stops} % of its stroke")
check(airs[4] <= m.FIRM_AIR < airs[3], f"four cycles reach a firm pedal ({airs[4]:.3f} <= {m.FIRM_AIR}), three don't")
check(airs[3] <= m.HALF_AIR < airs[2], f"three cycles get 70% out ({airs[3]:.3f} <= {m.HALF_AIR:.2f}), two don't")
check(lk.hazard(None) is None and lk.inflow(None, "caliper", "gas") < 1e-9, "no air came back in, no hazard")
check(lk.order_ok(None, ["open:bleeder", "close:bleeder", "open:bleeder", "close:bleeder"]),
      "the log shows the bleeder opening and shutting each cycle")

# the pedal let up with the bleeder still open: the returning piston pulls on the line and air comes in at the threads
drawn = []
for deg in (15, 19, 25, 30, 40, 90):
    new_trial()
    run(START + cycle(open_deg=deg, release_open=True))
    drawn.append((deg, round(lk.inflow(None, "caliper", "gas"), 3), lk.hazard(None, "air_in") is not None))
check(all(g == 0 and not hz for d, g, hz in drawn if d < 20) and all(hz for d, g, hz in drawn if d >= 25)
      and all(b[1] >= a[1] for a, b in zip(drawn, drawn[1:])),
      "pedal up with the bleeder open: (degrees, mL of air drawn in, air_in) " + str(drawn))
new_trial()
run(START + [(30, DOWN, SHUT, None), (5, UP_Q, SHUT, None), (10, UP_Q, OPEN, None), (20, UP_Q, SHUT, None)])
check(lk.hazard(None, "air_in") is None and lk.inflow(None, "caliper", "gas") == 0,
      "bleeder opened as the pedal comes up after a press with it shut: the piston only takes back what it pushed, no "
      "suction, no air")
new_trial()
run(START + [(30, DOWN, SHUT, None), (20, DOWN, OPEN, None), (2, DOWN, SHUT, None), (3, UP_Q, SHUT, None),
             (3, UP_Q, OPEN, None), (30, UP_Q, SHUT, None)])
check(lk.hazard(None, "air_in") is not None, "bleeder shut, pedal let up, then opened again while the piston is still "
      "coming back: air drawn in")
new_trial()
run(START + [(30, DOWN, SHUT, None), (20, DOWN, OPEN, None), (2, DOWN, SHUT, None), (50, UP_Q, SHUT, None),
             (20, UP_Q, OPEN, None), (20, UP_Q, SHUT, None)])
check(lk.hazard(None, "air_in") is None, "opened again only once the piston is back: no air drawn in")

# repeated wrong-order cycles never get the air down to firm
new_trial()
run(START)
for _ in range(10):
    run(cycle(release_open=True))
check(m.air(None) > 2 * m.FIRM_AIR, f"ten pedal-up-first cycles leave {m.air(None):.2f} mL: never firm")

PADHAND = {"left_gripper": tuple(m.pad_points({})[4])}  # closed fingers on the pad's top
cases = [
    ("do nothing", [], 0.0),
    ("pump the pedal ten times, bleeder shut", [(20, DOWN, SHUT, None), (30, UP_Q, SHUT, None)] * 10, 0.0),
    ("open the bleeder and shut it, no pedal", [(40, UP_Q, OPEN, None), (20, UP_Q, SHUT, None)], 0.0),
    ("one right cycle: real progress", cycle(), 0.25),
    ("two right cycles: progress, not yet 70% out", cycle() * 2, 0.25),
    ("three right cycles: 70% out, still a little soft", cycle() * 3, 0.5),
    ("four right cycles", cycle() * 4, 1.0),
    ("five right cycles", cycle() * 5, 1.0),
    ("bleeder opened before the press, pedal pushed to the floor, shut, let up: five cycles",
     [(10, UP_Q, OPEN, None), (40, DOWN, OPEN, None), (10, DOWN, SHUT, None), (30, UP_Q, SHUT, None)] * 5, 1.0),
    ("five right cycles, left arm parked away from home (home is not scored)",
     cycle(grp=None) * 5 + [(30, UP_Q, SHUT, AWAY)], 1.0),
    ("three right cycles, bleeder left open at the end (no 0.5 with it open)",
     cycle() * 3 + [(30, UP_Q, OPEN, None)], 0.0),
    ("three right cycles, then the mistake (latched, no 0.5)", cycle() * 3 + cycle(release_open=True), 0.0),
    ("three right cycles, pedal held down at the end (no 0.5)", cycle() * 3 + [(30, DOWN, SHUT, PADHAND)], 0.0),
    ("CORE MISTAKE: pedal up with the bleeder open, once", cycle(release_open=True), 0.0),
    ("CORE MISTAKE on the first cycle, then five right ones (latched)",
     cycle(release_open=True) + cycle() * 5, 0.0),
    ("five right cycles, then one mistake at the end (latched)", cycle() * 5 + cycle(release_open=True), 0.0),
    ("five right cycles, then the mistake with the bleeder cracked 30 degrees (latched)",
     cycle() * 5 + cycle(open_deg=30, release_open=True), 0.0),
    ("five right cycles, bleeder left open at the end", cycle() * 5 + [(30, UP_Q, OPEN, None)], 0.0),
    ("five right cycles, bleeder left 15 degrees open at the end (outside its detent)",
     cycle() * 5 + [(30, UP_Q, math.radians(15), None)], 0.0),
    ("five right cycles, bleeder 8 degrees from shut at the end (inside its detent)",
     cycle() * 5 + [(30, UP_Q, math.radians(8), None)], 1.0),
    ("five right cycles, pedal still held down at the end", cycle() * 5 + [(30, DOWN, SHUT, PADHAND)], 0.0),
    ("five right cycles, pedal propped 15% down at the end (piston not back)", cycle() * 5 + [(30, -0.15 * m.TRAVEL,
                                                                                            SHUT, None)], 0.0),
    ("five cycles with the bleeder only cracked 25 degrees, shut each time: too little out to count",
     cycle(open_deg=25) * 5, 0.0),
    ("five cycles with the bleeder opened only 45 degrees, shut each time: the pedal sinks slower, 70% out",
     cycle(open_deg=45) * 5, 0.5),
    ("bleeder shut before the pedal has sunk (opened 0.4 s): five short cycles, not firm",
     ([(10, DOWN, SHUT, None)] + [(1, DOWN, OPEN * i / 4, None) for i in range(1, 5)] + [(4, DOWN, OPEN, None)]
      + [(1, DOWN, OPEN * i / 4, None) for i in range(3, -1, -1)] + [(30, UP_Q, SHUT, None)]) * 5, 0.25),
    ("EXPLOIT: bleeder held cracked 22 degrees, pedal pumped 25 times, then shut",
     [(5, UP_Q, math.radians(22), None)] + [(10, DOWN, math.radians(22), None), (10, UP_Q, math.radians(22), None)] * 25
     + [(10, UP_Q, SHUT, None)], 0.0),
    ("EXPLOIT: bleeder held cracked 30 degrees, pedal pumped 25 times, then shut",
     [(5, UP_Q, math.radians(30), None)] + [(10, DOWN, math.radians(30), None), (10, UP_Q, math.radians(30), None)] * 25
     + [(10, UP_Q, SHUT, None)], 0.0),
    ("EXPLOIT: bleeder held open, pedal pumped 10 times, then shut",
     [(5, UP_Q, OPEN, None)] + [(20, DOWN, OPEN, None), (30, UP_Q, OPEN, None)] * 10 + [(10, UP_Q, SHUT, None)], 0.0),
]
ENDS = ("not home", "at the end")  # these cases' last phase is the end state itself
for name, phases, want in cases:
    got, ok = trial(phases, end=None if any(e in name for e in ENDS) else (30, UP_Q, SHUT, None))
    check(abs(got - want) < 1e-9 and ok == (want == 1.0), f"{got:.2f} (want {want:.2f}) {name} "
          f"({m.air(None):.2f} mL left, {lk.inflow(None, 'caliper', 'gas'):.3f} mL drawn in)")

# hands at the end, judged by contact (the gripper's own shape within 2 mm), not by a zone around the parts
from yam_arena.constants import FINGER_OPEN  # noqa: E402

pad_top = np.array(m.pad_points({})[4])  # the pad's top center, pedal up
mid = np.array(m.wrench_points({})[6])  # on the handle, 9.4 cm out, the wrench shut
GRIP_HANDLE = (m.WRENCH_W / 2) / abs(FINGER_OPEN)  # fingers closed onto the handle's 1.6 cm width


def bled_then(grippers, opening=None, steps=30):
    new_trial()
    run(START + cycle() * 5)
    out = None
    for _ in range(steps):
        out = step(UP_Q, SHUT, grippers, opening=opening)
    return out[0]


for name, g, op, want in [
    ("closed fingertips resting on the pedal pad (touching)", {"left_gripper": tuple(pad_top + [0, 0, TIP])}, None,
     0.0),
    ("closed fingertips 1 mm over the pad (inside the 2 mm contact tolerance)",
     {"left_gripper": tuple(pad_top + [0, 0, TIP + 0.001])}, None, 0.0),
    ("closed fingertips 5 mm over the pad, not touching (the old 3 cm zone scored 0)",
     {"left_gripper": tuple(pad_top + [0, 0, TIP + 0.005])}, None, 1.0),
    ("fingers closed on the wrench handle (holding it)", {"right_gripper": tuple(mid)},
     {"right_gripper": GRIP_HANDLE}, 0.0),
    ("closed fingertips resting on the wrench handle", {"right_gripper": tuple(mid + [0, 0, m.WRENCH_T / 2 + TIP])},
     {"right_gripper": 0.0}, 0.0),
    ("closed fingertips 5 mm over the wrench handle, not touching (the old zone scored 0)",
     {"right_gripper": tuple(mid + [0, 0, m.WRENCH_T / 2 + TIP + 0.005])}, {"right_gripper": 0.0}, 1.0),
    ("fingers open either side of the handle, 7 mm clear of it each side, not touching",
     {"right_gripper": tuple(mid)}, None, 1.0),
    ("the right hand let go 5 cm from the handle (not home: not scored)",
     {"right_gripper": tuple(np.array(m.wrench_points({})[-1]) + [0, -0.05, 0.03])}, None, 1.0),
]:
    got = bled_then(g, op)
    check(got == want, f"{got:.2f} (want {want:.2f}) bled, {name}")
check(m.gk.CONTACT == 0.002, "contact tolerance 2 mm, as the text says")

# moving parts at the end
new_trial()
run(START + cycle() * 5)
got, _ = step(UP_Q, SHUT, pvel=0.5)
check(got == 0.0, f"{got:.2f} (want 0.00) pedal still swinging at 0.5 rad/s")
got, _ = step(UP_Q, SHUT, wvel=0.5)
check(got == 0.0, f"{got:.2f} (want 0.00) wrench still swinging at 0.5 rad/s")

# cheap tricks: hovering over the pedal doing nothing, knocking the wrench, half presses
got, _ = trial([(60, UP_Q, SHUT, {"left_gripper": tuple(pad_top + [0, 0, 0.05])})])
check(got == 0.0, f"{got:.2f} (want 0.00) hovering over the pedal and nothing else: the air stays in")
got, _ = trial([(20, UP_Q, math.radians(30), None), (20, UP_Q, SHUT, None)] * 5)
check(got == 0.0 and lk.hazard(None) is None, f"{got:.2f} (want 0.00) wrench knocked 30 degrees and back, five times")
got, _ = trial([(20, -0.05, SHUT, None), (5, -0.05, OPEN, None), (20, -0.05, OPEN, None),
                (5, -0.05, SHUT, None), (30, UP_Q, SHUT, None)] * 5)
check(got == 0.0, f"{got:.2f} (want 0.00) light presses that stop short of the fluid, with the bleeder worked")

# episode reset: a hazard trial does not carry over; home re-recorded
trial(cycle(release_open=True))
check(lk.hazard(None, "air_in") is not None, "the mistake trial logged air_in")
got, ok = trial(cycle() * 5)
check(got == 1.0 and ok and lk.gas(None, "caliper") < m.FIRM_AIR,
      f"{got:.2f} (want 1.00) next trial starts clean, with the air back at 1.2 mL")
new_trial()
step(UP_Q, SHUT, AWAY)  # this trial starts with the left arm at AWAY: that is its home now
run([(n, p, w, AWAY) for n, p, w, _ in cycle() * 5])
got, _ = run([(30, UP_Q, SHUT, AWAY)])
check(got == 1.0, f"{got:.2f} (want 1.00) a trial run with the left arm elsewhere throughout")

# --- QC review (Nov 10, redone for the piston): partial returns, same-step releases, one-step pops
ph = []
for _ in range(5):
    ph += [(10, DOWN, SHUT, None), (40, DOWN, OPEN, None), (20, 0.71 * DOWN, OPEN, None), (5, 0.71 * DOWN, SHUT, None),
           (30, UP_Q, SHUT, None)]
got, _ = trial(ph)
check(got == 0.0 and lk.hazard(None, "air_in") is not None,
      f"{got:.2f} (want 0) pedal let 29% back up from the floor with the bleeder open: the piston pulls, air_in")
ph = []
for _ in range(5):
    ph += [(10, DOWN, SHUT, None), (40, DOWN, OPEN, None), (3, UP_Q, OPEN, None), (30, UP_Q, SHUT, None)]
got, _ = trial(ph)
check(got == 0.0 and lk.hazard(None, "air_in") is not None,
      f"{got:.2f} (want 0) wrench shut only 0.15 s after the pedal started up: air_in")
ph = []
for _ in range(5):
    ph += [(10, DOWN, SHUT, None), (40, DOWN, OPEN, None), (1, UP_Q, SHUT, None), (30, UP_Q, SHUT, None)]
got, _ = trial(ph)
check(lk.inflow(None, "caliper", "gas") < m.AIR_IN_ML,
      f"{got:.2f} wrench snapped shut on the same 50 ms step the pedal started up: the seat is shut before the piston "
      f"pulls ({lk.inflow(None, 'caliper', 'gas'):.3f} mL in)")
ph = cycle() * 4 + [(10, DOWN, SHUT, None), (20, DOWN, OPEN, None), (1, DOWN + 0.02, OPEN, None),
                    (20, DOWN, OPEN, None), (5, DOWN, SHUT, None), (30, UP_Q, SHUT, None)]
got, ok = trial(ph)
check(got == 1.0 and ok and lk.hazard(None) is None,
      f"{got:.2f} (want 1.00) a one-step 2 mm pop of the pad with the bleeder open is not a return")

# --- the scripted demo, replayed: pedal angle from the left tool point's height, wrench angle from the right tool
# point's bearing about the bleeder (lagging 0.06 rad behind, as the straddling fingers' slop lets it)
d = m.demo()
check(isinstance(d["left"][-1], kit.Home) and isinstance(d["right"][-1], kit.Home), "demo: both arms end at home")
TL, TR = sum(w.duration for w in d["left"]), sum(w.duration for w in d["right"])
check(abs(TL - TR) < 1e-6 and TL * 20 < 1200, f"demo: both arms end together, {TL * 20:.0f} steps (< 1,200)")


def sample(wps, t, home):
    """Linear interpolation of tool point and finger between waypoints (Home = the parked start)."""
    prev = np.array(home, float)
    t0 = 0.0
    for w in wps:
        tgt = np.array(home, float) if isinstance(w, kit.Home) else np.array(w.pos, float)
        if t <= t0 + w.duration:
            return prev + (tgt - prev) * (t - t0) / w.duration
        prev, t0 = tgt, t0 + w.duration
    return prev


pad_rest_top = m.PIVOT[2] + m.PAD_DZ + m.PAD[2] / 2
new_trial()
wq, pq, got, ok = SHUT, 0.0, None, False
sw_max = 0.0
for k in range(int(TL * 20) + 30):
    t = k * 0.05
    L = sample(d["left"], t, PARK["left_gripper"])
    Rr = sample(d["right"], t, PARK["right_gripper"])
    tip_z = L[2] - m.TOP - m.PAD_TIP_PAST_TCP
    near_pad = abs(L[0] - m._press_xy()[0]) < 0.03 and abs(L[1] - m.PIVOT[1]) < 0.03
    pq = max(-m.TRAVEL, min(0.0, -math.asin(min(1.0, max(0.0, pad_rest_top - tip_z) / m.LEVER)))) if near_pad else 0.0
    rb = Rr[:2] - np.array(m.BLEED)
    if np.linalg.norm(rb) < 0.10 and Rr[2] - m.TOP < m.WRENCH_Z + 0.03:
        bearing = math.atan2(rb[0], -rb[1])  # 0 along -y, +pi/2 along +x
        wq = min(max(wq, bearing - 0.06), bearing + 0.06)  # pushed by whichever finger leads
        wq = min(max(wq, SHUT - 0.05), OPEN + 0.05)
    sw_max = max(sw_max, wq)
    got, ok = step(pq, wq, {"left_gripper": tuple(L), "right_gripper": tuple(Rr)})
check(sw_max > OPEN - m.SHUT_TOL, f"demo replay: the wrench reaches the open detent ({math.degrees(sw_max):.0f} deg)")
check(lk.hazard(None) is None, "demo replay: no air drawn back in (the bleeder always shut before the pedal came up)")
check(got == 1.0 and ok, f"{got:.2f} (want 1.00) demo replay ends bled ({m.air(None):.3f} mL left)")
check(lk.import_order_ok(), "livekit was imported last")

print("all pass" if not fails else f"{fails} FAILED")
