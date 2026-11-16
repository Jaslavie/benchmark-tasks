"""livekit: things that are energized or pressurized, for a yam_arena task file (works with yamkit, and reads
cablekit, mechkit and twistkit parts when a task uses them).

A task declares a small network: electrical nodes joined by sources, wires, loads and conductive tools, and/or
pressure compartments joined by pipes, couplings and valves. The physical parts the robot moves decide which
connections are made (a cablekit clamp on a post, a mechkit breaker toggle past its throw, a twistkit coupling backed
off, a wrench whose bare metal touches two pads). Every control step (20 Hz, inside the hooked object_states, after
yamkit/cablekit/twistkit have updated) the kit solves the network, then appends to an ordered event log: connections
made and broken, sparks, and hazards, which are latched for the trial (an arc when a tool bridges a live terminal to
ground, a spray or hose whip when a coupling is opened under pressure, a backfeed when a source drives a line that
must be dead, liquid sucked back into a flask, air drawn into a brake line, a reservoir run dry).

    import yamkit as kit
    import cablekit as ck          # if the task has clamps or hoses
    import livekit as lk           # import it LAST, so its step runs after the other kits' (see "Hook" below)

    lk.setup(
        lk.Node("bat_pos", body="battery", at=(0.05, 0.0, 0.145), lamp=False),
        lk.Node("bat_neg", body="battery", at=(-0.05, 0.0, 0.145)),
        lk.Node("frame", ground=True),
        lk.Source("battery", plus="bat_pos", minus="bat_neg", volts=12.6, ohms=0.008),
        lk.Wire("neg_cable", "bat_neg", "frame", closed=lk.attached("neg_lead/lug", "neg_post")),
        lk.Wire("pos_cable", "bat_pos", "harness", closed=lk.attached("pos_lead/lug", "pos_post")),
        lk.Load("parasitic", "harness", "frame", ohms=400.0),
        lk.Tool("wrench", body="machine/wrench", tip=(0.0, 0, 0), tail=(0.13, 0, 0), radius=0.006,
                pads=(lk.Pad("bat_pos", body="battery", point=(0.05, 0, 0.15), radius=0.008),
                      lk.Pad("frame", body="frame_rail", point=(0, 0, 0), box=(0.10, 0.01, 0.01)))),
    )
    ...
    lk.hazard(objects, "short")                       # in score(): the first arc this trial, or None
    lk.order_ok(objects, [("break", "neg_cable"), ("break", "pos_cable")])

Electrical model (DC nodal analysis, solved with NumPy each step). Every `Wire` that is closed and every tool contact
is a small resistance (default 2 milliohm); a `Source` is an ideal EMF with an internal resistance (a 12 V lead-acid
starting battery has about 5-10 milliohm, so a dead short draws 1,000+ A: the arc that welds a wrench to a terminal);
a `Load` is a resistance (given as ohms, or watts at a voltage) with an optional starting surge (a motor draws 3-6x its
running current for about a second). Nodes flagged `ground` are tied to 0 V (the vehicle chassis, the earth bar);
a tiny leak (1 microsiemens) to 0 V keeps floating parts solvable. A source that is off is an open circuit. Results:
node voltages, the current in every element, which nodes are joined by low-resistance paths (nets).
Events: "make"/"break" when a wire or tool contact closes or opens; "spark" when that happens with more than
`spark_amps` flowing (after a make, before a break; a hazard if the element has `spark_hazard`, e.g. a clamp near a
gassing battery); "short" (hazard) when a tool (or a wire given its own `fault_amps`) carries more than that (tools default to 100 A); "backfeed"
(hazard) when a node flagged `dead` (a utility line someone is working on) rises above `live_volts`, or a source
flagged `no_tie` finds another source's terminal on its own net, or a source with `reverse_amps` is driven backward;
"overload" (hazard) when a source with `max_amps` delivers more than that for `overload_s`; "live"/"dead" when a
node flagged `watch` changes.

Pressure model (lumped compartments, explicit integration with 20 substeps per control step and a per-pipe flow
limiter so a stiff, nearly liquid-full compartment cannot oscillate). A `Tank` is a closed compartment of `volume` mL
holding gas (tracked as mL at 1 atm and 20 C, i.e. an amount) and liquid (mL, incompressible); its absolute pressure
is the ideal-gas pressure of the gas in the space the liquid leaves, plus a fixed cushion worth 0.5% of the volume
(the compliance of hoses and seals: a liquid-full line sits at 1 atm and pressurizes steeply as more is pushed in,
by up to 0.5% of its volume, which takes about 1,000 atmospheres), at the tank's temperature (optionally heated by a burner: first-order toward `hot_c`
or back to `start_c`). "ambient" is the open air at 1 atm. A `Pool` is liquid open to the air with a `head` (a jar,
a water trough, a faucet's water). A `Supply` holds a set gauge pressure while `on` (a compressor, a water aspirator
pulling a vacuum, a brake pedal held down), optionally limited to `capacity` mL per press (the pedal reaches the
floor) and refilled from a reservoir tank; when it stops it can pull back (`return_psi`, the master cylinder's piston
returning) and then either closes or opens to the reservoir at 1 atm (`idle="pool"`, the compensating port). A `Pipe`
joins two of these with a linear conductance (mL/s per psi of difference at full open) times its open fraction
(0..1, from a valve angle, a coupling's backed-off angle, a hose fitting on or off). What a pipe carries out of a tank
depends on where its port is: "top" (gas first), "bottom" (liquid first) or "mixed" (in proportion to the gas and
liquid volumes, for bubbles spread through a line). Events: "open"/"close" for pipes; "release" (hazard) when a pipe
flagged with `release_psi` goes fully open with more than that pressure across it (a coupling or cap undone under
pressure sprays or whips); "burst" (hazard) above a tank's `rating`; a tank with `protect="liquid"` or `"gas"` logs
`protect_kind` (default "backflow", hazard) once more than `protect_ml` of that medium has flowed into it (water
sucked back into a filtrate, air drawn back into a brake caliper); `dry_ml` logs `dry_kind` (hazard) when its liquid
falls below that (a master-cylinder reservoir run dry lets air into the whole system).

Simplifications (state them in a task's fidelity): DC, steady state each 50 ms step (no inductance, no AC phase, so
paralleling two equal AC sources is flagged by topology with `no_tie`, not by current); contact is geometric
(bare-metal segment of a tool within its radius of a pad: no contact force or resistance that depends on pressure);
arcs do not weld or burn anything and nothing is drawn but the marker; gas is ideal and isothermal within a step,
liquid is incompressible, flow is linear in pressure difference (laminar-like), there is no inertia (no water
hammer), no gravity head inside tanks, and a tank's gas and liquid are well mixed except as its ports say. Numbers
for a task come from real equipment and should be cited there.

Scorer API (all follow the trial; `objects` is what score() gets and is unused by most, kept for uniformity):
- electrical: `volts(objects, node, ref=None)` (to ground, or to `ref`), `live(objects, node, ref=None, volts=None)`,
  `amps(objects, element)` (a wire's, a load's, a source's output; a tool's largest contact current),
  `closed(objects, element)`, `contacts(objects, tool)` (the nodes a tool touches now), `connected(objects, a, b)`
  (same low-resistance net now).
- pressure: `pressure(objects, name, unit="psi")` (gauge; "kPa", "inH2O", "bar" also; tanks, supplies, pools),
  `gas(objects, tank)` (mL at 1 atm), `liquid(objects, tank)`, `inflow(objects, tank, medium)` (mL that came in this
  trial), `delivered(objects, supply)`, `flow(objects, pipe)` (mL/s last step, + from a to b), `opening(objects, pipe)`,
  `temp(objects, tank)`.
- events: `events(objects, kind=None, where=None)` (ordered list of `Event(step, kind, where, value, hazard, point)`),
  `hazard(objects, kind=None, where=None)` (the first hazard event, latched: it stays after the cause is gone),
  `hazards(...)`, `first(objects, kind, where=None)` (its step or None), `order_ok(objects, sequence)` (the
  (kind, where) pairs happened in that order, each after the one before: a subsequence of the log), `step()`.
- `log(kind, where, value=None, hazard=False, point=None)` adds a task event; `watch(fn)` registers fn(objects),
  called every step after the solve (a task's own rules: a pilot flame lighting, a needle, a counter);
  `set_gas(tank, ml)` / `set_liquid(tank, ml)` for such rules.

Condition helpers (give them to `closed=`, `on=`, `open=`; each is fn(objects) -> bool or 0..1): `attached(key,
target=None)` (cablekit), `joint_past(key, at, sign=1)`, `joint_fraction(key, shut, full)` (a valve handle),
`backed_off(key, crack_deg, open_deg)` (a twistkit coupling: 0 seated, 1 off), `inside(body, point, region_body,
center, half)` (a hose end inside a jar), `near(body, point, other, other_point, r)`, `all_of`, `any_of`, `not_`,
`times` (product of fractions: a bleeder's opening while its hose is in the jar).
Gauges: `gauge(name, full, sweep, zero=0.0, unit="psi")` returns a `fn(J, mem)` for `mechkit.Follow(fn=...)`, so a
needle link reads a pressure (one control step behind, since mechanism drives are set before this kit's step).

Visuals (Isaac Lab VisualizationMarkers, optional, never scored): a bright flash at each spark, arc or release (for
`flash_s`), a dark scorch mark that stays at each latched short, a small lamp sphere per node with `lamp=True` (red
when live, dark when dead), white bubbles at a pipe flagged `show=True` while gas flows through it, and a water-colored
bead at a tank flagged `show=True` while liquid flows into it, and a flat white disc at a tank flagged `gas_show=True`
whose area follows the gas left in it (radius `AIR_R` at `gas_full`, default the tank's starting gas; hidden under 1%).
Sizes are set for the 320 x 176 top camera (a 1.6 cm flash, a 1.2 cm lamp, a 3.2 cm air disc when full).

Demo helpers: `press(xyz, depth, hold)` (a closed-finger push held down: a pedal, a button, a knob pushed in) and
`push_across(start, end)` (a closed-finger stroke: a toggle thrown, a wrench or valve handle swung) return
yam_arena Waypoints; end a demo with `kit.Home`.

Hook. Like cablekit and twistkit, the module chains yamkit's scene patch: the wrapped object_states runs the inner
states (yamkit's mechanisms and grippers, cablekit's attachments, twistkit's thread accounting) and then this kit's
`update`, once per control step. Import livekit after the other kits so it wraps them (imported earlier, it would see
their state one step late). Plain Python and NumPy at import; Isaac code only runs inside the simulator and is guarded.

Episodes: everything resets when yamkit's `episode()` changes (the simulator reset the scene); `reset()` by hand.
Tests: `setup(...)`, then call `update(objects)` once per step with hand-made ObjectStates (conditions are plain
callables, so tests drive them through a dict); bump `kit._EPISODE["count"]` for a new trial.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field

import numpy as np

import yamkit as kit

PATM = 101.325  # kPa, one standard atmosphere
KPA_PER_PSI = 6.894757
KPA_PER_CM_WATER = 0.0980665
T0 = 293.15  # K: gas amounts are mL at 1 atm and 20 C
AMBIENT = "ambient"
CUSHION = 0.005  # a tank's compliance: as if 0.5% of its volume were a trapped air cushion that never leaves
UNITS = {"psi": KPA_PER_PSI, "kPa": 1.0, "bar": 100.0, "inH2O": 0.249089, "mmHg": 0.133322}


# -- electrical parts --------------------------------------------------------------------------------


@dataclass(frozen=True)
class Node:
    """An electrical node: a terminal, a bus, a chassis. `ground=True` ties it to 0 V. `dead` (True or fn(objects)):
    while it holds, the node must stay dead (a utility line in an outage, with crews on it): rising above `live_volts`
    is a "backfeed" hazard. `watch=True` logs "live" and
    "dead" when it changes. `lamp=True` draws an indicator. Its location (for markers and event points) is `at` in the
    frame of `body` (a task object, "mech/link" or a gripper), or a world point when `body` is None."""

    name: str
    ground: bool = False
    dead: object = False
    watch: bool = False
    lamp: bool = False
    live_volts: float | None = None
    body: str | None = None
    at: tuple | None = None


@dataclass(frozen=True)
class Source:
    """An EMF of `volts` from `minus` to `plus` with internal resistance `ohms`, on while `on` (True, or fn(objects)).
    `max_amps`/`overload_s`: an "overload" hazard when it delivers more for that long (a generator's rating).
    `reverse_amps`: a "backfeed" hazard when driven backward by more. `no_tie`: a "backfeed" hazard when another
    source's terminal is on its plus net through low-resistance paths (two supplies tied)."""

    name: str
    plus: str
    minus: str
    volts: float
    ohms: float = 0.01
    on: object = True
    max_amps: float | None = None
    overload_s: float = 0.5
    reverse_amps: float | None = None
    no_tie: bool = False


@dataclass(frozen=True)
class Wire:
    """A conductor between nodes `a` and `b`, closed while `closed` (True or fn(objects)): a cable, a clamp on a post,
    a breaker, a switch, a jumper. `fault_amps` (none by default: cables in a fault's path carry its current too; set
    it on a part that can itself be the short, like a loose jumper) makes a "short" hazard above it; `spark_amps` (default:
    the setup's) makes a "spark" on make or break; `spark_hazard` makes such a spark a hazard."""

    name: str
    a: str
    b: str
    closed: object = True
    ohms: float = 0.002
    fault_amps: float | None = None
    spark_amps: float | None = None
    spark_hazard: bool = False
    body: str | None = None
    at: tuple | None = None


@dataclass(frozen=True)
class Load:
    """A resistance between `a` and `b`: `ohms`, or `watts` at `volts`. On while `on`. When it is first energized
    (its voltage rises past half of its rated volts) it draws `surge` times its current for `surge_s` (motor start)."""

    name: str
    a: str
    b: str
    ohms: float | None = None
    watts: float | None = None
    volts: float | None = None
    on: object = True
    surge: float = 1.0
    surge_s: float = 0.0

    @property
    def r(self) -> float:
        if self.ohms is not None:
            return float(self.ohms)
        return float(self.volts) ** 2 / float(self.watts)


@dataclass(frozen=True)
class Pad:
    """A contact region on a body tied to `node`: a sphere of `radius` around `point`, or, with `box` (half sizes), a
    box around `point` aligned with the body's frame. `body` None: world coordinates."""

    node: str
    body: str | None = None
    point: tuple = (0.0, 0.0, 0.0)
    radius: float = 0.01
    box: tuple | None = None


@dataclass(frozen=True)
class Tool:
    """A conductive tool (a wrench, a screwdriver's shank, a jumper bar, a metal gripper): its bare metal is the
    segment `tip`-`tail` in the frame of `body`, with `radius`. It touches a pad when that segment comes within the
    radius (off again past radius + 2 mm, so it does not chatter). While it touches two or more pads it joins their
    nodes through `ohms`."""

    name: str
    body: str
    tip: tuple = (0.0, 0.0, 0.0)
    tail: tuple = (0.1, 0.0, 0.0)
    radius: float = 0.004
    pads: tuple = ()
    ohms: float = 0.002
    fault_amps: float | None = None
    spark_amps: float | None = None
    spark_hazard: bool = False


# -- pressure parts ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Tank:
    """A closed compartment: `volume` mL, starting at `psi` gauge with `liquid` mL in it at `start_c`. `rating` (psi
    gauge): a "burst" hazard above it. `protect` ("liquid" or "gas"): `protect_kind` hazard once more than
    `protect_ml` of that has flowed in. `dry_ml`: `dry_kind` hazard when the liquid falls below it. `heat` (fn or
    bool): heated toward `hot_c` with time constant `heat_s`, else cooling to `start_c` with `cool_s`. `make_gas`
    (mL/s at 1 atm while heated and above 60 C): gas generated (a reaction). `show=True` draws a bead while liquid
    flows in. `gas_show=True` draws a white disc at `at` whose radius is AIR_R * sqrt(gas / gas_full) (`gas_full`
    in mL at 1 atm, default the starting gas): the air left, readable on video."""

    name: str
    volume: float
    psi: float = 0.0
    liquid: float = 0.0
    rating: float | None = None
    protect: str | None = None
    protect_ml: float = 1.0
    protect_kind: str = "backflow"
    dry_ml: float | None = None
    dry_kind: str = "ran_dry"
    heat: object = None
    start_c: float = 20.0
    hot_c: float = 20.0
    heat_s: float = 5.0
    cool_s: float = 10.0
    make_gas: float = 0.0
    show: bool = False
    body: str | None = None
    at: tuple | None = None
    gas_show: bool = False
    gas_full: float | None = None


@dataclass(frozen=True)
class Supply:
    """Holds `psi` gauge (a number or fn(objects)) while `on`: a compressor, a regulator, a vacuum source (negative),
    a held brake pedal. `medium` "gas" or "liquid". `max_flow` caps its flow (mL/s, actual volume). `capacity` (mL)
    is what one activation can push before it stops (the pedal on the floor); `refill` names a tank whose liquid it
    draws (and returns to). After `on` goes off it holds `return_psi` for `return_s` (the piston returning), then is
    `idle`: "closed" (no flow) or "pool" (open to its reservoir at 1 atm, the compensating port)."""

    name: str
    psi: object = 0.0
    medium: str = "gas"
    on: object = True
    max_flow: float | None = None
    capacity: float | None = None
    refill: str | None = None
    return_psi: float | None = None
    return_s: float = 0.0
    idle: str = "closed"
    body: str | None = None
    at: tuple | None = None


@dataclass(frozen=True)
class Pool:
    """Liquid open to the air with `head_cm` of depth over the pipe's end (a jar, a trough, a faucet's water)."""

    name: str
    head_cm: float = 0.0
    body: str | None = None
    at: tuple | None = None


@dataclass(frozen=True)
class Pipe:
    """A path between `a` and `b` (tanks, supplies, pools or "ambient") passing `flow` mL/s per psi of difference at
    full open, times `open` (True/False, a fraction 0..1, or fn(objects) giving either). `port_a`/`port_b`: where it
    leaves a tank ("top", "bottom", "mixed"). `release_psi`: a `release_kind` hazard ("release") if it goes fully open
    with more than that across it. `one_way=True`: it passes flow from a to b only (a check valve; air drawn in
    around a loosened bleeder's threads, which weep only a little outward). `show=True` draws bubbles while gas
    passes."""

    name: str
    a: str
    b: str
    open: object = True
    flow: float = 1.0
    port_a: str = "bottom"
    port_b: str = "bottom"
    release_psi: float | None = None
    release_kind: str = "release"
    one_way: bool = False
    show: bool = False
    body: str | None = None
    at: tuple | None = None


@dataclass(frozen=True)
class Event:
    step: int
    kind: str
    where: str
    value: float | None = None
    hazard: bool = False
    point: tuple | None = None


# -- configuration and per-trial state ---------------------------------------------------------------

_CFG: dict = {"parts": {}, "order": [], "dt": 0.05, "fault_amps": 100.0, "spark_amps": 0.5, "live_volts": 1.0,
              "substeps": 20, "flash_s": 0.6, "watchers": []}


def setup(*parts, dt: float = 0.05, fault_amps: float = 100.0, spark_amps: float = 0.5, live_volts: float = 1.0,
          substeps: int = 20, flash_s: float = 0.6) -> None:
    """Declare the network (any mix of the part classes above). Call once at import of task.py."""
    names = {}
    for p in parts:
        if p.name in names or p.name == AMBIENT:
            raise ValueError(f"livekit: duplicate or reserved name {p.name!r}")
        names[p.name] = p
    nodes = {p.name for p in parts if isinstance(p, Node)}
    for p in parts:  # undeclared nodes are allowed (plain junctions); declared pressure ends must exist
        if isinstance(p, Pipe):
            for e in (p.a, p.b):
                if e != AMBIENT and not isinstance(names.get(e), (Tank, Supply, Pool)):
                    raise ValueError(f"livekit: pipe {p.name}: no tank, supply or pool {e!r}")
        if isinstance(p, Tool):
            for pad in p.pads:
                nodes.add(pad.node)
    for p in parts:
        if isinstance(p, Supply) and p.refill is not None and not isinstance(names.get(p.refill), Tank):
            raise ValueError(f"livekit: supply {p.name}: refill {p.refill!r} is not a tank")
    _CFG.update(parts=names, order=list(parts), dt=dt, fault_amps=fault_amps, spark_amps=spark_amps,
                live_volts=live_volts, substeps=max(1, int(substeps)), flash_s=flash_s, watchers=[])
    if not import_order_ok():
        print("livekit: WARNING: another kit was imported after livekit, so livekit reads its state one control step "
              "late; import livekit last", file=sys.stderr, flush=True)
    reset()


def import_order_ok() -> bool:
    """True if no kit chained yamkit's scene patch after this one (livekit was imported last, so its step runs after
    cablekit's attachments, twistkit's threads and yamkit's joints of the same control step)."""
    return kit._patch is _patch


def watch(fn) -> None:
    """Call fn(objects) every step after the network is solved (a task's own rules; may call log/set_gas)."""
    _CFG["watchers"].append(fn)


def _of(cls) -> list:
    return [p for p in _CFG["order"] if isinstance(p, cls)]


@dataclass
class _State:
    episode: int = -1
    step: int = 0
    events: list = field(default_factory=list)
    V: dict = field(default_factory=dict)  # node -> volts
    I: dict = field(default_factory=dict)  # element -> amps (a source: delivered out of plus)
    closed: dict = field(default_factory=dict)  # wire/"tool>node" -> bool
    contacts: dict = field(default_factory=dict)  # tool -> {node: point}
    nets: dict = field(default_factory=dict)  # node -> net id
    faulting: set = field(default_factory=set)
    live: dict = field(default_factory=dict)
    over: dict = field(default_factory=dict)  # source -> seconds over its rating
    energized: dict = field(default_factory=dict)  # load -> seconds since energized (None: not)
    tied: set = field(default_factory=set)
    dead_live: set = field(default_factory=set)
    reverse: set = field(default_factory=set)
    gas: dict = field(default_factory=dict)  # tank -> mL at 1 atm, 20 C
    liq: dict = field(default_factory=dict)  # tank -> mL
    tc: dict = field(default_factory=dict)  # tank -> deg C
    inflow: dict = field(default_factory=dict)  # (tank, medium) -> mL
    delivered: dict = field(default_factory=dict)  # supply -> mL this trial
    press_used: dict = field(default_factory=dict)  # supply -> mL this activation
    sup_on: dict = field(default_factory=dict)
    returning: dict = field(default_factory=dict)  # supply -> seconds left
    frac: dict = field(default_factory=dict)  # pipe -> open fraction
    flows: dict = field(default_factory=dict)  # pipe -> mL/s (+ a to b), last step
    gas_flow: dict = field(default_factory=dict)  # pipe -> mL/s of gas (actual), last step
    liq_in: dict = field(default_factory=dict)  # tank -> mL/s of liquid in, last step
    P: dict = field(default_factory=dict)  # endpoint -> kPa absolute (None: closed)
    latched: set = field(default_factory=set)  # one-shot hazards logged (kind, where)
    flashes: list = field(default_factory=list)  # (point, step)


_S = _State()


def _start_gas(t: Tank) -> float:
    """mL at 1 atm (20 C) of gas in the tank at the start of a trial."""
    c = CUSHION * t.volume
    vg = t.volume + c - t.liquid
    return max(0.0, vg * (PATM + t.psi * KPA_PER_PSI) / PATM * T0 / (t.start_c + 273.15) - c)


def reset() -> None:
    """Back to the start of a trial (the kit also does this whenever kit.episode() changes)."""
    global _S
    _S = _State(episode=kit.episode())
    for t in _of(Tank):
        _S.gas[t.name] = _start_gas(t)
        _S.liq[t.name] = float(t.liquid)
        _S.tc[t.name] = float(t.start_c)
    for s in _of(Supply):
        _S.delivered[s.name] = 0.0
        _S.press_used[s.name] = 0.0
        _S.sup_on[s.name] = False
        _S.returning[s.name] = 0.0


def _fresh() -> None:
    if _S.episode != kit.episode():
        reset()


def step() -> int:
    """Control steps the kit has run in this trial."""
    _fresh()
    return _S.step


def log(kind: str, where: str, value=None, hazard: bool = False, point=None) -> Event:
    """Append an event now (task rules use it from a `watch` function)."""
    e = Event(_S.step, kind, where, None if value is None else float(value), bool(hazard),
              None if point is None else tuple(float(v) for v in point))
    _S.events.append(e)
    if hazard or kind in ("spark", "short", "release"):
        if e.point is not None:
            _S.flashes.append((e.point, _S.step, kind))
    return e


def _once(kind: str, where: str, value=None, point=None) -> None:
    """A hazard event logged once per trial for (kind, where)."""
    if (kind, where) in _S.latched:
        return
    _S.latched.add((kind, where))
    log(kind, where, value, True, point)


# -- geometry ----------------------------------------------------------------------------------------


def _rotm(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _pose(objects, body):
    if body is None:
        return np.zeros(3), np.eye(3)
    s = objects.get(body) if hasattr(objects, "get") else objects[body]
    if s is None:
        return None
    return np.asarray(s.pos, float), _rotm(s.quat_xyzw)


def _world(objects, body, local) -> np.ndarray | None:
    if local is None:
        return None
    pr = _pose(objects, body)
    if pr is None:
        return None
    p, R = pr
    return p + R @ np.asarray(local, float)


def _where_point(objects, part) -> tuple | None:
    w = _world(objects, getattr(part, "body", None), getattr(part, "at", None))
    return None if w is None else tuple(w)


def _pad_dist(objects, pad: Pad, pts: np.ndarray) -> tuple[float, np.ndarray]:
    """Smallest distance from the sample points to the pad's surface (sphere: to its centre minus radius)."""
    pr = _pose(objects, pad.body)
    if pr is None:
        return math.inf, pts[0]
    p, R = pr
    if pad.box is None:
        c = p + R @ np.asarray(pad.point, float)
        d = np.linalg.norm(pts - c, axis=1) - pad.radius
    else:
        loc = (pts - p) @ R - np.asarray(pad.point, float)
        d = np.linalg.norm(np.maximum(np.abs(loc) - np.asarray(pad.box, float), 0.0), axis=1)
    i = int(np.argmin(d))
    return float(d[i]), pts[i]


def _tool_points(objects, tool: Tool, n: int = 13) -> np.ndarray | None:
    a = _world(objects, tool.body, tool.tip)
    b = _world(objects, tool.body, tool.tail)
    if a is None or b is None:
        return None
    s = np.linspace(0.0, 1.0, n)[:, None]
    return a + s * (b - a)


def _val(c, objects, default=True):
    if c is None:
        return default
    if callable(c):
        return c(objects)
    return c


# -- electrical solve --------------------------------------------------------------------------------


class _UF:
    def __init__(self, items):
        self.p = {i: i for i in items}

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def _electrical(objects) -> None:
    wires, loads, sources, tools = _of(Wire), _of(Load), _of(Source), _of(Tool)
    if not (wires or loads or sources or tools):
        return
    dt = _CFG["dt"]
    names = []
    for p in _of(Node):
        names.append(p.name)
    for e in wires + loads:
        names += [e.a, e.b]
    for s in sources:
        names += [s.plus, s.minus]
    for t in tools:
        names.append(f"tool:{t.name}")
        names += [pad.node for pad in t.pads]
    names = list(dict.fromkeys(names))
    idx = {n: i for i, n in enumerate(names)}
    nodes = {p.name: p for p in _of(Node)}

    # which conductors are closed now
    now_closed = {}
    for w in wires:
        now_closed[w.name] = bool(_val(w.closed, objects))
    contacts = {}
    for t in tools:
        pts = _tool_points(objects, t)
        prev = _S.contacts.get(t.name, {})
        touch = {}
        if pts is not None:
            for pad in t.pads:
                d, at = _pad_dist(objects, pad, pts)
                lim = t.radius + (0.002 if pad.node in prev else 0.0)
                if d <= lim and (pad.node not in touch or d < touch[pad.node][0]):
                    touch[pad.node] = (d, tuple(at))
        contacts[t.name] = {n: v[1] for n, v in touch.items()}
        for n in [pad.node for pad in t.pads]:
            now_closed[f"{t.name}>{n}"] = n in touch

    srcs_on = {s.name: bool(_val(s.on, objects)) for s in sources}
    loads_on = {ld.name: bool(_val(ld.on, objects)) for ld in loads}

    def solve():
        n = len(names)
        G = np.eye(n) * 1e-6
        b = np.zeros(n)

        def g(i, j, c):
            G[i, i] += c
            G[j, j] += c
            G[i, j] -= c
            G[j, i] -= c

        for p in nodes.values():
            if p.ground:
                G[idx[p.name], idx[p.name]] += 1e4
        for w in wires:
            if now_closed[w.name]:
                g(idx[w.a], idx[w.b], 1.0 / w.ohms)
        for t in tools:
            for node in contacts[t.name]:
                g(idx[f"tool:{t.name}"], idx[node], 2.0 / t.ohms)
        for ld in loads:
            if loads_on[ld.name]:
                k = ld.surge if (_S.energized.get(ld.name) is not None and _S.energized[ld.name] < ld.surge_s) else 1.0
                g(idx[ld.a], idx[ld.b], k / ld.r)
        for s in sources:
            if srcs_on[s.name]:
                g(idx[s.plus], idx[s.minus], 1.0 / s.ohms)
                b[idx[s.plus]] += s.volts / s.ohms
                b[idx[s.minus]] -= s.volts / s.ohms
        return np.linalg.solve(G, b)

    v = solve()
    V = {nm: float(v[idx[nm]]) for nm in names}
    I = {}
    for w in wires:
        I[w.name] = (V[w.a] - V[w.b]) / w.ohms if now_closed[w.name] else 0.0
    for t in tools:
        cur = [abs(V[f"tool:{t.name}"] - V[nd]) * 2.0 / t.ohms for nd in contacts[t.name]]
        I[t.name] = max(cur) if len(cur) >= 2 else 0.0
        for nd in contacts[t.name]:
            I[f"{t.name}>{nd}"] = (V[f"tool:{t.name}"] - V[nd]) * 2.0 / t.ohms
    for ld in loads:
        if loads_on[ld.name]:
            k = ld.surge if (_S.energized.get(ld.name) is not None and _S.energized[ld.name] < ld.surge_s) else 1.0
            I[ld.name] = (V[ld.a] - V[ld.b]) * k / ld.r
        else:
            I[ld.name] = 0.0
    for s in sources:
        I[s.name] = (s.volts - (V[s.plus] - V[s.minus])) / s.ohms if srcs_on[s.name] else 0.0

    # nets over low-resistance conductors (wires and tool contacts, not loads or sources)
    uf = _UF(names)
    for w in wires:
        if now_closed[w.name]:
            uf.union(w.a, w.b)
    for t in tools:
        for nd in contacts[t.name]:
            uf.union(f"tool:{t.name}", nd)
    nets = {nm: uf.find(nm) for nm in names}

    # make / break / spark
    spark_default = _CFG["spark_amps"]
    elems = [(w.name, w) for w in wires] + [(f"{t.name}>{pad.node}", t) for t in tools for pad in t.pads]
    for key, part in elems:
        was = _S.closed.get(key)
        now = now_closed[key]
        if was is None:  # first step of the trial: the starting state, no event
            continue
        if was != now:
            if isinstance(part, Tool):
                nd = key.split(">", 1)[1]
                pt = contacts[part.name].get(nd) if now else _S.contacts.get(part.name, {}).get(nd)
            else:
                pt = _where_point(objects, part)
            log("make" if now else "break", key, point=pt)
            amps = abs(I.get(key, 0.0)) if now else abs(_S.I.get(key, 0.0))
            lim = part.spark_amps if part.spark_amps is not None else spark_default
            if amps > lim:
                if part.spark_hazard:
                    _once("spark", key, amps, pt)
                else:
                    log("spark", key, amps, False, pt)

    # faults (onset of each)
    # tools always (default limit: the setup's); wires only with their own fault_amps, since the cables in a fault's
    # path carry the same current as the tool that caused it and the arc belongs to the tool
    for key, part in [(w.name, w) for w in wires if w.fault_amps is not None] + [(t.name, t) for t in tools]:
        lim = part.fault_amps if part.fault_amps is not None else _CFG["fault_amps"]
        a = abs(I.get(key, 0.0))
        if a > lim and key not in _S.faulting:
            _S.faulting.add(key)
            if isinstance(part, Tool):
                pts = list(contacts[part.name].values())
                pt = tuple(np.mean(np.array(pts), axis=0)) if pts else None
            else:
                pt = _where_point(objects, part)
            log("short", key, a, True, pt)
        elif a <= lim:
            _S.faulting.discard(key)

    # sources: overload, reverse, ties
    for s in sources:
        out = I[s.name]
        if s.max_amps is not None:
            _S.over[s.name] = _S.over.get(s.name, 0.0) + dt if out > s.max_amps else 0.0
            if _S.over[s.name] >= s.overload_s - 1e-9:
                _once("overload", s.name, out)
        if s.reverse_amps is not None and srcs_on[s.name] and -out > s.reverse_amps:
            _once("backfeed", s.name, -out)
        if s.no_tie and srcs_on[s.name]:
            for o in sources:
                if o.name != s.name and srcs_on[o.name] and (nets[o.plus] == nets[s.plus] or nets[o.minus] == nets[s.plus]):
                    _once("backfeed", f"{s.name}+{o.name}", None)

    # nodes: dead nodes that came alive, watched transitions
    lv = _CFG["live_volts"]
    for p in nodes.values():
        thr = p.live_volts if p.live_volts is not None else lv
        is_live = abs(V[p.name]) >= thr
        if bool(_val(p.dead, objects, False)) and is_live:
            _once("backfeed", p.name, V[p.name], _where_point(objects, p))
        if p.watch and _S.live.get(p.name) is not None and _S.live[p.name] != is_live:
            log("live" if is_live else "dead", p.name, V[p.name], point=_where_point(objects, p))
        _S.live[p.name] = is_live

    # loads: energized timers (for surge)
    for ld in loads:
        rated = ld.volts if ld.volts is not None else lv * 2
        on = loads_on[ld.name] and abs(V[ld.a] - V[ld.b]) >= 0.5 * rated
        if on:
            _S.energized[ld.name] = 0.0 if _S.energized.get(ld.name) is None else _S.energized[ld.name] + dt
        else:
            _S.energized[ld.name] = None

    _S.V, _S.I, _S.closed, _S.contacts, _S.nets = V, I, now_closed, contacts, nets


# -- pressure solve ----------------------------------------------------------------------------------


def _tank_vg(t: Tank) -> float:
    """The gas space: the volume the liquid leaves plus the cushion (hoses and seals give a little)."""
    return max(t.volume + CUSHION * t.volume - _S.liq[t.name], 0.0005 * CUSHION * t.volume)


def _tank_p(t: Tank) -> float:
    """Absolute kPa: the gas plus the cushion's own (fixed, 1 atm-worth) amount in the gas space."""
    return PATM * (_S.gas[t.name] + CUSHION * t.volume) / _tank_vg(t) * (_S.tc[t.name] + 273.15) / T0


def _supply_p(s: Supply, objects) -> float | None:
    if _S.sup_on[s.name]:
        if s.capacity is not None and _S.press_used[s.name] >= s.capacity - 1e-9:
            return None  # the pedal is on the floor: nothing more moves
        return PATM + float(_val(s.psi, objects, 0.0)) * KPA_PER_PSI
    if _S.returning[s.name] > 0:
        return PATM + float(s.return_psi) * KPA_PER_PSI
    if s.idle == "pool":
        return PATM
    return None


def _pressure(objects) -> None:
    tanks, supplies, pools, pipes = _of(Tank), _of(Supply), _of(Pool), _of(Pipe)
    if not (tanks or supplies or pipes):
        return
    dt = _CFG["dt"]
    parts = _CFG["parts"]
    # supplies on/off and their activations
    for s in supplies:
        on = bool(_val(s.on, objects))
        if on and not _S.sup_on[s.name]:
            _S.press_used[s.name] = 0.0
        if not on and _S.sup_on[s.name] and s.return_psi is not None:
            _S.returning[s.name] = s.return_s
        elif not on and _S.returning[s.name] > 0:
            _S.returning[s.name] = max(0.0, _S.returning[s.name] - dt)
        _S.sup_on[s.name] = on
    # temperatures and gas made
    for t in tanks:
        hot = bool(_val(t.heat, objects, False)) if t.heat is not None else False
        target, tau = (t.hot_c, t.heat_s) if hot else (t.start_c, t.cool_s)
        _S.tc[t.name] += (target - _S.tc[t.name]) * (1.0 - math.exp(-dt / max(tau, 1e-6)))
        if hot and t.make_gas and _S.tc[t.name] > 60.0:
            _S.gas[t.name] += t.make_gas * dt

    def p_of(name):
        if name == AMBIENT:
            return PATM
        e = parts[name]
        if isinstance(e, Tank):
            return _tank_p(e)
        if isinstance(e, Pool):
            return PATM + e.head_cm * KPA_PER_CM_WATER
        return _supply_p(e, objects)

    # pipe openings, open/close and release events (checked before any flow this step)
    for p in pipes:
        f = _val(p.open, objects)
        f = 1.0 if f is True else 0.0 if f is False or f is None else min(max(float(f), 0.0), 1.0)
        was = _S.frac.get(p.name)
        if was is not None:
            pt = _where_point(objects, p)
            if was <= 1e-3 < f:
                log("open", p.name, f, point=pt)
            elif f <= 1e-3 < was:
                log("close", p.name, f, point=pt)
            if p.release_psi is not None and f >= 0.999 > was:
                pa, pb = p_of(p.a), p_of(p.b)
                if pa is not None and pb is not None and abs(pa - pb) > p.release_psi * KPA_PER_PSI:
                    _once(p.release_kind, p.name, abs(pa - pb) / KPA_PER_PSI, pt)
        _S.frac[p.name] = f

    n = _CFG["substeps"]
    h = dt / n
    flows = {p.name: 0.0 for p in pipes}
    gflow = {p.name: 0.0 for p in pipes}
    liq_in = {t.name: 0.0 for t in tanks}
    ports = {}
    for p in pipes:
        for e in (p.a, p.b):
            ports[e] = ports.get(e, 0) + 1
    for _ in range(n):
        P = {name: p_of(name) for name in [AMBIENT] + [e.name for e in tanks + supplies + pools]}
        moves = []
        for p in pipes:
            f = _S.frac[p.name]
            pa, pb = P[p.a], P[p.b]
            if f <= 1e-3 or pa is None or pb is None:
                continue
            dp = pa - pb
            if p.one_way and dp <= 0:
                continue
            q = p.flow * f * dp / KPA_PER_PSI  # mL/s, actual volume at the upstream side
            dv = q * h
            # limiter: never move more than half of what would equalize the two sides (shared among a tank's pipes)
            caps = []
            for e in (p.a, p.b):
                ent = parts.get(e)
                if isinstance(ent, Tank):
                    caps.append(_tank_vg(ent) / max(P[e], 1e-3) / ports[e])
            if caps:
                ceq = 1.0 / sum(1.0 / c for c in caps)
                lim = 0.5 * abs(dp) * ceq
                dv = math.copysign(min(abs(dv), lim), dv)
            up, dn, port = (p.a, p.b, p.port_a) if dv > 0 else (p.b, p.a, p.port_b)
            vol = abs(dv)
            src = parts.get(up)
            if isinstance(src, Supply):
                if src.max_flow is not None:
                    vol = min(vol, src.max_flow * h)
                if _S.sup_on[up] and src.capacity is not None:
                    vol = min(vol, max(0.0, src.capacity - _S.press_used[up]))
            moves.append((p, up, dn, port, vol, P[up]))
        for p, up, dn, port, vol, pup in moves:
            if vol <= 0:
                continue
            src = parts.get(up)
            # what leaves the upstream side: (gas actual mL, liquid mL)
            if up == AMBIENT:
                g_act, l_act = vol, 0.0
            elif isinstance(src, Pool):
                g_act, l_act = 0.0, vol
            elif isinstance(src, Supply):
                g_act, l_act = (vol, 0.0) if src.medium == "gas" else (0.0, vol)
            else:  # a tank
                gas_act = _S.gas[up] * PATM / max(pup, 1e-3) * (_S.tc[up] + 273.15) / T0
                liq = _S.liq[up]
                if port == "top":
                    g_act = min(vol, gas_act)
                    l_act = min(vol - g_act, liq)
                elif port == "mixed":
                    tot = gas_act + liq
                    share = gas_act / tot if tot > 1e-12 else 1.0
                    g_act = min(vol * share, gas_act)
                    l_act = min(vol - g_act, liq)
                else:
                    l_act = min(vol, liq)
                    g_act = min(vol - l_act, gas_act)
            g_amt = g_act * pup / PATM * T0 / ((_S.tc[up] + 273.15) if up in _S.tc else T0)
            if up in _S.gas:
                _S.gas[up] = max(0.0, _S.gas[up] - g_amt)
                _S.liq[up] = max(0.0, _S.liq[up] - l_act)
            if isinstance(src, Supply):
                if _S.sup_on[up]:
                    _S.press_used[up] += g_act + l_act
                _S.delivered[up] += g_act + l_act
                if src.refill is not None and l_act > 0:
                    _S.liq[src.refill] = max(0.0, _S.liq[src.refill] - l_act)
            dst = parts.get(dn)
            if dn in _S.gas:
                _S.gas[dn] += g_amt
                room = dst.volume + 0.999 * CUSHION * dst.volume - _S.liq[dn]
                _S.liq[dn] += min(l_act, max(room, 0.0))
                _S.inflow[(dn, "gas")] = _S.inflow.get((dn, "gas"), 0.0) + g_amt
                _S.inflow[(dn, "liquid")] = _S.inflow.get((dn, "liquid"), 0.0) + l_act
                liq_in[dn] += l_act / dt
            elif isinstance(dst, Supply) and dst.refill is not None and l_act > 0:
                _S.liq[dst.refill] += l_act  # back to the reservoir
            sign = 1.0 if up == p.a else -1.0
            flows[p.name] += sign * (g_act + l_act) / dt
            gflow[p.name] += g_act / dt
    _S.flows, _S.gas_flow, _S.liq_in = flows, gflow, liq_in
    _S.P = {name: p_of(name) for name in [AMBIENT] + [e.name for e in tanks + supplies + pools]}

    for t in tanks:
        pt = _where_point(objects, t)
        if t.rating is not None and (_S.P[t.name] - PATM) / KPA_PER_PSI > t.rating:
            _once("burst", t.name, (_S.P[t.name] - PATM) / KPA_PER_PSI, pt)
        if t.protect in ("liquid", "gas") and _S.inflow.get((t.name, t.protect), 0.0) > t.protect_ml:
            _once(t.protect_kind, t.name, _S.inflow[(t.name, t.protect)], pt)
        if t.dry_ml is not None and _S.liq[t.name] < t.dry_ml:
            _once(t.dry_kind, t.name, _S.liq[t.name], pt)


# -- the step --------------------------------------------------------------------------------------


def update(objects) -> None:
    """One control step: solve the network from the object states and log events. The hook calls it once per
    control step; offline tests call it directly."""
    _fresh()
    _S.step += 1
    _electrical(objects)
    _pressure(objects)
    for fn in _CFG["watchers"]:
        fn(objects)


# -- scorer API ------------------------------------------------------------------------------------


def volts(objects, node: str, ref: str | None = None) -> float:
    """The node's voltage to ground (0 V), or to node `ref`."""
    _fresh()
    v = _S.V.get(node, 0.0)
    return v - (_S.V.get(ref, 0.0) if ref is not None else 0.0)


def live(objects, node: str, ref: str | None = None, volts_min: float | None = None) -> bool:
    """The node is at least `volts_min` (default the setup's live_volts) from ground or from `ref`."""
    lim = volts_min if volts_min is not None else _CFG["live_volts"]
    return abs(volts(objects, node, ref)) >= lim


def amps(objects, element: str) -> float:
    """The current in a wire (a to b), load (a to b), source (out of plus), tool (largest contact) or "tool>node"."""
    _fresh()
    return float(_S.I.get(element, 0.0))


def closed(objects, element: str) -> bool:
    _fresh()
    return bool(_S.closed.get(element, False))


def contacts(objects, tool: str) -> set:
    """The nodes the tool's bare metal touches now."""
    _fresh()
    return set(_S.contacts.get(tool, {}))


def connected(objects, a: str, b: str) -> bool:
    """Nodes a and b are on one low-resistance net now (through closed wires and tool contacts)."""
    _fresh()
    return a in _S.nets and _S.nets.get(a) == _S.nets.get(b)


def pressure(objects, name: str, unit: str = "psi") -> float:
    """Gauge pressure of a tank, supply or pool now (0 for a closed supply)."""
    _fresh()
    p = _S.P.get(name)
    if p is None:
        return 0.0
    return (p - PATM) / UNITS[unit]


def gas(objects, tank: str) -> float:
    """Gas in the tank, mL at 1 atm and 20 C."""
    _fresh()
    return float(_S.gas.get(tank, 0.0))


def liquid(objects, tank: str) -> float:
    _fresh()
    return float(_S.liq.get(tank, 0.0))


def room(objects, tank: str, psi: float) -> float:
    """mL of liquid the tank still takes before its gauge pressure reaches `psi` (its gas and its cushion squeezed
    to that pressure); negative if it is already above it. (Local addition, brake-bleed: a master cylinder's fluid
    stop, mechkit.Piston's `room`.)"""
    _fresh()
    t = _CFG["parts"][tank]
    p_abs = PATM + psi * KPA_PER_PSI
    squeezed = (_S.gas[tank] + CUSHION * t.volume) * PATM / p_abs * (_S.tc[tank] + 273.15) / T0
    return _tank_vg(t) - squeezed


def inflow(objects, tank: str, medium: str = "liquid") -> float:
    """mL of that medium ("liquid", or "gas" as mL at 1 atm) that has flowed into the tank this trial."""
    _fresh()
    return float(_S.inflow.get((tank, medium), 0.0))


def delivered(objects, supply: str) -> float:
    _fresh()
    return float(_S.delivered.get(supply, 0.0))


def flow(objects, pipe: str) -> float:
    """mL/s through the pipe in the last step (+ from a to b)."""
    _fresh()
    return float(_S.flows.get(pipe, 0.0))


def opening(objects, pipe: str) -> float:
    _fresh()
    return float(_S.frac.get(pipe, 0.0))


def temp(objects, tank: str) -> float:
    _fresh()
    return float(_S.tc.get(tank, 20.0))


def set_gas(tank: str, ml: float) -> None:
    _S.gas[tank] = max(0.0, float(ml))


def set_liquid(tank: str, ml: float) -> None:
    _S.liq[tank] = max(0.0, float(ml))


def _match(e: Event, kind, where) -> bool:
    return (kind is None or e.kind == kind) and (where is None or e.where == where)


def events(objects=None, kind: str | None = None, where: str | None = None) -> list:
    """The ordered event log of this trial, optionally filtered."""
    _fresh()
    return [e for e in _S.events if _match(e, kind, where)]


def hazards(objects=None, kind: str | None = None, where: str | None = None) -> list:
    return [e for e in events(objects, kind, where) if e.hazard]


def hazard(objects=None, kind: str | None = None, where: str | None = None) -> Event | None:
    """The first hazard event this trial (of that kind / at that element), or None. Latched: it stays."""
    h = hazards(objects, kind, where)
    return h[0] if h else None


def first(objects, kind: str, where: str | None = None) -> int | None:
    """The step of the first such event, or None."""
    e = events(objects, kind, where)
    return e[0].step if e else None


def last(objects, kind: str, where: str | None = None) -> int | None:
    e = events(objects, kind, where)
    return e[-1].step if e else None


def order_ok(objects, sequence) -> bool:
    """True if the (kind, where) pairs (or "kind:where" strings; where None = any) occur in the log in this order,
    each strictly after the one matched before it (a subsequence: other events may come between)."""
    log_ = events(objects)
    i = 0
    for item in sequence:
        kind, where = item.split(":", 1) if isinstance(item, str) else item
        while i < len(log_) and not _match(log_[i], kind, where):
            i += 1
        if i == len(log_):
            return False
        i += 1
    return True


# -- condition helpers -----------------------------------------------------------------------------


def attached(key: str, target: str | None = None):
    """fn(objects): a cablekit fitting `key` ("cable/fitting") is on `target` (or on anything)."""

    def f(objects):
        import cablekit as ck

        t = ck.attached(objects, key)
        return t is not None if target is None else t == target

    return f


def joint_past(key: str, at: float, sign: int = 1):
    """fn(objects): a yamkit joint ("mech/link") is at or past `at` in the `sign` direction (a breaker thrown)."""
    return lambda objects: sign * (kit.joint(objects, key).pos - at) >= 0


def joint_fraction(key: str, shut: float, full: float):
    """fn(objects) -> 0..1: a valve handle's opening, 0 at `shut`, 1 at `full` (either direction)."""

    def f(objects):
        q = kit.joint(objects, key).pos
        return min(max((q - shut) / (full - shut), 0.0), 1.0)

    return f


def backed_off(key: str, crack_deg: float, open_deg: float):
    """fn(objects) -> 0..1: a twistkit threaded coupling `key` ("body/cap"): 0 until backed off `crack_deg` past
    its seat (still sealing), rising to 1 at `open_deg`, and 1 once off its thread."""

    def f(objects):
        import twistkit as tk

        if tk.is_off(objects, key):
            return 1.0
        d = tk.backed_off_deg(objects, key)
        return min(max((d - crack_deg) / (open_deg - crack_deg), 0.0), 1.0)

    return f


def inside(body: str | None, point, region_body: str | None, center, half):
    """fn(objects): `point` on `body` is inside the box `center` +- `half` in `region_body`'s frame (a hose end
    under the water in a jar, a probe in a socket)."""

    def f(objects):
        w = _world(objects, body, point)
        pr = _pose(objects, region_body)
        if w is None or pr is None:
            return False
        p, R = pr
        loc = (w - p) @ R - np.asarray(center, float)
        return bool(np.all(np.abs(loc) <= np.asarray(half, float)))

    return f


def near(body: str | None, point, other: str | None, other_point, r: float):
    """fn(objects): two points on two bodies are within r of each other."""

    def f(objects):
        a, b = _world(objects, body, point), _world(objects, other, other_point)
        return a is not None and b is not None and float(np.linalg.norm(a - b)) <= r

    return f


def all_of(*cs):
    return lambda objects: all(bool(_val(c, objects)) for c in cs)


def any_of(*cs):
    return lambda objects: any(bool(_val(c, objects)) for c in cs)


def not_(c):
    return lambda objects: not bool(_val(c, objects))


def times(*cs):
    """fn(objects) -> the product of fractions/bools (a valve's opening while a hose is on)."""

    def f(objects):
        out = 1.0
        for c in cs:
            v = _val(c, objects)
            out *= 1.0 if v is True else 0.0 if v is False or v is None else float(v)
        return out

    return f


def gauge(name: str, full: float, sweep: float, zero: float = 0.0, unit: str = "psi"):
    """A `fn(J, mem)` for `mechkit.Follow(fn=...)`: the needle angle `zero + sweep * p / full`, clamped to the dial
    (p in `unit`, gauge; a vacuum gauge uses a negative `full`)."""

    def f(J, mem):
        p = pressure(None, name, unit)
        return zero + sweep * min(max(p / full, 0.0), 1.0)

    return f


# -- visuals ---------------------------------------------------------------------------------------

_VIS = {"markers": None, "failed": False}
FLASH_SLOTS, SCORCH_SLOTS, BUBBLE_SLOTS = 3, 3, 4
FLASH_R, SCORCH_R, LAMP_R, BUBBLE_R = 0.008, 0.007, 0.006, 0.004
AIR_R, AIR_H = 0.016, 0.002  # the air disc's radius at gas_full, and its thickness


def _markers():
    if _VIS["markers"] is not None or _VIS["failed"]:
        return _VIS["markers"]
    try:
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        def look(rgb, emissive=(0.0, 0.0, 0.0)):
            lin = tuple(kit._linear(c) for c in rgb)
            return sim_utils.PreviewSurfaceCfg(diffuse_color=lin, emissive_color=emissive, roughness=0.4)

        protos = {
            "flash": sim_utils.SphereCfg(radius=1.0, visual_material=look((1.0, 0.95, 0.55), (4.0, 3.6, 1.6))),
            "scorch": sim_utils.CylinderCfg(radius=1.0, height=1.0, axis="Z", visual_material=look((0.06, 0.05, 0.04))),
            "lamp_on": sim_utils.SphereCfg(radius=1.0, visual_material=look((1.0, 0.1, 0.05), (3.0, 0.2, 0.1))),
            "lamp_off": sim_utils.SphereCfg(radius=1.0, visual_material=look((0.25, 0.22, 0.2))),
            "bubble": sim_utils.SphereCfg(radius=1.0, visual_material=look((0.95, 0.97, 1.0))),
            "bead": sim_utils.SphereCfg(radius=1.0, visual_material=look((0.35, 0.6, 0.95))),
            "air": sim_utils.CylinderCfg(radius=1.0, height=1.0, axis="Z",
                                         visual_material=look((0.97, 0.97, 0.97), (0.6, 0.6, 0.6))),
        }
        _VIS["markers"] = VisualizationMarkers(VisualizationMarkersCfg(prim_path="/World/Visuals/livekit",
                                                                       markers=protos))
        _VIS["index"] = {k: i for i, k in enumerate(protos)}
    except Exception as e:  # noqa: BLE001  (visuals are optional)
        print(f"livekit: no visuals ({e!r})", file=sys.stderr, flush=True)
        _VIS["failed"] = True
    return _VIS["markers"]


def _draw(objects, origin) -> None:
    m = _markers()
    if m is None:
        return
    idx = _VIS["index"]
    tr, sc, ids = [], [], []

    def put(kind, p, r, h=None):
        tr.append(tuple(np.asarray(p, float) + origin))
        sc.append((r, r, h if h is not None else r))
        ids.append(idx[kind])

    def hide(kind):
        tr.append((0.0, 0.0, -10.0)), sc.append((1e-5,) * 3), ids.append(idx[kind])

    hold = max(1, int(round(_CFG["flash_s"] / _CFG["dt"])))
    recent = [f for f in _S.flashes if _S.step - f[1] < hold][-FLASH_SLOTS:]
    for p, s, kind in recent:
        put("flash", p, FLASH_R * (1.0 + 0.6 * ((_S.step - s) % 2)))  # flickers
    for _ in range(FLASH_SLOTS - len(recent)):
        hide("flash")
    burns = [e.point for e in _S.events if e.kind == "short" and e.point is not None][:SCORCH_SLOTS]
    for p in burns:
        put("scorch", (p[0], p[1], p[2] - 0.001), SCORCH_R, 0.0015)
    for _ in range(SCORCH_SLOTS - len(burns)):
        hide("scorch")
    for nd in _of(Node):
        if nd.lamp:
            p = _where_point(objects, nd)
            if p is None:
                hide("lamp_off")
            else:
                put("lamp_on" if _S.live.get(nd.name) else "lamp_off", p, LAMP_R)
    shown = [p for p in _of(Pipe) if p.show]
    for p in shown[:BUBBLE_SLOTS]:
        pt = _where_point(objects, p)
        g = abs(_S.gas_flow.get(p.name, 0.0))
        if pt is None or g < 0.05:
            hide("bubble")
        else:
            put("bubble", (pt[0], pt[1], pt[2] + 0.004 * (_S.step % 3)), BUBBLE_R * min(2.0, 0.7 + g / 5.0))
    for t in [t for t in _of(Tank) if t.show]:
        pt = _where_point(objects, t)
        if pt is None or _S.liq_in.get(t.name, 0.0) < 0.05:
            hide("bead")
        else:
            put("bead", pt, BUBBLE_R * 1.5)
    for t in [t for t in _of(Tank) if t.gas_show]:
        pt = _where_point(objects, t)
        full = t.gas_full if t.gas_full is not None else _start_gas(t)
        frac = _S.gas.get(t.name, 0.0) / full if full > 0 else 0.0
        if pt is None or frac < 0.01:
            hide("air")
        else:
            put("air", pt, AIR_R * math.sqrt(min(frac, 4.0)), AIR_H)
    if tr:
        # identity as (x, y, z, w), the convention pourkit's markers use and were verified with in the sim (Nov 8);
        # the earlier (1, 0, 0, 0) is a half turn in that convention (harmless for spheres, not for future shapes)
        qs = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (len(tr), 1))
        m.visualize(translations=np.array(tr, float), orientations=qs, scales=np.array(sc, float),
                    marker_indices=np.array(ids, int))


# -- hook into yam_arena.scene (after yamkit's and any kit imported before this one) -------------------

_orig_patch = kit._patch


def _patch(scene) -> None:
    _orig_patch(scene)
    if getattr(scene, "_livekit", False):
        return
    scene._livekit = True
    inner = scene.object_states
    last_key = {"key": None}

    def object_states(env, names):
        out = inner(env, names)
        base = env.unwrapped
        buf = getattr(base, "episode_length_buf", None)
        key = (kit.episode(), int(buf[0].item()) if buf is not None else None)
        if key[1] is None or key != last_key["key"]:  # once per control step
            last_key["key"] = key
            update(out)
            if os.environ.get("LIVEKIT_DEBUG"):
                print("livekit", _S.step, {k: round(v, 2) for k, v in _S.V.items()},
                      {k: round(pressure(out, k), 2) for k in _S.P if k != AMBIENT},
                      [(e.kind, e.where) for e in _S.events if e.step == _S.step], flush=True)
            try:
                _draw(out, base.scene.env_origins[0].detach().cpu().numpy())
            except Exception as e:  # noqa: BLE001  (never let the visuals break a run)
                if not _VIS["failed"]:
                    print(f"livekit: visuals off ({e!r})", file=sys.stderr, flush=True)
                _VIS["failed"] = True
        return out

    scene.object_states = object_states


kit._patch = _patch
if "yam_arena.scene" in sys.modules:
    _patch(sys.modules["yam_arena.scene"])


# -- demo helpers ------------------------------------------------------------------------------------


def press(xyz, depth: float, hold: float, *, yaw: float = 0.0, approach: float = 0.05, label: str = "press") -> list:
    """Waypoints for a closed-finger push straight down on a button, pedal pad or toggle at `xyz` (its top), `depth`
    m down, held for `hold` s, then back up: approach above, press, hold, release."""
    from yam_arena.constants import FINGER_CLOSED
    from yam_arena.demo import Waypoint

    x, y, z = (float(v) for v in xyz)
    return [Waypoint(1.0, (x, y, z + approach), FINGER_CLOSED, f"{label} above", yaw),
            Waypoint(0.8, (x, y, z + 0.003), FINGER_CLOSED, f"{label} touch", yaw),
            Waypoint(0.6, (x, y, z - depth), FINGER_CLOSED, f"{label} down", yaw),
            Waypoint(max(hold, 0.05), (x, y, z - depth), FINGER_CLOSED, f"{label} hold", yaw),
            Waypoint(0.6, (x, y, z + approach), FINGER_CLOSED, f"{label} up", yaw)]


def push_across(start, end, *, yaw: float = 0.0, steps: int = 4, seconds: float = 1.6, lift: float = 0.04,
                label: str = "throw") -> list:
    """Waypoints for a closed-finger stroke from `start` to `end` (a toggle thrown, a lever swung in its plane, a
    valve handle pushed round): lower at start, sweep in `steps` pieces, lift clear."""
    from yam_arena.constants import FINGER_CLOSED
    from yam_arena.demo import Waypoint

    a, b = np.asarray(start, float), np.asarray(end, float)
    out = [Waypoint(1.0, tuple(a + [0, 0, lift]), FINGER_CLOSED, f"{label} above", yaw),
           Waypoint(0.7, tuple(a), FINGER_CLOSED, f"{label} on", yaw)]
    for i in range(1, steps + 1):
        out.append(Waypoint(seconds / steps, tuple(a + (b - a) * i / steps), FINGER_CLOSED, f"{label} {i}", yaw))
    out.append(Waypoint(0.7, tuple(b + [0, 0, lift]), FINGER_CLOSED, f"{label} off", yaw))
    return out
