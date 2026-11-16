# Clean an SC fibre connector's ferrule on a cassette cleaner, then latch it into a panel adapter

A yellow 3 mm fibre patch cord runs from a cord clip on the table to a loose blue SC/UPC connector. The connector is
true size: a 9 x 7.4 mm housing 25 mm long, with a white 2.5 mm ferrule standing about 2 mm proud of its front (the 2 mm is an estimate: see Materials). A grey
cassette end-face cleaner (CLETOP-S size, 125 x 85 x 35 mm, 150 g) stands loose on its long edge on the left. It has
a 30 x 10 mm window of white cleaning tape near the top of its front face, with an arrow pointing down. On the right
is a 1U patch-panel section with one beige SC simplex adapter. A second connector is already latched into its back.

The robot must wipe the ferrule's end face once along the tape, flat and in the arrow's direction, for most of the
window's length. It must then push the connector into the adapter until the adapter's detent pins snap into the
housing's side notches, and let go. After the wipe, the end face must touch nothing except the adapter's sleeve (and
more tape). Touching the table, the panel face, the adapter's walls, the cleaner's body or a finger makes the trial
score 0.

This is the routine that comes before every fibre mating in a data center: "It is important that every fiber
connector be inspected and cleaned prior to mating" and "Never touch the end face of the fiber connectors" (Cisco).
One hand holds the cassette (and its lever) while the other draws the ferrule down the tape: "Pull the fiber tip
lightly down the exposed cleaning area in the direction of the arrow or from top to bottom" and "Do not scrub the
fiber against the fabric or clean over the same surface more than once" (Cisco).

Both arms are needed. The cleaner starts at x 30, y +25 cm.
- Offline IK (the arm's own solver, straight-down grip) leaves the right arm at least 83 mm short of the tape
  window there, at any wrist yaw. It is 38 mm short of even the cleaner's nearest corner.
- The adapter's port is out of the left arm's reach: it ends 86 to 133 mm short of the lined-up pose in front of
  the port.

So the left arm has to bring the cleaner to the right arm, and the right arm has to make the connection. In the demo
the left arm also holds the cleaner steady during the wipe, as a technician does. The cleaner is a free 150 g box,
and the stroke drags on it. The scorer does not require the cleaner to be held.

## Ability tested

Ability tested: keeping a process order on a millimetre-scale part. The robot makes one valid, one-way, flat wipe
of a 2.5 mm end face on a held tool. It then carries the part to a tight port and inserts it (0.25 mm per side) until
it latches, with the cleaned face never touching anything on the way.

## Closest tasks

Each shares an element with this task. The differences are in what is scored.

- TK-XU9FCD (patch cord into a tester): pushes each end of a cord into a cable-tester jack, with a plug body
  55 x 16 x 14 mm. There is no cleaning step and no rule about what the plug's tip touches.
- DataCenterBench Tier 1 "join": inserting an Ethernet plug into a coupler (as listed in that benchmark). It has no
  cleaning stage and no contamination rule.
- TK-7FS22W (swap an optical transceiver, with dust caps), TK-TWBUEN (route a patch cord round a radius guide),
  TK-MEUKHW (patch-cord slack, then plug in): fibre handling in a rack, without wiping an end face.
- TK-U3S7ZZ (an earlier submission of the clean-then-plug idea): a 55 x 14 x 12 mm connector body wiped across a
  flat pad in a stroke of at least 4 cm, then pushed at least 15 mm into a socket, judged by depth. This task
  differs in the connector and adapter size (9 x 7.4 mm housing, 2.5 mm ferrule, 0.25 mm per side), the detent
  latch, a cassette cleaner brought over by the other arm, and a touch rule tracked from physics contacts.
- TK-B3MNT7 (scrub an IV hub before attaching): the same clean-then-connect order, but in a clinical setting, with
  a scrubbing motion and a luer connection.
- TK-GCU8JK (wipe a monitor lead): wiping a cable, with no mating step.
- The Industrial Dexterity Benchmark (arXiv 2607.14021) defines a real-hardware "fiber cable maintenance" task:
  unplug a cable, brush the end of the connector on a cleaning pad, and re-insert it. Its baseline uses an SC
  simplex connector and a 3-D-printed circular pad, and its total score leaves out the cleaning phase.

## Materials

| Item | Value used | Source |
|---|---|---|
| SC plug housing | 9.0 x 7.4 mm, 25 mm long | Hirose HSC series catalogue (plug 9 x 7.4 mm, housing 25 mm) |
| Ferrule | 2.5 mm diameter (zirconia), about 2 mm proud of the housing | Hirose (ferrule 2.499 +/- 0.0005 mm); IEC 61754-4 (2.5 mm nominal). The protrusion is not given in any source checked and is an estimate |
| SC adapter | 27.4 mm long, 12.8 x 9.4 mm body, 1.8 mm flange | Hirose HSC adapter drawing |
| Panel cutout | 13 x 9.5 mm | Hirose HSC adapter panel cutout |
| Alignment sleeve | 2.5 mm bore (modelled as a rigid 3.2 mm square bore) | Senko zirconia sleeves (ID 2.49, OD 3.2 mm) |
| Latch | two detent pins, about 1 N preload each, snap into 0.6 mm side notches | Simplified. Hirose lists a simplex-plug engagement force of 19.6 N max (an upper limit) |
| Cassette cleaner | 125 x 85 x 35 mm (W x H x D), 150 g | CLETOP-S datasheet (bismon.com). AFL's series sheet lists 130 x 75 x 40 mm, 160 g |
| Tape window | 30 x 10 mm, arrow top to bottom | Estimated from product photos and the Cisco wording "from top to bottom" |
| Patch cord | 3 mm jacket, 28 cm free length | Common simplex patch cord |

Sources (quotes are verbatim):

- Cisco, "Inspection and Cleaning Procedures for Fiber-Optic Connections" (doc 51834):
  https://www.cisco.com/c/en/us/support/docs/optical/synchronous-digital-hierarchy-sdh/51834-cleanfiber2.html
- CLETOP-S user manual: "While pressing the lever and keeping your thumb depressed on it, press the optical plug
  ferrule endface against the cleaning tape surface and move it slowly in arrow's direction."
  https://manuals.online/ntt/ultrasonic-jewelry-cleaner/cletop-s-series/user-manual/no2o
- CLETOP-S datasheet (125 x 85 x 35 mm, 150 g): https://www.bismon.com/pdf/Cletop-S.pdf ; CLETOP series (AFL):
  https://assets.testequity.com/te1/Documents/pdf/AFL/Cletop_Cassette_Cleaner_Series_Data_Sheet.pdf
- Fluke Networks, "Golden Rule of Fiber Inspection": "Contaminated connections remain the number one cause of fiber
  related problems and test failures in data centers ..."
  https://www.flukenetworks.com/blog/cabling-chronicles/golden-rule-fiber-inspection-keeps-times
- FOA: "Clean and dry the end of the ferrule just before insertion."
  https://www.thefoa.org/tech/ref/termination/cleaning.html
- Hirose HSC catalogue: https://dtsheet.com/model/1946362 ; https://static.datasheets.com/doc/23976698-hirose-hscf-2a-d-p--ds.pdf ; IEC 61754-4 preview:
  https://www.vde-verlag.de/iec-normen/preview-pdf/info_iec61754-4%7Bed1.0%7Db.pdf
- Senko zirconia sleeves: https://www.senko.com/product/zirconia-sleeves/
- Cabling Installation & Maintenance on fibre cleaning in AI networks:
  https://cablinginstall.com/design-install/article/55279720/keeping-ai-and-ml-fiber-optic-networks-running-the-vital-role-of-fiber-cleaning

## Requirements

- `yam-isaac-lab-arena` with Isaac Lab-Arena installed (`source IsaacLab-Arena/yam.env`).
- `cablekit.py` is in this folder (the cable simulation).
  `task.py` adds the following on top of the kit:
  - the connector's parts on the cord's end plug;
  - the adapter and panel as fixed colliders;
  - the detent pins as sprung prismatic joints;
  - the cleaner's tape, frame and arrow (drawn only);
  - a tracker that runs at every physics substep.

## Run a trial

```bash
inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 --scorer task_score \
    --max-steps 2000 \
    --instruction "Clean the blue fibre connector on the cleaner, then latch it into the panel's SC adapter."
```

## Scripted demo

The demo script is not part of this submission. It was run for at most 1500 control steps.

The demo is not camera-driven: it reads the simulator's poses of the connector and the cleaner to plan its
moves and to correct them in closed loop. It shows that the task can be done, not that a camera policy can do it.

1. The two arms work at the same time:
   - The right arm pinches the connector 27 mm behind the ferrule tip, over the rear of the housing and the boot,
     lifts it and draws it back.
   - The left arm grips the cleaner's top edge 55 mm from its centre, with the fingers across its 35 mm thickness.
     It carries the cleaner 25 cm to the right, to x 30, y 0, sets it down and keeps holding it.
2. The right arm turns the connector to point into the cleaner's face. It comes in from 35 mm in front, then lines
   the ferrule up with the top of the tape window using the measured tip position.
3. It presses the end face onto the tape, 1.5 mm past the surface plus 2 mm of preload. It draws the face down 23 mm
   along the arrow in 2.5 s, then pulls a little further until the tip reaches the planned end.
4. It backs the connector off the tape and lifts it over the cleaner, to 14 cm above the table. It then swings to
   the panel side. Meanwhile the left arm lets go of the cleaner and moves away.
5. The connector sits a few degrees askew in the fingers after the wipe (4.7° of yaw in one run). Without a fix, this
   jams the housing in the 0.25 mm channel. So the right arm turns its wrist until the connector's axis points
   straight into the port.
6. It lines the ferrule up with the port (closed loop, to within 0.15 mm) and pushes the connector home. Both detent
   pins deflect about 0.3 mm and snap into the notches.
7. It opens the fingers and backs away.

In the recorded run (`demo/`), the valid stroke registered at 21 mm, and the full stroke was 23.4 mm. The connector
latched with the ferrule tip 13.4 mm deep, and the score was 1.0. Success ends the episode at step 813
(about 41 s at 20 Hz). `demo/` holds the three camera videos, `setup.png` and `result.json`.

What the cameras see:
- The right wrist camera looks down along the connector. It sees the connector, the white ferrule tip, the cleaner
  and the adapter and panel during insertion.
- During the wipe it looks steeply down onto the cleaner's top. The tape window, on the upright face, is seen almost
  edge-on.
- The left wrist camera mostly sees the table while it holds the cleaner.
- The top camera sees the whole layout.

## Scoring (`score()`, evaluated at the end)

The tracker runs at every 240 Hz physics substep. It reads the connector's live pose and the PhysX contacts
between the connector and every other body. A contact counts as "on the ferrule" when it lies on the 2 mm ferrule
protrusion: axially within 1.85 mm of the tip, and radially within 2.3 mm of the axis. The housing's front face is
2 mm back, so contacts there do not count.

- **Valid clean**: all of the following, in one stroke:
  - the ferrule is in contact with the cleaner inside the tape window;
  - its axis is within 10° of the tape's normal;
  - the whole end face (its rim) is inside the window;
  - the face slides at least 21 mm along the arrow;
  - it never moves back against the arrow by more than 1.5 mm. A reversal breaks the stroke (no scrubbing).

  Hovering a fraction of a millimetre off the tape makes no contact and does not count.
- **Latched to 0** for the rest of the trial if:
  - the ferrule enters the adapter before a valid clean; or
  - after the valid clean, the ferrule touches anything other than the tape window, the adapter's sleeve bore and
    mouth, or the far connector's ferrule. This includes the table, the cleaner body, the panel, the adapter face
    and walls, the gripper fingers and any other object.
- Otherwise, at the end:
  - **1.0**: latched, and no gripper within 35 mm of the connector.
    - Latched means the connector is lined up with the port (within 0.6 mm and 4°) and the housing is home (front
      face at 10.3–11.8 mm deep).
    - Both detent pins must also be back within 0.15 mm of rest, that is, snapped into the notches. A pin still
      pushed out means the housing is not home.
  - **0.75**: latched but still held.
  - **0.5**: ferrule in the adapter, not latched.
  - **0.25**: cleaned, not in the adapter.
  - **0**: not cleaned.

## Shortcuts and partial credit (each run in the sim, scored by `score()`)

Each variant is a scripted run of the same arms, scored by `score()`. The scripts are not part of this submission; the results are from recorded runs.

| `SC_DEMO=` | what it does | result |
|---|---|---|
| `nothing` | waits | 0 (not cleaned) |
| `dirty` | skips the cleaner; inserts and latches the connector, then lets go | 0 (entered the adapter before a valid clean; it was latched and released) |
| `short` | a correct flat stroke on the tape, but only 10 mm long (9.8 mm registered); then latches and lets go | 0 (no valid clean; latched and released) |
| `hover` | the whole wipe 1.2 mm off the tape, flat and along the arrow; then latches and lets go | 0 (no tape contact at all; latched and released) |
| `touch` | a valid wipe, then the ferrule touches the panel's steel face 3.5 cm beside the adapter; then latches and lets go | 0 (cleaned face touched `panel_l`; latched and released) |
| (accidental) | an early version of `touch` swung the connector into the left arm, which still held the cleaner | 0 (cleaned face touched a gripper finger, `tip_right`) |
| `clean_only` | a valid wipe, then holds the connector in the air | 0.25 |
| `no_latch` | a valid wipe, then pushes the connector only 6 mm into the adapter (ferrule tip 5.8 mm in) and stops there | 0.5 |
| `hold` | a valid wipe, then latches the connector but keeps holding it | 0.75 |
| `correct` | the full demo | 1.0 |

The "hover" and "short" rows show that the cleaning needs real contact over most of the window's length. A
near-miss is not enough. The "dirty", "touch" and gripper rows show that a fully latched connector still scores
0 if the order or the no-touch rule is broken.

## What is simplified (fidelity)

- Clearance:
  - The housing has 0.25 mm per side in the adapter channel.
  - The sleeve is a rigid square bore with 0.35 mm per side. A real split sleeve grips the ferrule, so alignment
    here is looser.
  - The ferrule's spring is not modelled: the tips stop 0.2 mm apart.
- Latch force: the detent pins preload about 1 N each. The Hirose SC simplex-plug engagement force limit is 19.6 N max,
  so the arm needs less push here than in reality.
- Cleaner:
  - The cassette's shutter and lever are drawn but not working. Holding the cassette stands in for holding the
    lever down.
  - The window's size and position are estimated.
  - The tape is the cassette's surface, with no fabric texture or compliance.
  - "Clean" is judged geometrically from contact, angle, and stroke length and direction. Dust is not simulated.
- Ferrule shape:
  - The ferrule collider is a 2.5 mm square prism, drawn round.
  - The end face is flat; a real UPC ferrule face is domed.
- The connector's cord is a chain of rigid segments with bending springs.
- Not modelled: end-face inspection with a scope; the connector's key orientation (the key is drawn only, and the
  channel is symmetric).

## Runtime and seeds

- Run limit 2,000 steps (100 s at 20 Hz), about 2.5 times the demo's 813 steps.
- One trial takes about 6 to 10 minutes of wall time on a shared GPU (2 to 3.5 steps/s), plus 1 to 2 minutes of
  Isaac Sim start-up.
- The layout is fixed: use `object_jitter=0`. The cord is laid straight from its clip at every reset.
