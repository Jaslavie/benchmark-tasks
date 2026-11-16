# Bleed the air out of a brake caliper, shutting the bleeder each cycle before the pedal comes back up

The job is to push the air out of a brake caliper, cycle after cycle, until the pedal is firm. A clear bench brake
trainer: a brake pedal on a stand works a master cylinder whose steel line runs to a disc-brake caliper.
An 11 mm box-end wrench is already on the caliper's bleeder screw, and a clear hose runs from the bleeder into a catch
jar. There is air in the caliper, so the pedal is spongy. "Bleed the air out of this brake caliper." In the two-person
pedal method one hand presses the pedal and keeps it down, the other opens the bleeder a quarter turn so fluid and air
push out and the pedal sinks to the floor, then shuts it while the pedal is still down, and only then does the pedal
come up; repeat until no bubbles come out. Letting the pedal up while the bleeder is open can draw air straight back in at the bleeder, because the master
cylinder's piston springs back and pulls on the line.

Run (from the repo root, Isaac Lab-Arena installed):

    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=tasks/brake-bleed-close-before-release/task.py \
      -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
      --instruction "Bleed the air out of this brake caliper." --scorer task_score --max-steps 1000
    python tasks/brake-bleed-close-before-release/test_task.py      # offline scorer checks, prints "all pass"

Use `--policy dream --max-steps 2400` for a Dream-1 trial. The scene is deterministic (no seeds, object_jitter=0).
The scripted demo plan is 842 steps (42.1 s: five bleed cycles of 7 s each, plus the approach and the return home)
at most; the trial ends once the right hand has lifted off the shut wrench and the bled state has held 1 s (the
arms going home is not scored), at step 789, so run it with `--max-steps 1000`; a full trial is 2,400 steps. Set
`LIVEKIT_DEBUG=1` to print the line's pressure and the event log every step.

Files: `task.py` (scene, network, scorer, demo), `test_task.py`, and the kits it imports, copied here: `yamkit.py`,
`mechkit.py`, `guardkit.py` (trimmed to the contact check) and `livekit.py` (imported last).

## Spec

Positions in cm from the midpoint between the two arm bases: x forward, y to the robot's left, table top z 0.

| Object | What it is | Where |
|---|---|---|
| Pedal stand | steel post 3 x 4 cm on a 12 x 10 cm foot plate | hinge at x 38, y 16, 8 cm up |
| Brake pedal | steel lever 11 cm long, black rubber pad 4.5 x 5 x 1.2 cm; 3.3 cm of travel at the pad (17 degrees) to a floor stop; return spring about 2 N at rest, 4.5 N fully down | pad centered x 27, y 16, top 9 cm up at rest |
| Master cylinder | aluminum, 3.2 cm across, 7 cm long along x, translucent reservoir 6 x 4 x 4.5 cm with MIN and MAX lines and amber fluid | behind the hinge, x 41 to 48, y 16 |
| Brake line | 4.8 mm steel tube along the table | from the master cylinder to the caliper's back |
| Caliper | black cast-iron disc-brake caliper 9 x 6 x 6 cm on a 5 x 4 x 3 cm steel bracket | centered x 30, y -12, 3 to 9 cm up |
| Bleeder screw | 11 mm hex (12.7 mm across its corners) with a yellow paint ring, nipple to 11.8 cm | x 29, y -11, on the caliper's top |
| Wrench | 11 mm chrome box-end wrench on the bleeder's hex; handle 12.5 cm from the axis, 1.6 cm wide, 0.8 cm thick | 10 cm up; points -y at shut (start), +x a quarter turn open; detents at both, stops 3 degrees past each |
| Bleeder hose | clear vinyl, 4.8 mm inside, 8 mm outside, from the nipple up and over to the jar, its end 2 cm under the fluid | |
| Catch jar | glass, 7 cm across, 9 cm tall, in a wire holder, 4.5 cm of dark old fluid; an inverted graduated tube (1 cm across) over the hose's end collects the air | x 21.5, y -4.5 |

Start: pedal up, bleeder shut, 1.2 mL of air in the caliper and its line (8 mL).

Mechanisms: the wrench is a hinge about z on the bleeder's axis with `mk.Detents` at shut and a quarter turn open
(10-degree catch zones). The pedal is a hinge about y with a return spring (preloaded against its up stop) and a
master cylinder, `mk.Piston` (in `mechkit.py`), coupled to the pressure network:
- The piston covers the compensating port 8% into the pedal's stroke and from there sweeps 4.2 mL by the floor stop
  (15.2 mL per radian of pedal; a 19 mm bore moving about 1.6 cm over the whole stroke).
- The fluid stop: every control step the pedal joint's lower limit is set where the piston would have pushed the
  line up to 6 psi, the push of the hand (about 6 N on the pad for the fluid, at most about 9 N with the return
  spring). With the bleeder shut that takes only the fluid that squeezes the air, so a pressed pedal stops at 16% of
  its stroke with the starting 1.2 mL of air, and at about 9% once bled. A hard joint limit, not a stiff spring, so a gripper pushing on the pad cannot pop through it.
- Held on the piston, the hand keeps the line at 6 psi (a livekit supply into the line); whatever leaves through the
  open bleeder moves the piston, and the stop with it, so the pedal sinks as fast as the bleeder passes fluid (test:
  in 1 s, 0% of the stroke at 15 degrees of wrench, 5% at 25, 19% at 40, 55% at a quarter turn) and reaches the floor.
- Let up, the pedal leaves the piston, which comes back on its own spring only as fast as fluid fills behind it, with
  5 psi of suction on the line: past the cup seal from the reservoir and, if the bleeder is open, up the hose from the
  jar and as air around the bleeder's threads. The port opens again only once the piston is back (about 1 s after a
  full stroke). With the bleeder shut, the piston only takes back the fluid it pushed.

Pressure network (`livekit`, solved every control step): the reservoir (open to the air), the closed side (the master
cylinder's chamber, the line and the caliper as one 8 mL compartment with 1.2 mL of air, guarded by `protect="gas"`:
more than 0.01 mL of air flowing in logs a latched `air_in` hazard), the hand's push (6 psi while the pedal is held
on the piston), the compensating port and the cup seal (reservoir to line, 1.0 mL/s per psi), the jar (fluid 2 cm over
the hose's end), the bleeder (0.6 mL/s per psi at a quarter turn, its opening rising from 0 at 20 degrees of wrench,
carrying air and fluid in proportion), and the bleeder's threads (a one-way pipe from the room into the line, 0.1 mL/s
per psi, open as soon as the bleeder is past 20 degrees: a bleeder seals at its seat, not its threads).

## Scoring

Air drawn back in is latched for the whole trial; everything else is read from the final state, which must hold 1 s
(SCORE_OVER = "end").

| Outcome | Score |
|---|---|
| More than 0.01 mL of air drawn into the caliper at any time, latched (the pedal let up with the bleeder past 20 degrees: the returning piston pulls air in at its threads) | 0 |
| A gripper still touching the pedal pad or the wrench at the end (its palm or fingers within 2 mm) | 0 |
| Pedal or wrench still moving faster than 0.2 rad/s | 0 |
| Wrench more than 10 degrees from shut (bleeder left open) | 0 |
| Pedal not back up: more than 8% down, or its piston not yet back past the port | 0 |
| Air left above 0.84 mL (less than 30% out; doing nothing ends here) | 0 |
| Air left 0.84 mL or less but above 0.36 mL | 0.25 |
| Air left 0.36 mL or less but above 0.2 mL | 0.5 |
| Air left 0.2 mL or less | 1.0 |

Possible scores: 0, 0.25, 0.5, 1.0. Where the arms end up does not count.

Why each rule, and how a real trial checks it (the volumes are scaled: 1.2 mL of air in an 8 mL line, so that a few
cycles clear it):
- The air left is the score because it is what makes a pedal spongy and a car slow to stop. Each correct cycle
  (pedal pressed, bleeder opened a quarter turn until the pedal is on the floor, shut, pedal up) pushes about 4 mL
  through the caliper and takes about 40% of the air left with it: 1.2, 0.72, 0.42, 0.25, 0.15 mL after one to four
  cycles (test_task). A real trial reads it as 1.2 mL minus the jar's inverted graduated tube.
- 1.0 at 0.2 mL or less: the firm, high pedal a technician checks for before the car goes back (the pedal then stops
  at about 9% of its stroke). Four correct cycles reach it, three do not.
- 0.5 at 0.36 mL or less (70% out): most of the air is out but the pedal is still a little soft, so the job is not
  done. Three correct cycles reach it, two do not; so do five cycles with the bleeder opened only 45 degrees (the
  pedal sinks slower and less fluid goes through).
- 0.25 at 0.84 mL or less (30% out): real progress, but the pedal is still spongy. One correct cycle reaches it;
  five cycles with the bleeder cracked only 25 degrees do not (0.9 mL left).
- 0 above 0.84 mL: doing nothing, pumping the pedal with the bleeder shut, or opening the bleeder without pressing
  leave all 1.2 mL in.
- Air drawn back in, more than 0.01 mL (one bubble about 3 mm across), latched: this is what the procedure's order
  prevents. With the bleeder shut before the pedal comes up, none comes in at all; letting the pedal up with the
  bleeder still open draws 0.02 mL at 25 degrees and 0.23 mL at a quarter turn (test_task), and pumping with the
  bleeder held cracked draws more on every stroke, so the one-handed shortcut (crack the bleeder, pump the pedal)
  scores 0. It stays latched although more strokes would push that air out again: ten pedal-up-first cycles in a row
  never reach a firm pedal, and the score should follow the procedure's one rule rather than how many extra strokes a
  run had time for. A real trial checks on the video that the wrench was inside the 20-degree witness mark before the
  pedal started up, and watches the clear-body caliper (see Real-world build) and the clear bleeder hose, where air
  drawn in shows as bubbles rising.
- 20 degrees: where the bleeder's seat lifts and it starts passing fluid; from there its threads let air in, since a
  bleeder seals at its seat, not its threads.
- Wrench within 10 degrees of shut at the end: the detent's catch zone, inside which the detent pulls the bleeder
  snug. Outside it the bleeder is left open and can leak fluid onto the caliper. Real trial: the 10-degree witness mark.
- Pedal within 8% of its stroke (2.6 mm at the pad) of the top, its piston back: the compensating port. A pedal left
  further down keeps the port shut, sealing the line off from the reservoir, so the brake can drag. Real trial: the
  ruler beside the pad.
- Hands off at the end: a hand still on the pedal or the wrench means the job is not finished, since a hand resting
  on the pad can hold the pedal down and one on the handle can still turn the bleeder. The check is contact: the
  gripper's own shape (palm and both fingers, boxes fitted to the YAM's collision meshes, at their present opening)
  within 2 mm of the pad or the wrench, the 2 mm being the solver's resting gap plus the fit of the boxes. A hand
  lifted clear, or fingers left open either side of the handle without touching it, does not count, however close.
  Real trial: the end of the video shows both grippers off the pad and the wrench (no gap needed beyond that).
- 6 psi in the line under the hand: set by what one arm pushes comfortably. The push and the fluid are tied (push
  times pad travel equals line pressure times the volume swept), so with the 4.2 mL per stroke that clears the air in
  four cycles, a lower pressure is what keeps the push light: at most about 9 N on the pad, spring included. The
  sim's shoulder joints are limited to 27 N*m (i2rt's yam_v1 settings), and 9 N
  pressed straight down about 0.3 m from the left arm's shoulder takes about 3 N*m, a ninth of that. 6 psi still
  drives fluid and air out of an open bleeder (gravity bleeding works on well under 1 psi). Real trial: a 0 to 15 psi
  gauge teed into the line. A car's pedal makes hundreds of psi; 6 psi is scaled to what the YAM can push, and that
  is enough because what is scored is the hydraulic order of events (press, open, shut, release), not the pressure.
- 0.2 rad/s and 1 s held: the pedal and the wrench at rest, not still springing back. Real trial: the video.

The modeled parts: the fluid, the air and the piston's pressure are a model (above), not simulated fluid. They behave
like the real rig because the pedal's travel and the air out both follow the fluid that actually leaves the bleeder.
A real trial checks the physical outcomes instead: the air collected in the graduated tube, the pedal's travel on the
ruler (on the first press it stops about a fifth of the way down, and it drops to the floor only while the bleeder is
open), bubbles in the clear hose, and the wrench against its witness marks.

In a real trial: inject 1.2 mL of air into the caliper by syringe at reset, refill and re-invert the graduated tube,
and check the items above from the video and a top photo at the end.

## Real-world build

- A bench brake-bleeding trainer from parts: a single-circuit master cylinder with a 19 mm (0.75 in.) bore and its
  own reservoir, on a stand with an 11 cm pedal lever and a short arm 5.3 cm above the pedal's hinge that works the
  pushrod, so the piston moves about 1.6 cm over the pad's 3.3 cm; a 4.8 mm (0.19 in.) steel brake line with flare fittings;
  and a single-piston disc-brake caliper on a steel bracket. One arm presses
  the pad: about 6 N holds the line at 6 psi, plus the return spring's 2 to 4.5 N, so at most about 9 N.
- A caliper whose bleeder screw has an 11 mm hex (measure it; 8 and 10 mm are also common, and the wrench must match),
  an 11 mm box-end wrench, about 1 ft of clear vinyl hose 4.8 mm (0.19 in.) inside (a snug push fit on the nipple), a glass
  jar in a wire holder, and a 2 mL graduated tube (or a 3 mL syringe barrel, tip sealed) inverted over the hose's end
  in the jar to collect the air.
- A clear bench rig: a clear-body caliper or a clear acrylic stand-in for it, so the air pocket can be seen.
- Fluid: propylene-glycol RV antifreeze (nontoxic, water-clean, safe on the robot's table), not brake fluid (DOT 3
  and 4 strip paint and irritate), not mineral oil (it swells the ordinary seals of a DOT-fluid master cylinder) and
  not DOT 5 silicone. Keep the rig over a tray.
- A 5 mL syringe with a short tube to inject 1.2 mL of air at reset through a fill port on the caliper.
- A light return spring on the pedal (about 4.5 N at the pad pressed), so trials are safe for a YAM setup. A small
  ruler on the stand next to the pad shows the pedal's travel on the camera, and paint witness marks on the caliper's
  top at 10 and 20 degrees from the wrench's shut position show where it is.

## How it differs from nearby board tasks

- Lower a cabinet with a hydraulic release valve: one hand meters one valve to control a load's
  descent speed. Here two hands must be timed against each other, cycle after cycle: the pedal hand may only let go
  after the wrench hand has shut the bleeder, and the score comes from the air the network moves.
- Hold a pedal while the other hand swaps a filter: one static hold, released once at the end; nothing depends
  on when it is released relative to a valve.
- Double block and bleed before removing a cap, verify zero pressure before opening: venting is a
  precondition for opening; here bleeding air out is the job.
- Fit a caliper over the disc / fit pads: the same equipment, no hydraulics.
- Connect jump leads in the safe order: a connection order for spark safety, nothing timed.
