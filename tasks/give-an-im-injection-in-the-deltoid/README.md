# Give a 1 mL intramuscular injection into a deltoid trainer with a retractable safety syringe

The robot uncaps a 3 mL retractable-needle syringe (25 G x 1 in, 1.0 mL of water standing in for a 1 mL IM vitamin B12
dose) and drops the cap in a waste cup. It finds the deltoid site 25-50 mm below the shoulder's bony edge on a strap-on
trainer whose height changes by trial, and puts the needle in square to the skin up to the hub. It keeps the needle in
the muscle and still while the other hand pushes 1 mL steadily (about 10 s) and on through the detent so the needle
retracts. Then it drops the syringe in a sharps container. The scorer is `score()` / `tier()` / `Trial` in
`im_injection.py`.

## Run

Needs `yam-isaac-lab-arena` with Isaac Lab-Arena (`source <ARENA_DIR>/yam.env`). `yam_bodies.py` (bodies, switchable
joints, contact readings) and `yam_demokit.py` (the scripted-demo runner) are in this folder.

```bash
inspect-robots run --embodiment yam_arena --policy dream -E task=im_injection.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 --scorer task_score --max-steps 3600 --instruction "$(jq -r .instruction task.json)"
ARENA_DIR=<arena checkout> YAM_EVAL_DIR=<log dir> \
tools/run_injection.sh inj-full 3600 scripted full 0       # the scripted demo
tools/run_injection.sh inj-phys 3600 scripted spike        # the demo with the physics checks ([spike] PASS/FAIL lines)
tools/run_injection.sh inj-steep 1200 scripted steep 0     # a 25 deg press must not go in
tools/run_injection.sh inj-push 560 scripted pushonly 0 INJ_PUSHONLY=1   # the left finger's push alone, barrel held on its rest
tools/run_injection.sh inj-dream 3600 dream full 0         # a policy (Dream-1) through the bridge
python -m pytest -q test_im_injection.py                   # scorer, layout, demo reach, clearance, views; no simulator
```

- Trial settings: `INJ_HEIGHT` (0, 1, 2 spacer layers) and `INJ_TRAINER_T` (m, 0.043-0.050) set the trial's height and
  skin x. The real trial follows a fixed rotation: trial k stands on k mod 3 layers of 18.3 mm plywood.
- Seed and jitter: one seed (0), no object jitter.
- Runtime: the scripted demo ran 1,101 control steps (55 s at 20 Hz, inside the 3,600-step run limit) and took 16
  minutes of wall-clock time on one GPU shared with other jobs. The Dream-1 run used all 3,600 steps and took 44 minutes.
- Diagnostics: `INJ_TRACE` (a trace every 0.25 s; `run_injection.sh` sets it), `INJ_PROBE` and `INJ_ZERO_WRENCH`. They
  only log or switch applied forces off for diagnosis, and are unset in scored runs.
- The full-mode run prints a scorer check: the step checks ran, the score before the stow is at most 0.75, and 1.0 after
  it. The check only prints and never voids a run. If the syringe is not stowed, the run prints each stow condition:
  barrel, rod, cap and both grippers' clearance.

## Recorded runs (scores as the scorer reported them)

Both runs are on the files in this folder, with no spacer (`INJ_HEIGHT=0`, acromion edge 25.0 cm) and the 50 mm trainer.

- Scripted demo (`tools/run_injection.sh <name> 3600 scripted full 0`): **1.0**, reached and held 1 s at step 1,101.
  One puncture, D 37.5 mm, L 0.3 mm, angle 0.3 deg, full depth, continuous in muscle; the needle moved at most 0.05 mm
  and 1.2 deg between S and the retraction. The stopper broke loose at 3.0 N, S was 1.0 s later at the 0.9 mL line, and
  the 0.8 / 0.6 / 0.4 / 0.2 / 0 lines were reached 0.94, 2.84, 4.85, 6.84 and 8.86 s after S (1.7 mm/s under a thumb
  load of 3.3-3.7 N); the retraction fired 1.4 s after the 0 line. Syringe in the container, cap in the cup.
- Dream-1 (`tools/run_injection.sh <name> 3600 dream`): **0.0**. It used all 3,600 steps: the arms hovered
  near the syringe rest and the sharps container, nothing was grasped and there was no puncture.

## Scene and assets

All task geometry is boxes built in code (`im_injection.py`, `yam_bodies.py`). That covers the stand, the trainer, the
syringe and its parts, the rest, the tray, the sharps container, the cup and the camera mounts. No third-party mesh, scan
or texture is used. The arms, cameras and table are the `yam_arena` station the benchmark supplies.

- Stand: 3/4 in plywood base 22 x 22 cm and spacers; 4 in Schedule 40 PVC floor flange, 19.0 cm across, with a
  13.5 x 5 cm collar; pipe 11.43 cm OD, 6.0 mm wall, 30 cm long, axis at (36.0, 0) cm.
- Trainer: Erler-Zimmer 7130 (17.5 x 8.5 x 5.0 cm), strapped to the pipe over a 1 mm silicone sheet. The colliders are
  boxes in its measured profile: a flat lateral face at the site, sides falling back 30 deg, and the shoulder cap 6 mm
  back above the acromion's edge. What the cameras see is a visual-only arm contour and acromion ridge, within the
  colliders or behind the skin's plane.
- Syringe: VanishPoint 3 mL, 25 G x 1 in, printed 0.1 mL graduations (longer every 0.5 mL), 3 mm marker bands at the
  2.0 mL line. The barrel's collider is a 10.8 mm square, so it cannot roll in a grip. The plunger, cap, needle and its
  carrier are their own bodies.
- Sharps container: nominal 2.5 L, outside 15 x 11 x 15 cm, lid opening 11 x 7 cm (the purchased one is checked
  against these). Waste cup: 9 cm across and 6 cm tall, its 10-sided wall and floor modelled.
- Table: the station's 1.10 x 1.30 m wooden top. The two C-clamps holding the stand are not modelled (the stand is
  fixed). The overhead camera's 1 m boom is drawn but is not a collider; its upright is not modelled.

## Scoring (task.json `scoring`; `Trial` in `im_injection.py`)

The scoring text is what a person applies to the two calibrated videos and the graduations. Where the simulation needs a
number for a visible outcome, it is given here.

- In muscle: hub-to-skin gap 5 mm or less.
- Full depth, "the hub touches the skin": simulation gap 1 mm or less.
- Lines and graduations: a line is reached when the stopper's front edge is on or past it. The simulation has no
  tolerance: the edge's volume is at or below the line.
- S: the edge reaches the graduation 0.1 mL below its reading at the first in-muscle frame, that reading taken to the
  nearest printed graduation (midway reads up). A 1.0 mL fill reads 1.0 whether the edge sits a hair above or below the
  line, so S is the 0.9 line: the edge has to travel at least half a graduation (0.05 mL, 0.85 mm) to reach it. S is
  therefore the start of the push and cannot fire at the insertion with the stopper at rest. A push begun before the hub
  is on the skin does start S there, and the rest of the insertion then counts against needle still. The first timed
  interval runs from S to the first 0.2 mL line below the edge at S and is prorated by that volume (from the 0.9
  line to the 0.8 line: 0.5-2 s).
- Steady push, from S until the edge reaches the 0 line. All three parts are needed for 1.0:
  - Rate: each 0.2 mL line is reached 1-4 s after the previous one, the first interval (S to the first 0.2 mL line)
    prorated by its volume. 1-4 s is half to twice the taught 10 s per mL (2 s per 0.2 mL).
  - No stop: no 3 s in which the edge reaches no new printed 0.1 mL graduation. At the slowest rate the band allows
    (4 s per 0.2 mL) a graduation is reached every 2 s, and at the taught rate every 1 s. So 3 s, 1.5 times the
    slowest spacing, flags only a stopped plunger.
  - Thumb on: no 3 consecutive side-camera frames show a 2 mm gap between the pushing finger and the thumb rest; frames
    hiding the thumb rest don't count, and over 1.0 s of them in total means thumb on is not credited. In simulation,
    the robot's contact load on the plunger under 0.2 N for 0.1 s or more (3 frames at 30 fps) is the thumb off; no
    frame is hidden for this clause there.
- Timing ends on the frame the edge reaches the 0 line. The 10 N push through the 0 line's stop (the retraction) and any
  wait before it are not timed.
- Stowed: the syringe with its plunger rod wholly inside the container below the rim, the cap in the cup, and both
  grippers at least 5 cm from both. The rule and the demo share one check (`stow_check`).
- Zeros (each an e-stop event on the real rig):
  - the unretracted bare needle touching anything but the skin or the container's inside, either arm included; the
    cap's inside is exempt only while uncapping, and the dropped cap is read from geometry;
  - more than 0.2 mL pushed before the first puncture;
  - the bare syringe dropped or set down outside the container, its rest included: at rest with no fingertip load
    for 0.5 s. A syringe uncapped while it lies unheld on its rest counts, so the cap comes off a held syringe;
  - the needle bent more than 10 deg off the barrel's axis on any side-camera frame;
  - the syringe let go (no fingertip load for 0.5 s) with its needle in the trainer;
  - a puncture less than 20 mm below the acromion.
- Fail-safes: a non-finite reading, a non-unit rotation, time not running forward, a missing body, or a built scene that
  does not match the constants (checked on the first step) voids the trial (0).

## Fidelity (task.json `fidelity` is the short form)

- Rigid bodies at 240 Hz on GPU PhysX, at real sizes.
- Needle in tissue: a rule on a D6 joint between the trainer and the syringe. The needle goes in when its tip presses the
  skin at 1.0 N or more within 20 deg of the local normal. It stops at the hub (25.4 mm). A steeper press records a
  puncture but never goes in. The trainer is rigid and the needle does not enter it: the needle's carrier slides back
  into the barrel as the barrel advances, which keeps the hub's approach, the depth and the angle real. Once in, the
  joint holds the syringe's rotation within 20 deg of the skin's normal.
- Tissue forces: 0.8 N of stick-slip along the needle and a 300 N/m hold on its line. Both read the needle tip's
  measured motion, its position differentiated and low-passed over 20 ms. The solver's velocity of the gripped barrel
  carries an offset with the barrel still, and fed that velocity, the tissue forces drove the stopper.
- Plunger stiction: a hard stop with 0.2 mm of play, released by the rule when the measured thumb load reaches 3 N.
- Plunger glide: about 2 N of elasto-plastic friction (0.5 mm presliding), applied as a force pair on the plunger and
  the barrel. After the retraction the glide keeps holding the plunger, in place of the real syringe's lock-down latch.
- Liquid through the needle: what makes a real push slow. Hagen-Poiseuille for water (1.0 mPa s) through the 25 G x
  1 in needle (bore 0.26 mm nominal, 25.4 mm long) from a 59.2 mm2 barrel gives 226 kPa per mL/s: 13.4 N per mL/s on
  the plunger, 1.3 N at the taught 0.1 mL/s (790 N s/m of stopper speed; 500-1,250 N s/m over the bores ISO 9626 allows
  a 25 G needle). A push of 3.3 N gives the taught rate, and twice the force about four times the rate, so a light
  touch cannot run the dose out. It is the plunger joint's damping: the barrel and plunger are one reduced-coordinate
  articulation (barrel the floating root, one prismatic joint), which the GPU solver integrates exactly. Measured on
  the GPU with known forces on the stopper and no robot: 1 N does not move it, 2.5 N and 4 N move it at 0.63 and
  2.53 mm/s (2.00 N + 790 N s/m). As two free bodies on an ordinary joint the same damping acted at under a tenth of
  its value whenever the barrel was held by contact, and the glide's force pair cancelled. The damping cannot be
  switched at run time, so it also acts over the detent's 2 mm past the 0 line and on the retracted plunger.
- 0 line: a stop that gives only to 10 N of thumb load. 2 mm past it the retraction fires and the needle is pulled into
  the barrel.
- Not simulated: needle bending under load, tearing, dimpling, layered tissue resistance, back-pressure, skin spreading,
  the finger sleeves and the C-clamps.
- Forces are nominal values, checked on the purchased parts before trials (table below).

## How the demo works

- Right hand: one grip from the rest to the sharps container, fingers forward and 30 deg down, pads on the barrel's +-y
  faces 30-52 mm behind the hub.
- Uncapping: at (14, -13, 22) cm, clear of the trainer. The left hand's level fingers pinch the cap from +y and pull it
  45 mm straight along the measured barrel axis, slowly through the fit.
- Push: the left's closed fingertips meet the thumb rest from 3 mm behind at 2 mm/s, stopping at 0.3 N of measured
  load. The press then ramps to the 3 N breakaway and the hand keeps pressing: it follows a 0.1 mL/s target against the
  glide and the liquid's resistance (about 3.3 N on the thumb rest), the fingertips on the thumb rest throughout, then
  2 mm/s through the detent.
- The `spike` mode adds physics checks:
  - the grip holding 15 N, the cap pull straight, the breakaway and the steady push;
  - the views, and the retraction;
  - a free stopper not drifting (at most 0.5 mm in 2 s with nothing pushing), after the breakaway and after the
    retraction;
  - the scorer itself (the step checks ran; the score at most 0.75 before the stow and 1.0 after it);
  - every joint torque at most 9 N m.

## Scoring one real trial (worksheet)

About 10 minutes per trial for one person with the two videos in a player that steps frame by frame (an estimate from
the list below; not yet timed on a real trial). In order:

1. After the trial, at the trainer: place the midline card, find the new puncture, measure D and L with the steel
   rule; count the punctures (raking-light photo). Weigh the used syringe (0.9 g or more used).
2. Side video: note any e-stop event (needle touch, bent needle, syringe set down or let go, 0.2 mL pushed early).
3. Side video, first in-muscle frame (hub 5 mm from the skin): read the stopper edge to the nearest graduation, the
   pitch; overhead video, same frame: the yaw. A is the worse of the two.
4. Side video: the frame the hub touches the skin (full depth); the frame S (edge on the graduation 0.1 mL below the
   reading in step 3); the frames the edge reaches the 0.8, 0.6, 0.4, 0.2 and 0 lines; the retraction frame. Write the
   six times down: the intervals give the rate, and no 3 s may pass without a new 0.1 mL graduation.
5. Between S and the 0 line: step through for a 2 mm gap between the pushing finger and the thumb rest (3 frames).
6. Between S and the retraction: at S, at each line and at the retraction, read the hub's x and z and the pitch (side)
   and the hub's y and the yaw (overhead) against their values at S (3 mm, 5 deg); count hidden frames.
7. End of the video: syringe below the container's rim, cap in the cup, both grippers 5 cm clear, held 1 s.
8. Read the tier from task.json `scoring`.

## Camera calibration (real trial)

The two scoring cameras stay fixed for a session. Each is calibrated with a printed 10 mm grid card (a 100 x 100 mm
grid, lines 0.3 mm, checked against the steel rule):

- Side camera, at session start: hold the card vertical in the y = 0 plane (against the midline card's notch), its
  lower edge on the base, and photograph it. This gives mm per pixel along x and z for the hub's position, the pitch
  and the hub-to-skin gap, and shows the image is square to the plane (grid lines within 1 deg of the frame's edges).
- Overhead camera, at session start and at each new spacer height: lay the card horizontal at the syringe's working
  height (on a block about 4 cm below the acromion edge, the site's height), centred on y = 0, and photograph it.
  This gives mm per pixel along x and y for the yaw, the hub's y and L.
- Check: the 100 mm scale bar on the base's +y face measures 100 +- 1 mm in the side view. A calibration is redone if
  a camera is bumped or the check fails.
- The barrel's own graduations (1.69 mm per 0.1 mL) are read directly in the side view and need no calibration.

## Reset

See task.json `reset_notes` (about 3 minutes).

## Nominal values, checked on the purchased parts before trials (constants at the top of `im_injection.py`)

| Value | In simulation | How |
|---|---|---|
| Trainer thickness t at the site; penetrable depth (go/no-go >= 28 mm) | 50 mm; 43 mm tested too | Calipers; a needle with a depth stop |
| Skin's local surface normal at the site | horizontal, -x | A square on the skin, in both calibration photos |
| Flange diameter | 190 mm | Calipers |
| Graduations from the hub (0 / 1.0 / 2.0 mL) | 10.0 / 26.9 / 43.8 mm | Calipers on a real unit |
| Cap pull-off | 6 N | Spring scale |
| Plunger breakaway; glide; push force at 0.1 mL/s with water | 3 N; 2 N; 3.3 N | Spring scale on the thumb rest, stopwatch |
| Retraction (detent) force and travel | 10 N over 2 mm | Spring scale on the thumb rest of a capped unit |
| Puncture force at 90 deg | engages at 1.0 N | Kitchen scale under the trainer |
| Syringe mass with 1 mL | 9 g + parts | 0.01 g scale |

## Real rig: parts and costs (US, with shipping)

| Part | Cost |
|---|---|
| Two Erler-Zimmer 7130 trainers (EUR 110.67 each incl. VAT), plus EU-US shipping and duty | about $300 |
| 4 in Sch 40 PVC pipe (2 ft) and floor flange; 3/4 in plywood; screws, 2 C-clamps | about $70 |
| 1 mm silicone anti-slip sheet; straps | about $15 |
| Needlestick level 4+ gloves, cut into finger sleeves | about $40 |
| 0.01 g scale; 10 N spring scale; tongs; steel rule; calipers | about $70 |
| Sharps containers, about 2.5 L, outside at most 15 x 11 x 15 cm, opening at least 11 x 7 cm (6) | about $30 |
| VanishPoint 3 mL 25 G x 1 in (box of 100) | about $60-90 |
| Polypropylene tray, absorbent pads, paper cups, tape | about $20 |
| Clamp stand and 1 m extrusion boom; printed calibration cards | about $60 |
| Two 1080p 30 fps USB cameras (if the lab has none) | about $80 |

## Sources

- O*NET OnLine: 31-9092.00 Medical Assistants ("...performing such tasks as giving injections or removing sutures"),
  29-2061.00 Licensed Practical and Licensed Vocational Nurses, 29-1141.00 Registered Nurses.
- CDC, "Administering Vaccines: Dose, Route, Site, and Needle Size" and the IM injection job aid: deltoid site 2-3
  fingerbreadths below the acromion, 90 degrees, 1 in needle for 60-70 kg adults. Paraphrased: the job aid's wording
  could not be checked word for word.
- Ernstmeyer K, Christman E (eds.), *Nursing Skills* (Open RN, Chippewa Valley Technical College; CC BY-SA 4.0),
  section 18.6 "Administering Intramuscular Medications"
  (https://med.libretexts.org/Bookshelves/Nursing/Nursing_Skills_(OpenRN)/18:_Administration_of_Parenteral_Medications/18.06:_Administering_Intramuscular_Medications,
  fetched 2026-11-18): the deltoid site "about 1 inch to 2 inches (2.5 cm to 5 cm) below the acromion process";
  "Intramuscular injections are administered at a 90-degree angle"; "administering medications at 10 seconds per mL is
  an effective rate for IM injections". The same rate is taught in Perry, Potter et al., *Clinical Nursing Skills and
  Techniques* (not checked page by page).
- Prasetyono TOH, Adhistana P. Laboratory study on injection force measurement on syringe and needle combinations.
  Malays J Med Sci 2019;26(2):66-76 (doi 10.21315/mjms2019.26.2.8): a 1 mL syringe's break-loose 0.75-1.15 N and glide
  0.13-0.71 N. The simulation's 3 N / 2 N are a firmer stopper than that.
- Erler-Zimmer Ref. 7130 strap-on vaccination trainer, maker's listing: 17.5 x 8.5 x 5.0 cm, 300 g, palpable shoulder
  anatomy, puncture-proof back.
