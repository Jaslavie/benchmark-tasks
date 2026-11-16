# Fit an O-ring onto a hydraulic plug: stretch it over the thread by no more than 50%, seat it square and untwisted

Fit an O-ring into its groove on a hydraulic O-ring boss (ORB) plug: put two fingertips into the ring, stretch it
just enough to pass over the thread, carry it down square to the groove at the hex, and draw the fingers out so it
closes into the groove. A ring stretched past 50% is scrap; one pushed down the thread unstretched rolls and is left
twisted; one let go tilted lands with a side on the thread until that side is pressed down into the groove.

Category: Impact on jobs. Industrial machinery mechanics, machinery maintenance workers and millwrights held about
547,300 US jobs in 2025 (BLS Occupational Outlook Handbook); repairing plant hydraulics, including fitting new
O-rings on ORB fittings and plugs, is a regular part of that work.

## Run

From the repository root, on a yam_arena install (Isaac Lab-Arena, inspect-robots):

    # scripted demo (scores 1.0)
    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=tasks/o-ring-into-groove/task.py \
      -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
      --instruction "Fit the O-ring into the groove at the base of the plug's thread." \
      --scorer task_score --max-steps 1500 --log-dir ~/yam-eval
    # Dream-1: the same with --policy dream --max-steps 2400

Offline scorer tests (no simulator): `python test_task.py` prints "all pass".

- Runtime: the demo finishes in 248 control steps (12.4 s at 20 Hz); a 2,400-step run takes about 10 minutes of wall time on
  one L40S. No seeds: the layout is fixed (`object_jitter=0`); the demo is deterministic.
- Files: `task.py` (scene, scorer, demo), `guardkit.py` (trimmed to `touching()`, the contact check for 0.75),
  `ringkit.py` (the ring-on-a-shaft model; this copy adds `FingerTips`, a
  gripper's two fingertips stretching an elastomer ring, and rolling when it is pushed instead of stretched),
  `yamkit.py` (modeled props, mechanisms and gripper states). The `FingerTips` copy also lets a fingertip press the
  high side of a ring that landed tilted down into its groove (see "At the groove" below). Both kits hook yam_arena's scene builder in memory;
  nothing on disk changes.

## Spec

Frame: cm from the midpoint between the two arm bases, x forward, y to the robot's left, table top z = 0.

| Object | Description | Position |
|---|---|---|
| Holder | blue anodized aluminum block, 9 x 9 x 4 cm, fixed | centered (34, -10) |
| Plug | zinc-plated steel SAE J1926 -24 O-ring boss hex plug, standing hex-down: 2.25 inch hex (57.15 mm across flats) | axis (34, -10), hex z 4.0 to 5.4 |
| Groove | the undercut between hex and thread, 3.6 mm long, 44.2 mm diameter, lower wall the hex face | z 5.4 to 5.76 |
| Thread | 1.875-12 UN, 47.6 mm across the crests, flat end face with a 1.5 mm chamfer | z 5.76 to 7.66 |
| O-ring | AS568-924, bright blue 90-durometer fluorocarbon (FKM), 43.7 mm inside diameter, 3.0 mm section | flat and centered on the end face, z 7.66 to 7.96 |

The ring's stretch is its inner circumference: the loop round the two fingertips' outer faces and, once both
fingers are outside the thread, round the thread as well. With the YAM fingertips (4 mm thick, 14 mm wide) a pad
gap of 57 mm (4.7 mm clear of the thread each side) stretches it 28.7% (56.2 mm); with the ring round the thread the
50% limit (65.5 mm) is reached at a pad gap of 73.7 mm, and fully open (94 mm) is 77%. Above the end face, with the
thread not inside the loop, the same 50% takes a pad gap of 80.9 mm. The sim reads the loop itself, so both apply;
a real trial reads the log the same way: over 73.7 mm with the ring round the thread, or over 80.9 mm above it.

The sim models one way of stretching the ring: one gripper's own two fingertips go into it and the gripper opens,
spreading them apart like a pair of pliers, so the stretch follows that gripper's pad gap. A two-hand stretch (a
fingertip of each gripper in the ring, the arms pulling apart) is outside the model: the sim does not stretch the
ring for it. A real trial may still see one, and no single gripper's opening shows it, so it is measured another
way: with the sizing cone (the ring's largest passed diameter), or from both arms' logged fingertip positions (the
loop round the two fingertips, and the thread if it is inside it, worked out as above; over 65.5 mm, 50%, scores 0).

Two parts of the ring model are stand-ins. The ring is tied to the plug's axis: it stays centered on the plug and
moves only along it and turns about it, as if the two fingertips kept it centered, where a real ring could slip off
one fingertip and hang crooked. A fingertip pressing a cocked ring's high side down 1.5 mm seats it by a rule (the
press law below), a stand-in for a real fingertip press; whether a real ring seats that way is checked, not assumed.
A real trial scores the ring itself on the top photo at the end: in the groove all round with no side on the thread,
and its parting line straight round the outside (untwisted).

Scoring (`score()` in task.py, evaluated at the end of the run). The score is the ring's final condition, after two
damage checks.

| Outcome | Score |
|---|---|
| Ring ever stretched past 65.5 mm inside diameter (50%) | 0, irreversible |
| More than 90 degrees of roll left in its section (pushed down the thread unstretched), wherever it is | 0 |
| Seated: in the groove all round (square, no side on the thread), closed onto its 44.2 mm bottom, 90 degrees of roll or less, with no fingertip in it and no gripper touching it, however it got there | 1.0 |
| Seated as for 1.0, but a gripper still touching the ring (its own shape within 2 mm of it) | 0.75 |
| At the groove (its middle within the 3.6 mm groove) but not seated: one side left on the thread, or the fingertips still in it | 0.5 |
| Stretched over the thread and still at least 5 mm down it at the end, not at the groove | 0.25 |
| Anything else, including doing nothing | 0 |

Where the arms end is not scored. Possible scores: 0, 0.25, 0.5, 0.75, 1.0.

Why each rule, and how a real trial checks it:

- Over-stretch, 65.5 mm (50%). Machinery's Handbook advises against stretching an O-ring's inside diameter more than
  50% to install it; a ring stretched further takes a permanent set, its section thins, and it no longer fills the
  groove, so the joint weeps. The sim models the stretch as the loop round the fingertips (and the thread), as above;
  73.7 mm of pad gap is the limit. A real trial measures a one-gripper stretch the same way, from the gripper's logged
  opening (the YAM reports its pad gap every step): a pad gap over 73.7 mm while the ring is round the thread (80.9 mm while it is still
  above the end face) is over 50%, and scores 0. That log is the main measure, since it sees a brief stretch that the ring may not keep. The secondary check is
  the ring itself: after the trial it is rolled off with a plastic pick and dropped on an O-ring sizing cone marked in
  0.5 mm diameters, beside an unused ring from the same bag. A ring that settles at a larger diameter than the new one
  has taken a set; a stretch just past 50% may not show at that resolution, which is why the log comes first. A
  two-hand stretch, which the sim does not model, is read from both arms' logged fingertip positions or the cone
  (see above).
- Twist, 90 degrees. A ring pushed along the thread without being stretched clear of it rolls 38 degrees per mm (its
  3.0 mm section rolling without slipping). After a quarter turn (2.4 mm of rolling) the mold parting line, which runs
  round the outside of an untwisted ring, has moved onto its face. A twisted ring can spiral in the groove and leak, so a
  fitter replaces it. The sim models the roll: only a fingertip arriving on the ring's near face rolls it, a ring
  already in the groove is held by the groove's wall, and a ring stretched clear of the thread again springs back
  untwisted. A real trial checks the parting line by eye and on the top photo: straight round the outside, not
  spiraling onto the face.
- Seated, 1.0: in the groove all round, closed onto its 44.2 mm bottom (1.2% installed stretch), no side on the
  thread, untwisted (90 degrees of roll or less, as above), no fingertip left in it, since a ring still held open
  on the fingers has not closed into the groove and cannot seal, and no gripper touching it (see 0.75). Only the
  ring as it ends is judged, not how it got there: let go square, stretched again and let go square, or pressed
  down into the groove after landing tilted all score 1.0. A real trial checks it on the top photo (the ring visible
  in the groove all round, its parting line straight, the grippers clear of it) and with a fingertip run round the
  ring after the trial.
- Seated but still touched, 0.75: the ring is seated just as for 1.0, but a gripper still touches it at the end. The
  fit is right but not handed off: a hand still on the ring can still roll it out of the groove. So it is worth less
  than a ring let go and left alone, and more than a ring at the groove that is not yet seated (0.5). In the sim
  touching is contact, not nearness: the gripper's own shape (palm, fingers and pads at their present opening) within
  2 mm of the ring, the solver's resting gap plus the fit of the shapes to the gripper's meshes (guardkit
  `touching()`). A hand hovering 3 mm over the ring, or down beside it, does not count. A real trial checks the last
  top and wrist camera frames for any part of the gripper resting on the ring.
- At the groove, 0.5: the ring is undamaged and in place but still on the fingertips or with one side on the thread.
  Closing it or pushing the high side down is all that is left, so it is worth half. In the sim a ring let go more
  than 4 degrees off square (the fingertips' height difference over their spacing, taken when the first fingertip
  leaves it) lands that way: across its 50 mm the tilt lifts the far side 3.5 mm, so its 3.0 mm section spans 3.5 to
  6.5 mm above the hex face, all above the 3.6 mm groove. That side then sits on the thread just above the groove, where it can be cut
  when the plug is screwed into its port. A fitter fixes it by pressing the high side down
  with a fingertip, and so can the robot: a fingertip end that comes down onto the high side (over its section,
  within 60 degrees of its highest point) and pushes it 1.5 mm, half its 3.0 mm section, takes it over the last crest,
  and the ring's own tension pulls it into the groove square. By the twist model that press rolls it 57 degrees
  (1.5 mm), under the 90-degree limit, so the ring ends seated and scores 1.0. A shorter press lets it spring back
  onto the thread, and pressing the side already in the groove does nothing. Stretching it again and letting it go
  square also seats it. The press is a stand-in law (see the spec above); a real trial reads the final ring on the top
  photo and by fingertip, not the release angle or the press.
- On the thread, 0.25: stretched over the thread and, at the end, at least 5 mm down it (where it is then, not the
  deepest it reached, so a ring worked back up toward the end face loses it). Its middle is then 3.5 mm past the
  1.5 mm chamfer, over more than a full 2.1 mm thread pitch, so it is clearly started rather than resting on the end
  face. The ring is undamaged and a quarter of the way through the job. A real trial reads it against the mm scale
  on the holder.
- Doing nothing, or hovering over or touching the ring while it still lies on the end face, scores 0.

0.2 mm diametral clearance lets the stretched ring slide over the thread.

## Real-world build

All parts are off the shelf: a -24 SAE J1926 (1.875-12 UN) O-ring boss hex plug, AS568-924 90-durometer FKM
O-rings in blue (as sold for ORB fittings; the color shows the ring and its parting line on camera), and a 9 x 9 x 4.2 cm aluminum block with a 2.25 inch hex pocket 2 mm deep, so the hex face is 5.4 cm up as in the sim (or a
bench vise with soft jaws holding the plug's hex so the hex face is 5.4 cm above the table). The robot's own
fingertips stretch the ring, so no tool is needed; add a printed mm scale on the holder in view of the top camera and
an O-ring sizing cone marked in 0.5 mm diameters for the stretch check.

A real trial is checked on the log, the top photo and the ring. Stretch: the gripper's logged pad gap, over 73.7 mm
with the ring round the thread (80.9 mm above the end face) failing the run, or for a two-hand stretch the loop round both arms' logged fingertip
positions over 65.5 mm; then, as a secondary check, the ring is rolled off and dropped on the
sizing cone next to an unused ring from the same bag (a larger settled diameter means a permanent set). Seating, on
the top photo at the end: the ring visible in the groove all round with the grippers clear of it (1.0), the same with
any part of a gripper still resting on it in the last top or wrist frames (0.75), at the groove with a side on the thread or
still on the fingertips (0.5), or on the thread at least 5 mm below the end face against the mm scale (0.25), then
confirmed with a fingertip run round it. Twist: the parting line straight round the outside on the top photo, not
spiraling onto the face.

Reset: roll the ring out of the groove with a plastic O-ring pick, check it on the sizing cone against a new ring and
its parting line, replace it if either is off, and lay it flat and centered on the plug's end face.
