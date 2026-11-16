# Lean a tall canister in over the shelf fence under a low shelf and stand it up

A shelf unit has a lower shelf with a 35 mm fence along its front edge and an upper shelf whose underside is 170 mm
above the lower shelf, starting 40 mm behind the fence. A 150 mm tall canister with a flat carry handle moulded along the front edge of its top stands
on the table in front. Lifted upright over the fence it would need 185 mm of height, so it has to be leaned, its
base passed over the fence, and stood up behind it. The robot is asked to stand the canister upright on the lower shelf,
behind the fence.

The task is `task.py` for the `yam_arena` embodiment of yam-isaac-lab-arena (the program's YAM embodiment for Isaac
Lab-Arena), on the `station` scene. It needs `parts.py` (multi-shape parts), `lots.py` (layout lots) and `tilt.py`
(tilted grips for the scripted demo), all in this folder; they import only NumPy and `yam_arena`. `near_misses.py`
lists near-successes and wrong outcomes with the score each must get; they are plain-Python checks of `score()` and
`success()` and need no simulator.

## Lots

Lot A (the screenshot): the canister 24 cm in front of the arm bases and 16 cm left of the midline. Lot B: 20 cm in
front and 5 cm left. The shelf does not move. Trial k uses lot k mod 2, lot A first, so a run needs `--epochs 2` (or a
multiple); `TASK_LOT=A` or `TASK_LOT=B` pins every trial to one lot. Replaying one lot's motions picks at empty table in
the other. The lots stop a single replayed layout, not a memorised replay of both.

## Run a trial

From this folder on a machine set up by the yam-isaac-lab-arena launchable:

```bash
I="Stand the canister upright on the lower shelf, behind the fence."

# Dream-1, both lots
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=$PWD/task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$I" --scorer task_score --max-steps 1200 --epochs 2 --store-frames --log-dir ~/yam-eval/canister-dream

# Scripted demo, both lots (each reaches 1.0 in about 14 s)
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=$PWD/task.py -E object_jitter=0 -E control_hz=20 -E cam_width=640 -E cam_height=360 \
  --instruction "$I" --scorer task_score --max-steps 600 --epochs 2 --store-frames --log-dir ~/yam-eval/canister-demo
```

The run prints `task_score`, the mean over the trials. Scoring is `score()` and `success()` in `task.py`, described
in words in `task.json`.

- **Runtime per trial:** at most 60 s of simulated time (1,200 steps at 20 Hz), about 1 to 2 minutes of wall time on
  an L40S once Isaac Sim has started; a scored run is two trials.
- **Seeds:** none needed. Run with `-E object_jitter=0`.

## Scoring details the simulator applies

- The canister's four base corners are computed from its pose. "Placed" means every corner is at or behind the
  fence's inner face, in front of the back wall and between the side walls, and the lowest corner is within 2 mm of
  the lower shelf's top. Tilt is the angle of the canister's axis from vertical.
- 1.0: placed, tilt at most 5 degrees. 0.5: placed, tilt over 5 and at most 30 degrees, and leaning back
  toward the robot (the canister's axis tips toward -x, within 45 degrees of straight back), which is how it rests on
  the fence. Each also needs the canister still (under 1 cm/s and 0.5 rad/s, `ObjectState.resting()`) and let go.
  Success (1.0) must hold 3 s; the 0.5 tier is read from the state at the end of the trial.
- **Let go:** a `parts.on_step` callback reads both arms' finger and gripper link poses every step and measures the
  shortest distance from their collision hulls (yam_arena.analysis) to the canister's body and handle; it must exceed
  5 mm. Offline (`near_misses.py`) there is no robot, so this check counts as clear.

## Scripted demo

`demo()` drives the left arm only, from the start spot of the trial's lot (which the cameras show); it reads nothing
from the simulator. It grips the handle from above with the pads closing front to back across its 12 mm thickness
(pads told to close 3 mm past each face), so the pads hold the canister's lean. It lifts, leans the grip 18 degrees
(the canister takes about 15) with the base leading toward the shelf, passes the base over the fence about 13 mm
above it, sets the canister's near bottom edge down 14 mm behind the fence, and stands it up about that edge in
three steps while still holding the handle, which stays in front of the upper shelf's edge. It opens to 20 mm, rises
30 mm, backs away 80 mm, then moves 16 cm back and 6 cm up, well clear of the canister, and waits. The lean uses `tilt.py`, which affects only the scripted demo, never Dream-1 or scoring.
The waypoints are aimed 15 mm further in and 12 mm higher while the arm carries the canister, because the loaded
arm lags its target by about that much (measured in the first trace).

## Measured in simulation

- **First trace, no lag compensation:** the base was dragged along the fence's top and the canister stood on the
  fence (score 0 in both lots).
- **Demo, lot A** (trace): the canister leaned 14.7 degrees in the pads for 18 asked, passed 13 mm over the fence,
  was stood up behind it and scored 1.0 at 13.75 s.
- **Released while leaning 22 degrees** (a separate check, not the demo): it settled leaning back on the fence at
  17 degrees and scored 0.5, so the 0.5 tier is a state that holds after release.
- **Upright carry:** carrying the canister straight down at the height needed to clear the fence is past the arm's
  straight-down reach here (55 mm IK error offline), and in the trace the canister swung over in the grip. The
  shelf rules it out anyway: upright, the canister's top would be 15 mm or more above the upper shelf's underside
  while its base passes over the fence.

## Other ways to succeed, and what scores 0

- **Also full score:** either arm, any grip, pushing it upright after a release, any spot on the lower shelf behind
  the fence.
- **0.5:** the base behind the fence, the canister left leaning back on the fence (up to 30 degrees). Leaning
  sideways or forward scores 0.
- **Score 0:** lying down anywhere; standing on the fence or straddling it; on the table; still held; still moving.

The cases in `near_misses.py` cover the fence line (on it, 1 mm over), the back and side walls, 2 mm and 3 mm above
the shelf, 5 and 6 degrees, 30 and 31 degrees, sideways and forward tilts, leaning on the fence, lying down, on the fence top,
on the table and a moving canister.

## Recorded runs

- Scripted demo, `--epochs 2`: 1.0 in lot A and in lot B (task_score 1.0).
- Dream-1, 1,200 steps, `--epochs 2`: 0.0 in both lots.
