# Get a heavy carton up onto a low platform

A 3 kg carton stands on the table in front of a 2 cm platform (a step, dock lip or plinth). It has a round handle
bar along the top of its front edge. The arms on that handle can raise its front edge but cannot lift its back off
the table, and pushed straight at the platform it jams against the 2 cm face. The way up is the movers' one: set
the front edge on the step, let go, then push the back until the carton's weight passes the step's edge, so it
tips level onto the platform by itself, and push it fully on, keeping it upright throughout.

The task is `task.py` for the `yam_arena` embodiment of yam-isaac-lab-arena (the program's YAM embodiment for Isaac
Lab-Arena, run through Inspect Robots), on the `station` scene, with `parts.py` (the carton and platform built from
several shapes) and `lots.py` (layout lots) next to it. `near_misses.py` lists near-successes and wrong outcomes with
the score each must get; it runs without a simulator.

## Run a trial

From this folder, on a machine set up by the yam-isaac-lab-arena launchable:

```bash
I="Move the carton up onto the platform, keeping it upright."

# Dream-1, both lots (trial k uses lot k mod 2; the score is the mean)
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=$PWD/task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$I" --scorer task_score --max-steps 900 --epochs 2 --store-frames

# Scripted demo, both lots
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=$PWD/task.py -E object_jitter=0 -E control_hz=20 -E cam_width=640 -E cam_height=360 \
  --instruction "$I" --scorer task_score --max-steps 900 --epochs 2 --store-frames
```

Put `TASK_LOT=A` or `B` in front to run every trial in one lot. Lot B starts the carton 5 cm to the left and 3 cm
further back; without `--epochs 2` only lot A runs. There is no choice in this task for a habit to win; the lots
only stop a replay of one fixed motion.

- **Runtime per trial:** at most 45 s of simulated time (900 steps at 20 Hz); about 1-2 minutes of wall time per
  trial on an L40S once Isaac Sim has started.
- **Seeds:** none needed. Run with `-E object_jitter=0`.

## Scoring details

`score()` and `success()` read the carton's pose (its eight body corners from its true size and orientation):

- **1.0, flat on the platform:** the four bottom corners within 1 mm of the platform's top and inside its outline,
  the top within 2 degrees of level.
- **0.75, level but overhanging the near edge:** the carton has tipped level onto the platform (four bottom corners
  within 1 mm of its top height, top within 2 degrees of level, its underside centre past the step's edge) but has
  not been pushed fully on: both back bottom corners are short of the step's edge. Front and sides inside the outline.
- **0.5, resting across the step:** both front bottom corners past the step's edge, inside the platform's outline
  sideways, and no lower than 1 mm under the platform's top (the carton's underside bears on the step's edge or the
  platform), both back corners on the table. That state is the method's first step and does not arise from shoving it at the step (it jams against the
  face) or lifting it whole (the arms cannot).
- **Tipped:** the carton's top tilted more than 20 degrees from level at any step of the trial gives 0 for the rest
  of the trial; the record is cleared at every reset. The method needs about 9 degrees.
- **Let go:** both arms' `gripper`, `tip_left` and `tip_right` collision hulls more than 5 mm from points on the
  carton's body and handle, measured every step. **Still:** under 1 cm/s and 0.5 rad/s. Success holds for 1 s.

## Measured on the GPU (L40S, stock stack)

- **Edge lift, not a whole lift (3 kg):** both arms on the handle commanded 50 mm up raised the front bottom edge
  27 mm with the back on the table; commanded 80 mm up and 4 cm back, the front rose to 68 mm and the carton tilted
  25 degrees about its back edge, which never left the table. At 1.5 kg the arms lifted the whole carton, so the
  mass is set at 3 kg.
- **Straight push (current 3 kg, 0.45/0.30 friction; trace, lot A, `~/runs/cartonfix-straight-push/
  trace-20261108T061115`, a scratch copy of this task with only `demo()` replaced):** both hands pushing its back
  3 cm up for 9 s slid the carton 4 cm to the step, where it stopped flat against the 2 cm face (front at 38.0 cm,
  level, score 0). As the fingers then rose behind it, they caught its back and it went over onto its front face
  against the platform (90 degrees, score 0).
- **Demo (trace, lot A):** the front edge rose 28 mm; set across the step, the carton rested on the step's edge with
  its front 4-6 mm above the platform (0.5 while it sat there, let go); pushed by its back it rode up to 15.7 degrees
  of tilt over the edge, then tipped level onto the platform once its centre passed the edge, and was pushed on
  (1.0). Traces before the friction was set to coated board on steel showed the two arms could not push the
  3 kg carton at the 1.2 friction used then.

- **Final runs (both lots):** demo 1.0 in each lot; Dream-1 0 in each lot (its grippers moved over the carton and its
  handle, but the carton stayed on the table).

## Scripted demo

`demo()` uses both arms with the grippers straight down. They grip the handle 19 cm apart (closer, the elbows
collide), raise the front edge (commanded 50 mm, as the loaded arms lag), carry it 75 mm forward over the step, set
it down and let go, rise clear of the handle, come down behind the carton with closed fingers, push its back low
until it tips level onto the platform, then push higher until its back is 3 cm past the step's edge, and move away.
The carton's start is the one the cameras show for the trial's lot.
