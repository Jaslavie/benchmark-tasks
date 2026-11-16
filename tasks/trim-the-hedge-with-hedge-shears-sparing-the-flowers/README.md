# Trim three overgrown shoots off a box hedge with hedge shears, without cutting the flowers

A cedar planter trough on the table holds a small clipped box hedge. Three overgrown green shoots stick up out of the
front of the hedge; behind them, in the back row, grow a pink and an orange flower. On a rubber mat in front of the
planter lie a pair of hedge shears, closed: two long steel blades on a pivot bolt, each blade one piece with a wooden
handle, the handles splaying back toward the arms. Both hands take the shears by their handles, lift them off the mat
and carry them up to the hedge's top, opening the blades. For each shoot the open blades go in round it just above
the hedge and close: the shoot's top is cut off and falls onto the hedge. Then the hands close the shears, lay them
back on the mat and both arms go back to where they started. The task is one task file,
[`trim_the_hedge_with_hedge_shears_sparing_the_flowers.py`](trim_the_hedge_with_hedge_shears_sparing_the_flowers.py),
for the `yam_arena` embodiment (scene, scoring and the scripted demo), plus [`bodykit.py`](bodykit.py), which it
imports from the same folder.

What it tests: working one two-handled tool with both hands together (carrying it level, opening and closing it by
moving the two handles against each other at the right moment), aiming the cut at what should go while missing what
should stay, then putting the tool back closed.

Why two hands: the pivot turns freely, so the blades open and close only when the two handles are moved against each
other; a hand on one handle can carry its own half but has nothing to close it against. With one hand (the right doing
its part of the demo, the left never moving) the demo scored 0: the left half swung open to its stop and hung from the
pivot, the shears drooped 20 degrees in the right hand and the blades never closed round a shoot. The left alone also
scored 0: the right half hung open at its stop and no shoot was ever between the blades as they closed.

## How it is modelled

`yam_arena` task files place single boxes, cylinders and spheres. The mat and the planter with its soil and hedge are
kinematic, fixed where they are put; each half of the shears is a loose rigid body made of shapes from `bodykit.py`,
the two joined by a `Hinge` about the pivot bolt (it turns freely against light friction, from 2 degrees past closed,
where the rubber bumpers meet, to 50 degrees open).

- The stems: below its cut height each shoot (and each flower) is drawn only; nothing collides with it. Above it, its
  top is a loose body held in place by a fixed joint (a `Glue` from `bodykit.py`, to a hidden fixed body under the
  table) and colliding with everything but the shears.
- The cut: after every control step the task file finds where each stem crosses the blades' plane and where that point
  lies against the blades (how far from the pivot, at what angle between the two blades). A stem between blades open
  at least 20 degrees, 2.5 to 15 cm from the pivot (and, for a shoot, within 1.8 cm of its cut height, 2 cm above the
  hedge), is in the blades; if the blades then close to 6 degrees open or less while it is still there, the stem is
  cut: the task file switches its joint off and gives the top a little push, so it tips over sideways onto the hedge.
- Lying closed on the mat: the pivot over the mat (at least 2 cm in from its edges), no more than 1.2 cm higher than
  when the shears lie flat, the blades within 10 degrees of level and open 6 degrees or less.

Importing `bodykit.py` hooks `yam_arena.scene` when the simulator starts, without changing yam_arena, gives `score()`
each body's own pose, and lets the task file act on the simulation after every control step.

In a real trial the cut is judged by observation, not by blade angles: a shoot counts once its top has come away
within 1.8 cm of its cut height, and a flower is cut if its stem is severed anywhere above the hedge. The simulation's
blade-angle rule (opened 20 degrees round the stem, closed to 6 degrees on it) only stands in for severing, which the
rigid stems cannot do by contact; a real shoot cut cleanly with a smaller blade movement counts the same.

## Run and score a trial

The scorer is `task_score`: 1.0 when `success()` (full score, which includes both arms back where they started, with
both halves of the shears still) has held for 1 s, otherwise `score()` at the end of the episode (0, 0.25, 0.5 or
1.0; see `task.json`). `-E object_jitter=0` keeps every object exactly where the task file puts it, so every trial
starts from the same layout. The cameras render at 1280 x 704 for sharp videos; Dream-1's client scales every frame to
320 x 176 before sending it, so `-E cam_width=320 -E cam_height=176` gives the policy the same input.

Dream-1 (needs a key in `~/.dreamlabs/credentials`):

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=trim_the_hedge_with_hedge_shears_sparing_the_flowers.py -E object_jitter=0 -E control_hz=20 -E cam_width=1280 -E cam_height=704 \
  --instruction "Trim the three tall green shoots off the box hedge with the hedge shears, cutting each just above the hedge, without cutting the flowers. Then leave the shears lying closed on the mat and take both arms back to where they started." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/shears-dream
```

The scripted demo, which reaches full score: both hands come straight down on the handles, fingers closing across
them, and lift the shears off the mat together. They carry them up to the hedge's top, opening the blades 40 degrees
on the way (each hand swinging its handle out round the pivot). For each shoot, left to right, the open blades go in
round it, the shoot 12 cm out along them, and close: the shoot's top is cut off and tips over onto the hedge; the
blades open again and come back out. Then the hands close the shears, carry them back over the mat, lay them down,
let go and go back to where they started:

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=trim_the_hedge_with_hedge_shears_sparing_the_flowers.py -E object_jitter=0 -E control_hz=20 -E cam_width=1280 -E cam_height=704 \
  --instruction "Trim the three tall green shoots off the box hedge with the hedge shears, cutting each just above the hedge, without cutting the flowers. Then leave the shears lying closed on the mat and take both arms back to where they started." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/shears-demo
```

Videos, one MP4 per camera next to each run's frames:

```bash
for log in ~/yam-eval/shears-dream/*.json; do inspect-robots video "$log"; done
```

Set `SHEARS_DEBUG=1` to print what the score is built from four times a second: where the pivot is, how far the
blades are open, how the shears are turned and tilted, which shoots and flowers are cut, where every stem in the
cutting part of the blades lies against them (distance from the pivot, angle, crossing height; a star when it is
between open blades), whether the shears lie on the mat, where each TCP is and where each shoot's top is.

## Runtime per trial

On one RTX 5090: about 7 minutes for a Dream-1 trial of 1,200 steps (396 s on the clock, with Dream answering in 0.55 s on average), and about 3 minutes for the scripted demo (690 steps, 34.5 s of simulated time, 198 s on the clock). Isaac Sim's first start on a new machine adds a few minutes while it compiles its shaders.

## Seeds

None needed: with `-E object_jitter=0` the layout is fixed, and the scripted demo follows fixed waypoints.
