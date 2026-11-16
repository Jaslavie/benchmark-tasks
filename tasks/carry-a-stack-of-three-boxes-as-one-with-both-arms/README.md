# Carry a stack of three boxes onto a shelf in one trip, squeezed between both arms

Three cardboard boxes stand in a stack, largest at the bottom. The bottom box is 200 x 120 mm, wider in both
directions than one gripper opens (94 mm), so the robot has to hold it between both arms, one pressing each side,
lift the whole stack, carry it 17 cm out, and set it on a 60 mm shelf. The boxes above ride on the bottom box;
none may be lifted off the one below it at any time, so taking the boxes over one by one scores 0.

The task is one file, `task.py`, for the `yam_arena` embodiment of yam-isaac-lab-arena (the program's YAM
embodiment for Isaac Lab-Arena, run through Inspect Robots), on the `station` scene. `near_misses.py` lists
near-successes and wrong outcomes with the score each must get; they are plain-Python checks of `score()` and
`success()` and need no simulator.

## Run a trial

From this folder on a machine set up by the yam-isaac-lab-arena launchable:

```bash
I="Move the stack of three boxes onto the shelf in one go, keeping them stacked the whole time: do not lift any box off the one below it."

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=$PWD/task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$I" --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/stack-dream

# Scripted demo (reaches 1.0 after about 16 s)
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=$PWD/task.py -E object_jitter=0 -E control_hz=20 -E cam_width=640 -E cam_height=360 \
  --instruction "$I" --scorer task_score --max-steps 600 --store-frames --log-dir ~/yam-eval/stack-demo
```

The run prints `task_score`. Scoring is `score()` and `success()` in `task.py`, described in words in `task.json`.

- **Runtime per trial:** at most 60 s of simulated time (1,200 steps at 20 Hz); about 1 to 2 minutes of wall time
  on an L40S once Isaac Sim has started (its first start takes several minutes).
- **Seeds:** none needed. Submitted runs use `-E object_jitter=0`, so every box starts exactly where `task.py`
  puts it. Jitter would shift the three boxes independently and start them off-centre on each other.
- **Trial history:** the scorer remembers whether the stack has been broken (an upper box more than 5 mm above the
  box below, more than 25 mm off its centre, or any box tilted past 15 degrees) and whether it has been lifted as
  one. A broken stack scores 0 for the rest of the trial. The record clears only on a scene reset: all three boxes
  within 0.5 mm and 1 degree of their starting poses and at rest, reached by a jump of more than 20 mm in one control
  step. Restacking the boxes at the start by hand does not clear it.
- **Grippers:** scoring sees only the boxes, so a stack resting on the shelf counts whether or not it is still held.
  Setting the stack down and regripping is allowed as long as it is never broken.

## Scripted demo

`demo()` drives both arms together with their fingers closed, turned so the fingers close front to back:

1. Each arm lowers its closed gripper beside one side of the bottom box, finger tips 10 mm above the table, and
   presses inward. The fingers widen toward the gripper body, so they press on the bottom box's top edges and stay
   clear of the narrower boxes above.
2. Both arms lift the stack, carry it out over the shelf, lower it until the bottom box is about 7 mm above the shelf
   top, and pull sideways off it so it drops onto the shelf.

Carrying the stack, both arms settle about 12 mm below their targets; the demo aims that much higher when it lowers
the stack. The scripted squeeze is a position target 30 mm past contact on each side; with less squeeze the stack
slipped out during the carry in testing.

## Checks without a simulator

```bash
python near_misses.py   # needs yam_arena importable; prints each case's score
```

The cases include: the stack centred on the shelf (1.0), held 1.5 mm above it (0), turned 10 degrees (1.0) and 20
degrees so a corner overhangs (0.5), an upper box 9 mm (1.0) and 12 mm (0.5) off centre, the stack lifted and set
down on the table 70 mm out (0.25) or 20 mm out (0), slid along the table without a lift (0), and the stack broken in
several ways and then rebuilt on the shelf (0): the top box lifted off (also when it is put back exactly at its start), boxes carried one at a time, the upper two
lifted together, the middle box slid 30 mm, the stack tipped 20 degrees.

## Scoring, exact definitions

This is the full rule that `score()` implements; the scoring field in task.json is a shorter summary of it.

Measured with a ruler and a photo, and judged over the whole trial. Every limit below is inclusive (a value exactly at a limit passes). A box is flat when its top face is within 5 degrees of level, either way up (a box turned upside down counts as flat, one on its side does not); at rest means moving less than 1 cm/s and turning less than 0.5 rad/s. A box rests on a surface when its lowest corner is within 1 mm of that surface (for an upper box, of the top of the box below); an upper box's offset is the horizontal distance between its centre and the centre of the box below it. The stack is broken, and the trial scores 0 whatever happens afterwards, if at any moment the middle box's lowest corner is more than 5 mm above the bottom box's top or the top box's lowest corner is more than 5 mm above the middle box's top, an offset exceeds 25 mm, or any box tilts more than 15 degrees (so carrying boxes one at a time, or lifting the upper boxes off and putting them back, scores 0). The stack is standing when all three boxes are flat and at rest, the middle box rests on the bottom box and the top box rests on the middle box. Score 1.0: the stack was never broken, the bottom box rests on the shelf top with all four of its bottom corners over the shelf top (no overhang), and the stack is standing with both offsets at most 10 mm. Score 0.5: the stack was never broken, the bottom box rests on the shelf top with its centre over the shelf top, and the stack is standing with both offsets at most 25 mm, but the bottom box overhangs an edge of the shelf or an offset is over 10 mm. Score 0.25: the stack was never broken, it was lifted as one at some moment in the trial (the bottom box's lowest corner at least 20 mm above the table), and it now stands on the table (bottom box resting on the table) with both offsets at most 25 mm and the bottom box's centre at least 30 mm from where it started. Score 0 in every other case: the stack left where it was or slid along the table without being lifted, held more than 1 mm above the shelf, any box on the table or knocked off, or the stack broken at any time. Scoring looks only at the boxes, so a stack resting fully on the shelf (or, for 0.25, on the table) counts whether or not the grippers are still holding it. Setting the stack down and taking a new grip is allowed, as long as the stack is never broken. The trial ends at 1.0 once the 1.0 condition has held for 1 s; otherwise it scores the condition at the last step. The record of the trial (broken, lifted) starts afresh only on a scene reset: all three boxes within 0.5 mm and 1 degree (any rotation) of their starting poses and at rest, having got there by a jump of more than 20 mm in one control step; restacking the boxes at the start by hand does not clear it.
