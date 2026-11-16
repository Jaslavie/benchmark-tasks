# Free the ice cubes: twist an ice-cube tray between both hands, then turn it over to tip the cubes into the dish

A light blue plastic ice-cube tray of eight frozen cubes hangs across the top of a glass dish by the tabs at its two
ends. Holding the tray by both tabs, the hands twist it, one end one way and the other end the other way, so the cubes
crack loose; then they lift it off the dish's rim, turn it over above the dish so the cubes fall in, turn it back, set
it back across the dish and go back to where they started. The task is one task file,
[`twist_an_ice_cube_tray_and_tip_the_cubes_into_a_dish.py`](twist_an_ice_cube_tray_and_tip_the_cubes_into_a_dish.py),
for the `yam_arena` embodiment (scene, scoring and the scripted demo), plus [`bodykit.py`](bodykit.py), which it
imports from the same folder.

What it tests: applying opposite twists with the two hands to one object; then moving the same object together with
both hands, turning it over and back without dropping anything outside the dish.

Why two hands: twisting the tray takes a turning force at each end in opposite directions. With one hand on one tab,
its other end either follows through the tray's stiffness or stays where the dish's lugs hold it, and neither cracks
the cubes loose: the task file counts a twist only when both ends are turned, opposite ways.
With one hand (the right doing its part of the demo, the left never moving) the demo scored 0: the whole tray turned with the right end, about 85 degrees, and the cubes stayed frozen in.

## How it is modelled

`yam_arena` task files place single boxes, cylinders and spheres. The dish is kinematic, fixed where it is put; the
tray and the cubes are loose rigid bodies made of shapes from `bodykit.py`.

- The tray: two rigid halves, each an end tab (what a hand holds) with its four compartments fixed to it (a `Glue` from
  `bodykit.py`), joined at the tabs by a hinge along the tray's length (a `Hinge`) with a spring (0.01 N m per degree)
  standing in for the plastic's stiffness in twist, up to 30 degrees either way. Small lugs on the dish's end walls,
  either side of each tab, keep the hanging tray centred.
- The ice: each cube is held in its compartment by a fixed joint to its half's tab (a `Glue`), so it stays in however
  the tray is turned. After every control step the task file reads how far the two tabs are turned against each other
  about the tray's length; at 12 degrees, with each end turned at least 5 degrees from level, the two opposite ways (so
  that the lugs, which can hold one resting end still, cannot stand in for the second hand), it switches every cube's
  joint off and the cubes lie loose in their compartments (slippery: friction 0.15), falling out once the tray is
  turned over.

Importing `bodykit.py` hooks `yam_arena.scene` when the simulator starts, without changing yam_arena, gives `score()`
each body's own pose, and lets the task file act on the simulation after every control step.

## Run and score a trial

The scorer is `task_score`: 1.0 when `success()` (full score, which includes both arms back where they started, with
the tray and every cube still) has held for 1 s, otherwise `score()` at the end of the episode (0, 0.25, 0.5 or 1.0;
see `task.json`). `-E object_jitter=0` keeps every object exactly where the task file puts it, so every trial starts
from the same layout. The cameras render at 1280 x 704 for sharp videos; Dream-1's client scales every frame to
320 x 176 before sending it, so `-E cam_width=320 -E cam_height=176` gives the policy the same input.

Dream-1 (needs a key in `~/.dreamlabs/credentials`):

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=twist_an_ice_cube_tray_and_tip_the_cubes_into_a_dish.py -E object_jitter=0 -E control_hz=20 -E cam_width=1280 -E cam_height=704 \
  --instruction "Get the ice cubes into the glass dish: hold the ice-cube tray by its two end tabs and twist it, one end one way and the other end the other way, so the frozen cubes crack loose; then lift it off the dish's rim, turn it over above the dish so all eight cubes fall in, turn it back and set it back across the dish, and take both arms back to where they started." \
  --scorer task_score --max-steps 800 --store-frames --log-dir ~/yam-eval/ice-dream
```

The scripted demo, which reaches full score: each hand comes down on one end tab from above, level and pointing along
the tray, fingers closing across the tab. They twist the tray, the left end's top toward the far side and the right
end's toward the arms, 12 degrees each, and back; lift it 3 cm off the dish's rim, turn it together 115 degrees about
its length so the cubes fall into the dish, hold it there, turn it back, set it back across the dish, let go and go back
to where they started:

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=twist_an_ice_cube_tray_and_tip_the_cubes_into_a_dish.py -E object_jitter=0 -E control_hz=20 -E cam_width=1280 -E cam_height=704 \
  --instruction "Get the ice cubes into the glass dish: hold the ice-cube tray by its two end tabs and twist it, one end one way and the other end the other way, so the frozen cubes crack loose; then lift it off the dish's rim, turn it over above the dish so all eight cubes fall in, turn it back and set it back across the dish, and take both arms back to where they started." \
  --scorer task_score --max-steps 800 --store-frames --log-dir ~/yam-eval/ice-demo
```

Videos, one MP4 per camera next to each run's frames:

```bash
for log in ~/yam-eval/ice-dream/*.json; do inspect-robots video "$log"; done
```

Set `ICE_DEBUG=1` to print what the score is built from twice a second: how far each tab is turned about the tray's
length and the twist between them (now and the most so far), whether the cubes are loose, how many cubes are in the
dish and the lowest cube's height, where each tab is and whether both are back on the rim, and where each TCP is.

## Runtime per trial

On one RTX 5090: about 2 minutes for a Dream-1 trial of 800 steps (113 s on the clock, with Dream answering in 0.89 s on average), and about 3 minutes for the scripted demo (395 steps, 19.8 s of simulated time, 209 s on the clock). Isaac Sim's first start on a new machine adds a few minutes while it compiles its shaders.

## Seeds

None needed: with `-E object_jitter=0` the layout is fixed, and the scripted demo follows fixed waypoints.
