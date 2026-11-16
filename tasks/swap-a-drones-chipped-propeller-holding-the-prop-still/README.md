# Replace a drone's chipped propeller with the new one and tighten its prop nut

A quadcopter drone stands on its skids on the bench, its rear motor nearest the arms. That motor's orange propeller has
a chipped blade. The propeller sits on the motor's bell, centred by its 24 mm hole on a 23 mm boss, and is clamped down by a tall red prop
nut on the motor's threaded shaft. The motor is switched off and spins freely, so turning the nut alone just turns the
nut, the propeller and the motor together. A new propeller of the same kind, printed side up, lies on a stand to the
right; a blue bin for worn parts stands in front of the drone and a small steel parts dish to the right.

The instruction gives the goal: the new propeller on the motor, printed side up, its nut tight, the chipped one in the
bin and both arms back. Done as in the demo: hold the chipped propeller still by a blade with one hand while the other
unscrews the nut (anticlockwise) and lifts it off; put the chipped propeller in the blue bin; set the new propeller on
the boss, printed side up, hold it still by a blade and screw the nut back on (clockwise) until it is tight; then take
both arms back to where they started. The task is one task file,
[`swap_a_drones_chipped_propeller_holding_the_prop_still.py`](swap_a_drones_chipped_propeller_holding_the_prop_still.py),
for the `yam_arena` embodiment (scene, scoring and the scripted demo), plus [`bodykit.py`](bodykit.py), which it
imports from the same folder.

**What it tests:**
- holding a part still against a turning load while the other hand turns a fastener on it;
- a fastener that needs several strokes, letting go and re-gripping between them;
- taking a part off a shaft and fitting a new one in its place;
- putting things where they belong (the nut aside, the worn part in the bin) and finishing with both hands free.

**Why two hands:** the motor is off and spins freely on its bearings. Turned alone, the nut drags the motor and
the propeller round with it (thread friction and friction under the nut, against almost nothing at the bearings), so it
never unscrews and never tightens. Held still by a blade, the propeller, and through its clamp the motor, stays put
while the nut turns on the thread. The one-hand check scored 0: the right hand did its part as in the demo while the
left never moved. The motor and propeller spun round with each stroke, the nut turned only a tenth of a turn against
the motor and never came off.

## How it is modelled

`yam_arena` task files place single boxes, cylinders and spheres. The airframe (body, arms, skids and the three
motors not being serviced, with their propellers and nuts), the bin, the stand and the dish are kinematic bodies made
of shapes from `bodykit.py`. The rear motor's bell, both propellers and the nut are rigid bodies.

- **The motor:** the bell turns on a hinge (a `Hinge` from `bodykit.py`) about the motor's axis, with a light drag
  (1e-4 N m per degree/s) for its bearings. The propeller is a loose part on its boss (a 24 mm hole on a 23 mm boss, 0.5 mm of play; the boss's top edge is
  chamfered as a lead-in): only friction under its hub
  and under the nut makes it turn the bell.
- **The thread:** the nut sits on the shaft on a `Thread` joint from `bodykit.py` that keeps it on the shaft's axis
  and lets it slide along it, pulled by a spring (8,000 N/m, at most 4 N) toward a target. Turning on it is resisted
  by 0.15 N m s/rad of thread friction, which drags the free motor round with the nut. After every control step
  the task file measures how far the nut has turned against the bell and moves the target 13.5 mm per turn (a coarse
  quick-release thread): anticlockwise lifts it, clockwise lowers it. Turned on past seating, the target goes below the seat, so
  the nut presses the hub down (4 N at most): the score counts the nut tight once it has turned at least 16 degrees past
  where it first seated (0.6 mm of target). This is the turn a paint line across nut and shaft shows in a real trial,
  where the nut must also hold 0.5 N m on a torque screwdriver; the simulation does not model the torque itself. Turned anticlockwise until its bottom is 8 mm above the hub's
  top, the nut comes off the thread (the joint is switched off) and can be lifted away. Brought back over the shaft
  (within 8 mm of its axis, upright within 10 degrees, its bottom less than 7 mm above the hub's top), it catches the
  thread again. A nut just unscrewed must leave the shaft before it can catch again.
- The nut never collides with the bell or the shaft (it slides along it on the joint); it does collide with the hub,
  the fingers, the dish and the table.

Importing `bodykit.py` hooks `yam_arena.scene` when the simulator starts, without changing yam_arena. It gives
`score()` each body's own pose and lets the task file act on the simulation after every control step.

## Run and score a trial

The scorer is `task_score`. It gives 1.0 when `success()` has held for 1 s; `success()` means full score (which
includes both arms back where they started) with the new propeller and the nut still. Otherwise it gives `score()` at
the end of the episode: 0, 0.25, 0.5 or 1.0 (see `task.json`).

- `-E object_jitter=0` keeps every object exactly where the task file puts it, so every trial starts from the same
  layout.
- The cameras render at 1280 x 704 for sharp videos. Dream-1's client scales every frame to 320 x 176 before sending
  it, so `-E cam_width=320 -E cam_height=176` gives the policy the same input.

Dream-1 (needs a key in `~/.dreamlabs/credentials`):

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=swap_a_drones_chipped_propeller_holding_the_prop_still.py -E object_jitter=0 -E control_hz=20 -E cam_width=1280 -E cam_height=704 \
  --instruction "Replace the drone's chipped propeller with the new one from the stand, printed side up, with its nut tight; put the chipped propeller in the blue bin, then take both arms back to where they started." \
  --scorer task_score --max-steps 1300 --store-frames --log-dir ~/yam-eval/prop-dream
```

The scripted demo reaches full score:
1. The left hand comes in from its side along the chipped propeller's left blade, tilted 20 degrees down, and pinches
   the blade between its fingers.
2. The right hand comes straight down on the nut, fingers across its flats, and unscrews it in three strokes of a
   third of a turn anticlockwise: turn, let go, turn back with the fingers wide open, take the nut again, following
   it up the thread. The nut comes off the thread; the right lifts it and sets it in the steel dish.
3. The left lets go and draws back. The right takes the chipped propeller by its hub, lifts it off the shaft and drops
   it into the blue bin.
4. The right takes the new propeller off its stand by its hub, carries it over the motor and sets it on the boss;
   the left pinches its left blade.
5. The right takes the nut from the dish, sets it over the shaft (it catches the thread) and screws it down in three
   strokes clockwise until it is tight.
6. Both hands let go and both arms go back to where they started.

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=swap_a_drones_chipped_propeller_holding_the_prop_still.py -E object_jitter=0 -E control_hz=20 -E cam_width=1280 -E cam_height=704 \
  --instruction "Replace the drone's chipped propeller with the new one from the stand, printed side up, with its nut tight; put the chipped propeller in the blue bin, then take both arms back to where they started." \
  --scorer task_score --max-steps 1300 --store-frames --log-dir ~/yam-eval/prop-demo
```

Videos, one MP4 per camera next to each run's frames:

```bash
for log in ~/yam-eval/prop-dream/*.json; do inspect-robots video "$log"; done
```

Set `PROP_DEBUG=1` to print what the score is built from twice a second:
- whether the nut is on the thread, its height above its seat, the thread's target and how far it has turned;
- whether it has come off, how far it is off the shaft's axis and tilted, and the bell's turn;
- where the nut and both TCPs are;
- where each propeller is, its turn and its tilt.

## Runtime per trial

On one RTX 5090: about 5 minutes for a Dream-1 trial of 1,300 steps (296 s on the clock, with Dream answering in 0.84 s on average), and about 3 minutes for the scripted demo (1,239 steps, 62.0 s of simulated time, 208 s on the clock). Isaac Sim's first start on a new machine adds a few minutes while it compiles its shaders.

## Seeds

None needed: with `-E object_jitter=0` the layout is fixed, and the scripted demo follows fixed waypoints.
