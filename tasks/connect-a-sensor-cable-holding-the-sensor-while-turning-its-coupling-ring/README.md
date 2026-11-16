# Connect a robot's sensor cable: hold the loose proximity sensor still while turning the M12 coupling ring tight

A robot cell's pick station stands on the table: an aluminium fixture plate with a part in its nest, the cell's robot
arm parked behind it, and, aimed down at the part from an angled bracket, a new inductive proximity sensor (an M30
steel barrel) that has just been fitted. Its jam nuts are not tightened yet, so it turns freely in its bracket, and a
red protective cap covers the M12 connector at its back end. The robot's sensor cable comes out of the cable duct and
ends in an M12 connector with a knurled coupling ring, parked on a parking receptacle beside the sensor. One hand pulls
the cap off the sensor's connector and drops it into the parts tray, takes the cable's connector off its parking
receptacle and pushes it onto the sensor's connector. The other hand holds the sensor's barrel still while the first
turns the coupling ring, in strokes, until it is tight and the sensor's green power light comes on. Then both hands let
go and both arms go back to where they started. The task is one task file,
[`connect_a_sensor_cable_holding_the_sensor_while_turning_its_coupling_ring.py`](connect_a_sensor_cable_holding_the_sensor_while_turning_its_coupling_ring.py),
for the `yam_arena` embodiment (scene, scoring and the scripted demo), plus [`bodykit.py`](bodykit.py), which it
imports from the same folder.

What it tests: steps in the right order (the cap has to come off before the connector can go on), lining a connector
up with an angled thread and pushing it home, turning a ring in repeated strokes (let go, turn back, take it again),
and holding a loose part still against the drag of the hand that works on it.

Why two hands: the coupling ring's thread drags the loose sensor round with it, so turning the ring of a connector on
a sensor nobody holds just turns the sensor, and the connector never tightens. With one hand (the right doing its part
of the demo, the left never moving) the demo scored 0.25: the cap came off and the connector went on and caught its
thread, but through four strokes of the ring the sensor turned with it, and the ring never got more than 0.2 of a turn
onto the sensor (tight takes one).

## How it is modelled

`yam_arena` task files place single boxes, cylinders and spheres. The fixture plate, the bracket, the parking
receptacle, the cable duct, the parts tray and the robot arm are kinematic, fixed where they are put; the sensor, the
cap, the cable's connector, the three cable segments and the part are loose rigid bodies made of shapes from
`bodykit.py`.

- The sensor turns in its bracket on a `Hinge` about its axis (the loose jam nuts: light friction).
- The cap and the connector are hollow at the front (their walls are rings of boxes), so they sit over the sensor's
  connector, or the parking receptacle's dummy one, held there by gravity; the connector cannot reach the sensor's
  connector while the cap is still on it.
- The cable is three stiff segments joined to each other, to the duct's mouth and to the connector by damped ball
  joints (`bodykit.ball`).
- The thread: after every control step the task file checks where the connector's front is against the sensor's back
  end. Within 4 mm of it along the axis, 3 mm of the axis and 15 degrees of it, the thread has caught: a joint (a
  `Thread` from `bodykit.py`) switches on and keeps the connector on the axis, free to turn. From then on the task file
  counts how far the connector turns against the sensor: clockwise seen from behind screws it on; one full turn makes
  it tight, and the green light (a lit ring placed over the dark one) comes on; turned back 0.6 turns, it comes off.
  While it is not yet tight, the ring drags the sensor round with it: the task file moves the hinge's drive target with
  every turn of the ring and switches on a spring toward it (at most 5 degrees behind, 0.03 N m), so a sensor nobody
  holds turns with the ring, and only a sensor held still lets the ring screw on.

Importing `bodykit.py` hooks `yam_arena.scene` when the simulator starts, without changing yam_arena, gives `score()`
each body's own pose, and lets the task file act on the simulation after every control step.

## Run and score a trial

The scorer is `task_score`: 1.0 when `success()` (full score, which includes both arms back where they started, with
the connector, the sensor and the cap still) has held for 1 s, otherwise `score()` at the end of the episode (0, 0.25,
0.5 or 1.0; see `task.json`). `-E object_jitter=0` keeps every object exactly where the task file puts it, so every
trial starts from the same layout. The cameras render at 1280 x 704 for sharp videos; Dream-1's client scales every
frame to 320 x 176 before sending it, so `-E cam_width=320 -E cam_height=176` gives the policy the same input.

Dream-1 (needs a key in `~/.dreamlabs/credentials`):

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=connect_a_sensor_cable_holding_the_sensor_while_turning_its_coupling_ring.py -E object_jitter=0 -E control_hz=20 -E cam_width=1280 -E cam_height=704 \
  --instruction "Connect the robot's sensor cable to the new proximity sensor: pull the red protective cap off the sensor's connector and drop it in the blue parts tray; take the cable's connector off its parking holder and push it onto the sensor's connector; the sensor turns in its bracket because its nuts are not tightened yet, so hold the sensor still with one hand while the other turns the connector's knurled coupling ring until it is tight and the sensor's green light comes on. Then let go of both and take both arms back to where they started." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/m12-dream
```

The scripted demo, which reaches full score: the right hand, pointing along the sensor's axis with its fingers closing
across it, slides its open fingers round the red cap, takes it, pulls it off and drops it into the parts tray. It comes
down round the parked connector's coupling ring, takes it, pulls it off the parking receptacle, carries it across and
pushes it onto the sensor's connector. The left hand, pointing in along the barrel from its sensing end and tilted a
little down, slides its open fingers round the barrel below the bracket and holds it still. The right turns the ring
clockwise (seen from behind) in four strokes of 135 degrees, letting go and turning back open between strokes; after
the third it is tight and the green light comes on. Both hands let go, draw back and go back to where they started:

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=connect_a_sensor_cable_holding_the_sensor_while_turning_its_coupling_ring.py -E object_jitter=0 -E control_hz=20 -E cam_width=1280 -E cam_height=704 \
  --instruction "Connect the robot's sensor cable to the new proximity sensor: pull the red protective cap off the sensor's connector and drop it in the blue parts tray; take the cable's connector off its parking holder and push it onto the sensor's connector; the sensor turns in its bracket because its nuts are not tightened yet, so hold the sensor still with one hand while the other turns the connector's knurled coupling ring until it is tight and the sensor's green light comes on. Then let go of both and take both arms back to where they started." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/m12-demo
```

Videos, one MP4 per camera next to each run's frames:

```bash
for log in ~/yam-eval/m12-dream/*.json; do inspect-robots video "$log"; done
```

Set `M12_DEBUG=1` to print what the score is built from twice a second: whether the thread has caught, where the
connector's front is against the sensor's back end (along and off the axis, tilt), how many turns it has been screwed
on and whether it is tight, how far the connector and the sensor have turned and the sensor's drive target, where the
cap is, and where each TCP is from the coupling ring and from where the barrel is held.

## Runtime per trial

On one RTX 5090: about 5 minutes for a Dream-1 trial of 1,200 steps (329 s on the clock, with Dream answering in 0.91 s on average), and about 4 minutes for the scripted demo (659 steps, 33.0 s of simulated time, 223 s on the clock). Isaac Sim's first start on a new machine adds a few minutes while it compiles its shaders.

## Seeds

None needed: with `-E object_jitter=0` the layout is fixed, and the scripted demo follows fixed waypoints.
