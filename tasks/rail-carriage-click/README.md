# Slide a linear-guide carriage onto its rail past a spring plunger

A grey T-rail lies along x on the table, its open end toward the arms and an orange end stop at the far end. An orange
spring plunger sits in the rail head 4 cm before the stop. A blue carriage with a grip post starts on a yellow mat in
front of the right arm. The robot must pick the carriage up by its post, fit it onto the open end of the rail, push it
along the rail over the plunger until it touches the end stop, and let go. The scene, scorer and scripted demo are in
`rail_carriage_click.py`.

Linear guide carriages on robot gantries, 3D printers and wafer handlers are slid onto their rails by hand during
assembly. A ball-plunger detent marks the seated position, and a carriage left short of it can drift. The task tests
three things together: fitting a part with 1 mm clearances that can only go on from the open end (its lips sit under
the rail head, so it cannot be dropped on from above), pushing it through a spring detent that resists, and releasing
it cleanly. It is a guided slide fit, not a placement.

## Note for reviewers

This builds the idea of TK-GBSVDR after its revision hold lapsed. The code and text were written from scratch. The
score is computed from the end state only: nothing is remembered, so a carriage that was seated and then lifted off
scores 0 (tested, below).

## Files

| File | What |
|---|---|
| `rail_carriage_click.py` | The yam_arena task file: objects, `score()` and the scripted demo |
| `partkit.py` | Rigid parts built from boxes and cylinders, with spring-loaded sliders (from the accepted `din-rail-mount` task) |
| `task.json` | The submission details |

## Run a trial

In the Isaac Lab-Arena environment (`source /data/IsaacLab-Arena/yam.env`), from the repository root:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=tasks/rail-carriage-click/rail_carriage_click.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Pick up the blue carriage by its post, fit it onto the open end of the grey rail, and push it along the rail past the orange plunger until it touches the end stop. Then let go and keep both fingers at least 3 cm from the post." \
  --scorer task_score --store-frames --max-steps 1200 --log-dir runs/dream
```

Scripted demo: the same flags with `--policy yam_scripted --max-steps 400`. The score is printed as `task_score`.
Wrong attempts for the shortcut checks: set `RAIL_VARIANT` to `nothing`, `half`, `hold`, `drop_on_top`, `elsewhere` or
`liftoff` before the scripted run. Render videos with `inspect-robots video <log>.json`.

## Scoring

Computed at the end of the episode from the last second of poses and, for the release, the arms' finger joints and pad
tips. Nothing earlier counts.

- 1.0: the carriage is on the rail (guide lips under the head: centre 1 to 5 mm above its resting height and within the
  rail's length, level within 3 degrees, lateral offset 3 mm or less); wholly past the plunger (its rear face at least
  1 mm beyond the plunger's back face) with the plunger back up behind it; its front face within 3 mm of the end stop;
  released (both fingers at least 90% open and both pad tips at least 3 cm from the post's axis, for the last 1 s); and
  still (moving less than 1 mm over the last 1 s).
- 0.5: on the rail at the end but not all of the above (short of the plunger, still held, or still moving).
- 0: not on the rail at the end, whatever happened earlier.

## Reach and layout

- One arm only. The carriage starts at x 26, y +2, the rail axis is at y -15 and the end stop face at x 41.5. Offline
  inverse kinematics: every waypoint of the demo is reached by the right arm (worst error 0.2 mm). The rail is out of
  reach of the left arm (the right arm reaches the rail up to about x 40 only at y -15 or nearer its base).
- Layout (cm, from the arm-base line and the midline): carriage start (26, +2); rail from x 22.0 to 41.5 at y -15;
  plunger back face x 37.5; end stop x 41.5 to 43.5.

## Shortcut and robustness checks (scripted)

| Variant | What it does | Score |
|---|---|---|
| Demo | Fits, pushes to the stop, lets go, parks | 1.0 |
| `nothing` | Arm parks, carriage untouched | 0 |
| `half` | Fits the carriage onto the rail, lets go short of the plunger | 0.5 |
| `hold` | Pushes it home but keeps gripping it to the end | 0.5 |
| `drop_on_top` | Lowers it onto the rail head from above at the seated position | 0 |
| `elsewhere` | Puts it down at the middle of the table | 0 |
| `liftoff` | Seats it and lets go, then regrips and lifts it | 0 |
| Dream-1 (1,200 steps) | One run | 0.0 |

The `liftoff` test is the reviewer fix: the earlier version kept partial credit after the carriage left the rail. Here
the lifted carriage rises above 5 mm over its resting height, is no longer on the rail, and the score drops from the
seated value to 0.

## Runtime and seeds

- Scripted demo: 400 steps (20 s simulated), about 75 s of wall time plus simulator start.
- Dream-1: 1,200 steps (1 minute simulated), about 3 minutes of wall time on one GPU.
- No randomness: run with `-E object_jitter=0`. No seeds.

## Results

Dream-1: 0.0 (the arm goes to the rail but never picks up the carriage). Demo: 1.0.

## Fidelity

The parts are rigid bodies built from boxes and one cylinder. The rail, end stop and plunger housing are one 6 kg part
that does not move. The plunger is a prismatic joint with a spring drive (0.2 N/mm, 4 mm travel). Its ramp rises 1 mm
over 1.2 cm to 1 mm above the rail head, so the push it needs is small; a real ball plunger is stiffer. Friction is
1.2 everywhere, clearances are 1 mm, contact offsets 0.5 mm. A carriage held by its cylindrical post can pitch about
the pinch axis, so any hard resistance at the plate tips it; the gentle ramp keeps the push inside what the arm can
deliver. No contact force cap is enforced.

## Real-world build

A 3D-printed or machined T-rail about 20 cm long, with a stop at one end and a spring plunger (1 mm proud of the rail
head) 4 cm before it, and a printed carriage with undercut lips and a 3 cm post, 80 g. Real carriages are far more
precise; here the 1 mm clearances stand in for that.

## Licence

The task is released under the MIT licence. It uses no library or third-party assets: the rail, end stop, plunger,
carriage and mats are built from plain boxes and a cylinder, and are released under the task licence. `partkit.py` is
reused from the accepted `din-rail-mount` task in this repository (same licence as the repository).
