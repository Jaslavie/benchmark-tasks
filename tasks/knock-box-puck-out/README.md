# Knock a spent espresso puck out of the portafilter into the knock box

Instruction given to the policy: `Empty the used coffee out of the portafilter into the knock box, then put the portafilter back on its stand.`

Files: `task.json`, `knock_box_puck_out.py` (scene, `success()`, partial-credit `score()`, the scripted `demo()` and an offline
self-test), `mechkit.py` (helper module imported by the task file: the puck's breakable bond to the basket;
it patches yam_arena in memory at import, nothing on disk changes; written for this task set, no third-party code),
`README.md`.

## Environment

The program's Brev launchable (Isaac Lab-Arena 0.3.1, Isaac Sim, L40S) with yam-arena and inspect-robots as it
installs them. Before every shell: `source ~/IsaacLab-Arena/yam.env`; the Dream key is in `~/.dreamlabs/credentials`.

## Offline check, no GPU

From the root of this repository, with yam-arena importable:

```bash
python tasks/knock-box-puck-out/knock_box_puck_out.py
```

It asserts the scoring rules on hand-built states and replays every arm's demo plan through the kinematics at 20 Hz
(every joint inside the URDF limits, IK error under 0.5 mm). Expected output:

```
scoring: 16 cases pass
right: 28.7 s plan, worst IK 0.090 mm / 0.019 deg, largest joint step 10.3 deg
```

## Run and score one Dream-1 trial

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=tasks/knock-box-puck-out/knock_box_puck_out.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Empty the used coffee out of the portafilter into the knock box, then put the portafilter back on its stand." \
  --scorer task_score --max-steps 1200 --seed 0 --no-prompt --store-frames --log-dir ~/yam-eval/knock-box-puck-out-dream
```

`task_score` is 1.0 when `success()` has held for 1 s (the episode ends with `termination_reason="success"`),
otherwise `score()` at the last step. The CLI prints `task_score: <value>`.

## Run and score the full-score demo

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=tasks/knock-box-puck-out/knock_box_puck_out.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Empty the used coffee out of the portafilter into the knock box, then put the portafilter back on its stand." \
  --scorer task_score --max-steps 1200 --seed 0 --no-prompt --store-frames --log-dir ~/yam-eval/knock-box-puck-out-demo
```

The demo is `demo()` in the task file: open loop and deterministic, 29 s of plan (about 774 steps
including the success hold), inside the run limit of 1,200 steps.

## Videos and screenshot

```bash
for log in ~/yam-eval/knock-box-puck-out-*/*.json; do inspect-robots video "$log"; done
ffmpeg -y -i ~/yam-eval/knock-box-puck-out-demo/frames/<run>/scene-0-e0_top_cam.mp4 -frames:v 1 setup.png
```

## Run limit, runtime and seeds

- Run limit: 1,200 steps (60 seconds at 20 Hz).
- One fixed layout: `-E object_jitter=0`, so the seed does not change the start. Dream-1 is hosted and not seeded;
  the demo is deterministic.
- Runtime per trial on an L40S: roughly 1 minute of wall clock per 400 steps for Dream-1, plus Isaac Sim warm-up on
  the first run.

## Reset between trials

Put a puck back on the basket floor and re-bond it (on a real rig: a printed puck held by an O-ring friction fit, or a fresh spent puck), set the portafilter basket-up on its holder and empty the knock box; the holder and the box do not move.
