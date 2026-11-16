# Rinse a glass on a bar's push-down glass rinser, then leave it to drain

Instruction given to the policy: `Rinse the glass on the glass rinser, then leave it to drain on the drip mat.`

Files: `task.json`, `bar_glass_rinser.py` (scene, `success()`, partial-credit `score()`, the scripted `demo()` and an
offline self-test), `mechkit.py` (multi-box rigid parts, the sprung prismatic joint and per-episode hooks),
`README.md`.

## What is modelled

- The rinse plate is a free 60 g body on a vertical prismatic joint with a 300 N/m spring preloaded 3.6 N against
  its upper stop, so the 180 g glass resting on it does not open the valve; pushing about 3 N more takes it past
  the 4 mm valve point. The plate stops 9 mm down.
- Rinse time accrues only while the valve is open with the glass rim-down, centred within 1.2 cm and seated on the
  plate. The valve open with the plate uncovered (pushed by a finger, or the glass off centre) sprays the bar and
  latches a score of 0, as does knocking the glass over.
- Score = 0.6 x min(1, rinse time / 1.5 s) + 0.4 for the glass left rim-down, wholly on the drip mat, let go and
  still.

## Environment

Isaac Lab-Arena 0.3.1 with yam-arena and inspect-robots.

## Offline check, no GPU

```bash
python tasks/bar-glass-rinser/bar_glass_rinser.py
```

It asserts the scoring rules on hand-built states (rinse timer and both latches) and replays the demo plan through
the kinematics at 20 Hz. Expected output:

```
scoring: 14 cases pass
right: 26.1 s plan, worst IK 0.089 mm / 0.013 deg
```

## Run and score one trial

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=tasks/bar-glass-rinser/bar_glass_rinser.py \
  -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Rinse the glass on the glass rinser, then leave it to drain on the drip mat." \
  --scorer task_score --max-steps 900 --seed 0 --no-prompt --store-frames --log-dir runs/bar-glass-rinser
```

For Dream-1 use `--policy dream` with the same arguments.

## Results (seed 0, no jitter)

| Run | task_score |
| --- | --- |
| Scripted demo (pick, press 2.5 s, set on the mat) | 1.0 |
| Shortcut: carry the glass straight to the mat, no rinse | 0.4 |
| Shortcut: push the plate down with the fingers first, then the full demo | 0 |
| Dream-1 | 0 |
