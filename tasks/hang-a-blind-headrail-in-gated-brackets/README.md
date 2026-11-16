# Hang a blind headrail in two gated brackets

A white blind headrail lies on two blocks on the table. At the back is a half-scale window: a pale blue glass pane in
front of a grey wall, with a dark box bracket at each end. Each bracket has an orange hinged gate across its front, shut,
with a white grip tab on its free end. The robot must swing both gates open, lift the headrail with both hands (it has
a blue grip tab for each), carry it level onto the two bracket floors, slide it back until it bears on the wall, and
swing both gates shut over it. A gate shut first blocks the rail, and a rail left about 1 cm or more off the wall stops its
gates short of shut. Pushing the rail along the table cannot work: the floors are 6 cm above the blocks.

## Files

- `hang_a_blind_headrail_in_gated_brackets.py`: the task file for the `yam_arena` embodiment (scene, `score()`,
  `success()`, the closed-loop `demo_script`).
- `mechkit.py`, `belimits.py`, `closedloop.py`: shared helpers for multi-part objects, geometry checks and the
  closed-loop scripted demo, kept beside the task file.
- `task.json`: the task's details.

## Run a trial

On an instance set up by the simulation walkthrough, from this folder:

```bash
INSTR="Hang the white blind headrail in the two window brackets."

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=hang_a_blind_headrail_in_gated_brackets.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$INSTR" --scorer task_score \
  --max-steps 3600 --store-frames --log-dir ~/yam-eval/headrail-dream

# Scripted demo (full score): both arms open the gates, carry the rail in by its tabs, slide it to the wall,
# then push the gates shut and park
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=hang_a_blind_headrail_in_gated_brackets.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$INSTR" --scorer task_score \
  --max-steps 3600 --store-frames --log-dir ~/yam-eval/headrail-demo
```

- **Seed and layout:** `-E object_jitter=0` is required: it keeps every object exactly where the task file puts it.
- **Runtime per trial:** the demo takes about 50 s of simulated time (about 1,000 steps); a Dream-1 trial runs up to
  3,600 steps (3 minutes simulated). Wall-clock time on one L40S, including Isaac Sim start-up: about 220 s for the demo and about 550 s for a full Dream-1 trial.
- **Score:** `task_score` reports 1.0 when `success()` holds for 8 s (rail home, gates shut, rail still), otherwise
  `score()` at the end of the run. The long hold lets the scripted demo open its grippers and park before the run ends.

## Scoring

At the end of the run (grippers opened in a real trial):

1. **Rail home** means: underside on both bracket floors (within 0.4 cm of the floor tops), back face within
   0.5 cm of the wall at both ends, tilted 2 degrees or less, each end 3 cm or more inside its bracket. A rail whose
   underside is more than 1 cm above the floors (still held, perched on a lip, left on the blocks) is not home.
2. **Gates shut** means both gates within 0.1 rad of shut: each gate plate in line with its bracket front (top photo).
3. **1.0** rail home and both gates shut. **0.5** rail home, a gate open. **0.25** both gates were swung open 1.2 rad
   or more at the same time at any point in the run, rail not home. Otherwise 0.

Doing nothing scores 0. Tape-measure checks: rail back face to the wall at each end; rail underside against the
floors; top photo for the gates. In simulation the checks use the rail's actual pose, so a skewed or tilted rail is
judged by its real ends.

## Real-world reset

Shut both gates, then lay the headrail on the two blocks at its marked spot, square to the table edge. Keep fingers
clear of the gate hinges when closing.
