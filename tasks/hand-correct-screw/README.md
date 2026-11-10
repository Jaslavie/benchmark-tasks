# Hand the correct screw

Instruction given to the policy: `The person asks for an M4 x 20 mm Phillips pan-head machine screw. Take one screw from the bin whose front label reads M4 and place it in their open palm; then let go.`

Files in this folder: `task.json`, `task.py` (scene, success, partial credit and the scripted demo), `README.md`.

## Task design

Three open bins in a row on the station table, front labels **M3 / M4 / M5** (left to right from behind the arms). Each bin holds **three** steel Phillips pan-head machine screws:

| Bin label | Screws (×3) | Shank | Head | Mass each |
|-----------|-------------|-------|------|-----------|
| M3 | M3 × 16 mm | Ø 3 mm × 16 mm | Ø 5.5 mm × 2 mm | 1.6 g |
| M4 | M4 × 20 mm | Ø 4 mm × 20 mm | Ø 7 mm × 2.5 mm | 3.1 g |
| M5 | M5 × 25 mm | Ø 5 mm × 25 mm | Ø 8.5 mm × 3 mm | 5.4 g |

A fixed manikin forearm and open palm sit on the robot's left; a request card reads **M4 × 20 mm**. The job is to read the request, open the M4 bin, hand over one M4 screw, and let go.

## How scoring works

Judged at the end of the trial (after the arms stop). You get **one** of these scores — they do not add together:

1. **0 — failed**
   - You are still holding a screw (either gripper within 3 cm of any screw), **or**
   - Any screw from the M3 or M5 bin is on the person's palm (wrong size), **or**
   - Nothing useful happened (no lift, no handoff).

2. **0.25 — you started the job**
   - At some point you lifted an M4 screw at least 3 cm off the table, but you never completed a clean handoff.

3. **0.5 — right screw, messy handoff**
   - Exactly one M4 is on the palm, but another screw is also on or right next to the palm (or forearm).

4. **1.0 — success**
   - Exactly one M4 × 20 mm screw lies alone on the open palm,
   - you have let go of it,
   - it stays still for 1 second, **and**
   - the other eight screws are still in the bins they started in.

A screw “on the palm” means its centre is inside the 9 cm palm disc (top view), its lowest point is within 4 mm of the palm’s top face, and it is at rest.

| Score | Short condition |
|-------|-----------------|
| 1.0 | One M4 alone on the palm, released, still 1 s; other eight in home bins |
| 0.5 | One M4 on the palm, but another screw is also near the palm |
| 0.25 | An M4 was lifted ≥ 3 cm, but no clean handoff |
| 0 | Wrong size on palm, still gripping a screw, or nothing scored |

## SimReady asset references

The submitted scene is built from `yam_arena` primitives (`Box` / `Cylinder`) so it runs on the program stack without downloading meshes. For a higher-fidelity Isaac Sim stage, or a real-lab twin, use these [NVIDIA SimReady Assets](https://huggingface.co/datasets/nvidia/simready-assets) (CC-BY-4.0) as visual / physics stand-ins:

| Role in this task | Suggested SimReady asset | USD entry point |
|-------------------|--------------------------|-----------------|
| M4-class fastener | `factory_bolt_m04_loose` (also `factory_bolt_m04_tight`) | `factory_bolt_m04_loose/simready_usd/sm_factory_bolt_m04_loose_01.usd` |
| Generic machine screw | `hardware_screw_95_a01`, `hardware_screw_99_physics_a01`, `screw_high_res` | `…/simready_usd/sm_<name>_01.usd` |
| M4 nut (optional clutter) | `factory_nut_m4_loose` / `factory_nut_m4_tight` | same pattern |
| Open bin / tray | `bin_waste_a01`–`a04`, `bin_waste_b01`, or small box assets such as `airSystem_air_filter_box_a01` | same pattern |

Browse the tree: https://huggingface.co/datasets/nvidia/simready-assets/tree/main

Credit: NVIDIA SimReady Assets, CC-BY-4.0. In this task file, screws are rigid boxes sized to each screw's overall envelope (threads and Phillips recesses not modeled); bin “labels” are coloured front plates standing in for printed M3/M4/M5 text (the scorer keys off start bin and diameter, not OCR).

## Environment

Built for the program's Brev launchable (Isaac Lab-Arena, Isaac Sim, L40S). Before every shell: `source ~/IsaacLab-Arena/yam.env`. Dream key in `~/.dreamlabs/credentials`.

## Offline check, no GPU

From the root of this repository, with the yam-arena package importable:

```bash
python tasks/hand-correct-screw/task.py
```

Asserts the scoring rules on hand-built states (start 0; one M4 alone on the palm 1.0; wrong size on the palm 0; two screws on the palm 0.5; lifted but not placed 0.25) and replays each arm's demo plan through the kinematics at 20 Hz.

## Run and score one Dream-1 trial

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=tasks/hand-correct-screw/task.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "The person asks for an M4 x 20 mm Phillips pan-head machine screw. Take one screw from the bin whose front label reads M4 and place it in their open palm; then let go." \
  --scorer task_score --max-steps 2400 --seed 0 --no-prompt --store-frames \
  --log-dir ~/yam-eval/hand-correct-screw-dream
```

## Run and score the full-score demo

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=tasks/hand-correct-screw/task.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "The person asks for an M4 x 20 mm Phillips pan-head machine screw. Take one screw from the bin whose front label reads M4 and place it in their open palm; then let go." \
  --scorer task_score --max-steps 2400 --seed 0 --no-prompt --store-frames \
  --log-dir ~/yam-eval/hand-correct-screw-demo
```

The demo is `demo()` in `task.py`: the left arm picks the middle M4 screw (`screw_m4_1`) and places it on the palm centre; the right arm stays clear. Use `-E object_jitter=0`.

## Run limit, runtime and seeds

- Run limit: 2,400 control steps at 20 Hz (120 s), `--max-steps 2400`.
- Seeds: `--seed 0` with `-E object_jitter=0` for a fixed layout (required for the open-loop demo).

## Reset between trials

Return every screw to its labeled bin (three M3, three M4, three M5), long axis along x, heads toward +x. Check the three bin labels face the robot and the request card still shows M4 × 20 mm. Nothing is consumed.
