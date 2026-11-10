# Hand the correct screw

Instruction: `The person asks for an M4 x 20 mm Phillips pan-head machine screw. Take one screw from the bin whose front label reads M4 and place it in their open palm; then let go.`

Files: `task.json`, `hand_correct_screw.py`, `README.md`

## Task design

Three open bins on the station table, front labels **M3 / M4 / M5** left to right from behind the arms. Each bin holds three steel Phillips pan-head screws:

| Bin | Screws | Shank | Head | Mass |
|-----|--------|-------|------|------|
| M3 | M3 × 16 mm ×3 | Ø 3 mm × 16 mm | Ø 5.5 mm × 2 mm | 1.6 g |
| M4 | M4 × 20 mm ×3 | Ø 4 mm × 20 mm | Ø 7 mm × 2.5 mm | 3.1 g |
| M5 | M5 × 25 mm ×3 | Ø 5 mm × 25 mm | Ø 8.5 mm × 3 mm | 5.4 g |

A fixed manikin forearm and open palm sit on the robot's left. A request card reads **M4 × 20 mm**. Place one M4 screw alone in the palm and release.

## Scoring

End-of-trial score. Tiers do not stack; single best tier, else 0:

- **0**: still holding a screw, gripper within 3 cm, or any M3/M5 on the palm
- **0.25**: lifted an M4 by at least 3 cm without a clean handoff
- **0.5**: one M4 on the palm with another screw on or next to the palm
- **1.0**: one M4 alone on the palm, released, still 1 s; other eight still in start bins

On the palm: screw centre inside the 9 cm palm disc, lowest point within 4 mm of the palm top, at rest.

## SimReady assets

Scene uses `yam_arena` `Box` / `Cylinder` primitives. For higher-fidelity stages, see [NVIDIA SimReady Assets](https://huggingface.co/datasets/nvidia/simready-assets), CC-BY-4.0:

| Role | Asset | USD |
|------|-------|-----|
| M4 fastener | `factory_bolt_m04_loose` | `factory_bolt_m04_loose/simready_usd/sm_factory_bolt_m04_loose_01.usd` |
| Generic screw | `hardware_screw_95_a01`, `hardware_screw_99_physics_a01`, `screw_high_res` | `…/simready_usd/sm_<name>_01.usd` |
| M4 nut clutter | `factory_nut_m4_loose`, `factory_nut_m4_tight` | same pattern |
| Bin / tray | `bin_waste_a01`–`a04`, `bin_waste_b01`, `airSystem_air_filter_box_a01` | same pattern |

Tree: https://huggingface.co/datasets/nvidia/simready-assets/tree/main

Screws here are rigid boxes sized to the envelope. Bin labels are coloured front plates. Scorer keys off start bin and diameter, not OCR.

## Environment

Brev launchable: Isaac Lab-Arena, Isaac Sim, L40S. Before every shell: `source ~/IsaacLab-Arena/yam.env`. Dream key: `~/.dreamlabs/credentials`.

## Offline check, no GPU

```bash
python tasks/hand-correct-screw/hand_correct_screw.py
```

Prints a JSON report: scoring cases, plan duration, worst IK per arm.

## Dream-1 trial

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=tasks/hand-correct-screw/hand_correct_screw.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "The person asks for an M4 x 20 mm Phillips pan-head machine screw. Take one screw from the bin whose front label reads M4 and place it in their open palm; then let go." \
  --scorer task_score --max-steps 2400 --seed 0 --no-prompt --store-frames \
  --log-dir ~/yam-eval/hand-correct-screw-dream
```

## Full-score demo

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=tasks/hand-correct-screw/hand_correct_screw.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "The person asks for an M4 x 20 mm Phillips pan-head machine screw. Take one screw from the bin whose front label reads M4 and place it in their open palm; then let go." \
  --scorer task_score --max-steps 2400 --seed 0 --no-prompt --store-frames \
  --log-dir ~/yam-eval/hand-correct-screw-demo
```

`demo()` in `hand_correct_screw.py`: left arm places `screw_m4_1` on the palm centre; right stays clear. Requires `-E object_jitter=0`.

## Limits and seeds

- Run limit: 2400 steps at 20 Hz, 120 s, `--max-steps 2400`
- Fixed layout: `--seed 0` with `-E object_jitter=0`

## Reset between trials

Return every screw to its labeled bin, three each of M3, M4, M5. Long axis along x, heads toward +x. Bin labels face the robot. Request card shows M4 × 20 mm. Nothing is consumed.
