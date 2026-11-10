# Hand the correct screw

Instruction: `The person asks for an M4 x 20 mm Phillips pan-head machine screw. Take one screw from the white bin and place it in the tray; then let go.`

Files: `task.json`, `hand_correct_screw.py`, `README.md`, `assets/simready/`

## Task design

Three short bins on the station table, fully coloured **yellow / white / green** for **M3 / M4 / M5**. Each has a floor and holds three steel Phillips pan-head screws:

| Bin | Colour | Screws | SimReady mesh | Mass |
|-----|--------|--------|---------------|------|
| M3 | yellow | M3 × 16 mm ×3 | `hardware_screw_m3_a02` | 1.6 g |
| M4 | white | M4 × 20 mm ×3 | `factory_bolt_m04_loose` | 3.1 g |
| M5 | green | M5 × 25 mm ×3 | `screw_high_res` | 5.4 g |

Bins are 22×18×9 cm open Box walls on the table floor so grippers fit and screws stay in. Screws use SimReady meshes at near-metric size. Handoff target is fixed SimReady `dishware_plastic_tray_a01`. A request card reads **M4 × 20 mm**. Place one M4 screw alone in the tray and release.

Placement randomization: free objects with mass use `-E object_jitter`. The tray has no mass, so it stays fixed. Open-loop `demo()` needs `-E object_jitter=0`.

## Scoring

End-of-trial score. Tiers do not stack; single best tier, else 0:

- **0**: still holding a screw, gripper within 3 cm, or any M3/M5 in the tray
- **0.25**: lifted an M4 by at least 3 cm without a clean handoff
- **0.5**: one M4 in the tray with another screw on or next to it
- **1.0**: one M4 alone in the tray, released, still 1 s; other eight still in start bins

## SimReady assets

NVIDIA [simready-assets](https://huggingface.co/datasets/nvidia/simready-assets) (CC-BY-4.0) under `tasks/hand-correct-screw/assets/simready/` (gitignored):

| Role | Asset | Entry |
|------|-------|-------|
| M4 | `factory_bolt_m04_loose` | `…/sm_factory_bolt_m04_loose_01.usda` |
| M3 | `hardware_screw_m3_a02` | `…/sm_hardware_screw_m3_a02_01.usda` |
| M5 | `screw_high_res` | `…/sm_screw_high_res_01.usda` |
| Tray | `dishware_plastic_tray_a01` | `…/sm_dishware_plastic_tray_a01_01.usd` |

## Environment

Brev launchable: Isaac Lab-Arena, Isaac Sim, L40S. Before every shell: `source ~/IsaacLab-Arena/yam.env`. On multi-GPU hosts pin with `export CUDA_VISIBLE_DEVICES=1` and `-E device=cuda:0`.

## Offline check, no GPU

```bash
python tasks/hand-correct-screw/hand_correct_screw.py
```

## Full-score demo

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=tasks/hand-correct-screw/hand_correct_screw.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "The person asks for an M4 x 20 mm Phillips pan-head machine screw. Take one screw from the white bin and place it in the tray; then let go." \
  --scorer task_score --max-steps 2400 --seed 0 --no-prompt --store-frames \
  --log-dir ~/yam-eval/hand-correct-screw-demo
```

## Limits and seeds

- Run limit: 2400 steps at 20 Hz, `--max-steps 2400`
- Fixed layout: `--seed 0` with `-E object_jitter=0`
