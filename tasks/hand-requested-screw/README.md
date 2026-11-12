# Hand the requested metric screw

Choose the bin that matches the spoken M4 x 20 mm request, pick one screw, and place it alone in the tray.

Files: `task.json`, `hand_requested_screw.py`, `README.md`, `assets/simready/`

## SimReady assets

NVIDIA [simready-assets](https://huggingface.co/datasets/nvidia/simready-assets) (CC-BY-4.0) under `tasks/hand-requested-screw/assets/simready/` (not committed):

| Role | Asset | Entry |
|------|-------|-------|
| M4 | `factory_bolt_m04_loose` | `…/sm_factory_bolt_m04_loose_01.usda` |
| M3 | `hardware_screw_m3_a02` | `…/sm_hardware_screw_m3_a02_01.usda` |
| M5 | `screw_high_res` | `…/sm_screw_high_res_01.usda` |
| Tray | `dishware_plastic_tray_a01` | `…/sm_dishware_plastic_tray_a01_01.usd` |

## Environment

Brev launchable: Isaac Lab-Arena, Isaac Sim, L40S. Before every shell: `source ~/IsaacLab-Arena/yam.env`. On multi-GPU hosts pin with `export CUDA_VISIBLE_DEVICES=1` and `-E device=cuda:0`.

## Offline checks

```bash
python tasks/hand-requested-screw/hand_requested_screw.py
```

## Full-score demo

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=tasks/hand-requested-screw/hand_requested_screw.py \
  -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "The person asks for an M4 x 20 mm Phillips pan-head machine screw. Choose the matching size from the three bins, place one screw alone in the tray, then let go." \
  --scorer task_score --max-steps 2400 --seed 0 --no-prompt --store-frames \
  --log-dir ~/yam-eval/hand-requested-screw-demo
```

## Dream-1

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=tasks/hand-requested-screw/hand_requested_screw.py \
  -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "The person asks for an M4 x 20 mm Phillips pan-head machine screw. Choose the matching size from the three bins, place one screw alone in the tray, then let go." \
  --scorer task_score --max-steps 2400 --seed 0 --no-prompt --store-frames \
  --log-dir ~/yam-eval/hand-requested-screw-dream
```
