# Match the request and hand one metric screw

Choose the bin that matches the spoken M4 x 20 mm request, pick one screw, and place it alone in the tray.

## Environment

Brev launchable: Isaac Lab-Arena, Isaac Sim, L40S. Before every shell: `source ~/IsaacLab-Arena/yam.env`. On multi-GPU hosts pin with `export CUDA_VISIBLE_DEVICES=1` and `-E device=cuda:0`.

Place NVIDIA SimReady packages under `assets/simready/` (or keep the symlink to the shared packages).

## Offline checks

```bash
python tasks/match-request-hand-screw/match_request_hand_screw.py
```

## Full-score demo

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=tasks/match-request-hand-screw/match_request_hand_screw.py \
  -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "The person asks for an M4 x 20 mm Phillips pan-head machine screw. Choose the matching size from the three bins, place one screw alone in the tray, then let go." \
  --scorer task_score --max-steps 2400 --seed 0 --no-prompt --store-frames \
  --log-dir ~/yam-eval/match-request-hand-screw-demo
```

## Dream-1

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=tasks/match-request-hand-screw/match_request_hand_screw.py \
  -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "The person asks for an M4 x 20 mm Phillips pan-head machine screw. Choose the matching size from the three bins, place one screw alone in the tray, then let go." \
  --scorer task_score --max-steps 2400 --seed 0 --no-prompt --store-frames \
  --log-dir ~/yam-eval/match-request-hand-screw-dream
```
