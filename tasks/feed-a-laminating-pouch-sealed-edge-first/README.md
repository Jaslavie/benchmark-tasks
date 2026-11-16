# Laminate a flash card: feed its pouch into the laminator sealed edge first

A task file for the `yam_arena` embodiment ([yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena), Isaac Lab-Arena 0.3.1).

A warmed-up desktop pouch laminator lies in the middle of the table, its entry slot facing the right arm. In line with
the slot, to the robot's right, a loaded card pouch lies on a box of pouches, backwards: its open end toward the slot. The instruction is
only "Laminate the flash card." Pouches go into a laminator sealed edge first; fed open end first, the loose films catch
at the rollers and jam. The robot must find the sealed end (a narrow frosted seam with the card pushed against it; the
other end shows 13 mm of clear film), turn the pouch 180 degrees, feed it square into the 8 mm slot, and let go once the
rollers take it.

## Run a trial

Dream-1:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Laminate the flash card." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir runs/dream
```

Scripted demo (scores 1.0): the same command with `--policy yam_scripted`. To check the scorer, set `TASK_DEMO` to
`failure` (fed open end first: a jam, 0), `held` (still pinched after 2 cm of draw: a jam, 0), `skewed` (20 degrees off
square: 0.5), `turned` (turned but never fed: 0) or `nothing` (0).

The run prints `task_score`: 1.0 once `success()` (a square bite, sealed edge first, and the pouch out of the back of the
laminator) has held for 1 s; otherwise `score()` at the end of the trial, which follows the scoring in `task.json`.

## How the laminator works here

`task.py` registers a per-step helper (`scenekit.on_step`). The first step at which any part of the pouch reaches the
rollers' nip line, 2.5 cm inside the slot, is the bite; the helper records the pouch's heading then. After a square or
skewed bite it draws the pouch straight through at 10 mm/s by setting its pose every control step, and lets it drop
behind the laminator once all of it is past the back face. After an open-edge-first bite, or if a gripper is still
closed on the pouch once 2 cm has been drawn, it draws the pouch 2 cm at most and holds it there: a jam, which the
scorer latches until the next reset.

## Runtime and seeds

- Demo: 751 steps (37.6 s simulated), about 5 minutes including start-up.
- Dream-1: 2,400 steps (2 minutes simulated).
- `object_jitter=0` keeps the layout fixed, so no seed is needed. The scripted demo is open loop and deterministic.

## Sources

- All meshes are built in `assets.py` from boxes and cylinders (`usdkit.py`); textures are drawn by `make_textures.py`
  (our own artwork, no text or logos). No third-party assets.
- Sizes: an A4 office pouch laminator (about 34 x 11 x 7 cm, 9.25 in / 235 mm entry); a jumbo card laminating pouch,
  2-15/16 x 4-1/8 in (75 x 105 mm); a playing-card-size flash card, 2.5 x 3.5 in (63 x 88 mm); a box of 100 card
  pouches. Feeding rules from pouch laminator manuals and troubleshooting tables (sealed edge first; open end first,
  off-centre or crooked entry jams; let the rollers take the pouch).
