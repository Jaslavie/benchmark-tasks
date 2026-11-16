# Hang washed wine glasses upside down in an overhead bar rack

Behind a bar, washed wine glasses are hung bowl-down from an overhead rack so they drain and stay clear of dust.
The rack is a channel with two lips, and a glass can only go in through its open end: the foot is wider than the gap
between the lips, so the stem has to enter the 25 mm gap at the end and run along it, the foot riding just above the
lips, before the glass is lowered to hang by its foot. The only place to hold a glass is its 9 mm stem, and the rack
is overhead: the lips are 22 cm up, so the gripping hand has to stay under them, coming in nearly level, while it
carries the foot over them. Two glasses stand bowl-down on a bar mat, as they come out of the glass washer.

Instruction: *Hang both wine glasses in the overhead rack.*

Files: `task.py` (the task file: objects, `success`, `distance`, `score`, `demo`), `robocurve_ext.py` (shipped in the
folder, no `yam_arena` edits; builds the rack and posts as compound rigid bodies of box pieces and each glass from
`glass.obj`), `glass.obj` (the glass: foot, stem and bowl as three convex parts), `task.json`, `test_task.py` (offline
checks).

## What is on the table

Positions in metres from the midpoint between the arm bases (x out, y left), heights above the table.

- **Overhead glass rack** (fixed, brass): a single channel along y at x 0.44, from its open end at y -0.20 to a closed
  end at y +0.10 (300 mm, room for four glasses). Two lips with a 25 mm channel between them, 89 mm opening, 38 mm
  tall; the lips' top is 0.220 above the table. Two square brass posts, 20 mm across and 254 mm tall, at its far side (x 0.50) hold it up.
- **Bar mat** (fixed, black rubber): 200 x 120 x 6 mm at (0.485, -0.30).
- **Glass A** (loose, 180 g) stands bowl-down on the mat at (0.44, -0.30); **glass B** behind it at (0.53, -0.30).
- Each glass: a 75 mm foot, 3 mm thick; a 9 mm stem, 80 mm long; a 95 mm bowl (74 mm rim, 84 mm at its widest);
  178 mm tall. Hanging, its foot lies on both lips and its rim is 42 mm above the table.

Rack sizes: single-channel bar glass racks have a 1 in (25 mm) channel, a 3 1/2 in (89 mm) opening and are 1 1/2 in
(38 mm) tall (KegWorks "Glass hanger rack, brass, 24 in"); they take glasses with bases up to 3 to 3.25 in, 76 to
83 mm (WebstaurantStore: Regency 16 in rack 460GHRG16, American Metalcraft GR1435). The glass sizes and mass are
those of a typical 300 ml wine glass (assumed).

## Run and score a trial

Built against the YAM Isaac Lab-Arena embodiment at commit `bd62771`. Copy this folder to `~/hang-stemware-rack` on
the GPU machine. Always run with `-E object_jitter=0`.

```bash
export INSTR="Hang both wine glasses in the overhead rack."

# a policy (here Dream-1), 900 steps = 45 seconds at 20 Hz
inspect-robots run --embodiment yam_arena --policy dream -E task=$HOME/hang-stemware-rack/task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 --instruction "$INSTR" --scorer task_score --max-steps 900 --store-frames --log-dir ~/yam-eval/hang-stemware-rack-dream

# the scripted demo
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=$HOME/hang-stemware-rack/task.py -E object_jitter=0 -E control_hz=20 --instruction "$INSTR" --scorer task_score --max-steps 900 --store-frames --log-dir ~/yam-eval/hang-stemware-rack-demo

# one MP4 per camera, in each log's frames/ folder
for log in ~/yam-eval/hang-stemware-rack-*/*.json; do inspect-robots video "$log"; done
```

The run prints `task_score` at the end. Offline checks: `YAM_ARENA=~/yam-isaac-lab-arena python -m pytest -q
test_task.py` (NumPy and pytest only).

## Scoring

Measured on each glass once everything is still (`SCORE_OVER = "end"`). A glass **hangs** when its foot's underside
is within 3 mm of the lips' top, its stem is within 8 mm of the channel's centre line, the whole foot is between the
open end and the closed end, and it hangs straight (tilt <= 5 deg).

- **1.0**: both glasses hang, nothing moves, and both TCPs are at least 3 cm from each glass (released).
- **0.5**: the same with one glass hanging.
- **0** otherwise, including a glass still held, and any glass knocked over (tipped past 45 deg) at any time during
  the trial (latched: standing it back up does not undo it).

## What makes this task distinctive

There is one way in and one place to hold. The foot is wider than the channel, so each glass has to be brought to
the rack's open end and threaded in sideways, stem first along the 25 mm gap, and only lowered once its foot is over
the lips. And it can only be held by its 9 mm stem, under an overhead rack: the foot has to ride above the lips
22 cm up while the hand that holds the stem stays below them, so the hand comes in nearly level and grips the stem
low. The glass is not set down on anything: it ends up hanging bowl-down by its foot from two 32 mm lips.

## Verification

Run on an RTX 5090 with `yam-isaac-lab-arena` at `bd62771`, using the commands above (`object_jitter=0`, default
seed):

- Scripted demo: `task_score` 1.0, ended by success at step 640 (32 s of trial time); 89 s wall clock including
  Isaac start-up. The right hand takes each glass by its stem, the fingers pointing out along the table, carries the
  foot 6-10 mm over the lips (the glass tilting at most 3.8 deg in the hand), slides it in along the channel, lowers
  it and backs out. Glass A hangs at y +0.053 and glass B at y -0.042, both with the foot on the lips (0.0 mm), the
  stem 0.5 mm off the centre line and tilt 0.0 deg.
- Dream-1: `task_score` 0.0 after the full 900-step run limit; 107 s wall clock. The right arm reached toward the
  glasses and nudged glass A (it moved about 7 mm and tilted 2.3 deg, then settled back upright), but neither glass
  was lifted; both stayed on the mat.
- Offline tests: 22 passed.
- Top camera: in the first frame the rack on its posts, the mat and both glasses are visible. In the demo's last
  frame both glasses hang bowl-down under the rack and the mat is empty; the rack's top hides the feet, and the
  bowls show below it.

## Third-party assets

none
