# Fit up two bevelled coupons for a V-groove weld

Two steel coupons lie loose on a welding table, each with one edge bevelled for welding and each turned the wrong
way. The robot fits them up for a single-V groove butt weld: it turns each coupon so the two bevels face each other
across the joint, making a 60-degree V, and sets them flat side by side with a root opening of 1/16 to 1/8 inch (1.6
to 3.2 mm) all along the joint and their ends in line. One coupon needs a quarter turn and the other turning end for
end, and nothing on the table marks where the joint goes.

## The real step
A welding procedure gives the fit-up: "Root Opening: 1/16"-1/8"" ([LANL WPS
3501-11B](https://engstandards.lanl.gov/esm/welding/welding_specs/3501-11B.pdf), an AWS groove-weld procedure for
steel), and "Parts to be joined by butt welding shall be carefully aligned" ([LANL WFP 2-04, AWS D1.1 Structural
Welding Steel](https://engstandards.lanl.gov/esm/welding/vol2/WFP%202-04%20Procedure-R1.pdf), 8.0 D). Done by:
welders, cutters, solderers and brazers, and structural metal fabricators and fitters.

What goes wrong in real life: a root opening that closes up leaves the root unfused; one that opens too wide burns
through; a coupon turned the wrong way puts a square edge in the joint, so there is no groove for the root pass. A
photo from above shows all three.

The coupons are full-size: 1/2-inch plate, 3 x 3 inches, each with a 30-degree bevel down to a 1/8-inch root face
(60 degrees across the groove), the proportions of the prequalified single-V groove joint in AWS D1.1. What we
simplified: the coupons are rigid and perfectly flat; the table is a fixed steel plate; nothing is tacked or clamped,
so the coupons stay where the robot lets go of them. The bevel, the root face and a 1-inch band of the top face beside
the bevel are ground bright and the rest of each coupon is dark mill scale, as a coupon looks once it is prepared and
cleaned beside the joint.

## Run a trial
The coupons are custom bodies (a bevel isn't a box), so this task is a task-local plugin for the stock `yam_arena`
adapter: it registers the embodiment `yam_weld_fitup` (the stock robot, cameras, 14-D joint action and observations,
unchanged, plus the two coupons, scored from their measured poses). The scripted demo is the task file's `demo()`,
played by the stock policy `yam_scripted`.

On the Robocurve launchable, from this folder:

```bash
source /data/IsaacLab-Arena/yam.env
uv pip install --python /data/IsaacLab-Arena/.venv/bin/python --no-deps -e .
```

Elsewhere, install the pinned adapter first (`yam-isaac-lab-arena`'s `setup.sh`), then `uv pip install --no-deps -e .`
into the same environment. Then, from this folder:

```bash
INSTR="$(python3 -c 'import json; print(json.load(open("task.json"))["instruction"])')"
COMMON=(--embodiment yam_weld_fitup -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176
        --instruction "$INSTR" --scorer task_score --max-steps 2400 --store-frames)

# The scripted demo (full score)
inspect-robots run "${COMMON[@]}" --policy yam_scripted --log-dir runs/demo

# Dream-1 (needs a Dream API key in ~/.dreamlabs/credentials)
inspect-robots run "${COMMON[@]}" --policy dream --log-dir runs/dream

# One MP4 per camera, next to each run's frames
for log in runs/*/*.json; do inspect-robots video "$log"; done
```

The run prints `task_score` at the end. The trial ends at 1.0 once the complete fit-up has held for 1 s; otherwise
it scores the partial credit below at the last step.

## Setup
Positions in cm on the table, from the midpoint between the two arm bases: x forward (away from the arms), y to the
robot's left.

| Object | Description | Position |
|---|---|---|
| welding table | fixed dark steel plate, 30 x 32 x 1 cm | centred at (34, 20.5) |
| coupon_a | loose steel coupon, 7.62 x 7.62 x 1.27 cm (3 x 3 x 1/2 in), 0.56 kg, dark mill scale; one edge bevelled 30 degrees from vertical down to a 3.2 mm root face; the bevel, root face and a 2.5 cm band of the top face beside it ground bright | centred at (40, 29), bevel facing away from the arms (+x) |
| coupon_b | the same | centred at (40, 13), bevel facing right (-y), away from coupon_a |

Friction: steel on steel 0.6 static, 0.5 dynamic for the coupons; the welding table 1.2 / 1.0 (the task-file
default; PhysX averages the two surfaces in contact). Start: both grippers open at the ready pose. One fixed layout:
`-E object_jitter=0`, so the seed doesn't change the start.

## Scoring
A coupon's bevel faces the other coupon when the other's centre lies ahead of its bevelled edge and within that
edge's length. The root opening is measured at both ends of the joint, from each coupon's root edge (the bottom of
its bevelled side) to the other coupon's bevelled side, so a joint that is wider at one end reads wider there. Flat
means tilted at most 5 degrees and lying on the table (centre within 3 mm of its height lying flat); still means
moving under 1 cm/s and turning under 0.5 rad/s.

| Score | When, at the end |
|---|---|
| 1.0 | both coupons flat and still, both bevels facing each other, the root opening 1.6-3.2 mm at both ends of the joint, the ends in line within 5 mm; held for 1 s, which ends the trial (if time runs out first, this state scores 0.9) |
| 0.5 | both flat and still, both bevels facing each other and parallel within 10 degrees, side by side (opening 0-10 mm at both ends, ends within 2 cm), but the opening or the ends outside the fit-up limits |
| 0.25 | both flat and still, side by side and parallel within 10 degrees with only one bevel at the joint: a square edge in the joint |
| 0 | otherwise, including a coupon off the table, on edge or on top of the other, or either moving |

The joint can be anywhere on the table and run any way, and either coupon can be on either side. On a real bench, a
1/16-inch and a 1/8-inch gauge (or welding rod) check the opening at each end of the joint, and a straightedge
across the ends checks that they line up.

## The scripted demo
The left arm grips coupon_a at its middle, the fingers on the two square edges either side of the bevel, lifts it,
turns it a quarter turn while carrying it and sets it down with its bevel facing right, 22 cm left of centre. It
then grips coupon_b the same way, turns it end for end and sets it down 2.4 mm from coupon_a, its bevel facing left.
Each coupon is set down slowly and let go 1 mm above the table, the fingers opening slowly to 9 cm.

## Runtime and seeds
- The scripted demo's plan takes about 43 s of simulated time (856 steps at 20 Hz), within the 2,400-step limit.
- Official runs (2026-11-08): the demo 1.0, ending on success at 40 s, with a root opening of 2.4-2.6 mm and the
  ends 0.4 mm apart; Dream-1 0.0 over the full 2,400 steps (it never moved a coupon).
- The scoring and reach checks run without the simulator: `python -m pytest -q test_weld_fitup.py`.
- One fixed layout: `-E object_jitter=0`, so the seed doesn't change the start.
