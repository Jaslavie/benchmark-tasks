# Feed a hex shaft through two bearings

Two bearing plates stand fixed on the table, 20.3 cm apart, each with a bearing on one shared line. A 1/2-inch hex
shaft lies on the table beside them. The robot slides the shaft through both bearings until each end sticks out at
least 1 cm past the outside of its bearing, room for the collar that will lock it. The bores leave a millimetre each
side, and the fingers can't follow the shaft through a plate: the robot has to push, let go, grip the shaft again
further along and push again, holding it on the line to a millimetre throughout. It is the step after the bearings are
pressed into the plates, from WCP's GreyT Shooter assembly instructions.

## The real step
"Slide the cut shooter wheel shaft into position." ([WCP, GreyT Shooter (4"-6"), Additional Hardware, Step
2](https://docs.wcproducts.com/welcome/greyt/greyt-shooter-4-6/assembly-instructions/additional-hardware/steps-1-5).)
Two collars then lock it: "The Collars should be flat with the ThunderHex Bearing flanges." ([GreyT Shooter (9.5"),
Belt Drive](https://docs.wcproducts.com/welcome/greyt/greyt-shooter-9.5/assembly-instructions/belt-drive).) Done by:
robot builders and electromechanical assemblers.

The parts are the kit's, at full size ([complete
kit](https://docs.wcproducts.com/greyt-shooter-9.5/general-info/kit-contents/complete-kit)): 1/2" hex shaft stock, "1/2"
ThunderHex ID x 1.125" OD x 0.313" WD (Flanged Bearing)" (WCP gives the ID as 13.75 mm), and 8" standoffs between the
plates. What we simplified: the hex is a 12.7 mm square bar and the bore a 14.7 mm square, 1 mm each side. Real
ThunderHex bearings fit snug and real shaft ends are chamfered; the extra room stands in for the chamfer that guides a
real shaft in. The bearings are square blocks with no flange, and the plates are cut down to 10 x 7 cm around them.

## Run a trial
On a machine with the official simulator (the Robocurve Brev launchable, or `yam-isaac-lab-arena`'s `setup.sh`), with
a Dream API key in `~/.dreamlabs/credentials`, from this folder:

```bash
# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=feed_shaft.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Slide the hex shaft through both bearings so that each end sticks out at least 1 cm past the outside of its bearing, room for a collar on each end." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/feed-shaft-dream

# The scripted demo, which reaches full score
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=feed_shaft.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Slide the hex shaft through both bearings so that each end sticks out at least 1 cm past the outside of its bearing, room for a collar on each end." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/feed-shaft-demo

# One MP4 per camera, next to each run's frames
for log in ~/yam-eval/feed-shaft-dream/*.json; do inspect-robots video "$log"; done
```

The run prints `task_score` at the end. The trial ends at 1.0 once the shaft has been through both bearings, with
room for both collars and still, for 1 s; otherwise it scores the partial credit below at the last step.

## Scoring
The shaft runs through a bearing when its centre line crosses the middle of that bearing's plate inside the bore. The
shaft can go in from either side.

| Score | For |
|---|---|
| 1.0 | through both bearings, each end at least 1 cm past the outside of its bearing, and still, held for 1 s (the trial then ends; if time runs out first, this state scores 0.9) |
| 0.5 | through both, but an end has less than 1 cm for its collar, or the shaft is still moving |
| 0.25 | through one bearing |
| 0 | otherwise |

Still means moving under 1 cm/s and turning under 0.5 rad/s. Centred, each end is out 1.82 cm, so the shaft's middle
can sit up to 8.2 mm either way of halfway. A policy that pushes once from its first grip stops when the fingers meet
the first plate, with the shaft through one bearing: 0.25. On a real bench, a ruler measures how far each end sticks
out past its bearing's flange.

## The scripted demo
The left arm grips the shaft off the table at its middle, waits for the grip to settle and lifts it slowly, carries it
over the plates, turns it along the bores' line and lowers it beside the first plate, its leading end 1 cm short of the
bearing. Stroke 1 pushes it through the first bearing until the grip is 2.8 cm from the plate (the fingers can't pass
it), 9.9 cm through, and lets go; the shaft tips in the bore until its far end rests on the table, about 12 degrees.
Stroke 2 grips it again between the plates where it lies, raises the grip onto the line, which swings the shaft about
the first bearing, and pushes it through the second until each end has room for its collar. It lets go gently and
leaves straight up.

## Runtime and seeds
- The scripted demo's plan takes 33 s of simulated time (660 steps at 20 Hz), within the 2,400-step limit.
- Official runs (2026-11-08): the demo 1.0, ending on success at 31.9 s (637 steps); Dream-1 0.0 over the full
  2,400 steps.
- The scoring checks run without the simulator: `python -m pytest -q test_feed_shaft.py`.
- One fixed layout: `-E object_jitter=0`, so the seed doesn't change the start.
