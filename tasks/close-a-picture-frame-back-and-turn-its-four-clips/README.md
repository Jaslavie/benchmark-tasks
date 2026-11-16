# Close a picture frame's back: press it flush with one hand while the other turns its clips, swapping hands

A picture frame of oak moulding lies face down on a felt mat. Its back (a hardboard backboard on the mount and foam
behind the picture) stands 12 mm proud of the frame until it is pressed down. Four brass turn buttons, two on each long
side, lie open along the frame. One hand presses the back down flush; the other turns the two clips on its side a
quarter turn each, over the back's edge. Then the hands swap: the second hand presses the back while the first turns the
clips on the other side. Both hands let go, the back held down by its four clips, and both arms go back to where they
started. The task is one task file,
[`close_a_picture_frame_back_and_turn_its_four_clips.py`](close_a_picture_frame_back_and_turn_its_four_clips.py),
for the `yam_arena` embodiment (scene, scoring and the scripted demo), plus [`bodykit.py`](bodykit.py), which it
imports from the same folder.

What it tests: keeping a sprung part pressed down with one hand while the other works small fittings around it, then
swapping roles between the hands; pushing small swivel clips round a quarter turn.

Why two hands: a clip cannot swing over a back that stands proud (the back's edge is in its way), and the back springs
up as soon as nothing presses it. With one hand (the right turning its two clips as in the demo, the left never moving)
the demo scored 0.0: both clips it turned stopped against the edge of the back, which stood 12 mm proud, after about 28 degrees.

## How it is modelled

`yam_arena` task files place single boxes, cylinders and spheres. The mat is fixed; the frame, its back and the four
clips are loose rigid bodies made of shapes from `bodykit.py`.

- The back: one rigid slab (backboard, mount and foam) joined to the frame by a joint (a `Thread` from `bodykit.py`,
  switched on throughout) that lets it slide up to 12 mm out of the frame and tilt up to 6 degrees. A spring pushes it
  up, so it stands 12 mm proud until about 9 N presses it flush, and a spring on its tilt brings it back level, so a
  side not held by a clip springs up.
- The clips: each a hinge about its screw (a `Hinge` from `bodykit.py`), with a spring holding it where it last
  stopped; after every control step, a clip pushed more than 3 degrees past its hold has given, and the task file
  moves the hold there, so a clip stays where it is left. Turned a quarter turn, a clip's plate lies over the back's
  edge, just above a back pressed flush.

Importing `bodykit.py` hooks `yam_arena.scene` when the simulator starts, without changing yam_arena, gives `score()`
each body's own pose, and lets the task file act on the simulation after every control step.

## Run and score a trial

The scorer is `task_score`: 1.0 when `success()` (full score, both arms back where they started, with the frame, its back and the clips still) has held for 1 s, otherwise `score()` at the end of the episode (0, 0.25, 0.5 or 1.0;
see `task.json`). `-E object_jitter=0` keeps every object exactly where the task file puts it, so every trial starts
from the same layout. The cameras render at 1280 x 704 for sharp videos; Dream-1's client scales every frame to
320 x 176 before sending it, so `-E cam_width=320 -E cam_height=176` gives the policy the same input.

Dream-1 (needs a key in `~/.dreamlabs/credentials`):

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=close_a_picture_frame_back_and_turn_its_four_clips.py -E object_jitter=0 -E control_hz=20 -E cam_width=1280 -E cam_height=704 \
  --instruction "Close the back of the picture frame: with one hand press its back down flush and keep it there while the other hand turns the two brass clips on its side a quarter turn each, over the back's edge; then swap, pressing the back down with the second hand while the first turns the two clips on the other side. Let go, leaving the back held down by its four clips, and take both arms back to where they started." \
  --scorer task_score --max-steps 1600 --store-frames --log-dir ~/yam-eval/frame-dream
```

The scripted demo, which reaches full score: the left hand comes down on the middle of the back, fingers closed, and
presses it flush. The right hand turns the two clips on the right side, fingertips down beside each clip's knob and
round a quarter turn. The left lets go (the back's left side springs up) and moves over to its side; the right presses
the back flush in the middle while the left turns the two clips on the left. Both let go and rise, the back staying
down, and both arms go back to where they started:

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=close_a_picture_frame_back_and_turn_its_four_clips.py -E object_jitter=0 -E control_hz=20 -E cam_width=1280 -E cam_height=704 \
  --instruction "Close the back of the picture frame: with one hand press its back down flush and keep it there while the other hand turns the two brass clips on its side a quarter turn each, over the back's edge; then swap, pressing the back down with the second hand while the first turns the two clips on the other side. Let go, leaving the back held down by its four clips, and take both arms back to where they started." \
  --scorer task_score --max-steps 1600 --store-frames --log-dir ~/yam-eval/frame-demo
```

Videos, one MP4 per camera next to each run's frames:

```bash
for log in ~/yam-eval/frame-dream/*.json; do inspect-robots video "$log"; done
```

Set `FRAME_DEBUG=1` to print what the score is built from twice a second: how far the back stands proud and how far it
is tilted, how far each clip has turned, whether both hands are clear, and where each TCP is.

## Runtime per trial

On one RTX 5090: about 7 minutes for a Dream-1 trial of 1,600 steps (430 s on the clock, with Dream answering in 0.84 s on average), and about 3 minutes for the scripted demo (716 steps, 35.8 s of simulated time, 152 s on the clock). Isaac Sim's first start on a new machine adds a few minutes while it compiles its shaders.

## Seeds

None needed: with `-E object_jitter=0` the layout is fixed, and the scripted demo follows fixed waypoints.
