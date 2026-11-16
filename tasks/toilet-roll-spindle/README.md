# Change a toilet roll on a spring-loaded spindle

A chrome two-post toilet-roll holder is fixed low on a white wall panel that stands across the far side of the
table. A spring-loaded telescoping spindle sits in the two small sockets on the posts' inner faces with an empty
cardboard tube hanging on it. A new, full roll lies on its side on the table to the front right, and a grey tray
stands in front of the holder for the empty tube. The robot must take the spindle out, get the empty tube off it
and into the tray, slide the new roll onto the spindle and fit the spindle back so that the roll hangs in the
holder.

Changing the toilet roll is something almost everyone does at home. The hard part is the refit: once the roll is
on, only the spindle's two short ends stick out, so it can only be held by one end. One end goes into its socket,
the spindle is pushed shorter between that socket and the fingers until the other end clears its post, and only
then is it let go so that it springs out into the second socket (the far post's chamfered socket edge can do part
of the squeezing as the end swings in). Let go early and the free end swings down and the roll falls. The roll is as
wide as most of the gap between the posts, so the spindle can only be brought in at an angle along a narrow path.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: holder, spindle, tube, roll, tray; `success()` and `score()` |
| `rollkit.py` | Rigid parts built from boxes, cylinders and spheres, with spring-loaded sliding or hinged links, for yam_arena task files |
| `demo.py` | The scripted full-score demo (closed loop, same embodiment, same scorer), with video recording, and the cheap-strategy variants |
| `task.json` | The submission text |
| `LICENSE` | MIT |

`rollkit.py` hooks `yam_arena.scene` and the embodiment's step and reset without modifying yam_arena, so the task
runs with the standard `yam_arena` embodiment and `task_score` scorer.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough, from this folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Replace the empty toilet roll with the new one: take out the spindle, put the empty tube in the grey tray, slide the new roll onto the spindle and fit the spindle back in the holder." \
  --scorer task_score --max-steps 3000 --store-frames --log-dir ~/yam-eval/toilet-roll-dream
```

`task_score` prints the trial's score: 1.0 on success (held for 1 s), otherwise `score()` at the last step.

## Scripted demo

The demo is closed loop, so it is not a `demo()` plan for `--policy yam_scripted`. `demo.py` runs it in the same
`yam_arena` embodiment with the same task file, layout (`object_jitter=0`, 20 Hz, 320 x 176 cameras) and limits as
a trial, sends 14 joint commands per step like a policy, and scores the last step with the same `task_score`
scorer, which it prints.

```bash
python demo.py task.py demo_out                                  # demo.mp4, setup.png, actions.jsonl, result.json
DEMO_VARIANT=nothing python demo.py task.py out --no-record        # both arms stay still
DEMO_VARIANT=early_release python demo.py task.py out --no-record  # let go once the first end is in its socket
DEMO_VARIANT=no_squeeze python demo.py task.py out --no-record     # swing the free end in without squeezing
DEMO_VARIANT=lift_only python demo.py task.py out --no-record      # stop with the roll held on the spindle
```

The demo plans from the fixed layout (where the sockets, the tray and the roll are) and, while it runs, from the
poses of the spindle's two ends, the tube and the roll, which a camera can see, and from the arms' own joint
angles. It reads no forces, contact flags or physical parameters.

1. The left arm grips the empty tube in the middle with a light squeeze (fingers about 38 mm apart on the 45 mm
   tube) and pulls it toward the robot: the tube drags the spindle forward, its domed ends ride out over the chamfered
   front edges of the sockets (the spindle shortens as they do) and it comes out. The left arm lays the tube, with the
   spindle in it, in the tray and moves away.
2. The right arm grips the spindle's right end where it sticks out of the tube, lifts it to the middle of the tube's
   bore and draws it out sideways; the tube stays in the tray.
3. It lays the spindle on the table beside the tray and takes it again 12 mm in from its right tip, so it lies
   level in the fingers.
4. It brings the spindle in from the side, level with the core of the new roll lying on the table, and pushes it
   through until 2 cm of it sticks out of the far side. It lets go (the spindle rests in the core) and takes the right
   end again from the front, the hand pointing forward and down so that the fingers close up and down on it, 20 mm
   in from the tip, and lifts the roll on the spindle. Held this way the roll's weight cannot twist the spindle out
   of line in the fingers; while it carries the roll the hand also turns and rises by what the spindle's seen pose
   shows it has drooped.
5. It backs away from the posts, turns the spindle 20 degrees (right end toward the robot), brings the left end in
   between the posts from the front and then along the spindle into the left socket.
6. It pushes the spindle 13 mm shorter against the left socket and swings the right end round the left socket until
   the spindle lies across the holder, its right tip 2 mm inside the right post.
7. It opens its fingers: the spindle springs out into the right socket and the roll hangs on it. The hand backs out
   the way it came and both arms move clear.

The demo prints `task_score: 1 (success) after 1500 steps` (75 s simulated, including the 1 s hold).

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`); the episode ends
early on success held for 1 s.

- An end is **seated** when its tip is at least 3 mm past its post's inner face and inside the 16 mm socket (within
  7 mm of its centre front to back and up and down). A tip that far past the face can only be in the socket.
- The new roll is **on the spindle** when its axis is within 16 mm of the spindle's axis (it hangs about 10 mm below
  it) and within 15 degrees of parallel, its whole width lies between the spindle's tips (5 mm tolerance), and its
  lowest point is at least 1 cm above the table (it hangs, rather than rests on the table).
- The empty tube is **off the spindle** when the same test fails for it.
- **Hands off**: no finger pad within 3 cm of the spindle's surface or of the roll's surface.

Score 0 if the new roll is not on the spindle or the empty tube is still on it. Otherwise:

| Score | Condition |
|---|---|
| 1.0 (success, held 1 s) | both ends seated, hands off, spindle and roll still (under 1 cm/s, roll turning under 1 rad/s), empty tube in the tray (centre inside the walls, at most 4 cm above the table, still) |
| 0.75 | both ends seated and hands off, the tube elsewhere |
| 0.5 | one end seated and hands off |
| 0.25 | any other state with the roll on the spindle and the tube off it (held in the air, or let go clear of the table) |

Seated credit needs both grippers clear, so holding an end in a socket earns nothing. There are no force
conditions. All of these can be checked with a ruler and photos of the holder from the front and from above.

### Cheap strategies (measured in Isaac Sim with the demo variants)

| Strategy | Score | What happens |
|---|---|---|
| Do nothing for 3,000 steps (official CLI, `--policy yam_scripted` with no waypoints) | 0 | nothing moves; the spindle stays seated with the empty tube on it and the new roll lies on the table |
| Full demo | 1.0 | success after 1,500 steps |
| Let go as soon as the first end is in its socket (`early_release`) | 0 | the free end swings down, the spindle drops out and the roll ends on the table |
| Swing the free end in without squeezing the spindle first (`no_squeeze`) | 1.0 | the dome rides over the right socket's chamfered edge, which squeezes the spindle as it swings in: a valid way to fit it, as on a real holder |
| Stop with the roll held on the spindle in the air (`lift_only`) | 0.25 | roll on the spindle, empty tube off it, no end seated |

Other cheap outcomes are covered by the rules above: pulling the old spindle out and leaving it, or laying the new
roll and the spindle on the table, scores 0 (the roll is not hanging clear of the table); the new roll cannot go
on while the empty tube is still on the spindle (the two together are longer than the spindle), and holding an end
in a socket earns no seated credit.

## Runtime and seeds

The layout is fixed (`-E object_jitter=0`); there are no seeds to choose. The demo reaches success after 1,500
control steps (75 s simulated). The run limit of 3,000 steps gives a policy twice the demo's time. One recorded demo
run takes about 11 minutes on an RTX A6000, including 2 to 3 minutes for Isaac Sim to start. The demo is deterministic apart
from small differences in the physics between machines.

## Fidelity

- All parts are rigid bodies built from boxes, cylinders and spheres. The holder, wall panel and tray are fixed.
- The spindle is two tubes (22 and 19 mm) on a prismatic joint with a linear spring of 200 N/m toward its relaxed
  length of 156 mm and 30 mm of travel (1.2 N when fitted, 6 N fully squeezed). Real roller springs are of this order
  (a few newtons); the value is assumed, not measured. For contact the tubes are 12-sided prisms; the ends are 12 mm
  spheres (domes). The spindle's sizes follow common replacement rollers (about 6 in long, telescoping, for posts
  with holes of about 16 mm).
- The new roll is a rigid hollow cylinder (11.5 cm across, 10 cm wide, 4.3 cm core hole, 130 g) made of 24 flat
  staves; the paper's softness is not modelled, and the task never squeezes the roll. The empty tube is a rigid
  2 mm-walled cylinder of 24 staves (real cardboard cores have 1 to 1.5 mm walls).
- Finger contacts on these parts resist twisting about the contact normal over a 10 mm patch (PhysX torsional
  patch radius), as rubber finger pads do; without it the spindle spins in a point grip.
- Friction: chrome holder 0.3, spindle 0.3, paper and cardboard 0.7, the table and tray as the station's default.

## Real-world build

A standard chrome two-post toilet-roll holder with 14 cm between the posts' inner faces and sockets of about 16 mm
(any holder that takes a spring roller), screwed to a board standing on the table, its sockets 10 cm above the
table; a spring-loaded telescoping replacement roller of about 6 in (15.5 cm); an empty cardboard core; a standard
roll of toilet paper (about 10 cm wide and 11.5 cm across); a plastic tray of about 16 x 22 cm taped to the table.

## Third-party assets and licenses

The SDK is yam-isaac-lab-arena, the program's official dependency, installed in the user's environment. Nothing
from it is redistributed here. Licenses are given as each source states them.

| Asset | Used for | Source | License as stated by the source | Bundled |
|---|---|---|---|---|
| YAM arm and gripper (URDF, meshes, collision hulls, converted USD) | both arms | SDK `yam_arena/assets/yam/`, from i2rt | MIT, Copyright (c) I2RT Robotics | no |
| Station floor and table | room | SDK `station.py` / `scene.py`: boxes with flat colours generated in code; no mesh or texture file | SDK README: "This repository has no license yet" | no |
| Top-camera stand (foot, mast, arm, housing) | room | SDK `station.camera_stand`: boxes and capsules generated in code | as the row above | no |
| Dome light and lamp | lighting | SDK `scene.py`, Isaac Lab-Arena dome light and an Isaac Lab sphere light | as the row above | no |
| top_cam, left_cam, right_cam | policy RGB | SDK pinhole camera sensors at the calibrated poses | as the row above | no |
| Station fruit, bowl, plate (Arena object library) and water bottle | not used: this task's own objects replace them | - | - | no |
| Holder, wall panel, spindle, tube, roll, tray | task objects | this folder (`task.py`, `rollkit.py`) | the authors' own geometry generated in code, MIT (this folder's LICENSE) | yes |

Isaac Lab-Arena ships the Apache License 2.0. Isaac Sim and Isaac Lab are runtime dependencies under their own
licenses; none of them is redistributed here. All task-specific code and geometry in this folder is the authors'
own work under the MIT License.
