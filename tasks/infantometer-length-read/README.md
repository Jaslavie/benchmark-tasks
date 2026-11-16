# Straighten and hold a newborn training doll's two sprung legs flat while the other arm closes the footboard

The robot straightens the two sprung legs of a 45 cm newborn training doll and holds them flat on an infant length
board with one arm, while the other arm slides the blue footboard onto both soles. Both arms then let go with the
footboard left at the reading (450 mm). This is the recumbent-length measurement that paediatric nurses, medical
assistants and community health workers make at every well-baby and growth-monitoring visit (WHO length-board method).

What makes it hard is the doll, not the board. Each leg is a sprung two-joint chain: at rest the hips sit at 80 deg
and the knees at 95 deg with the feet in the air, so the soles reach only the 391 mm mark. Each wrong plan below was
run in the simulator and fails on physics:

- Sliding the footboard up to the feet touches them at 390 mm. Pushing on folds the hips further (80 to 85 deg), so
  the reading only gets shorter.
- Pulling the ankles straight drags the 0.33 kg trunk 8 mm off the headpiece while the knees are still bent 56 deg.
- Pressing one thigh flat leaves the other leg curled, even when the footboard ends at the right mark.
- Pressing both thighs flat and then fetching the footboard fails: the legs spring back as soon as the hand lifts,
  and the footboard closes on curled feet at 389 mm.

Only one plan works: one hand straightens both legs and holds both thighs flat while the other closes the footboard
onto the soles. A single gripper cannot do both, because the thighs and the footboard handle are about 15 cm apart.

Earlier hold-while-closing tasks hold a part that is already in shape. In TK-RGRUBN (a pin held in a caliper) and
TK-48JHRV (a block held in a vise) the held part is one rigid piece that only has to stay put, and in TK-M6GCBU (an M.2
drive) a single sprung part is pressed down at one point. Here the held part is two sprung two-joint chains, and the
hand has to change their shape before it can hold them:

- The legs must first be uncurled. In our runs a finger pushing on the knee ends slipped off at about 40 deg of hip
  flexion, so the hand has to move onto the thigh tops part way.
- Both legs must be held flat at once by one hand. Each is free to curl on its own (`one_knee`).
- The chain buckles under end load. Closing the footboard on a bent leg folds it further instead of stopping it
  (`footboard_push`), and pulling the feet drags the doll off the headpiece (`pull_ankles`).
- The legs re-curl the moment the hand lifts (`press_then_fetch`), so the hold has to last until the footboard is
  on the soles.

The legs read the true length only when both thighs are pressed flat at once, with the heels sliding down the board.

Instruction given to the policy (20 words; it states the goal, names the doll as a training doll, and never says how
to hold the legs):

`Measure the training doll's length on the white length board, leaving the blue footboard at the reading, then let go.`

Files in this folder:

- `task.json`
- `task.py`: scene, whole-run record, scoring and the scripted demo
- `linkkit.py`: articulated rigid-link bodies for yam_arena
- `linkdemo.py`: closed-loop scripted-demo runner
- `tests/test_scoring.py`: scorer edge cases
- `README.md`

## Requirements

Built on the program's Brev launchable (Isaac Lab-Arena 0.3.0, Isaac Sim 6.1, Isaac Lab 17.0.2, one L40S). Versions
used for the submitted runs:

- inspect-robots 1.0.0 (`inspect-robots --version`)
- yam-arena at commit 0059d02a1a91 (`git -C ~/yam-isaac-lab-arena rev-parse --short=12 HEAD`)
- inspect-robots-dream 0.1.0, Dream checkpoint `dreamlabs/dream-1-beta`
- task file SHA-256, first 12 hex: b618ecb0e915 (`sha256sum task.py | cut -c1-12`; equals
  `eval.environment_revision` in the log)

Before every shell: `source ~/IsaacLab-Arena/yam.env` (on our VM, `/data/IsaacLab-Arena/yam.env`).

`linkkit.py` and `linkdemo.py` must sit next to `task.py`. Importing `linkkit` installs an import hook. The hook
teaches `yam_arena.scene` to spawn the board and the doll as PhysX articulations: a prismatic footboard joint with
rail friction, and sprung revolute hips and knees with limits and drives, spawned through Isaac Lab's
`ArticulationCfg`. It also calls the task's per-step history record after every control step. There is no fork of
yam_arena and there are no third-party assets: every body is built from box and capsule colliders in `task.py`.

## Offline check, no simulator

```bash
python -m pytest -q tests
```

37 tests feed synthetic control steps to the scorer and check the scene with the kit's own kinematics. They import
`task.py`, so they need yam_arena installed, but no simulator.

Scorer cases:

- The demo's trace scores 1.0.
- After a touch that read 450.5 mm, a footboard left at 445.5 or 455.5 mm passes; 445.4 or 455.6 mm scores 0.5.
- A valid touch that reads 444, 456 or 459 mm, with the footboard left there, scores 1.0.
- A 0.3 s touch at 459 mm is not a measurement and does not widen the window. Two measured touches at 447 and
  456 mm accept a footboard left anywhere from 442 to 461 mm.
- A touch held 0.45 s is not a measurement; 0.50 s is.
- A sole 2.1 mm off the footboard, passed by it by 3.1 mm, or beside the plate blocks the touch.
- A crown gap of 2.1 mm blocks both the touch and the straight hold.
- A hip over 12 deg or a knee over 15 deg breaks "legs straight"; 12.0 and 15.0 deg pass.
- A footboard moving at 5.1 mm/s is not a touch.
- A single 20.1 mm lift, or the trunk pushed off the board, scores 0 even after a perfect measurement; 19.9 mm does
  not.
- A pad coming within 29 mm restarts the 1 s release, and a gripper moving at 6 mm/s blocks it.
- Holding on scores 0.75, moving the footboard off the mark after measuring scores 0.5, press-then-fetch scores 0.25
  and the footboard set on the tape mark without a touch scores 0.
- The measurement survives the legs curling up again, and the record clears only on reset.

Geometry cases:

- The design pose is 450 mm from crown to sole and reads as a touch.
- Inside the touch limits (hips up to 12 deg, knees up to 15 deg, nothing through the board) the crown-to-sole
  length stays between 448 and 456 mm.
- The curled legs first touch the footboard between 385 and 400 mm.
- Every demo pose is reachable by `yam_arena.ik.solve` within 2 mm and 1 deg.

## Run and score one Dream-1 trial

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Measure the training doll's length on the white length board, leaving the blue footboard at the reading, then let go." \
  --scorer task_score --max-steps 1200 --seed 0 --no-prompt --store-frames --log-dir ~/yam-eval/infantometer-length-read-dream
```

Scoring is automatic. `task_score` reports 1.0 when the task file's `success()` holds for 1.0 s and the episode ends
with `termination_reason="success"`. Otherwise it is the task file's `score()` at the end of the episode
(`SCORE_OVER = "end"`). For repeat trials add `--epochs K`; with `object_jitter=0` the seed changes nothing in the
scene.

The same command with `--policy yam_scripted` was run on these files to check the CLI path (`--max-steps 1200`,
`--seed 0`). It gave status `success` and `task_score 0`, because the scripted policy only holds the ready pose. The
run ended by `policy_stop`, and `eval.environment_revision` equalled the task file's SHA-256 prefix.

## Scripted demo and cheat variants

```bash
python linkdemo.py task.py demo_out                            # full-score demo
python linkdemo.py task.py cheat_out --variant footboard_only  # a cheat variant (or DEMO_VARIANT=footboard_only)
```

`linkdemo.py` runs `demo_script()` from `task.py` closed loop at 20 Hz. It writes `top_cam.mp4`, `left_cam.mp4`,
`right_cam.mp4`, `setup.png` and `result.json` (score, steps, IK error and the scorer's record). The script reads only
what a camera could measure: link poses, hip and knee angles computed from those poses, the footboard's pose, and the
robot's own gripper state. It never reads joint values or the history record.

1. Both arms together. The right gripper comes down on the footboard handle, pads closing across it, and grips it
   (22 mm handle, pads closed to 16 mm). The left gripper, pointing straight down with its pads 46 mm apart so that
   one finger sits over each thigh, lowers its fingertips 40 mm headward of the curled knees, at knee height.
2. Phase A. The left fingers swing about the seen hip axis (16 deg/s) and push the knees footward. This stops once
   the seen hips are below 52 deg (after 41.6 deg of swing).
3. Phase B. The fingertips move onto the thigh tops 65 mm from the hips, 20 mm short of the knees. They follow the
   thighs down (15 deg/s, leading the seen angle by 8 deg) to 4 deg below flat. On the way the heels land and slide
   footward along the board. A closed loop presses 3 mm deeper while either hip is above 1.5 deg or either knee is
   above 3 deg (6 mm extra: hips 0.7 and 1.0 deg, knees 1.4 and 2.0 deg).
4. Close. The right arm slides the footboard toward the soles: 25 mm/s until the larger seen sole gap is 6 mm, then
   5 mm/s until both soles touch with the crown on the headpiece, or until the footboard stalls. It then pushes
   0.5 mm more and holds 1 s: reading 451.3 mm, sole gaps 0.2 and 0.0 mm, crown 1.1 mm from the headpiece, hips 0.0
   and 0.3 deg, knees 0.0 and 0.5 deg. The touch holds for 1.35 s and the measurement is recorded at
   t = 17.2 s.
5. Release. The left fingers let the thighs follow them back up to 40 deg, open off the thighs (the legs curl up
   between them) and lift away. The curling feet push the held footboard to 458.6 mm. The right arm slides it
   back to the reading it saw at the touch (451.0 mm), then opens and lifts away. Both hold still.

Result: the measured readings (touch held 0.5 s or more) ran from 451.3 to 451.7 mm, and every touch read 451.3 to
453.1 mm. The crown was never more than 1.1 mm off the headpiece and the doll never lifted. The footboard was left at
451.0 mm, 0.3 mm from the measured readings, and both grippers were clear and still for 1.95 s. Success holds for 1 s
and ends the episode at step 628 (31.4 s simulated); task_score 1.0. A reset afterwards (`--check-reset`) puts the
footboard back at 520 mm, every doll joint within 0.22 deg of its rest angle and the trunk within 0.15 mm of its
start, and clears the record. The cheat variants below ran in one later simulator session (a development driver that
resets between variants and runs the same `demo_script`), and none inherited an earlier variant's record.
`press_then_fetch` was also run on its own: inside the shared session a reset left the free doll 1.6 mm footward of its
start (crown 2.6 mm off the headpiece), so its straight hold never had the crown on the headpiece and it scored 0.0
there. A fresh start puts the crown 0.9 mm off the headpiece. Each variant can be reproduced on its own with
`python linkdemo.py task.py out --variant <name>`.

Cheat variants, with their simulator results from that session (the files in this folder, `object_jitter=0`):

| Variant | What it does | Result | task_score |
|---|---|---|---|
| `nothing` | Both arms stay at the ready pose. | legs curled, footboard at 520 mm | 0.0 |
| `footboard_only` | The right arm alone slides the footboard up to the curled feet, holds 1 s and lets go. | first touch at 389.7 mm; the legs are never straight | 0.0 |
| `footboard_push` | The right arm alone pushes the footboard on to the 380 mm mark and lets go. | the feet fold the hips from 80 to 85 deg; footboard at 380.2 mm | 0.0 |
| `set_by_tape` | The right arm alone sets the footboard face on the 450 mm tape mark by eye and lets go. | footboard at 450.0 mm, 60 mm short of the curled feet; never measured | 0.0 |
| `pull_ankles` | The left arm pinches both shins just above the feet and drags them 80 mm footward and down at 20 mm/s and holds; the right arm closes the footboard onto the soles; both let go. | the trunk slides 8 mm off the headpiece with the knees still at 56 deg; closing the footboard pushes the doll back and folds the legs (hips 57 deg) | 0.0 |
| `one_knee` | The left arm, fingers closed, presses only the far thigh flat; the right arm closes the footboard onto the far sole; both let go. | far leg flat and footboard left at 450.4 mm, but the near leg stays curled (hips 73 deg) | 0.0 |

Wrong plans that earn partial credit, also run in the simulator:

| Variant | What it does | Result | task_score |
|---|---|---|---|
| `press_then_fetch` | The left arm presses both thighs flat, holds 2 s and lifts away; then the right arm closes the footboard onto the feet. Run on its own. | legs straight with the crown on the headpiece for 4.45 s; the legs spring back up and the footboard closes on curled feet at 389.4 mm | 0.25 |
| `hold_on` | The full demo without the release. | measured at t = 16.95 s (read 450.4 to 450.5 mm), footboard at 450.5 mm, both grippers still on | 0.75 |

## Scoring (`score()`, at the end of the episode; `success()` must hold 1 s)

Every control step from reset, the record (`record()` into a `linkkit.History`, cleared only by the simulator reset)
reads these values:

- both hip and knee angles (from the joints; 0 = straight)
- the gap from the crown to the headpiece face
- each sole's gap to the footboard face (nearest of its four sole-face corners)
- the footboard reading (headpiece face to footboard face, from the rail joint) and its speed
- the doll's lowest collider point and its trunk position
- each gripper's pad clearance to the doll and the footboard, and its speed

Two derived conditions feed the ladder:

- Touch: both hips at most 12 deg and both knees at most 15 deg (legs straight); the crown within 2 mm of the
  headpiece; both soles within 2 mm of the footboard face and in front of it; the footboard moving slower than
  5 mm/s.
- Measured: a touch held for 0.5 s at any time in the run. It stays set when the legs re-curl. Every control step of
  a touch that has held 0.5 s adds its footboard reading to the measured readings. Within the touch limits the
  crown-to-sole length is 448 to 456 mm, so a touch reads about 445 to 460 mm on the 450 mm doll.

Gates, any time in the run (each gives 0):

- the doll's lowest point more than 20 mm above the board (lifted)
- the trunk centre leaving x 8 to 32 cm or y -20 to 44 cm (pushed off the board)

Ladder:

| Score | Rule |
|---|---|
| 0.25 | both legs held straight with the crown on the headpiece for 1 s |
| 0.5 | measured |
| 0.75 | measured, and the footboard at the end within 5 mm of the measured readings (the instruction's "at the reading") |
| 1.0 | and for the last 1 s every pad of both grippers at least 30 mm from the doll and the footboard, both grippers slower than 5 mm/s (success, which must also hold 1 s) |

Moving the legs or the footboard back never clears a gate or a measurement. `score()` returns the rung,
`success()` is true on the 1.0 rung, and `distance()` is 1 - score.

Real trial: read the footboard face against the tape on a top photo (to 5 mm). A side photo during the hold shows
both knees flat, the crown on the headpiece and both soles on the footboard.

## Run limit, runtime and seeds

- Run limit: 1,200 control steps at 20 Hz (60 s simulated), `--max-steps 1200`. The demo succeeds at step 628.
- Runtime on one L40S: the demo took 10 to 14 min wall clock including Isaac Sim start-up on our shared VM, with
  other simulations running (0.8 to 1.2 control steps/s). A Dream-1 trial takes 4 min for
  1200 steps (Dream round trips pause the simulator). The first run after boot adds a few minutes of
  warm-up.
- Seeds: `--seed 0` for every submitted run. `-E object_jitter=0` fixes the layout (the board and the doll are never
  jittered). The scripted demo is deterministic; Dream-1 is a hosted model without a seed.

## Submitted videos and scores

- Dream-1 run: log `adhoc_43a84023.json`, trial `scene-0-e0`, `task_score = 0.0`, ended by `max_steps` after 1200 steps.
- Demo: `python linkdemo.py task.py demo_out`, `demo_out/result.json`, `task_score = 1.0`, ended by `success` after
  628 steps. The video is `demo_out/top_cam.mp4` and the screenshot `demo_out/setup.png` (frame 0 of the top
  camera).

```bash
inspect-robots inspect adhoc_43a84023.json
jq '{status, scores: .samples[0].epochs, ends: .samples[0].termination_reasons, limit: .eval.max_steps, seed: .eval.seed, task_sha12: .eval.environment_revision}' adhoc_43a84023.json
```

## Materials

- Length board: white melamine-faced plywood, 680 x 260 x 18 mm, about 2 kg, fixed to the table. Its laminate top is
  smooth (friction 0.35 static, 0.30 dynamic). A fixed white headpiece, 260 x 15 x 160 mm, stands at the 0 mm end,
  and a yellow printed tape 20 mm wide with black 50 mm ticks runs from 0 to 600 mm along the far edge.
- Footboard: blue painted plywood plate, 200 x 12 x 160 mm, on a 200 x 60 x 8 mm shoe that runs in the board's rail,
  with a 22 x 50 x 22 mm beech handle on its back 120 mm above the board. It weighs 0.30 kg and slides with 1.5 N of
  static friction (1.2 N moving), so it stays where it is left.
- Doll: a 45 cm, 0.45 kg vinyl newborn length-measurement training doll with a white cloth nappy, fitted with a
  torsion-spring hinge at each hip and knee (a jointed vinyl doll with a small torsion spring in each joint). Friction
  0.6 static, 0.5 dynamic; the shins and feet are smooth glossy vinyl, 0.25 static, 0.2 dynamic. The head, trunk and
  arms are one rigid 0.33 kg body. Each leg has a thigh
  (38 mm thick, 85 mm hip to knee, 35 g) and a shin with its foot (80 mm knee to sole, 25 g; foot 32 x 20 x 70 mm,
  toes up). The hips and knees are torsion-sprung hinges: hips 0.07 N m/rad toward 95 deg of flexion, knees
  0.05 N m/rad toward 81 deg. They settle at hips 80 deg and knees 95 deg. With every joint straight the doll is
  450 mm from crown to sole.
- Positions and colours for every body are in `task.json` (setup) and at the top of `task.py`.

## Reset between trials

Nothing is consumed and the board is fixed. Lay the doll face up on its tape outline, midline on the board's centre
line, crown 1 mm from the headpiece, and let the legs spring up to rest (hips about 80 deg, knees about 95 deg, feet
in the air). Slide the footboard to the 520 mm mark.

## Licence

MIT. Copyright (c) 2026 the task authors. `linkkit.py` and `linkdemo.py` carry the full MIT text; `task.py` and
`tests/test_scoring.py` carry the SPDX identifier. This is our own code, written with public Isaac Lab and USD Physics
techniques; it copies no code from other tasks in this repository. There are no third-party assets.
