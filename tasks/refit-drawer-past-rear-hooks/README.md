# Refit a loaded drawer on its roller slides, carried in above riding height so the rear hooks clear, then pull-test it

A small wooden drawer cabinet with an 18 mm top stands on the table with its drawer taken out, as at home after a
drawer was taken out to get back something that fell behind it, to clean it, or to move the chest. The drawer, loaded
with a 0.8 kg box of parts and fitted with a bar handle, stands on the table between the arms and the cabinet. The robot must put the drawer back on its slides, close
it, and then pull it open again by about 5 cm to show that it runs.

The cabinet's slides are roller slides. Each side wall carries a steel rail with a small roller at its front end;
each side of the drawer carries a steel runner with an orange hook plate under its rear half. A seated drawer rides
on the rollers (under the runners) and on the rails (under the hook plates). To put it in, the hooks must pass over
the rollers, so the loaded drawer has to be held level, about a centimetre or two higher than where it will ride,
carried in until the hooks are past the rollers, and set down. Pulled out, the hooks stop the drawer against the
rollers after 75 mm, so a seated drawer cannot be pulled out by accident. The cabinet top closes the opening from
above, so the drawer can only go in through the front: with the drawer on its slides there is 24 mm between its side
walls and the underside of the top and 14 mm above its front panel.

The common mistake is to push the drawer in low: it then slides along the cabinet floor under the slides and also
closes, but it is not on its slides. The instruction only names the goal ("on its slides", then "pull it open about
5 cm"); it does not say how to get past the hooks or how to tell the two cases apart.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, `success()`, `score()` and the scripted `demo()` |
| `drawerkit.py` | Rigid parts built from several boxes (free or fixed, each box with its own friction) for yam_arena task files |
| `record_demo.py` | Plays `demo()` through the same embodiment and scorer, prints the drawer's state and records MP4s |
| `LICENSE` | MIT License for the code and geometry in this folder |

`drawerkit.py` hooks `yam_arena.scene` and the embodiment's step and reset without modifying yam_arena, so the task
runs with the standard `yam_arena` embodiment and `task_score` scorer.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough, from this folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Put the drawer back into the cabinet on its slides and close it. Then pull it open again by about 5 cm and leave it there." \
  --scorer task_score --max-steps 1600 --store-frames --log-dir ~/yam-eval/drawer-refit-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode.

## Scripted demo

The same command with `--policy yam_scripted` plays `demo()`: open loop, planned only from the fixed start layout;
it never reads the simulator's state. To record it:

```bash
python record_demo.py task.py demo_out                                  # full score: demo_*_cam.mp4, setup.png
DEMO_VARIANT=floor python record_demo.py task.py out --no-record        # push it in low, under the slides
DEMO_VARIANT=short python record_demo.py task.py out --no-record        # lift too little: hooks meet the rollers
DEMO_VARIANT=insert_only python record_demo.py task.py out --no-record  # seat it and stop
DEMO_VARIANT=close_only python record_demo.py task.py out --no-record   # seat it, close it, stop
DEMO_VARIANT=pull:12 python record_demo.py task.py out --no-record      # pull 12 cm instead of 5
DEMO_VARIANT=floor,pull:12 python record_demo.py task.py out --no-record  # variants combine with commas
DEMO_VARIANT=above python record_demo.py task.py out --no-record        # try to lower it in from above the cabinet
DEMO_VARIANT=sidegrip python record_demo.py task.py out --no-record     # hands follow the drawer in from above
```

1. Both arms grip the drawer's side walls from above at its centre of mass (fingers closing across the wall) and
   lift it level.
2. They carry it into the opening with the hook plates above the rollers, as far as the hands can go before the
   cabinet face, and lower it until the rear of the hook plates rests on the rollers.
3. One arm at a time lets go of its side wall and grips the front panel where it overhangs the drawer box (fingers
   closing across the panel); the drawer rests on the rollers and the other hand meanwhile.
4. Holding the front panel at that height they push the drawer in until the hooks are 15 mm past the rollers, set it
   down on the rails 6 cm short of closed, and let go.
5. Closed fingers push the front panel until it is shut; the arms back off.
6. The left arm grips the bar handle, pulls it 5 cm toward the robot, lets go and moves away.

The demo reaches 1.0 (success) at step 849 (42 s), both through `inspect-robots run --policy yam_scripted` and in
`record_demo.py`.

## Scoring

Evaluated every control step; the state at the end of the episode is scored (`SCORE_OVER = "end"`).

- **On its slides**: the drawer bottom 66 +/- 4 mm above the table (measured at the front panel's foot), level
  within 2.5 degrees, square to the cabinet within 3 degrees and centred within 12 mm, and open no more than 78 mm
  (it stops at 75 mm when the hooks meet the rollers). Pushed in under the slides the drawer sits at 12 mm.
- **Closed**: the front panel within 5 mm of the cabinet face. **Hands off**: every point of both grippers at least
  2 cm from the drawer (handle included) and the box.
- 0 if the box ever leaves the drawer or the drawer falls off the table.
- 1.0 (the trial ends as a success once this has held for 1 s): earlier the drawer was closed on its slides with hands off for 1 s; now it is on its
  slides, open 30 to 80 mm, the box inside, hands off.
- 0.5: it was closed on its slides with hands off for 1 s, and it is still on its slides.
- 0.25: it is on its slides.
- Otherwise 0.

All of it can be checked with a ruler on the video and a photo of the end.

Measured in simulation (scores as printed by the live scorer; `record_demo.py` with the given `DEMO_VARIANT`):

| Strategy | Score |
|---|---|
| Scripted demo (`inspect-robots run --policy yam_scripted`) | 1.0 (success at step 849) |
| Do nothing (1,600 steps, `inspect-robots run`) | 0 |
| Push it in low (`floor`): carried just over the cabinet floor, it slides in under the slides; the closed fingers catch the end of the handle bar, so it stops 29 mm short of closed, and the open-loop handle grip misses | 0: front panel foot at 12 mm, not on its slides |
| Lift too little (`short`): the hooks meet the rollers, the drawer jams and tips 12.5 degrees | 0 |
| Lower it in from above (`above`): side-wall grip, lifted to clear the 19.8 cm cabinet top; at the cabinet the arms are past their reach, the drawer pitches 12.6 degrees, slips from the grip and drops back onto the table | 0 |
| Hands follow the drawer in from above (`sidegrip`, the route an open cabinet allows): the fingers hit the cabinet top's front edge, the drawer tips and ends leaning out of the opening at 23.5 degrees | 0 |
| Seat it on its slides and stop (`insert_only`) | 0.25 |
| Seat it, close it, let go and stop (`close_only`) | 0.5 |
| Full demo but pull 12 cm (`pull:12`): the drawer stops at its hooks | 1.0 (75 mm open is inside 30 to 80 mm) |
| Push it in low and pull 12 cm (`floor,pull:12`) | 0 (as `floor`) |

Pulling the drawer open without having closed it on its slides first scores at most 0.25.

## How the hidden property can be observed

Once the drawer is in, the cameras cannot see whether its hooks and runners sit on the rails: they are behind the
drawer's side walls. Two things show it:

- the height of the front panel: on its slides its foot is 66 mm above the table, under the slides 12 mm;
- a pull test: pulled by its front panel, a drawer on its slides runs out level at the same height and stops when
  the hooks meet the rollers; a drawer under the slides drags out along the floor with no stop.

Measured with `record_demo.py` (it prints the drawer's state every 20 steps: `height_mm` is the front panel's foot
above the table, `opened_mm` the distance from closed, `tilt` in degrees), after the drawer was put in and closed:

| Drawer state | Closed: front panel foot above the table | Pulled 5 cm | Pulled 12 cm |
|---|---|---|---|
| On its slides (demo, `pull:12`, by the handle) | 66.0 mm | runs out 48.4 mm, still 66.0 mm high, tilt 0.0 deg | stops at 75.0 mm (hooks on the rollers), 66.0 mm high, tilt 0.0 deg |
| Under the slides (`floor`, `floor,pull:12`) | 12.0 mm | drags out 33.0 mm at 12.0 mm | no stop: comes out 107.4 mm and its front drops to the table (0.2 mm, tilt 5.4 deg) |

The under-the-slides pulls were measured by the front panel's top edge before the cabinet top and the handle were
added; nothing below the top changed. With the top in place the `floor` variant shows the same 12.0 mm front panel
height.

```bash
DEMO_VARIANT=pull:12 python record_demo.py task.py probe --no-record
DEMO_VARIANT=floor python record_demo.py task.py probe --no-record
DEMO_VARIANT=floor,pull:12 python record_demo.py task.py probe --no-record
```

## Runtime and seeds

About 42 s of simulated time for the demo; a full 1,600-step trial takes about 2 minutes on an RTX A6000 plus about
1.5 minutes for Isaac Sim to start. The layout is fixed (`-E object_jitter=0`); there are no seeds to choose.

## Fidelity

All parts are rigid boxes. The rollers and ball bearings of real roller slides are modelled as low-friction (0.05)
sliding contacts and a fixed 6 mm step at the front of each rail; there is no self-closing spring or soft-close
damper. The drawer (1.0 kg) and its load (0.8 kg) are lighter than a typical full kitchen drawer so that two YAM arms
(27 N m at the large joints, 7 N m at the wrist, about 40 N per finger) can carry it level by its side walls.

## Real-world build

A plywood drawer box 300 mm wide, 202 mm deep and 90 mm tall with an 18 mm front 390 x 100 mm and a bar handle
(160 mm centres, standing 35 mm off the front); a 40 cm wide, 26 cm deep carcass with 18 cm sides and an 18 mm top
(19.8 cm tall). Bottom-mount roller slides (cabinet member with a front roller, drawer member with a rear
hook) fixed 90 mm above the bottom of the carcass, or the equivalent built from aluminium angle and a 6 mm step.
A box of parts weighing 0.8 kg in the drawer. Mark the drawer's start position on the table with tape.

## Third-party assets and licenses

The SDK is yam-isaac-lab-arena at 0059d02a1a91bc6834c1d0846022891df0c78730, the program's official dependency,
installed by the user's environment. Nothing from it is redistributed here. Licenses are given as each source states
them.

| Asset | Used for | Source | License as stated by the source | Bundled |
|---|---|---|---|---|
| YAM arm and gripper (URDF, meshes, collision hulls, converted USD) | both arms | SDK `yam_arena/assets/yam/`, from i2rt (SOURCE_COMMIT 120c3c81400171174604e503943f8d1ebc891058) | MIT, Copyright (c) I2RT Robotics (SDK `LICENSE.i2rt`) | no |
| link2/link3 meshes split into black and white parts | station arm colours | SDK `yam_arena/assets/yam/dream/`, from the dreamlabs 1.0.23 wheel | "MIT, see ../LICENSE.i2rt" (SDK `dream/SOURCE`) | no |
| Station floor and table | room | SDK `station.py`/`scene.py`: boxes with flat colours generated in code; no mesh or texture file loaded | SDK README: "This repository has no license yet" (Placeholder) | no |
| Top-camera stand | room | SDK `station.py`: boxes and capsules generated in code | as the row above | no |
| Dome light and lamp | lighting | SDK `scene.py`, using Isaac Lab-Arena's DomeLight and an Isaac Lab sphere light; no HDR image | as the row above | no |
| top_cam, left_cam, right_cam | policy RGB | SDK pinhole camera sensors at the calibrated poses; no camera mesh | as the row above | no |
| Station poses, sizes and colours | numbers in the SDK code | SDK README: a replica of Dream Labs' calibrated YAM station (dreamlabs 1.0.23, `yam_mujoco`) | license not stated by the SDK | no |
| Station fruit, bowl, plate (Arena object library) and water bottle | not used: this task's own objects replace them | - | - | no |
| Cabinet, slides, drawer, handle, parts box | task objects | this folder (`task.py`, `drawerkit.py`): boxes generated in code | the authors' own, MIT License (`LICENSE`) | yes |

Isaac Lab-Arena release/0.3.1 (c7edf3a9d5d8276565208f72422f8f8a0806d6cf) ships the Apache License 2.0 in its
LICENSE.md. Isaac Sim and Isaac Lab are runtime dependencies under their own licenses. None of them is redistributed
here. All task-specific code and geometry in this folder is the authors' own and is released under the MIT License
in `LICENSE`.
