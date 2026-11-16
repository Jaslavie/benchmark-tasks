# Drag a file into the green folder on a computer screen with a real mouse

A robot uses a computer through the interface built for people: it drags the red file on the screen into the green folder with an ordinary mouse.

What makes it hard is the hand, not the choice. Anyone can see which folder is green. But the mouse is a relative, high-gain pointer:

- **Precision.** 1 mm of mouse travel moves the pointer 31.5 pixels. A folder's icon is 150 x 128 px, so the drop must land within about 2 mm of mouse travel either way: 2.4 mm across the folder, 1.9 to 2.2 mm up and down.
- **The held button.** The file follows the pointer only while the left button stays down. If the press slips off or the finger lifts, the file drops where it is, and a second press within 0.5 s is a double click, which opens what is under the pointer.
- **The flat sensor.** The sensor under the mouse tracks only while it is within 1.5 mm of the mat. Tilt or lift the mouse and the pointer stops.
- **Its own axes.** The sensor reads motion in the mouse's own frame, so a mouse turned on the mat moves the pointer at a slant.

The nearest task elsewhere is the DexJoCo suite's 'Click Mouse', which sets a mouse on its pad and clicks it; here the button is held through a drag, and the file is landed on a folder within about 2 mm of mouse travel by watching the on-screen pointer.

The folders' order, the file's start, the pointer's start and the mouse's place and turn on the mat change with the seed.

**Both arms.** One arm is enough: a fingertip can press the button and drag the mouse by friction. The demo uses only the right arm. Using two arms, one on the body and one on the button, is allowed.

## Requirements

- `yam-isaac-lab-arena` with Isaac Lab-Arena 0.3.1 (`source <ARENA_DIR>/yam.env`).
- `mechkit.py` and `demokit.py`, in this folder. Importing `mechkit` hooks `yam_arena.scene` and `yam_arena.inspect_embodiment` without modifying them.

## Run a trial

From this folder:

```bash
inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 -E control_hz=20 \
    -E cam_width=320 -E cam_height=176 --policy dream -P rows_per_request=20 --scorer task_score --max-steps 1200 --seed 1 \
    --instruction "Using the mouse, drag the red file into the green folder on the screen, then let go of the mouse."
```

`--seed` sets the layout through the seed that `inspect-robots run` derives from it and passes to the reset: `--seed 1` gives reset seed 2566101560, the scene in the screenshot and the demo video. With no `--seed`, the reset seed is 2001288867. `demokit.py --seed` passes its seed to the reset as it is.

## The demo

```bash
python demokit.py task.py OUT --seed 2566101560     # the full-score run on the scene of --seed 1, with video
python demokit.py task.py OUT --seed 2001288867     # the scene of a run with no --seed
python check_scorer.py                              # every score line on hand-built states, with no simulator
```

The demo is a scripted reference, not a policy. It reads the screen through the top camera: it samples the camera's picture where a grid of screen pixels shows, and finds the pointer (a small white arrow, its tip the white's upper-left extreme), the red file and the green folder by colour. It reads the mouse's pose, which is in view, from the simulator. A policy gets only the cameras. It never reads the seed, the button's joint or the scorer.

One arm does it all. It pushes the level top of the palm hump to bring the pointer onto the file with the button up. It then presses the left button, starts the drag, and checks in the picture that the file moved with the pointer, pressing again if not. It drags by the button until the pointer is on the green folder, and lifts off. It scored 1.0 in GPU physics, reading the screen through the camera, on reset seeds 2566101560 (the scene of `--seed 1`, 13.0 s of simulated time, the demo video), 377265627 (`--seed 2`, 14.6 s), 2001288867 (no `--seed`, 12.0 s) and 1 (12.9 s). A 1,200-step trial takes about 2 minutes on a 32 GB RTX GPU.

## Scoring

`--scorer task_score` prints the total. `score()` returns each line below itself, and `success()` is true at any step that meets the 1.0 line's end state, so the harness's 1 s hold and `score()` agree.

- **1.0:** the file is in the green folder, the button is up, every finger is at least 1 cm clear of the mouse, and the mouse is on its mat and upright (tilted 30 degrees or less), held for 1 s.
- **0.5:** the file is in the green folder, but that end state is not held for 1 s.
- **0.25:** the file has been let go on the desktop, and its centre is at most half as far, in a straight line, from the green folder's centre as at the start.
- **0:** otherwise.
- **0 at any moment, and for the rest of the trial:** the file dropped into another folder; a folder dragged; a folder or the file opened by a double click; the mouse leaving its mat (its centre goes past the mat's edge, or it tilts over 30 degrees; lifting it level over the mat is allowed); a finger touching the monitor.

The file goes into a folder when the button comes up while the file is being dragged (it has moved 6 px or more with the button held on it) and the pointer's tip is on that folder's icon, 150 x 128 px. A single click on a folder does nothing. Dropped elsewhere, the file stays where it was let go, and can be dragged again. Records: a screen recording and a video of the table and hands, in sync, or one video that shows the screen, the mat and the hands.

## Real trials: the build

- **Computer and monitor.** Any computer with a 24-inch 1920 x 1080 monitor, scaling 100%. Stand the monitor at the back of the table, its picture's bottom edge 57 cm in front of the arm bases, centred between them, 4.5 cm above the table, and leaned back 20 degrees.
- **Desktop.** A plain dark blue wallpaper. Four folders, coloured blue, green, yellow and purple with the operating system's folder-colour option, for example on a current macOS. Desktop icons at 150 px, the pointer at a large accessibility size, about 145 px tall, and a red document file. Place the icons at the pixel positions that `layout_for(seed)` in `task.py` gives, by hand or with a short script. Turn automatic arranging off, so that a dropped file stays where it lands: on macOS, set Sort By to None and turn Snap to Grid off. Turn spring-loaded folders off, so that a folder does not open while a file is held over it.
- **Pointer speed.** A 1,600 cpi optical mouse at a 0.5 pointer multiplier, with pointer acceleration off: 31.5 px per mm. Calibrate it: with the pointer near the screen's left edge, slide the mouse 50 mm to the right along a ruler on the mat; the pointer must move 1,575 px. Adjust the speed setting until it does.
- **Mouse.** A wireless optical mouse about 116 x 62 x 38 mm and 85 g, matte plastic, on PTFE feet. Its left button should click at about 0.6 N: a 65 g weight (0.64 N) set on the middle of the button clicks it, and 50 g (0.49 N) does not.
- **Mat.** A cloth-topped mouse mat, 24 cm along x by 26 cm along y and 3 mm thick on a rubber base, centred 34 cm forward and 13 cm right of the midline between the arm bases, and taped to the table along its edges. Check the feet's friction: tilt the mat with the mouse on it; the mouse should start to slide at about 8.5 degrees (friction 0.15).
- **Logging.** Record the screen, and film the table and the hands from above and to the side, in sync with the screen recording (or film both in one view). The screen recording shows each drop, any folder that moves and any window that opens. The table video shows the fingers clearing the mouse, whether the mouse stays on its mat and level, and any touch on the monitor.

## The model behind the screen

The screen is drawn by code, as the computer would draw it. Each control step, the sensor's motion is taken in the mouse's own axes: forward moves the pointer up and left moves it left, at 31.5 px per mm. It counts only while the sensor is within 1.5 mm of the mat, and the pointer stops at the screen's edges. The left button is a sprung shell on a slide, 1,500 N/m, with a 0.6 mm stop. Its switch closes at 0.4 mm (0.6 N) and opens again when the press falls below 0.3 mm. Friction is 0.15 between the feet and the mat, and 0.6 for a fingertip on the shell. A selected file shows a highlight behind it, and a folder under a dragged file shows one too. There is no mouse cable.

## Assets

Every mesh is generated by `task.py` and `mechkit.py`. There are no textures and no third-party assets.

## Licence

MIT
