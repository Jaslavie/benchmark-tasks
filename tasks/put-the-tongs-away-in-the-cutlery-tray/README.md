# Lock the spring tongs and put them away in the cutlery tray

A pair of 22 cm stainless spring tongs with red silicone heads lies open on a kitchen counter. A fixed cutlery drawer
tray stands beyond them. Its only compartment long enough for the tongs is 65 mm wide inside, and the open tongs are
94 mm across their heads. The tongs have a pull-tab lock at the hinge end. The tab can only be pulled out to lock them
while they are squeezed shut. The robot is asked to put the tongs away in the tray, locked shut.

The task is `task.py` for the `yam_arena` embodiment of yam-isaac-lab-arena (the program's YAM embodiment for Isaac
Lab-Arena), on the `station` scene. It needs `parts.py` (jointed multi-shape parts) and `lots.py` (layout lots), both
in this folder; they import only NumPy and `yam_arena`. `near_misses.py` lists near-successes and wrong outcomes with
the score each must get. They are plain-Python checks of `score()` and `success()` and need no simulator.

## Layout

One fixed start layout (the screenshot): the hinge end 10 cm to the robot's right, the tongs pointing left. The
`lots.py` helper is used only for its fresh settle before the first frame, so the tongs have sprung open when the
trial starts. A second layout with the tongs turned end for end was tried and dropped: moving the jointed tongs per
trial did not place them reliably in simulation.

## Run a trial

From this folder on a machine set up by the yam-isaac-lab-arena launchable:

```bash
I="Put the tongs away in the cutlery tray, locked shut."

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=$PWD/task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$I" --scorer task_score --max-steps 1200 --store-frames --log-dir ./logs/tongs-dream

# Scripted demo
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=$PWD/task.py -E object_jitter=0 -E control_hz=20 -E cam_width=640 -E cam_height=360 \
  --instruction "$I" --scorer task_score --max-steps 600 --store-frames --log-dir ./logs/tongs-demo
```

The run prints `task_score`. Scoring is `score()` and `success()` in `task.py`, described
in words in `task.json`.

- **Runtime per trial:** at most 60 s of simulated time (1,200 steps at 20 Hz), about 1 to 2 minutes of wall time
  on an L40S once Isaac Sim has started.
- **Seeds:** none needed. Run with `-E object_jitter=0`.

## The tongs and the lock in the simulator

- The tongs are one loose jointed part. The body is the hinge knuckle. Each arm (strip plus head) is a link on its
  own vertical hinge pin, limited to 0 (shut) to 10 degrees (open stop). The pull tab is a third link that slides
  0 to 10 mm out of the hinge end.
- The shapes are written shut. At every reset an `on_reset` callback points both hinge springs at 0.20 rad past
  the open stop (0.8 N m/rad each), so the tongs spring open against their stops before the first frame. With the
  tongs shut, that is about 0.3 N m per arm, or about 1.9 N needed at the squeeze point 160 mm from the hinge.
- **The lock:** an `on_step` callback watches the joints. When the tab is out at least 7 mm and both arms are within
  0.03 rad of shut, the lock catches. Both hinge springs then pull toward shut, and the tab's spring holds it out.
  When the tab is pushed back to under 3 mm, the lock lets go and the springs open the tongs again. Pulled out while
  the tongs are open, the tab springs back in. Scoring reads only the tab and arm positions, which a person can see
  and measure.
- The two arms do not collide with each other (two moving links of one part do not collide in this stack). Their
  shut stop is the hinge limit.

## Scoring details the simulator applies

- **Let go:** the `on_step` callback reads both arms' finger and gripper link poses every step and measures the
  shortest distance from their collision hulls (`yam_arena.analysis`) to every shape of the tongs. It must exceed
  5 mm. Scoring fails closed: in the simulator, a step where this could not be measured counts as held, and earns
  nothing. Offline (`near_misses.py`, no simulator loaded) there is no robot, so it counts as clear.
- **Still:** the tongs move under 1 cm/s and turn under 0.5 rad/s (`ObjectState.resting()`). Success must hold 2 s.
- **In the long compartment:** every corner of every shape of the tongs (knuckle, both arms and heads, tab; the
  knuckle cylinder taken as its box) lies within the compartment's inner faces, up to 1 mm into a face for contact.
  None is above the rim (50 mm up), and the lowest corner is within 3 mm above the floor (4 mm up).
- **Locked:** tab at least 7 mm out and each arm at most 0.03 rad from shut.
- **On the table:** the tongs' lowest corner is at most 5 mm below the counter. Tongs dropped off the table score 0.
- Tiers: 1.0 locked and in the compartment; 0.5 locked anywhere else on the table; 0.25 in the compartment, not
  locked; 0 otherwise. Every tier requires let go, still and on the table.

## Scripted demo

`demo()` uses both arms. The arm on the heads' side comes down over the open arms 160 mm from the hinge, with its
fingers closing front to back across the tongs, and squeezes them shut (pads commanded 4 mm past the shut pair's
faces). The other arm comes down on the pull tab, pinches it and pulls it 13 mm along the tongs (3 mm past its travel;
the arm gives a little), lets go and moves away. The first arm lifts the locked tongs, carries them over the long
compartment's centre, lowers them to 6 mm above its floor, opens 24 mm (clear of the walls), rises and waits. The demo
plays open-loop from the published start layout and reads nothing from the simulator.

## Other ways to succeed, and what scores 0

- **Also full score:** either arm doing either job, any grip that holds the tongs shut while the tab is pulled,
  the tongs pointing either way in the compartment.
- **0.5:** locking the tongs and leaving them on the counter, on the tray's rim or across the tray.
- **0.25:** squeezing the tongs, lowering them into the long compartment and letting go without locking them (they
  spring open against its walls).
- **0:** open tongs laid across the tray (they are too wide for the long compartment and too long for the short
  ones), the tab pulled while the tongs are open (it springs back), tongs still held or moving, tongs off the table.

The cases in `near_misses.py` cover the 7 mm tab and 0.03 rad arm limits on both sides, the 3 mm floor limit, the
compartment's end wall with 0.5 mm to spare and 2 mm over, turns of 8 and 12 degrees in the compartment, one end on the
rim, tongs sprung open against the walls, open tongs on the walls, locked tongs on the counter, moving tongs, and tongs
dropped off the table.

## Recorded runs

- Scripted demo: 1.0. The squeeze shut the arms, the pulled tab caught the lock, and the
  locked tongs were let go in the long compartment and stayed still.
- Dream-1, 1,200 steps: 0.0.
