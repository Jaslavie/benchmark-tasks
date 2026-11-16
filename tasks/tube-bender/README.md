# Bend a copper tube to 90 degrees in a lever bender, allowing for springback

Refrigeration mechanics and plumbers bend copper tube by hand. Tubes that look alike spring back by 4 or 12 degrees, seen only once the lever is relaxed, and the tube can only bend further. A tolerance of 3 degrees means that no fixed overbend passes both. So the way through is to pull, relax, read, and pull again.

**Both arms.** The bender stands free on the bench and is light, 0.4 kg, on a plate that rests directly on the bench, with friction 0.4 at rest and 0.3 sliding. Pulling the lever with one arm spins the bender: a run with one arm turned it 138 degrees, moved it 10.5 cm, and bent the tube 23.5 degrees. Pressing down on the bender with one arm while pulling, by a tilted grip or otherwise, was not tried, so this may not force two arms.

## Requirements

- `yam-isaac-lab-arena` with Isaac Lab-Arena 0.3.1 (`source <ARENA_DIR>/yam.env`).
- `mechkit.py` and `demokit.py`, in this folder. Importing `mechkit` hooks `yam_arena.scene` and `yam_arena.inspect_embodiment` without modifying them.

## Run a trial

From this folder. `--seed` sets the tube, through the seed that `inspect-robots run` derives from it and passes to the reset: an odd reset seed springs back 4 degrees, and an even one 12. The default `--seed 0` gives reset seed 1415158438, and so 12; `--seed 2` gives 377265627, and so 4. `demokit.py --seed` passes its seed as it is, and with none the tube springs back 12.

```bash
inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 -E control_hz=20 \
    -E cam_width=320 -E cam_height=176 --policy dream -P rows_per_request=20 --scorer task_score --max-steps 800 \
    --instruction "Bend the tube in the bender to the 90 degree mark on the card, judged with the lever fully relaxed, then take both grippers away."
```

## The demo

```bash
python demokit.py task.py OUT                   # the full-score run, with video: about 20 s of simulated time
python demokit.py task.py OUT --seed 1          # the other tube
python demokit.py task.py OUT --mode one_arm    # one arm pulls, with nothing holding the bender: scores 0
```

It is closed loop on the lever's and the tube's joints and the bender's pose, and reads no scorer state or springback value. A trial of about 400 steps takes about 1 minute on a 32 GB RTX GPU.

`python check_scorer.py` checks every score line on hand-built states, with no simulator.

## Scoring

`--scorer task_score` prints the total.

`score()` returns each line below itself, and `success()` is true at any step that meets the 1.0 line, so the harness's 1 s hold and `score()` agree.

- **1.0:** the tube set from 87 to 93 degrees; the lever tip within 1.5 cm of its start line, in line with the form handle (5 degrees); both grippers 3 cm clear of both handles; all still for 1 s.
- **0.5:** lever tip within 1.5 cm of its start line and both grippers 3 cm clear, all still for 1 s, and the tube reads from 80 to under 87, or over 93 up to 96.
- **0.25:** lever tip within 1.5 cm of its start line and both grippers 3 cm clear, all still for 1 s, and the tube reads from 70 to under 80.
- **0:** otherwise, or if the tube is set past 96 degrees, or the bender leaves the bench or tips over.

"Still" means the lever moves less than 0.5 degrees over any 0.3 s.

## Real trials

- **Tube:** copper (99.9% copper), 2.5 mm outside diameter, 0.3 mm wall, bought in straight lengths in the as-drawn (hard) temper. Cut it 29 cm long and lay it 16 cm on the form handle's side of the wheel's centre and 13 cm on the lever's side. The stiff temper is the tube as drawn, with a 0.2% proof stress of about 300 MPa. The soft one is the same stock annealed: heat each length to a dull red with a torch and let it cool. Clean both kinds with an abrasive pad so that they look alike. The simulation draws the tube 1 cm thick so that it reads from above.
- **Why this size:** springback grows with the bend radius over the tube's diameter, and with the metal's yield stress over its stiffness. Bent to 90 degrees at a 35 mm radius, elastic-plastic bending theory gives this tube 10 to 14 degrees as drawn (proof stress 280 to 340 MPa) and 3 to 5 degrees annealed. A 6 mm copper tube on the same radius springs back only about 5 degrees hard and 2 to 3 annealed, too little for this task.
- **Bender:** a lever bender with a 35 mm bend radius: a form wheel 70 mm across, grooved for 2.5 mm tube, with the form handle and lever each 17 cm from the pivot and rubber grips 15 cm out, on a plate 20 x 34 cm that rests loose on the bench. A white card under the tube, 25 cm across, marked green at 0, a red band from 87 to 93 degrees, and black ticks at 70, 80 and 96.
- **Return spring:** a torsion spring at the pivot of 0.6 N m per radian, at rest with the lever in line with the form handle. Check it with a spring balance at the lever grip, pulling square to the lever: 3.1 N holds the lever at 45 degrees and 6.3 N at 90.
- **Springback:** calibrate each batch before trials. Bend three offcuts on the lever to 90 degrees on the card, relax the lever and read the card: the shortfall is the springback. Use batches whose means are within 1 degree of 12 (as drawn) and 4 (annealed), and more than 6 degrees apart, so that no single overbend puts both tempers in the 6 degree band. Set other batches aside.
- **Load:** the simulation's only load on the lever is the return spring, 6.3 N at the grip at 90 degrees. The tube's own resistance to bending, which the simulation leaves out, adds about 3 N there for the hard tube and 1 N for the annealed one, by the same theory.

## The model behind the bend

The tube is not simulated as a material. With the lever at L degrees, the tube's set angle is the largest value yet of L less the springback. Its drawn angle is the larger of the set angle and L, so the tube follows the lever down to its set angle, and stays there. The lever is on a spring of 0.6 N m per radian with its rest at the start.

## Assets

Every mesh is generated by `task.py` and `mechkit.py`. There are no textures and no third-party assets.

## Licence

MIT
