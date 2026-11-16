# Open a bottle with a detented safety loop and rocker latch

**Instruction:** Open the bottle and keep it upright on the table.

The active task is `yam/task.py` in the official YAM/Isaac Lab-Arena station.
The free-standing bottle has a hinged safety loop, a spring-return rocker and
a spring-opening lid. Lower the loop, then press the lower rocker while keeping
the bottle upright. The loop and hook constrain the mechanism through contact;
there is no software rule that releases the lid after a policy completes a stage.
`task.json` defines the complete setup, numeric scoring, reset and fidelity limits.

## Verified results

| Controller | Score | Control steps | Simulated time | Wall time including startup |
| --- | --- | --- | --- | --- |
| Scripted YAM contact demo | 1.0 | 405 | 20.25 s | about 6 min |
| Dream-1 beta, continuous session | 0.0 | 2,400 | 120 s | 35.1 min |

Both scores were independently recomputed from evaluator measurements. The
Dream trial used seed0 and `rows_per_request=4`:592requests, one server session,
zero expired responses or retries. The arms approach the bottle but do not
open its loop/lid.851steps had guard modifications (749delta-limit and136clamp
events, with overlaps); remaining interface/model effects are not isolated.
The model and demo videos include the reset observation (2401 and406frames,
respectively, at20fps). They are H.264/yuv420p and fully decode.

These are nominal fixed-layout trials, not a success-rate estimate. The earlier
48-row Dream trial also scored0.0 but restarted the server session at every
chunk boundary; `yam/dream1-initial-verification.json` preserves that result.
The latest model report is `yam/dream1-verification.json`; the nominal scripted
report is `yam/verification.json`. Media IDs are recorded in `task.json` after
upload, separately from source control.

## Environment

Use the official simulation launchable's activated Arena environment. Tested:
Isaac Sim6.1.0.0, Arena0.3.1, YAM revision
`0059d02a1a91bc6834c1d0846022891df0c78730`, one NVIDIA L40S48GB.
`provenance.json` records source revisions and adaptation details.

```bash
# On the official launchable; adjust if its environment is installed elsewhere.
source /data/IsaacLab-Arena/yam.env
cd tasks/detent-latch-bottle
inspect-robots list embodiments
```

Run the following commands from this task directory. This folder contains all
custom runtime, scoring and evaluation helpers; it imports no sibling project
or repository-level script. Do not install `requirements.txt` or `pyproject.toml`
into Arena: those describe the optional, historical MuJoCo/Panda reference.
YAM runs use Arena's existing dependencies and official robot assets.

## Scripted full-score demonstration

Choose a fresh output directory. Only14robot joint/gripper values are commanded;
no bottle pose or bottle joint is set by the controller.

```bash
out_dir="$PWD/artifacts/scripted-check"
mkdir -p "$out_dir"
export BOTTLE_TRACE="$out_dir/measurements.jsonl"
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=yam/task.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction 'Open the bottle and keep it upright on the table.' \
  --scorer task_score --max-steps 600 --seed 0 --store-frames \
  --log-dir "$out_dir"
python yam/analyze_evaluation.py "$out_dir"
```

The demo normally finishes before its600step diagnostic cap, inside the task's
2400step limit. `yam/motion.py` defines a side approach: one arm steadies the
body while the other lowers the safety loop, releases it, and presses the rocker.

## Dream-1 evaluation

Configure Dream credentials privately as documented by the official launchable.
The runner checks presence without printing the key, and refuses existing output
directories. This command makes paid model calls; the offline analyzer does not.

```bash
bash yam/evaluate.sh "$PWD/artifacts/dream-check" 2400 4
```

The task-local runner snapshots source/settings and wraps the installed official
Dream session in this process. Changed/missing server sessions cause a trial
error before that chunk is returned for execution. The completed JSON, trace and
continuity record must agree; process exit0 alone does not establish a valid run.
The wrapper does not change model outputs, physics, images, instruction or SDK
source. Four predicted rows are executed between calls; this increases request
count compared with the default48rows but avoids the observed idle expiry in
this slow1kHz scene. This is not a measured gateway TTL or a guarantee under
other system loads. `validation.exit` is0 only after the runner's checks pass.

The default checkpoint is fixed to `dreamlabs/dream-1-beta`, the evaluated model.
A short continuity probe uses56steps (`32`preroll plus six4-row requests):

```bash
bash yam/evaluate.sh "$PWD/artifacts/continuity-check" 56 4
```

## Scoring and videos

```bash
# Works offline; does not import Isaac or call a model.
python yam/analyze_evaluation.py artifacts/dream-check
python -m detent_bottle.scoring artifacts/dream-check/measurements.jsonl
# Pass the final adhoc_*.json, excluding any *.live.json:
inspect-robots video artifacts/dream-check/FINAL_LOG.json
```

`FINAL_LOG.json` is a placeholder for the generated final filename. The official
video exporter produces one stream per camera. The submitted model/demo each
combine the top and right-wrist streams, with original timing.

At20Hz, each credit threshold must hold for8consecutive ticks: loop>=150deg gives
0.25, lid>15deg gives0.5, lid>60deg gives1.0. The current terminal score counts.
A bottle-axis/world-up dot<=0.90, baseZ<0.7375m, nonfinite robot/bottle state,
missing/nonfinite wrist readings, or wrist reaction force>100N for4consecutive
ticks permanently veto credit. Use the larger wrist norm, including gravity.
`bottle_runtime.measure` translates the YAM tabletop height into the shared
scorer's reference height; the actual YAM fall threshold is10mm below the table.
The circular base's low edge must also remain within10mm of the tabletop, with
its centre X in[-.154,.874]m and Y in[-.614,.614]m. Low-edge height accounts for
base radius36mm and tilt; this is geometric proximity, not a table-force sensor.
Airborne/off-edge states permanently invalidate, even after returning. This
condition was added after recording: all405demo and2400model samples satisfy it;
regrading preserves the official scores and credit times. No dynamics changed.

Sensor static-load validation checked10N/20N increments and unloading recovery;
it does not establish dynamic or hardware calibration.

The evaluator reads bottle geometry and wrist sensors. The policy receives only
the official three RGB views and14robot joint/gripper state/action values.

## Checks and mechanism audit

```bash
python -m unittest discover -s yam -p test_continuity.py -v
(cd yam && python -m unittest -v test_contract)
python -m unittest test_task.ScoringTests test_support -v
# No model calls; attempts the rocker press without lowering the loop.
python yam/audit_press_only.py --out "$PWD/artifacts/press-only-check"
python yam/analyze_evaluation.py artifacts/press-only-check
```

The negative contact probe must be inspected for actual approach/contact as well
as score: a zero from failing to reach the mechanism is not proof of an interlock.
It changes only the scripted robot trajectory, not task geometry or scoring.
`yam/submission-audit.json` records which submission checks were actually done
and their limits. It does not claim an exhaustive robustness or exploit search.

## Distinction from nearby tasks

The task board was checked against accepted, pending and previous-cohort ideas.
The closest medicine-bottle task uses simultaneous axial push and cap rotation;
the bottle-and-cup task permits either an unscrew or a simple flip-top and scores
pouring; the accepted two-latch case uses simultaneous prismatic latch release
and lid removal. Here a free bottle has a sequential hinged safety loop and a
separate spring-return rocker, with passive hook/receiver contacts and no cap
unscrewing or pouring. This is a mechanical difference rather than a color or
layout variation; novelty acceptance remains the reviewers' judgment.

## Fidelity and limitations

The bottle preserves four rigid links, three revolute joints,73colliders and
294source contact pairs. PhysX runs at1kHz with50substeps per20Hz control tick,
64position/8velocity solver iterations and explicit internal contact filtering.
Lid/rocker springs and loop torque are passive. The loop uses an approximate
piecewise detent law plus COM gravity compensation, not a resolved detent cam.
No bottle pose corrections, software latch gates or contact switching run.

Dimensions, inertias, friction and springs are estimates. Native dry joint
friction and full dynamic bias are not reproduced, and PhysX material mixing
differs from the MuJoCo reference. Cosmetic seals/rings/spout surfaces are visual
only. Liquid, deformation, breakage and seal compression are not modeled. The
real-world analogue is an empty room-temperature bottle; autonomous hardware
operation has not been validated. The force ceiling is an evaluator safeguard,
not a measured real bottle opening-force specification.

`detent_bottle/` retains standalone geometry/scoring and the optional MuJoCo
reference. Its scripted score1.0 at714steps is separate from the submitted YAM
result. `task.mujoco-reference.json` preserves its historical metadata. The active
submission entry point remains `yam/task.py`.
