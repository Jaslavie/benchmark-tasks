#!/usr/bin/env bash
# Dream-1 evaluation in the activated official Arena/YAM environment.
set -euo pipefail
task_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
run_root=${1:-$task_root/artifacts/dream1-bottle-$(date -u +%Y%m%dT%H%M%SZ)}
max_steps=${2:-2400}
rows_per_request=${3:-4}
if [[ ! -s "$HOME/.dreamlabs/credentials" && -z "${DREAM_API_KEY:-}" ]]; then
    printf 'Dream credentials are not configured. No evaluation started.\n' >&2
    exit 2
fi
if [[ -e "$run_root" ]]; then
    printf 'Refusing to overwrite an existing evaluation directory.\n' >&2
    exit 2
fi
mkdir -p "$run_root/source/yam" "$run_root/source/detent_bottle"
cp "$task_root"/yam/*.py "$run_root/source/yam/"
cp "$task_root"/detent_bottle/*.py "$run_root/source/detent_bottle/"
cp "$task_root/task.json" "$run_root/source/"
cp "$task_root/yam/dream_continuity.py" "$run_root/source/"
cp "${BASH_SOURCE[0]}" "$run_root/source/runner.sh"
cp "$task_root/yam/analyze_evaluation.py" "$run_root/source/"
python - "$run_root" "$max_steps" "$rows_per_request" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1])
manifest={str(p.relative_to(root/'source')):hashlib.sha256(p.read_bytes()).hexdigest()
          for p in (root/'source').rglob('*') if p.is_file()}
(root/'source-sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
(root/'registration.json').write_text(json.dumps(dict(
    policy='dream',model='dreamlabs/dream-1-beta',max_steps=int(sys.argv[2]),rows_per_request=int(sys.argv[3]),
    continuity_gate=True,control_hz=20,
    object_jitter=0,seed=0,instruction='Open the bottle and keep it upright on the table.',
    scorer='task_score',epochs=1,scripted_reference_score=1.0),indent=2)+'\n')
PY
export BOTTLE_TRACE="$run_root/measurements.jsonl"
export BOTTLE_MODEL_REPORT="$run_root/model.json"
printf 'Dream-1 run directory: %s\n' "$run_root"
set +e
timeout --signal=TERM --kill-after=30 5400 python "$run_root/source/dream_continuity.py" \
    --continuity-log "$run_root/continuity.jsonl" run \
    --embodiment yam_arena --policy dream -P model=dreamlabs/dream-1-beta -P rows_per_request="$rows_per_request" \
    -E task="$task_root/yam/task.py" -E object_jitter=0 -E control_hz=20 \
    -E cam_width=320 -E cam_height=176 \
    --instruction 'Open the bottle and keep it upright on the table.' \
    --scorer task_score --max-steps "$max_steps" --seed 0 --store-frames \
    --log-dir "$run_root" > "$run_root/run.log" 2>&1
trial_exit=$?
set -e
printf '%s\n' "$trial_exit" > "$run_root/run.exit"
printf 'Inspect Robots exited with code%s; inspect the final JSON before accepting a result.\n' "$trial_exit"
if [[ "$trial_exit" != 0 ]]; then exit "$trial_exit"; fi
python "$task_root/yam/analyze_evaluation.py" "$run_root" > "$run_root/analysis.stdout.json"
set +e
python - "$run_root/analysis.json" "$max_steps" <<'PYEND'
import json,sys
report=json.load(open(sys.argv[1]))
if not report['valid_completed_trial'] or not report['continuous_model_evaluation']:
    raise SystemExit('Trial failed completion or session continuity checks')
if int(sys.argv[2]) < 2400 and report['dream_request_count'] < 6:
    raise SystemExit('Preflight requires at least six requests')
PYEND
validation_exit=$?
set -e
printf '%s\n' "$validation_exit" > "$run_root/validation.exit"
exit "$validation_exit"
