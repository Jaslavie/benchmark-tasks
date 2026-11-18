#!/bin/bash
# Run one trial of the IM injection task on one GPU, without touching anyone else's work.
# usage: run_injection.sh NAME MAX_STEPS scripted|dream [DEMO_MODE] [GPU] [ENV=VAL ...]
#   DEMO_MODE (scripted only; ignored for dream): full (default), spike, steep
#   ENV=VAL: e.g. INJ_HEIGHT=0|1|2 (spacer: none, one, two 18.3 mm layers), INJ_TRAINER_T=0.050|0.043 (m)
#   EPOCHS=n: n trials in one process (the scene is reset between them)
#   GPU: the index to run on (default 0). Refuses to start unless it has NEED_MB (default 24000) MiB free.
#   This script never kills a process.
# Paths from the environment: ARENA_DIR (the IsaacLab-Arena checkout, default /workspace/$USER/IsaacLab-Arena) and
# YAM_EVAL_DIR (logs, default /workspace/$USER/yam-eval).
# Logs: $YAM_EVAL_DIR/NAME (inspect log and frames) and NAME.log (stdout).
set -u
NAME=${1:?name}; MAX=${2:?max steps}; POLICY=${3:?scripted or dream}; MODE=${4:-full}; GPU=${5:-0}
shift 3; [ $# -gt 0 ] && shift; [ $# -gt 0 ] && shift
case "$GPU" in ''|*[!0-9]*) echo "GPU must be an index, not $GPU" >&2; exit 2;; esac
case "$POLICY" in scripted|dream) ;; *) echo "policy must be scripted or dream, not $POLICY" >&2; exit 2;; esac

# Share the GPU only if there is room: never kill or crowd anyone else's process.
FREE=$(nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits 2>&1)
if [ $? -ne 0 ]; then echo "nvidia-smi failed on GPU $GPU: $FREE" >&2; exit 3; fi
NEED_MB=${NEED_MB:-24000}
if [ "$FREE" -lt "$NEED_MB" ]; then
  echo "GPU $GPU has only ${FREE} MiB free (< ${NEED_MB}); not starting." >&2; exit 4
fi

HERE=$(cd "$(dirname "$0")/.." && pwd)
ARENA=${ARENA_DIR:-/workspace/$USER/IsaacLab-Arena}
L=${YAM_EVAL_DIR:-/workspace/$USER/yam-eval}
mkdir -p "$L"
source "$ARENA/yam.env" >/dev/null 2>&1
export CUDA_VISIBLE_DEVICES=$GPU       # this GPU only; inside the process it is cuda:0
export TASK_DEMO="$MODE" INJ_TRACE=1
for kv in "$@"; do export "$kv"; done
INSTR="Give the 1 mL dose in the syringe as an intramuscular injection in the training arm's deltoid, pushing the plunger slowly and steadily. Push it all the way so the needle retracts, then put the syringe in the sharps container and the needle cap in the waste cup."
COMMON=(--embodiment yam_arena -E task="$HERE/im_injection.py" -E device=cuda:0 -E object_jitter=0 -E control_hz=20
        -E cam_width=320 -E cam_height=176 --instruction "$INSTR" --scorer task_score --max-steps "$MAX" --seed 0
        --no-prompt --store-frames --log-dir "$L/$NAME" ${EPOCHS:+--epochs "$EPOCHS"})
cd "$HERE"
if [ "$POLICY" = scripted ]; then
  inspect-robots run --policy yam_scripted "${COMMON[@]}" > "$L/$NAME.log" 2>&1
  echo "exit $?" >> "$L/$NAME.log"
else
  "$ARENA/.venv/bin/inspect-robots-dream-bridge" --warm --port 8766 > "$L/$NAME-bridge.log" 2>&1 &
  BR=$!
  trap 'kill $BR 2>/dev/null' EXIT
  for i in $(seq 1 360); do
    grep -q "listening" "$L/$NAME-bridge.log" && break
    kill -0 $BR 2>/dev/null || { echo "bridge died" >> "$L/$NAME.log"; exit 1; }
    sleep 10
  done
  inspect-robots run --policy dream -P server_url=http://127.0.0.1:8766 "${COMMON[@]}" > "$L/$NAME.log" 2>&1
  echo "exit $?" >> "$L/$NAME.log"
fi
