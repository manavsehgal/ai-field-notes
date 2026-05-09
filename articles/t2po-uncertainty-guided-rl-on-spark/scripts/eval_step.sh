#!/usr/bin/env bash
# eval_step.sh — one-off eval of a specific T²PO step's adapter against the held-out 158.
#
# Usage:
#   eval_step.sh --run-dir <2026-05-08-t2po-grpo path> --step 45 \
#                --heldout <tasks-heldout-158.jsonl> \
#                --eval-base-traj <qwen-base trajectories.jsonl>
#
# Mirrors the run_eval() function in t2po_loop.sh but for a single named step,
# so we can eval the strongest training-side checkpoint after a run completes
# (or any prior step) without touching the loop's main schedule.

set -euo pipefail

CONTAINER="tllm-build"
VLLM_URL="http://172.17.0.3:8000/v1"
VLLM_GPU_UTIL=0.85
HOST_VENV="/tmp/fk-t2po/bin/python3"
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_DIR=""
STEP=""
HELDOUT=""
EVAL_BASE_TRAJ=""

while (( $# )); do
    case "$1" in
        --run-dir) RUN_DIR="$2"; shift 2 ;;
        --step) STEP="$2"; shift 2 ;;
        --heldout) HELDOUT="$2"; shift 2 ;;
        --eval-base-traj) EVAL_BASE_TRAJ="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

[[ -z "$RUN_DIR" || -z "$STEP" || -z "$HELDOUT" ]] && {
    echo "missing required args" >&2; exit 2; }

step_padded=$(printf '%03d' "$STEP")
adapter="$RUN_DIR/step-$step_padded/adapter"
eval_dir="$RUN_DIR/eval-step-$step_padded"
[[ ! -d "$adapter" ]] && { echo "adapter not found: $adapter" >&2; exit 1; }
mkdir -p "$eval_dir"

container_adapter="/work/t2po-run/eval-step-$step_padded-adapter"

echo ""
echo "=== EVAL @ STEP $STEP  $(date '+%F %T') ==="
echo "  adapter: $adapter"
echo "  eval_dir: $eval_dir"

# Stage adapter into container
docker exec "$CONTAINER" bash -c "rm -rf $container_adapter"
docker cp "$adapter" "$CONTAINER:$container_adapter"

# Start vLLM with this adapter
echo "[vllm] starting with adapter $container_adapter"
docker exec -d "$CONTAINER" bash -c "
    cd /work && python3 -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen2.5-7B-Instruct --port 8000 --max-model-len 8192 \
        --gpu-memory-utilization $VLLM_GPU_UTIL --enable-lora \
        --lora-modules clawgym=$container_adapter --max-lora-rank 16 \
        > /work/clawgym-sft/vllm-serve.log 2>&1
"
t0=$(date +%s)
until curl -sf --max-time 3 "$VLLM_URL/models" 2>/dev/null \
        | grep -q '"id":"clawgym"'; do
    if (( $(date +%s) - t0 > 360 )); then
        echo "[vllm] FAILED to come up in 6 min" >&2
        exit 1
    fi
    sleep 5
done
echo "[vllm] ready after $(( $(date +%s) - t0 ))s"

# Roll out 158 held-out
"$HOST_VENV" "$SCRIPTS_DIR/rollout.py" \
    --tasks "$HELDOUT" \
    --out-dir "$eval_dir/" \
    --max-turns 12 \
    --nim-base-url "$VLLM_URL" \
    --model clawgym \
    | tee "$eval_dir/rollout.log"

# Stop vLLM
echo "[vllm] stopping"
docker exec "$CONTAINER" bash -c "pkill -9 -f 'vllm|EngineCore' 2>&1; \
    sleep 2; pkill -9 -f 'multiprocessing.resource_tracker' 2>&1 || true" || true
sleep 3

# Compare against Qwen-base baseline
if [[ -n "$EVAL_BASE_TRAJ" && -f "$EVAL_BASE_TRAJ" ]]; then
    "$HOST_VENV" "$SCRIPTS_DIR/compare_phase5.py" \
        --base "$EVAL_BASE_TRAJ" \
        --sft  "$eval_dir/trajectories.jsonl" \
        --out-json "$eval_dir/comparison.json" \
        | tee "$eval_dir/compare.log" || true
fi

# Cleanup container-side staged adapter
docker exec "$CONTAINER" bash -c "rm -rf $container_adapter" || true

echo "=== EVAL @ STEP $STEP done $(date '+%F %T') ==="
