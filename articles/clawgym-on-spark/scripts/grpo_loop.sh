#!/usr/bin/env bash
# Phase 6 GRPO outer loop for ClawGym-on-Spark.
#
# Architecture (kill-and-restart):
#
#   for step in 1..N:
#     1. Start vLLM (container-side) with the current LoRA adapter.
#     2. Sample M training tasks from the pool.
#     3. Run phase6_smoke.py against vLLM → trajectory_bundle.jsonl.
#     4. Stop vLLM (frees ~108 GB unified mem at GPU_UTIL=0.85).
#     5. Run grpo_train.py → updated adapter at adapter-step-N/.
#     6. (every EVAL_EVERY steps) full eval against the 158 held-out:
#        a. Restart vLLM with the new adapter.
#        b. rollout.py against held-out → trajectories.jsonl.
#        c. compare_phase5.py vs Phase 5 SFT baseline.
#
# WHY kill-restart and not co-residence: vLLM 0.20 in tllm-build does NOT
# expose /v1/load_lora_adapter (verified 2026-05-05 — endpoint 404s,
# openapi.json has zero lora endpoints). So even if vLLM and trainer
# co-reside (gpu-memory-utilization 0.4 was tested + works), we still
# need to restart vLLM each step to load the NEW adapter from disk.
# Co-residence buys nothing on wall in this architecture.
#
# Future optimization path: persistent-trainer (load Qwen ONCE, ingest
# bundles via stdin) + co-resident vLLM with per-step restart. Saves
# ~90 min of trainer-loads on a 50-step run. Not implemented this iter.
#
# Wall (post-dry-run estimate):
#   per step (no eval): vLLM start ~120s + rollouts ~30s + vLLM kill ~5s
#                       + trainer load+step ~130s = ~5 min
#   per eval:           vLLM start ~120s + rollout 158 × ~28s = ~75 min
#   N=50 steps + 2 evals (every 25 steps): 50*5 + 2*75 = 400 min ≈ 6.7 hr
#
# Usage:
#   bash grpo_loop.sh \
#       --pool      .../tasks-grpo-pool.jsonl \
#       --heldout   .../tasks-heldout-158.jsonl \
#       --init      /work/clawgym-sft/adapter-v1 \
#       --out-dir   .../runs/<date>-phase6-grpo/ \
#       --steps     50 \
#       --tasks-per-step 8 \
#       --k         4 \
#       --eval-every 25 \
#       --container tllm-build \
#       --vllm-url  http://172.17.0.3:8000/v1
#
# All paths under --out-dir are host-side; the loop docker-cp's bundles
# in and adapters back. Designed to be Ctrl-C-safe: each step's
# adapter+bundle+log are written before the next step starts, so
# resumption from a step boundary is `--init <last-adapter> --resume-step N+1`.

set -euo pipefail

# ────────────────────────────────────────────────────────────────
# Defaults + arg parsing
# ────────────────────────────────────────────────────────────────
POOL=""
HELDOUT=""
INIT=""
OUT_DIR=""
STEPS=50
TASKS_PER_STEP=8
K=4
EVAL_EVERY=25
CONTAINER="tllm-build"
VLLM_URL="http://172.17.0.3:8000/v1"
TEMPERATURE=0.8
LR=1e-5
KL_BETA=0.04
RESUME_STEP=1
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_VENV="/tmp/fk-clawgym/bin/python3"
VLLM_GPU_UTIL=0.85  # vLLM only — we kill it before trainer runs

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pool) POOL="$2"; shift 2 ;;
        --heldout) HELDOUT="$2"; shift 2 ;;
        --init) INIT="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --steps) STEPS="$2"; shift 2 ;;
        --tasks-per-step) TASKS_PER_STEP="$2"; shift 2 ;;
        --k) K="$2"; shift 2 ;;
        --eval-every) EVAL_EVERY="$2"; shift 2 ;;
        --container) CONTAINER="$2"; shift 2 ;;
        --vllm-url) VLLM_URL="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --kl-beta) KL_BETA="$2"; shift 2 ;;
        --resume-step) RESUME_STEP="$2"; shift 2 ;;
        *) echo "unknown flag $1" >&2; exit 2 ;;
    esac
done

for var in POOL HELDOUT INIT OUT_DIR; do
    if [[ -z "${!var}" ]]; then echo "missing --${var,,}" >&2; exit 2; fi
done
mkdir -p "$OUT_DIR"

LOG="$OUT_DIR/loop.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== grpo_loop.sh started $(date '+%F %T') ==="
echo "  POOL=$POOL"
echo "  HELDOUT=$HELDOUT"
echo "  INIT=$INIT"
echo "  OUT_DIR=$OUT_DIR"
echo "  STEPS=$STEPS  TASKS_PER_STEP=$TASKS_PER_STEP  K=$K  EVAL_EVERY=$EVAL_EVERY"
echo "  TEMPERATURE=$TEMPERATURE  LR=$LR  KL_BETA=$KL_BETA"
echo "  RESUME_STEP=$RESUME_STEP"

# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────

container_path() {
    # Convert host adapter-vN path to container-side /work/clawgym-grpo/...
    # Assumes adapter is staged into container under /work/clawgym-grpo/.
    echo "/work/clawgym-grpo/$(basename "$1")"
}

start_vllm() {
    local adapter_path="$1"
    local container_adapter
    container_adapter="$(container_path "$adapter_path")"
    echo "[vllm] starting with adapter $container_adapter"
    docker exec -d "$CONTAINER" bash -c "
        cd /work && python3 -m vllm.entrypoints.openai.api_server \
            --model Qwen/Qwen2.5-7B-Instruct --port 8000 --max-model-len 8192 \
            --gpu-memory-utilization $VLLM_GPU_UTIL --enable-lora \
            --lora-modules clawgym=$container_adapter --max-lora-rank 16 \
            > /work/clawgym-sft/vllm-serve.log 2>&1
    "
    local t0; t0=$(date +%s)
    until curl -sf --max-time 3 "$VLLM_URL/models" 2>/dev/null \
            | grep -q '"id":"clawgym"'; do
        if (( $(date +%s) - t0 > 360 )); then
            echo "[vllm] FAILED to come up in 6 min" >&2; return 1
        fi
        sleep 5
    done
    echo "[vllm] ready after $(( $(date +%s) - t0 ))s"
}

stop_vllm() {
    echo "[vllm] stopping"
    docker exec "$CONTAINER" bash -c "pkill -9 -f 'vllm|EngineCore' 2>&1; \
        sleep 2; pkill -9 -f 'multiprocessing.resource_tracker' 2>&1 || true" || true
    sleep 3
    free -h | head -2 | tail -1
}

stage_adapter() {
    local adapter_path="$1"
    local container_adapter
    container_adapter="$(container_path "$adapter_path")"
    docker exec "$CONTAINER" bash -c "rm -rf $container_adapter"
    docker cp "$adapter_path" "$CONTAINER:$container_adapter"
}

sample_task_ids() {
    local n="$1"
    "$HOST_VENV" -c "
import json, random
random.seed($((RANDOM)))
tasks=[]
with open('$POOL') as f:
    for line in f:
        t=json.loads(line); tasks.append(t['task_id'])
random.shuffle(tasks)
print(','.join(tasks[:$n]))
"
}

run_step() {
    local step="$1"
    local current_adapter="$2"
    local step_dir="$OUT_DIR/step-$(printf '%03d' "$step")"
    mkdir -p "$step_dir"
    echo ""
    echo "=== STEP $step  $(date '+%F %T') ==="
    echo "  current_adapter=$current_adapter"

    # 1. Stage adapter into container; start vLLM
    stage_adapter "$current_adapter"
    start_vllm "$current_adapter"

    # 2. Sample tasks + run rollouts → bundle
    local task_ids
    task_ids=$(sample_task_ids "$TASKS_PER_STEP")
    echo "[step $step] sampled tasks: $task_ids"
    "$HOST_VENV" "$SCRIPTS_DIR/phase6_smoke.py" \
        --tasks "$POOL" \
        --task-ids "$task_ids" \
        --k "$K" \
        --temperature "$TEMPERATURE" \
        --vllm-base-url "$VLLM_URL" \
        --model clawgym \
        --out-dir "$step_dir/" \
        | tee "$step_dir/rollout.log"

    # 3. Stop vLLM (free unified memory for trainer)
    stop_vllm

    # 4. Trainer step → save adapter
    docker cp "$step_dir/trajectory_bundle.jsonl" "$CONTAINER:/work/clawgym-grpo/dryrun/"
    # KL reference is FIXED to the SFT-init ($INIT) for every step — classic
    # GRPO fixed-SFT-init reference. The reference adapter is staged once
    # at /work/clawgym-grpo/_reference_adapter/ before the loop starts.
    docker exec "$CONTAINER" bash -c "
        cd /work/clawgym-grpo && python3 grpo_train.py \
            --bundle dryrun/trajectory_bundle.jsonl \
            --tasks-pool tasks-grpo-pool.jsonl \
            --adapter-init $(container_path "$current_adapter") \
            --reference-adapter /work/clawgym-grpo/_reference_adapter \
            --out-dir /work/clawgym-grpo/adapter-step-$(printf '%03d' "$step")/ \
            --base-model Qwen/Qwen2.5-7B-Instruct \
            --lr $LR --kl-beta $KL_BETA --check-weight-delta
    " 2>&1 | tee "$step_dir/trainer.log"

    # 5. Pull adapter back to host
    docker cp "$CONTAINER:/work/clawgym-grpo/adapter-step-$(printf '%03d' "$step")/" \
        "$step_dir/adapter/"

    echo "$step_dir/adapter" > "$OUT_DIR/.current_adapter"

    # 6. Optional eval
    if (( step % EVAL_EVERY == 0 )); then
        run_eval "$step" "$step_dir/adapter"
    fi
}

run_eval() {
    local step="$1"
    local adapter="$2"
    local eval_dir="$OUT_DIR/eval-step-$(printf '%03d' "$step")"
    mkdir -p "$eval_dir"
    echo ""
    echo "=== EVAL @ STEP $step  $(date '+%F %T') ==="
    stage_adapter "$adapter"
    start_vllm "$adapter"
    "$HOST_VENV" "$SCRIPTS_DIR/rollout.py" \
        --tasks "$HELDOUT" \
        --out-dir "$eval_dir/" \
        --max-turns 12 \
        --nim-base-url "$VLLM_URL" \
        --model clawgym \
        | tee "$eval_dir/rollout.log"
    stop_vllm
    "$HOST_VENV" "$SCRIPTS_DIR/compare_phase5.py" \
        --base "$OUT_DIR/../2026-05-04-phase5-eval/qwen-base/trajectories.jsonl" \
        --sft  "$eval_dir/trajectories.jsonl" \
        --out-json "$eval_dir/comparison.json" \
        | tee "$eval_dir/compare.log"
}

# ────────────────────────────────────────────────────────────────
# Main loop
# ────────────────────────────────────────────────────────────────

current_adapter="$INIT"
if [[ "$RESUME_STEP" -gt 1 ]]; then
    prior_step=$((RESUME_STEP - 1))
    current_adapter="$OUT_DIR/step-$(printf '%03d' "$prior_step")/adapter"
    echo "Resuming from step $RESUME_STEP using $current_adapter"
fi

# One-time staging of the tasks pool + frozen SFT-init reference adapter
docker cp "$POOL" "$CONTAINER:/work/clawgym-grpo/tasks-grpo-pool.jsonl"
docker exec "$CONTAINER" bash -c "rm -rf /work/clawgym-grpo/_reference_adapter && \
    mkdir -p /work/clawgym-grpo/_reference_adapter"
docker cp "$INIT/." "$CONTAINER:/work/clawgym-grpo/_reference_adapter/"
echo "Staged frozen reference adapter (SFT-init = $INIT) → /work/clawgym-grpo/_reference_adapter/"

t_loop=$(date +%s)
for step in $(seq "$RESUME_STEP" "$STEPS"); do
    run_step "$step" "$current_adapter"
    current_adapter="$OUT_DIR/step-$(printf '%03d' "$step")/adapter"
done

echo ""
echo "=== loop complete in $(( $(date +%s) - t_loop ))s ==="
echo "final adapter: $current_adapter"
