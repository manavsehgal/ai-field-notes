#!/usr/bin/env bash
# Phase 5 pipeline driver — runs after rollout #1 (Qwen base) finishes.
# 1. Launches rollout #2 (Qwen + clawgym LoRA) on the held-out 158.
# 2. Runs the comparison + writes NOTES skeleton.
set -euo pipefail

RUN=/home/nvidia/ai-field-notes/articles/clawgym-on-spark/evidence/runs/2026-05-04-phase5-eval
SCRIPTS=/home/nvidia/ai-field-notes/articles/clawgym-on-spark/scripts
PY=/tmp/fk-clawgym/bin/python3
NIM_BASE=http://172.17.0.3:8000/v1

# --- guard: rollout #1 must exist and be 158 lines ---
TRAJ_BASE=$RUN/qwen-base/trajectories.jsonl
N=$(wc -l < $TRAJ_BASE)
if [ "$N" -lt 158 ]; then
    echo "ABORT: rollout #1 trajectories=$N (<158)"
    exit 1
fi
echo "rollout #1 OK: $N trajectories"

# --- launch rollout #2: Qwen + clawgym adapter ---
echo "launching rollout #2..."
cd $SCRIPTS
nohup $PY rollout.py \
    --tasks $RUN/tasks-heldout-158.jsonl \
    --out-dir $RUN/qwen-sft/ \
    --model clawgym \
    --nim-base-url $NIM_BASE \
    > $RUN/qwen-sft/rollout.log 2>&1 &
echo "rollout #2 PID=$!"
disown

echo "rollout #2 launched. Monitor with:"
echo "  wc -l $RUN/qwen-sft/trajectories.jsonl"
