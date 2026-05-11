#!/bin/bash
# Post-retriever-download sequencer.
#
# Runs INSIDE the a2tgpo-spark container.
# Waits for the wiki-18 download to finish, then pulls Qwen3-4B + e5-base-v2,
# starts the retrieval server on port 5003, and writes a READY marker.
#
# Idempotent: re-running after partial failure resumes from the next missing step.

set -uo pipefail

LOG=/work/post-retriever.log
echo "--- post_retriever_setup.sh starting at $(date -u +%Y-%m-%dT%H:%M:%SZ) ---" | tee -a $LOG

# Step 1: wait for retriever PREP_DONE
echo "[1/5] waiting for /work/retriever-download.log PREP_DONE marker" | tee -a $LOG
until grep -q "PREP_DONE" /work/retriever-download.log 2>/dev/null; do
  sleep 30
done
echo "[1/5] PREP_DONE seen. Verifying artifacts..." | tee -a $LOG

if [ ! -f /work/retriever-data/e5_Flat.index ]; then
  echo "ERROR: /work/retriever-data/e5_Flat.index missing after PREP_DONE" | tee -a $LOG
  exit 11
fi
if [ ! -f /work/retriever-data/wiki-18.jsonl ]; then
  echo "ERROR: /work/retriever-data/wiki-18.jsonl missing after PREP_DONE" | tee -a $LOG
  exit 12
fi
echo "[1/5] retriever artifacts OK: $(du -sh /work/retriever-data/e5_Flat.index /work/retriever-data/wiki-18.jsonl | tr '\n' ' ')" | tee -a $LOG

# Step 2: pull Qwen3-4B
echo "[2/5] pulling Qwen/Qwen3-4B-Base to /work/models/Qwen3-4B" | tee -a $LOG
mkdir -p /work/models
if [ ! -f /work/models/Qwen3-4B/config.json ]; then
  python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Qwen/Qwen3-4B-Base',
    local_dir='/work/models/Qwen3-4B',
)
print('Qwen3-4B downloaded to /work/models/Qwen3-4B')
" 2>&1 | tee -a $LOG
else
  echo "[2/5] Qwen3-4B already present, skipping" | tee -a $LOG
fi

# Step 3: pull intfloat/e5-base-v2 retriever model
echo "[3/5] pulling intfloat/e5-base-v2 to /work/models/e5-base-v2" | tee -a $LOG
if [ ! -f /work/models/e5-base-v2/config.json ]; then
  python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='intfloat/e5-base-v2',
    local_dir='/work/models/e5-base-v2',
)
print('e5-base-v2 downloaded')
" 2>&1 | tee -a $LOG
else
  echo "[3/5] e5-base-v2 already present, skipping" | tee -a $LOG
fi

# Step 4: launch retrieval server (background)
echo "[4/5] starting retrieval_server on port 5003" | tee -a $LOG
if ! curl -sf http://127.0.0.1:5003/docs >/dev/null 2>&1; then
  nohup python3 /a2tgpo/evidence/repo-snapshot/rag_server/retrieval_server.py \
    --index_path /work/retriever-data/e5_Flat.index \
    --corpus_path /work/retriever-data/wiki-18.jsonl \
    --topk 3 \
    --retriever_model /work/models/e5-base-v2 \
    > /work/retrieval_server.log 2>&1 &
  echo "retrieval_server pid=$!" | tee -a $LOG
  # wait for it to be alive (faiss index load can take 5-15 min)
  echo "[4/5] waiting for /retrieve endpoint to come up (faiss index load)..." | tee -a $LOG
  for i in $(seq 1 60); do
    if curl -sf -X POST -H "Content-Type: application/json" \
         -d '{"queries":["test"],"topk":1,"return_scores":true}' \
         http://127.0.0.1:5003/retrieve >/dev/null 2>&1; then
      echo "[4/5] retrieval_server alive after ${i} × 30s = $((i*30))s" | tee -a $LOG
      break
    fi
    sleep 30
  done
  if ! curl -sf -X POST -H "Content-Type: application/json" \
        -d '{"queries":["test"],"topk":1,"return_scores":true}' \
        http://127.0.0.1:5003/retrieve >/dev/null 2>&1; then
    echo "ERROR: retrieval_server did not come up within 30 min" | tee -a $LOG
    tail -50 /work/retrieval_server.log | tee -a $LOG
    exit 14
  fi
else
  echo "[4/5] retrieval_server already running" | tee -a $LOG
fi

# Step 5: smoke a real query for latency
echo "[5/5] smoke retriever with sample query" | tee -a $LOG
START_NS=$(date +%s%N)
curl -sf -X POST -H "Content-Type: application/json" \
     -d '{"queries":["When was the Eiffel Tower completed?"],"topk":3,"return_scores":true}' \
     http://127.0.0.1:5003/retrieve > /work/retriever-smoke.json
END_NS=$(date +%s%N)
LATENCY_MS=$(( (END_NS - START_NS) / 1000000 ))
echo "[5/5] sample query latency: ${LATENCY_MS} ms" | tee -a $LOG
echo "[5/5] first 600 bytes of response:" | tee -a $LOG
head -c 600 /work/retriever-smoke.json | tee -a $LOG
echo "" | tee -a $LOG

# Write READY marker
echo "READY=$(date -u +%Y-%m-%dT%H:%M:%SZ) latency_ms=${LATENCY_MS}" > /work/POST_RETRIEVER_READY
echo "--- post_retriever_setup.sh DONE at $(date -u +%Y-%m-%dT%H:%M:%SZ) ---" | tee -a $LOG
