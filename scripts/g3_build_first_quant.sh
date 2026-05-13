#!/usr/bin/env bash
# G3 Track B end-to-end runner — produce first Orionfold GGUF on Spark.
#
# Owns: model download → quantize_gguf → measure perplexity + tok/s + thermal
# → dry-run publish_quant. Stops before the HF push so Track A (HF org / token)
# can be ungated independently — the final push is a one-liner once HF_TOKEN
# lands.
#
# Defaults target the revised front-runner from HANDOFF §2 (Track B1):
# nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 — Bartowski gap as of
# 2026-05-12. Override via env vars.
#
# Usage:
#   ./scripts/g3_build_first_quant.sh
#   MODEL_ID=nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 ./scripts/g3_build_first_quant.sh  # fallback
#   QUANT_VARIANTS=Q4_K_M,Q8_0 ./scripts/g3_build_first_quant.sh  # subset
#   SKIP_DOWNLOAD=1 ./scripts/g3_build_first_quant.sh  # reuse existing weights
#
# Prereqs:
#   - llama.cpp built on Spark (CUDA on, GGML_CUDA=ON). Default search path
#     /home/nvidia/llama.cpp; override LLAMA_CPP_DIR.
#   - huggingface_hub CLI in a venv. Default /tmp/fk-test; override HF_VENV.
#   - ~70 GB free disk for the source weights + ~150 GB for all five GGUF
#     variants. Total ~220 GB at peak. Spark home volume should have ≥ 250 GB.

set -euo pipefail

# --- Config (env-overridable) -----------------------------------------------

MODEL_ID="${MODEL_ID:-nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16}"
MODEL_SLUG="${MODEL_SLUG:-$(basename "$MODEL_ID")}"
BASE_MODEL_ARG="${BASE_MODEL_ARG:-$MODEL_ID}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-/home/nvidia/llama.cpp}"
LLAMA_CPP_BIN="${LLAMA_CPP_BIN:-${LLAMA_CPP_DIR}/build/bin}"
LLAMA_CPP_CONVERT="${LLAMA_CPP_CONVERT:-${LLAMA_CPP_DIR}/convert_hf_to_gguf.py}"
HF_VENV="${HF_VENV:-/tmp/fk-test}"
MODELS_DIR="${MODELS_DIR:-/home/nvidia/data/models}"
QUANTS_DIR="${QUANTS_DIR:-/home/nvidia/data/quants}"
STAGE_DIR="${STAGE_DIR:-/tmp/orionfold-stage}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-/home/nvidia/ai-field-notes/src/content/artifacts}"
ARTICLE_SLUG="${ARTICLE_SLUG:-becoming-a-gguf-publisher-on-spark}"
QUANT_VARIANTS="${QUANT_VARIANTS:-Q4_K_M,Q5_K_M,Q6_K,Q8_0,F16}"
WIKITEXT_CORPUS="${WIKITEXT_CORPUS:-/home/nvidia/data/calibration/wikitext-2-raw/wiki.test.raw}"
REPO_NAME="${REPO_NAME:-${MODEL_SLUG}-GGUF}"

# --- Logging helpers --------------------------------------------------------

log() { printf '\033[1;36m[g3]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[g3 FATAL]\033[0m %s\n' "$*" >&2; exit 1; }

# --- Step 1: preflight ------------------------------------------------------

step_preflight() {
  log "preflight — checking llama.cpp + huggingface_hub + disk"
  for bin in llama-quantize llama-perplexity llama-bench; do
    if [[ ! -x "${LLAMA_CPP_BIN}/${bin}" ]]; then
      die "llama.cpp binary not found: ${LLAMA_CPP_BIN}/${bin} — build llama.cpp first or set LLAMA_CPP_DIR"
    fi
  done
  if [[ ! -f "$LLAMA_CPP_CONVERT" ]]; then
    die "convert_hf_to_gguf.py not found at $LLAMA_CPP_CONVERT — set LLAMA_CPP_CONVERT"
  fi
  if [[ ! -x "${HF_VENV}/bin/hf" ]]; then
    log "installing huggingface_hub into ${HF_VENV}"
    if [[ ! -x "${HF_VENV}/bin/python" ]]; then
      die "venv ${HF_VENV} not found — create one with python3 -m venv ${HF_VENV}"
    fi
    "${HF_VENV}/bin/pip" install --quiet 'huggingface_hub>=1.14'
  fi
  local avail_gb
  avail_gb=$(df --output=avail -BG "$MODELS_DIR" 2>/dev/null | tail -1 | tr -dc '0-9' || echo 0)
  if [[ -z "$avail_gb" || "$avail_gb" -lt 250 ]]; then
    log "warn: <250 GB free at $MODELS_DIR; may run out of space at peak (~220 GB needed)"
  fi
  mkdir -p "$MODELS_DIR" "$QUANTS_DIR" "$STAGE_DIR" "$ARTIFACTS_DIR"
}

# --- Step 2: download source model -----------------------------------------

step_download() {
  local model_dir="${MODELS_DIR}/${MODEL_SLUG}"
  if [[ -n "${SKIP_DOWNLOAD:-}" ]]; then
    log "skip-download: reusing $model_dir"
    return
  fi
  if [[ -f "${model_dir}/config.json" ]]; then
    log "model already present at ${model_dir} — skipping download (set SKIP_DOWNLOAD=0 to force re-pull)"
    return
  fi
  log "downloading ${MODEL_ID} → ${model_dir} (this can take 30–120 min depending on bandwidth)"
  "${HF_VENV}/bin/hf" download "$MODEL_ID" \
    --local-dir "$model_dir"
}

# --- Step 3: probe convert support (cheap — config.json only) --------------

step_probe_convert() {
  local model_dir="${MODELS_DIR}/${MODEL_SLUG}"
  log "probe — checking if convert_hf_to_gguf.py accepts this architecture"
  if "${HF_VENV}/bin/python" "$LLAMA_CPP_CONVERT" --help >/dev/null 2>&1; then
    log "convert script callable"
  else
    die "convert_hf_to_gguf.py not executable — check Python env has the requirements"
  fi
  # The actual architecture check happens at convert time. Documented as a
  # known risk in HANDOFF §2: omnimodal Nemotron may need a forked llama.cpp
  # branch or text-decoder-only extraction.
}

# --- Step 4: quantize via fieldkit.quant -----------------------------------

step_quantize() {
  local model_dir="${MODELS_DIR}/${MODEL_SLUG}"
  local out_dir="${QUANTS_DIR}/${MODEL_SLUG}"
  mkdir -p "$out_dir"
  log "quantizing ${MODEL_SLUG} → ${out_dir} (variants: ${QUANT_VARIANTS})"
  LLAMA_CPP_BIN="$LLAMA_CPP_BIN" LLAMA_CPP_CONVERT="$LLAMA_CPP_CONVERT" \
    "${HF_VENV}/bin/python" - <<PYEOF
import os
from pathlib import Path
from fieldkit.quant import quantize_gguf, LlamaCppPaths

paths = LlamaCppPaths().resolve()
variants = tuple("${QUANT_VARIANTS}".split(","))
report = quantize_gguf(
    model="${model_dir}",
    outdir="${out_dir}",
    variants=variants,
    paths=paths,
    base_model_id="${BASE_MODEL_ARG}",
)
print("variants written:", list(report.variant_files.keys()))
for v, info in report.variant_files.items():
    print(f"  {v}: {info.get('size','?')}")
PYEOF
}

# --- Step 5: measure perplexity + tok/s ------------------------------------

step_measure() {
  local out_dir="${QUANTS_DIR}/${MODEL_SLUG}"
  if [[ ! -f "$WIKITEXT_CORPUS" ]]; then
    log "warn: wikitext corpus not found at $WIKITEXT_CORPUS — skipping perplexity pass"
    log "      download via: hf download Salesforce/wikitext --local-dir /home/nvidia/data/calibration --include 'wikitext-2-raw-v1/*'"
    return
  fi
  log "measuring perplexity + tok/s per variant"
  LLAMA_CPP_BIN="$LLAMA_CPP_BIN" \
    "${HF_VENV}/bin/python" - <<PYEOF
import json
from pathlib import Path
from fieldkit.quant import (
    LlamaCppPaths,
    measure_perplexity_gguf,
    measure_tokens_per_sec_gguf,
)

paths = LlamaCppPaths().resolve()
out_dir = Path("${out_dir}")
variants = "${QUANT_VARIANTS}".split(",")
report = {"perplexity": {}, "tokens_per_sec": {}}

for v in variants:
    gguf = out_dir / f"model-{v}.gguf"
    if not gguf.exists():
        continue
    print(f"measuring {v} …")
    ppl = measure_perplexity_gguf(
        gguf_path=gguf,
        corpus_path="${WIKITEXT_CORPUS}",
        paths=paths,
    )
    tps = measure_tokens_per_sec_gguf(
        gguf_path=gguf,
        paths=paths,
    )
    report["perplexity"][v] = ppl
    report["tokens_per_sec"][v] = tps
    print(f"  perplexity={ppl} tok/s={tps}")

(out_dir / "measurements.json").write_text(json.dumps(report, indent=2))
PYEOF
}

# --- Step 6: dry-run publish -----------------------------------------------

step_dry_run_publish() {
  local out_dir="${QUANTS_DIR}/${MODEL_SLUG}"
  log "dry-run publish_quant — staging at ${STAGE_DIR}/${MODEL_SLUG}"
  "${HF_VENV}/bin/python" - <<PYEOF
import json
from pathlib import Path
from types import SimpleNamespace
from fieldkit.publish import publish_quant

out_dir = Path("${out_dir}")
variants = "${QUANT_VARIANTS}".split(",")
measurements = json.loads((out_dir / "measurements.json").read_text()) if (out_dir / "measurements.json").exists() else {"perplexity": {}, "tokens_per_sec": {}}

variant_files = {}
for v in variants:
    gguf = out_dir / f"model-{v}.gguf"
    if gguf.exists():
        variant_files[v] = {"path": str(gguf), "rel": gguf.name, "size": ""}

report = SimpleNamespace(
    format="gguf",
    variants=tuple(variants),
    variant_files=variant_files,
    perplexity=measurements.get("perplexity", {}),
    tokens_per_sec=measurements.get("tokens_per_sec", {}),
    sustained_load_minutes=None,
)

result = publish_quant(
    quant_report=report,
    base_model="${BASE_MODEL_ARG}",
    repo_name="${REPO_NAME}",
    staging_dir="${STAGE_DIR}/${MODEL_SLUG}",
    artifacts_dir="${ARTIFACTS_DIR}",
    article_slug="${ARTICLE_SLUG}",
    article_title="Becoming a GGUF publisher on the Spark",
    dry_run=True,
)
print()
print("=== Dry-run result ===")
print("hf_repo:    ", result.hf_repo)
print("card_path:  ", result.card_path)
print("manifest:   ", result.manifest_path)
print("files staged:")
for f in result.files_uploaded:
    print(f"  {f}")
PYEOF
}

# --- Main -------------------------------------------------------------------

case "${1:-all}" in
  preflight)          step_preflight ;;
  download)           step_preflight && step_download ;;
  probe)              step_preflight && step_probe_convert ;;
  quantize)           step_preflight && step_quantize ;;
  measure)            step_preflight && step_measure ;;
  publish-dryrun)     step_preflight && step_dry_run_publish ;;
  all)
    step_preflight
    step_download
    step_probe_convert
    step_quantize
    step_measure
    step_dry_run_publish
    log "done — review staged card at ${STAGE_DIR}/${MODEL_SLUG}/README.md"
    log "next: flip dry_run=False with HF_TOKEN set to actually push"
    ;;
  *)
    cat <<EOF
Usage: $0 <step>

Steps:
  preflight       — verify llama.cpp + venv + disk
  download        — pull source model from HF
  probe           — check convert script accepts the architecture
  quantize        — produce GGUF variants via fieldkit.quant
  measure         — perplexity + tok/s per variant (needs wikitext)
  publish-dryrun  — stage card + manifest via fieldkit.publish (dry_run=True)
  all             — run every step in order (default)

Env overrides:
  MODEL_ID, MODEL_SLUG, LLAMA_CPP_DIR, MODELS_DIR, QUANTS_DIR, STAGE_DIR,
  ARTIFACTS_DIR, ARTICLE_SLUG, QUANT_VARIANTS, WIKITEXT_CORPUS, REPO_NAME,
  SKIP_DOWNLOAD
EOF
    exit 1
    ;;
esac
