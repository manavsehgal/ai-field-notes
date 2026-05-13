#!/usr/bin/env python3
# Copyright 2026 Manav Sehgal
# SPDX-License-Identifier: Apache-2.0
"""Vertical-curator quant lineage for the first Orionfold finance GGUF.

Writes the V2 baseline trial — the source BF16 fine-tune we're about to
quantize — into the canonical `fieldkit.lineage.LineageStore` TSV. Captures
the eval-bench / corpus / license metadata up front so the lineage entry
exists *before* the quantization runs (the gate per HANDOFF §2 Track B V2).

The B4 step (quantize + measure) will extend this TSV with one row per
GGUF variant — Q4_K_M / Q5_K_M / Q6_K / Q8_0 / F16 — carrying real
wikitext-2 perplexity, llama-bench tok/s, sustained-load minutes, and the
FinanceBench mini-eval accuracy from `fieldkit.eval.VerticalBench`. The
`core_metric` for variant rows is the FinanceBench accuracy (higher is
better — comparator flipped via `lower_is_better=False`); `val_bpb` is the
wikitext-2 perplexity (lower is better, used for sanity-check cross-walk).

Run:
    python evidence/lineage-demo.py
"""

from __future__ import annotations

from pathlib import Path

from fieldkit.lineage import FailureLabel, LineageStore, Trial


# --- Static metadata for the v0 vertical-curator quant ---------------------

DOMAIN = "vertical-curator-finance"
BASELINE_HF_REPO = "instruction-pretrain/finance-Llama3-8B"
BASELINE_ARCH = "LlamaForCausalLM (Llama-3 8B)"
BASELINE_LICENSE = "Llama-3 Community License"
BASELINE_PARAMS = "8.03B"
BASELINE_BYTES = 32_121_044_992  # 7-shard safetensors total, ~32 GB FP32 storage

# Quant target — Orionfold v0 publishing repo.
TARGET_HF_REPO = "orionfoldllc/finance-Llama3-8B-GGUF"
TARGET_VARIANTS = ("Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16")

# Eval-bench metadata (Q16 = (b) Spark-overlay).
BENCH_NAME = "FinanceBench (public open subset)"
BENCH_HF_DATASET = "PatronusAI/financebench"
BENCH_FILE = "financebench_merged.jsonl"
BENCH_SIZE = "150 Q&A pairs (open subset; full bench is 10.2K)"
BENCH_LICENSE = "CC-BY-NC-4.0"
BENCH_ARXIV = "2311.11944"
BENCH_SCORER = "fieldkit.eval.numeric_match (rel_tolerance=0.01)"

# Calibration corpus (used by `measure_perplexity_gguf` for the variant rows).
CALIBRATION_PATH = "/home/nvidia/data/calibration/wikitext-2-raw-v1/wiki.test.raw"
CALIBRATION_DESC = "wikitext-2 raw test split (4,358 lines, 1.3 MB)"


def make_variant_trial(
    *,
    exp_id: str,
    variant: str,
    timestamp: str,
    finance_accuracy: float | None,
    wikitext_perplexity: float | None,
    delta_vs_best_acc: float | None,
    quantize_seconds: float | None,
    total_seconds: float | None,
    gguf_size_bytes: int | None,
    tokens_per_sec_tg: float | None,
    tokens_per_sec_pp: float | None,
    sustained_load_minutes: float | None,
    status: FailureLabel = FailureLabel.KEEP,
) -> Trial:
    """Build a per-variant trial row.

    Called by `B4` once `fieldkit.quant` measurements land. `core_metric` is
    the FinanceBench numeric_match accuracy (higher is better), `val_bpb` is
    wikitext-2 perplexity (lower is better — sanity cross-walk vs Bartowski).
    `train_s` carries quantize wall-time; `total_s` adds the measurement
    sweep. Per-variant tok/s and thermal envelope live in `notes` since the
    schema is fixed.
    """
    notes_bits = []
    if tokens_per_sec_tg is not None:
        notes_bits.append(f"tg_tok_per_s={tokens_per_sec_tg:.1f}")
    if tokens_per_sec_pp is not None:
        notes_bits.append(f"pp_tok_per_s={tokens_per_sec_pp:.1f}")
    if sustained_load_minutes is not None:
        notes_bits.append(f"sustained_load_min={sustained_load_minutes:.1f}")
    if gguf_size_bytes is not None:
        notes_bits.append(f"gguf_size_bytes={gguf_size_bytes}")
    notes_bits.append(f"bench={BENCH_HF_DATASET}")
    notes_bits.append(f"corpus={CALIBRATION_DESC}")
    return Trial(
        exp_id=exp_id,
        timestamp=timestamp,
        specialist=f"orionfold-curator/{variant}",
        parent_exp="000",
        baseline_exp="000",
        domain=DOMAIN,
        hypothesis=f"{variant} quant of {BASELINE_HF_REPO} — Spark-tested measurement layer",
        expected_delta=f"FinanceBench accuracy delta vs F16 baseline",
        status=status,
        core_metric=finance_accuracy,
        val_bpb=wikitext_perplexity,
        delta_vs_best=delta_vs_best_acc,
        train_s=quantize_seconds,
        total_s=total_seconds,
        job_name=f"orionfold-finance-llama3-8b-{variant.lower()}",
        snapshot_path=f"/home/nvidia/data/quants/finance-Llama3-8B/model-{variant}.gguf",
        notes=" ; ".join(notes_bits),
    )


def make_baseline_trial() -> Trial:
    """The V2 row — captures everything we know before quantization runs."""
    notes_parts = [
        f"hf_repo={BASELINE_HF_REPO}",
        f"arch={BASELINE_ARCH}",
        f"license={BASELINE_LICENSE}",
        f"params={BASELINE_PARAMS}",
        f"weight_bytes={BASELINE_BYTES}",
        f"target_repo={TARGET_HF_REPO}",
        f"variants={'/'.join(TARGET_VARIANTS)}",
        f"bench={BENCH_NAME} ({BENCH_HF_DATASET}, {BENCH_LICENSE}, arxiv:{BENCH_ARXIV})",
        f"bench_file={BENCH_FILE}",
        f"bench_size={BENCH_SIZE}",
        f"bench_scorer={BENCH_SCORER}",
        f"calibration={CALIBRATION_DESC} @ {CALIBRATION_PATH}",
    ]
    return Trial(
        exp_id="000",
        timestamp="2026-05-13T18:46:00Z",
        specialist="orionfold-curator",
        parent_exp="",
        baseline_exp="000",
        domain=DOMAIN,
        hypothesis=(
            "Quantize finance-Llama3-8B (Microsoft Instruction Pre-Training NeurIPS 2024) "
            "to five GGUF variants and ship as orionfoldllc/finance-Llama3-8B-GGUF with "
            "FinanceBench mini-eval + Spark perplexity + tok/s + sustained-load minutes "
            "on every card — the vertical-curator differentiation pattern."
        ),
        expected_delta="",
        status=FailureLabel.BASELINE,
        core_metric=None,  # FinanceBench accuracy lands here on variant rows
        val_bpb=None,  # wikitext-2 perplexity lands here on variant rows
        delta_vs_best=None,
        train_s=None,
        total_s=None,
        job_name="orionfold-finance-llama3-8b-v0",
        snapshot_path="",
        notes=" ; ".join(notes_parts),
    )


def main() -> None:
    """Write the baseline row to `evidence/lineage/results.tsv`.

    FinanceBench accuracy is higher-is-better, so `LineageStore` runs with
    `lower_is_better=False`. The V2 row carries no metrics yet, so the
    comparator is academic until B4 fills variants in.
    """
    evidence_root = Path(__file__).parent / "lineage"
    evidence_root.mkdir(parents=True, exist_ok=True)
    store = LineageStore(evidence_root, lower_is_better=False)

    baseline = make_baseline_trial()
    store.append(baseline)

    print(f"wrote {evidence_root}/results.tsv")
    print(f"  baseline: {baseline.exp_id}  specialist={baseline.specialist}")
    print(f"  hf_repo={BASELINE_HF_REPO}")
    print(f"  bench={BENCH_HF_DATASET} ({BENCH_LICENSE}, arxiv:{BENCH_ARXIV})")
    print()
    print("Next: B4 (quantize + measure) extends this TSV with one row per variant.")


if __name__ == "__main__":
    main()
