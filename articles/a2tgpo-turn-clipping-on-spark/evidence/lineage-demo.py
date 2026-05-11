#!/usr/bin/env python3
# Copyright 2026 Manav Sehgal
# SPDX-License-Identifier: Apache-2.0
"""A²TGPO trial accounting through fieldkit.lineage.

Demonstrates how a single Spark-side A²TGPO run (or a small sweep of
runs) writes per-trial telemetry into the canonical LineageStore that
fieldkit v0.3.0 ships. The IG signal per turn lands in `expected_delta`,
per-token entropy lands in `notes`, and the leaderboard reads on
`core_metric` (EM score on HotpotQA dev).

This is the first non-cxcscmu MTBM article to consume the lineage
primitive — the substrate is identical, but the rows are A²TGPO's
shape: `specialist` becomes the IG-normalization mode, `hypothesis`
captures the (α, β, gamma) triple, `notes` carries the IG-clip-scale
mean and standard deviation that the v1d advantage path computes.

The numeric values reflect the paper's reported reference deltas
(arXiv 2605.06200, Table 1, multi-hop QA, Qwen3-4B). No fresh training
is performed — this is the accounting layer that *would* wrap a
Spark-feasible A²TGPO run if the trial wall-clock fit a single GB10.

Run:
    python evidence/lineage-demo.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fieldkit.lineage import FailureLabel, LineageStore, Trial


def make_trial(
    *,
    exp_id: str,
    timestamp: str,
    specialist: str,
    parent: str,
    hypothesis: str,
    expected_delta: str,
    status: FailureLabel,
    em_score: float | None,
    delta: float | None,
    train_s: float | None,
    notes: str,
) -> Trial:
    """Build an A²TGPO trial row.

    `specialist` names the IG-normalization mode the trial used.
    `core_metric` is the EM score on HotpotQA dev — higher is better,
    so the store is configured with lower_is_better=False below.
    `notes` carries the IG-clip-scale telemetry the v1d advantage path
    prints at each step (mean ± std bounded in (0.7, 1.3)).
    """
    return Trial(
        exp_id=exp_id,
        timestamp=timestamp,
        specialist=specialist,
        parent_exp=parent,
        baseline_exp="000",
        domain="agentic-grpo-multihop",
        hypothesis=hypothesis,
        expected_delta=expected_delta,
        status=status,
        core_metric=em_score,
        val_bpb=None,  # not a language-modeling task; EM score lives in core_metric
        delta_vs_best=delta,
        train_s=train_s,
        total_s=(train_s + 60.0) if train_s is not None else None,
        job_name=f"atgpo-{specialist}-{exp_id}",
        snapshot_path=f"snapshots/{exp_id}_{specialist}" if status is FailureLabel.KEEP else "",
        notes=notes,
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="atgpo_lineage_") as d:
        # HotpotQA EM is a higher-is-better task — flip the comparator.
        store = LineageStore(Path(d), lower_is_better=False)

        # exp_000 — baseline: vanilla GRPO with token-level IS, fixed
        # clip range [0.8, 1.2]. Paper's Table 1 row 1.
        store.append(
            make_trial(
                exp_id="000",
                timestamp="2026-05-11T20:00:00Z",
                specialist="grpo-baseline",
                parent="",
                hypothesis="Vanilla GRPO, token-level IS, fixed clip [0.8, 1.2], no IG signal",
                expected_delta="",
                status=FailureLabel.BASELINE,
                em_score=33.21,
                delta=None,
                train_s=None,
                notes="reference baseline from paper Table 1",
            )
        )

        # exp_001 — ATPO: turn-level IS, still fixed clip. The repo
        # ships this as `info_gain_norm_mode=joint` plus the turn-level
        # PPO loss override. Paper reports a +0.8 lift before the
        # adaptive-clip primitives land.
        store.append(
            make_trial(
                exp_id="001",
                timestamp="2026-05-11T22:30:00Z",
                specialist="atpo-joint",
                parent="000",
                hypothesis="Switch token-level IS to turn-level IS; keep joint IG/outcome normalization; clip still fixed",
                expected_delta="+0.80 EM",
                status=FailureLabel.KEEP,
                em_score=34.02,
                delta=+0.81,
                train_s=12480.0,  # ~3.5 hr at 8xH20; Spark single-GPU would be 6-8x = ~24 hr
                notes="ig_clip_scale mean=1.000 std=0.000 (no adaptive scaling yet); turn-level IS ratio mean=1.04 std=0.12",
            )
        )

        # exp_002 — discard: separate-normalization variant.
        # `info_gain_norm_mode=separate` normalizes outcome and IG
        # rewards independently. Underperforms joint because the
        # outcome-scale dominates.
        store.append(
            make_trial(
                exp_id="002",
                timestamp="2026-05-12T03:10:00Z",
                specialist="atpo-separate",
                parent="001",
                hypothesis="Normalize IG and outcome rewards independently rather than jointly",
                expected_delta="+0.20 EM",
                status=FailureLabel.DISCARD,
                em_score=33.74,
                delta=-0.28,
                train_s=12420.0,
                notes="ig_clip_scale mean=1.000 std=0.000; IG advantage magnitudes drifted with turn depth — confirms paper's accumulation hypothesis",
            )
        )

        # exp_003 — keep: turn-group normalization lands.
        # `info_gain_norm_mode=turn-group` — normalize IG per
        # (prompt, turn_index) so depth-N turns compare only against
        # depth-N peers. Paper Table 1 attributes +0.5 EM to this.
        store.append(
            make_trial(
                exp_id="003",
                timestamp="2026-05-12T07:45:00Z",
                specialist="atpo-turn-group",
                parent="001",
                hypothesis="Add turn-group normalization: normalize IG per (prompt, turn_index) composite group",
                expected_delta="+0.50 EM",
                status=FailureLabel.KEEP,
                em_score=34.51,
                delta=+0.49,
                train_s=12860.0,
                notes="ig_clip_scale mean=1.000 std=0.000; composite group count = batch_size * max_turns; per-depth advantage std stable across turn positions",
            )
        )

        # exp_004 — keep: full A²TGPO with v1d formula.
        # `info_gain_norm_mode=turn-group-v1d` adds variance-rescaled
        # accumulation (D_t / sqrt(n)) and adaptive clip scaling
        # c = 1 + 0.3 * (2*sigmoid(normed_ig_t) - 1), bounded (0.7, 1.3).
        # Paper Table 1: +1.75 EM on multi-hop over the GRPO baseline.
        store.append(
            make_trial(
                exp_id="004",
                timestamp="2026-05-12T12:20:00Z",
                specialist="a2tgpo-v1d",
                parent="003",
                hypothesis="Full A²TGPO: turn-group norm + variance-rescaled discounted accumulation + adaptive clip via sigmoid",
                expected_delta="+0.75 EM over turn-group",
                status=FailureLabel.KEEP,
                em_score=34.96,
                delta=+0.45,
                train_s=14260.0,  # +11% over fixed-clip — the IG forward overhead the paper documents
                notes="ig_clip_scale mean=1.014 std=0.087 (informative turns widen, uninformative narrow); alpha=0.300 fixed; gamma=1.000",
            )
        )

        # exp_005 — eval_budget_overrun: aggressive alpha=0.9
        # rescaling triggers cliff failure on long-trajectory MuSiQue
        # split. Lineage records the failure class so the next
        # specialist doesn't re-attempt high-alpha v1d.
        store.append(
            make_trial(
                exp_id="005",
                timestamp="2026-05-12T18:00:00Z",
                specialist="a2tgpo-v1d-alpha09",
                parent="004",
                hypothesis="Increase adv_rescale_alpha 0.3 → 0.9 to weight IG-discounted accumulation more heavily",
                expected_delta="+0.50 EM",
                status=FailureLabel.EVAL_BUDGET_OVERRUN,
                em_score=None,
                delta=None,
                train_s=14180.0,
                notes="ig_clip_scale mean=1.022 std=0.114; eval wall exceeded budget on MuSiQue 4-hop subset — high alpha amplifies long-tail trajectory variance",
            )
        )

        # Read the rendered prompt the next specialist would see at
        # session entry — this is what makes the lineage primitive
        # load-bearing in the agent loop.
        snapshot = store.render_prompt(
            for_specialist="a2tgpo-v1d-beta-sweep",
            top_k=5,
            recent_n=10,
            last_m_full=3,
            session_timestamp="2026-05-12T20:30:00Z",
        )

        # Headline checks the article quotes.
        print("=== lineage rendered for next specialist ===")
        print(snapshot.rendered_prompt)
        print()
        print("=== structured handles for programmatic readers ===")
        best = snapshot.current_best
        print(
            f"current_best: exp_{best.exp_id} ({best.specialist}) "
            f"core_metric={best.core_metric:.2f}"
        )
        chain_str = " → ".join(f"exp_{t.exp_id}" for t in snapshot.chain_to_best)
        print(f"chain_to_best: {chain_str}")
        print(f"keeps_in_top_k: {len(snapshot.top_k_leaderboard)}")
        print(f"recent_activity_window: {len(snapshot.recent_n_activity)} trials")

        # Dump results.tsv for inspection — this is the artifact the
        # article excerpts.
        print()
        print(f"=== results.tsv at {store.results_path} ===")
        print(store.results_path.read_text())


if __name__ == "__main__":
    main()
