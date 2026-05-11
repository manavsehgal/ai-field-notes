# Provenance: a2tgpo-turn-clipping-on-spark

Promoted from Frontier Scout on 2026-05-08; published 2026-05-11.

- arXiv: [2605.06200](https://arxiv.org/abs/2605.06200)
- Repo: [github.com/CuSO4-Chen/A-TGPO](https://github.com/CuSO4-Chen/A-TGPO)
- Fast verdict: spark-feasible
- Deep verdict: spark-feasible (algorithm fits at ~30 GB working set; per-trial wall on single GB10 is 6–8× the published 8×H20 schedule, so promotion uses **study-from-source** mode rather than a fresh-data reproduction).

## Study mode

This article does not reproduce the paper's headline numbers on Spark. The published recipe trains on 8×H20 for ~14k steps per configuration; single-GB10 projection is six-to-eight days of continuous wall per single config, which exceeds the 6-hour per-trial budget the memory note `feedback_spark_scaling_optimism` documents. The released artifact this article studies is therefore the *source code* — `verl_atgpo/verl/trainer/ppo/core_algos.py` lines 1190–1400 — not a checkpoint or a training log.

## Source material walked

- `evidence/paper.pdf` — full A²TGPO paper, Tables 1–3 for EM lifts.
- `evidence/feasibility-eval.md` — Frontier Scout's spark-feasibility eval (verdict: in-envelope at 4B bf16).
- `evidence/spark-recipe.md` — recipe-level plan (now superseded by the source-walk in §4 of the article).
- `evidence/repo-snapshot/ATGPO/verl_atgpo/verl/trainer/ppo/core_algos.py` — vendored upstream; the three primitives live at lines 1264–1400 (`compute_atgpo_advantage` for normalization + accumulation, lines 1367–1375 for the σ-bounded clip-scale construction) and lines 920–924 (the PPO loss override consuming `ig_clip_scale`).

## Lineage demo

`evidence/lineage-demo.py` — A 6-trial worked example that walks a representative A²TGPO sweep (vanilla GRPO baseline → ATPO joint → separate norm discard → turn-group keep → full v1d keep → α=0.9 eval-budget-overrun) and writes each into a `fieldkit.lineage.LineageStore`. Renders the prompt the next specialist would see at session entry. The EM scores reflect the paper's reported reference deltas on HotpotQA dev (Qwen3-4B); no live training was run for this article.

Generated artifacts:
- `evidence/lineage-rendered.txt` — the full Markdown block the demo prints.
- `evidence/results.tsv` — the 6-row TSV the demo writes.

## Open follow-ups

- `fieldkit.training.rl` extraction (deferred to v0.4 fieldkit release): three primitives — `InformationGain`, `TurnGroupNormalizer`, `AdaptiveTurnClipper` — each ~50 LOC pure-torch.
- A future MTBM article will run a single A²TGPO configuration to wall-clock completion on Spark and write the row to a real `LineageStore`. This article scopes the receipt; that article scopes the run.
