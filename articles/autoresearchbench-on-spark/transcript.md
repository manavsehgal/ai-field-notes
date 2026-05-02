# Provenance: autoresearchbench-on-spark

Promoted from Frontier Scout on 2026-05-02.

- arXiv: 2604.25256 — AutoResearchBench: Benchmarking AI Agents on Complex Scientific Literature Discovery
- Repo: https://github.com/CherYou/AutoResearchBench (29⭐, Apache-2.0, Python+Shell, last push 2026-04-24)
- Dataset: https://huggingface.co/datasets/Lk123/AutoResearchBench
- Fast verdict: spark-feasible
- Deep verdict: spark-feasible

The full agent eval is at `evidence/feasibility-eval.md`. The proposed Spark recipe is at `evidence/spark-recipe.md`. Use these as the starting outline; replace with measured numbers as the experiment progresses.

## Source material map

- `evidence/paper.pdf` — arxiv PDF, 3.2 MB
- `evidence/paper-meta.json` — papers.json entry (popularity 26/100, 27 HF upvotes)
- `evidence/feasibility-eval.md` — full Frontier Scout eval (immutable)
- `evidence/spark-recipe.md` — extracted Proposed Spark recipe section as runbook
- `evidence/repo-snapshot/` — shallow clone of CherYou/AutoResearchBench (Apache-2.0)

## Experiment status

Not started. **This is the most-shippable of the five** — the upstream repo is mature, Apache-2.0, dataset on HF, and the inference entrypoint takes `OPENAI_API_KEY` + `OPENAI_API_BASE` from `.env`, which NIM exposes natively. Likely 1–2 hours to first numbers, plus 4–6 hours for the 3-model comparative leaderboard.
