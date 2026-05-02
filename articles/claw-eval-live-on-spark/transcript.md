# Provenance: claw-eval-live-on-spark

Promoted from Frontier Scout on 2026-05-02.

- arXiv: 2604.28139 — Claw-Eval-Live: A Live Agent Benchmark for Evolving Real-World Workflows
- Repo: _(no public repo at promotion time; project page at claw-eval-live.github.io)_
- Fast verdict: spark-feasible
- Deep verdict: spark-feasible

The full agent eval is at `evidence/feasibility-eval.md`. The proposed Spark recipe is at `evidence/spark-recipe.md`. Use these as the starting outline; replace with measured numbers as the experiment progresses.

## Source material map

- `evidence/paper.pdf` — arxiv PDF, 4.7 MB
- `evidence/paper-meta.json` — papers.json entry (popularity 25/100, 22 HF upvotes)
- `evidence/feasibility-eval.md` — full Frontier Scout eval (immutable)
- `evidence/spark-recipe.md` — extracted Proposed Spark recipe section as runbook
- `evidence/repo-snapshot/README.txt` — stub (no public repo)

## Experiment status

Not started. Highest infrastructure cost of the five — service mocks (HR, ticketing, file workspace) need to be built, plus the sandbox-orchestration glue. Honest scope: 5–10 hand-authored tasks per family, simplest-possible mocks. Article frames as protocol replication on a downsized corpus, not full benchmark reproduction.
