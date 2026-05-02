# Provenance: test-time-distilling-for-exploration

Promoted from Frontier Scout on 2026-05-02.

- arXiv: 2604.24927 — Large Language Models Explore by Latent Distilling
- Repo: https://github.com/LinesHogan/tLLM (33⭐, Python, last push 2026-04-26)
- Fast verdict: spark-feasible
- Deep verdict: spark-feasible

The full agent eval is at `evidence/feasibility-eval.md`. The proposed Spark recipe is at `evidence/spark-recipe.md`. Use these as the starting outline; replace with measured numbers as the experiment progresses.

## Source material map

- `evidence/paper.pdf` — arxiv PDF, 6.6 MB
- `evidence/paper-meta.json` — papers.json entry (popularity 31/100, 59 HF upvotes)
- `evidence/feasibility-eval.md` — full Frontier Scout eval (immutable)
- `evidence/spark-recipe.md` — extracted Proposed Spark recipe section as runbook
- `evidence/repo-snapshot/` — shallow clone of LinesHogan/tLLM (vLLM-extension)

## Experiment status

Not started. The repo is real and recently pushed — this is the most-shippable of the five from a "their code runs as-is" standpoint. Caveat: vLLM is not the project's verified production path (NIM/TRT-LLM is). Article should call that out and frame the run as standalone-vLLM-on-Spark.
