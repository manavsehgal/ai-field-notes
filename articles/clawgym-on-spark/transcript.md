# Provenance: clawgym-on-spark

Promoted from Frontier Scout on 2026-05-02.

- arXiv: 2604.26904 — ClawGym: A Scalable Framework for Building Effective Claw Agents
- Repo: _(github.com/ClawGym org exists; no code yet)_
- Fast verdict: spark-feasible
- Deep verdict: spark-feasible

The full agent eval is at `evidence/feasibility-eval.md`. The proposed Spark recipe is at `evidence/spark-recipe.md`. Use these as the starting outline; replace with measured numbers as the experiment progresses.

## Source material map

- `evidence/paper.pdf` — arxiv PDF, 1.7 MB
- `evidence/paper-meta.json` — papers.json entry (popularity 30/100, 44 HF upvotes)
- `evidence/feasibility-eval.md` — full Frontier Scout eval (immutable)
- `evidence/spark-recipe.md` — extracted Proposed Spark recipe section as runbook
- `evidence/repo-snapshot/README.txt` — stub (no public repo)

## Experiment status

Not started. Next session should either wait for upstream's 13.5K dataset drop or generate a 1K-task subset from the persona-driven recipe described in the paper. Either way, sandbox + SFT + RL plumbing can be exercised end-to-end on the smaller corpus first.
