---
arxiv_id: 2605.03596
title: "Workspace-Bench 1.0: Benchmarking AI Agents on Workspace Tasks with Large-Scale File Dependencies"
published: 2026-05-04
primary_category: unknown
hf_upvotes: 2
popularity_score: 9
suggested_stage: agentic
suggested_series: Autoresearch
fast_verdict: spark-feasible
relevance_score: 0.78
has_deep_eval: false
abs_url: https://arxiv.org/abs/2605.03596
pdf_url: https://arxiv.org/pdf/2605.03596
hf_paper_url: https://huggingface.co/papers/2605.03596
---

# Workspace-Bench 1.0: Benchmarking AI Agents on Workspace Tasks with Large-Scale File Dependencies

**Verdict:** spark-feasible · **Series:** Autoresearch · **Stage:** agentic · **Relevance:** 0.78 · **Popularity:** 9/100

> Workspace-Bench: agents over real file-dependency graphs — close cousin of ClawGym, ready Spark harness.

## Abstract

Workspace learning requires AI agents to identify, reason over, exploit, and update explicit and implicit dependencies among heterogeneous files in a worker's workspace, enabling them to complete both routine and advanced tasks effectively. Despite its importance, existing relevant benchmarks largely evaluate agents on pre-specified or synthesized files with limited real-world dependencies, leaving workspace-level evaluation underexplored. To this end, we introduce Workspace-Bench, a benchmark for evaluating AI agents on Workspace Learning invOlving Large-Scale File Dependencies. We construct realistic workspaces with 5 worker profiles, 74 file types, 20,476 files (up to 20GB) and curate 388 tasks, each with its own file dependency graph, evaluated across 7,399 total rubrics that require cross-file retrieval, contextual reasoning, and adaptive decision-making. We further provide Workspace-Bench-Lite, a 100-task subset that preserves the benchmark distribution while reducing evaluation costs by about 70%. We evaluate 4 popular agent harnesses and 7 foundation models. Experimental results show that current agents remain far from reliable workspace learning, where the best reaches only 68.7%, substantially below the human result of 80.7%, and the average performance across agents is only 47.4%.

## Why this matters for ai-field-notes

- **Topic tags:** agentic, benchmarks, workspace, file-dependencies, evals
- **NVIDIA stack:** NemoClaw, OpenClaw
- **Fast verdict rationale:** Workspace-Bench: agents over real file-dependency graphs — close cousin of ClawGym, ready Spark harness.

## Repos

_No public repo yet._

## Citations

`citations: not yet indexed`

## Links

- [arXiv abstract](https://arxiv.org/abs/2605.03596)
- [PDF](https://arxiv.org/pdf/2605.03596)
- [HuggingFace daily papers](https://huggingface.co/papers/2605.03596)

