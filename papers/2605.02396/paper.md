---
arxiv_id: 2605.02396
title: "HeavySkill: Heavy Thinking as the Inner Skill in Agentic Harness"
published: 2026-05-03
primary_category: unknown
hf_upvotes: 9
popularity_score: 18
suggested_stage: agentic
suggested_series: Autoresearch
fast_verdict: spark-feasible
relevance_score: 0.7
has_deep_eval: false
abs_url: https://arxiv.org/abs/2605.02396
pdf_url: https://arxiv.org/pdf/2605.02396
hf_paper_url: https://huggingface.co/papers/2605.02396
---

# HeavySkill: Heavy Thinking as the Inner Skill in Agentic Harness

**Verdict:** spark-feasible · **Series:** Autoresearch · **Stage:** agentic · **Relevance:** 0.70 · **Popularity:** 18/100

> Internalizing 'heavy thinking' into the model rather than the harness — directly testable on Spark's NIM-hosted thinking models.

## Abstract

Recent advances in agentic harness with orchestration frameworks that coordinate multiple agents with memory, skills, and tool use have achieved remarkable success in complex reasoning tasks. However, the underlying mechanism that truly drives performance remains obscured behind intricate system designs. In this paper, we propose HeavySkill, a perspective that views heavy thinking not only as a minimal execution unit in orchestration harness but also as an inner skill internalized within the model's parameters that drives the orchestrator to solve complex tasks. We identify this skill as a two-stage pipeline, i.e., parallel reasoning then summarization, which can operate beneath any agentic harness. We present a systematic empirical study of HeavySkill across diverse domains. Our results show that this inner skill consistently outperforms traditional Best-of-N (BoN) strategies; notably, stronger LLMs can even approach Pass@N performance. Crucially, we demonstrate that the depth and width of heavy thinking, as a learnable skill, can be further scaled via reinforcement learning, offering a promising path toward self-evolving LLMs that internalize complex reasoning without relying on brittle orchestration layers.

## Why this matters for ai-field-notes

- **Topic tags:** agentic, reasoning, thinking, harness, internalization
- **NVIDIA stack:** NIM, NemoClaw
- **Fast verdict rationale:** Internalizing 'heavy thinking' into the model rather than the harness — directly testable on Spark's NIM-hosted thinking models.

## Repos

_No public repo yet._

## Citations

`citations: not yet indexed`

## Links

- [arXiv abstract](https://arxiv.org/abs/2605.02396)
- [PDF](https://arxiv.org/pdf/2605.02396)
- [HuggingFace daily papers](https://huggingface.co/papers/2605.02396)

