---
arxiv_id: 2605.04036
title: "OpenSeeker-v2: Pushing the Limits of Search Agents with Informative and High-Difficulty Trajectories"
published: 2026-05-04
primary_category: cs.AI
hf_upvotes: 36
popularity_score: 28
suggested_stage: fine-tuning
suggested_series: Autoresearch
fast_verdict: spark-feasible
relevance_score: 0.78
has_deep_eval: false
abs_url: https://arxiv.org/abs/2605.04036
pdf_url: https://arxiv.org/pdf/2605.04036
hf_paper_url: https://huggingface.co/papers/2605.04036
---

# OpenSeeker-v2: Pushing the Limits of Search Agents with Informative and High-Difficulty Trajectories

**Verdict:** spark-feasible · **Series:** Autoresearch · **Stage:** fine-tuning · **Relevance:** 0.78 · **Popularity:** 28/100

> SFT-only recipe for deep-search agents at small scale — high-signal trajectories + LoRA on a 7B base fits the Spark.

## Abstract

Deep search capabilities have become an indispensable competency for frontier Large Language Model (LLM) agents, yet their development remains dominated by industrial giants. The typical industry recipe involves a highly resource-intensive pipeline spanning pre-training, continual pre-training (CPT), supervised fine-tuning (SFT), and reinforcement learning (RL). In this report, we show that when fueled with informative and high-difficulty trajectories, a simple SFT approach could be surprisingly powerful for training frontier search agents. By introducing three simple data synthesis modifications: scaling knowledge graph size for richer exploration, expanding the tool set size for broader functionality, and strict low-step filtering, we establish a stronger baseline. Trained on merely 10.6k data points, our OpenSeeker-v2 achieves state-of-the-art performance across 4 benchmarks (30B-sized agents with ReAct paradigm): 46.0% on BrowseComp, 58.1% on BrowseComp-ZH, 34.6% on Humanity's Last Exam, and 78.0% on xbench, surpassing even Tongyi DeepResearch trained with heavy CPT+SFT+RL pipeline, which achieves 43.4%, 46.7%, 32.9%, and 75.0%, respectively. Notably, OpenSeeker-v2 represents the first state-of-the-art search agent within its model scale and paradigm to be developed by a purely academic team using only SFT. We are excited to open-source the OpenSeeker-v2 model weights and share our simple yet effective findings to make frontier search agent research more accessible to the community.

## Why this matters for ai-field-notes

- **Topic tags:** agentic, search-agents, sft, lora, trajectories, evals
- **NVIDIA stack:** NIM, NeMo, NemoClaw
- **Fast verdict rationale:** SFT-only recipe for deep-search agents at small scale — high-signal trajectories + LoRA on a 7B base fits the Spark.

## Repos

_No public repo yet._

## Citations

`citations: not yet indexed`

## Links

- [arXiv abstract](https://arxiv.org/abs/2605.04036)
- [PDF](https://arxiv.org/pdf/2605.04036)
- [HuggingFace daily papers](https://huggingface.co/papers/2605.04036)

