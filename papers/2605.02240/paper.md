---
arxiv_id: 2605.02240
title: "PhysicianBench: Evaluating LLM Agents in Real-World EHR Environments"
published: 2026-05-03
primary_category: unknown
hf_upvotes: 6
popularity_score: 15
suggested_stage: agentic
suggested_series: Autoresearch
fast_verdict: borderline
relevance_score: 0.55
has_deep_eval: false
abs_url: https://arxiv.org/abs/2605.02240
pdf_url: https://arxiv.org/pdf/2605.02240
hf_paper_url: https://huggingface.co/papers/2605.02240
---

# PhysicianBench: Evaluating LLM Agents in Real-World EHR Environments

**Verdict:** borderline · **Series:** Autoresearch · **Stage:** agentic · **Relevance:** 0.55 · **Popularity:** 15/100

> Long-horizon EHR agent benchmark — domain-specific but the harness shape mirrors ClawGym; runnable against a hosted NIM.

## Abstract

We introduce PhysicianBench, a benchmark for evaluating LLM agents on physician tasks grounded in real clinical setting within electronic health record (EHR) environments. Existing medical agent benchmarks primarily focus on static knowledge recall, single-step atomic actions, or action intent without verifiable execution against the environment. As a result, they fail to capture the long-horizon, composite workflows that characterize real clinical systems. PhysicianBench comprises 100 long-horizon tasks adapted from real consultation cases between primary care and subspecialty physicians, with each task independently reviewed by a separate panel of physicians. Tasks are instantiated in an EHR environment with real patient records and accessed through the same standard APIs used by commercial EHR vendors. Tasks span 21 specialties (e.g., cardiology, endocrinology, oncology, psychiatry) and diverse workflow types (e.g., diagnosis interpretation, medication prescribing, treatment planning), requiring an average of 27 tool calls per task. Solving each task requires retrieving data across encounters, reasoning over heterogeneous clinical information, executing consequential clinical actions, and producing clinical documentation. Each task is decomposed into structured checkpoints (670 in total across the benchmark) capturing distinct stages of completion graded by task-specific scripts with execution-grounded verification. Across 13 proprietary and open-source LLM agents, the best-performing model achieves only 46% success rate (pass@1), while open-source models reach at most 19%, revealing a substantial gap between current agent capabilities and the demands of real-world clinical workflows. PhysicianBench provides a realistic and execution-grounded benchmark for measuring progress toward autonomous clinical agents.

## Why this matters for ai-field-notes

- **Topic tags:** agentic, benchmarks, ehr, long-horizon, tool-use, evals
- **NVIDIA stack:** NemoClaw, Guardrails
- **Fast verdict rationale:** Long-horizon EHR agent benchmark — domain-specific but the harness shape mirrors ClawGym; runnable against a hosted NIM.

## Repos

_No public repo yet._

## Citations

`citations: not yet indexed`

## Links

- [arXiv abstract](https://arxiv.org/abs/2605.02240)
- [PDF](https://arxiv.org/pdf/2605.02240)
- [HuggingFace daily papers](https://huggingface.co/papers/2605.02240)

