---
arxiv_id: 2605.00553
title: "Stable-GFlowNet: Toward Diverse and Robust LLM Red-Teaming via Contrastive Trajectory Balance"
published: 2026-04-30
primary_category: unknown
hf_upvotes: 15
popularity_score: 21
suggested_stage: fine-tuning
suggested_series: LLM Wiki
fast_verdict: spark-feasible
relevance_score: 0.65
has_deep_eval: false
abs_url: https://arxiv.org/abs/2605.00553
pdf_url: https://arxiv.org/pdf/2605.00553
hf_paper_url: https://huggingface.co/papers/2605.00553
---

# Stable-GFlowNet: Toward Diverse and Robust LLM Red-Teaming via Contrastive Trajectory Balance

**Verdict:** spark-feasible · **Series:** LLM Wiki · **Stage:** fine-tuning · **Relevance:** 0.65 · **Popularity:** 21/100

> Stable GFlowNet for diverse LLM red-team prompt generation — small-scale RL on small attacker LM, safety-arc material.

## Abstract

Large Language Model (LLM) Red-Teaming, which proactively identifies vulnerabilities of LLMs, is an essential process for ensuring safety. Finding effective and diverse attacks in red-teaming is important, but achieving both is challenging. Generative Flow Networks (GFNs) that perform distribution matching are a promising methods, but they are notorious for training instability and mode collapse. In particular, unstable rewards in red-teaming accelerate mode collapse. We propose Stable-GFN (S-GFN), which eliminates partition function Z estimation in GFN and reduces training instability. S-GFN avoids Z-estimation through pairwise comparisons and employs a robust masking methodology against noisy rewards. Additionally, we propose a fluency stabilizer to prevent the model from getting stuck in local optima that produce gibberish. S-GFN provides more stable training while maintaining the optimal policy of GFN. We demonstrate the overwhelming attack performance and diversity of S-GFN across various settings.

## Why this matters for ai-field-notes

- **Topic tags:** red-teaming, guardrails, rl, gflownet, safety
- **NVIDIA stack:** NeMo, Guardrails, NIM
- **Fast verdict rationale:** Stable GFlowNet for diverse LLM red-team prompt generation — small-scale RL on small attacker LM, safety-arc material.

## Repos

_No public repo yet._

## Citations

`citations: not yet indexed`

## Links

- [arXiv abstract](https://arxiv.org/abs/2605.00553)
- [PDF](https://arxiv.org/pdf/2605.00553)
- [HuggingFace daily papers](https://huggingface.co/papers/2605.00553)

