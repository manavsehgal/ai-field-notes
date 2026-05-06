---
arxiv_id: 2604.28123
title: "Beyond SFT-to-RL: Pre-alignment via Black-Box On-Policy Distillation for Multimodal RL"
published: 2026-04-30
primary_category: unknown
hf_upvotes: 34
popularity_score: 27
suggested_stage: fine-tuning
suggested_series: LLM Wiki
fast_verdict: spark-feasible
relevance_score: 0.75
has_deep_eval: false
abs_url: https://arxiv.org/abs/2604.28123
pdf_url: https://arxiv.org/pdf/2604.28123
hf_paper_url: https://huggingface.co/papers/2604.28123
---

# Beyond SFT-to-RL: Pre-alignment via Black-Box On-Policy Distillation for Multimodal RL

**Verdict:** spark-feasible · **Series:** LLM Wiki · **Stage:** fine-tuning · **Relevance:** 0.75 · **Popularity:** 27/100

> Black-box on-policy distillation as pre-alignment for multimodal RL — small-student distillation is in the Spark envelope.

## Abstract

The standard post-training recipe for large multimodal models (LMMs) applies supervised fine-tuning (SFT) on curated demonstrations followed by reinforcement learning with verifiable rewards (RLVR). However, SFT introduces distributional drift that neither preserves the model's original capabilities nor faithfully matches the supervision distribution. This problem is further amplified in multimodal reasoning, where perception errors and reasoning failures follow distinct drift patterns that compound during subsequent RL. We introduce PRISM, a three-stage pipeline that mitigates this drift by inserting an explicit distribution-alignment stage between SFT and RLVR. Building on the principle of on-policy distillation (OPD), PRISM casts alignment as a black-box, response-level adversarial game between the policy and a Mixture-of-Experts (MoE) discriminator with dedicated perception and reasoning experts, providing disentangled corrective signals that steer the policy toward the supervision distribution without requiring access to teacher logits. While 1.26M public demonstrations suffice for broad SFT initialization, distribution alignment demands higher-fidelity supervision; we therefore curate 113K additional demonstrations from Gemini 3 Flash, featuring dense visual grounding and step-by-step reasoning on the hardest unsolved problems. Experiments on Qwen3-VL show that PRISM consistently improves downstream RLVR performance across multiple RL algorithms (GRPO, DAPO, GSPO) and diverse multimodal benchmarks, improving average accuracy by +4.4 and +6.0 points over the SFT-to-RLVR baseline on 4B and 8B, respectively. Our code, data, and model checkpoints are publicly available at https://github.com/XIAO4579/PRISM.

## Why this matters for ai-field-notes

- **Topic tags:** distillation, rl, multimodal, post-training, lora
- **NVIDIA stack:** NeMo, NIM
- **Fast verdict rationale:** Black-box on-policy distillation as pre-alignment for multimodal RL — small-student distillation is in the Spark envelope.

## Repos

_No public repo yet._

## Citations

`citations: not yet indexed`

## Links

- [arXiv abstract](https://arxiv.org/abs/2604.28123)
- [PDF](https://arxiv.org/pdf/2604.28123)
- [HuggingFace daily papers](https://huggingface.co/papers/2604.28123)

