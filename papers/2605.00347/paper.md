---
arxiv_id: 2605.00347
title: "Odysseus: Scaling VLMs to 100+ Turn Decision-Making in Games via Reinforcement Learning"
published: 2026-04-30
primary_category: unknown
hf_upvotes: 12
popularity_score: 19
suggested_stage: training
suggested_series: Autoresearch
fast_verdict: borderline
relevance_score: 0.65
has_deep_eval: false
abs_url: https://arxiv.org/abs/2605.00347
pdf_url: https://arxiv.org/pdf/2605.00347
hf_paper_url: https://huggingface.co/papers/2605.00347
---

# Odysseus: Scaling VLMs to 100+ Turn Decision-Making in Games via Reinforcement Learning

**Verdict:** borderline · **Series:** Autoresearch · **Stage:** training · **Relevance:** 0.65 · **Popularity:** 19/100

> 100+ turn VLM RL on Mario — small-VLM fits, but 100+ turn rollouts × RL may push the unified-memory budget; verify in eval.

## Abstract

Given the rapidly growing capabilities of vision-language models (VLMs), extending them to interactive decision-making tasks such as video games has emerged as a promising frontier. However, existing approaches either rely on large-scale supervised fine-tuning (SFT) on human trajectories or apply reinforcement learning (RL) only in relatively short-horizon settings (typically around 20--30 turns). In this work, we study RL-based training of VLMs for long-horizon decision-making in Super Mario Land, a visually grounded environment requiring 100+ turns of interaction with coordinated perception, reasoning, and action. We begin with a systematic investigation of key algorithmic components and propose an adapted variant of PPO with a lightweight turn-level critic, which substantially improves training stability and sample efficiency over critic-free methods such as GRPO and Reinforce++. We further show that pretrained VLMs provide strong action priors, significantly improving sample efficiency during RL training and reducing the need for manual design choices such as action engineering, compared to classical deep RL trained from scratch. Building on these insights, we introduce Odysseus, an open training framework for VLM agents, achieving substantial gains across multiple levels of the game and at least 3 times average game progresses than frontier models. Moreover, the trained models exhibit consistent improvements under both in-game and cross-game generalization settings, while maintaining general-domain capabilities. Overall, our results identify key ingredients for making RL stable and effective in long-horizon, multi-modal settings, and provide practical guidance for developing VLMs as embodied agents.

## Why this matters for ai-field-notes

- **Topic tags:** agentic, vlm, rl, long-horizon, multimodal
- **NVIDIA stack:** NeMo, NemoClaw
- **Fast verdict rationale:** 100+ turn VLM RL on Mario — small-VLM fits, but 100+ turn rollouts × RL may push the unified-memory budget; verify in eval.

## Repos

_No public repo yet._

## Citations

`citations: 0`

## Links

- [arXiv abstract](https://arxiv.org/abs/2605.00347)
- [PDF](https://arxiv.org/pdf/2605.00347)
- [HuggingFace daily papers](https://huggingface.co/papers/2605.00347)

