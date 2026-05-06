---
arxiv_id: 2605.02801
title: "Reinforcement Learning for LLM-based Multi-Agent Systems through Orchestration Traces"
published: 2026-05-03
primary_category: unknown
hf_upvotes: 2
popularity_score: 9
suggested_stage: training
suggested_series: Autoresearch
fast_verdict: borderline
relevance_score: 0.7
has_deep_eval: false
abs_url: https://arxiv.org/abs/2605.02801
pdf_url: https://arxiv.org/pdf/2605.02801
hf_paper_url: https://huggingface.co/papers/2605.02801
---

# Reinforcement Learning for LLM-based Multi-Agent Systems through Orchestration Traces

**Verdict:** borderline · **Series:** Autoresearch · **Stage:** training · **Relevance:** 0.70 · **Popularity:** 9/100

> RL over multi-agent orchestration traces — Spark can train one agent at a time, multi-agent rollouts may need careful memory choreography.

## Abstract

As large language model (LLM) agents evolve from isolated tool users into coordinated teams, reinforcement learning (RL) must optimize not only individual actions but also how work is spawned, delegated, communicated, aggregated, and stopped. This paper studies RL for LLM-based multi-agent systems through orchestration traces: temporal interaction graphs whose events include sub-agent spawning, delegation, communication, tool use, return, aggregation, and stopping decisions.
  Using this lens, we identify three technical axes. First, reward design spans eight families, including orchestration rewards for parallelism speedup, split correctness, and aggregation quality. Second, reward and credit signals attach to eight credit- or signal-bearing units from token to team; explicit counterfactual message-level credit remains especially sparse in our curated pool. Third, orchestration learning decomposes into five sub-decisions: when to spawn, whom to delegate to, how to communicate, how to aggregate, and when to stop. In our curated pool as of May 4, 2026, we found no explicit RL training method for the stopping decision.
  We connect academic methods to public industrial evidence from Kimi Agent Swarm, OpenAI Codex, and Anthropic Claude Code. The resulting scale gap is a gap between publicly reported deployment envelopes and open academic evaluation regimes, not independent verification of industrial training traces. We release the artifact at https://github.com/xxzcc/awesome-llm-mas-rl, including an 84-entry tagged paper pool, a 32-record exclusion log, scripted corpus statistics, and a minimal JSON schema for replayable orchestration traces.

## Why this matters for ai-field-notes

- **Topic tags:** rl, multi-agent, orchestration, agentic, credit-assignment
- **NVIDIA stack:** NeMo, NemoClaw
- **Fast verdict rationale:** RL over multi-agent orchestration traces — Spark can train one agent at a time, multi-agent rollouts may need careful memory choreography.

## Repos

_No public repo yet._

## Citations

`citations: not yet indexed`

## Links

- [arXiv abstract](https://arxiv.org/abs/2605.02801)
- [PDF](https://arxiv.org/pdf/2605.02801)
- [HuggingFace daily papers](https://huggingface.co/papers/2605.02801)

