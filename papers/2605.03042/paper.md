---
arxiv_id: 2605.03042
title: "ARIS: Autonomous Research via Adversarial Multi-Agent Collaboration"
published: 2026-05-03
primary_category: unknown
hf_upvotes: 65
popularity_score: 33
suggested_stage: agentic
suggested_series: Autoresearch
fast_verdict: spark-feasible
relevance_score: 0.85
has_deep_eval: false
abs_url: https://arxiv.org/abs/2605.03042
pdf_url: https://arxiv.org/pdf/2605.03042
hf_paper_url: https://huggingface.co/papers/2605.03042
---

# ARIS: Autonomous Research via Adversarial Multi-Agent Collaboration

**Verdict:** spark-feasible · **Series:** Autoresearch · **Stage:** agentic · **Relevance:** 0.85 · **Popularity:** 33/100

> Open-source autonomous-research harness with adversarial review — direct Autoresearch-arc material; runs over hosted NIM endpoints.

## Abstract

This report describes ARIS (Auto-Research-in-sleep), an open-source research harness for autonomous research, including its architecture, assurance mechanisms, and early deployment experience. The performance of agent systems built on LLMs depends on both the model weights and the harness around them, which governs what information to store, retrieve, and present to the model. For long-horizon research workflows, the central failure mode is not a visible breakdown but a plausible unsupported success: a long-running agent can produce claims whose evidential support is incomplete, misreported, or silently inherited from the executor's framing. Therefore, we present ARIS as a research harness that coordinates machine-learning research workflows through cross-model adversarial collaboration as a default configuration: an executor model drives forward progress while a reviewer from a different model family is recommended to critique intermediate artifacts and request revisions. ARIS has three architectural layers. The execution layer provides more than 65 reusable Markdown-defined skills, model integrations via MCP, a persistent research wiki for iterative reuse of prior findings, and deterministic figure generation. The orchestration layer coordinates five end-to-end workflows with adjustable effort settings and configurable routing to reviewer models. The assurance layer includes a three-stage process for checking whether experimental claims are supported by evidence: integrity verification, result-to-claim mapping, and claim auditing that cross-checks manuscript statements against the claim ledger and raw evidence, as well as a five-pass scientific-editing pipeline, mathematical-proof checks, and visual inspection of the rendered PDF. A prototype self-improvement loop records research traces and proposes harness improvements that are adopted only after reviewer approval.

## Why this matters for ai-field-notes

- **Topic tags:** agentic, autoresearch, multi-agent, orchestration, evals
- **NVIDIA stack:** NIM, NemoClaw, Guardrails
- **Fast verdict rationale:** Open-source autonomous-research harness with adversarial review — direct Autoresearch-arc material; runs over hosted NIM endpoints.

## Repos

_No public repo yet._

## Citations

`citations: not yet indexed`

## Links

- [arXiv abstract](https://arxiv.org/abs/2605.03042)
- [PDF](https://arxiv.org/pdf/2605.03042)
- [HuggingFace daily papers](https://huggingface.co/papers/2605.03042)

