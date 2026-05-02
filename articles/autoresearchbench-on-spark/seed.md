---
title: 'AutoResearchBench on Spark — Spark reproduction notes'
date: 2026-05-02
author: 'Manav Sehgal'
product: 'NemoClaw'
stage: observability
difficulty: 'intermediate'
time_required: '~30 min read'
hardware: 'NVIDIA DGX Spark'
tags: [agentic, benchmark, rag, retrieval, literature-search, nim, evaluation]
summary: 'Run AutoResearchBench Deep + Wide literature-discovery tasks against three NIM-hosted Spark models (Llama 8B, Nemotron Super 49B, Llama 70B fp8) and chart where local-first agents land on a benchmark where even frontier LLMs sit under 10 percent.'
status: upcoming
series: 'Autoresearch'
---

## Source paper

- arXiv: [2604.25256](https://arxiv.org/abs/2604.25256) — AutoResearchBench: Benchmarking AI Agents on Complex Scientific Literature Discovery
- Repo: [CherYou/AutoResearchBench](https://github.com/CherYou/AutoResearchBench) (29⭐, Apache-2.0, Python+Shell, last push 2026-04-24)
- Dataset: [Lk123/AutoResearchBench](https://huggingface.co/datasets/Lk123/AutoResearchBench) (HF, obfuscated bundle)
- Popularity: **26/100** · 27 HF upvotes · 0 citations

## Frontier Scout verdict

**spark-feasible** — pure inference, NIM exposes the OpenAI-compatible endpoint the bench expects, the 29⭐ Apache-2.0 repo + HF dataset are real and pushable, and the largest in-envelope NIM model (70B fp8) leaves enough room for the retriever stack; only blocker is internet-egress for the search tools.

## Proposed Spark recipe

1. **Clone + install**: already snapshotted at `evidence/repo-snapshot/`. `cd` into it and `/opt/venv/bin/python3 -m pip install -r requirements.txt`.
2. **Start NIM with Llama 3.3 70B fp8** (or smaller in-envelope model first for plumbing). Note the OpenAI-compatible base URL (`http://localhost:8000/v1`).
3. **Configure `.env`** with `MODEL`, `OPENAI_API_KEY=local`, `OPENAI_API_BASE=http://localhost:8000/v1`, `INPUT_FILE=input_data/academic_deepsearch_example.jsonl`.
4. **Download + decrypt the bench bundle** from HF (`decrypt_benchmark.py` against the released `.obf.json`).
5. **Run inference**: `bash run_inference.sh`. The agent uses two ship-with-the-repo tools — `tool_deepxivsearch.py` (academic search) and `tool_websearch.py` (general web) — both need internet egress and likely an API key for the academic backend.
6. **Run evaluation**: `bash evaluate/run_evaluate.sh deep ...` and `bash evaluate/run_evaluate.sh wide ...`.
7. **Comparative table**: same bench against `llama-3.1-8b-instruct` (NIM), `nemotron-super-49b` (NIM), and Nemotron via NemoClaw — Spark-stack-internal leaderboard.

Full recipe with stack-map references in [`evidence/spark-recipe.md`](./evidence/spark-recipe.md).

## Open questions for the experiment

- Live search dependency: DeepXiv + web search tools need internet and likely API keys.
- Headline accuracy is low for everyone (~9% Deep, ~9% Wide). Local Spark models likely lower still — frame as "where are we on the absolute scale" not "we beat SOTA."
- Bundle decryption flow may have license/key gating; soft blocker.
- Ground-truth file for Wide eval may need separate fetching.

## Suggested article shape

- **Stage:** observability
- **Series:** Autoresearch
- **Tags:** agentic, benchmark, rag, retrieval, literature-search, nim, evaluation
- **Voice:** essay on *what literature discovery actually asks of an agent*, why even frontier LLMs sit under 10%, and what that says about the autoresearch-loop ceiling on a Spark-local stack. Most-shippable of the five — the upstream repo is real and the OpenAI-compatible endpoint drops in cleanly.
