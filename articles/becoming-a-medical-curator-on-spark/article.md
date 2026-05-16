---
title: "Orionfold/II-Medical-8B-GGUF on Spark — five medical-reasoning variants, MedMCQA mini-eval, ChatML reasoning format"
date: 2026-05-16
author: Manav Sehgal
product: llama.cpp
stage: deployment
difficulty: intermediate
time_required: "planned ~5 hours end-to-end on a DGX Spark"
hardware: "NVIDIA DGX Spark"
tags: [gguf, quantization, medical, healthcare, orionfold, medmcqa, qwen3, chatml, reasoning, fieldkit, spark-tested]
summary: "Vertical 4 in the Orionfold series — five GGUF variants of Intelligent-Internet's II-Medical-8B (Qwen3-8B base, SFT + DAPO reasoning recipe) measured on a DGX Spark. Q5_K_M lands at 36.4 tok/s, 5.45 GB, and 52% on a MedMCQA n=50 mini-eval — slightly above F16. First reasoning-recipe pick in the series."
status: upcoming
series: Machine that Builds Machines
book_chapters: [10, 11]
fieldkit_modules: [quant, publish, eval, lineage]
also_stages: [observability]
hf_url: https://huggingface.co/Orionfold/II-Medical-8B-GGUF
---

This is the publishing receipt for the **fourth Orionfold vertical** — [`Orionfold/II-Medical-8B-GGUF`](https://huggingface.co/Orionfold/II-Medical-8B-GGUF), five GGUF quantizations of [Intelligent-Internet/II-Medical-8B](https://huggingface.co/Intelligent-Internet/II-Medical-8B), the Qwen3-8B base with an SFT + DAPO reasoning recipe targeting clinical Q&A. After finance numeric reasoning, legal binary classification, and cyber MCQ, medical is the first vertical in the series to ship a model whose generation budget actually has to account for a `<think>` block before the answer token — and that single shift exposed a footgun in the preflight harness that the prior three cards never had to face.

The narrative thread: this is the second vertical in a row to ship with **zero new code in `fieldkit` itself**. The infrastructure shape — `g3_build_first_quant.sh` per-model case, JSONL bench loader, `mcq_letter` scorer, prompt-format wrapper — absorbed the medical pick as a configuration change. The one real footgun was a missing `chatml` branch in the preflight prompt-format detector that was silently dropping `<|im_start|>` wrapping and would have eaten any future ChatML model. That patch landed in `scripts/g3_preflight_bench.py` and now stands ready for whichever ChatML pick comes next.

This article is in flight — pairs with the [becoming a GGUF publisher on Spark](/articles/becoming-a-gguf-publisher-on-spark/) deep-dive (shared evidence at `evidence/lineage-II-Medical-8B/`). Full draft to follow once the live HF push lands.
