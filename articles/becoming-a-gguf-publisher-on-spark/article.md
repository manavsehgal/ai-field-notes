---
title: "Becoming a GGUF publisher on the Spark — first quant, first 10K downloads, the Spark-tested angle"
date: 2026-05-12
author: Manav Sehgal
product: llama.cpp + HuggingFace Hub + Nemotron family
stage: deployment
difficulty: intermediate
time_required: "14-day milestone: 5 quants published, one in HF trending sidebar"
hardware: "NVIDIA DGX Spark"
tags: [gguf, llama-cpp, huggingface, orionfold, quantization, nemotron, spark-tested, publishing, dgx-spark]
summary: "Bartowski's audience has 11,869 followers; TheBloke's 26.9K-follower seat has been dormant since 2024-01-31. The Spark's 128 GB unified memory uniquely positions it for the 70B+ tier where consumer GPUs can't compete. A 14-day plan to ship the first 5 Orionfold GGUFs — Nemotron-family models, every card carrying perplexity, tok/s, and thermal envelope measured on a real GB10."
status: upcoming
series: Machine that Builds Machines
book_chapters: [10, 11]
fieldkit_modules: [quant, publish, lineage]
also_stages: [observability]
---

## What this article will answer

This article documents the build-out of the **Orionfold** GGUF-publisher brand on the Spark — the v0 of MTBM Pick #1. The thesis (per `ideas/mtbm-use-cases.md` §6): there is an unclaimed audience seat in the GGUF publishing surface, and a 128 GB unified-memory Spark is the right rig to claim it from. The article walks through:

- Why GGUF, why now (TheBloke dormant since 2024-01-31; `qwopqwop200` 404'd; Bartowski covers ~70% of newly-released models within 24 h — the gap is the 30%).
- The Orionfold positioning angle — Nemotron-family models layered with **Spark-tested-and-runs** measurement: every quant card carries perplexity (wikitext-2), sustained `tok/s` on GB10, and a thermal-envelope duty-cycle disclosure. None of the incumbent publishers ships all three.
- The `fieldkit.quant` + `fieldkit.publish` pipeline that turns one Nemotron release into Q4_K_M / Q5_K_M / Q6_K / Q8_0 / F16 GGUFs with a deterministic model card, an HF push, and a per-artifact YAML manifest that Mac-side catalog pages render automatically.
- The 14-day milestone: **5 quants published, ≥ 50 followers on `huggingface.co/orionfoldllc`, ≥ 10K aggregate downloads, one quant in the HF "trending GGUFs" sidebar, first GitHub Sponsor secured.**

## NVIDIA technologies to be covered

- **DGX Spark (GB10)** — the 128 GB unified memory budget is what makes 70B-class GGUFs realistic in a one-machine pipeline. The article quantifies how much headroom Q4_K_M / Q5_K_M / Q6_K each leave at typical chat batch sizes.
- **Nemotron family** — Nemotron-3-Nano-30B-A3B and adjacent open releases. Why the editorial continuity with the blog's NVIDIA line gives Orionfold a defensible niche the model-agnostic Bartowski-shape can't claim.
- **TensorRT-LLM + NVFP4 (forward reference)** — the v0.5 `fieldkit.quant.quantize_nvfp4` stub points at the FP4-aware quantization Blackwell will surface; this article notes the stub and explains why the v0 stays GGUF-only.

## What I expect to find

The bottleneck is not quantization wall-clock — `llama-quantize` is fast once the F16 GGUF exists. The bottleneck is the **measurement triple**: perplexity over wikitext-2 (~30 min per variant on GB10), sustained-load thermal probing (~60 min per variant to capture throttle onset), and the `llama-bench` tok/s pass (~10 min). At five variants per model, that's ~8 hours of measurement to produce one model card. The article will document how the `fieldkit.quant.ThermalProbe` loop overlaps with the perplexity pass so the wall stays under 3 hours end-to-end.

I expect the first quant to clear the HF "trending GGUFs" sidebar within 72 hours of publication, based on the Bartowski 24-hour acceleration pattern. The disconfirming signal would be that the Nemotron audience is too narrow and Q1's (a)-vs-(d) choice in the HANDOFF should flip to (d) standalone (model-agnostic "Spark-tested" positioning) by the second model.

## Where it sits in the arc

Anchor article for the **MTBM forge station** at the publishing surface. Builds directly on `auto-research-loop-on-spark` (which surfaced `fieldkit.lineage`) and reuses every prior `fieldkit.eval` and `fieldkit.lineage` pattern. The next article after this is either:

- (sequential path, per HANDOFF Q10) **`from-brand-brief-to-civitai-1`** — G9 LoRA publisher v0 once the first 5 GGUFs are out the door.
- (parallel path, if Q10 flips) the G3 second-cycle article documenting `fieldkit.quant.quantize_awq` extension.

## Field-evidence sketch (to be filled by the v0 run)

- 5 Orionfold GGUFs at `huggingface.co/orionfoldllc` (Bartowski-shape repo names: `orionfoldllc/<model>-GGUF`).
- 5 corresponding manifests at `src/content/artifacts/<slug>.yaml` (Phase-2 sync contract: Mac destination renders `/artifacts/quant/` catalog page).
- One reproducible `fieldkit.lineage` run per quant, hashed against the wikitext-2 calibration corpus.
- A `compare-table.md` cross-checking Orionfold perplexity numbers against Bartowski's published values for the same model where overlap exists (build trust by reproducing the easy cases, then differentiating on Spark-tested measurement).
