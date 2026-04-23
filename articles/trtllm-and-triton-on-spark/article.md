---
title: "TensorRT-LLM + Triton on a DGX Spark — Life Without NIM"
date: 2026-05-21
author: Manav Sehgal
product: TensorRT-LLM + Triton Inference Server
stage: deployment
difficulty: advanced
time_required: "planned ~1 day including engine build"
hardware: "NVIDIA DGX Spark"
tags: [deployment, tensorrt-llm, triton, fp8, engine-build, cold-start, second-brain, dgx-spark]
summary: "A planned rebuild of the 8B-NIM baseline using raw TensorRT-LLM engines served by Triton. The prize is lower per-token latency and a tighter memory footprint; the cost is a multi-step engine-build pipeline. For whom does the flexibility outweigh NIM's batteries-included comfort?"
status: upcoming
---

## What this article will answer

NIM's convenience has a tax: a fixed serving stack and limited knobs. For a Second Brain that hits one endpoint thousands of times a day, is it worth the complexity to drop to TensorRT-LLM + Triton directly?

## NVIDIA technologies to be covered

- **TensorRT-LLM engine build** — `trtllm-build` against an FP8 Llama 3.1 8B checkpoint; max batch, max input/output, paged KV cache sizing.
- **Triton Inference Server** — `tensorrtllm_backend`, per-model config, dynamic batching, concurrent model instances on a single GPU.
- **FP8 quantization quirks** — calibration sets, accuracy regressions vs the NIM baseline, when FP8 is a free lunch and when it isn't.
- **Docker Compose + systemd** — making the engine survive a reboot without hand-holding.

## What I expect to find

Raw TRT-LLM should shave 10–20 ms off first-token latency for the Second Brain's query pattern, at the cost of a 30-minute engine rebuild every time the model version moves. The article closes with a decision rule for when to go raw and when to stay on NIM.

## Where it sits in the arc

First **Second Brain** specialization (S1). Presumes the foundation install chain is complete and the 8B-NIM baseline is what we're trying to beat.
