# Transcript — source material for gpu-sizing-math-for-fine-tuning

## The gap that prompted this article

User asked: *"based on what body of knowledge we have covered in this project — are we able to answer a question like: how many and what type of NVIDIA GPUs do I need if I want to fine-tune a 100B parameter Nemotron model?"*

A survey of the corpus returned a clear "no":

- **LoRA coverage stops at 9B.** `lora-on-your-own-qa-pairs` fine-tunes a 3B Qwen2.5 base (~20 GB peak measured). `lora-fine-tune-nemotron-on-spark` planned 9B. Both are single-Spark LoRA pieces.
- **One ceiling statement for full fine-tune.** `nemo-framework-continued-pretraining-on-spark` says: *"8B at BF16 with activations checkpointed should fit. 49B won't."* That is the most explicit parameter-count-vs-memory landmark in the corpus, and it's limited to the Spark's 128 GB budget.
- **Nothing on VRAM math for >49B.** No per-parameter byte accounting, no Adam-state breakdown, no QLoRA memory formula.
- **No multi-GPU sizing.** FSDP, TP, PP are name-dropped in the NeMo piece but not sized.
- **No multi-node GPU catalog.** H100/H200/B200 per-GPU VRAM, NVLink vs PCIe, InfiniBand — not covered. Corpus is Spark-only (GB10, 128 GB unified).

The Spark is the lens of this blog, so the corpus' narrow focus is a feature, not a bug. But a reader wanting to answer "100B Nemotron fine-tune sizing" from this blog hit a dead end. This article closes the gap by teaching the math, anchored on the Spark's measurable numbers.

## Canonical sizing numbers used in the article

### Full BF16 fine-tune with Adam, mixed precision — "~16 bytes per parameter for state"

- 2 bytes weights (bf16)
- 2 bytes gradients (bf16)
- 12 bytes Adam state (fp32 momentum 4 + fp32 variance 4 + fp32 master weights 4)

At 100B: 1,600 GB state + activation term (~200 GB checkpointed, realistic) = ~1,800 GB.

Canonical references: DeepSpeed ZeRO paper (Rajbhandari et al. 2020); FSDP docs; the "20 bytes/param" figure cited in HuggingFace `Accelerate` memory docs (which adds ~4 bytes fp32 sharded grads for safety).

### LoRA rank-16 on all linears

- Trainable fraction: ~0.5-1% of parameters depending on architecture. Taken as 1% ceiling.
- Adapter state bill: 12 bytes × 1% = 0.12 bytes/param effective.

At 100B:
- Frozen base: 200 GB
- Adapter weights + grads + Adam: ~16 GB
- Activations (unchanged from full FT): ~30-60 GB
- Total: ~250 GB

### QLoRA (NF4 base + bf16 adapter + paged 8-bit optimizer)

- Frozen base: 0.5 bytes/param (NF4)
- Adapter state: same as LoRA
- Dequantization scratch: ~5 GB constant factor

At 100B:
- Frozen base: 50 GB
- Dequant scratch: ~5 GB
- Adapter bill: ~10 GB
- Activations: ~30-60 GB
- Total: ~65 GB tight, ~100 GB comfortable

Canonical reference: Dettmers et al. 2023 (QLoRA paper).

## Spark data points referenced in the article

### 3B rank-16 LoRA on Qwen2.5-3B-Instruct — "peak ~20 GB"

From `lora-on-your-own-qa-pairs/article.md` line 158:
> "I stopped the 8B NIM for the training run (freed ~10 GiB of unified memory) and left the 1B embedding NIM up. Headroom during training: ~20 GiB peak GPU memory."

Config (line 156): 29.93M trainable parameters (0.96% of 3.09B base), batch 4, grad-accum 2, bf16 SFT with gradient checkpointing.

### 49B won't fit — from the NeMo continued-pretraining piece

From `nemo-framework-continued-pretraining-on-spark/article.md` line 29:
> "8B at BF16 with activations checkpointed should fit. 49B won't."

Confirms the 16-bytes/param × 8B = 128 GB ceiling identity for the Spark's unified memory budget.

## GPU per-device VRAM used in the article

| GPU | VRAM | Source |
|---|---:|---|
| H100 | 80 GB | NVIDIA spec |
| H200 | 141 GB | NVIDIA spec |
| B200 | 192 GB | NVIDIA spec (Blackwell) |

Aggregate numbers in the article (24× H100 = 1,920 GB, 8× H100 = 640 GB, etc.) are straight multiplication.

## Editorial decisions

- **Article lives outside the three use-case arcs.** It is a foundations piece, not a step in Second Brain / LLM Wiki / Autoresearch. The editorial memory explicitly allows this: *"Does not gate articles outside the arcs. Foundational pieces, standalone comparisons, or one-offs are welcome."*
- **Stage is `foundations`, not `fine-tuning`.** The article teaches math that applies to any fine-tune; it does not run one. `also_stages: [fine-tuning]` ensures it surfaces on the fine-tuning stage filter page too.
- **Product is `Foundation`.** Not tied to a single NVIDIA product — the math applies across NeMo, HF Transformers, TRL, PEFT.
- **No screenshots.** This is a math piece, not a setup piece. The inline fn-diagram carries the thesis-in-one-glance work that screenshots would do for a hands-on piece.
- **Cross-links.** Two explicit slug-based cross-links: `lora-on-your-own-qa-pairs` (Spark 3B LoRA anchor) and `nemo-framework-continued-pretraining-on-spark` (49B landmark). These are the two existing-corpus references the "verification" section hangs on.

## What was deliberately NOT covered

- **Training data volume sizing.** Token counts, dataset memory, shuffle buffers. Separate article if needed.
- **Distillation, pruning, MoE fine-tuning.** Each changes the math non-trivially. Separate articles.
- **Inference-time KV cache sizing.** Flagged in the closing section as the next foundations piece.
- **Interconnect pricing.** The article mentions InfiniBand as a second line item but doesn't price it. Procurement-adjacent rather than sizing-math.
- **Numbers for specific NeMo-Customizer LoRA runs at 100B.** NeMo Customizer's own best-practice defaults vary; the article uses generic rank-16 all-linears as a clean ceiling.
