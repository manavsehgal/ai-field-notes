# Source material: unsloth-on-spark-feasibility

Cleaned session log and provenance for this article. The article walks six gates Unsloth had to clear before the v2 patent-strategist train commits to it instead of TRL+PEFT. Evidence lives on disk at `/home/nvidia/data/aifn-train-lora/` and in `/tmp/hf-scout/2026-05-19/patent-strategist-8B/`.

_Populated on 2026-05-19._

## Companion docs

- `ideas/c2-progress-2026-05-19.md` — the canonical evidence doc; Q1+Q1b+Q2+Q3+Q5+Q6 gate table, eight surprises, install profile, comparison vs s40 baseline, artifacts catalog.
- `ideas/unsloth-strategy.md` — the s38 strategy frame (5.4k words) the article responds to.
- `/tmp/hf-scout/2026-05-19/patent-strategist-8B/report.md` — the C1 base-model pick that landed on `nvidia/Llama-3.1-Nemotron-Nano-8B-v1`.

## Evidence files (logs on disk)

- `/home/nvidia/data/aifn-train-lora/unsloth-install-2026-05-19.log` — gate 1 pip install transcript.
- `/home/nvidia/data/aifn-train-lora/unsloth-install-profile-2026-05-19.json` — pinned versions snapshot for the reproducibility appendix.
- `/home/nvidia/data/aifn-train-lora/unsloth-load-smoke-v2.log` — gate 2 warm-cache load (110 s, 16.94 GB peak, GEN OK).
- `/home/nvidia/data/aifn-train-lora/unsloth-smoke-train.py` + `unsloth-smoke-train.log` — gate 3 (100 steps, 1.21 s/step, BF16 preserved, adapter saved).
- `/home/nvidia/data/aifn-train-lora/unsloth-smoke-2026-05-19/adapter/` — the trained LoRA adapter (54 MB safetensors + tokenizer + chat_template.jinja).
- `/home/nvidia/data/aifn-train-lora/unsloth-gguf-roundtrip.py` + `unsloth-gguf-roundtrip.log` — gate 5 (LOAD OK, MERGE OK, BOTH QUANTS OK in 207 s).
- `/home/nvidia/data/aifn-train-lora/unsloth-smoke-2026-05-19/gguf_gguf/Llama-3.1-Nemotron-Nano-8B-v1.{Q4_K_M,Q8_0}.gguf` — gate 5 deliverables (4.6 GB + 8.0 GB).

## Gate-by-gate event log (cleaned)

### Gate 1 — install

```bash
docker exec ps-train pip install --no-deps unsloth unsloth_zoo bitsandbytes
# Three packages, six seconds of resolution. s40 stack untouched:
# transformers 5.8.1 / peft 0.19.1 / trl 1.4.0 / accelerate 1.13.0 /
# torchao 0.16.0 / flash_attn 2.7.4.post1+25.11.
#
# python3 -m bitsandbytes  →  SUCCESS  CUDA_VERSION=130  CC=(12,1)
# Unsloth import:  6.3 s, FA2: True, Xformers: None
```

### Gate 2 — load

```python
# Cold cache (first run): download 4 shards in 333s + load 291 weight files
# in ~122s → total 455s.
# Warm cache (rerun): 110s. Peak alloc 16.94 GB.
# Chat template length: 2004 chars. <|start_header_id|> marker present.
# Gen smoke: "Reply with exactly one word: hello." → 'Hello.' exactly.
```

### Gate 3 — wrap + train

100-step LoRA SFT on 50 inline arithmetic rows. Loss curve:

```
step  10  →  5.83
step  20  →  0.83
step  30  →  0.42
step  40  →  0.37
step  50  →  0.29
step  60  →  0.14
step  70  →  0.12
step  80  →  0.12
step  90  →  0.12
step 100  →  0.12
```

Per-step time: 1.21 s. Peak alloc end-to-end: 16.94 GB (identical to gate 2). First-param dtype after train: `torch.bfloat16` — BF16 preserved. Adapter saved at `/home/nvidia/data/aifn-train-lora/unsloth-smoke-2026-05-19/adapter/` (54 MB safetensors + tokenizer + chat_template.jinja).

### Gate 5 — `save_pretrained_gguf()`

Single call:
```python
model.save_pretrained_gguf(
    "/home/nvidia/data/aifn-train-lora/unsloth-smoke-2026-05-19/gguf",
    tokenizer,
    quantization_method=["q8_0", "q4_k_m"],
)
```

Wall: 207.4 s end-to-end (110.5 s reload + 96.9 s merge/convert/quantize). Output sequence:

```
Unsloth: Merging weights into 16bit:   0%|          | 0/4 [00:00<?, ?it/s]
Unsloth: Merging weights into 16bit:  25%|██▌       | 1/4 [00:11<00:35, 11.96s/it]
Unsloth: Merging weights into 16bit:  50%|█████     | 2/4 [00:25<00:26, 13.13s/it]
Unsloth: Merging weights into 16bit:  75%|███████▌  | 3/4 [00:38<00:12, 12.91s/it]
Unsloth: Merging weights into 16bit: 100%|██████████| 4/4 [00:40<00:00, 10.17s/it]
Unsloth: Converting to GGUF format...
==((====))==  Unsloth: Conversion from HF to GGUF information
   \\   /|    [0] Installing llama.cpp might take 3 minutes.
O^O/ \_/ \    [1] Converting HF to GGUF bf16 might take 3 minutes.
\        /    [2] Converting GGUF bf16 to ['q8_0', 'q4_k_m'] might take 10 minutes each.
 "-____-"     In total, you will have to wait at least 16 minutes.

Unsloth: Installing llama.cpp. This might take 3 minutes...
Unsloth: Cloning llama.cpp repository...
Unsloth: Building llama.cpp - please wait 1 to 3 minutes
Unsloth: Successfully installed llama.cpp!
Unsloth: Preparing converter script...
Unsloth: [1] Converting model into bf16 GGUF format.
Unsloth: Initial conversion completed!
Unsloth: [2] Converting GGUF bf16 into q8_0. This might take 10 minutes...
Unsloth: [2] Converting GGUF bf16 into q4_k_m. This might take 10 minutes...
Unsloth: All GGUF conversions completed successfully!
Generated files:
  Llama-3.1-Nemotron-Nano-8B-v1.Q4_K_M.gguf
  Llama-3.1-Nemotron-Nano-8B-v1.Q8_0.gguf
```

Final size check (manual ls; the script's `os.walk()` returned empty because of the `_gguf` suffix quirk):

```
Llama-3.1-Nemotron-Nano-8B-v1.Q4_K_M.gguf  4.6 GB  (predicted band 4.0–5.5)
Llama-3.1-Nemotron-Nano-8B-v1.Q8_0.gguf    8.0 GB  (predicted band 7.5–9.5)
```

### Gate 6 — llama.cpp round-trip

```bash
/home/nvidia/llama.cpp/build/bin/llama-completion \
    -m .../Llama-3.1-Nemotron-Nano-8B-v1.Q4_K_M.gguf \
    -p "What is 5+5?" -n 64 --temp 0
# Output: "5 + 5 = 10."
# Perf: 25 tokens in 251 ms (Q4_K_M) / 375 ms (Q8_0).
# Memory: model 4403 MiB + 16K ctx 16384 MiB = 21 GB total (Q4_K_M).
#         model 7605 MiB + 16K ctx 16384 MiB = 24 GB total (Q8_0).
```

First attempt of this gate failed because `llama-cli -no-cnv` is no longer honored — see the article's tradeoffs section. The 5 GB log file of empty `>` prompts that produced was deleted; `llama-completion` is the one-shot binary the b1 build steers users toward.

## Decisions made during the session

- **Article-scaffold branch, not feasibility-decision doc.** All four gate-5 pass criteria green → strategy doc says scaffold the article (this one).
- **Single inline fn-diagram, not two.** The flow-pipeline-with-recurring-16.94-GB visual carries both the architectural-context job AND the thesis-reinforcement job — a second diagram would have been decoration. Kept the ceiling at one.
- **`product: Foundation`, not a specific NVIDIA product.** Same convention as the W1 and W2 predecessor articles in this thread — the Unsloth + Nemotron + llama.cpp combo doesn't map to one of the listed NVIDIA-product slots, and `Foundation` is the established escape hatch.
- **Kept intermediate `gguf/` merged-BF16 safetensors (15 GB) on disk.** They're throwaway but disk has 2.5T free. Cleanup can wait for the v2 production train or a deliberate disk-pressure pass.

## What was scrubbed before publishing

- No secrets, keys, hostnames, or PII appear in the source material; nothing was redacted.
- Internal SYNC-HANDOFF Q-references and HANDOFF banner text were paraphrased, not quoted verbatim, to keep this article reader-facing rather than internal-planning-facing per voice-and-style.md's customer-link audit guidance.
