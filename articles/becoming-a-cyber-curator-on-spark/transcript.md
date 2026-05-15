# Source material: becoming-a-cyber-curator-on-spark

Cleaned session log and provenance for this article. Raw material that became evidence in article.md lives here.

_Populated on 2026-05-15._

## Session origin

Third installment of the vertical-curator series. Spawned after the Saul HF push (`Orionfold/Saul-7B-Instruct-v1-GGUF`) was verified live (HTTP 200, all 5 GGUFs visible), and fieldkit v0.4.1 had shipped on PyPI. The handoff from session 8 named the cyber pick: `ZySec-AI/SecurityLLM` from the `hf-model-scout` cyber-7B report (decision-gated on writing `scripts/cyber_merge.py` since cyber lacks a canonical bench like FinanceBench / LegalBench).

## Model pick — ZySec-AI/SecurityLLM

Verified via the HF API + raw card files before committing to download:

- **Arch:** Mistral 7B (`MistralForCausalLM`, hidden 4096, layers 32, vocab 32000, ctx 32768) — llama.cpp-supported.
- **License:** apache-2.0 (Orionfold-commercial-tier compatible).
- **Chat template:** Zephyr (`<|user|>\n{content}{eos}\n<|assistant|>\n`) — present in `tokenizer_config.json`.
- **Training:** Zephyr-7B-beta lineage, DPO over 30+ cyber domains (cryptography, network security, governance, etc.).
- **Storage:** 3× safetensors files (4.9 GB + 5.0 GB + 4.5 GB = 14.4 GB FP16).

All four scout traps from [[project_orionfold_parent_brand]] cleared in the same gate that caught the V1 finance failure: not an `instruction-pretrain/*` model, has a real chat_template, llama.cpp-supported arch, fits in unified memory.

## Bench pick — tihanyin/CyberMetric

Cyber has no FinanceBench/LegalBench-equivalent canonical bench. Surveyed candidates:

- **CyberMetric (tihanyin/CyberMetric)** — 4-option MCQ, arxiv 2402.07688, **apache-2.0**, balanced 20/20/20/20 across A/B/C/D. Picked.
- **SecQA (zefang-liu/secqa)** — apache-2.0 alternative but CC-BY-NC-SA-4.0; skip for commercial-tier card.
- **CTI-Bench (AI4Sec/cti-bench)** — CC-BY-NC-SA-4.0; same skip reason.
- **CyberBench / E2E-Cyber-Bench (Leop0ld/e2e-cyber-bench)** — license unclear.

CyberMetric-80 (the smallest release) sampled to 50 rows deterministically with `random.seed(42)` — matches the finance/legal mini-eval scale of 50 questions.

## Three scripts deltas (no fieldkit changes)

Workflow shape carried forward from the Saul release. The only delta vs that pipeline was three local script patches:

1. **`scripts/cyber_merge.py`** — new. Emits 50 rows of `{id, text, answer, task}` JSONL from `tihanyin/CyberMetric`'s 80-question release. Prompt template builds the 4-option MCQ + "reply with only the single letter A, B, C, or D" instruction. Gold = the solution letter.

2. **`scripts/g3_measure_variants.py`** — patched. Added `cybermetric` to the `VERTICAL_BENCH` valid list (alongside `financebench` / `legalbench`). Added `_DEFAULT_DOMAIN["cybermetric"] = "vertical-curator-cyber"`, baseline=ZySec-AI/SecurityLLM, dataset=tihanyin/CyberMetric. Added `CYBERBENCH_JSONL` + `CYBERBENCH_LIMIT` env knobs. New local `mcq_letter` scorer (~15 lines, regex-based, prefers "Answer: X" markers, falls back to first word-bounded `[A-D]`). New `_wrap_zephyr` prompt wrapper alongside the existing `_wrap_inst`. Per-vertical wrapper dispatch in `measure_variant`.

3. **`scripts/g3_preflight_bench.py`** — patched. Added `cybermetric` to the preflight valid list. Extended `_detect_prompt_format` to recognize zephyr (`<|user|>` + `<|assistant|>` in chat_template). Extended `_format_prompt` to emit the zephyr wrapper. Local `mcq_letter` scorer (same impl as measure script). Vertical-bench dispatch in `main()`.

4. **`scripts/g3_build_first_quant.sh`** — patched. Added a `ZySec-AI/SecurityLLM` case to the model-override switch (`MODEL_LICENSE=apache-2.0`, `CHAT_FORMAT=zephyr`, `VERTICAL_BENCH=cybermetric`, `ARTICLE_SLUG=becoming-a-cyber-curator-on-spark`). Added `CYBERBENCH_JSONL` env propagation through `step_preflight_bench` and `step_measure`.

`fieldkit` source files modified: **zero**. PyPI package version on this release's commit is `0.4.1` — same as the prior legal release.

## Preflight gate — 3/5 PASS

The preflight bench fired clean on F16 GGUF (converted from FP source via `convert_hf_to_gguf.py`, 6.9 sec). Five CyberMetric questions × ~15 sec each via llama-server on GPU. Score: 3/5, well above the abort-on-zero threshold.

The two failures were compliance failures, not factual failures:

- Q2 expected="B" — gold answer was "CISO" (option B). Model output: "The Chief Information Security Officer (CISO) is indeed responsible for implementing the planning, budgeting, and performance of an organization's information security components..." → `mcq_letter` correctly returned 0 because no word-bounded "B" appeared.
- Q4 expected="C" — gold answer was a surveillance-technique option. Model output: "In the context of computer/network surveillance, it is essential to understand the various forms of monitoring and surveillance techniques used to ensure security and compliance. \n\n1. Keyboard Monitor..." → `mcq_letter` returned 0 because the response launched into a numbered list and was truncated by the n_predict cutoff before any A/B/C/D letter appeared.

Both are real compliance gaps — the model knew the answer but didn't follow the "reply with only the letter" instruction. A `contains` scorer would have hidden these as false positives (matching "B" in "CISO" or "C" in "Keyboard"). The `mcq_letter` scorer surfaces them honestly, which is the right tradeoff for vibe-bench validity.

## Smoke-test (scorer correctness)

Before running the long measure cycle, validated `mcq_letter` against 11 hand-crafted predicted-vs-gold pairs. All 11 passed:

```
[ok] pred='B'                            want=1.0 got=1.0
[ok] pred='b'                            want=1.0 got=1.0
[ok] pred='B.'                           want=1.0 got=1.0
[ok] pred='Answer: B'                    want=1.0 got=1.0
[ok] pred='The answer is B.'             want=1.0 got=1.0
[ok] pred='I think it is option B'       want=1.0 got=1.0
[ok] pred='A'                            want=0.0 got=0.0
[ok] pred=''                             want=0.0 got=0.0
[ok] pred='Buffer overflow'              want=0.0 got=0.0
[ok] pred='C) Buffer overflow'           want=0.0 got=0.0
[ok] pred='I would say A or B'           want=0.0 got=0.0
```

## Open at scaffold time

- Measure step (5 variants × 4 axes × 50 questions) was still running when this scaffold was written. Estimated ~1 hour.
- Numbers in the article body table are `TODO` placeholders until measure lands.
- HF push to `Orionfold/SecurityLLM-GGUF` is queued behind measure.
- Article is `status: upcoming` until measure completes + numbers + at least one inline `fn-diagram` are added in a polish pass.

## References

- [HuggingFace: ZySec-AI/SecurityLLM](https://huggingface.co/ZySec-AI/SecurityLLM)
- [HuggingFace: tihanyin/CyberMetric (dataset)](https://huggingface.co/datasets/tihanyin/CyberMetric)
- [arxiv 2402.07688 — CyberMetric paper](https://arxiv.org/abs/2402.07688)
- Prior verticals in the series:
  - [becoming-a-gguf-publisher-on-spark](/articles/becoming-a-gguf-publisher-on-spark/) (finance)
  - [becoming-a-legal-curator-on-spark](/articles/becoming-a-legal-curator-on-spark/) (legal)
- Memories carried into this session: `[[feedback_preflight_bench_before_quant]]`, `[[feedback_chat_vs_continued_pretrain_trap]]`, `[[feedback_hf_upload_resilient_api]]`, `[[project_orionfold_parent_brand]]`.
