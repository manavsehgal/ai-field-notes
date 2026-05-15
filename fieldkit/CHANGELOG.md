# Changelog

All notable changes to `fieldkit` are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). While the package is on `0.x`, minor versions may include breaking changes. `1.0` will mark API stability.

## [Unreleased]

### Added — `fieldkit.publish` card-rendering polish (v0.4.2 candidates)

- **`ModelCard.llama_cpp_example_prompt: Optional[str]`** — new field. Threads through `publish_quant(..., llama_cpp_example_prompt=...)` and from a duck-typed report's `.llama_cpp_example_prompt` attribute. The default `## How to run` body's `llama-cpp-python` snippet now uses this string for the user-message; when omitted it falls back to a neutral `"Summarize the key idea in one paragraph."` placeholder instead of the previously-hardcoded `"Explain working capital."` (which leaked into the legal + cyber vertical cards on first push). Multi-line MCQ-shaped prompts are JSON-escaped (`\n`) so the snippet stays single-line + valid Python — caller passes the raw prompt, the renderer handles escaping.
- **Side fix:** the previous renderer rendered the hardcoded finance prompt on every vertical card; the cyber + legal cards on HF were patched out-of-band on 2026-05-15. Going forward, every `publish_quant` call should pass `llama_cpp_example_prompt=...` matching the article's "Using this release" section, per `[[feedback_customer_link_audit]]`.

## [0.4.1] — 2026-05-14

Patch release. The `fieldkit.eval.VerticalBench` overlay introduced in v0.4.0 needed two kwargs to score FinanceBench correctly (open-book context-prepend) and to bound a JSONL slice (subset filter on `question_type`). Both lifts came out of the 2026-05-13 V1 attempt on `AdaptLLM/finance-chat` (0/50 closed-book vs. 14–18%/50 open-book on the same JSONL) and the 2026-05-14 legal-curator scoring run on `Equall/Saul-7B-Instruct-v1`. The two scripts under `scripts/g3_*` that carried duplicated loaders now call into the package surface. No new modules, no new public classes — additive kwargs only.

### Added — `fieldkit.eval.VerticalBench` open-book mode

- **`VerticalBench.from_jsonl(..., open_book=...)`** — new kwarg. When `True`, FinanceBench rows have their `evidence[*].evidence_text` prepended to the question (templated as "Context from <doc>: …\n\nQuestion: …\n\nAnswer with just the numeric value.") so the model sees the 10-K excerpt the gold answer was derived from. Default `None` auto-resolves to `True` for `financebench` and `False` for `legalbench` / `generic` — the right defaults per benchmark convention. Lifts inline `_load_finbench_open_book` helpers from `scripts/g3_preflight_bench.py` and `scripts/g3_measure_variants.py` into the package surface; both scripts now call `VerticalBench.from_jsonl(open_book=True, subset=…)` instead of carrying duplicated loaders. The 2026-05-13 V1 attempt on AdaptLLM/finance-chat scored 0/50 closed-book and 14–18%/50 open-book on the same JSONL — open-book is the load-bearing flag for FinanceBench scoring.
- **`VerticalBench.from_jsonl(..., subset=...)`** — new kwarg. FinanceBench-only convenience filter on the `question_type` column. Drops non-matching rows before the loader hits the `limit` cap, so callers can score the `metrics-generated` subset with `limit=50` and get 50 metrics-generated questions (not 50 mixed rows of which N are metrics-generated).

### Test suite

**+8 new tests** on `TestOpenBook` in `tests/test_vertical_bench.py` covering: auto-default for financebench, explicit `False` keeps closed-book, missing-evidence falls back to closed-book, legalbench / generic are no-ops, list-of-strings evidence shape, subset filter, subset × limit composition. Total: **375 passed, 3 skipped** offline (`pytest -q`). The 3 skips are the two `--spark`-gated live-integration tests + the `torch`-import skip in `test_training.py` (CPU-only venv).

### Articles in this release

- [`becoming-a-legal-curator-on-spark`](https://ainative.business/field-notes/becoming-a-legal-curator-on-spark/) — second Orionfold quant card, swaps FinanceBench for a curated 5-task LegalBench subset. Drives the `subset` kwarg's first non-finance use (LegalBench tasks via `legalbench` format) and validates that the `open_book` default-off branch is correct for LegalBench JSONLs.

### Verified on Spark

- **Live HF push:** `Orionfold/Saul-7B-Instruct-v1-GGUF` (5 GGUF variants + README, ~37 GB) shipped 2026-05-14 via the same `publish_quant(dry_run=False)` path the finance-chat card used a week earlier. Zero source changes in `fieldkit.publish` between the two pushes — the v0.4.0 surface generalized as designed.

## [0.4.0] — 2026-05-14

Fourth public release. Two new top-level modules (`fieldkit.publish` + `fieldkit.quant`) for the G3 GGUF / Quantization Publisher pick (MTBM Pick #1 per `ideas/mtbm-use-cases.md` §6), the v0.4.x **vertical-curator overlay** on `fieldkit.eval` (`VerticalBench`), and post-dry-run card-rendering fixes that landed the first live HF push (`Orionfold/finance-chat-GGUF`). The two new modules together unlock most of Cluster G; this cut implements the GGUF critical path and stubs the other quant formats with named entry points pointing at the v0.5+ roadmap.

### Added — `fieldkit.publish` (new module)

HuggingFace Hub adapter + auto model card builder from `fieldkit.lineage`. Three public surfaces:

- **`fieldkit.publish.ModelCard`** — frontmatter + body builder. Renders the canonical card every Orionfold artifact gets: YAML frontmatter (license, library_name, base_model, pipeline_tag, tags, model_creator), a title + elevator, a **Spark-tested** block (per-variant perplexity + tok/s + thermal envelope), a variants table, **How to run** (`ollama pull` + `from_pretrained` snippets), an optional **Lineage** block (rendered from a `fieldkit.lineage.LineageStore` if provided), a **Methods** backlink to `ainative.business/field-notes/<slug>/`, and a footer attributing the publication to Orionfold LLC.
- **`fieldkit.publish.ArtifactManifest`** — frozen dataclass for the `src/content/artifacts/<slug>.yaml` Phase-2 sync record (per memory `project_artifact_manifests_phase2`). `to_yaml()` emits via a hand-rolled stdlib emitter so the module has no runtime YAML dep. The source repo writes one of these per push; the Mac destination renders `/artifacts/<kind>/` catalog pages from `getCollection('artifacts')`.
- **`fieldkit.publish.HFHubAdapter`** — lazy-`huggingface_hub` wrapper. Defaults to `dry_run=True` (stages files on disk, logs the would-be calls, no network). Flip `dry_run=False` to push via `HfApi().upload_folder(...)`. Token resolution order: explicit `token=` → `HF_TOKEN` env → cached login. The dry-run path is fully testable offline.

Plus an orchestrator: **`fieldkit.publish.publish_quant(...)`** — one-line caller that ingests a `QuantReport`-shaped object (duck-typed; produced by `fieldkit.quant.quantize_gguf`), renders the card, writes the manifest, stages the variant files, and pushes (or dry-runs) the HF commit.

Branded constants: `ORIONFOLD_BRAND = "Orionfold LLC"`, `ORIONFOLD_HF_HANDLE = "Orionfold"` (was `ORIONFOLD_HF_ORG = "orionfoldllc"` until 2026-05-14, when publishing moved to the existing user-account handle — Bartowski-shape personal handle precedent). Per the 2026-05-12 HANDOFF Q3 decision: Orionfold LLC is the parent brand for all AI-artifact publishing surfaces; repo names follow the Bartowski shape (`Orionfold/<model>-GGUF`, `Orionfold/<model>-LoRA`). `ORIONFOLD_HF_ORG` is retained as a back-compat alias pointing at the new constant; will be dropped at the next major cut.

### Added — `fieldkit.quant` (new module)

Quantization dispatcher. GGUF path implemented; AWQ/GPTQ/EXL3/MLX/NVFP4 declared as named stubs pointing at the roadmap.

- **`fieldkit.quant.quantize_gguf(...)`** — wraps `llama.cpp/convert_hf_to_gguf.py` + `llama-quantize` to emit one GGUF file per requested variant (canonical Orionfold set: `Q4_K_M`, `Q5_K_M`, `Q6_K`, `Q8_0`, `F16`). Auto-derives F16 from a HF Transformers checkpoint when the source isn't already a GGUF. `dry_run=True` enumerates the would-be subprocess commands into `report.notes` without invoking them — used by tests and CI.
- **`fieldkit.quant.measure_perplexity_gguf(...)`** — wraps `llama-perplexity`. Parses output via `parse_perplexity_output()` which recognizes the standard `Final estimate: PPL = N.NNN` shape and the lowercase `perplexity = N.NNN` fallback. Returns `None` on parse failure (cards ship without a perplexity column if measurement was skipped).
- **`fieldkit.quant.measure_tokens_per_sec_gguf(...)`** — wraps `llama-bench`. Parses output via `parse_llama_bench_output()` for `tg` (text-gen, default) or `pp` (prompt-process) tok/s.
- **`fieldkit.quant.ThermalProbe`** — pure-stdlib `nvidia-smi` poll loop. Reports sustained-load minutes before throttle, per the 2026-05-12 HANDOFF Q9 decision to publish duty-cycle limits on every Orionfold card.
- **`fieldkit.quant.LlamaCppPaths`** — locator for `llama-quantize` / `llama-perplexity` / `llama-bench` / `convert_hf_to_gguf.py`. Env defaults: `LLAMA_CPP_BIN` directory, `LLAMA_CPP_CONVERT` script path. Override any field directly.
- **`fieldkit.quant.QuantReport`** — canonical dataclass output. The contract `fieldkit.publish.publish_quant()` consumes.
- **`fieldkit.quant.quantize_awq` / `quantize_gptq` / `quantize_exl3` / `quantize_mlx` / `quantize_nvfp4`** — named entry-point stubs. Raise `NotImplementedError` with a one-liner pointing at `ideas/mtbm-use-cases.md` §7. Locks the v0.4 public surface so v0.5+ implementations slot in without an API break.

### Added — `fieldkit.eval.VerticalBench` (v0.4.x — vertical-curator overlay)

Lightweight JSONL-loader wrapper around `fieldkit.eval.Bench` for vertical-domain accuracy scoring (FinanceBench / LegalBench / SemEval / generic). Drives the **vertical-curator pivot** announced 2026-05-13 (HANDOFF §2 + `ideas/mtbm-use-cases.md` §6 Pick #1.b + §8.5.1): every Orionfold quant card now ships with a vertical-domain accuracy axis, not just wikitext perplexity. Lives in `fieldkit/src/fieldkit/eval/vertical.py`; re-exported at the package root for `from fieldkit.eval import VerticalBench`.

- **`fieldkit.eval.VerticalBench`** + **`VerticalQA`** — bench shape, JSONL loader, scorer plumbing. Accepts any `Callable[[str], str]` as the model function so subprocess (`llama-cli`), in-process (`llama-cpp-python`), or NIM-backed scoring all slot in. Per-call latency aggregates alongside accuracy + refusal via the underlying `Bench`.
- **`fieldkit.eval.VerticalBench.from_jsonl(path, format='auto', ...)`** — auto-detects `financebench` / `legalbench` / `generic` JSONL shapes from the first row's field signature. Per-row metadata (company, doc_period, question_type, task) flows into per-call tags for slice-by aggregation downstream.
- **Scorers** — `exact_match`, `contains`, `numeric_match` (with configurable `rel_tolerance`, default 1% — FinanceBench convention). The bench picks `numeric_match` by default for FinanceBench-shape JSONL, `exact_match` for LegalBench-shape.

### Added — license + How-to-run defaults on `fieldkit.publish` (v0.4.x — `Orionfold/finance-chat-GGUF` dry-run found two card bugs)

- **`ModelCard.license`** is now reachable from `publish_quant(..., model_license=...)` (and the duck-typed `quant_report.model_license` attribute). Previously the kwarg didn't exist and every card defaulted to `apache-2.0` — wrong for any Llama / Gemma / Qwen / CC-BY-NC base. AdaptLLM/finance-chat now correctly publishes with `license: llama2`.
- **`ArtifactManifest.model_license`** mirrors the same value into the Astro manifest under `license.model:`. Astro Zod schema (`src/content.config.ts`) extended with `license.model: z.string().optional()` so destination catalog pages and HF badges stay in sync. The `license.tier:` field (commercial-distribution tier — `free` / `pro`) stays distinct from this upstream-license field.
- **`ModelCard.hf_repo`** + **`ModelCard.chat_format`** + **`ModelCard.recommended_variant`** — three new fields that drive an auto-rendered default `## How to run` body. Before this fix, cards with no explicit `ollama_pull_handle` / `transformers_snippet` rendered an empty section header (the second finance-chat bug). The new renderer auto-builds three code blocks templated from `hf_repo` + a featured variant: `huggingface-cli download`, `llama-server` (OpenAI-compatible serve), and `llama-cpp-python` (in-process, threading `chat_format` if set). When all three new fields are absent + no explicit handle/snippet supplied, the section is omitted entirely (no more empty headers).
- **`publish_quant(..., model_license=, chat_format=, recommended_variant=)`** kwargs added — orchestrate all three through to card + manifest. Same duck-typed fallback through `quant_report` attributes.
- **`scripts/g3_build_first_quant.sh`** — `MODEL_LICENSE` / `CHAT_FORMAT` / `RECOMMENDED_VARIANT` env knobs added with case-statement overrides (`AdaptLLM/finance-chat → llama2 + llama-2`). Default `MODEL_LICENSE=apache-2.0` + `RECOMMENDED_VARIANT=Q5_K_M` for greenfield runs.
- **`scripts/g3_push_first_quant.py`** (new) — one-shot live-push helper that reuses the existing dry-run stage (no 32 GB re-copy via `publish_quant(dry_run=False)`); calls `HFHubAdapter.push_folder()` directly. Bakes in xet-safety env (`HF_HOME=/home/nvidia/data/.hf-cache` + `HF_HUB_DISABLE_XET=1`) per the Spark-side `~/.cache/huggingface/` permission landmine; sources `HF_TOKEN` from `.env.local` (chmod 600).
- **+11 tests** (full suite: 379 passed, 2 skipped offline). Covers: model_license override flow, default apache-2.0 fallback, default GGUF How-to-run rendering, `recommended_variant` override, `hf_repo`-less skip-section behavior, manifest `license.model` emission.

### Added — vertical-eval surface on `fieldkit.publish`

`ModelCard` + `ArtifactManifest` + `publish_quant(...)` extended to thread per-variant vertical-eval scores through to the rendered card and the Phase-2 sync manifest:

- **`ModelCard.vertical_eval: dict[str, float]`** + **`ModelCard.vertical_eval_name: str`** — when set, the **Spark-tested** block renders a 5-column table (Variant / Size / Perplexity / tok/s / *Vertical-eval-name*) instead of the 4-column default, and the introductory copy switches from "measurement triple" to "measurement quad". Accuracy values render as percentages (`62.0%`). Cards without vertical eval render identically to v0.4.0 — backwards-compatible.
- **`ArtifactManifest.vertical_eval` + `vertical_eval_name`** — written into the YAML manifest under the same key names. Mac destination Zod schema (`src/content.config.ts`) extended to accept both. Manifests without vertical eval skip the field entirely.
- **`publish_quant(..., vertical_eval=, vertical_eval_name=)`** — explicit kwargs override whatever the duck-typed `quant_report` carries. Useful when scoring happens out-of-band from quantization (the canonical path on Spark: quantize 5 variants → measure each variant via `g3_measure_variants.py`, which calls `VerticalBench.run(llama_cli_fn)` and then feeds the resulting accuracy dict back into `publish_quant`).

### Schema changes

- `src/content.config.ts` — `FIELDKIT_MODULES` extended to include `'quant'` and `'publish'` in canonical order (`capabilities, nim, rag, eval, training, lineage, quant, publish, cli`).
- `src/content.config.ts` — new `artifacts` Astro collection (Phase 2 sync contract). Loads YAML manifests from `src/content/artifacts/*.yaml`; Zod schema mirrors `fieldkit.publish.ArtifactManifest`. `ARTIFACT_KINDS` enum exposed alongside `FIELDKIT_MODULES` for downstream filtering. `src/content/artifacts/` directory created (empty + `.gitkeep`); first manifest will land when the first quant ships.
- `src/content.config.ts` — `artifacts` schema extended with optional `vertical_eval: Record<string, number>` + `vertical_eval_name: string` (vertical-curator pivot 2026-05-13).

### Test suite

**130 new tests** across `tests/test_publish.py` (42, +16 from v0.4 scaffold incl. +11 for the model_license + How-to-run defaults fix), `tests/test_quant.py` (37), and `tests/test_vertical_bench.py` (39, new file), plus targeted regression coverage. Total: **379 passed, 2 skipped** offline (`pytest -q`). The 2 skips are `--spark`-gated live integration tests (chat NIM + pgvector); the v0.3 torch module-level skip has been resolved by lazy-importing torch only inside the training entry points. All new tests run offline — `dry_run=True` paths for `HFHubAdapter`, `publish_quant`, and `quantize_gguf` exercise the full code path without `huggingface_hub`, llama.cpp binaries, or `nvidia-smi` available. `VerticalBench` tests run without a model — `model_fn` is a callable, so a plain `lambda` exercises the full scoring + bench-aggregation path.

### Articles in this release

- [`becoming-a-gguf-publisher-on-spark`](https://ainative.business/field-notes/becoming-a-gguf-publisher-on-spark/) — G3 v0 anchor article. 3,388 words; documents the five-variant `Orionfold/finance-chat-GGUF` release end-to-end (Spark-tested perplexity / tok/s / sustained-load minutes / FinanceBench accuracy across F16, Q8_0, Q6_K, Q5_K_M, Q4_K_M) plus the V0 preflight-bench gate and the V1 chat-vs-continued-pretrain lesson. `hf_url:` frontmatter threads the live HF receipt onto the article.

### Verified on Spark

- **Live HF push:** `Orionfold/finance-chat-GGUF` shipped 2026-05-14 at <https://huggingface.co/Orionfold/finance-chat-GGUF> — 5 GGUF variants + auto-rendered README in 1h 57min. Repo returns HTTP 200, all 6 files present. `publish_quant(dry_run=False)` path exercised end-to-end.
- **Five-variant measurement card** (F16 / Q8_0 / Q6_K / Q5_K_M / Q4_K_M) with the four Spark-tested axes — perplexity (wikitext-2), tg + pp tok/s (`llama-bench`), sustained-load minutes (`ThermalProbe` via `nvidia-smi`), and FinanceBench accuracy (n=50, `numeric_match`, open-book) — all produced via `fieldkit.quant.measure_*` + `fieldkit.eval.VerticalBench.run(...)` on GB10.

### Deferred to v0.5

- `fieldkit.image-lora` + `fieldkit.civitai` — Pick #2 (G9) prep. Deferred per the 2026-05-12 HANDOFF Q10 decision to sequence G3 → G9 rather than parallelize. Will land once G3 v0 proves the `fieldkit.publish` infra.
- Non-GGUF formats in `fieldkit.quant` (AWQ, GPTQ, EXL3, MLX, NVFP4). The G3 v0 niche-positioning is Nemotron-family GGUFs with the Spark-tested layer; other formats are pure surface-area expansion and can wait for an audience signal.

## [0.3.0] — 2026-05-11

Third public release. One new top-level module (`fieldkit.lineage`) lifted from the [auto-research-loop-on-spark article](https://ainative.business/field-notes/auto-research-loop-on-spark/) — the portable part of cxcscmu's *Auto-Research-Recipes* harness, decomposed into a pure-stdlib substrate any harness on the Spark can write into.

### Added — `fieldkit.lineage` (new module)

The portable part of cxcscmu's *Auto-Research-Recipes* harness, extracted into a top-level submodule. The case for the primitive is in the released `pg_ablation_lineage_on` vs `pg_ablation_lineage_off` runs: same agent, same prompt template, same 201-trial budget on Parameter Golf — only whether the agent's session prompt includes the rendered lineage block differs. With lineage on: 16 keeps (8.0%), 38 eval-budget overruns. Without: 3 keeps (1.5%), 123 eval-budget overruns. **5.3× more keeps · 3.2× fewer wall-wastes**, with no model change, no compute change, no prompt-template change. ([extract from #auto-research-loop-on-spark])

The new module is pure-stdlib (no torch, no numpy) — ~200 LOC of public surface, ~330 LOC including docstrings + renderer helpers.

- **`fieldkit.lineage.FailureLabel`** — 10-class string enum (`keep`, `discard`, `crash`, `eval_budget_overrun`, `train_budget_overrun`, `size_blocked`, `preflight_crash`, `harness_abort`, `disqualified`, `baseline`). `.value` round-trips byte-identically to cxcscmu TSVs. The `is_informational` property is the cxcscmu `_QUARANTINED_STATUSES` rule as a method — returns `False` only for `harness_abort` (bookkeeping kills); every other class carries usable signal for the next agent.
- **`fieldkit.lineage.Trial`** — frozen dataclass for one TSV row. 17 fields in canonical order. `core_metric` is the task-agnostic primary metric (so the module works for Parameter Golf, NanoChat-D12, CIFAR, and any future task in the arc); `val_bpb` is preserved alongside for direct interop with cxcscmu-shaped data. `Trial.header()` / `Trial.to_row()` / `Trial.from_row(dict)` give exact TSV round-trip — `None` floats serialize as empty strings (matches cxcscmu convention).
- **`fieldkit.lineage.LineageStore(root, *, lower_is_better=True)`** — append-only TSV writer at `root/results.tsv` with `fcntl.flock` exclusive locking across header + row writes (concurrent specialists can write without interleaving). Read-side accessors: `all_trials()`, `latest(n)`, `best()`, `chain_to(exp_id)` (walks `parent_exp` pointers root-first, terminates on missing or self-referential parents), and `render_prompt(...)` — the deterministic Markdown emitter.
- **`fieldkit.lineage.LineageSnapshot`** — frozen dataclass returned by `render_prompt`. Carries the rendered Markdown string plus the underlying structured data (`current_best`, `chain_to_best`, `top_k_leaderboard`, `recent_n_activity`, `last_m_with_full_hypothesis`) so callers can index in without re-parsing.
- **`fieldkit.lineage.RecipeEdit`** — pairs a keep trial with its workdir `snapshot_path` and `parent_snapshot_path`. `diff()` computes a unified diff of every text file in the snapshot vs the parent (binary files elide with a `Binary files ... differ` marker); baseline trials with no parent return an empty diff.

Rendered Markdown output mirrors cxcscmu's `release_artifacts/example_lineage_pg_lineage_on_arch.txt` shape: header line + `## LEADERBOARD.md` (current best + top-K kept table) + `## KNOWLEDGE.md` (current-best lineage as a nested `└─` chain + recent-activity table + last-M detailed entries). Determinism is tested — same TSV state in produces byte-identical Markdown across calls.

### Test suite

**29 new tests** for `fieldkit.lineage` (`tests/test_lineage.py`): `FailureLabel` value parity + `is_informational` predicate + 10-class enum surface lock; `Trial` round-trip via TSV; `LineageStore` append / latest / best / `chain_to` correctness across linear and branched topologies; `render_prompt` determinism, top-K filtering, chain rendering with `← BEST` marker; `RecipeEdit.diff()` against parent snapshots including new-file detection.

Total fieldkit test count: **249 passed, 3 skipped** offline (`pytest -q`) — the 3 skips are 1 module-level torch importorskip in `test_training.py` and 2 `--spark`-gated live integration tests.

### Articles in this release

- [`auto-research-loop-on-spark`](https://ainative.business/field-notes/auto-research-loop-on-spark/) — anchor article. Walks the 17-column schema, the 10-class enum semantics, and the cxcscmu lineage ablation that proves the primitive's value.

### Schema change — `FIELDKIT_MODULES`

`src/content.config.ts` extended to include `'lineage'` in the `FIELDKIT_MODULES` tuple (order: `capabilities, nim, rag, eval, training, lineage, cli`). Required so articles can declare `fieldkit_modules: ['lineage']` in their frontmatter.

[extract from #auto-research-loop-on-spark]: https://github.com/manavsehgal/ai-field-notes/tree/main/articles/auto-research-loop-on-spark

## [0.2.0] — 2026-05-05

Second public release. One new module (`fieldkit.training`) plus four extensions to the v0.1 `fieldkit.eval` surface, all lifted from articles in [ai-field-notes](https://ainative.business/field-notes/) — primarily the `clawgym-on-spark` and Frontier Scout arcs. The `fieldkit.agents` and `fieldkit.inference` modules originally targeted for v0.2 are deferred to v0.3+ because their public APIs need a second article's use case to lock in (see "Deferred to v0.3+" below).

### Added — `fieldkit.training` (new module)

Fine-tuning primitives for any RL or SFT loop on the DGX Spark's unified-memory GB10. Both classes use lazy `torch` imports so `import fieldkit.training` costs nothing in environments that don't run training.

- **`fieldkit.training.WeightDeltaTracker`** — pre/post snapshot of trainable params with L2 and `max|Δ|` reporting. Sanity-check that any fine-tuning step actually moves weights — the first time someone debugs "why didn't my LoRA update?" they'll wish for this. Source: `articles/clawgym-on-spark/scripts/grpo_train.py` (`--check-weight-delta` block). ([extract from #clawgym-on-spark-grpo])
- **`fieldkit.training.LoraReferenceSnapshot`** — CPU-resident snapshot of a peft adapter's LoRA tensors with a context manager that swaps the snapshot into the live model for one no-grad forward pass and restores trainable weights on exit. Two construction modes: snapshot from current policy at step start (online) vs. `from_disk(adapter_dir)` for a fixed reference (classic GRPO fixed-SFT-init reference, with the safetensors `.<adapter_name>.weight ↔ .weight` key transform peft 0.19+ requires). Solves a real bug: peft 0.19's `load_adapter(..., is_trainable=False)` crashes with `KeyError` under `device_map="auto"` whenever the GPU has anything else resident — peft's offload-detection over-triggers on Spark unified memory. The CPU-snapshot/swap dance sidesteps the offloader entirely. Source: `articles/clawgym-on-spark/scripts/grpo_train.py` (`--reference-adapter` + snapshot/swap blocks). ([extract from #clawgym-on-spark-grpo])

### Added — extends `fieldkit.eval`

Four new primitives that extend the v0.1 eval surface (`Bench`, `Judge`, `Trajectory`, `is_refusal`) with programmatic grading, code-bench pass@k, agent-loop schemas, and ablation comparison support.

- **`fieldkit.eval.AssertionGrader`** — pure-function grader over five file-system assertion primitives (`file_exists`, `file_not_exists`, `file_unchanged`, `file_contents_contain`, `file_contents_match_regex`). Accepts either a SynthTask-shaped dict (auto-derives `seed_files` from `workspace_seed.files`) or a bare list of assertion dicts, so the grader stays usable without coupling to the deferred `fieldkit.agents.SynthTask` shape. Sibling to `Judge` — programmatic verification where it applies. Source: `articles/clawgym-on-spark/scripts/grader.py`. ([extract from #clawgym-on-spark])
- **`fieldkit.eval.PassAtK`** + **`pass_at_k_estimator`** — verifier-loop primitive: per-task grader + `n`-sample iterator → `pass@1`, `pass@k` via the unbiased estimator (Chen et al. 2021). Decoupled from the model — caller supplies pre-generated samples + a grader callable, `PassAtK` aggregates. Two entry points: `score(problems, samples, grader)` for fresh runs and `from_rows(rows)` for offline pass@k math against pre-graded `(task_id, n, passed)` tuples. Tested on HumanEval + AIME 2024 across baseline vs. ESamp modes. Source: `articles/runtime-frontier-six-patches-on-spark/scripts/passatk_a2.py`. ([extract from #pass-at-k-after-the-seventh-patch])
- **`fieldkit.eval.AgentRun`** + **`TurnDetail`** + **`summarize_agent_runs`** — per-question, per-turn schema for any third-party agent bench. Default constructor handles the AutoResearchBench JSONL shape (`input_data.arxiv_id`, `inference_results[0].turn_details/total_time/...`); `from_record(...)` accepts field-name overrides for other bench layouts. `TurnDetail` carries five canonical fields (turn, action, duration_s, input/output tokens) plus an `extras` dict so bench-specific fields (e.g. `papers_retrieved`, `parse_errors`) survive round-tripping. `summarize_agent_runs()` rolls up status counts + `wall_seconds` / `turns` / `candidates` / `tool_calls` / `tool_format_errors` summaries. Source: `articles/autoresearchbench-on-spark/scripts/analyze_run.py`. ([extract from #autoresearchbench-on-spark])
- **`fieldkit.eval.MatchedBaseComparison`** + **`GroupStats`** + **`MatchedBaseComparisonResult`** — held-out task split + two-rollout driver + per-group / per-assertion-kind delta. The "filter held-out by training-set membership, run rollout twice with different `--model`, emit B-A comparison" pattern is reusable for any LoRA / adapter ablation. Default `group_extractor` splits `synth-<persona>-NN` task IDs into the persona; pass any `Callable[[str], str]` for other task-id schemes, or `None` to disable per-group breakdown. Accepts trajectories as in-memory dicts or a JSONL path. `.report()` returns a markdown summary table. Source: `articles/clawgym-on-spark/scripts/compare_phase5.py`. ([extract from #clawgym-on-spark])

### Articles in this release

Articles whose `fieldkit_modules` frontmatter assumes v0.2 (added since v0.1.0):

- [`autoresearchbench-on-spark`](https://ainative.business/field-notes/autoresearchbench-on-spark/) — surfaced `fieldkit.eval.AgentRun`.
- [`test-time-distilling-for-exploration`](https://ainative.business/field-notes/test-time-distilling-for-exploration/) — surfaced the deferred `fieldkit.inference.VLLMClient`.
- [`runtime-frontier-six-patches-on-spark`](https://ainative.business/field-notes/runtime-frontier-six-patches-on-spark/) — surfaced `fieldkit.eval.PassAtK` (matured in the seventh-patch follow-up).
- [`pass-at-k-after-the-seventh-patch`](https://ainative.business/field-notes/pass-at-k-after-the-seventh-patch/) — anchor article for `fieldkit.eval.PassAtK`.
- [`clawgym-on-spark`](https://ainative.business/field-notes/clawgym-on-spark/) — surfaced `fieldkit.eval.AssertionGrader`, `fieldkit.eval.MatchedBaseComparison`, plus the deferred `fieldkit.agents` substrate.
- [`clawgym-on-spark-grpo`](https://ainative.business/field-notes/clawgym-on-spark-grpo/) — surfaced the entire `fieldkit.training` module (`LoraReferenceSnapshot`, `WeightDeltaTracker`).

### Test suite

**232 passed, 2 skipped** offline (`pytest -q`) — covers all v0.1 surface plus 16 + 19 + 16 + 12 + 12 = 75 new tests for the v0.2 additions. Reproduce: `pip install fieldkit[dev]` then `pytest`. The `fieldkit.training` tests gate on `pytest.importorskip("torch")` so the suite skips cleanly in pure-inference dev envs and runs end-to-end in any env with torch installed. v0.1's live `--spark` integration tests still pass against warm NIMs + pgvector — none were modified in this release.

### Deferred to v0.3+

The full design doc at `articles/clawgym-on-spark/scripts/fieldkit_agents_v0_2_sketch.md` charts the larger `fieldkit.agents` substrate; the candidates below need a second article's use case before extraction is sound.

- **`fieldkit.agents` module** (7 symbols — `Persona`, `WorkspaceSeed`/`WorkspaceFile`, `SynthTask`, `TaskAuthor`, `Sandbox`/`LocalTempSandbox`, `RolloutDriver`, `Trajectory`/`TurnRecord`). The whole agent-trajectory training substrate that the `clawgym-on-spark` arc walks. Coupled enough that the public API needs a second consuming article to lock in. Source: `articles/clawgym-on-spark/scripts/synth_tasks.py` + `rollout.py`. ([extract from #clawgym-on-spark])
- **`fieldkit.inference.VLLMClient`** — mirror of `fieldkit.nim.NIMClient` for vLLM-side experiments. Deferred because the canonical interface needs a second runtime-frontier article (post-test-time-distilling) to converge. Source: `articles/runtime-frontier-six-patches-on-spark/scripts/bench_a2.py`. ([extract from #test-time-distilling-for-exploration])
- **`fieldkit.agents.replay_messages_from_trajectory`** — reconstruct the exact `(system, user, assistant, observation, …)` message list a policy saw at rollout time. Required for any off-policy training (log-prob recompute is meaningless if reconstruction differs by even a token). Logic currently lives in two byte-identical places (`rollout.py:RolloutDriver.rollout()` forward + `grpo_train.py:reconstruct_messages()` reverse). Right callable interface for `(system_prompt, user_prompt_template, observation_formatter)` won't be obvious until a second article exercises it. ([extract from #clawgym-on-spark-grpo])

[extract from #autoresearchbench-on-spark]: https://github.com/manavsehgal/ai-field-notes/tree/main/articles/autoresearchbench-on-spark
[extract from #test-time-distilling-for-exploration]: https://github.com/manavsehgal/ai-field-notes/tree/main/articles/test-time-distilling-for-exploration
[extract from #pass-at-k-after-the-seventh-patch]: https://github.com/manavsehgal/ai-field-notes/tree/main/articles/pass-at-k-after-the-seventh-patch
[extract from #clawgym-on-spark]: https://github.com/manavsehgal/ai-field-notes/tree/main/articles/clawgym-on-spark
[extract from #clawgym-on-spark-grpo]: https://github.com/manavsehgal/ai-field-notes/tree/main/articles/clawgym-on-spark-grpo

## [0.1.0] — 2026-05-02

First public release. Four library modules + a CLI + a docs site section, lifted from 25+ articles in [ai-field-notes](https://ainative.business/field-notes/).

### Added

- **`fieldkit.capabilities`** — typed Python facade over `spark-capabilities.json`. `Capabilities.load()` cached singleton (with `.hardware`, `.memory_budget_rules_of_thumb`, `.stack`, `.in_envelope_signals`, `.out_of_envelope_signals`, `.stage_routing_hints`, `.series_routing_hints`), plus canonical math helpers `kv_cache_bytes()`, `weight_bytes()`, `practical_inference_envelope()`. Numbers pinned to `kv-cache-arithmetic-at-inference` and `gpu-sizing-math-for-fine-tuning`. ([#capabilities])
- **`fieldkit.nim`** — OpenAI-compatible `NIMClient` over `httpx` with `tenacity`-backed retries on 429 / 503 / `ConnectError` / timeouts. `NIMClient.chat()` runs a pre-flight context check and raises `NIMContextOverflowError` with the estimated token count *before any network call*, so the opaque NIM 400 from a >8192-token request never surfaces. Helpers: `chunk_text()` (paragraph→sentence→word splitting under a `max_tokens` budget), `estimate_tokens()` (1 tok ≈ 4 chars), `wait_for_warm()` (polls `/v1/models` for the ~90s NIM cold start). Constants: `NIM_CONTEXT_WINDOW = 8192`, `DEFAULT_CHUNK_TOKENS = 1024`. Errors: `NIMError` → `NIMHTTPError`, `NIMTimeoutError`, `NIMContextOverflowError`. ([#nim])
- **`fieldkit.rag`** — composable ingest → retrieve → rerank → fuse pipeline backed by pgvector + a NIM embedder + the strict-context grounded prompt from `naive-rag-on-spark`. `Pipeline.ingest()` chunks via `fieldkit.nim.chunk_text` and upserts in batches of 32; `Pipeline.retrieve()` does pgvector cosine top-K; `Pipeline.rerank()` is a pass-through when `rerank_url=None`; `Pipeline.fuse()` builds the strict-context messages list and calls the generator; `Pipeline.ask()` chains all three. Embed and rerank inherit `NIMClient.chat`'s retry policy so co-resident memory pressure doesn't fail the pipeline. ([#rag])
- **`fieldkit.eval`** — `Bench` (latency aggregation with the same `{summary, calls}` JSON shape as the article evidence files), `Judge` (LLM-as-judge with built-in `correctness` / `faithfulness` / `relevance` rubrics + a static `Judge.parse()` JSON-then-regex extractor), `Trajectory` (agent-loop JSONL analyzer with `knob_coverage / repeat_rate / mode_dominance / cumulative_best`), `is_refusal()` (regex catalog unioned across the project's articles), `summarize_metric()`. ([#eval])
- **`fieldkit.cli`** — Typer wrapper exposing `fieldkit version`, `fieldkit envelope <size>`, `fieldkit feasibility <model_id> [--ctx --batch --dtype]`, `fieldkit bench rag`. On `$PATH` after `pip install`. ([#cli])
- **Astro docs site** — `/fieldkit/` landing page with install + quickstart + module grid, and `/fieldkit/api/<module>/` reference pages backed by a new `fieldkit_docs` content collection. Articles can opt-in via `fieldkit_modules:` frontmatter to display a "USES fieldkit.X" chip on cards and appear under each module's "Articles that use fieldkit.<module>" footer. 11 articles opted in for the v0.1.0 launch.
- **Samples**: `samples/feasibility-math.py` (capabilities reproduction of the kv-cache article's table), `samples/hello-nim.py` (Python equivalent of the curl one-liner), `samples/naive-rag.py` (end-to-end RAG in <30 lines), `samples/bench-rag.py` (offline `Bench` + `Judge.parse()` walkthrough).
- `scripts/sync_capabilities.py` keeps the package-bundled `spark-capabilities.json` in sync with the source-of-truth at `scripts/lib/spark-capabilities.json` (pre-commit-enforced).
- `pytest --spark` flag (via `tests/conftest.py`) gates integration tests that need a live NIM / pgvector on the DGX Spark; default runs skip them.

### Changed

- `frontier-scout` skill (`refresh` and `eval` modes, plus `references/feasibility-prompt.md` and `references/classifier-prompt.md`) now teaches the typed `from fieldkit.capabilities import …` API as the preferred grounding path; raw JSON read is the documented fallback.

### Verified on Spark

Phases 3, 4, 5 were live-verified end-to-end against the chat NIM (Llama 3.1 8B, port 8000), the embed NIM (Nemotron Embed 1B v2, port 8001), and pgvector (port 5432) before being committed. Phase 5 in particular rewrote `articles/naive-rag-on-spark/evidence/benchmark.py` against `fieldkit.eval.Bench` + `fieldkit.rag.Pipeline.fuse` and reproduced the original article's behavioral fingerprint (5 of 6 refusals incl. the canonical Google-IPO false refusal, plus the Ian Thorpe grounded answer).

### Distribution

Published to PyPI on 2026-05-02: <https://pypi.org/project/fieldkit/0.1.0/>. Canonical install is now `pip install fieldkit`; the git-tag install (`pip install "git+…@fieldkit/v0.1.0#subdirectory=fieldkit"`) remains supported for unreleased commits between tags. Subsequent releases publish to both git and PyPI in one flow via `fieldkit-curator release`.

### Test suite

**157 passing, 2 skipped** without `--spark` (151 passing with `--spark` against warm NIMs + pgvector). Reproduce: `pip install fieldkit/[dev]` then `pytest`; for the live tests, `pytest --spark`.

[#capabilities]: https://github.com/manavsehgal/ai-field-notes/tree/main/fieldkit/src/fieldkit/capabilities
[#nim]: https://github.com/manavsehgal/ai-field-notes/tree/main/fieldkit/src/fieldkit/nim
[#rag]: https://github.com/manavsehgal/ai-field-notes/tree/main/fieldkit/src/fieldkit/rag
[#eval]: https://github.com/manavsehgal/ai-field-notes/tree/main/fieldkit/src/fieldkit/eval
[#cli]: https://github.com/manavsehgal/ai-field-notes/tree/main/fieldkit/src/fieldkit/cli
