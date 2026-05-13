# Changelog

All notable changes to `fieldkit` are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). While the package is on `0.x`, minor versions may include breaking changes. `1.0` will mark API stability.

## [Unreleased]

Fourth release in flight. Two new top-level modules (`fieldkit.publish` + `fieldkit.quant`) scaffolded for the G3 GGUF / Quantization Publisher pick (MTBM Pick #1 per `ideas/mtbm-use-cases.md` §6). The two modules together unlock most of Cluster G; this cut implements the GGUF critical path and stubs the other quant formats with named entry points pointing at the v0.5+ roadmap.

### Added — `fieldkit.publish` (new module)

HuggingFace Hub adapter + auto model card builder from `fieldkit.lineage`. Three public surfaces:

- **`fieldkit.publish.ModelCard`** — frontmatter + body builder. Renders the canonical card every Orionfold artifact gets: YAML frontmatter (license, library_name, base_model, pipeline_tag, tags, model_creator), a title + elevator, a **Spark-tested** block (per-variant perplexity + tok/s + thermal envelope), a variants table, **How to run** (`ollama pull` + `from_pretrained` snippets), an optional **Lineage** block (rendered from a `fieldkit.lineage.LineageStore` if provided), a **Methods** backlink to `ainative.business/field-notes/<slug>/`, and a footer attributing the publication to Orionfold LLC.
- **`fieldkit.publish.ArtifactManifest`** — frozen dataclass for the `src/content/artifacts/<slug>.yaml` Phase-2 sync record (per memory `project_artifact_manifests_phase2`). `to_yaml()` emits via a hand-rolled stdlib emitter so the module has no runtime YAML dep. The source repo writes one of these per push; the Mac destination renders `/artifacts/<kind>/` catalog pages from `getCollection('artifacts')`.
- **`fieldkit.publish.HFHubAdapter`** — lazy-`huggingface_hub` wrapper. Defaults to `dry_run=True` (stages files on disk, logs the would-be calls, no network). Flip `dry_run=False` to push via `HfApi().upload_folder(...)`. Token resolution order: explicit `token=` → `HF_TOKEN` env → cached login. The dry-run path is fully testable offline.

Plus an orchestrator: **`fieldkit.publish.publish_quant(...)`** — one-line caller that ingests a `QuantReport`-shaped object (duck-typed; produced by `fieldkit.quant.quantize_gguf`), renders the card, writes the manifest, stages the variant files, and pushes (or dry-runs) the HF commit.

Branded constants: `ORIONFOLD_BRAND = "Orionfold LLC"`, `ORIONFOLD_HF_ORG = "orionfoldllc"`. Per the 2026-05-12 HANDOFF Q3 decision: Orionfold LLC is the parent brand for all AI-artifact publishing surfaces; repo names follow the Bartowski shape (`orionfoldllc/<model>-GGUF`, `orionfoldllc/<model>-LoRA`).

### Added — `fieldkit.quant` (new module)

Quantization dispatcher. GGUF path implemented; AWQ/GPTQ/EXL3/MLX/NVFP4 declared as named stubs pointing at the roadmap.

- **`fieldkit.quant.quantize_gguf(...)`** — wraps `llama.cpp/convert_hf_to_gguf.py` + `llama-quantize` to emit one GGUF file per requested variant (canonical Orionfold set: `Q4_K_M`, `Q5_K_M`, `Q6_K`, `Q8_0`, `F16`). Auto-derives F16 from a HF Transformers checkpoint when the source isn't already a GGUF. `dry_run=True` enumerates the would-be subprocess commands into `report.notes` without invoking them — used by tests and CI.
- **`fieldkit.quant.measure_perplexity_gguf(...)`** — wraps `llama-perplexity`. Parses output via `parse_perplexity_output()` which recognizes the standard `Final estimate: PPL = N.NNN` shape and the lowercase `perplexity = N.NNN` fallback. Returns `None` on parse failure (cards ship without a perplexity column if measurement was skipped).
- **`fieldkit.quant.measure_tokens_per_sec_gguf(...)`** — wraps `llama-bench`. Parses output via `parse_llama_bench_output()` for `tg` (text-gen, default) or `pp` (prompt-process) tok/s.
- **`fieldkit.quant.ThermalProbe`** — pure-stdlib `nvidia-smi` poll loop. Reports sustained-load minutes before throttle, per the 2026-05-12 HANDOFF Q9 decision to publish duty-cycle limits on every Orionfold card.
- **`fieldkit.quant.LlamaCppPaths`** — locator for `llama-quantize` / `llama-perplexity` / `llama-bench` / `convert_hf_to_gguf.py`. Env defaults: `LLAMA_CPP_BIN` directory, `LLAMA_CPP_CONVERT` script path. Override any field directly.
- **`fieldkit.quant.QuantReport`** — canonical dataclass output. The contract `fieldkit.publish.publish_quant()` consumes.
- **`fieldkit.quant.quantize_awq` / `quantize_gptq` / `quantize_exl3` / `quantize_mlx` / `quantize_nvfp4`** — named entry-point stubs. Raise `NotImplementedError` with a one-liner pointing at `ideas/mtbm-use-cases.md` §7. Locks the v0.4 public surface so v0.5+ implementations slot in without an API break.

### Schema changes

- `src/content.config.ts` — `FIELDKIT_MODULES` extended to include `'quant'` and `'publish'` in canonical order (`capabilities, nim, rag, eval, training, lineage, quant, publish, cli`).
- `src/content.config.ts` — new `artifacts` Astro collection (Phase 2 sync contract). Loads YAML manifests from `src/content/artifacts/*.yaml`; Zod schema mirrors `fieldkit.publish.ArtifactManifest`. `ARTIFACT_KINDS` enum exposed alongside `FIELDKIT_MODULES` for downstream filtering. `src/content/artifacts/` directory created (empty + `.gitkeep`); first manifest will land when the first quant ships.

### Test suite

**63 new tests** across `tests/test_publish.py` (26) and `tests/test_quant.py` (37). Total: **312 passed, 3 skipped** offline (`pytest -q`). The 3 skips are unchanged from v0.3.0 (1 module-level torch importorskip + 2 `--spark`-gated live integration tests). All new tests run offline — `dry_run=True` paths for `HFHubAdapter`, `publish_quant`, and `quantize_gguf` exercise the full code path without `huggingface_hub`, llama.cpp binaries, or `nvidia-smi` available.

### Articles in this release

- `articles/becoming-a-gguf-publisher-on-spark/` — placeholder (`status: upcoming`) scaffolded for the G3 v0 anchor article. Will be promoted to `status: published` after the first 5 Orionfold GGUF quants ship and the 14-day milestone in HANDOFF §2 is met.

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
