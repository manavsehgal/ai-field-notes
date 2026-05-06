# Changelog

All notable changes to `fieldkit` are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). While the package is on `0.x`, minor versions may include breaking changes. `1.0` will mark API stability.

## [Unreleased]

Second public release in progress. One new module (`fieldkit.training`) plus four extensions to the v0.1 `fieldkit.eval` surface, all lifted from articles in [ai-field-notes](https://ainative.business/field-notes/) — primarily the `clawgym-on-spark` and Frontier Scout arcs. The `fieldkit.agents` and `fieldkit.inference` modules originally targeted for v0.2 are deferred to v0.3+ because their public APIs need a second article's use case to lock in (see "Deferred to v0.3+" below).

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

### Verified on Spark

All 220 unit tests pass offline (`pytest -q`) plus 12 additional `fieldkit.training` tests gate on `pytest.importorskip("torch")` so the suite skips cleanly in pure-inference dev envs and runs end-to-end in any env with torch installed. The torch-dependent tests reproduce the snapshot/swap and L2 / max|Δ| math byte-identical to the GRPO trainer's behavior.

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
