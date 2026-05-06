# Changelog

All notable changes to `fieldkit` are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). While the package is on `0.x`, minor versions may include breaking changes. `1.0` will mark API stability.

## [Unreleased]

### Added — proposed module: `fieldkit.agents`

The agent-trajectory training substrate that the `clawgym-on-spark` arc walks. One module covers task synthesis, sandbox rollout, and trajectory records — `Persona` produces tasks, `RolloutDriver` runs them, `Trajectory` records what happened, and `fieldkit.eval.AssertionGrader` (below) grades the post-state. The four are a tight loop. Full design doc at `articles/clawgym-on-spark/scripts/fieldkit_agents_v0_2_sketch.md`.

- **`fieldkit.agents.Persona`** — frozen dataclass `(role, context, skill_focus)`. Source: `articles/clawgym-on-spark/scripts/synth_tasks.py:Persona`. ([extract from #clawgym-on-spark])
- **`fieldkit.agents.WorkspaceSeed`** + **`WorkspaceFile`** — JSON-serializable workspace template; `WorkspaceSeed.materialize(root)` writes the seed to disk for one rollout. Source: `articles/clawgym-on-spark/scripts/synth_tasks.py`. ([extract from #clawgym-on-spark])
- **`fieldkit.agents.SynthTask`** — frozen JSON-serializable record bundling intent + assertions + workspace. The seam between `TaskAuthor` (produces) and `AssertionGrader` (consumes). Source: `articles/clawgym-on-spark/scripts/synth_tasks.py`. ([extract from #clawgym-on-spark])
- **`fieldkit.agents.TaskAuthor`** — LLM-driven task generator with retry + balanced-brace JSON extractor. Wraps any `NIMClient`-shaped chat client. Source: `articles/clawgym-on-spark/scripts/synth_tasks.py`. ([extract from #clawgym-on-spark])
- **`fieldkit.agents.Sandbox`** + **`LocalTempSandbox`** — abstract base (`materialize` / `exec` / `list_files` / `cleanup`) + a tempdir+subprocess concrete implementation. `NemoClawSandbox` deferred until an article actually uses it. Source: `articles/clawgym-on-spark/scripts/rollout.py`. ([extract from #clawgym-on-spark])
- **`fieldkit.agents.RolloutDriver`** — agent loop with bash-block-per-turn protocol, parse-error corrective hints, and observation injection back into chat history. Generic over any `Sandbox` and `NIMClient`-shaped client. Source: `articles/clawgym-on-spark/scripts/rollout.py:RolloutDriver`. ([extract from #clawgym-on-spark])
- **`fieldkit.agents.Trajectory`** + **`TurnRecord`** — JSONL-friendly record types for SFT/GRPO consumption. Source: `articles/clawgym-on-spark/scripts/rollout.py`. ([extract from #clawgym-on-spark])

### Added — proposed module: `fieldkit.training`

New module for fine-tuning primitives that recur across SFT and RL articles. Two utilities surfaced in the Phase 6 GRPO work; future training articles will extend.

- **`fieldkit.training.LoraReferenceSnapshot`** — CPU-resident snapshot of a peft adapter's LoRA tensors (loaded via safetensors with `.{adapter_name}.weight` ↔ `.weight` key transform); a context manager swaps the snapshot in for one `no_grad` forward pass and restores trainable weights. Solves a real bug: peft 0.19's `load_adapter(adapter_name="reference", is_trainable=False)` crashes with a `KeyError` under `device_map="auto"` whenever the GPU has anything else resident — verified both with vLLM co-resident *and* with the trainer alone, because peft's offload-detection over-triggers on Spark unified memory. ~30 lines. Two reference modes: snapshot from current policy at step start (online) vs. load LoRA weights from a fixed adapter on disk (classic GRPO fixed-SFT-init reference). Source: `articles/clawgym-on-spark/scripts/grpo_train.py` (`--reference-adapter` + snapshot/swap blocks, lines 207–260 and 336–348). Opens a new `fieldkit.training` module. ([extract from #clawgym-on-spark-grpo])
- **`fieldkit.training.WeightDeltaTracker`** — pre/post snapshot of trainable params with L2 and `max|Δ|` reporting. ~15 lines. Sanity-check that any fine-tuning step actually moves weights — the first time someone debugs "why didn't my LoRA update?" they'll wish for this. Source: `articles/clawgym-on-spark/scripts/grpo_train.py` (`--check-weight-delta` block, lines 268–273 and 394–406). ([extract from #clawgym-on-spark-grpo])

### Added — proposed module: `fieldkit.inference`

New module for inference-server clients beyond NIM. One vLLM client surfaced from the test-time-distilling work; future runtime articles will extend.

- **`fieldkit.inference.VLLMClient`** — mirror of `fieldkit.nim.NIMClient` for vLLM-side experiments. Wraps `make_llm`, `SamplingParams`, and the throughput-measurement boilerplate that recurs across the runtime-frontier articles. Source: `articles/test-time-distilling-for-exploration/evidence/repo-snapshot/` + `articles/runtime-frontier-six-patches-on-spark/scripts/bench_a2.py`. ([extract from #test-time-distilling-for-exploration])

### Added — extends `fieldkit.eval`

Four new primitives that extend the v0.1 eval surface (`Bench`, `Judge`, `Trajectory`, `is_refusal`) with agent-loop, programmatic-grading, code-bench, and ablation-comparison support.

- **`fieldkit.eval.AgentRun`** — per-question, per-turn schema `(action, duration, papers_retrieved, input_tokens, output_tokens, parse_errors)` for any third-party agent bench that reads `OPENAI_API_BASE` from a `.env`. Source: `articles/autoresearchbench-on-spark/scripts/analyze_run.py` + `compare_runs.py`. ([extract from #autoresearchbench-on-spark])
- **`fieldkit.eval.PassAtK`** — verifier-loop primitive: per-task grader + `n`-sample iterator → `pass@1`, `pass@k` via the unbiased estimator. Tested on HumanEval + AIME 2024, two model families, baseline vs. ESamp modes. Source: `articles/runtime-frontier-six-patches-on-spark/scripts/passatk_a2.py`. ([extract from #pass-at-k-after-the-seventh-patch])
- **`fieldkit.eval.AssertionGrader`** — pure-function grader over five assertion primitives (`file_exists`, `file_not_exists`, `file_unchanged`, `file_contents_contain`, `file_contents_match_regex`) over a post-state directory root. Sibling to `Judge`, not a replacement — programmatic verification where it applies. Source: `articles/clawgym-on-spark/scripts/grader.py:grade`. ([extract from #clawgym-on-spark])
- **`fieldkit.eval.MatchedBaseComparison`** — held-out task split + two-rollout driver + per-persona / per-assertion-kind delta. The "filter held-out by training-set membership → run rollout twice with different `--model` → emit B − A comparison" pattern is reusable for any LoRA / adapter ablation. Source: `articles/clawgym-on-spark/scripts/compare_phase5.py` + `run_phase5_pipeline.sh`. ([extract from #clawgym-on-spark])

### Deferred

- **`fieldkit.agents.replay_messages_from_trajectory`** — *deferred until a second article supplies a second use case.* Reconstruct the exact `(system, user, assistant, observation, …)` message list a policy saw at rollout time from a saved `Trajectory`. Required for any off-policy training (log-prob recompute is meaningless if reconstruction differs by even a token). The logic currently lives in two places and must stay byte-identical: `articles/clawgym-on-spark/scripts/rollout.py:RolloutDriver.rollout()` (forward, building messages as it generates) and `articles/clawgym-on-spark/scripts/grpo_train.py:reconstruct_messages()` (reverse, rebuilding from saved `Trajectory`, line 64). The right callable interface for `(system_prompt, user_prompt_template, observation_formatter)` won't be obvious until a second article exercises it. ([extract from #clawgym-on-spark-grpo])

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
