<!--
  🆕 STATUS: NEW — pending Mac sweep.
  This file is one feature/release at a time, not a running log.
  At the next release prompt, **clear this entire file and start fresh** (do NOT append to existing sections).
  Last reset: 2026-05-14 (this rotation supersedes the v0.4.0 cycle scope; the v0.4.0 PyPI release commits e322af2..2190824 are bundled here under the v0.4.1 umbrella).

  Prior Mac sweep receipts (preserved here since SYNC-HANDOFF is per-release-not-running-log; Mac will sweep this v0.4.1 cycle next):
  - 2026-05-14 v0.4.0 cycle: swept at destination commit manavsehgal/ainative-business.github.io@f7ea7aa (Mac PR #4 against this repo — conflicted on rotation; safe to close, receipt captured here).
  - 2026-05-14 Orionfold/finance-chat-GGUF cycle: swept at destination commit manavsehgal/ainative-business.github.io@85f9307 (Mac PR #3, merged).
  - 2026-05-12 Autoresearch→MTBM rename: swept at destination commit manavsehgal/ainative-business.github.io@71293af (Mac PR #2, merged).
-->
---
release_slug: 2026-05-14-fieldkit-v0.4.1
status: NEW
source_range: 7f1159e..HEAD
articles_added:
  - becoming-a-legal-curator-on-spark      # new long-form deep-dive; status: published; customer-link audited; signature SVG reused from VerticalCuratorRetry (no new SVG this cycle)
articles_updated:
  - becoming-a-gguf-publisher-on-spark    # evidence/lineage TSV gained Q5_K_M + Q6_K re-run rows for audit-trail completeness (commit 8ae42e4 — published article body unchanged)
artifacts_added:
  - src/content/artifacts/saul-7b-instruct-v1-gguf.yaml   # second real-world Phase-2 manifest (after finance-chat-gguf.yaml from prior cycle) — license.tier=free, license.model=mit, vertical_eval populated from LegalBench n=50
artifacts_updated: []
fieldkit_modules_changed:
  - eval                                   # VerticalBench.from_jsonl gains open_book + subset kwargs (v0.4.1 lift, additive only, no breaking changes)
papers_added:
  - 2605.13779                             # MinT — Managed Infrastructure for LoRA Training + Serving Millions of LLMs (MTBM Pick #1 analog, 137 HF upvotes)
  - 2605.09942                             # HAGE — RL-Driven Weighted Graph Evolution for Agentic Memory (Second Brain)
  - 2605.12978                             # Useful Memories Become Faulty When Continuously Updated (Second Brain counter-narrative)
  - 2605.12975                             # Retrieval is Cheap, Show Me the Code — Executable Multi-Hop Reasoning for RAG (LLM Wiki)
  - 2605.12501                             # Covering Human Action Space for Computer Use (MTBM/clawnav-adjacent)
papers_classify_count: 13                  # total new in papers.json after relevance≥0.5 filter (5 marked dive-deep, 8 catalog-only)
renames_to_replay: []
removes: []
new_top_level_pages: []
breaking_changes: []
destination_overrides_to_preserve: []
hf_repos_added:
  - Orionfold/Saul-7B-Instruct-v1-GGUF     # 5 GGUF variants of Equall/Saul-7B-Instruct-v1, mistral chat_format, MIT license, LegalBench-scored (n=50, contains)
civitai_artifacts_added: []
fieldkit_release:
  version: 0.4.1
  tag: fieldkit/v0.4.1
  pypi_url: https://pypi.org/project/fieldkit/0.4.1/
  release_commit: 0b6986e
  stats_commit: b90de2c
post_rotation_commits:                     # appended after SYNC-HANDOFF rotation, still in same NEW window
  - 8ae42e4                                # lineage(gguf-publisher): TSV audit-trail rows
  - 08f3d72                                # frontier-scout: refresh 2026-05-14
  - 82d95ed                                # docs(fieldkit.eval): v0.4.1 open_book + subset kwargs catch-up
---

## Headline

Two bundled cycles ship in one Mac sweep window: (1) second Orionfold quant card [`Orionfold/Saul-7B-Instruct-v1-GGUF`](https://huggingface.co/Orionfold/Saul-7B-Instruct-v1-GGUF) — legal vertical, five-variant Spark-tested shape, LegalBench n=50 mini-eval; (2) **fieldkit v0.4.1 PyPI release** — `VerticalBench.from_jsonl` gains `open_book=...` + `subset=...` kwargs, lifted from inline script helpers into the package surface. Released at <https://pypi.org/project/fieldkit/0.4.1/>; git tag `fieldkit/v0.4.1`.

The release validates two hypotheses in one window:

1. **Publishing surface generalizes across verticals.** The Saul card needed *zero* changes in `fieldkit.publish` — only a new `VERTICAL_BENCH=legalbench` dispatch in `scripts/g3_measure_variants.py` + a new `scripts/legalbench_merge.py` helper. Card rendering, manifest generation, and the push pipeline all worked unmodified.
2. **VerticalBench.from_jsonl needed two kwargs to be production-correct.** `open_book=True` (auto-enabled for FinanceBench) prepends `evidence[*].evidence_text` to the question so the model sees the 10-K excerpt; this lifted accuracy from 0/50 closed-book to 14–18%/50 open-book on the same JSONL during the V1 retry. `subset=...` filters FinanceBench by `question_type` before the `limit` cap. Both are additive kwargs; no breaking changes.

## What Mac CC sweeps

Straight mirror across both bundled cycles — no destination-side rewrites. Concrete files / paths:

### Saul cycle (article + manifest + scripts)

- **`articles/becoming-a-legal-curator-on-spark/article.md`** — new long-form. ~1,700 words, customer-link audited. Frontmatter: `status: published`, `series: Machine that Builds Machines`, `book_chapters: [10, 11]`, `fieldkit_modules: [quant, publish, eval, lineage]`, `also_stages: [observability]`, `hf_url: https://huggingface.co/Orionfold/Saul-7B-Instruct-v1-GGUF`.
- **`articles/becoming-a-legal-curator-on-spark/evidence/lineage-Saul-7B-Instruct-v1/results.tsv`** — 5 variant rows with four-axis measurement. Q5_K_M = 0.72 LegalBench (best on bench, beats F16 = 0.68 within n=50 sampling variance), F16 = 5.917 ppl / 10.9 tg, Q4_K_M = 29.4 tg (throughput pick).
- **`src/content/artifacts/saul-7b-instruct-v1-gguf.yaml`** — second Phase-2 manifest. Catalog templates can render side-by-side with `finance-chat-gguf.yaml`. `license.tier=free`, `license.model=mit`, `vertical_eval_name="LegalBench (n=50, contains)"`.
- **`scripts/g3_measure_variants.py`** — `VERTICAL_BENCH` dispatch + scorer selection + lineage label generalization. Drives both finance + legal pipelines from one script.
- **`scripts/g3_build_first_quant.sh`** — Saul case in the model-id switch + thread-through of `VERTICAL_BENCH` and `LEGALBENCH_JSONL` to the measure step.
- **`scripts/legalbench_merge.py`** — new helper. Merges 5 task TSVs from `nguha/legalbench/data/<task>/test.tsv` into one JSONL the v0.4.1 `VerticalBench.from_jsonl(format="legalbench")` consumes directly.

### fieldkit v0.4.1 cycle (release + source lift)

- **`fieldkit/src/fieldkit/_version.py`** — `0.4.0` → `0.4.1`.
- **`fieldkit/src/fieldkit/eval/vertical.py`** — `VerticalBench.from_jsonl` signature widens with `open_book: bool | None = None` (auto-True for FinanceBench, False for LegalBench/generic) and `subset: str | None = None` (FinanceBench-only `question_type` filter). The two kwargs are additive — callers who don't pass them get the v0.4.0 behavior.
- **`fieldkit/tests/test_vertical_bench.py`** — `TestOpenBook` class added, +8 tests covering auto-default, explicit False, missing-evidence fallback, list-of-strings shape, subset filter, subset × limit composition.
- **`fieldkit/CHANGELOG.md`** — `[Unreleased]` block moved to `[0.4.1] — 2026-05-14`, with **Added**, **Test suite**, **Articles in this release**, and **Verified on Spark** sub-sections.
- **`scripts/g3_preflight_bench.py` + `scripts/g3_measure_variants.py`** — inline `_load_finbench_open_book` helpers deleted in favor of `VerticalBench.from_jsonl(open_book=True, subset=…)`. ~150 LOC lifted out of `scripts/` into `fieldkit/`.

### Auto-refreshed

- **`src/data/project-stats.json`** + **`README.md`** — 36 articles (+1 from prior sweep), 121,613 words (+1,520), 24,185 LOC (+159 from the v0.4.1 lift). Deployment stage gains an entry; Observability count nudges from 7 to 8.

### Post-rotation adds (also in this sweep window)

Two commits landed after the SYNC-HANDOFF was rotated but still belong to the same Mac-sweep window. Both touch Spark-authoritative globs per `SYNC-CONTRACT.md`; Mac mirrors them as-is.

- **`articles/becoming-a-gguf-publisher-on-spark/evidence/lineage/results.tsv`** (commit `8ae42e4`) — two re-run rows added (exp_id 003 Q5_K_M, exp_id 004 Q6_K) from a session-4 measurement re-cycle. The published article's variant table is unchanged; rows ship purely for audit-trail completeness.
- **`papers/**`** (commit `08f3d72`) — `frontier-scout refresh 2026-05-14`. New files: `papers/2605.{13779,09942,12978,12975,12501}/paper.md`, `papers/runs/2026-05-14/refresh-summary.md`. Updated: `papers/README.md`, `papers/papers.json`, `papers/runs/index.md`. Top dive-deep picks (in order): MinT (LoRA infra, MTBM), HAGE (RL graph memory, Second Brain), Useful-Memories-Faulty (Second Brain counter-narrative), Retrieval-is-Cheap (executable RAG, LLM Wiki), CUA-Action-Space (MTBM agentic).
- **`fieldkit/docs/api/eval.md`** (commit `82d95ed`) — v0.4.1 kwarg catch-up. Adds `open_book` + `subset` to the `VerticalBench.from_jsonl` signature line + example block + the v0.4.x-additions overview. The audit-docs script's symbol-coverage check passed at release time because `VerticalBench` itself is mentioned — kwarg drift on existing methods slipped through. A new `--strict-kwargs` flag on `audit_docs.py` (skill-local, in `~/.claude/skills/fieldkit-curator/scripts/`) catches this class going forward.

## What Mac CC does NOT need to do

- **No rename replays.** No new entries in `SYNC-RENAMES.log` this cycle. Existing entries remain fully `complete` after the prior `orionfoldllc → Orionfold` and `Autoresearch → Machine that Builds Machines` sweeps.
- **No new top-level pages.** Article lives at `/field-notes/becoming-a-legal-curator-on-spark/`; sorted by ordinal-desc per the existing convention. Manifest renders via the catalog template (already in place).
- **No destination-prose rewrites.** New article slots into the existing Astro article collection without schema or template changes. The `hf_url` field and the `artifacts` collection are reused.
- **No new skill IA mirroring.** `hf-publisher`, `hf-model-scout`, and `fieldkit-curator` all live in `~/.claude/skills/` (Spark CC user config), not in the source repo.
- **No breaking change handling.** The v0.4.1 surface additions are kwargs-only with sensible defaults; existing callers (including the v0.4.0-era finance-chat path) continue to work unmodified.

## Why both cycles bundle cleanly

The Saul card and the v0.4.1 release are causally linked: the kwargs that landed in v0.4.1 (`open_book`, `subset`) were the kwargs the Saul measurement run consumed via the script-side path. The release moves those kwargs from `scripts/` into `fieldkit/`. Mac sweep treating them as one cycle keeps the diff sequence coherent — the article references the release surface, the release CHANGELOG references the article as an in-window consumer.

## Source range

`7f1159e..HEAD` — six commits beyond the prior SYNC-HANDOFF reset:

1. `e0e599e` — Saul-7B-Instruct-v1-GGUF + LegalBench mini-eval (article + manifest + scripts + stats).
2. `0b6986e` — fieldkit v0.4.1 release commit (version bump + CHANGELOG finalization).
3. `b90de2c` — stats + README refresh post-v0.4.1 (LOC bump + missed Saul Deployment-stage entry).
4. `8ae42e4` — lineage TSV audit-trail rows for the gguf-publisher article (post-rotation add).
5. `08f3d72` — frontier-scout refresh 2026-05-14 (+13 papers, top 5 dive-deep — post-rotation add).
6. `82d95ed` — fieldkit/docs/api/eval.md catch-up: documents the v0.4.1 `open_book` + `subset` kwargs on `VerticalBench.from_jsonl` that shipped undocumented for ~30 min after the v0.4.1 PyPI release. Spawned the new `audit_docs.py --strict-kwargs` mode in the fieldkit-curator skill (skill-local, not in this repo).

## Spark-side gates that ran

- **Saul card pipeline:** `g3_preflight_bench.py` (1/5 FinanceBench, confirms open-book Q&A) → `g3_build_first_quant.sh all` (~5h end-to-end) → `hf-publisher` full-auto mode (5/5 verify_stage → detached `hf_push.py` upload, ~2h ETA for 36 GB).
- **fieldkit v0.4.1 release:** `audit_docs.py` 8/9 PASS (cli SKIP, no fails) → `audit_landing.py` 4/4 PASS → `pytest tests/` 375 passed, 3 skipped → git install verify ✅ → `twine check` PASSED for both wheel + sdist → PyPI upload → PyPI install verify ✅ (one CDN-lag retry, propagation < 90s).

## What Mac CC should look for after sweep

- The second vertical card should render side-by-side with `finance-chat-gguf` on whatever Mac-side catalog page exists for `artifacts/quants/`. Suggested ordering: chronological-desc (newest first) so legal sits above finance.
- The article should appear in the Machine that Builds Machines series listing. It's a peer to `becoming-a-gguf-publisher-on-spark` in series + book_chapters.
- The home-page "At a glance" infographic now reads 36 articles. The product counts shift (DGX Spark 29 → 30, Ollama 6 → 7); model-list panel may still miss Mistral/Saul-7B-Instruct depending on `compute_stats.py` MODELS regex coverage — minor cosmetic miss, not blocking.
- The fieldkit landing page (`/fieldkit/`) `version` prop should now display `0.4.1`. The CLI demo + hero + CTA footer all read `version` from `_version.py` via the Astro page; no manual edit needed, but a Mac-side build + smoke-screen should confirm the new version surfaced.
