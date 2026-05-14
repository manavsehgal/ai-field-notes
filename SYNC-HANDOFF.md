<!--
  🆕 STATUS: NEW — pending Mac sweep.
  This file is one feature/release at a time, not a running log.
  At the next release prompt, **clear this entire file and start fresh** (do NOT append to existing sections).
  Last reset: 2026-05-14 (prior content covered the fieldkit v0.4.0 PyPI cut + landing-page drift fix — sweep status NEW awaiting Mac CC. THIS release supersedes that scope for the Mac sweep; the v0.4.0 PyPI cut work landed in commits e322af2..2190824 and is bundled here.).
-->
---
release_slug: 2026-05-14-orionfold-saul-7b-instruct-v1-gguf
status: NEW
source_range: 7f1159e..HEAD
articles_added:
  - becoming-a-legal-curator-on-spark      # new long-form deep-dive; status: published; customer-link audited; signature SVG reused from VerticalCuratorRetry (no new SVG this cycle)
articles_updated: []
artifacts_added:
  - src/content/artifacts/saul-7b-instruct-v1-gguf.yaml   # second real-world Phase-2 manifest (after finance-chat-gguf.yaml from prior cycle) — license.tier=free, license.model=mit, vertical_eval populated from LegalBench n=50
artifacts_updated: []
fieldkit_modules_changed: []               # no fieldkit source changes this cycle; the v0.4.1 lift (open_book/subset on VerticalBench.from_jsonl) landed in commit 7f1159e at the prior session end — already in [Unreleased], no new code here
renames_to_replay: []
removes: []
new_top_level_pages: []
breaking_changes: []
destination_overrides_to_preserve: []
hf_repos_added:
  - Orionfold/Saul-7B-Instruct-v1-GGUF     # 5 GGUF variants of Equall/Saul-7B-Instruct-v1, mistral chat_format, MIT license, LegalBench-scored (n=50, contains)
civitai_artifacts_added: []
---

## Headline

Second Orionfold quant card ships: [`Orionfold/Saul-7B-Instruct-v1-GGUF`](https://huggingface.co/Orionfold/Saul-7B-Instruct-v1-GGUF). Same five-variant Spark-tested shape as `finance-chat-GGUF` from the prior cycle, swapping FinanceBench for a curated 5-task LegalBench subset (overruling, abercrombie, proa, contract_nli_confidentiality_of_agreement, diversity_1 — 10 questions each, scored with `fieldkit.eval.contains`). Saul is a Mistral-7B-Instruct SFT (MIT license) from Equall, paper at arXiv:2403.03883.

The release validates that the v0.4.0 publishing surface generalizes across verticals: no `fieldkit.publish` changes were needed. Two measurement-script extensions did land, both in `scripts/` (not `fieldkit/`): a new `VERTICAL_BENCH={financebench,legalbench}` env knob in `g3_measure_variants.py` that dispatches between numeric_match (finance) and contains (legal) scorers, and a per-model case in `g3_build_first_quant.sh` that auto-resolves `MODEL_LICENSE`, `CHAT_FORMAT`, and `ARTICLE_SLUG` for Saul. A new helper at `scripts/legalbench_merge.py` produces the merged 50-question JSONL from the upstream `nguha/legalbench` 162-task dataset.

## What Mac CC sweeps

Straight mirror — no destination-side rewrites. Concrete files / paths:

- **`articles/becoming-a-legal-curator-on-spark/article.md`** — new long-form. ~1,700 words, customer-link audited (no V1 retry narrative, no strategy leak, no competitor punches; voice stays Manav-deep-dive). Frontmatter: `status: published`, `series: Machine that Builds Machines`, `book_chapters: [10, 11]`, `fieldkit_modules: [quant, publish, eval, lineage]`, `also_stages: [observability]`, `hf_url: https://huggingface.co/Orionfold/Saul-7B-Instruct-v1-GGUF`. Same Methods-link convention as the finance article.
- **`articles/becoming-a-legal-curator-on-spark/evidence/lineage-Saul-7B-Instruct-v1/results.tsv`** — 5 variant rows (one per GGUF), each with the four-axis measurement: F16 = 0.68 / 5.917 / 10.9 tg / 5.2 sustain; Q5_K_M = 0.72 (best on bench) / 5.938 / 20.2 tg / 2.4 sustain; Q4_K_M = 0.62 / 5.986 / 29.4 tg / 1.7 sustain (throughput pick); Q6_K = 0.68 / 5.925 / 22.4 tg / 2.5 sustain; Q8_0 = 0.66 / 5.914 / 7.3 tg / 7.2 sustain (anomalously slow — same pattern as finance-chat Q8_0).
- **`src/content/artifacts/saul-7b-instruct-v1-gguf.yaml`** — second Phase-2 artifact manifest. The catalog page templates (Mac-side editorial) can render it alongside `finance-chat-gguf.yaml` as the second vertical card. Fields populated: 5 variants, perplexity per variant, spark_tokens_per_sec per variant, sustained_load_minutes (worst-case across variants, per Q9 convention), vertical_eval as `{variant: legalbench_accuracy}`, vertical_eval_name `"LegalBench (n=50, contains)"`, license `{tier: free, model: mit}`.
- **`scripts/g3_measure_variants.py`** — `VERTICAL_BENCH` dispatch + scorer selection + lineage label generalization. Drives both finance + legal pipelines from one script. Mac doesn't need to mirror, but the source path may show up in cross-link audits.
- **`scripts/g3_build_first_quant.sh`** — Saul case in the model-id switch + thread-through of `VERTICAL_BENCH` and `LEGALBENCH_JSONL` to the measure step.
- **`scripts/legalbench_merge.py`** — new helper. Merges 5 task TSVs from `nguha/legalbench/data/<task>/test.tsv` into one JSONL the v0.4.1 `VerticalBench.from_jsonl(format="legalbench")` consumes directly. Mac doesn't need to mirror; lives in source-side scripts/ only.
- **`src/data/project-stats.json`** + **`README.md`** — auto-refreshed: 36 articles (+1), 121,613 words (+1,520), 24,185 LOC (+159).

## What Mac CC does NOT need to do

- **No fieldkit source mirroring this cycle.** No changes under `fieldkit/src/` since the prior `7f1159e` commit. The v0.4.1 release stays in `[Unreleased]` — bundling decision (cut now vs cut with next vertical) is queued.
- **No destination-prose rewrites.** New article slots into the existing Astro article collection without schema or template changes. The `hf_url` field, introduced in the prior cycle, is reused here.
- **No rename replays.** No new entries in `SYNC-RENAMES.log` this cycle. Existing entries remain fully `complete` after the prior `orionfoldllc → Orionfold` and `Autoresearch → Machine that Builds Machines` sweeps.
- **No new skill IA.** `hf-publisher` was used to push (sibling to `fieldkit-curator`), but it lives in `~/.claude/skills/` (Spark CC user config), not in the source repo. Same protocol as the prior `hf-publisher` and `audit_landing.py` introductions.
- **No new top-level pages.** Article lives at `/field-notes/becoming-a-legal-curator-on-spark/`; sorted by ordinal-desc per the existing convention. Manifest renders via the catalog template (already in place from prior cycle).

## Why a second vertical card matters

The publishing-surface hypothesis from the prior cycle was: "the four-axis card shape generalizes across verticals". This release tests that. Two signals support the generalization:

1. **No `fieldkit.publish` changes were needed.** The `ModelCard`, `ArtifactManifest`, `publish_quant`, and verify_stage gate all worked unmodified. The `vertical_eval` + `vertical_eval_name` kwargs were already public.
2. **The script-side work was small + structured.** Adding `VERTICAL_BENCH=legalbench` is one switch statement and one scorer-selection branch; no plumbing was added that wasn't already implied by the existing FinanceBench path.

The remaining vertical-curator scaling question — "can we add a third vertical (cyber, medical) without further script changes?" — is the next test. The shape of `legalbench_merge.py` suggests yes (any bench with `text + answer` rows + an instruction template can use the existing path).

## Source range

`7f1159e..HEAD` — pending the post-push commit cluster. Expected commit shape: one bundled commit covering the article + manifest + lineage + scripts + stats refresh, with the live HF push URL in the body.

## Spark-side gates that ran

- `g3_preflight_bench.py` — 1/5 on FinanceBench (threshold ≥1). Confirms Saul does open-book Q&A (not a continued-pretrain trap). LegalBench-shaped preflight extension still queued.
- `verify_stage.sh` — 5/5 PASS after article scaffold landed (initial run failed on the Methods-link check; resolved by drafting the article before the live push, per the customer-link audit convention).
- `g3_build_first_quant.sh all` — preflight → download (cached, 5/6 shards from a prior partial; one resume) → preflight-bench → probe → quantize (5 variants in ~3 min each) → measure (4 axes × 5 variants in ~13 min each) → publish-dryrun. ~5 hours wall-time end-to-end.
- `hf-publisher` skill (full-auto mode) — 5/5 verify_stage, then detached `hf_push.py` upload. ETA 1.5–2 hours for the 36 GB push.

## What Mac CC should look for after sweep

- The second vertical card on a hypothetical Mac catalog page (artifacts/quants/) should render side-by-side with `finance-chat-gguf` as proof the shape generalizes. Suggested ordering: chronological-desc (newest first) so legal sits above finance.
- The article should appear in the Machine that Builds Machines series at `/series/machine-that-builds-machines/` (or whatever the Mac-side IA names the listing page). It's a peer to `becoming-a-gguf-publisher-on-spark` in series + book_chapters.
- The home-page "At a glance" infographic now reads 36 articles. The `pgvector` / Spark / NIM counts shift; check the model-list panel includes Mistral / Saul if the regex caught either (currently the MODELS list in `compute_stats.py` covers Llama and Qwen families; Mistral/Saul-7B-Instruct may not be counted as a separate model entry — minor cosmetic miss, not blocking).
