<!--
  ✅ STATUS: SHIPPED — Mac swept at destination commit manavsehgal/ainative-business.github.io@df0066c on 2026-05-17.
  This file is one feature/release at a time, not a running log.
  At the next release prompt, **clear this entire file and start fresh** (do NOT append to existing sections).
  Last reset: 2026-05-17 (patent-strategist v1 baseline article + fieldkit v0.4.3 + skills-in-git).

  Prior Mac sweep receipts (preserved here since SYNC-HANDOFF is per-release-not-running-log):
  - 2026-05-17 patent-strategist-v1-baseline cycle: swept at destination commit manavsehgal/ainative-business.github.io@df0066c (this PR).
  - 2026-05-16 medical-vertical / II-Medical-8B-GGUF cycle: swept at destination commit manavsehgal/ainative-business.github.io@10a74a5 (Mac PR #8, merged 2026-05-16 at ac1b427).
  - 2026-05-15 fieldkit v0.4.2 cycle: swept at destination commit manavsehgal/ainative-business.github.io@495196d (Mac PR #7, merged 2026-05-16 at d332d28).
  - 2026-05-15 cyber-vertical cycle: swept at destination commit manavsehgal/ainative-business.github.io@135bcad (Mac PR #6 merged 2026-05-15).
  - 2026-05-14 v0.4.1 cycle: swept at destination commit manavsehgal/ainative-business.github.io@e1b16de (Mac PR #5 merged 2026-05-14).
  - 2026-05-14 v0.4.0 cycle: swept at destination commit manavsehgal/ainative-business.github.io@f7ea7aa (Mac PR #4 against this repo — conflicted on rotation; safe to close, receipt captured here).
  - 2026-05-14 Orionfold/finance-chat-GGUF cycle: swept at destination commit manavsehgal/ainative-business.github.io@85f9307 (Mac PR #3, merged).
  - 2026-05-12 Autoresearch→MTBM rename: swept at destination commit manavsehgal/ainative-business.github.io@71293af (Mac PR #2, merged).
-->
---
release_slug: 2026-05-17-patent-strategist-v1-baseline
status: SHIPPED
source_range: f005b52..17f59ee
articles_added:
  - slug: patent-strategist-v1-baseline-on-spark    # status: published — three-mode bracket on R1-0528-Qwen3-8B Q5_K_M, signature: PatentBracketSignature
articles_updated: []
signatures_added:
  - component: src/components/svg/PatentBracketSignature.astro    # 300x200 three-bar D-mcq ladder (closed 0.62 / retrieval 0.85 / oracle 0.95) + dashed random-4-choice baseline + dashed lift connectors (+0.225 / +0.100)
artifacts_added: []                                  # no HF push this cycle — bench/eval article, not a model release
artifacts_updated: []
fieldkit_modules_changed:
  - eval                                             # patent-strategist format wired into VerticalBench.from_jsonl; 4 new scorers (patent_claim_validity, office_action_argument, irac_structure, prior_art_relevance + _full); mcq_letter promoted to top-level + bug-fix to findall-last so reasoning-model elimination-then-answer prose scores on the conclusion
fieldkit_release:
  version: 0.4.3
  pypi: https://pypi.org/project/fieldkit/0.4.3/
  tag: fieldkit/v0.4.3
  notes: |
    Adds patent-strategist scorer family (4 new scorers + dispatch map + rubric markdown
    bundled in wheel). mcq_letter promoted from internal to top-level export. No schema
    breakage. PyPI live; install with `pip install fieldkit==0.4.3`.
papers_added: []
papers_classify_count: 0
renames_to_replay: []
removes: []
new_top_level_pages: []
breaking_changes: []
destination_overrides_to_preserve:
  - "Live home page lists 40 articles vs local's 39 published. The 2-article gap is the two Mac-authored landing pages already on live (`ai-transformation`, `solo-builder-case-study`) — Mac owns those, no Spark sweep action needed. If Mac wants to formalize this in `mirrors/destination-overrides.md`, the placeholder file is still empty pending the first inventory PR."
hf_repos_added: []
civitai_artifacts_added: []
post_rotation_commits:
  - 17f59ee  # patent-strategist: flip baseline article upcoming → published
---

## Headline

The patent-strategist v1 **three-mode bracket** ships as a published article. `DeepSeek-R1-0528-Qwen3-8B Q5_K_M` ran the 200-row `patent-strategist-v0.1` bench under closed-book, retrieval, and oracle context modes; D-mcq accuracy climbs **0.625 → 0.850 → 0.950**; overall mean across the 90 scorer-supported rows (B + D-mcq + D-irac) lands **0.397 → 0.489 → 0.541**. Closed-to-retrieval lift is **2.25× the retrieval-to-oracle gap** — the bracket says fine-tune the model first, retriever upgrades later. Three scaffold bugs surfaced mid-flight (options-blind D-mcq prompts, first-Option-wins `mcq_letter` regex, max-tokens truncation mid-`<think>`); two are patched in this release, one is deferred to W4.

This cycle also ships `fieldkit v0.4.3` on PyPI — the patent-strategist scorer family (`patent_claim_validity`, `office_action_argument`, `irac_structure`, `prior_art_relevance` + `_full`), the `mcq_letter` findall-last fix, and a 26-item kwarg-drift docs cleanup that takes the `--strict-kwargs` audit to **0 FAIL / 0 WARN**.

## What Mac CC sweeps

Spark-authoritative files (per `[[reference_sync_contract]]`). No destination-side rewrites, no schema changes, no renames.

- **`articles/patent-strategist-v1-baseline-on-spark/article.md`** — NEW article, **`status: published`** with the full 9-section draft. Frontmatter: `product: Foundation`, `stage: fine-tuning`, `difficulty: advanced`, `time_required: "~10 hours (mostly automated overnight sweeps)"`, `tags: [eval, rag, reasoning-models, llama-cpp, deepseek-r1, vertical-bench, patent-strategist]`, `signature: PatentBracketSignature`. Body carries the three-mode bracket fn-diagram (color-distinct indigo/blue/primary lanes converging on a haloed model + scorer endpoint with right-edge score callouts), the eval-scaffold-and-bench overview, the journey (picking the model, picking the quantization, standing up llama-server with embedded console code blocks, building the eval driver, plausibility checks), the results table with the full closed/retrieval/oracle ladder, three findings, three bug retros, the targeted-fine-tune unlock, and a closing tie-back to the next-vertical scaffold.
- **`articles/patent-strategist-v1-baseline-on-spark/transcript.md`** — provenance for the article (drafted mid-T10 sweep). One-page session-source record.
- **`src/components/svg/PatentBracketSignature.astro`** — NEW signature (300×200) for the card thumbnail on `/stage/fine-tuning/`. Three bars labeled `closed / retrieval / oracle` at heights 0.625 / 0.850 / 0.950, indigo→blue→primary gradient, with the dashed 0.25 random-4-choice baseline, halo behind the oracle bar, and two dashed connector lines carrying the `+0.225` and `+0.100` lift labels. Footer caption hammers the headline thesis: *"closed-to-retrieval lift is 2.25× the retrieval-to-oracle gap → fine-tune the model first, retriever upgrades later"*.
- **`evidence/patent-strategist/baseline-runs/`** — NEW evidence root. Five run-dirs:
    - `20260517-102017-retrieval-retonly-4da81a/` — retrieval-only 3-row smoke (no inference)
    - `20260517-104509-retrieval-518c10/` — 5-row retrieval smoke (post-server-up)
    - `20260517-104908-retrieval-136ef4/` — full 200-row retrieval sweep, 3h22m, overall 0.489
    - `20260517-141203-oracle-e6885f/` — full 200-row oracle sweep, 2h52m, overall 0.541
    - `20260517-170410-closed-b8cfe9/` — full 200-row closed-book sweep, overall 0.397 (initial 175 + 13min E-tail resume after power outage)
- **`scripts/seed_patent_bench.py`** — NEW (~530 LOC, stdlib-only). 200-row bench seeder via Claude Opus through `claude -p` OAuth; landed in session 25.
- **`scripts/review_patent_bench.py`** — NEW (~370 LOC, stdlib-only). Human-in-the-loop review CLI with atomic per-decision JSONL rewrite; synthesized-rows-first priority; smoke-tested round-trip clean.
- **`scripts/run_rag_baseline.py`** — NEW (~500 LOC). T10 driver, three modes (closed/retrieval/oracle) + retrieval-only side-mode; BGE-small/FAISS lookup; OpenAI-compatible chat client; scorer dispatch to `PATENT_STRATEGIST_SCORER_FNS`. Now renders D-mcq `options` as labeled choices (the prior version dropped them on the floor — caught on a 5-row smoke that scored 5/5 because R1 invented plausible options).
- **`scripts/rescore_predictions.py`** — NEW small helper. Replays scorers against an existing `predictions.jsonl` in place after a scorer-fix; regenerates `scores.json`. Used here after the `mcq_letter` findall-last fix lifted retrieval D-mcq 0.775 → 0.850 without re-running inference.
- **`scripts/resume_closed_e_tail.py`** — NEW one-off recovery script. Power outage truncated the closed-book sweep at row 175/200; this resumes the missing 25 E-shape rows against the same llama-server endpoint and appends to the same `predictions.jsonl`.
- **`scripts/build_rag_index.py`** — NEW (W1 T3). BGE-small + FAISS IndexFlatIP build over the patent-strategist corpus; 39,777 vectors total.
- **`scripts/pull_mpep_static.py`** — NEW (T2 followup). USPTO static-HTML MPEP mirror puller; 2,047 subsections, 4,437 chunks; takes prosecution-query retrieval ceiling from 0.74 → 0.85.
- **`scripts/pull_patentmatch_naumann.py`** — NEW (T2 followup). Reverse-engineered HiDrive share-token flow for canonical HPI-Naumann PatentMatch; 25,340 EPO claim-pair rows.
- **`fieldkit/src/fieldkit/eval/__init__.py`** + **`fieldkit/tests/eval/test_mcq_letter.py`** — EXTENDED. `mcq_letter` now uses `re.findall(...)[-1]` so reasoning-model elimination prose (`"Option A is incorrect ... Option D is incorrect ... Answer: B"`) scores on the conclusion rather than the first eliminated distractor. New test `test_concluding_answer_wins_over_elimination` locks the behavior; +3 correct rows on retrieval D-mcq (40-row slice).
- **`fieldkit/` v0.4.3 on PyPI** — patent-strategist scorers shipped (`fieldkit.eval.vertical.PATENT_STRATEGIST_SCORER_FNS`), rubric markdown bundled in wheel. 507 passed / 2 skipped in suite. **No code-only change required on Mac side** — `fieldkit/docs/api/` pages already mirror the new symbols (audit pass landed in `ae85b66`).
- **`src/data/project-stats.json`** — auto-refreshed twice this cycle: first after the upcoming-placeholder commit (`1c3e23a` — articles `38 published + 5 upcoming`), then again after the `upcoming → published` flip (`17f59ee` — articles `39 published + 4 upcoming = 43 total`). Word total **130,727** (up from 127,405 on the flip — the article's prose was excluded while it carried `status: upcoming`). LOC unchanged at 25,508. Fine-tuning stage moved from 8 → 9. `DGX Spark` product climbed to 31 (tied with NIM).
- **`.claude/skills/` tree** — NEW in git. The operational skills layer is now versioned alongside the source. Mac CC's skill tree is separate (`~/.claude/skills/` on Mac); this is the Spark CC user-config layer (`tech-writer`, `hf-publisher`, `hf-model-scout`, `fieldkit-curator`, `frontier-scout`, `nvidia-learn-stats`, etc.). Per `[[reference_sync_contract]]`: skills are Mac-and-Spark each-own-their-own; **no Mac sweep action needed**.

### Auto-refreshed (no Mac-side action needed)

The stats infographic + README article index reflect the new article entry and the `upcoming → published` flip. Top-level numbers shifted as noted above. The `/stage/fine-tuning/` page now renders the patent-strategist card with the `PatentBracketSignature` thumbnail (verified via Playwright smoke; zero console errors).

## What Mac CC does NOT need to do

- **No fieldkit destination action.** `fieldkit/docs/api/` is the canonical reference layer and already mirrors v0.4.3's exports (the kwarg-drift baseline was cleared in `ae85b66`). PyPI publishing is automated from the fieldkit-curator skill — Mac doesn't republish.
- **No HF push.** This is a bench/eval cycle, not a model release. No `Orionfold/*` repo lands this cycle.
- **No artifact manifest.** `src/content/artifacts/` is unchanged; no new YAML.
- **No rename replays.** `SYNC-RENAMES.log` unchanged since the 2026-05-14 Orionfold rename (already swept complete at Mac PR #3).
- **No new top-level pages.** The new article folder is under the existing `articles/` content collection; no new section, no new stage page (existing `fine-tuning` filter renders it via `signature: PatentBracketSignature`).
- **No skill IA mirroring.** Skills live under each side's `.claude/skills/` independently.
- **No destination-overrides change from us.** The 2-article gap between local home (39 published) and live home (40 articles) is the two Mac-authored landing pages (`ai-transformation`, `solo-builder-case-study`); this remains a Mac-owned concern. If Mac wants to surface those counts in `src/data/project-stats.json` so the local home matches the live count, the proper channel is the placeholder `mirrors/destination-overrides.md` per `[[reference_destination_overrides_mirror]]`.

## Notes for the next sync

- **patent-strategist W3 — fine-tune kickoff** is the next track. The bracket measured a closed-to-retrieval gap of 22.5 percentage points on D-mcq; W3 will GRPO-fine-tune R1-0528-Qwen3-8B against the deterministic-scorable shapes (D-mcq + D-irac) to close that specific gap. That cycle will likely cut a new fieldkit minor (`fieldkit.training` extensions for the patent reward function) and ship a new article. ETA ~2 weeks of overnight runs.
- **A-shape + D-oa scorer signature mismatch** is a small W4 follow-up. `score_prediction(shape, prediction, gold_label)` raises TypeError on `patent_claim_validity` (A) and `office_action_argument` (D-oa) for both retrieval and oracle runs — same pattern across all three modes, so the inter-mode comparison is unaffected, but the scorer signature wants either a `judge=` kwarg or a structured-gold wrapper. Tracked in spec §3.5 Judge-backend follow-up.
- **Spec evolution.** `specs/patent-strategist-v1.md` predates the eval matrix's actual results; §3.5 (Judge backend), §5.5 (1000-row ramp), §5.6 (reasoning-budget cap) all want a post-bracket revision. Holding for the W3 follow-up.
