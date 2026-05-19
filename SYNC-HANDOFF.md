<!--
  ⚠️ STATUS: NEW — drop received, awaiting Mac sweep.
  This file is one feature/release at a time, not a running log.
  At the next release prompt, **clear this entire file and start fresh** (do NOT append to existing sections).
  Last reset: 2026-05-19 (patent-strategist W3 data-prep article + corpus-contamination signature).

  Prior Mac sweep receipts (preserved here since SYNC-HANDOFF is per-release-not-running-log):
  - 2026-05-17 patent-strategist-v1-baseline cycle: swept at destination commit manavsehgal/ainative-business.github.io@df0066c (Mac PR against this repo).
  - 2026-05-16 medical-vertical / II-Medical-8B-GGUF cycle: swept at destination commit manavsehgal/ainative-business.github.io@10a74a5 (Mac PR #8, merged 2026-05-16 at ac1b427).
  - 2026-05-15 fieldkit v0.4.2 cycle: swept at destination commit manavsehgal/ainative-business.github.io@495196d (Mac PR #7, merged 2026-05-16 at d332d28).
  - 2026-05-15 cyber-vertical cycle: swept at destination commit manavsehgal/ainative-business.github.io@135bcad (Mac PR #6 merged 2026-05-15).
  - 2026-05-14 v0.4.1 cycle: swept at destination commit manavsehgal/ainative-business.github.io@e1b16de (Mac PR #5 merged 2026-05-14).
  - 2026-05-14 v0.4.0 cycle: swept at destination commit manavsehgal/ainative-business.github.io@f7ea7aa (Mac PR #4 against this repo — conflicted on rotation; safe to close, receipt captured here).
  - 2026-05-14 Orionfold/finance-chat-GGUF cycle: swept at destination commit manavsehgal/ainative-business.github.io@85f9307 (Mac PR #3, merged).
  - 2026-05-12 Autoresearch→MTBM rename: swept at destination commit manavsehgal/ainative-business.github.io@71293af (Mac PR #2, merged).
-->
---
release_slug: 2026-05-19-patent-strategist-W3-data-prep-article
status: NEW
source_range: 17f59ee..280ad3c
articles_added:
  - slug: fine-tune-data-prep-decisions-on-spark    # status: published — three-misdiagnoses field report on the W3 fine-tune, signature: CorpusContaminationLayers
articles_updated: []
signatures_added:
  - component: src/components/svg/CorpusContaminationLayers.astro    # 300x200 horizontal-stacked bars showing 2797/5000 (55.94 percent) family-prefix leakage as the dominant accent, plus diversify (20.24 percent) and duplicate-of (6.22 percent) secondary bars. Footer: "the trainer was fine / the corpus was 56% contaminated"
signatures_updated:
  - component: src/components/svg/PatentBracketSignature.astro       # stroke-width normalization: 0.4/0.6 → 0.5 (grid), 0.7 → 1 (data trace). Incidental fix to unblock the validator's hard-invariant gate on this commit; no visual change worth flagging.
artifacts_added: []                                  # no HF push this cycle — the W3 patent-strategist model is SHELVED pending corpus rebuild
artifacts_updated: []
fieldkit_modules_changed: []                         # no fieldkit changes this cycle
fieldkit_release: null
papers_added: []
papers_classify_count: 0
renames_to_replay: []
removes: []
new_top_level_pages: []
breaking_changes: []
destination_overrides_to_preserve:
  - "Live home page count and stage counts shift with this release — 40 published articles now (was 39). The 2-article gap to the Mac-authored landing pages (`ai-transformation`, `solo-builder-case-study`) persists; Mac owns those. If Mac wants to formalize, the `mirrors/destination-overrides.md` placeholder remains pending the first inventory PR."
hf_repos_added: []
civitai_artifacts_added: []
post_rotation_commits: []
---

## What shipped this cycle

One published deep-dive essay at `articles/fine-tune-data-prep-decisions-on-spark/` (~3400 words) reporting on the W3 patent-strategist fine-tune cycle that failed three times in a row. The article's narrative arc walks the three rounds of misdiagnosis the author hit before finding the actual bug:

1. **Round 1 — BOS/EOS misdiagnosis (session 39).** First train completed in 128 min wall with a clean monotonic loss curve. Probe row 1 ran the full `max_new_tokens=4096` budget without closing `</think>`. Diagnosed as missing BOS+EOS bookends in the training text. Memory `feedback_sft_eos_bos_explicit` saved. Fix patched in `scripts/build_train_jsonl.py` and the corpus was re-encoded.

2. **Round 2 — catastrophic forgetting (session 40).** Retrain ran for 131 min — identical loss curve to s39. Probe row 1 *still* failed (399s wall, `has_think=False`). Investigation: read TRL 1.4's `DataCollatorForLanguageModeling.torch_call()` source. It pads labels with literal `-100`, not by checking pad-token-id positions. Real EOS at the end of unpadded rows is in the gradient. The round-1 diagnosis was wrong about the layer. Reproduced AIME row 1 in isolation — model opens `<think>`, emits Korean Hangul mojibake, falls into a degenerate repetition loop (`1000=2^3*5^3 …` repeating ~40×), never closes `</think>`. Catastrophic forgetting on out-of-distribution input — real but incomplete.

3. **Round 3 — corpus contamination (the actual bug).** Patent-only probe (10 rows) showed presence=1.00 but think length=126 tok vs baseline 1252 (10× shorter). Reading per-row output exposed: no spaces between words, synth-pipeline meta-state leaking at start of every `<think>` block (`A1duplicateof3886.Diversifybyemphasizingthe§103…`), and hallucinated case cites (`Mayo Clinic v. Klein Electric` — Klein Electric doesn't exist; real case is *Mayo v. Prometheus*). Audited the source corpus: 2797/5000 rows (**55.94 percent**) have family-prefix meta in `<think>`, 311 (6.22 percent) have "duplicate of N", 1012 (20.24 percent) have "diversify by". The synth pipeline's producer-subagent state leaked verbatim into 56 percent of training rows.

The article lands the lesson upstream of the trainer: corpus-quality gates are 20× cheaper than retrain-and-debug, and an emission-only probe gate can't catch corpus contamination — a content gate (LLM judge on think coherence) is the missing piece.

Voice is the standard ai-field-notes deep-dive essay: first-person, fork-by-fork, measured numbers, every claim backed by a specific incident. 8 explainers (2× `:::define`, 2× `:::why`, 2× `:::pitfall`, 1× `:::math`, 1× `:::deeper`, 1× `:::hardware`). One inline `<figure class="fn-diagram">` (dual-path comparison: intended pipeline vs actual three-rounds-of-misdiagnosis trajectory; 900×440 viewBox). New signature SVG `CorpusContaminationLayers.astro` (300×200 horizontal-stacked bars — 56 percent family-prefix as accent, 20 percent diversify and 6 percent duplicate-of as secondaries; footer *"the trainer was fine / the corpus was 56% contaminated"*).

## Mac sweep guidance

This is a content-only release — one new article, two SVG components (one new, one stroke-width patch), three forensic probe JSONs, one helper script. No fieldkit release, no HF push, no renames, no schema changes.

**Files Mac should sweep on the destination side:**

- `articles/fine-tune-data-prep-decisions-on-spark/article.md` (new) — published, `series: Machine that Builds Machines`, `stage: fine-tuning`, `signature: CorpusContaminationLayers`
- `articles/fine-tune-data-prep-decisions-on-spark/transcript.md` (new) — provenance file, not user-rendered but synced for parity
- `src/components/svg/CorpusContaminationLayers.astro` (new signature)
- `src/components/svg/PatentBracketSignature.astro` (stroke-width fix only — no visual diff worth re-eyeballing on destination, but the file changed)
- `src/data/project-stats.json` (refreshed: now 40 articles / 134,203 words / 25,508 LOC)
- `README.md` (regenerated by `refresh_readme.py`)

**Files that are source-only (do not sweep):**

- `probes/patent-strategist-v1-2026-05-19.json` — forensic probe output (4-row aborted run from s39)
- `probes/patent-strategist-v1-2026-05-19.patent-only.json` — patent-only probe (10 rows, s40), referenced by the article as evidence
- `probes/patent-only-10q.jsonl` — filtered probe set (10 patent rows)
- `scripts/build_train_jsonl.py` — corpus→text-shape conversion (with the BOS/EOS fix that turned out unnecessary at trainer level)

All in `probes/` and `scripts/` per the SYNC-CONTRACT — Spark-authoritative, Mac doesn't render these.

**Confirmation points:**

1. Home page "At a glance" infographic should re-render with article count = 40 (was 39 last cycle).
2. Stage filter `/stage/fine-tuning/` should show the new card with `Article №NN` (ordinal is git-first-add derived; let `article-order.mjs` compute it on destination).
3. Series filter `/series/machine-that-builds-machines/` should include the new article.
4. Card on home + stage pages should render the `CorpusContaminationLayers` signature thumbnail (300×200, three horizontal bars, indigo-primary accent on the 56 percent bar).
5. The inline fn-diagram in the article body should render at 900×440 viewBox via the existing `.prose .fn-diagram` breakout CSS. No new diagram-system changes required.

## What did NOT ship

- **The W3 patent-strategist model is SHELVED** (16.38 GB merged-BF16 sits on the Spark, not promoted to HF). No `Orionfold/patent-strategist-v1-GGUF` push this cycle. The paired bench dataset `Orionfold/patent-strategist-bench-v0.1` is held pending corpus rebuild.
- **No fieldkit release.** No promotion candidates landed in `[Unreleased]` this cycle.
- **No renames.** SYNC-RENAMES.log unchanged.
- **No new top-level pages.** Mac chrome surfaces (`/book/`, `/pricing/`, `/about/`, `/artifacts/<kind>/`) are untouched.

## Next cycle expectations

The patent-strategist arc resumes after a corpus rebuild step (write `scripts/clean_corpus_meta.py` to strip the leaked meta-state; patch `claude-corpus-synth/verify_chunk.py` to refuse contaminated chunks at write time; rerun the 5k synth with the cleaner inline; retrain for ~131 min; reprobe with a content gate added). Best-case timeline for the next paired-drop publish (model + bench): 2-3 sessions, gated on corpus rebuild + content-judging probe scaffold. The next sync drop from this branch will either be that paired drop or a smaller article-only sync if the corpus rebuild blocks.
