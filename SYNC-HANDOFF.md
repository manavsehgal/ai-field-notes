<!--
  ⚠️ STATUS: SHIPPED — Mac swept at destination commit manavsehgal/ainative-business.github.io@7b1b62e on 2026-05-19.
  This file is one feature/release at a time, not a running log.
  At the next release prompt, **clear this entire file and start fresh** (do NOT append to existing sections).
  Last reset: 2026-05-19 (patent-strategist bench v0.1 HF push + synth-skill meta-state gates).

  Prior Mac sweep receipts (preserved here since SYNC-HANDOFF is per-release-not-running-log):
  - 2026-05-19 patent-strategist-bench-v0.1-published-plus-synth-skill-patches cycle: swept at destination commit manavsehgal/ainative-business.github.io@7b1b62e (Mac PR #11 against this repo).
  - 2026-05-19 patent-strategist-W3-data-prep-article cycle: swept at destination commit manavsehgal/ainative-business.github.io@c71a5bb (Mac PR #10 against this repo).
  - 2026-05-17 patent-strategist-v1-baseline cycle: swept at destination commit manavsehgal/ainative-business.github.io@df0066c (Mac PR #9 against this repo).
  - 2026-05-16 medical-vertical / II-Medical-8B-GGUF cycle: swept at destination commit manavsehgal/ainative-business.github.io@10a74a5 (Mac PR #8, merged 2026-05-16 at ac1b427).
  - 2026-05-15 fieldkit v0.4.2 cycle: swept at destination commit manavsehgal/ainative-business.github.io@495196d (Mac PR #7, merged 2026-05-16 at d332d28).
  - 2026-05-15 cyber-vertical cycle: swept at destination commit manavsehgal/ainative-business.github.io@135bcad (Mac PR #6 merged 2026-05-15).
  - 2026-05-14 v0.4.1 cycle: swept at destination commit manavsehgal/ainative-business.github.io@e1b16de (Mac PR #5 merged 2026-05-14).
  - 2026-05-14 v0.4.0 cycle: swept at destination commit manavsehgal/ainative-business.github.io@f7ea7aa (Mac PR #4 against this repo — conflicted on rotation; safe to close, receipt captured here).
  - 2026-05-14 Orionfold/finance-chat-GGUF cycle: swept at destination commit manavsehgal/ainative-business.github.io@85f9307 (Mac PR #3, merged).
  - 2026-05-12 Autoresearch→MTBM rename: swept at destination commit manavsehgal/ainative-business.github.io@71293af (Mac PR #2, merged).
-->
---
release_slug: 2026-05-19-patent-strategist-bench-v0.1-published-plus-synth-skill-patches
status: SHIPPED
source_range: c1722c2..HEAD
articles_added: []
articles_updated: []
signatures_added: []
signatures_updated: []
artifacts_added:
  - slug: patent-strategist-bench-v0.1                    # kind: bench, hf_repo: Orionfold/patent-strategist-bench-v0.1, license tier: free / model: cc-by-4.0; manifest at src/content/artifacts/patent-strategist-bench-v0.1.yaml; pairs with the patent-strategist-v1-baseline-on-spark + fine-tune-data-prep-decisions-on-spark methodology articles
artifacts_updated: []
fieldkit_modules_changed: []                              # no fieldkit changes this cycle (skill patches are .claude/skills/, not fieldkit)
fieldkit_release: null
papers_added: []
papers_classify_count: 0
renames_to_replay: []
removes: []
new_top_level_pages: []
breaking_changes: []
destination_overrides_to_preserve:
  - "Article counts unchanged this cycle (40 published, 4 upcoming). Mac chrome stays in sync. The 2-article gap to the Mac-authored landing pages (`ai-transformation`, `solo-builder-case-study`) persists; Mac owns those."
hf_repos_added:
  - repo: Orionfold/patent-strategist-bench-v0.1         # https://huggingface.co/datasets/Orionfold/patent-strategist-bench-v0.1, commit d3dbf76; repo_type: dataset; CC-BY-4.0; 200 rows × 7 shapes (A=50, B=40, C=20, D-irac=10, D-mcq=40, D-oa=10, E=30); pushed 2026-05-19T21:52Z
civitai_artifacts_added: []
post_rotation_commits: []
---

## What shipped this cycle

The **paired bench dataset for the patent-strategist arc** — `Orionfold/patent-strategist-bench-v0.1` — landed on HuggingFace as the lasting customer-facing artifact from the W1+W2 work. The W3 model itself is still shelved pending the planned corpus rebuild on the new NVIDIA-base + Unsloth stack (see HANDOFF.md Phase C); shipping the bench independently of the model is deliberate. The bench was always a sound, decoupled methodology artifact — anchored to public sources (USPTO MPEP, HPI-Naumann PatentMatch, BIGPATENT), oracle-context attached to every row, three-mode evaluation built in — and the two shipped methodology articles already reference it.

**Bench HF push details:**

- Dataset slug: `Orionfold/patent-strategist-bench-v0.1` (repo_type=dataset, CC-BY-4.0)
- Commit on HF: `d3dbf76` — pushed via `scripts/publish_patent_bench.py --push` against the staged `/tmp/hf-stage/patent-strategist-bench-v0.1/` tree (README.md 8.6 KB + data/train.jsonl 611 KB)
- Schema: 200 rows × 7 shapes (A=50 / B=40 / C=20 / D-irac=10 / D-mcq=40 / D-oa=10 / E=30), columns include `qid`, `question`, `family`, `shape`, `oracle_context`, `gold_label`, plus per-source `source_metadata` JSON

**Dataset-card customer-link audit (per `feedback_customer_link_audit`):**

- The companion paragraph now links **both** shipped methodology articles — `patent-strategist-v1-baseline-on-spark` (W1 baseline) and `fine-tune-data-prep-decisions-on-spark` (W2/W3 data-prep field report) — not just the W1 baseline as before
- The v1.0 roadmap line was de-committed from the shelved `DeepSeek-R1-0528-Qwen3-8B` base; it now reads "specific base model pending; see the methodology articles for the constraints the next attempt has to clear." This avoids promising a paired drop on a model that's been retired in favor of an NVIDIA-base + Unsloth research path
- All other card content — quick stats, schema, sources & licensing, limitations, citation — unchanged from the audit baseline

**`claude-corpus-synth` skill patches (preventing the next corpus rebuild from repeating the 56% meta-state contamination):**

- `.claude/skills/claude-corpus-synth/scripts/verify_chunk.py` — three new content-gates (`META_FAMILY_PREFIX_RE` / `META_DUPLICATE_OF_RE` / `META_DIVERSIFY_RE`) reject any chunk whose `<think>` body starts with `A1/A2/A4/E1/E2`, contains `"duplicate of N"`, or contains `"diversify by"`. Tested against the contaminated 5000-row source corpus — exact match to the diagnosis: 56 % family-prefix (2797 rows), 6 % duplicate-of (275 rows), 8 % diversify-by within `<think>` body (375 rows). Pre-patch the verifier was symbol-coverage-only (line count / `<think>` presence / length); the meta-state was invisible
- `.claude/skills/claude-corpus-synth/references/producer-subagent-prompt.md` — explicit "⚠️ Producer working-notes MUST NOT leak into `<think>`" section added, with rewrite example and a per-row self-check: "would a working patent attorney recognize this `<think>` as their own scratch reasoning?"
- New feedback memory `feedback_synth_meta_state_gate.md` saved, linking back to `feedback_audit_docs_kwarg_blind_spot` (the symbol-coverage-is-not-content-coverage twin lesson on the fieldkit side) and `feedback_keep_scorer_local_until_reuse` (don't promote the regex to `fieldkit.eval` until a second vertical's corpus reuses it)

**Phase 2 artifact manifest:**

- New `src/content/artifacts/patent-strategist-bench-v0.1.yaml` — first `kind: bench` manifest in the repo (existing four manifests at `finance-chat-gguf`, `ii-medical-8b-gguf`, `saul-7b-instruct-v1-gguf`, `securityllm-gguf` are all `kind: quant`). Schema-compliant per `src/content.config.ts` (`ARTIFACT_KINDS` already includes `bench`). The manifest carries the HF repo URL, license tier (free, model CC-BY-4.0), and a back-pointer to `articles/patent-strategist-v1-baseline-on-spark/`. No `variants` array (benches aren't quantized); no `perplexity` / `spark_tokens_per_sec` block (irrelevant for datasets)

## Mac sweep guidance

This cycle is **artifact-side**, not content-side. No new article, no new signature SVG, no new prose. Mac's sweep is therefore much lighter than the last few cycles.

**Files Mac should sweep on the destination side:**

- `src/content/artifacts/patent-strategist-bench-v0.1.yaml` (new) — the first `kind: bench` manifest. Mac's `/artifacts/<kind>/` catalog rendering should pick this up via `getCollection('artifacts')` and surface it under whatever bench-catalog page Mac maintains (or scaffold one if `/artifacts/bench/` doesn't exist yet). If Mac chooses to defer the `bench` catalog page until a second bench lands, that's fine — the source manifest is forward-compatible
- `dataset-cards/patent-strategist-bench-v0.1/README.md` (modified) — second-article link added + v1.0 roadmap line de-committed. Optional sweep; the dataset card primarily lives on HF, this file is the source of truth that gets re-pushed on the next bench rev
- `src/data/project-stats.json` (regenerated; no numeric change since no article was published — bytes may differ via the `generated_at` timestamp)
- `README.md` (regenerated; article counts unchanged so likely byte-identical)

**Files Mac should NOT sweep (Spark-only):**

- `.claude/skills/claude-corpus-synth/**` — Claude Code skills are Spark-side tooling. Mac's `.claude/skills/` is independent. The patches do not propagate
- The new feedback memory under `~/.claude/projects/-home-nvidia-ai-field-notes/memory/` — auto-memory store, not in this repo

**Files NOT changed this cycle:**

- No `articles/**` additions or updates
- No `src/components/svg/**` changes
- No `fieldkit/**` changes (no PyPI release)
- No `papers/**` changes
- No `evidence/**` changes (the W3 baseline-runs evidence was shipped last cycle)
- No renames (`SYNC-RENAMES.log` unchanged)
- No new top-level pages (Mac chrome `/book/`, `/pricing/`, `/about/`, `/artifacts/<kind>/` untouched from Spark side)

**Confirmation points after sweep:**

1. `getCollection('artifacts')` should include `patent-strategist-bench-v0.1` with `kind: bench` and the `Orionfold/patent-strategist-bench-v0.1` HF repo URL
2. If Mac renders an `/artifacts/bench/` catalog page, the patent-strategist bench should appear as its first (and only) entry; the "free" license tier should surface; the back-link to `articles/patent-strategist-v1-baseline-on-spark/` should resolve
3. The HF dataset page at `https://huggingface.co/datasets/Orionfold/patent-strategist-bench-v0.1` should render the dataset card cleanly — Quick stats table, three-mode evaluation table, schema block, limitations, citation. The dataset viewer should load `data/train.jsonl` as a parquet-cached table with the 13-column schema
4. `src/data/project-stats.json` numbers remain at 40 articles / 134,203 words / 25,508 LOC (no drift; no new article was published)

## What did NOT ship

- **The W3 patent-strategist model remains SHELVED.** The s40 merged-BF16 (16.38 GB) was deleted from `/home/nvidia/data/aifn-train-lora/` this session; 748 MB of root-owned LoRA checkpoints under `runs/checkpoint-{200,400,600,626}/` survive pending a sudo cleanup. The model is not coming back on this base + corpus combination — see HANDOFF.md Phase C for the NVIDIA-base + Unsloth research path that replaces it
- **No fieldkit release.** The skill patches live under `.claude/skills/` (Spark-side tooling), NOT under `fieldkit/`. No `[Unreleased]` promotion candidates this cycle
- **No new article.** The methodology articles for both bench-publish (this cycle) and synth-skill patches (this cycle) live inside the two existing W1/W2/W3 articles already shipped (`patent-strategist-v1-baseline-on-spark` and `fine-tune-data-prep-decisions-on-spark`). A future article on the producer-subagent meta-state mechanics is queued behind the v2 corpus build per HANDOFF.md, so the article-side stays empty for this rotation

## Next cycle expectations

The patent-strategist arc resumes on a new foundation (Phase C in HANDOFF.md): NVIDIA-family base model (Llama-3.1-Nemotron-8B-Instruct is the prime candidate; Mistral-NeMo-Minitron-8B-Instruct + Nemotron-Nano-9B-v2 are alternate candidates pending llama.cpp + Unsloth compatibility checks) plus Unsloth as the training framework (NVIDIA partner; ~2× speed + ~50 % memory savings; integration with `fieldkit.train` deferred until a second model build validates the helper abstractions per `feedback_keep_scorer_local_until_reuse`). The next sync drop from this branch will either be a feasibility article under `articles/unsloth-on-spark-feasibility/` (if the Unsloth integration ships a clean smoke train on the Spark) or a smaller decision doc under `ideas/unsloth-feasibility-decision.md` (if the integration gate doesn't clear). Either is publishable.
