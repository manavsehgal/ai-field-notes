<!--
  ⚠️ STATUS: NEW — Mac sweep pending.
  This file is one feature/release at a time, not a running log.
  At the next release prompt, **clear this entire file and start fresh** (do NOT append to existing sections).
  Last reset: 2026-05-16 (medical-vertical / II-Medical-8B-GGUF live HF push).

  Prior Mac sweep receipts (preserved here since SYNC-HANDOFF is per-release-not-running-log; Mac will sweep this 2026-05-16 medical-vertical cycle next):
  - 2026-05-15 fieldkit v0.4.2 cycle: swept at destination commit manavsehgal/ainative-business.github.io@495196d (Mac PR #7, merged 2026-05-16 at d332d28).
  - 2026-05-15 cyber-vertical cycle: swept at destination commit manavsehgal/ainative-business.github.io@135bcad (Mac PR #6 merged 2026-05-15).
  - 2026-05-14 v0.4.1 cycle: swept at destination commit manavsehgal/ainative-business.github.io@e1b16de (Mac PR #5 merged 2026-05-14).
  - 2026-05-14 v0.4.0 cycle: swept at destination commit manavsehgal/ainative-business.github.io@f7ea7aa (Mac PR #4 against this repo — conflicted on rotation; safe to close, receipt captured here).
  - 2026-05-14 Orionfold/finance-chat-GGUF cycle: swept at destination commit manavsehgal/ainative-business.github.io@85f9307 (Mac PR #3, merged).
  - 2026-05-12 Autoresearch→MTBM rename: swept at destination commit manavsehgal/ainative-business.github.io@71293af (Mac PR #2, merged).
-->
---
release_slug: 2026-05-16-medical-vertical-II-Medical-8B
status: NEW
source_range: f23efb3..HEAD
articles_added:
  - slug: becoming-a-medical-curator-on-spark    # status: upcoming placeholder; full draft promotion pending (see Open question below)
articles_updated: []
artifacts_added:
  - manifest: src/content/artifacts/ii-medical-8b-gguf.yaml
    hf_repo: Orionfold/II-Medical-8B-GGUF
    recommended_variant: Q5_K_M
    license: apache-2.0
    chat_format: chatml
    vertical_eval_name: "MedMCQA (n=50, mcq_letter)"
artifacts_updated: []
fieldkit_modules_changed: []                     # zero fieldkit source-code changes this cycle — second vertical in a row to land via configuration only
papers_added: []
papers_classify_count: 0
renames_to_replay: []
removes: []
new_top_level_pages: []
breaking_changes: []
destination_overrides_to_preserve: []
hf_repos_added:
  - Orionfold/II-Medical-8B-GGUF                 # 5 GGUFs + README + .gitattributes; live 2026-05-16 05:15 UTC after 2h32m push
civitai_artifacts_added: []
fieldkit_release: null                           # no fieldkit cut this cycle; current PyPI stays at 0.4.2
post_rotation_commits: []                        # rotation happens at end-of-cycle; any post-rotation commits captured in next sweep
---

## Headline

Vertical 4 (**medical**) ships live to HuggingFace at <https://huggingface.co/Orionfold/II-Medical-8B-GGUF>. Five GGUF variants of Intelligent-Internet's II-Medical-8B (Qwen3-8B base, SFT + DAPO reasoning recipe targeting clinical Q&A), measured end-to-end on a DGX Spark with the same four-axis card the prior three Orionfold verticals carry. **Q5_K_M is the narrative-recommended variant** — perplexity essentially equal to F16 (16.24 vs 16.27), MedMCQA accuracy slightly above F16 (52% vs 48% at n=50), 2.3× throughput (36.4 vs 15.9 tok/s), 5.45 GB on disk.

This cycle is the **second vertical in a row to ship with zero new code in `fieldkit` itself** — the publishing surface generalized in v0.4.0–v0.4.1 absorbed the medical pick as a configuration change. One real footgun got patched in the preflight harness (a missing `chatml` branch in `_detect_prompt_format` that would have silently dropped `<|im_start|>` wrapping on any future ChatML model), but it lives in `scripts/g3_preflight_bench.py` until the `mcq_letter` scorer hits its third reuse and triggers Phase 8.5 promotion of the bench layer.

## What Mac CC sweeps

Spark-authoritative files (per `[[reference_sync_contract]]`). No destination-side rewrites, no schema changes, no renames.

- **`src/content/artifacts/ii-medical-8b-gguf.yaml`** — NEW artifact manifest (auto-emitted by `publish_quant(dry_run=True)`). Carries `recommended_variant: Q5_K_M` (the v0.4.2 field; this is its first cycle with a fresh push that uses it), `license.model: apache-2.0`, `chat_format: chatml`, `vertical_eval_name: "MedMCQA (n=50, mcq_letter)"`, 5 variants with full Spark-tested metrics. Destination catalog can render the "Sweet spot" badge directly from `recommended_variant:` — no hand-pin required (contrast with cyber's PR #6, where Mac had to hand-pin `Q4_K_M` because the manifest was on pre-v0.4.2 schema).
- **`articles/becoming-a-medical-curator-on-spark/article.md`** — NEW article in `status: upcoming` placeholder shape. Frontmatter mirrors the sibling finance/legal/cyber-curator articles exactly (`product: llama.cpp`, `stage: deployment`, `series: Machine that Builds Machines`, `fieldkit_modules: [quant, publish, eval, lineage]`, `also_stages: [observability]`, `hf_url`). Body is a 3-paragraph stub (intent + footgun-of-the-cycle + pairs-with-gguf-publisher pointer). **Full draft promotion is pending** — see "Open question" below.
- **`articles/becoming-a-gguf-publisher-on-spark/evidence/lineage-II-Medical-8B/results.tsv`** — NEW evidence directory under the gguf-publisher arc article (same shape the cyber + legal + finance verticals use). 5 lineage rows, exp_ids 001–005, status=keep.
- **`scripts/medmcqa_merge.py`** — NEW; samples N=50 from `openlifescienceai/medmcqa` validation split into the cyber/legal `{id,text,answer,task}` JSONL shape. The `mcq_letter` scorer's second reuse.
- **`scripts/g3_preflight_bench.py`** — EXTENDED with the medmcqa whitelist, `MEDMCQA_JSONL` env, **and** a new `chatml` branch in `_detect_prompt_format`/`_format_prompt` (the silent-failure fix for ChatML templates).
- **`scripts/g3_measure_variants.py`** — EXTENDED with medmcqa whitelist, `_wrap_chatml` prompt wrapper, MEDMCQA env, and medical defaults (domain / baseline / bench dataset maps).
- **`scripts/g3_build_first_quant.sh`** — EXTENDED with a new per-model case for `Intelligent-Internet/II-Medical-8B` (license, chat-format, vertical, article slug, MEDMCQA env threading).
- **`README.md`** + **`src/data/project-stats.json`** — auto-refreshed via `nvidia-learn-stats` + `tech-writer refresh_readme.py` after the upcoming-placeholder commit. Article count now 37 (5 upcoming including this placeholder).

### Auto-refreshed (no Mac-side action needed)

The stats infographic + README article index reflect the new article entry (with the 🔜 placeholder marker). Top-level numbers shifted only marginally — the upcoming placeholder counts toward `stages_upcoming` but not toward word / LOC totals until promoted.

## What Mac CC does NOT need to do

- **No fieldkit release.** PyPI stays at 0.4.2. No new module, no kwarg drift, no schema change. The v0.4.2 cut absorbed everything this push needed (`ModelCard.llama_cpp_example_prompt` neutral default + `ArtifactManifest.recommended_variant` flow-through).
- **No rename replays.** `SYNC-RENAMES.log` unchanged.
- **No new top-level pages.** The new article folder is under the existing `articles/` content collection; no new section, no new stage page (existing `deployment` + `observability` filter pages already render it).
- **No HF README patches.** The card rendered correctly from the v0.4.2 codepath on first push — no in-place HF edits needed (contrast with the finance + legal + cyber cycles which all required post-push fixups).
- **No skill IA mirroring.** `hf-publisher`, `hf-model-scout`, `tech-writer`, `fieldkit-curator` all live in `~/.claude/skills/` (Spark CC user config), not in the source repo.

## Verification (Spark-side)

- **HF live:** <https://huggingface.co/Orionfold/II-Medical-8B-GGUF> — HTTP 200; 7 files (5 GGUFs + README + .gitattributes) listed via `HfApi.list_repo_files`.
- **Push:** `hf_push_resilient.py` via `upload_large_folder` with `num_workers=1` per `[[feedback_hf_upload_resilient_api]]`. Started 2026-05-16 02:43, completed 05:15:19 — wall clock **2h 32m 33s** for ~40 GB at ~3.6 MB/s effective (5.2 MB/s raw rate minus per-file LFS commit handshakes). Zero retries triggered, zero httpx errors observed — second consecutive Spark push to ship without crashing (after the cyber cycle).
- **Preflight bench:** F16 GGUF preflight scored 2/5 on MedMCQA at `LLAMA_CLI_NPREDICT=256` (`<think>`-block truncation; the reasoning-recipe trap that drove the new `[[feedback_reasoning_model_npredict]]` memory). Bumped to 1024 for the full measure sweep; F16 scored 0.48 on n=50.
- **Variant table (n=250 perplexity, n=50 MedMCQA, llama-bench tg/pp):**

  | Variant | ppl | tg tok/s | pp tok/s | MedMCQA | Size |
  |---|---|---|---|---|---|
  | F16 | 16.27 | 15.94 | 2262.2 | 0.48 | 15.3 GB |
  | Q4_K_M | 16.55 | 43.57 | 2773.2 | 0.42 | 4.68 GB |
  | **Q5_K_M** ⭐ | **16.24** | **36.36** | 2579.5 | **0.52** | **5.45 GB** |
  | Q6_K | 16.01 | 32.80 | 2332.2 | 0.46 | 6.26 GB |
  | Q8_0 | 16.30 | 28.42 | 2523.3 | 0.48 | 8.11 GB |

- **`scripts/verify_article.sh`** — not run yet (article is still `status: upcoming`; the publish-gate checks apply after promotion).
- **`scripts/verify_stage.sh`** — 5/5 PASSED at push time (with `APACHE_VERIFIED=1` for the upstream license check).

## Release-commit chain (this cycle)

- **`713a1d0`** — `medical: vertical 4 infra — Intelligent-Internet/II-Medical-8B (Q5_K_M recommended)` (6 files: 3 script extensions + new `scripts/medmcqa_merge.py` + auto-emitted manifest + new lineage directory).
- **`4392ab6`** — `Upcoming: becoming-a-medical-curator-on-spark (vertical 4 placeholder)` (3 files: new article placeholder + stats refresh + README refresh).
- *(SYNC-HANDOFF rotation commit will land at the end of this cycle; captured in next sweep's `post_rotation_commits` if any post-rotation work lands before Mac sweeps.)*

Two commits this cycle (excluding the rotation itself + the v0.4.2 sweep-receipt merge `d332d28` which closed the prior cycle).

## Open question — answer to PR #7

**Decision: Option 2 — destination-authoritative catalog-footer override.**

Mac CC raised a recurring drift signal in PR #7 (v0.4.2 sweep-receipt): the cyber + gguf-publisher + legal articles get overwritten on every sync diff because the **destination** renders a trailing "↗ catalog: /artifacts/quants/\<slug\>/" footer block that the **source** doesn't carry. Mac restored those post-sync this cycle but flagged it would re-trigger every release.

Two options were on the table:
1. **Source-authoritative**: push the catalog-footer convention down to Spark; articles get a `/artifacts/quants/<slug>/` block when an HF artifact exists. Means tech-writer skill grows a footer step.
2. **Destination-authoritative**: add a narrow `articles/**` rule to `mirrors/destination-overrides.md` scoped to "trailing catalog footer when matching artifact exists." Mac owns the footer; Spark never touches that block; sync diff stops flagging it.

**Decision: Option 2.** Reasoning:
- The footer is **rendered chrome**, not editorial content — it points at destination-side catalog URLs (`/artifacts/quants/<slug>/`) that Mac already owns per the sync-contract chrome boundary table.
- Coupling Spark's tech-writer pipeline to destination URL shape would push site-layout concerns into the article-authoring path, which the sync-contract is explicitly designed to keep apart.
- The override is narrow (one trailing block, gated on artifact existence) so it doesn't expand `articles/**` into a contested area.
- Zero ongoing cost at source — Spark CC never has to remember to emit the footer or update it when destination URL conventions evolve.

**Action for Mac CC after consuming this:** add the override rule + scope language to `mirrors/destination-overrides.md`, then sweep-receipt as usual. No source-side change required for this decision. (`mirrors/destination-overrides.md` is Mac-authored per `[[reference_destination_overrides_mirror.md]]`.)

## What Mac CC should look for after sweep

- The new artifact manifest at `src/content/artifacts/ii-medical-8b-gguf.yaml` arrives with `recommended_variant: Q5_K_M` populated from source — the destination catalog can render the "Sweet spot" badge directly without hand-pinning (this is the first push to fully exercise the v0.4.2 manifest field).
- The new article `becoming-a-medical-curator-on-spark` arrives with `status: upcoming` — destination should show it as a placeholder on the `/stage/deployment/` and `/stage/observability/` filter pages with the muted-card + "Upcoming" badge. Excluded from the home index by default.
- After the catalog-footer override rule lands on Mac side, the next cyber + gguf-publisher + legal sync diff should run clean (no article overwrites flagged).
- HF catalog set grows from 3 → 4: `finance-chat-GGUF`, `Saul-7B-Instruct-v1-GGUF`, `SecurityLLM-GGUF`, **`II-Medical-8B-GGUF` (new)**.
