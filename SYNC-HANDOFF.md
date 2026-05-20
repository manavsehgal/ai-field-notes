<!--
  ⚠️ STATUS: NEW — Mac sweep pending.
  This file is one feature/release at a time, not a running log.
  At the next release prompt, **clear this entire file and start fresh** (do NOT append to existing sections).
  Last reset: 2026-05-19 (Unsloth-on-Spark feasibility article + signature SVG).

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
release_slug: 2026-05-19-unsloth-on-spark-feasibility-article
status: NEW
source_range: 538d3a9..HEAD                # main was at 538d3a9 before this cycle's commit
articles_added:
  - slug: unsloth-on-spark-feasibility     # series: Machine that Builds Machines; stage: fine-tuning; product: Foundation; signature: UnslothEnvelopeFlat; one inline fn-diagram (5-stage flow pipeline, accent on save_gguf); 11 explainers (4 define / 1 why / 3 pitfall / 1 math / 1 deeper / 1 hardware); ~2900 words; reproducibility appendix with container + package pins + recipe; covers C2 Q1+Q1b+Q2+Q3+Q5+Q6 gates green on `nvcr.io/nvidia/pytorch:25.11-py3` + Llama-3.1-Nemotron-Nano-8B-v1 + Unsloth 2026.5.5; headline metric is 16.94 GB peak GPU alloc across every GPU-resident stage (~5 GB under the s40 baseline of ~22 GB)
articles_updated: []
signatures_added:
  - name: UnslothEnvelopeFlat              # src/components/svg/UnslothEnvelopeFlat.astro; 300x200 viewBox; four equal-height bars (load / train / save_gguf / reload) all topped at 16.94 GB with a horizontal envelope line connecting tops + a dashed s40-baseline reference line at 22 GB; accent on save_gguf bar; uses the same data-svg-animate + svg-reveal-d* + bar-grow patterns as the existing CorpusContaminationLayers signature
signatures_updated: []
artifacts_added: []                         # smoke-train adapter + GGUFs live on /home/nvidia/data/ — not promoted to a permanent HF artifact this cycle (that comes with the v2 production train)
artifacts_updated: []
fieldkit_modules_changed: []                # gate 4 deliberately deferred — no fieldkit.training.unsloth helper until a 2nd vertical reuses the recipe
fieldkit_release: null
papers_added: []
papers_classify_count: 0
renames_to_replay: []
removes: []
new_top_level_pages: []
breaking_changes: []
destination_overrides_to_preserve:
  - "Article count moves 40 → 41 published (4 upcoming unchanged). Mac chrome stays in sync via the standard article-add path. The 2-article gap to the Mac-authored landing pages (`ai-transformation`, `solo-builder-case-study`) persists; Mac owns those."
hf_repos_added: []
civitai_artifacts_added: []
post_rotation_commits: []
---

## What shipped this cycle

The **Unsloth-on-Spark feasibility article** at `articles/unsloth-on-spark-feasibility/` — the third instalment in the patent-strategist arc on the **Machine that Builds Machines** track. It walks the six gates that had to clear before the v2 production train commits to Unsloth + Llama-3.1-Nemotron-Nano-8B-v1 + llama.cpp instead of the s40 TRL+PEFT stack. All six cleared in one container in roughly fifty minutes of wall. Headline finding: **peak GPU allocation flat at 16.94 GB across load / train / save_gguf / reload — same envelope as the BF16 base, ~5 GB under the s40 baseline of ~22 GB**.

**Article details:**

- Slug: `unsloth-on-spark-feasibility`
- Series: Machine that Builds Machines (third post in the patent-strategist sub-arc, after `patent-strategist-v1-baseline-on-spark` and `fine-tune-data-prep-decisions-on-spark`)
- Stage: `fine-tuning`; product: `Foundation` (Unsloth + Nemotron + llama.cpp doesn't map to the listed NVIDIA-product slots; same convention as W1 / W2)
- Signature SVG: `UnslothEnvelopeFlat` (300×200, four equal-height bars + envelope line, accent on save_gguf, dashed s40 reference at 22 GB)
- One inline fn-diagram (flow pipeline archetype, 5 stages, accent on save_gguf, recurring 16.94-GB annotations as thesis punch)
- 11 explainers: 4 define (LoRA r=16 attention-only / Unified memory on GB10 / Gradient checkpointing — Unsloth's "unsloth" flavor / GGUF), 1 why (Stack coherence beats raw speed on a personal rig), 3 pitfall (`bitsandbytes` self-check / `save_pretrained_gguf` writes to `<out>_gguf/` / `llama-cli -no-cnv` no longer honored), 1 math (per-step economics vs s40), 1 deeper (Unsloth release notes + W1/W2 cross-links + bench + base-model card), 1 hardware (same recipe scaled to H100/H200)
- Reproducibility appendix: container + package pins (all canonical s40 pins survived `--no-deps`), one-liners (`pip install --no-deps unsloth unsloth_zoo bitsandbytes` + `cd /tmp` trap dodge), recipe snippet, verification snippet, measured gate-timings table

**Article structure (8-section essay form per `references/article-structure.md`):**

1. Opening hook — train-time peak vs base-load peak on a single GPU; the recurring 16.94 GB is the thesis
2. Why this matters — stack coherence > raw speed on a personal rig; headroom for sibling workloads
3. Architectural context — where Unsloth sits (between transformers and the GPU); the kernel layer; `save_pretrained_gguf()` collapsing the publish surface; **inline fn-diagram** anchored here
4. The journey — gate-by-gate walk (gate 1 install / gate 2 load / gate 3 wrap+train / gate 4 deferred / gate 5 GGUF / gate 6 llama.cpp)
5. Verification — what success feels like on a Spark (5 GB headroom for a critic checkpoint)
6. Tradeoffs and surprises — Triton JIT gcc-specs trap, `_gguf` suffix output path, silent `-no-cnv` failure
7. What this unlocks — v2 production train / Orionfold NVIDIA-stack narrative / critic-model arc headroom
8. Closing — same Spark, same 128 GB, more room for what comes next

## Mac sweep guidance

This cycle is **content-side** — one new article + one new signature SVG + the standard stats + README refresh. Mac's sweep is the regular article-add shape (no schema changes, no path renames, no artifact catalogs touched).

**Files touched on source:**

```
NEW    articles/unsloth-on-spark-feasibility/article.md
NEW    articles/unsloth-on-spark-feasibility/transcript.md
NEW    articles/unsloth-on-spark-feasibility/{assets,screenshots}/   (empty placeholder dirs from new_article.sh)
NEW    src/components/svg/UnslothEnvelopeFlat.astro
MOD    src/data/project-stats.json     (article count 40 → 41; total words +Δ; stages.fine-tuning +1)
MOD    README.md                       (regenerated from project-stats.json + per-article frontmatter)
```

**What Mac needs to verify after pulling:**

1. `npm run build` validates the new article's frontmatter (it does — `series: Machine that Builds Machines` is in the enum, `signature: UnslothEnvelopeFlat` points at the new component, `fn-diagram` invariants pass `scripts/verify_svg.sh`)
2. The signature renders correctly on the home page and the `/stage/fine-tuning/` page — UnslothEnvelopeFlat uses the same `data-svg-animate` + `svg-reveal-d*` + `bar-grow` patterns as the existing `CorpusContaminationLayers` signature; should drop into the same animation pipeline cleanly
3. Series filter `/series/machine-that-builds-machines/` includes the new article alongside the W1+W2 predecessors
4. Article counts on the home "At a glance" infographic and across the destination chrome pick up the new total (41 published)

**No artifact-manifest changes this cycle.** The smoke-train adapter and GGUFs (`/home/nvidia/data/aifn-train-lora/unsloth-smoke-2026-05-19/`) are local research artifacts that will be replaced by a production-grade pair (model + bench co-publish) when the v2 production train ships. The bench manifest at `src/content/artifacts/patent-strategist-bench-v0.1.yaml` already carries the back-pointer to its companion methodology articles; once a v2 model card lands, a paired `kind: lora` (or `kind: quant` depending on the publish shape) manifest will reference both this article and the bench.

**No fieldkit changes this cycle.** Gate 4 (a `fieldkit.training.unsloth` helper) was deferred on purpose per `feedback_keep_scorer_local_until_reuse` — first use stays in `scripts/`; the second vertical that reaches for the same Unsloth recipe is the trigger to lift it into the package.

## Open questions for Mac

None. Standard article-add cycle, no schema changes, no chrome decisions waiting on Mac.

## Next cycle expectations

The patent-strategist arc next produces a **v2 model + paired-drop**. The patched `claude-corpus-synth` skill (from the prior cycle) generates a fresh five-thousand-row corpus with the meta-state gates in place; the same Unsloth recipe walked in this article scales to real `max_seq_length` and real LoRA rank; `save_pretrained_gguf()` ships the GGUF in one call; the existing bench at `Orionfold/patent-strategist-bench-v0.1` gets a paired model card with the concrete base finally named. That release will land as a `hf_repos_added` + `artifacts_added` cycle here (likely `kind: quant` or `kind: lora`), plus a model-card writeup either as a polish of this article or as a new release-receipt article.
