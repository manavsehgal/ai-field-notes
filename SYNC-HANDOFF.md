<!--
  ⚠️ STATUS: NEW — content below ships in the head commit on `main` after this release lands.
  At the next release prompt, **clear this entire file and start fresh** (do NOT append to existing sections).
  The file is *one feature/release at a time* — not a running log.
  Last reset: 2026-05-10 (prior content covered the explainer trailing-pair inline fix + Phase A/B visualization pass; both shipped through `0126de8` and have been consumed).
-->

# SYNC-HANDOFF — Visualization Quality Pass v2 (full-corpus sweep, 2026-05-10)

A full-corpus visualization quality pass across 8 published articles. Adds 6 inline companion figures to thin-coverage articles and fixes 2 mechanical text-bleed / dashed-line nits flagged from the prior pass. Pure article-prose changes; **no schema, CSS, plugin, or component edits**.

**Source range:** head of `main` (single commit on top of `0126de8`).

---

## TL;DR — what shipped

| Article | Change | Archetype |
|---|---|---|
| `articles/distill-architect-lora-from-trajectories/article.md` | Phase 2 fix: shorten 3 narrow-segment labels (`d_model=1536` → `dm=1536`, `d_ff=4096` → `ff=4096`, `d_ff=6144` → `ff=6144`) + add "3 singletons" annotation | mode-collapse-comparison (existing) |
| `articles/baseline-training-loop-on-spark/article.md` | Phase 2 fix: drop `pathLength="100"` from 2 dashed seq=2048 polylines so dasharray renders smoothly in user units | timeline (existing) |
| `articles/clawgym-on-spark/article.md` | Phase 4 add: per-persona per-assertion delta waterfall — 8 personas, 6 gain (ml-engineer +30.6 accent green) + 1 regression (data-science-researcher −9.9 red) | waterfall (new) |
| `articles/clawgym-on-spark-grpo/article.md` | Phase 4 add: 34-step training timeline — mean turns (accent solid) + task_complete% (dashed), step-29 minimum 3.69 highlighted | dual-trace timeline (new) |
| `articles/kv-cache-arithmetic-at-inference/article.md` | Phase 4 add: KV-per-token by Llama 3.1 family × precision — 8B/70B/405B at FP16/FP8 + dashed-red counterfactual at "405B if KV scaled" | paired-bar matrix (new) |
| `articles/gpu-sizing-math-for-fine-tuning/article.md` | Phase 4 add: method × model-size hardware-tier heatmap — 5 sizes × 3 methods, color-coded by hardware tier (Spark/H200/H100/SuperPOD) | heatmap (new) |
| `articles/lora-on-your-own-qa-pairs/article.md` | Phase 4 add: 3-epoch training-loss + eval-loss sparkline, eval-min at epoch 2 highlighted | sparkline (new) |
| `articles/pgvector-on-spark/article.md` | Phase 4 add: recall@10 vs p50 latency Pareto scatter — HNSW (solid green accent) vs IVFFlat (dashed) + exact ghost dot, ef=10 frontier highlighted | Pareto scatter (new) |

Plus stats refresh: `src/data/project-stats.json` regenerated (no article-count change — figure-only edits).

---

## File-level diff inventory

| File | LOC delta | Kind |
|---|---:|---|
| `articles/distill-architect-lora-from-trajectories/article.md` | +4 / −3 | label edits |
| `articles/baseline-training-loop-on-spark/article.md` | +2 / −2 | attr removal |
| `articles/clawgym-on-spark/article.md` | +69 / 0 | new figure |
| `articles/clawgym-on-spark-grpo/article.md` | +83 / 0 | new figure |
| `articles/kv-cache-arithmetic-at-inference/article.md` | +71 / 0 | new figure |
| `articles/gpu-sizing-math-for-fine-tuning/article.md` | +110 / 0 | new figure |
| `articles/lora-on-your-own-qa-pairs/article.md` | +74 / 0 | new figure |
| `articles/pgvector-on-spark/article.md` | +83 / 0 | new figure |
| `src/data/project-stats.json` | +1 / −1 | timestamp |

Total: **9 files, ~500 insertions, ~10 deletions**.

---

## Behavior contract (what destination must keep stable)

1. **No new CSS, no new components, no new dependencies.** Every figure uses the existing `.fn-diagram__*` BEM classes from `src/styles/diagrams.css`. The IntersectionObserver wiring in `src/layouts/BaseLayout.astro` already handles motion gating via `.fn-diagram--visible`. The 9 OKLCH accent tokens in `src/styles/global.css:75–83` are the palette source; this pass deploys 5 of them (`--svg-accent-green`, `--svg-accent-cyan`, `--svg-accent-red`, `--svg-accent-orange` plus `--color-primary` indigo).

2. **Every new SVG follows the existing inline-figure conventions:**
   - `<figure class="fn-diagram" aria-label="…">` wrapper
   - `<svg viewBox="0 0 900 N" role="img" aria-label="…" preserveAspectRatio="xMidYMid meet">`
   - `<g class="fn-diagram__edges">` / `__nodes` / `__labels` / `__annotations` child groups
   - `pathLength="100"` on stroke-drawn paths only (omit on `--dashed` and `--ghost` edges to preserve user-unit dasharray rendering — this is the lesson from the baseline-training-loop F1 fix in this same release)
   - **No blank lines inside any `<figure>` block** — markdown breaks out of HTML mode and renders the rest as code (memory: `feedback_fn_diagram_no_blank_lines.md`).

3. **Color rule (max 3 hues per figure, semantic only):** default stays blue/indigo; second hue only when 2+ data series share a figure (e.g., FP16 vs FP8 in kv-cache); third hue reserved for accent / warning / regression (e.g., the red regression bar in clawgym persona waterfall, the red counterfactual rect in kv-cache).

4. **Stats consumed:** `src/data/project-stats.json` updated only for `generated_at` timestamp — article count, word count, LOC unchanged from the prior `0126de8` snapshot. Home-page `ProjectStats.astro` reads the same shape.

---

## Verification steps for the destination CC

```bash
# 1. Pull the head commit on main
git pull origin main

# 2. Clear Astro cache (recommended after every content-only sync — figure SVGs are
#    inline in markdown, so the article cache key may not invalidate cleanly)
rm -rf .astro node_modules/.astro

# 3. Build
npm run build  # expect 60 pages clean, ~5 s
```

Then spot-check the 6 articles with new figures:

| Article URL | Should see |
|---|---|
| `/articles/clawgym-on-spark/` | 2 figures: existing 5-phase substrate + new persona waterfall (8 bars, +30.6 accent left, −9.9 red right) |
| `/articles/clawgym-on-spark-grpo/` | 2 figures: existing 5-node GRPO loop + new dual-trace training timeline (mean turns + task_complete% across 34 steps) |
| `/articles/kv-cache-arithmetic-at-inference/` | 2 figures: existing 5-tier scaling chart + new 3×2 family-precision matrix (FP16 blue + FP8 green bars per family) |
| `/articles/gpu-sizing-math-for-fine-tuning/` | 2 figures: existing 100B-three-ways chart + new 5×3 heatmap (3B → 100B columns × Full/LoRA/QLoRA rows) |
| `/articles/lora-on-your-own-qa-pairs/` | 2 figures: existing 4-layer stack + new 3-epoch loss sparkline (eval-min at epoch 2 accent green) |
| `/articles/pgvector-on-spark/` | 2 figures: existing pgvector layered stack + new recall-vs-latency Pareto scatter (8 dots, HNSW ef=10 frontier highlighted) |

And the 2 fixed figures:

| Article URL | Should see |
|---|---|
| `/articles/distill-architect-lora-from-trajectories/` | F3 mode-collapse: 3 narrow segments now read `dm=1536`, `ff=4096`, `ff=6144` (no overflow) with "3 singletons" annotation above |
| `/articles/baseline-training-loop-on-spark/` | F1 16-config sweep: 2 seq=2048 dashed traces now render as smooth `5 4` user-unit dashes (not broken ticks) |

---

## Conflict avoidance

- **No chrome edits.** Nav, Footer, Logo, ThemeToggle, ReaderSettings, BookmarkStar, ArticleArcNav, FieldkitCTA — all untouched.
- **No new files.** All 8 article changes + 1 stats JSON. The destination's own design-system files are not in this diff.
- **No localStorage keys added.** The `afn:*` namespace is unchanged.
- **No package.json changes.** No new dependencies.
- **Cache discipline:** `rm -rf .astro node_modules/.astro` before the first build is a safe-but-conservative habit; not strictly required for figure-only edits but harmless.

---

## Out of scope (intentionally)

- **Phase 3 archetype REPLACE.** The audit's REPLACE candidates (nim-first-inference, nemo-framework, rag-eval, autoresearch-agent-loop) all turned out to be at archetype best-fit on close inspection — the existing signature SVG components were good. Skipping was the right call.
- **Companion figures for 11 other thin articles.** Selected the 6 with the cleanest data-to-figure mappings; the remaining thin articles (mcp-second-brain, runtime-frontier-six-patches, trtllm-and-triton, bigger-generator-grounding, guardrails-on-the-retrieval-path, naive-rag, nemo-retriever-embeddings-local, nemo-framework-on-spark, autoresearchbench, test-time-distilling, one-substrate-three-apps) either lack quantitative data in prose for a second figure or already convey their thesis with a single inline figure. Future passes can revisit individually if a thesis evolves.
- **Mobile / theme deep-smoke.** Spot-check at desktop 1400×900 only (per user-confirmed scope). The motion + theme system already passes prior-pass verification; no infrastructure changed in this release.
- **`scripts/verify_svg.sh` validator.** Still aspirational per HANDOFF; the `feedback_fn_diagram_no_blank_lines.md` rule was caught only at browser render time on this pass — worth porting to a build gate in a future release.

---

## Why this works as a feature-only sync

This release is **inert at the schema and infrastructure layer**. Every change lives inside an `articles/<slug>/article.md` body or in a single auto-generated JSON file. There is no behavior contract change, no new component to register, no new directive type, no new localStorage key. A destination that builds clean before pulling will build clean after pulling.

---

## Addendum 2026-05-10 — v2.1 figure-overflow follow-ups (commit `f341514`)

Three small dark-mode-surfaced figure issues caught after v2 landed. Same scope rules as v2: pure article-prose edits, no CSS / component / schema changes.

| Article | Symptom | Fix |
|---|---|---|
| `articles/pass-at-k-after-the-seventh-patch/article.md` | (a) orphan italic primer paragraph read like a figcaption with no figure; (b) seven-patch through-line crossed the semi-transparent accent container holding patch #7 | (a) primer wrapped as `:::define[ESamp]:::` block, joining the existing Pass@k / Test-time-scaling define stack; (b) through-line shortened from `M 60 140 L 860 140` to `M 60 140 L 680 140` so it terminates at the dashed test-surface divider instead of bleeding through the accent gradient |
| `articles/clawgym-on-spark/article.md` | (a) two waterfall annotations centered at SVG `x=100` / `x=800` extended ~95 / ~88 vb-units past the `0..900` viewBox; (b) extra dark inner card behind the chart added a redundant dark-on-dark plate in dark mode | (a) re-anchored: `text-anchor="start" x="60"` for the two left annotations, `text-anchor="end" x="840"` for the right one; (b) removed the `<rect ... fill="var(--svg-card)" opacity="0.4"/>` plate at `x=40 y=40 width=820` |
| `articles/test-time-distilling-for-exploration/article.md` | most labels in the runtime diagram bled outside their 160-vb-wide corner nodes (longest label was 243 vb-units) and two annotations sat inside `tLLM` or overlapped `BotR` | corner nodes widened 160 → 240 (`x=60→40` left, `x=700→620` right); labels re-anchored `x=75→55` and `x=715→635`; four diagonal edges shortened to start/end at the new node edges; three annotations relocated — top two moved to the `y=176` inter-row gap, bottom long annotation rewritten as `post-filter intervention · reweight after top-k / top-p / min-p` and parked at `y=420 x=450` (figure-bottom sub-line) |

**Why these slipped through v2's smoke pass:** all three render correctly in light mode (where the accent gradient is opaque enough to mask the through-line, and the inner card adds visible separation). They surface only in dark mode, which the v2 audit didn't deep-smoke.

### File-level diff inventory

| File | LOC delta | Kind |
|---|---:|---|
| `articles/pass-at-k-after-the-seventh-patch/article.md` | +4 / −2 | define block + path d= |
| `articles/clawgym-on-spark/article.md` | +3 / −4 | text-anchor + rect removal |
| `articles/test-time-distilling-for-exploration/article.md` | +23 / −23 | rect/x/edge re-coordinate |
| `src/data/project-stats.json` | +2 / −2 | regen |

Total: **+32 / −31** across 4 files.

### Test plan (delta from v2)

```bash
cd /home/nvidia/ai-field-notes
rm -rf .astro node_modules/.astro
npm run build  # expect 60 pages clean
npm run dev    # http://localhost:4321/
```

Then dark-mode spot-check:

| Article URL | Should see |
|---|---|
| `/articles/pass-at-k-after-the-seventh-patch/` | (1) ESamp now appears as a `:::define` aside in the term index alongside Pass@k and Test-time scaling; (2) timeline through-line ends at the dashed divider, accent #7 box is unmarked by the line |
| `/articles/clawgym-on-spark/` | (1) waterfall: both side annotations sit fully inside the figure box at desktop widths; (2) no dark inner card behind the bars |
| `/articles/test-time-distilling-for-exploration/` | every node label fits inside its node, no overlap with the diagonal edge lines or other nodes; bottom sub-line "post-filter intervention · reweight after top-k / top-p / min-p" reads cleanly under all rows |

### Conflict avoidance (unchanged from v2)

Same as the parent v2 release: no chrome, no new files, no `localStorage` keys, no `package.json`. Pure article-body + regenerated stats JSON.
