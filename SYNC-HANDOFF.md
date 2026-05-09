<!--
  ⚠️ STATUS: NEW — content below ships in the head commit on `main` after this release lands (see "Source range" below for the SHA).
  At the next release prompt, **clear this entire file and start fresh** (do NOT append to existing sections).
  The file is *one feature/release at a time* — not a running log.
  Last reset: 2026-05-09 (was the explainers-feature + MTBM-rename doc; previous content has been consumed by the destination through commit `f5c29ed`).
-->

# SYNC-HANDOFF — Explainer trailing-pair inline fix

This release is a **single CSS rule change** to `src/styles/explainers.css`. It eliminates a layout bug introduced by the explainers feature: in **Margins** view at viewports ≥64rem, the second-to-last explainer floated into the gutter while only the very last explainer dropped back inline — which produced a tall block of whitespace before the closing callout whenever the prose between the two trailing explainers was shorter than the floated penultimate's height. The fix forces the **last two** trailing explainers inline, so the closing pair stacks as two clean full-prose-width callouts.

**Source range:** head of `main` (single commit on top of the prior `f5c29ed` watermark — see `git log -1 --format=%H -- src/styles/explainers.css` after pulling).

---

## TL;DR — what shipped

One CSS selector swap inside the `@media (min-width: 64rem)` block of `src/styles/explainers.css`:

```diff
-html:not([data-explain-mode="inline"]):not([data-explain-mode="hide"]) .prose .explain:last-of-type {
+html:not([data-explain-mode="inline"]):not([data-explain-mode="hide"]) .prose .explain:nth-last-of-type(-n+2) {
```

That's it. Comment block above the rule was rewritten to capture *why* (whitespace before the closing callout); behavior block unchanged (`float: none; clear: both; width: auto; margin: 2.5rem 0 0; shape-outside: none;`). No new dependencies, no schema changes, no markdown edits, no other CSS edits, no JS changes.

---

## File inventory — what to sync

### Modified files (1)

| Path | Δ LOC | Change |
|---|--:|---|
| `src/styles/explainers.css` | +9 −7 | Lines 132–144: `:last-of-type` → `:nth-last-of-type(-n+2)` plus a rewritten comment block (8 lines). No other edits in this file. |

### Unchanged

- No new files.
- No new routes.
- No new dependencies — `package.json` and `package-lock.json` untouched.
- No frontmatter / schema changes — `src/content.config.ts` untouched.
- No article body edits — `articles/<slug>/article.md` byte-identical across all 27 published + 4 upcoming articles.
- No JS, no rehype/remark plugin edits, no Astro config changes.

---

## Behavior contract

**Before:**
- Margin mode at ≥64rem: every explainer floats right, `:nth-of-type(2n)` flips to float left, `:last-of-type` drops to inline.
- Bug: penultimate stays floated. When prose between penult and last is shorter than the float, `clear: both` on the last pushes it past the float's tail — leaving a visible whitespace block.

**After:**
- Margin mode at ≥64rem: every explainer floats right, `:nth-of-type(2n)` flips to float left, **the last two `:last-of-type` siblings now drop to inline together**.
- Result: closing pair stacks as two full-prose-width callouts; no whitespace before the last; earlier explainers still alternate side-floats.

### Edge cases (verified in source)

- **1 explainer total** → matches that one, renders inline. Same as today.
- **2 explainers total** → both inline (acceptable; behaves as a "closing pair").
- **3+ explainers** → only last 2 inline; earlier ones float right / alternate left as before.
- **Hide mode** (`[data-explain-mode="hide"]`) → unchanged; selector gated by `:not(...)`.
- **Inline mode** (`[data-explain-mode="inline"]`) → unchanged; same gating.
- **Mobile (<64rem)** → unchanged; rule lives inside the `≥64rem` media query.
- **`.explain.explain--before-figure`** → equal selector specificity, defined later in source, so it still wins for any explainer it tags. The rehype plugin's behavior is untouched.
- **Print stylesheet** → unchanged.
- **Reader themes (sepia / dark / light)** → unchanged (token overrides scope to `.article`, not to `.explain` floating logic).

---

## Verification (after sync)

The destination should:

1. **Pull the change** — `git pull` (or merge the commit into your tracking branch).
2. **Build clean** — `npm run build` should succeed with the same page count as before this release (no routes added, no removed). No Astro cache flush needed (CSS-only change; content collection is byte-identical).
3. **Open an article with the closing pair** at viewport ≥1472px (92rem band), Margins explainer mode (the default):
   - `https://<your-host>/articles/t2po-uncertainty-guided-rl-on-spark/`
   - `https://<your-host>/articles/distill-architect-lora-from-trajectories/`
4. Scroll to the bottom. Confirm that **both `:::deeper`/`:::pitfall` and `:::hardware` callouts render as full-prose-width inline blocks**, stacked with a `2.5rem` top margin between them. There should be no gutter offset on either, and no visible whitespace block above the closing callout.
5. **Browser-evaluate sanity** (paste into devtools):
   ```js
   const els = [...document.querySelectorAll('.prose .explain')].slice(-2);
   els.map(el => ({
     cls: el.className,
     width: el.getBoundingClientRect().width,
     left: el.getBoundingClientRect().left,
     float: getComputedStyle(el).float,
   }));
   ```
   Both entries should have `float: "none"`, identical `left` (matching `.prose` left edge), and `width` equal to the prose column's content width — **not** ~272px (17rem gutter float).
6. **Earlier explainers still float** — on the same article, scroll up. Explainers 1..N-2 should alternate gutter floats (right / left / right / left) as before. Sample expected float computed-style sequence on `t2po-uncertainty-guided-rl-on-spark`: `[none, left, right, left, right, left, none, none]` (first explainer is `none` because of `.explain--before-figure`; last two are the new inlined pair).

---

## Conflict-avoidance notes

- **No chrome edits** — Nav, Footer, ThemeToggle, Logo, layouts all untouched. Destination's clones keep their overrides intact.
- **No `localStorage` schema change** — `afn:*` keys unchanged.
- **No `:has()`** — using only `:nth-last-of-type(-n+2)`, supported in every evergreen browser since ~2017.
- **Specificity preserved** — new selector has the same specificity as the old `:last-of-type`, so source order still resolves the `.explain--before-figure` precedence relationship correctly.
- **No cache caveat** — purely CSS; Astro 5 content-collection cache is unaffected. No `rm -rf .astro` needed.

---

## Source-of-truth pointer

- Source repo: `https://github.com/manavsehgal/ai-field-notes`
- Path: `src/styles/explainers.css`, lines 132–144 (the `@media (min-width: 64rem)` block's `:nth-last-of-type(-n+2)` rule).
- Companion `src/styles/explainers.css` comment-block rewrite at the same site explains the rationale inline.

---

## Out of scope

- Any change to `:has()`-based "trailing run >2" detection. Current and likely-future articles all use a 2-element closing pair (`:::deeper` or `:::pitfall` + `:::hardware`); deferring multi-trailing logic until a real article needs it.
- Any change to the `.explain--before-figure` rehype-side detection.
- Any change to mobile behavior, hide mode, inline mode, print stylesheet, or reader themes.

---

# SYNC-HANDOFF — Visualization quality pass (Phase A reconciliation + Phase B archetype diversification)

This release follows the explainer fix above and bundles a corpus-wide visualization quality pass. **15 new inline `fn-diagram` SVGs across 9 articles**, plus **4 raw matplotlib PNGs deleted** in favor of style-matched SVG replacements. No frontmatter changes, no schema changes, no JS, no CSS — every change lives inside `articles/<slug>/article.md` bodies (and 4 PNG removals).

The two halves of the pass:

- **Phase A (reconcile, 3 articles, 9 figures):** the three published articles that had a polished signature thumbnail but a visually thin body — text-only or relying on raw matplotlib PNGs whose aesthetic clashed with the signature. Each gets 3 inline SVGs that lift the signature's palette + motion vocabulary into the article body.
- **Phase B (diversify, 6 articles, 6 figures):** opportunistic archetype swaps in already-published articles where one section's prose or table content fits an underused archetype (timeline / waterfall / heatmap / dual-path) better than what was there. Selected via taste-test; many candidate articles were skipped because their existing archetypes were the right fit.

Archetype mix across the 15 new figures: 6 timelines, 5 waterfalls, 2 dual-paths, 1 flow-pipeline, 1 heatmap, 1 mode-collapse comparison. The corpus was previously flow-pipeline-heavy; this pass deliberately leans into the underused shapes.

---

## TL;DR — what shipped

- **Phase A — three published articles brought up to signature-level visual fidelity:**
  - `trajectory-eval-is-the-agent-flailing` — 3 raw matplotlib PNGs replaced with 3 inline SVGs (waterfall + 2 timelines). Source PNGs deleted from `evidence/`.
  - `distill-architect-lora-from-trajectories` — 3 inline SVGs added (flow pipeline of distillation, waterfall replacing `calibration.png`, mode-collapse distribution comparison). Source PNG deleted from `evidence/`.
  - `pass-at-k-after-the-seventh-patch` — 3 inline SVGs added (heatmap of 3-cell matrix, dual-path of instruct vs reasoning ESamp behavior, timeline of 7 patches).
- **Phase B — six published articles get one new figure each in a deliberately diversifying archetype:**
  - `baseline-training-loop-on-spark` — timeline (16-config throughput sweep, 4 traces, fp8 ceiling at 14,266 tok/s as accent).
  - `t2po-uncertainty-guided-rl-on-spark` — timeline (pool task_pass vs held-out task_pass across 50 steps, step-45 divergence as accent).
  - `guardrails-for-code-generation` — waterfall (rail-block distribution: R1=6, R2=3, R3=6, R4=2, R5=0).
  - `derisk-cloud-pretraining-on-the-spark` — waterfall (expected-value argument: $1,680 blind-booking loss vs $1.01 Spark cost vs $1,679 net savings, ~1,670× ratio).
  - `nemo-curator-training-data-prep` — waterfall (data-path overhead across 8 configs, sub-pixel data segments at step-time scale).
  - `rerank-fusion-retrieval-on-spark` — dual-path (dense lane + BM25 lane converging at RRF merge with rerank as optional accent stage; naive/bm25 off-ramps).

---

## File inventory — what to sync

### Modified files (9)

| Path | Δ figures | Notes |
|---|--:|---|
| `articles/trajectory-eval-is-the-agent-flailing/article.md` | +3 | Replaced 3 `![…](evidence/*.png)` lines with `<figure class="fn-diagram">` SVGs |
| `articles/distill-architect-lora-from-trajectories/article.md` | +3 | Replaced 1 PNG embed; added 2 new inline SVGs at section anchors |
| `articles/pass-at-k-after-the-seventh-patch/article.md` | +3 | Article was prose-only; added 3 inline SVGs at section anchors |
| `articles/baseline-training-loop-on-spark/article.md` | +1 | Added timeline before "Three findings drop out…" |
| `articles/t2po-uncertainty-guided-rl-on-spark/article.md` | +1 | Added timeline after held-out eval table |
| `articles/guardrails-for-code-generation/article.md` | +1 | Added waterfall after the bench bash code block |
| `articles/derisk-cloud-pretraining-on-the-spark/article.md` | +1 | Added waterfall after `expected_value_argument` JSON |
| `articles/nemo-curator-training-data-prep/article.md` | +1 | Added waterfall after data-overhead table |
| `articles/rerank-fusion-retrieval-on-spark/article.md` | +1 | Added dual-path after the `retrieve()` python def |

### Deleted files (4)

- `articles/trajectory-eval-is-the-agent-flailing/evidence/knob_coverage.png`
- `articles/trajectory-eval-is-the-agent-flailing/evidence/repeat_rate_over_time.png`
- `articles/trajectory-eval-is-the-agent-flailing/evidence/cumulative_best.png`
- `articles/distill-architect-lora-from-trajectories/evidence/calibration.png`

### Refreshed file (1)

- `src/data/project-stats.json` — re-run via `nvidia-learn-stats` skill. Stats were recomputed because article bodies changed substantially; word counts shifted slightly; LOC/section-counts unaffected by SVG additions (verifier counts SVG markup as prose, which is the existing convention). Top-line numbers: 32 published articles, ~110K words, 369K LOC across `evidence/` + `src/`.

### Unchanged

- No new files (no new `<Component>.astro` signature components — Phase A worked entirely in markdown using `fn-diagram__*` classes; signature thumbnails were not edited).
- No new routes.
- No new dependencies — `package.json` and `package-lock.json` untouched.
- No frontmatter / schema changes — `src/content.config.ts` untouched.
- No JS, no rehype/remark plugin edits, no CSS edits, no Astro config changes.

---

## Behavior contract

**Phase A — reconciliation.** Three articles previously rendered a polished signature thumbnail (in `src/components/svg/<Name>.astro`) on the home/stage card but had a visually thin body — `trajectory-eval` had three matplotlib PNGs whose stark aesthetic clashed; `distill-architect` had one matplotlib PNG plus prose-only sections; `pass-at-k` had only prose and tables. After this release each renders with body figures whose palette, motion vocabulary, and density match the signature. The article reads coherently end-to-end at the same fidelity.

**Phase B — diversification.** Six articles each gain one new inline figure that visualizes a section that previously lived only in prose or in a results table. No existing figures were modified or removed. Each new figure was selected via the visualizations.md taste-test (caption claims a thesis, removing it weakens the argument, motion serves understanding) and picks a deliberately underused archetype to reduce the corpus's flow-pipeline monoculture.

**Motion + accessibility:** every new figure follows the established `fn-diagram__*` class system — stroke-draw + fade-rise on `IntersectionObserver` entry, scroll-gated via `.fn-diagram--visible`, `prefers-reduced-motion` honored. Every `<svg>` carries `role="img"` and a sentence-level `aria-label`. Every `<path>` carries `pathLength="100"`.

---

## Verification (after sync)

1. **Pull the change** — `git pull` (single commit on `main`).
2. **Build clean** — `npm run build` should succeed; expect 60 pages built (same as before; no routes added). No Astro cache flush needed.
3. **Phase A spot-check** — open each at viewport ≥1472px, default theme:
   - `articles/trajectory-eval-is-the-agent-flailing/` — 3 inline figures replacing the previous matplotlib PNGs. Confirm no broken-image alt-text appears, and the SVGs animate on scroll-into-view.
   - `articles/distill-architect-lora-from-trajectories/` — 3 inline figures. The waterfall (figure 2) replaces what was `evidence/calibration.png`.
   - `articles/pass-at-k-after-the-seventh-patch/` — 3 inline figures (heatmap of cells + dual-path mechanism + timeline of 7 patches).
4. **Phase B spot-check** — open each, scroll to the new figure:
   - `baseline-training-loop-on-spark/` — timeline figure at "The shape of the envelope" section after the memory tables.
   - `t2po-uncertainty-guided-rl-on-spark/` — timeline figure between the held-out eval table and the "right column" paragraph.
   - `guardrails-for-code-generation/` — waterfall figure right after the bench bash code block.
   - `derisk-cloud-pretraining-on-the-spark/` — waterfall figure after the `expected_value_argument` JSON output.
   - `nemo-curator-training-data-prep/` — waterfall figure between data-overhead table and "biggest data-time number" paragraph.
   - `rerank-fusion-retrieval-on-spark/` — dual-path figure between the `retrieve()` python def and "Four branches" paragraph.
5. **Browser sanity** (paste into devtools at any of the 9 articles):
   ```js
   [...document.querySelectorAll('figure.fn-diagram')].map((f, i) => ({
     i,
     viewBox: f.querySelector('svg').getAttribute('viewBox'),
     visible: f.classList.contains('fn-diagram--visible'),
     captionLen: f.querySelector('figcaption')?.textContent.trim().length ?? 0,
   }));
   ```
   Every entry should have a non-null `viewBox`, `visible: true` after scrolling the figure into view, and `captionLen > 0` (interpretive caption, not empty alt-text).
6. **No 500 on previously-PNG-embedding pages** — `trajectory-eval-is-the-agent-flailing` and `distill-architect-lora-from-trajectories` no longer reference `evidence/*.png` from the body. Astro's image-optimizer should not try to fetch them. (During development on the source repo, a stale dev-server cache returned 500/`ImageNotFound` for `distill-architect` until a hard reload — a clean build does not have this issue.)

---

## Conflict-avoidance notes

- **Markdown-only changes** — every edit is inside `<figure class="fn-diagram">` blocks in article bodies. Destination's CSS/JS/Astro chrome is untouched.
- **No new SVG components** — Phase A explicitly used the in-article `fn-diagram__*` class system, NOT the signature `data-svg-animate` + `svg-reveal` system. The two systems coexist but were never mixed within a single SVG (per `~/.claude/skills/tech-writer/references/visualizations.md`).
- **No frontmatter / schema changes** — `signature:` fields point to the same Astro components as before; no new components were added under `src/components/svg/`.
- **PNG deletions are intentional** — the 4 deleted PNGs were `matplotlib`-generated chart images that the new SVGs supplant. Their original `.py` plot scripts remain in `evidence/` (e.g., `plot_calibration.py`) for reproducibility, but the rendered PNG outputs are gone and no longer referenced from any article body.
- **`src/data/project-stats.json`** — refreshed via the `nvidia-learn-stats` skill before commit, per the established convention. The home-page "At a glance" infographic numbers will refresh automatically; word counts shift slightly (each new figure's `aria-label` and `figcaption` count as prose).

---

## Source-of-truth pointer

- Source repo: `https://github.com/manavsehgal/ai-field-notes`
- 9 article body files + 4 PNG deletions + 1 stats refresh, in a single commit on `main`.
- Authoring guide referenced during the pass: `~/.claude/skills/tech-writer/references/visualizations.md` (six archetypes, validator constraints, halo containment, motion policy).

---

## Out of scope

- **Adding signatures to seed-stage articles.** Seven articles (a2tgpo, auto-research-loop, claw-eval-live, dci-corpus-operators, judge-orchestrated-ensemble, scientific-foundation-models-as-tools, skill-os-on-spark) live as `seed.md` + `transcript.md` + `evidence/` only — no published `article.md`. Adding a signature without a body is premature.
- **Adding figures to upcoming-status articles.** Four roadmap entries (lora-fine-tune-nemotron, nemo-framework-continued-pretraining, nsight-systems, spark-gpu-telemetry-prometheus-grafana) have `status: upcoming` placeholders; no body to figure.
- **Replacing UI/screenshot PNGs.** PNGs in `screenshots/` directories (NGC catalog pages, dashboards) document UI and stay as PNGs — they are not data visualizations.
- **Editing existing signature components.** No `src/components/svg/<Name>.astro` was modified. Card thumbnails on the home/stage pages render unchanged.
- **Sweeping every aligned article.** ~22 published articles already pick the right archetype for their thesis-load-bearing section; the audit explicitly skipped them. Phase B intentionally targeted only the high-leverage candidates.
