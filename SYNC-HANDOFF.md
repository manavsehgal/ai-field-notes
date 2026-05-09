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
- Project-stats refresh (no article publish, no content delta).
- Any change to mobile behavior, hide mode, inline mode, print stylesheet, or reader themes.
