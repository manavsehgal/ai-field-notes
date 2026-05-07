# SYNC-HANDOFF — Explainers feature (Phases 1–3 + layout fix)

This document is the canonical inventory for syncing the **explainers feature** (and its companion Kindle-style reader UX) from the source repo `manavsehgal/ai-field-notes` to the destination repo that publishes `https://ainative.business/field-notes/`. The destination consumer should read this end-to-end and use the file lists, dependency changes, and behavior contracts below to identify every change worth porting.

**Source range:** commits `a85dd6a` → `5b51787` (4 commits, ~3,400 line diff including article content; ~700 lines of net code/component additions).

---

## TL;DR — what shipped

A **learning-aid layer** that turns each article's prose into something readable by two audiences in one document:

- The **advanced reader** who wants the deep-dive essay flow uninterrupted.
- The **cross-learning reader** (e.g. Go engineer new to LLMs, data scientist learning the inference stack) who needs *just-in-time* term definitions, *why-this-matters* framings, and *go-deeper* pointers without leaving the page.

Authors annotate articles with six directive types (`:::define`, `:::why`, `:::deeper`, `:::pitfall`, `:::math`, `:::hardware`). The rendering pipeline floats those annotations as cards into the gutter between the prose column and the article frame on desktop, collapses them inline on mobile, and auto-collects every `:::define[term]` to a `/glossary/` index page. A reader settings menu lets the reader switch between **Hide / Margin / Inline** for explainers, and tune **Font Size**, **Line Height**, and **Reader Theme** (Light / Sepia / Dark) — all persisted in `localStorage` under the `afn:*` namespace. Each article also gets a **bookmark star + scroll-position resume banner**, a **section-aware ToC drawer with scroll-spy**, and **previous/next nav within its editorial series**.

All 31 published articles now ship 8–10 explainers with a balanced palette. The glossary auto-grew from 0 → **108 entries**.

---

## File inventory — what to sync

### New files (10)

| Path | LOC | Purpose |
|---|--:|---|
| `src/lib/remark-explainers.mjs` | 159 | Remark plugin: parses `:::define / :::why / :::deeper / :::pitfall / :::math / :::hardware` container directives → `<aside class="explain explain--<kind>">` nodes. Also (a) neutralizes false-positive directives like `:59 UTC`, `:8001`, `3:2` so timestamps/ports/ratios in prose don't break, (b) collects every `:::define[term]` into `file.data.glossaryEntries`, (c) adds `explain--before-figure` class to explainers preceding a wide figure within ~3 sibling positions. |
| `src/lib/rehype-explainer-figure.mjs` | 39 | Rehype plugin: belt-and-suspenders pass at the hast stage that adds `explain--before-figure` to explainer asides preceding `<figure>` or bare `<p><img></p>` blocks. Catches cases the mdast walk in `remark-explainers` misses (raw-HTML figures stitched in differently by Astro's pipeline). |
| `src/lib/article-glossary.mjs` | 72 | Re-runs the `remark-directive + remark-explainers` pipeline against each article's raw body to extract `glossaryEntries` for the `/glossary/` page. Avoids importing rendered MDX modules at build time. |
| `src/styles/explainers.css` | 304 | Full stylesheet for `.explain` asides — base styling, six per-kind accent borders + eyebrow labels, the wide-viewport float-into-gutter rule with `--explainer-width` / `--explainer-claim` / `--explainer-gap` custom properties, three-band media-query cascade (13rem at ≥64rem, 15rem at ≥80rem, 17rem at ≥92rem), `:nth-of-type(2n)` left-side alternation, `:last-of-type` and `.explain.explain--before-figure` inline overrides, print stylesheet, sepia/dark/light reader themes scoped to `.article`, and the `.article__resume-banner` styling. |
| `src/components/ReaderSettings.astro` | 243 | Top-right dropdown menu with four reader controls — Font Size (S/M/L), Line Height (Compact/Default/Relaxed), Reader Theme (Light/Sepia/Dark), Explainer Mode (Hide/Margin/Inline). Writes to `localStorage` keys `afn:font-scale`, `afn:line-height`, `afn:reader-theme`, `afn:explain-mode`, and to `document.documentElement.dataset` so CSS attribute selectors react instantly. |
| `src/components/ReaderScript.astro` | 31 | Inline `<script>` injected in `<head>` that reads the four `afn:*` localStorage keys and writes them to `document.documentElement.dataset` *before* paint, preventing FOUC of the wrong reader theme/font/explain-mode. Companion to the existing `ThemeScript`. |
| `src/components/BookmarkStar.astro` | 150 | Star toggle in the article header. Click to bookmark; bookmarks persist as `afn:bookmark:<slug>` (timestamp + scroll position). On article load, if a bookmark exists, surfaces the `.article__resume-banner` ("Pick up where you left off?") with Resume / Dismiss buttons. |
| `src/components/TocDrawer.astro` | 260 | "Sections" pill that opens a slide-out drawer listing the article's `<h2>` / `<h3>` headings with active-section highlighting (IntersectionObserver scroll-spy). Click a heading → smooth-scroll to that section. |
| `src/components/ArticleArcNav.astro` | 149 | Renders Previous / Next arrows for the article's editorial series (Foundations, Second Brain, LLM Wiki, Autoresearch, Frontier Scout, Looking Beyond Spark). Uses git-derived publish ordinal from `src/lib/article-order.mjs`. |
| `src/components/TermsInThisPiece.astro` | 123 | Expandable summary block above the prose listing every `:::define[term]` in this article with anchor links — gives readers a quick map of the domain terms before diving in. |

### New routes (2)

| Path | LOC | URL | Purpose |
|---|--:|---|---|
| `src/pages/glossary/index.astro` | 216 | `/glossary/` | A-Z reference of every `:::define[term]` site-wide. Auto-built by importing every article's `body` and running `article-glossary.mjs`. Renders A–Z navigation chips at top, then alphabetized cards with definition + sources (article title + anchor). Currently 108 entries. |
| `src/pages/bookmarks.astro` | 263 | `/bookmarks/` | Client-hydrated list of articles the reader has bookmarked (read from `afn:bookmark:*` localStorage keys). Shows article title, last-bookmarked timestamp, and Resume button. Empty-state message when no bookmarks exist. |

### Modified files (8)

| Path | Δ LOC | Change |
|---|--:|---|
| `astro.config.mjs` | +11 −2 | Imports `remark-directive`, `remark-explainers`, `rehype-explainer-figure`. Adds `remarkDirective` and `remarkExplainers` to `markdown.remarkPlugins` (in that order — `remark-directive` parses the `:::name[label]` syntax; `remark-explainers` interprets it). Adds `rehypeExplainerFigure` to `markdown.rehypePlugins` after `rehypeCaption`. |
| `package.json` | +1 | Adds dependency `remark-directive: ^4.0.0`. |
| `src/layouts/BaseLayout.astro` | +2 | Imports and renders `<ReaderScript />` next to the existing `<ThemeScript />`. Both must run before paint. |
| `src/pages/articles/[slug].astro` | +24 −1 | Imports `TermsInThisPiece`, `ReaderSettings`, `BookmarkStar`, `TocDrawer`, `ArticleArcNav`. Wraps `.article__breadcrumb` + the new `.article__reader-controls` cluster in a flex `.article__header-row`. Adds the `.article__resume-banner` (hidden by default, shown by `BookmarkStar` JS). Mounts `<TermsInThisPiece body={article.body} />` between the header and the prose. Mounts `<ArticleArcNav slug={article.id} />` after `<FieldkitCTA />`. |
| `src/styles/global.css` | +30 −4 | (a) `@import "./explainers.css"` near the top. (b) `.article` `max-width: 80rem` → **`92rem`** for the wider gutter. (c) `.prose p > img:only-child` and `.prose figure.figure--bleed img` width cap `80rem` → **`88rem`** so figures stay visually proportional in the wider frame. (d) New `.article__header-row` and `.article__reader-controls` flex rules with a `max-width: 640px` stack. |
| `src/styles/diagrams.css` | +8 −1 | `.prose .fn-diagram` width cap `80rem` → **`88rem`** (matches the figure-bleed bump). Comment updated to explain the explainer-side inline-render contract. |
| `src/data/project-stats.json` | regen | Refreshed to reflect 31 articles · 106,881 words · 45,889 LOC. The home page's "At a glance" infographic reads from this. |

### Article content changes (31 files)

All 31 published `articles/<slug>/article.md` files received 8–10 explainer directives each, ~278 explainers total. **Each article body is byte-identical to the prior version *except* for the inserted `:::define / :::why / :::deeper / :::pitfall / :::math / :::hardware` blocks** (placed between paragraphs, never inside `<figure>` HTML blocks). The destination should sync these markdown files in full from source.

The list of touched articles:

```
autoresearch-agent-loop                       (9 explainers)
autoresearchbench-on-spark                    (9)
baseline-training-loop-on-spark               (9)
bigger-generator-grounding-on-spark           (10)
clawgym-on-spark                              (9)
clawgym-on-spark-grpo                         (9)
derisk-cloud-pretraining-on-the-spark         (9)
dgx-spark-day-one-access-first                (9)
distill-architect-lora-from-trajectories      (9)
gpu-sizing-math-for-fine-tuning               (10)  pilot
guardrails-for-code-generation                (8)
guardrails-on-the-retrieval-path              (10)
kv-cache-arithmetic-at-inference              (10)  pilot
lora-on-your-own-qa-pairs                     (9)
mcp-second-brain-in-claude-code               (9)
naive-rag-on-spark                            (10)
nemoclaw-vs-openclaw-dgx-spark                (9)
nemo-curator-training-data-prep               (9)
nemo-framework-on-spark                       (8)
nemo-retriever-embeddings-local               (10)
nim-first-inference-dgx-spark                 (10)
one-substrate-three-apps                      (8)   bridge
pass-at-k-after-the-seventh-patch             (9)
pgvector-on-spark                             (10)
rag-eval-ragas-and-nemo-evaluator             (10)
rerank-fusion-retrieval-on-spark              (10)
runtime-frontier-six-patches-on-spark         (9)
test-time-distilling-for-exploration          (9)
trajectory-eval-is-the-agent-flailing         (8)
trtllm-and-triton-on-spark                    (9)
what-the-agent-actually-built                 (8)
```

The 4 `status: upcoming` placeholder articles (spark-gpu-telemetry-prometheus-grafana, lora-fine-tune-nemotron-on-spark, nemo-framework-continued-pretraining-on-spark, nsight-systems-on-spark) were not edited — they have no published body yet.

---

## Dependency change

A single new npm dependency:

```
"remark-directive": "^4.0.0"
```

This parses the `:::name[label]` container syntax into `containerDirective` nodes that `remark-explainers` then transforms. Already in `package-lock.json`.

---

## Plugin chain — required wiring in `astro.config.mjs`

Order is load-bearing — `remark-directive` must come before `remark-explainers`:

```js
import remarkDirective from 'remark-directive';
import remarkExplainers from './src/lib/remark-explainers.mjs';
import rehypeExplainerFigure from './src/lib/rehype-explainer-figure.mjs';

export default defineConfig({
  // ...
  markdown: {
    remarkPlugins: [
      [remarkFixLinks, { base, repoBase: REPO_BASE }],
      remarkDirective,
      remarkExplainers,
    ],
    rehypePlugins: [rehypeCaption, rehypeExplainerFigure],
    // ...
  },
});
```

---

## Authoring contract — six directive types

Authors write blocks like:

```markdown
prose paragraph that introduces the term…

:::define[KV cache]
Per-token attention state cached during decode so the model skips
recomputing attention over the full context every step.
:::

next prose paragraph continues the discussion…
```

The six recognized types (anything else is passed through as plain text):

| Directive | Eyebrow label | Accent | Auto-collected to /glossary/? |
|---|---|---|---|
| `:::define[term]` | DEFINE | cyan | **Yes** — `term` becomes the glossary entry name (must be unique within an article) |
| `:::why[bold-headline]` | WHY THIS MATTERS | amber | No |
| `:::deeper` | GO DEEPER | violet | No (use as bullet list of links) |
| `:::pitfall[bold-headline]` | PITFALL | rose | No |
| `:::math[bold-headline]` | IN PLAIN WORDS | emerald | No |
| `:::hardware[bold-headline]` | BEYOND SPARK | gold | No (frontier-hardware extrapolation; usually closes the article) |

**Per-article budget:** 8–10 explainers, balanced palette of 3–4 `:::define` + 1–2 `:::why` + 0–1 `:::pitfall` + 0–1 `:::math` + 1 `:::deeper` + 1 `:::hardware`. Bracket labels are mandatory for `:::why`, `:::pitfall`, `:::math`, `:::hardware` and should read like the article's argument spine when scanned in isolation.

False-positive directives (`:59`, `:8001`, `3:2`) are neutralized — they render as plain text without escaping.

---

## CSS architecture — explainers.css

The full stylesheet is `src/styles/explainers.css` (304 lines). Key behaviors:

- **Floats into the gutter** between `.prose` (48rem) and `.article` (now **92rem**). On wide viewports the aside floats right with `clear: right`, and `:nth-of-type(2n)` floats left so a long sequence alternates sides.
- **Tokenized width** via `.prose { --explainer-width; --explainer-claim; --explainer-gap; }` custom properties. Three-band media cascade scales the float with available gutter:
  - `≥64rem` viewport: 13rem wide / 14rem claim
  - `≥80rem`: 15rem / 16.25rem
  - `≥92rem`: 17rem / 18.25rem (~36ch — readability sweet spot)
- **`shape-outside: margin-box`** + **`shape-margin: 0.5rem`** so prose paragraphs hug the float edge cleanly.
- **`:last-of-type` inline override** — the closing `:::hardware` per article renders as a full-prose-width inline callout, so the gutter doesn't trail past the article body's natural end.
- **`.explain.explain--before-figure` inline override** — explainer asides tagged by remark + rehype as preceding a wide figure within ~3 sibling positions also render inline above the figure (full prose width). Avoids both overlap (figure covers float) and the side-gap that `clear: both` would create.
- **Print stylesheet** drops floats so explainers don't break across pages.
- **Reader theme overrides** (sepia / dark / light) scope CSS variables onto `.article` only — Nav / Footer / home page keep the global theme.

Companion CSS:

- `src/styles/global.css` — `.article max-width: 92rem`, figure-bleed cap `88rem`, `.article__header-row` flex rules.
- `src/styles/diagrams.css` — `.prose .fn-diagram` cap `88rem`.

---

## Reader settings — `localStorage` contract

The destination must reserve these `localStorage` keys (namespaced `afn:*` to coexist with the destination's own `/book/` reader keys, which live under different prefixes):

| Key | Values | Owner | Read by |
|---|---|---|---|
| `afn:font-scale` | `S` / `M` / `L` | ReaderSettings | ReaderScript (sets `data-font-scale` on `<html>`) |
| `afn:line-height` | `compact` / `default` / `relaxed` | ReaderSettings | ReaderScript (sets `data-line-height`) |
| `afn:reader-theme` | `light` / `sepia` / `dark` | ReaderSettings | ReaderScript (sets `data-reader-theme`) |
| `afn:explain-mode` | `hide` / `margin` / `inline` | ReaderSettings | ReaderScript (sets `data-explain-mode`) |
| `afn:bookmark:<slug>` | JSON `{at: ISO timestamp, scroll: pixel offset}` | BookmarkStar | BookmarkStar (toggle), `/bookmarks/` page (list), article slug page (resume banner) |

**FOUC contract:** `ReaderScript` is a synchronous `<script>` in `<head>`, sibling to `ThemeScript`. It must execute before first paint to avoid the wrong theme/font flashing in.

---

## Reader UX — what users see and do

1. **Article header** now has a right-aligned cluster of pill buttons: ★ Bookmark / ☰ Sections / ⚙︎ Reading.
2. **Bookmark star** — toggle to remember position. On article reload, if a bookmark exists, a resume banner ("Pick up where you left off?") appears under the header with **Resume** (smooth-scrolls to saved position) and **Dismiss** (clears bookmark for this slug) buttons.
3. **Sections** opens a slide-out drawer listing the article's headings; the current heading highlights via scroll-spy; click any heading to jump there.
4. **Reading** opens a settings popover with four radio groups:
   - **Font Size:** S · M (default) · L
   - **Line Height:** Compact · Default · Relaxed
   - **Reader Theme:** Light · Sepia · Dark (scoped to `.article` only — Nav/Footer keep the global theme)
   - **Explainer Mode:** Hide · Margin (default) · Inline
5. **Terms in this piece** — an expandable summary block under the article header listing every `:::define[term]` with anchor links. Gives the cross-learning reader a domain map before reading.
6. **Explainers in margin mode** float into the gutter alternating left/right. The first explainer in a logical "near-figure" position renders inline as a full-prose-width callout above the figure (avoids overlap/gap). The last explainer per article also renders inline at full prose width — looks like a closing callout.
7. **Explainer hide mode** — all `.explain` blocks `display: none`; prose flows normally.
8. **Explainer inline mode** — all explainers render inline as full-prose-width blocks.
9. **`/glossary/`** — A–Z navigation, alphabetized entries, definition + source-article links. Auto-grows from `:::define[term]` site-wide.
10. **`/bookmarks/`** — client-hydrated list of bookmarked articles with resume buttons; empty-state when none.
11. **ArticleArcNav** at article foot — Previous / Next within the editorial series.

---

## Layout fix details (commit `5b51787`)

The most recent commit addressed seams that surfaced in vibe-test of the Phase 3 backfill:

1. **Wider gutter** — `.article` 80rem → 92rem. Figure breakouts (`.fn-diagram`, `.figure--bleed`, `.figure--wide`, bare `<p><img>`) bumped to 88rem cap so they remain visually proportional in the wider frame.
2. **Roomier explainer floats** — three-band tokenized cascade (13/15/17rem) up from a flat 13rem, for ~36ch readability at the widest band.
3. **No overlap, no gap with figures** — the `explain--before-figure` class drops float on explainers preceding figures, rendering them as full-prose-width inline callouts above the figure. Detection runs in both the remark plugin (mdast siblings) and the rehype plugin (hast siblings) for resilience.
4. **No tail whitespace** — `:last-of-type` rule renders the closing `:::hardware` inline so the gutter doesn't extend past the article body's natural end.
5. **Print stylesheet** — explainers go inline when printing.

---

## Verification (after sync)

The destination should run, in order:

1. **Install** — `npm install` (picks up the new `remark-directive` dependency).
2. **Build clean** — `npm run build` should produce **59 pages** with no warnings (12 figures optimized).
3. **Page count diff** — destination's prior render had whatever count it had; the Phase 1 → Phase 3 chain added 2 new routes (`/glossary/` and `/bookmarks/`), so expect +2 pages over the pre-sync baseline.
4. **Glossary populated** — visit `/glossary/`. The masthead should read **"108 entries from 31 articles"**. Check that A–Z chips render and a few entries (e.g. "KV cache", "REINFORCE", "NIM — NVIDIA Inference Microservices") are present with definition + source link.
5. **Reader settings work** — open an article (e.g. `/articles/clawgym-on-spark-grpo/`), click ⚙︎ Reading. Cycle through Font Size / Line Height / Reader Theme / Explainer Mode. Verify the page responds instantly and the choice persists across reloads (localStorage).
6. **Bookmark + resume** — click ★, scroll to mid-article, reload. The resume banner should appear; clicking Resume should smooth-scroll to saved position.
7. **Explainer layout sanity** — at a viewport ≥1472px (92rem):
   - Floats sit in the right gutter, alternating left for `:nth-of-type(2n)`. Explainer width should be ~272px (17rem).
   - On `clawgym-on-spark-grpo`, the `:::define[REINFORCE]` explainer renders **inline above** the GRPO loop fn-diagram (no overlap, no side gap).
   - On any article, the closing `:::hardware` block renders inline at full prose width.
   - At 1280px viewport, explainer narrows to 240px (15rem) and the layout still works without horizontal scroll.
8. **Article body byte-stability** — for any article, the prose between the `:::name[…]:::` blocks should be byte-identical to the pre-sync version. The destination's existing renderer should not produce stale diff in non-explainer paragraphs.

---

## Conflict-avoidance notes

- **No edits to chrome** (Nav, Footer, ThemeToggle, Logo) so the destination's own clone of those components keeps its overrides intact.
- **`afn:*` localStorage namespace** chosen specifically to coexist with the destination's `/book/` reader keys (which use a different prefix).
- **Reader theme** scopes CSS variable overrides to `.article` only — Nav/Footer and home page keep the destination's global theme.
- **No new icons or fonts** beyond what the source already had.
- **`remark-directive`** is the only new npm dependency. If the destination uses a different bundler/registry, ensure the package resolves before building.
- **Astro 5 cache caveat** — after pulling, `rm -rf .astro node_modules/.astro` before the first build, otherwise the content collection cache may serve stale rendered articles without the new `:::` blocks.

---

## Source-of-truth pointers

For deeper reference if needed during sync:

- Authoring rules + worked examples: source repo `/.claude/skills/tech-writer/references/explainers.md` (this is the playbook future articles follow).
- Phase summaries in source: commits `a85dd6a` (Phase 1), `90392ad` (Phase 2), `ebe2994` (Phase 3), `5b51787` (layout fix).
- Live source dev URL: `http://localhost:4321/` on Spark, `http://10.0.0.209:4321/` on LAN.
