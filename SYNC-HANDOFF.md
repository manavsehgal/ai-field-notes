<!--
  ⚠️ STATUS: NEW — content below ships in the head commit on `main` after this release lands.
  At the next release prompt, **clear this entire file and start fresh** (do NOT append to existing sections).
  The file is *one feature/release at a time* — not a running log.
  Last reset: 2026-05-10 (prior content covered the v2 visualization quality pass; shipped through `6a51bcc`/`f341514`/`febeb29` and consumed).
-->

# SYNC-HANDOFF — Auto-Research-Loop article published (2026-05-10)

One new published article (`auto-research-loop-on-spark`) + one new signature SVG (`LineageAblationKeeps5x.astro`) + refreshed project-stats. Promotes the previously-upcoming Frontier Scout placeholder for cxcscmu's Auto-Research-Recipes (arXiv 2605.05724, ICML 2026) to a published study based on the release_artifacts/ — no fresh Spark experiment was run (faithful trial replication on 1×GB10 is 10–25 hr setup + multi-day wall, deferred). Article framed as reading the lineage primitive from cxcscmu's frozen release.

**Source range:** head of `main` (single commit on top of `6a51bcc`).

---

## TL;DR — what shipped

| Path | Change | Notes |
|---|---|---|
| `articles/auto-research-loop-on-spark/article.md` | **NEW** (5,139 words) — replaces previous `upcoming` preview | status: `upcoming` → `published`. Frontmatter: `series: Machine that Builds Machines`, `book_chapters: [10, 11]`, `stage: agentic`, `also_stages: [training, foundations]`, `fieldkit_modules: [capabilities, training]`. Title: "Reading the Lineage Primitive — cxcscmu Auto-Research, Studied from release_artifacts" |
| `src/components/svg/LineageAblationKeeps5x.astro` | **NEW** signature SVG | Paired-bar archetype. 16 keeps (lineage on) vs 3 keeps (lineage off) — the 5.3× ratio is the article's load-bearing claim. Style matches `T2poPoolVsHeldout` (300×200 viewBox, OKLCH tokens, `data-svg-animate`, `svg-reveal-d1..d4` motion gating). |
| `src/data/project-stats.json` | refreshed via `nvidia-learn-stats` | 33 published articles (+1), 114,166 words (+4,139), 369,415 LOC (+87). `stages.agentic` 12→13, `stages.foundations` 11→12, `stages.training` 10→11. |

No other files modified. No schema / CSS / plugin / component edits beyond the new signature file.

---

## Article shape

- **Lede.** cxcscmu's 1,704-trial public release has agentic infrastructure, ten specialists, an MCP toolchain — but the load-bearing intervention is a 17-column TSV called `results.tsv`. The paper's own `pg_ablation_lineage_on` vs `pg_ablation_lineage_off` runs prove the claim: same agent, same prompt template, same trial budget — only the rendered lineage block differs, and the lineage-on run produces 16 keeps where lineage-off produces 3.
- **Body.** Schema walkthrough (17 columns, with `(hypothesis, status, notes)` as the load-bearing triple); waterfall figure of the ablation status histograms; anatomy of the released `example_lineage_pg_lineage_on_arch.txt` worked-example prompt; per-class semantics of the 9 status enum values; the proposed `fieldkit.lineage` module (Trial, RecipeEdit, LineageSnapshot, LineageStore, FailureLabel enum — ~200 LOC sketch in the article).
- **Closing.** Why Spark reproduction is deferred: 4,800 H100-GPU-seconds-per-trial scales to ~15–20 hr wall per trial on 1×GB10, plus 10–25 hr of setup (FA3 build, FineWeb10B SP8192 re-tokenization, scheduler patches). The released 1,704-trial corpus is a richer dataset than a Spark smoke would produce. Future-work scaffold at `/home/nvidia/work/auto-research/` carries OAuth patches for when DGX Cloud time becomes available.

---

## Figures

1. **Signature** (`LineageAblationKeeps5x.astro`) — paired-bar archetype. Lineage on (16 keeps, accent green, halo) vs lineage off (3 keeps, primary indigo). 5.3× callout between bars in accent green. Used on the card thumbnail (home + stage/agentic pages).
2. **Inline figure** (in article, `fn-diagram` class, viewBox 880×360) — paired waterfall of the full status-class distribution. Five status bars per side (keep, eval_budget_overrun, discard, crash, size_blocked) scaled by absolute count, with the lineage-off `eval_budget_overrun · 123` label rendered *inside* the wide red bar (the wall-waste is so large the label sits in the bar to avoid SVG overflow).

Both figures vibe-tested at 1366×768 via Playwright MCP after a fresh `npm run build` clean (60 → 61 pages, ~5.4 s). Inline figure passed the blank-line trap check (memory `feedback_fn_diagram_no_blank_lines`) — `wc -l` between `<figure ...>` and `</figure>` shows no empty lines inside.

---

## Behavior contract (destination CC instance)

- The article's `status: published` makes it appear in the home-page article list (cards sorted by `article-order.mjs` git-derived ordinal, descending; this article will be №33 — the highest currently).
- The signature SVG is referenced by the article's `signature: LineageAblationKeeps5x` frontmatter field. The card thumbnail renderer (`src/components/ArticleCard.astro` or equivalent) imports from `src/components/svg/LineageAblationKeeps5x.astro` by name.
- `series: "Machine that Builds Machines"` adds the article to `/series/machine-that-builds-machines/`; it joins `autoresearch-agent-loop`, `trajectory-eval-is-the-agent-flailing`, and `distill-architect-lora-from-trajectories` in that arc.
- `book_chapters: [10, 11]` is a forward-compatible declaration; this repo doesn't render `/book/` backlinks yet, but the field is preserved so destination renderers can pull it.
- `fieldkit_modules: [capabilities, training]` uses the existing enum. The article *proposes* a new `fieldkit.lineage` module for v0.3, but the enum doesn't include `lineage` yet — adding it would require a coordinated schema bump that's out of scope for this release.

---

## Verification (already run, locally)

- `npm run build` — clean. 61 pages built in ~5.4 s. No schema or content-collection errors.
- Playwright MCP at 1366×768 — article header, signature card on home, inline waterfall figure, signature SVG on the article body (where the signature renders again). All confirmed.
- `nvidia-learn-stats` skill — re-ran, JSON timestamp updated; stage counts and article count incremented correctly.
- Smoke screenshots cleaned from `.playwright-mcp/aifn-smoke/auto-research-loop/` before commit per `feedback_browser_smoke_snapshots_tmp`.

---

## Conflict avoidance for destination

- No schema changes — `src/content.config.ts` untouched.
- No CSS changes — relies entirely on existing OKLCH tokens and `.fn-diagram` styles.
- No new dependencies — signature SVG is plain Astro + inline SVG.
- No changes outside the three paths listed in the TL;DR.

If the destination CC instance has cached the previous `upcoming` preview of `auto-research-loop-on-spark`, the cache should drop for that slug — the article body, status, and signature have all changed.

---

## Out of scope (intentionally deferred)

- **Live Spark reproduction.** Working scaffold exists at `/home/nvidia/work/auto-research/` (host-side, outside this repo): fresh clone of `cxcscmu/Auto-Research-Recipes`, dedicated venv with `claude-agent-sdk` + `filelock` + `psutil`, and patched `agent_core/harness/credentials.py` + `multi_agent_pg/harness/credentials.py` accepting `MAGENT_USE_OAUTH=1` to bypass the hard `ANTHROPIC_API_KEY` env-var check. Verified the Claude Agent SDK OAuth path against the local Claude Code session (Haiku 4.5 round-trip). Entry point when DGX Cloud or multi-H100 lab time is available: `MAGENT_USE_OAUTH=1 python -m multi_agent_pg.supervisor --state-root ./magent_state_pg_smoke --deadline-hours 24`.
- **`fieldkit.lineage` module landing.** This article proposes the module shape. Actual implementation lands in a future `fieldkit` release (v0.3 candidate). The `FIELDKIT_MODULES` enum in `src/content.config.ts` will need a `lineage` entry at that point.
- **Wave-2 retrospective on prior MTBM articles.** The autoresearch-agent-loop / trajectory-eval / Phase 6 GRPO / T²PO articles all produced lineage-shaped artifacts in ad-hoc formats; a future pass could rewrite their evidence/ to use the same `fieldkit.lineage.LineageStore` API once the module ships.
- **Next Frontier Scout article.** Priority 2 in the queue is `a2tgpo-turn-clipping-on-spark` (defines `fieldkit.training.rl`); it's the next foundational installment in the same MTBM arc.
