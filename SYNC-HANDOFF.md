<!--
  ⚠️ STATUS: NEW — content below ships in the head commit on `main` after this release lands.
  At the next release prompt, **clear this entire file and start fresh** (do NOT append to existing sections).
  The file is *one feature/release at a time* — not a running log.
  Last reset: 2026-05-11 (prior content covered the fieldkit v0.3.0 release window — shipped through `53fe370`, `ab125e5`, `f53a9dc`, `87a2ebe`, plus landing-page follow-ups `0ba1fd1`, `2a08e6f`, `500e9f4`; all consumed).
-->

# SYNC-HANDOFF — a2tgpo article publish + first MTBM consumer of fieldkit.lineage (2026-05-11)

One article published. The piece [`a2tgpo-turn-clipping-on-spark`](articles/a2tgpo-turn-clipping-on-spark/) is the first Machine-that-Builds-Machines article whose Python code blocks import the `fieldkit.lineage` module that v0.3.0 shipped. It promotes the previously-staged Frontier-Scout placeholder (status: upcoming, scaffolded 2026-05-08) into a published deep-dive (status: published, 2026-05-11). The article is **study-from-source** mode — the A²TGPO paper (arXiv 2605.06200) trains on 8×H20, which projects to six-to-eight days of wall-clock per single configuration on a single GB10. Per the `feedback_spark_scaling_optimism` memory rule that caps per-trial wall at ~6 hours, this promotion does not attempt a fresh reproduction; the released artifact studied is the source code at `verl_atgpo/verl/trainer/ppo/core_algos.py:1264–1400`.

The article walks the three primitives the paper adds to the GRPO loss family — turn-group normalization, variance-rescaled discounted accumulation, adaptive turn-level clipping — and demonstrates the lineage primitive recording each A²TGPO trial's IG-clip-scale telemetry via a working `evidence/lineage-demo.py` that prints the same Markdown block the next specialist would see at session entry.

**Source range:** one new commit on top of `500e9f4` (the head after the fieldkit v0.3.0 landing-page follow-ups).

---

## TL;DR — what shipped

| Path | Change | Notes |
|---|---|---|
| `articles/a2tgpo-turn-clipping-on-spark/article.md` | **NEW** — ~4,600 words, full 8-section deep-dive | Replaces the prior `seed.md` placeholder. Frontmatter: `status: published`, `stage: fine-tuning`, `also_stages: [agentic, training]`, `series: Machine that Builds Machines`, `book_chapters: [10]`, `fieldkit_modules: [capabilities, training, lineage]`. Two inline `<figure class="fn-diagram">` figures (layered-stack of the GRPO family tree + flow pipeline of the three primitives) + 8 sidebar explainers (4 define · 1 why · 1 math · 1 pitfall · 1 hardware · 1 deeper). |
| `articles/a2tgpo-turn-clipping-on-spark/seed.md` | **DELETED** — promoted to `article.md` | Standard upcoming → published transition. |
| `articles/a2tgpo-turn-clipping-on-spark/transcript.md` | **UPDATED** — provenance refresh | Documents the study-from-source mode + the lineage-demo evidence artifacts. |
| `articles/a2tgpo-turn-clipping-on-spark/evidence/lineage-demo.py` | **NEW** — ~165 LOC, pure-stdlib (modulo fieldkit) | Six-trial worked example (baseline → ATPO joint → separate-norm discard → turn-group keep → full v1d keep → α=0.9 eval-budget-overrun). Writes into `fieldkit.lineage.LineageStore`, configured `lower_is_better=False` since HotpotQA EM is higher-is-better. Renders the next-specialist prompt + structured handles. Exit-cleanly tempdir. |
| `articles/a2tgpo-turn-clipping-on-spark/evidence/lineage-rendered.txt` | **NEW** — 51 lines | The rendered Markdown block the demo prints, captured verbatim. Article §5 excerpts the KNOWLEDGE.md slice. |
| `articles/a2tgpo-turn-clipping-on-spark/evidence/results.tsv` | **NEW** — 8 lines (header + 6 rows + trailing) | The exact `results.tsv` the demo writes — six rows demonstrating the canonical 17-column schema with the A²TGPO trial shape. |
| `src/components/svg/AdaptiveClipBand.astro` | **NEW** — 300×200 signature component | Card-thumbnail visualisation: 6-turn adaptive-clip band centered on PPO ratio 1.0, modulating bounded (0.7, 1.3) via `c = 1 + 0.3·(2σ(IG)−1)`, with the paper's `+1.75 EM` headline number. Passes all hard invariants. |
| `src/components/svg/LineageAblationKeeps5x.astro` | **UPDATED** — `stroke-width="1.2"` → `1.5` (×2) | Pre-existing hard-invariant violations surfaced by `verify_svg.sh` (the validator scans every signature on every article publish, not just the article's own). Stroke values normalized to the `{0.5, 1, 1.5, 2}` hierarchy. No visual regression — the green-accent bar's stroke is now in the secondary-flow weight rather than off-grid. |
| `src/data/project-stats.json` | **UPDATED** — stats refresh | `articles_published`: 33 → **34**. `words_total`: ~113,000 → **117,403** (+~4,600 from the new article). `code.evidence_loc`: 13,839 → **14,052** (+213 from `evidence/lineage-demo.py`). `code.fieldkit_loc` unchanged at 6,971. `code.total_loc`: 20,810 → **21,023**. Per-stage counts: `fine-tuning` 7 → **8**. Upcoming count: 4 → **3** (one upcoming consumed). |
| `README.md` | **UPDATED** — regenerated via `refresh_readme.py` | Picks up the new article in the `Machine that Builds Machines` series row + the fine-tuning stage table + the LOC sub-line. The article surfaces at `№33` on the home page (ordinal derived from git first-add time via `src/lib/article-order.mjs`). |

---

## Where to look

- **The article itself**: `articles/a2tgpo-turn-clipping-on-spark/article.md` (frontmatter + body).
- **The working lineage demo**: `articles/a2tgpo-turn-clipping-on-spark/evidence/lineage-demo.py`. Run with `PYTHONPATH=fieldkit/src python3 articles/a2tgpo-turn-clipping-on-spark/evidence/lineage-demo.py` to reproduce the rendered prompt + TSV from a fresh tempdir.
- **The captured evidence artifacts**: `evidence/lineage-rendered.txt` (the rendered Markdown block) and `evidence/results.tsv` (the 6-row TSV).
- **The signature**: `src/components/svg/AdaptiveClipBand.astro`.

---

## Out of scope (intentionally — not blocking)

- **`fieldkit.training.rl` extraction.** The article scopes three named primitives that should land in a future `fieldkit.training.rl` submodule: `InformationGain` (per-turn logit-diff), `TurnGroupNormalizer` (composite-group reduction), `AdaptiveTurnClipper` (σ-bounded multiplier). Each ~50 LOC pure-torch. Deferred to a v0.4 fieldkit cut after a follow-up article runs a single A²TGPO configuration to wall-clock completion on Spark.
- **Fresh Spark reproduction of A²TGPO.** A future article will pick this up as the next MTBM Ch10-forge station — single configuration, overnight wall-clock, lineage row written to a real `LineageStore`. The receipt this article scopes is what that article's run produces.
- **Wave-2 retrospective on prior MTBM articles.** Carried forward from prior HANDOFF deferred lists: `autoresearch-agent-loop`, `trajectory-eval-is-the-agent-flailing`, `clawgym-on-spark-grpo`, `t2po-uncertainty-guided-rl-on-spark` all produced lineage-shaped artifacts in ad-hoc formats. A future polish pass could rewrite their `evidence/` to use the canonical `LineageStore`. Not blocking the next article.
- **LOC lineage infographic on `/about/`.** Companion visual for per-article original vs vendored LOC. Data already in `project-stats.json` (`evidence_loc` / `fieldkit_loc` / `vendored_loc`); only the visual is missing.

---

## Conflict-avoidance notes for destination CC instance

- The `LineageAblationKeeps5x.astro` stroke-width fix is unrelated to the article publish but lands in the same commit. Both `1.2` instances become `1.5`. The visual is essentially indistinguishable; the change is hard-invariant compliance, not a redesign.
- The home page card ordering is preserved — the article gets the next available `№` (33) based on git first-add time, which orders it just below the auto-research-loop article (№34) by descending site ordinal. No re-numbering needed for prior articles.
- The schema (`src/content.config.ts`) is unchanged. `fieldkit_modules` enum already includes `lineage` (added in v0.3.0). `series` enum already includes `Machine that Builds Machines`. `also_stages` accepts the `[agentic, training]` value (both in `STAGES`).
- The article has no `signature`-component collision — `AdaptiveClipBand` is a new name.
- The verify_article.sh + verify_svg.sh gates ran green on every commit-eligible diff after the stroke-width fix.

---

## Build/verify gates passed

- `bash ~/.claude/skills/tech-writer/scripts/verify_article.sh a2tgpo-turn-clipping-on-spark` → all green (frontmatter, image refs, slug match, secret/PII scan, SVG hard invariants).
- `npm run build` → 63 pages, 5.30 s. Clean. No broken refs.
- `PYTHONPATH=fieldkit/src python3 evidence/lineage-demo.py` → reproduces the captured `evidence/lineage-rendered.txt` and `evidence/results.tsv` byte-equivalent.
- Playwright vibe-test at 1400×900 (desktop wide) and 700×900 (mobile-narrow): clean. Both inline diagrams render at full width with no overflow; the home card at narrow viewport stacks signature-below-summary as expected; the №33 ordinal watermark renders behind the card.

---

## Commit shape

One commit:

```
feat(article): publish a2tgpo-turn-clipping-on-spark — first MTBM consumer of fieldkit.lineage
```

Files:
- `articles/a2tgpo-turn-clipping-on-spark/article.md` (new, ~4,600 words)
- `articles/a2tgpo-turn-clipping-on-spark/seed.md` (deleted)
- `articles/a2tgpo-turn-clipping-on-spark/transcript.md` (updated)
- `articles/a2tgpo-turn-clipping-on-spark/evidence/lineage-demo.py` (new)
- `articles/a2tgpo-turn-clipping-on-spark/evidence/lineage-rendered.txt` (new)
- `articles/a2tgpo-turn-clipping-on-spark/evidence/results.tsv` (new)
- `src/components/svg/AdaptiveClipBand.astro` (new)
- `src/components/svg/LineageAblationKeeps5x.astro` (pre-existing hard-invariant fix)
- `src/data/project-stats.json` (refresh)
- `README.md` (refresh)
- `SYNC-HANDOFF.md` (this file)
