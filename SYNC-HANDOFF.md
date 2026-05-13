<!--
  ✅ STATUS: SHIPPED — consumed by Mac CC on 2026-05-12 (sweep commit: manavsehgal/ainative-business.github.io@71293af). Content below remains for historical reference until the next release rotation clears the file.
  At the next release prompt, **clear this entire file and start fresh** (do NOT append to existing sections).
  The file is *one feature/release at a time* — not a running log.
  Last reset: 2026-05-11 (prior content covered the fieldkit v0.3.0 release window — shipped through `53fe370`, `ab125e5`, `f53a9dc`, `87a2ebe`, plus landing-page follow-ups `0ba1fd1`, `2a08e6f`, `500e9f4`; all consumed).
-->

# SYNC-HANDOFF — a2tgpo article publish + first MTBM consumer of fieldkit.lineage (2026-05-11)

One article published. The piece [`a2tgpo-turn-clipping-on-spark`](articles/a2tgpo-turn-clipping-on-spark/) is the first Machine-that-Builds-Machines article whose Python code blocks import the `fieldkit.lineage` module that v0.3.0 shipped. It promotes the previously-staged Frontier-Scout placeholder (status: upcoming, scaffolded 2026-05-08) into a published deep-dive (status: published, 2026-05-11). The article is **study-from-source** mode — the A²TGPO paper (arXiv 2605.06200) trains on 8×H20, which projects to six-to-eight days of wall-clock per single configuration on a single GB10. Per the `feedback_spark_scaling_optimism` memory rule that caps per-trial wall at ~6 hours, this promotion does not attempt a fresh reproduction; the released artifact studied is the source code at `verl_atgpo/verl/trainer/ppo/core_algos.py:1264–1400`.

The article walks the three primitives the paper adds to the GRPO loss family — turn-group normalization, variance-rescaled discounted accumulation, adaptive turn-level clipping — and demonstrates the lineage primitive recording each A²TGPO trial's IG-clip-scale telemetry via a working `evidence/lineage-demo.py` that prints the same Markdown block the next specialist would see at session entry.

**Source range:** the a2tgpo article publish (commit `17bb24b`) plus the overnight-harness preflight (`88ca4be`) plus the stats refresh (`34e33d9`), all on top of `500e9f4` — *plus* the sync-contract bootstrap landing in this same pull cycle (see §"Mac CC action bundle" below).

**Also bundled in this pull cycle:** the sync-contract bootstrap (Phase 1 of the new Spark ↔ Mac sync strategy — see `SYNC-CONTRACT.md`, `SYNC-RENAMES.log`, `mirrors/destination-overrides.md` at source repo root). Bundled because it is a one-time additive contract that must land *before* any further releases use the new YAML-frontmatter schema. After Mac CC executes the bundle actions below, the next release rotation starts fresh with the new schema.

---

## Mac CC action bundle — sync-contract bootstrap (one-time, this pull only)

These actions are intentionally explicit so Mac CC's next consumption pass picks them up without needing a follow-up prompt from the user. Execute in order: A → B → C → D.

### A. Memory entries to save (Mac side)

| Memory | Type | Slug | Body |
|---|---|---|---|
| Sync contract pointer | reference | `reference_sync_contract` | Pointer to `SYNC-CONTRACT.md` at source repo root. By-glob authoritative-side table (articles/fieldkit/papers/evidence/scripts/ideas/content/stats/svg are Spark; book/pricing/about/landing/artifacts-chrome/skills are Mac). Forward YAML-frontmatter schema for SYNC-HANDOFF effective next release. |
| Renames log pointer | reference | `reference_sync_renames_log` | Append-only YAML log at source repo root. Replay protocol: on every pull, read the log, find entries with `status: destination-needs-replay`, apply, flip to `complete` via PR back to source (PR title: `mirror: SYNC-RENAMES.log status update — <date> — <summary>`). |
| Destination overrides mirror | reference | `reference_destination_overrides_mirror` | Mac-authored file at `mirrors/destination-overrides.md` in source repo. Mac CC keeps it current; opens PRs to source updating only this file when destination IA changes. Heartbeat: update `Last reverse-sync` date on every consumption cycle, even if content unchanged. |
| New SYNC-HANDOFF schema | feedback | `feedback_sync_handoff_frontmatter_schema` | Starting after this release, SYNC-HANDOFF.md gets YAML frontmatter (fields: `release_slug`, `status`, `source_range`, `articles_added`, `articles_updated`, `artifacts_added`, `artifacts_updated`, `fieldkit_modules_changed`, `renames_to_replay`, `removes`, `new_top_level_pages`, `breaking_changes`, `destination_overrides_to_preserve`, `hf_repos_added`, `civitai_artifacts_added`). Parse mechanically with `yaml.safe_load`; prose below the frontmatter stays human-readable. **Why:** structured deltas catch the rename + override + new-page drift modes that the 2026-04 → 2026-05 series rename surfaced. **How to apply:** sync-field-notes skill reads frontmatter first, then prose. |
| Phase 2 artifact manifests plan | project | `project_artifact_manifests_phase2` | When `fieldkit v0.4` lands `fieldkit.publish`, source begins writing `src/content/artifacts/<slug>.yaml` per artifact (quants, LoRAs, adapters, embedders, datasets, spaces, benchmarks). Mac renders `/artifacts/<kind>/` catalog pages from `getCollection('artifacts')`. **Why:** MTBM pivot ships AI-native artifacts at G3-quant cadence (3–5 days first ship, then ~weekly); manual SYNC-HANDOFF entries don't scale. **How to apply:** when first manifest appears on a pull, scaffold the `/artifacts/<kind>/` index + detail templates on Mac. |

### B. sync-field-notes skill updates

Add the following capabilities. Each can land as one self-contained PR or batched into one PR titled `feat(sync-field-notes): contract-aware sweep`.

1. **Parse SYNC-HANDOFF YAML frontmatter.** Detect the `---\n...\n---` block at the top; if present, parse with `yaml.safe_load`. Drive the sweep from structured fields. Fall back to prose-only parsing if frontmatter is absent (this current release — the bundle below acts as the inline equivalent).
2. **Replay SYNC-RENAMES.log.** On every pull: read the YAML log, find entries with `status: destination-needs-replay`, apply each rename across `articles/**` frontmatter + prose + tag/index pages, then open a PR back to source flipping the status to `complete`.
3. **Respect destination-overrides.** Before editing any path, check it against `mirrors/destination-overrides.md` globs. If the path is destination-owned, skip silently. (This is the no-clobber gate that prevents Spark from stomping `/book/`, `/pricing/`, etc.)
4. **Render `/artifacts/<kind>/` catalogs (Phase 2 stub).** No-op until `src/content/artifacts/` collection exists in a future pull. When it appears: query `getCollection('artifacts')`, filter by `kind`, render cards. Detail pages at `/artifacts/<kind>/<slug>/`.
5. **Acknowledge SYNC-HANDOFF consumption.** After sweep, flip the file's HTML-comment status marker from `STATUS: NEW` → `STATUS: SHIPPED` in a PR (or as part of the SYNC-RENAMES.log status-flip PR).

### C. One-time PRs from Mac → Source

1. **`mirrors/destination-overrides.md` inventory PR.** Replace the TBD placeholders with the actual destination-owned inventory: top-level pages (book, pricing, about, landing — confirm globs), style overrides (design tokens, OKLCH palette, font stack divergences), build config (`astro.config.mjs` prod base = `/field-notes/`, deploy hooks, redirects, 404). Update the `Last reverse-sync` heartbeat date. PR title: `mirror: destination-overrides update — 2026-05-12 — initial Mac inventory`.
2. **SYNC-RENAMES.log status flip PR** (after the Autoresearch → MTBM sweep finishes — see §D). Flip the 2026-05-08 entry from `destination-needs-replay` to `complete`. PR title: `mirror: SYNC-RENAMES.log status update — 2026-05-12 — autoresearch→MTBM swept`.
3. **SYNC-HANDOFF.md SHIPPED flip** (can ride along with PR #2 or stand alone). Flip the `STATUS: NEW` comment to `STATUS: SHIPPED` so the next release rotation has a clean signal.

### D. Autoresearch → MTBM rename sweep target

Per the 2026-05-08 entry in `SYNC-RENAMES.log`:

- **`frontmatter.series`**: any article with `series: Autoresearch` → `series: Machine that Builds Machines`. Affected articles (from memory `project_nvidia_learn_editorial` + recent SYNC-HANDOFFs): `autoresearch-agent-loop`, `trajectory-eval-is-the-agent-flailing`, `clawgym-on-spark-grpo`, `t2po-uncertainty-guided-rl-on-spark`, plus any newer MTBM entries. Spot-check by grepping `src/content/articles/**/*.md` for `series:.*Autoresearch`.
- **Prose mentions**: case-insensitive `Autoresearch` → `Machine that Builds Machines` *only* inside articles/pages content under `src/content/articles/**` and rendered `/articles/` pages. **Do not touch** historical mentions inside `HANDOFF.md`, prior SYNC-HANDOFF.md entries (already SHIPPED), memory files, or git history — those are historical records.
- **Tag/index pages**: any rendered tag-cloud or series-index page that still surfaces "Autoresearch" gets regenerated from the updated frontmatter.
- **Article slugs are unchanged**. `articles/autoresearch-agent-loop` stays — the rename was series-level, not slug-level (per `feedback_article_prose_numbering` memory: prefer slug links over numbers because slugs survive reordering).

After the sweep, open the flip-status PR per §C.2.

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
