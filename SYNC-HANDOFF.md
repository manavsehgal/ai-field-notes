<!--
  ⚠️ STATUS: NEW — content below ships in the head commit on `main` after this release lands.
  At the next release prompt, **clear this entire file and start fresh** (do NOT append to existing sections).
  The file is *one feature/release at a time* — not a running log.
  Last reset: 2026-05-11 (prior content covered the auto-research-loop article publish; shipped through `649c534` and consumed).
-->

# SYNC-HANDOFF — Project-stats methodology change: count original code only (2026-05-11)

Single-file release. The home-page "At a glance" infographic was reporting **369,415 LOC**, which conflated four categories: original article evidence scripts, the fieldkit Python package, the Astro site infrastructure, and (the dominant contributor at ~93%) **vendored Frontier-Scout `articles/<slug>/evidence/repo-snapshot/` upstream-repo dumps** kept for study. This release narrows `total_loc` to **code written for this project only**: article evidence (excluding `/repo-snapshot/`) plus `fieldkit/{src,tests,samples,scripts}/`. The Astro site under `src/` is infrastructure, not the deliverable, and is no longer counted. New `total_loc` is **19,860** (94.6% reduction).

**Source range:** head of `main` (single commit on top of `649c534`).

---

## TL;DR — what shipped

| Path | Change | Notes |
|---|---|---|
| `src/data/project-stats.json` | refreshed via `nvidia-learn-stats` skill with new methodology | Schema: `code.total_loc` 369,415 → 19,860; new keys `code.fieldkit_loc` (6,021) and `code.vendored_loc` (344,711, tracked-but-excluded); removed key `code.src_loc`. `code.evidence_loc` 358,550 → 13,839 (originals only, snapshot dirs filtered). `code.by_language` collapses from 8 buckets to 3 (python 19,832 / sql 15 / shell 13) — Astro/HTML/CSS/TS/JS drop out because they only live in the now-excluded `src/`. |

**No other files in the repo are modified.** No schema, component, plugin, route, CSS, or article edits.

---

## Schema diff (destination action: zero, but read this)

```diff
 "code": {
-  "evidence_loc": 358550,
+  "evidence_loc": 13839,
-  "src_loc": 10865,
+  "fieldkit_loc": 6021,
+  "vendored_loc": 344711,
-  "total_loc": 369415,
+  "total_loc": 19860,
   "by_language": { ... },
   "data_lines": { ... }
 }
```

`ProjectStats.astro` reads `s.code?.total_loc ?? 0` and tolerates missing keys, so **no component change is required on the destination**. The headline KPI re-renders with the new number on the next build.

If the destination keeps a copy of the `nvidia-learn-stats` skill in `~/.claude/skills/`, propagate the methodology change there too (see "Skill change, out of band" below). Otherwise the destination's next ad-hoc `compute_stats.py` run will overwrite `project-stats.json` with the old shape.

---

## Why this matters

The 369K headline was honest arithmetic but misleading storytelling. ~93% of it was upstream code that the Frontier-Scout convention drops into each scouted article's `evidence/repo-snapshot/` directory as background material for the deep-dive — the cxcscmu Auto-Research-Recipes clone alone contributes 189,758 LOC. Reporting that as "code written for field notes" oversold output. The corrected number is closer to the truth: ~14K LOC of article-side scripts (drivers, training loops, eval pipelines, analysis helpers) plus ~6K LOC of fieldkit (the reusable Python package extracted from those articles).

The `vendored_loc` field is kept in the JSON for transparency — anyone curious about the discrepancy with `git ls-files | wc -l` can see the 344K excluded share is accounted for, not lost.

---

## Behavior contract

- **No new dependencies.** Same Astro / Tailwind / remark / rehype versions.
- **No build-cache flush needed.** Schema change is additive + one rename; `.astro/` cache stays valid.
- **No localStorage / reader-state implications.** Pure data file refresh.
- **No new routes.** Page count stays at 61.

---

## Verification

- `npm run build` clean: 61 pages, 5.84 s.
- Home page `dist/index.html` contains the string `19,860` under the `kpi__value` for "Lines of code" — confirmed by grep.
- Skill self-test: `python3 ~/.claude/skills/nvidia-learn-stats/scripts/compute_stats.py` from the repo root prints `code: 19,860 LOC  (evidence 13,839 · fieldkit 6,021 · vendored excluded 344,711)`.

---

## Skill change, out of band

The methodology lives in `~/.claude/skills/nvidia-learn-stats/scripts/compute_stats.py` (not tracked in this repo). Edits in this release:

- `walk_code()` signature gained `exclude_substrings` parameter and a 4-tuple return adding an `excluded_loc` field.
- New module constants: `VENDORED_MARKER = "/repo-snapshot/"`, `FIELDKIT_SUBDIRS = ("src", "tests", "samples", "scripts")`. `SKIP_DIR_NAMES` extended with `.pytest_cache` and `.ruff_cache`.
- `main()` replaces the `src/` LOC walk with a `fieldkit/` walk and gates evidence on the vendored marker.
- `SKILL.md` updated: working-directory path (`nvidia-learn` → `ai-field-notes`), output-shape example, and "When to invoke" entry for fieldkit / Frontier-Scout events.

If the destination CC instance has a clone of this skill, mirror the same edits before the next stats run. If it doesn't, no action needed.

---

## Conflict-avoidance notes

- This release does not touch CSS, components, plugins, or article markdown. Destination's chrome customizations cannot conflict.
- The `project-stats.json` file is *generated* on both sides; the destination's local skill run will reproduce the same numbers as long as the methodology matches.
- No new fields are *required* by `ProjectStats.astro`; the component reads `total_loc` and ignores the rest. Adding the new `fieldkit_loc` and `vendored_loc` keys is safe.

---

## Out of scope (intentionally deferred)

- **Surfacing `fieldkit_loc` and `vendored_loc` in the infographic itself.** The KPI tile still shows a single number ("Lines of code"). Splitting it into "Article code · Package code · (excluded: vendored)" would require a component edit; the data is in the JSON when that visual lands.
- **A "code lineage" companion figure** that breaks down per-article LOC vs vendored — could host on the home page or on `/about/` if a future article needs the meta-thread.
- **Historical project-stats values** — `git log src/data/project-stats.json` already gives a timeline; no archival shim needed.
