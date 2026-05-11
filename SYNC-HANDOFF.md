<!--
  ⚠️ STATUS: NEW — content below ships in the head commit on `main` after this release lands.
  At the next release prompt, **clear this entire file and start fresh** (do NOT append to existing sections).
  The file is *one feature/release at a time* — not a running log.
  Last reset: 2026-05-11 (prior content covered the auto-research-loop article publish; shipped through `649c534` and consumed).
-->

# SYNC-HANDOFF — fieldkit v0.3.0 release + project-stats methodology change + LOC tile breakdown (2026-05-11)

Three changes bundled into one sync window — all three commits sit on top of `649c534` and represent a single editorial unit (the v0.3.0 release of fieldkit, plus the supporting stats methodology change and LOC-tile UI surface that landed at the same time as v0.2.0.post1's stats refresh).

The lead change for this sync window is **fieldkit v0.3.0 on PyPI**, which introduces the `fieldkit.lineage` module (PyPI <https://pypi.org/project/fieldkit/0.3.0/>, git tag `fieldkit/v0.3.0`, commit `53fe370`). It's the v0.3 candidate sketched in the `auto-research-loop-on-spark` article — the portable part of cxcscmu's *Auto-Research-Recipes* harness, decomposed into a pure-stdlib substrate.

The two supporting changes are the project-stats methodology rewrite (commit `87a2ebe`) that narrowed `total_loc` from 369,415 to 19,860 by excluding the vendored Frontier-Scout repo snapshots and the Astro site infrastructure, and the LOC-tile UI surface (commit `f53a9dc`) that added the `14k article · 7k package` sub-line beneath the headline. After v0.3.0's stats refresh (commit `ab125e5`), the sub-line now reads `14k article · 7k package` (was `14k article · 6k package` pre-v0.3, since the lineage module added +950 LOC to `code.fieldkit_loc`).

**Source range:** head of `main` — four commits on top of `649c534`: `87a2ebe` (stats methodology) · `f53a9dc` (LOC tile breakdown UI) · `53fe370` (fieldkit v0.3.0 release) · `ab125e5` (stats refresh post-release).

---

## TL;DR — what shipped

| Path | Change | Notes |
|---|---|---|
| `fieldkit/src/fieldkit/lineage/__init__.py` | **NEW** — pure-stdlib module, ~330 LOC | Public surface: `FailureLabel` (10-class string enum + `is_informational`), `Trial` (frozen 17-field dataclass with TSV round-trip), `LineageStore` (fcntl-locked append + `latest`/`best`/`chain_to` + `render_prompt`), `LineageSnapshot`, `RecipeEdit` (snapshot + lazy unified-diff vs parent). |
| `fieldkit/tests/test_lineage.py` | **NEW** — 29 tests | FailureLabel value parity + 10-class surface lock + `is_informational` predicate; Trial round-trip via TSV; LineageStore append / latest / best (lower- and higher-is-better, empty store, no-metric fallback) / chain_to (linear + branched + missing-id raises); render_prompt (empty store, specialist name, top-K filters to keeps only, chain has `← BEST`, recent_n caps, determinism); RecipeEdit.diff (empty baseline, vs parent, new-file detection); LineageSnapshot frozen contract. |
| `fieldkit/samples/hello-lineage.py` | **NEW** — 5-trial worked example | Baseline + 2 keeps + discard + eval_budget_overrun. Prints the rendered prompt the next specialist sees at session entry. |
| `fieldkit/docs/api/lineage.md` | **NEW** — module reference page | Renders at `/fieldkit/lineage/` (Astro slot under fieldkit docs site). Order: 6, after training. |
| `fieldkit/src/fieldkit/_version.py` | `0.2.0.post1` → `0.3.0` | Bump per Decision M (semver, v0.x permits minor breaks; this is purely additive). |
| `fieldkit/CHANGELOG.md` | New `## [0.3.0] — 2026-05-11` section | Lists the new module, the 29-test addition, the offline test count (249 passed, 3 skipped), the `auto-research-loop-on-spark` article as the anchor, and the `FIELDKIT_MODULES` schema bump. |
| `src/content.config.ts` | `FIELDKIT_MODULES` extended with `'lineage'` | New order: `capabilities, nim, rag, eval, training, lineage, cli`. Required so articles can declare `fieldkit_modules: ['lineage']` in frontmatter. |
| `articles/auto-research-loop-on-spark/article.md` | frontmatter `fieldkit_modules` | `[capabilities, training]` → `[capabilities, training, lineage]`. |
| `src/data/project-stats.json` | refreshed via `nvidia-learn-stats` skill | Schema: `code.total_loc` 19,860 → **20,810**; `code.fieldkit_loc` 6,021 → **6,971** (+950 from lineage module + tests); `code.evidence_loc` unchanged at 13,839; `code.vendored_loc` unchanged at 344,711 (tracked-but-excluded). Earlier this sync window: `code.total_loc` 369,415 → 19,860 (methodology change) and a new `code.fieldkit_loc` / `code.vendored_loc` schema (was `code.src_loc`). |
| `README.md` | regenerated via `tech-writer` `refresh_readme.py` | Picks up the new LOC numbers + 37 articles surface (33 published + 4 upcoming placeholders). |
| `src/components/ProjectStats.astro` | conditional LOC-tile breakdown line | Three new const reads (`LOC_EVIDENCE` / `LOC_FIELDKIT` / `LOC_VENDORED`), a `kfmt(n)` helper, and one new `<div class="kpi__breakdown">` rendered only on the LOC tile (between the value and label), plus `title={locTitle}` tooltip on the tile's `<article>`. (From the earlier `f53a9dc` commit; no further changes in v0.3.) |
| `src/styles/global.css` | 10-line `.kpi__breakdown` rule | Mono 0.6rem, dim color, 0.04em tracking, line-height 1.25, `text-overflow: ellipsis`. (From the earlier `f53a9dc` commit; no further changes in v0.3.) |

**Distribution side:** `fieldkit 0.3.0` published to PyPI at <https://pypi.org/project/fieldkit/0.3.0/> (wheel 71.6 kB + sdist 118.0 kB; both `twine check` PASSED before upload). Git tag `fieldkit/v0.3.0` pushed to `origin/main`. Both install-verify gates (git source + PyPI source) PASSED in fresh venvs.

---

## Schema diff (destination action: validate `FIELDKIT_MODULES`)

The `FIELDKIT_MODULES` Zod enum used by `src/content.config.ts` widened from 6 → 7 values. Articles previously valid with `fieldkit_modules: [capabilities, training]` are still valid; the new `lineage` value is the only addition. Both publish-time `astro check` and `astro build` pass cleanly in the source repo (62 pages built, 4.51 s).

```diff
-export const FIELDKIT_MODULES = ['capabilities', 'nim', 'rag', 'eval', 'training', 'cli'] as const;
+export const FIELDKIT_MODULES = ['capabilities', 'nim', 'rag', 'eval', 'training', 'lineage', 'cli'] as const;
```

`project-stats.json` schema added these keys earlier in this sync window:

```diff
- "src_loc": <number>,
+ "fieldkit_loc": <number>,
+ "vendored_loc": <number>,        # tracked but excluded from total_loc
+ "excluded_loc": <number>,        # alias of vendored_loc for clarity
```

`ProjectStats.astro` reads via optional-chain (`s.code?.total_loc ?? 0`), tolerates the schema change without component-level action.

---

## Why this matters

**For fieldkit v0.3.0**, the case is in cxcscmu's own ablation runs. Same agent, same prompt template, same 201 trials of search on Parameter Golf, same Claude Opus on each specialist. The *only* difference between `pg_ablation_lineage_on` and `pg_ablation_lineage_off` is whether the agent's session prompt includes the rendered lineage block. With lineage on: 16 keeps (8.0%), 38 eval-budget overruns (19%), best `val_bpb` 1.073142. Without: 3 keeps (1.5%), **123 eval-budget overruns (61%)**, best `val_bpb` 1.077413. **5.3× more keeps · 3.2× fewer wall-wastes**, with no model change, no compute change, no prompt-template change. The lineage primitive is the portable part of that result — pure-stdlib Python, runs anywhere.

The `auto-research-loop-on-spark` article is the editorial anchor; v0.3 is the substrate that backs every future Machine-that-Builds-Machines article's trajectory artifact. The next MTBM article (`a2tgpo-turn-clipping-on-spark`, Frontier Scout Priority 2) will be the first to write directly to `fieldkit.lineage.LineageStore` from the start.

**For the project-stats methodology change**, the prior headline of 369,415 LOC was misleading: ~93% of it was vendored Frontier-Scout `articles/<slug>/evidence/repo-snapshot/` upstream-repo dumps kept for study (cxcscmu Auto-Research-Recipes alone = 189,758 LOC), plus ~3% Astro site infrastructure. Neither is code written for this project. The new methodology counts only article evidence (excluding `/repo-snapshot/`) and `fieldkit/{src,tests,samples,scripts}/`. The 94.6% reduction is the honest number.

---

## Behavior contract

- `fieldkit.lineage` imports are zero-cost in environments that don't use it (pure stdlib).
- `LineageStore.append` is concurrency-safe via `fcntl.flock`: multiple specialists can write to the same `results.tsv` without interleaving.
- `Trial.to_row()` ↔ `Trial.from_row(dict)` are byte-stable identity round-trip; `None` floats serialize as empty strings (matches cxcscmu TSV convention).
- `LineageStore.render_prompt(...)` is deterministic — same TSV state + same parameters → byte-identical Markdown output. Tested explicitly.
- `FailureLabel.value` is verbatim-compatible with cxcscmu TSV `status` columns; existing `results.tsv` files from `release_artifacts/` parse without modification.

---

## Verification

- `pytest tests/` → **249 passed, 3 skipped** offline (1 module-level torch `importorskip` in `test_training.py`, 2 `--spark`-gated live integration tests).
- Git-source install verify (`/tmp/fk030`): `pip install git+...@fieldkit/v0.3.0#subdirectory=fieldkit` → `fieldkit version` → `0.3.0`; `fieldkit.lineage` imports cleanly; `Trial.header()` has 17 fields; `FailureLabel` has 10 classes.
- PyPI build: `python -m build` → `fieldkit-0.3.0-py3-none-any.whl` (71.6 kB) + `fieldkit-0.3.0.tar.gz` (118.0 kB). `twine check dist/*` → both PASSED.
- PyPI upload: live at <https://pypi.org/project/fieldkit/0.3.0/>.
- PyPI install verify (`/tmp/fk-pypi`): `pip install --no-cache-dir fieldkit==0.3.0` → `fieldkit version` → `0.3.0`; LineageStore round-trip OK.
- `npm run build` clean — 62 pages, 4.51 s. New `/fieldkit/lineage/` page builds, LOC tile reads `14k article · 7k package`.

---

## Skill change, out of band

The `nvidia-learn-stats` skill at `~/.claude/skills/nvidia-learn-stats/` was updated earlier this sync window (commit `87a2ebe`) — out-of-repo, so propagate manually if the destination CC instance keeps a copy:

- `walk_code()` gained an `exclude_substrings` parameter and an `excluded_loc` return.
- New constants `VENDORED_MARKER = "/repo-snapshot/"` and `FIELDKIT_SUBDIRS = ("src","tests","samples","scripts")`.
- `SKIP_DIR_NAMES` extended with `.pytest_cache` + `.ruff_cache`.
- `main()` swapped its `src/` walk for a `fieldkit/` walk.
- `SKILL.md` updated paths (`nvidia-learn` → `ai-field-notes`) + output-shape example + "when to invoke" entry.

No further skill changes in v0.3.

---

## LOC tile breakdown (UI surface)

The home-page "At a glance" infographic's "Lines of code" tile now reads `20,810` (headline) with `14k article · 7k package` as a dim mono sub-line, and a `title` tooltip with the full breakdown: `13,839 article evidence + 6,971 fieldkit package = 20,810 original. 344,711 vendored repo-snapshot LOC tracked but excluded.`

The component-level code is from commit `f53a9dc` (additive — three const reads, a `kfmt` helper, one conditional `<div class="kpi__breakdown">` between value and label, `title={locTitle}` on the tile's `<article>`, plus a 10-line `.kpi__breakdown` CSS rule). v0.3.0's stats refresh changes only the displayed numbers, not the component logic.

---

## Conflict-avoidance notes

- The `FIELDKIT_MODULES` enum is the only schema widening. No removals, no renames. Existing articles with no `lineage` reference are unaffected.
- The new `fieldkit/docs/api/lineage.md` file lands a new route at `/fieldkit/lineage/`. No existing routes change.
- The `articles/auto-research-loop-on-spark/article.md` frontmatter change is the single character-level edit to an existing article: `[capabilities, training]` → `[capabilities, training, lineage]`. The article body is unchanged.
- `project-stats.json` schema changes are tolerated by `ProjectStats.astro` via optional-chaining; no destination-side component edits required.

---

## Out of scope (intentionally deferred)

- **Wave-2 retrospective on prior MTBM articles** — `autoresearch-agent-loop`, `trajectory-eval-is-the-agent-flailing`, `clawgym-on-spark-grpo`, `t2po-uncertainty-guided-rl-on-spark` all produced lineage-shaped artifacts in ad-hoc formats. Now that `fieldkit.lineage.LineageStore` ships, a future pass could rewrite their `evidence/` to use the canonical primitive. Not blocking the next article.
- **Live Spark reproduction of cxcscmu Parameter Golf** — working scaffold ready at `/home/nvidia/work/auto-research/` (out-of-repo). Hold for DGX Cloud time or a weekend window where the box can be dedicated; 3–8 days continuous run is the projected wall.
- **Companion infographic figure for code lineage** (per-article original vs vendored) — data is in `project-stats.json` already (`evidence_loc`, `fieldkit_loc`, `vendored_loc`); only the visual is missing. Could host on `/about/` if a future meta-article wants it.
- **Splitting the LOC tile further** — current breakdown is `14k article · 7k package`. Could add a `· (344k vendored excluded)` segment but it would crowd the tile and the tooltip carries that data already.
