<!--
  ✅ STATUS: SHIPPED — Mac sweep complete.
  This file is one feature/release at a time, not a running log.
  At the next release prompt, **clear this entire file and start fresh** (do NOT append to existing sections).
  Last reset: 2026-05-14 (prior content covered the Orionfold/finance-chat-GGUF release + customer-link audit + ORIONFOLD_HF_ORG → HANDLE brand rename, consumed by Mac CC on 2026-05-14 — sweep commit `manavsehgal/ainative-business.github.io@85f9307`).
  Current cycle consumed by Mac CC on 2026-05-14 — sweep commit `manavsehgal/ainative-business.github.io@f7ea7aa`.
-->
---
release_slug: 2026-05-14-fieldkit-v0.4.0
status: SHIPPED
swept_at: 2026-05-14
swept_by: manavsehgal/ainative-business.github.io@f7ea7aa
source_range: e322af2..HEAD
articles_added: []
articles_updated:
  - becoming-a-gguf-publisher-on-spark   # frontmatter gains `hf_url: https://huggingface.co/Orionfold/finance-chat-GGUF` — first article to use the new optional field
artifacts_added: []
artifacts_updated: []
fieldkit_modules_changed:
  - publish                              # license/chat_format/recommended_variant/hf_repo plumbing + auto-rendered ## How to run (v0.4.x fix); first public release of the module
  - quant                                # first public release of the module
  - eval                                 # v0.4.x VerticalBench + VerticalQA + exact_match/contains/numeric_match scorers
  - capabilities                         # docs gain Supporting types subsection (Hardware, MemoryBudgetRulesOfThumb, StackEntry — already public, prior docs page didn't mention them)
  - nim                                  # docs gain ChatMessage type-alias section
  - rag                                  # docs gain Tunable constants table (DEFAULT_EMBED_BATCH, CHUNKS_PER_DOC_MAX, DEFAULT_RERANK_MODEL)
  - cli                                  # docs `order:` 7 → 9 (was colliding with the new quant.md at canonical order 7); example version 0.2.0 → 0.4.0
renames_to_replay: []
removes: []
new_top_level_pages: []
breaking_changes: []                     # v0.x SemVer; v0.4 adds modules + extends eval surface, no public removals
destination_overrides_to_preserve: []
hf_repos_added: []                       # finance-chat-GGUF shipped in the prior release window; nothing new on HF this cycle
civitai_artifacts_added: []
---

## Headline

`fieldkit v0.4.0` shipped to PyPI at <https://pypi.org/project/fieldkit/0.4.0/> with the git tag `fieldkit/v0.4.0` on `origin/main` (commit `d9efb2f`). Two new top-level modules (`fieldkit.publish` + `fieldkit.quant`) plus the v0.4.x `fieldkit.eval.VerticalBench` overlay — the same surface that produced the `Orionfold/finance-chat-GGUF` card live-pushed in the previous sync window. Pre-release work also closed four `audit-docs` drift gaps in pre-existing modules and added an optional `hf_url:` field to the articles content-collection schema.

A second commit-cluster fixed the Spark-side fieldkit landing page (`/fieldkit/`) — the stats / modules / CLI demo sections had hardcoded values that drifted with each release. They now derive from `FIELDKIT_MODULES` (`src/content.config.ts`) and `__version__` (`fieldkit/src/fieldkit/_version.py`); a new sibling audit script in the `fieldkit-curator` skill (`audit_landing.py`) gates the drift before tag-push. The Mac mirror, which renders `/fieldkit/` from the same `src/components/sections/fieldkit/*.astro` + `src/pages/fieldkit/index.astro`, will pick up the dynamic behavior automatically once the sweep lands.

## What Mac CC sweeps

The Mac sweep is a straight mirror — no destination-side rewrites needed this cycle. Concrete files / paths:

- **`fieldkit/docs/api/*.md`** — six pages touched: `publish.md` (new, 1,279 words), `quant.md` (new, 1,090 words), `capabilities.md` / `nim.md` / `rag.md` / `eval.md` (drift fixes), and `cli.md` (frontmatter `order: 7 → 9` + example version bump). The Astro nav for `/fieldkit/api/<module>/` sorts by `order:`; the cli.md collision was the v0.4 trap — without the bump, `cli` would silently swap card position with `quant` on the modules grid.
- **`fieldkit/CHANGELOG.md`** — `[Unreleased]` → `[0.4.0] — 2026-05-14`, with four bundled storylines (publish + quant modules, VerticalBench overlay, model_license + How-to-run defaults fix, the live HF-push verification). Test count corrected from the prior session's stale `356/3` to actual `379/2`. New **Verified on Spark** sub-section.
- **`fieldkit/src/fieldkit/_version.py`** — `0.3.0 → 0.4.0`. `FieldkitHero`, `FieldkitCli`, and `FieldkitCTAFooter` all read this at build time on both sides.
- **`src/content.config.ts`** — `articles` schema gains optional `hf_url: z.string().url().optional()`. Backwards-compatible; existing articles render identically.
- **`src/components/sections/fieldkit/FieldkitProblem.astro`** — stat value + module-list source now derive from `FIELDKIT_MODULES.length` / `FIELDKIT_MODULES.join(', ')`. Source string gains spaces after commas + `break-words` + `leading-snug` + `text-[11px]` so the 9-module list wraps cleanly inside the 1/3-grid card at every viewport.
- **`src/components/sections/fieldkit/FieldkitModules.astro`** — headline "fieldkit in N imports" now reads `docs.length` via a number-word map. Tagline map gains entries for `quant` and `publish`.
- **`src/components/sections/fieldkit/FieldkitCli.astro`** — accepts a `version` prop; CLI demo output reads it instead of hardcoding `0.2.0`. `src/pages/fieldkit/index.astro` threads `fieldkitVersion` through.
- **`articles/becoming-a-gguf-publisher-on-spark/article.md`** — frontmatter gains `hf_url: https://huggingface.co/Orionfold/finance-chat-GGUF`. The article body is unchanged.
- **`src/data/project-stats.json`** + **`README.md`** — auto-refreshed by `nvidia-learn-stats` and `tech-writer/refresh_readme.py` post-release. New numbers: 35 articles, 120,093 words, 24,026 LOC (was 23,728 — fieldkit gained the `publish` + `quant` modules + tests).

## What Mac CC does NOT need to do

- **No destination-prose rewrites.** The `hf_url:` field is optional; only one article uses it. The page templates can read it where useful or ignore it.
- **No HF-repo replays.** The `Orionfold/finance-chat-GGUF` push landed in the prior sync window and was swept at `85f9307`. Nothing new on HF this cycle.
- **No rename replays.** `SYNC-RENAMES.log` is fully `complete` after the prior `orionfoldllc → Orionfold` sweep. No entries flipped to `destination-needs-replay`.
- **No Phase-2 artifact-manifest catalog work.** `src/content/artifacts/` got no new entries (`finance-chat-gguf.yaml` from the prior window is the only one); the catalog page templates remain Mac-side editorial.
- **No skill mirroring.** The new `audit_landing.py` lives in `~/.claude/skills/fieldkit-curator/scripts/` (Spark CC user config), not in the source repo. The Phase 2 cross-vendor `/skills/` IA is still deferred; nothing for Mac to render.

## Spark-side gates that ran

- `fieldkit-curator audit-docs` — 8/9 PASS, 1 skip (`cli` has no explicit `__all__`).
- `fieldkit-curator audit-landing` — 4/4 PASS (module_count_dynamic, no_hardcoded_versions, module_taglines, docs_order_matches_modules). This is the new gate; it caught the four drift points above + the cli.md `order:` collision before tagging.
- `pytest tests/` — 379 passed, 2 skipped offline (the 2 skips are `--spark`-gated live integration tests for `fieldkit.nim` + `fieldkit.rag`). No `--spark` paths touched in v0.4.0; live re-run deferred to a release that needs it.
- `python -m build` + `twine check` — both wheels clean; PyPI upload succeeded.
- Fresh-venv install verifies: ✅ from git tag, ✅ from PyPI (one ~60s CDN-propagation retry needed before the PyPI mirror caught up).

## Why a new `audit-landing` gate

The v0.4.0 release process surfaced a class of drift that nothing in `astro build` or `pytest` catches: hardcoded numbers and version strings in landing-page copy. The pre-release page kept saying "**7** modules / fieldkit in **seven** imports / `fieldkit.{capabilities,nim,rag,eval,training,lineage,cli}`" after `quant` + `publish` had already shipped — because the page copy is plain TSX literals, not derived. The new `audit_landing.py` codifies the four checks (`module_count_dynamic`, `no_hardcoded_versions`, `module_taglines`, `docs_order_matches_modules`) and the `fieldkit-curator release` flow now hard-stops on FAIL at step 2c (parallel to the existing 2b `audit-docs` gate). Source: `/home/nvidia/.claude/skills/fieldkit-curator/SKILL.md` "Mode: audit-landing".

Mac side has no parallel gate today; if Mac develops its own landing-page chrome that displays the same dynamic surface (module count, version, module list), the equivalent guardrail belongs in the `sync-field-notes` skill.

## Source range

`e322af2..2190824` — five commits since the prior Mac sweep:

```
2190824  fix(fieldkit landing): unstale module count, version, taglines, cli order
f8775ea  Refresh stats + README post-fieldkit-v0.4.0
d9efb2f  fieldkit v0.4.0: publish + quant modules, VerticalBench overlay, first live HF push
86bde6d  docs(fieldkit): close audit-docs gaps + article hf_url — v0.4.0 release prep
ab6e385  fix(fieldkit.publish): model_license plumbing + auto-rendered ## How to run defaults
```

Note: `ab6e385` predates this session and was the model_license + How-to-run fix that landed alongside the Orionfold push, but it shipped *after* the previous SYNC-HANDOFF was written and consumed. It rides along here for the Mac sweep's benefit; the diff is small and self-contained.
