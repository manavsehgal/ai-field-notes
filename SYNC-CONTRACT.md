# SYNC-CONTRACT — source-of-truth by path

> Mirrored in: `manavsehgal/ainative-business.github.io` (Mac destination repo).
> Last updated: 2026-05-12.

This file declares which side — **Spark** source (`manavsehgal/ai-field-notes`) or **Mac** destination (`manavsehgal/ainative-business.github.io`) — is authoritative for each path glob in the joint ai-field-notes + fieldkit + ainative.business stack. Spark CC reads this before editing; Mac CC's `sync-field-notes` skill reads this before sweeping.

Companion files:

- `SYNC-RENAMES.log` — append-only rename history with replay status (Spark writes, Mac applies).
- `SYNC-HANDOFF.md` — per-release delta (NEW → SHIPPED rotation; new YAML frontmatter schema applies starting next release).
- `mirrors/destination-overrides.md` — destination-owned IA inventory (Mac writes via PR; Spark CC reads to avoid clobbering).

## Authoritative side by glob

| Path glob | Authoritative | Notes |
|---|---|---|
| `articles/**` | **Spark** | All deep-dive content; Mac renders, never edits. PR back to source if Mac wants to typo-fix. |
| `fieldkit/**` | **Spark** | Python package; published to PyPI by the `fieldkit-curator` skill. Mac renders `/fieldkit/` chrome around it. |
| `papers/**` | **Spark** | Frontier-scout reports + `papers.json` sidecar. |
| `evidence/**`, `scripts/**`, `ideas/**` | **Spark** | Working corpus + brainstorms + helper scripts. |
| `src/content.config.ts` | **Spark** | Astro content-collection schema (shared; Mac mirrors). |
| `src/content/articles/**` | **Spark** | Article frontmatter + body. |
| `src/content/artifacts/**` *(Phase 2, when fieldkit v0.4 lands)* | **Spark** | Artifact manifests written by `fieldkit.publish`; Mac renders catalog pages from these. |
| `src/data/project-stats.json` | **Spark** | Refreshed by the `nvidia-learn-stats` skill on every article publish. |
| `src/components/svg/**` | **Spark** | Signature components; `verify_svg.sh` is the gate. |
| `SYNC-HANDOFF.md`, `SYNC-RENAMES.log` | **Spark** writes, **Mac** consumes | Per-release delta + append-only renames. |
| `mirrors/destination-overrides.md` | **Mac** writes via PR, **Spark** consumes | The one Mac → Spark reverse channel. |
| `/book/**` | **Mac** | Ch10–11 MTBM thesis, all book chapters. Never touched from Spark. |
| `/pricing/**`, `/about/**`, root landing | **Mac** | Marketing IA. Spark never edits. |
| `/artifacts/<kind>/*.astro` (page chrome) *(Phase 2)* | **Mac** | Catalog page templates; data comes from `src/content/artifacts/`. |
| `/skills/**` *(Phase 2, cross-vendor SKILL.md catalog)* | **Mac** | If/when this surface lands. |
| Mac CC skill scripts (e.g. `sync-field-notes`) | **Mac** | Lives in Mac's `.claude/skills/`, never in source repo. |

## Merge zones

Today none. `src/data/project-stats.json` is Spark-authoritative for *production* numbers; Mac may override values locally for previews but should not commit overrides without a PR back to source.

## Forward schema — `SYNC-HANDOFF.md` YAML frontmatter (effective next release)

The current `SYNC-HANDOFF.md` (2026-05-11 a2tgpo release) retains its existing prose-only format. Starting the **next** release after this contract lands, `SYNC-HANDOFF.md` gets a YAML frontmatter block:

```yaml
---
release_slug: <yyyy-mm-dd>-<short-name>
status: NEW              # NEW → SHIPPED after Mac sweeps
source_range: <commit>..HEAD
articles_added: []
articles_updated: []
artifacts_added: []      # Phase 2 onwards: slugs from src/content/artifacts/
artifacts_updated: []
fieldkit_modules_changed: []
renames_to_replay: []    # ids from SYNC-RENAMES.log to apply this release
removes: []              # paths Mac should delete
new_top_level_pages: []  # new /artifacts/<kind>/ IA, /skills/, etc.
breaking_changes: []
destination_overrides_to_preserve: []
hf_repos_added: []
civitai_artifacts_added: []
---
```

Mac CC's `sync-field-notes` parses the frontmatter mechanically; prose below stays human-readable for the next-specialist read.

## Sequencing in three phases

1. **Phase 1 (this commit)** — Contract hardening: this file + `SYNC-RENAMES.log` + `mirrors/destination-overrides.md` placeholder.
2. **Phase 2 (during fieldkit v0.4)** — Artifact manifests in `src/content/artifacts/`, written by `fieldkit.publish`, rendered by Mac's catalog templates. Schema lands in `src/content.config.ts` (new `artifacts` collection). New gate: `scripts/verify_artifact.sh`.
3. **Phase 3 (deferred)** — Shared-package carve-out reconsidered when ≥ 5 manifests exist and Mac's catalog renderer has real shape.

Full Spark-side plan: `/home/nvidia/.claude/plans/this-repo-is-source-cached-sky.md`.
