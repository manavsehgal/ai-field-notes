<!--
  ✅ STATUS: SHIPPED — Mac swept at destination commit manavsehgal/ainative-business.github.io@495196d on 2026-05-15.
  This file is one feature/release at a time, not a running log.
  At the next release prompt, **clear this entire file and start fresh** (do NOT append to existing sections).
  Last reset: 2026-05-15 (this rotation supersedes the 2026-05-15 cyber-vertical cycle scope, which Mac swept at destination commit manavsehgal/ainative-business.github.io@135bcad / source-merged at 520faa8).

  Prior Mac sweep receipts (preserved here since SYNC-HANDOFF is per-release-not-running-log):
  - 2026-05-15 fieldkit v0.4.2 cycle: swept at destination commit manavsehgal/ainative-business.github.io@495196d (this PR).
  - 2026-05-15 cyber-vertical cycle: swept at destination commit manavsehgal/ainative-business.github.io@135bcad (Mac PR #6 merged 2026-05-15).
  - 2026-05-14 v0.4.1 cycle: swept at destination commit manavsehgal/ainative-business.github.io@e1b16de (Mac PR #5 merged 2026-05-14).
  - 2026-05-14 v0.4.0 cycle: swept at destination commit manavsehgal/ainative-business.github.io@f7ea7aa (Mac PR #4 against this repo — conflicted on rotation; safe to close, receipt captured here).
  - 2026-05-14 Orionfold/finance-chat-GGUF cycle: swept at destination commit manavsehgal/ainative-business.github.io@85f9307 (Mac PR #3, merged).
  - 2026-05-12 Autoresearch→MTBM rename: swept at destination commit manavsehgal/ainative-business.github.io@71293af (Mac PR #2, merged).
-->
---
release_slug: 2026-05-15-fieldkit-v0.4.2
status: SHIPPED
source_range: dd81a29..f23efb3
articles_added: []
articles_updated: []
artifacts_added: []
artifacts_updated: []
fieldkit_modules_changed:
  - publish                                # ArtifactManifest gains recommended_variant field + publish_quant threads it through; ModelCard.llama_cpp_example_prompt already landed on main in ff1b92f
papers_added: []
papers_classify_count: 0
renames_to_replay: []
removes: []
new_top_level_pages: []
breaking_changes: []
destination_overrides_to_preserve: []
hf_repos_added: []
civitai_artifacts_added: []
fieldkit_release: 0.4.2                    # https://pypi.org/project/fieldkit/0.4.2/ ; git tag fieldkit/v0.4.2 on 81aca8f
post_rotation_commits: []                  # rotation happens at end-of-cycle; any post-rotation commits captured in next sweep
---

## Headline

fieldkit v0.4.2 cut, tagged, and shipped to PyPI. Patch release with two additive lifts on `fieldkit.publish` — both driven by the 2026-05-15 cyber-vertical cycle. Zero new modules, zero new public classes, zero breaking changes. The headline: every vertical card that ships through `publish_quant` from now on writes the article's recommended-variant pick into the `<slug>.yaml` manifest automatically, and the hardcoded finance prompt that leaked into Saul + cyber HF cards on first push can no longer leak into vertical 4.

What changed in v0.4.2:

1. **`ModelCard.llama_cpp_example_prompt: Optional[str]`** (already on main in `ff1b92f` since 2026-05-15) — kills the hardcoded `"Explain working capital."` user-message in the default `## How to run` body's `llama-cpp-python` snippet. Renderer now reads `card.llama_cpp_example_prompt` (or a neutral `"Summarize the key idea in one paragraph."` placeholder). Threads through `publish_quant(..., llama_cpp_example_prompt=...)` and from a duck-typed report's `.llama_cpp_example_prompt` attribute. Multi-line MCQ prompts are JSON-escaped (`\n`) so the snippet stays single-line and valid Python.
2. **`ArtifactManifest.recommended_variant: Optional[str]`** (NEW this release) — was already on `ModelCard` (so the README's How-to-run snippets template against the article's pick) but did NOT flow into the `<slug>.yaml` manifest, so the destination catalog couldn't see the article's narrative choice and had to either run its own rank-avg picker or hand-pin (which Mac did for cyber's `Q4_K_M` in PR #6). `publish_quant` now threads `recommended_variant` into both surfaces from one kwarg. Source `src/content.config.ts` adds the matching `recommended_variant: z.string().optional()` so the manifest YAML passes validation.

## What Mac CC sweeps

Straight mirror across the v0.4.2 cut — no destination-side rewrites, no schema changes, no renames. Concrete files:

- **`fieldkit/src/fieldkit/_version.py`** — `0.4.1 → 0.4.2`. The fieldkit landing page (`/fieldkit/`) reads this at build time; its version prop will display `0.4.2` after the next destination build. The `FieldkitCli.astro` demo (and any other section that displays the version) updates automatically — no Mac-side patch needed per the audit-landing rule that hardcoded version literals are not allowed.
- **`fieldkit/CHANGELOG.md`** — new `## [0.4.2] — 2026-05-15` section with the two Added bullets, a **Test suite** sub-section (378 passed, 3 skipped; +3 new tests), an **Articles in this release** sub-section (the cyber article from the prior cycle), and a **Verified on Spark** sub-section (no new HF push, but the v0.4.1 → v0.4.2 source delta was driven by the live cyber push). The `[Unreleased]` heading is now empty (fresh slate for the next cycle).
- **`fieldkit/src/fieldkit/publish/__init__.py`** — `ArtifactManifest` dataclass gains the `recommended_variant: Optional[str] = None` field; `to_dict()` emits it under `recommended_variant:` when set; `publish_quant` passes `recommended_variant=recommended_variant` into the `ArtifactManifest(...)` construction. No public-symbol additions to `__all__`; no breaking changes to existing call sites.
- **`fieldkit/docs/api/publish.md`** — `ArtifactManifest` example block grows a `recommended_variant="Q5_K_M"` line and a new paragraph below the license-block description explaining the v0.4.2 sweet-spot semantics.
- **`fieldkit/tests/test_publish.py`** — three new tests added (`+3 → 378 total`): two on `ArtifactManifest` (round-trip + elision-when-unset) and one on `publish_quant` (kwarg threads to both card and manifest YAML).
- **`src/content.config.ts`** — `artifacts` collection schema gains `recommended_variant: z.string().optional()` so manifest YAML files emit-pass-through cleanly. This mirrors what Mac already added to its own artifacts schema in PR #6 — Mac can either drop the duplicate (now that source emits the field directly) or keep it as defensive depth.

### Auto-refreshed

- **`src/data/project-stats.json`** + **`README.md`** — fieldkit LOC nudge from the +3 new tests + the new `recommended_variant` field. Article count unchanged (cyber was already counted in the prior cycle).

## What Mac CC does NOT need to do

- **No new articles.** The v0.4.2 release ships with no new article — the cyber-vertical article (which drove the lifts) was already swept at `135bcad` in the prior cycle.
- **No new artifacts.** No new HF push this cycle. Existing manifests (`finance-chat-gguf.yaml`, `saul-7b-instruct-v1-gguf.yaml`, `securityllm-gguf.yaml`) do NOT carry the new `recommended_variant` field yet — they were written before v0.4.2 landed. Mac's hand-pin for cyber's `Q4_K_M` in PR #6 remains the source of truth on the destination until the next push runs through v0.4.2; no retro re-emit required. The field is additive + optional.
- **No rename replays.** `SYNC-RENAMES.log` unchanged.
- **No new top-level pages.** No new article, no new product card, no new section.
- **No HF README patches.** The Saul + SecurityLLM HF READMEs were patched in-place on 2026-05-15 (HF commits `365dfe2` + `0824439`) to remove the leaked finance prompt — those are HF-side commits, not source repo commits. Mac doesn't need to do anything; flagging only for visibility.
- **No skill IA mirroring.** `fieldkit-curator` (which drove this release) lives in `~/.claude/skills/` (Spark CC user config), not in the source repo.

## Verification (Spark-side)

- **PyPI:** <https://pypi.org/project/fieldkit/0.4.2/> — wheel + sdist; `twine check PASSED` on both; fresh-venv PyPI install verify ✅ (`fieldkit version → 0.4.2`).
- **Git tag:** `fieldkit/v0.4.2` (annotated, unsigned per convention) on `origin/main` at commit `81aca8f`. Tag pushed cleanly; fresh-venv git-source install verify ✅ (`fieldkit version → 0.4.2`).
- **Audit gates:** `audit-docs` 8/9 PASS, 1 SKIP, 0 FAIL (4 pre-existing kwarg WARNs on `publish` are technical debt from earlier releases, not from this cut; the new `recommended_variant` + `llama_cpp_example_prompt` kwargs are both documented in `docs/api/publish.md`). `audit-landing` 4/4 PASS.
- **Test suite:** 378 passed, 3 skipped offline (`/tmp/fk/bin/pytest tests/ -q`). The 3 skips are the two `--spark`-gated live-integration tests + the `torch`-import skip in `test_training.py` (CPU-only venv).
- **`scripts/verify_article.sh`** — not applicable this cycle (no article changes).

## Release-commit chain (this cycle)

- **`ff1b92f`** — `fieldkit.publish: make llama_cpp example prompt overridable` (pre-rotation; the `ModelCard.llama_cpp_example_prompt` lift from the post-cyber audit. Landed before this release prompt; the v0.4.2 cut bundles it with the new manifest field).
- **`81aca8f`** — `fieldkit v0.4.2: card-rendering polish — overridable llama_cpp prompt + manifest recommended_variant` (the release commit: `_version.py` bump, CHANGELOG finalization, `ArtifactManifest.recommended_variant` field + thread-through, `src/content.config.ts` mirror, +3 tests, docs update).
- **`f23efb3`** — `Refresh stats + README post-fieldkit-v0.4.2` (post-tag stats refresh: `src/data/project-stats.json` + `README.md`).

Three commits this cycle; tag was created on `81aca8f` and pushed before the stats refresh.

## What Mac CC should look for after sweep

- The fieldkit landing page (`/fieldkit/`) should display `0.4.2` everywhere a version is rendered (`FieldkitHero`, `FieldkitCli` demo line, `FieldkitCTAFooter`). The `audit-landing` rule guarantees no hardcoded version literal exists in any `*.astro` file under `src/components/sections/fieldkit/`, so this should update cleanly with the `_version.py` read.
- The `Articles in this release` section on the v0.4.1 → v0.4.2 transition lists `becoming-a-cyber-curator-on-spark` (per CHANGELOG). Mac-side article-listing surfaces should reflect this if/when they index by `fieldkit_modules` × release coupling.
- The `recommended_variant` schema field on `artifacts` is now source-side authoritative. Once a vertical-4 push runs through v0.4.2, its `<slug>.yaml` will arrive with `recommended_variant:` populated — Mac's destination catalog can then render the "Sweet spot" badge from the manifest directly instead of running the rank-avg picker or hand-pinning.
- No new HF repos; the three existing Orionfold cards (`finance-chat-GGUF`, `Saul-7B-Instruct-v1-GGUF`, `SecurityLLM-GGUF`) remain the catalog set.
