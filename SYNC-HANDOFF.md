<!--
  ⚠️ STATUS: NEW — not yet consumed.
  This file is one feature/release at a time, not a running log.
  At the next release prompt, **clear this entire file and start fresh** (do NOT append to existing sections).
  Last reset: 2026-05-14 (prior content covered the a2tgpo article publish + sync-contract bootstrap pull cycle, consumed by Mac CC on 2026-05-12 — sweep commit `manavsehgal/ainative-business.github.io@71293af`).
-->
---
release_slug: 2026-05-14-orionfold-finance-chat-gguf
status: NEW
source_range: f32cd1f..HEAD
articles_added: []
articles_updated:
  - becoming-a-gguf-publisher-on-spark   # promoted from upcoming → published; customer-link audit applied; new fn-diagram + signature SVG
artifacts_added:
  - finance-chat-gguf                    # first Phase-2 src/content/artifacts/ manifest — Orionfold/finance-chat-GGUF
artifacts_updated: []
fieldkit_modules_changed:
  - publish                              # ORIONFOLD_HF_ORG → ORIONFOLD_HF_HANDLE rename; value orionfoldllc → Orionfold; back-compat alias kept
renames_to_replay:
  - 2026-05-14-orionfoldllc-to-orionfold # see SYNC-RENAMES.log entry below
removes: []
new_top_level_pages:
  - /artifacts/quant/                    # Phase 2 catalog kicks in — first manifest of `kind: quant` landed
breaking_changes: []
destination_overrides_to_preserve: []
hf_repos_added:
  - Orionfold/finance-chat-GGUF          # staged dry-run only; HF push pending — Mac side: no action required on push, but the article links to this URL
civitai_artifacts_added: []
---

# SYNC-HANDOFF — Orionfold/finance-chat-GGUF first quant + customer-link audit (2026-05-14)

This release lands two coupled changes:

1. **First Orionfold quant release** (`Orionfold/finance-chat-GGUF`) — five GGUF variants of AdaptLLM/finance-chat measured end-to-end on a DGX Spark. Article at [`becoming-a-gguf-publisher-on-spark`](articles/becoming-a-gguf-publisher-on-spark/) promotes from `status: upcoming` to `status: published` and now serves as the HF model card's "Methods" link. The companion Phase-2 artifact manifest at `src/content/artifacts/finance-chat-gguf.yaml` is the first real-world `kind: quant` payload — Mac side: this is when the `/artifacts/quant/` catalog page renders for the first time.
2. **Brand-handle rename** — HF publishing surface moved from org `orionfoldllc` to user handle `Orionfold` (Bartowski-precedent). `fieldkit.publish.ORIONFOLD_HF_ORG` → `ORIONFOLD_HF_HANDLE`; old constant lingers as a back-compat alias. Mac-side action: replay the rename across prose mentions (see SYNC-RENAMES.log entry, below).

The article was also surgically de-leaked for customer-link audit (per new memory `feedback_customer_link_audit`) — it's now reader-facing-first while preserving the Manav-voice deep-dive. The tech-writer skill grew a new section in `references/voice-and-style.md` ("When the article is linked from a product card") plus updates to `references/article-structure.md` Sections 7 + 8 capturing the four audit failure modes.

## Mac CC action bundle

### A. Memory entry to save

| Memory | Type | Slug | Body |
|---|---|---|---|
| Customer-link article audit | feedback | `feedback_customer_link_audit` | Articles linked from public product cards (HF README's Methods line, Civitai resource pages) need a customer-POV audit before publish — strip strategy leak, competitor punches, failure-narrative front-loading, roadmap detail. Voice stays Manav-deep-dive everywhere else. Patterns to look for: "the seat I want", "the moat", named competitor dismissals, HANDOFF Q-references, "next article in series" roadmap. Failure modes documented in tech-writer skill `references/voice-and-style.md` ("When the article is linked from a product card") + `references/article-structure.md` Sections 7+8. Optional frontmatter field `customer_linked: true` is the future tooling hook for a verifier pass. Lesson learned 2026-05-14 on the Orionfold/finance-chat-GGUF release where the HF Methods link initially pointed at a personal-journey deep-dive that admitted V1 failure in paragraph three. |

### B. Rename replay

Apply across destination prose + frontmatter + tags. Both casings of the old name occur in the wild (`orionfoldllc` in URLs, `Orionfoldllc` rare).

```yaml
- date: 2026-05-14
  kind: brand
  old: orionfoldllc
  new: Orionfold
  affected: [prose_mentions, hf_repo_urls, frontmatter_product_field, model_card_attribution]
  status: destination-needs-replay
  notes: |
    Bartowski-precedent personal HF handle. The org orionfoldllc was a fallback
    registered 2026-05-13 when the short slug `orionfold` was unavailable; the
    user account `Orionfold` already held it. Switched 2026-05-14 in favor of
    publishing under the user handle directly — simpler surface for a one-person
    publisher, cleaner repo URL. The LLC parent brand survives off-HF (legal
    entity, billing, footer attribution); the HF surface is the user.
```

Mac side: edit `SYNC-RENAMES.log` to flip this entry's status to `complete` after the sweep finishes (or PR back to source).

### C. Phase-2 artifact catalog kickoff

This release ships the first real `src/content/artifacts/<slug>.yaml` manifest — `finance-chat-gguf.yaml`. The Astro `artifacts` collection schema landed in `src/content.config.ts` with v0.4. Mac side: time to scaffold the `/artifacts/<kind>/` catalog templates (per Phase-2 plan in `SYNC-CONTRACT.md` and memory `project_artifact_manifests_phase2`):

- `/artifacts/quant/` index page — list all artifacts with `kind: quant`, show slug + base_model + variant count + license tier + linked article.
- `/artifacts/quant/[slug]/` detail page — render the full manifest (variants, perplexity, spark_tokens_per_sec, vertical_eval, sustained_load_minutes) as a four-axis card mirroring the HF card layout.

The manifest fields and shape are stable — see `fieldkit/src/fieldkit/publish/__init__.py` `ArtifactManifest` dataclass for the source of truth.

## What landed

### `becoming-a-gguf-publisher-on-spark` (promoted)

- Frontmatter: `status: upcoming` → `status: published`, `signature: VerticalCuratorRetry` (new component at `src/components/svg/VerticalCuratorRetry.astro`), title rewritten, summary rewritten.
- Body: 3,388 words (down from 4,351 — customer-link audit cut ~22%). 6 explainers (1 define + 2 why + 1 pitfall + 1 hardware + 1 deeper). One inline fn-diagram (5-stage release pipeline) replacing the prior V1-vs-V1-retry dual-path comparison. New *Using this release* section with variant picker + run snippets + vs-origin comparison.
- Evidence: `articles/becoming-a-gguf-publisher-on-spark/evidence/lineage-finance-chat/results.tsv` is the live release lineage (baseline + 5 variant rows). `evidence/lineage/results.tsv` is the V1 audit-trail (preserved at user instruction).

### `fieldkit` v0.4 surface

The `fieldkit.publish` module's constant rename + the first `publish_quant()` end-to-end dry run that emitted a real Phase-2 artifact manifest. Full fieldkit suite: 368 passed / 2 skipped.

### Scripts under `scripts/`

- `g3_preflight_bench.py` (new) — V0 preflight gate.
- `g3_build_first_quant.sh` (modified) — wired in the preflight step; defaults repointed to AdaptLLM/finance-chat.
- `g3_measure_variants.py` (modified) — open-book FinanceBench loader; `[INST]` prompt wrapping; per-attempt lineage directory.

## What didn't land (deferred to next release)

- **HF push** — `publish_quant(..., dry_run=False)` for `Orionfold/finance-chat-GGUF` is gated on two `fieldkit.publish.ModelCard` bug fixes (hardcoded `license: apache-2.0` should pull from manifest; empty "How to run" section body needs the template port from the article's *Using this release* section). User has not flipped the flag.
- **`/artifacts/quant/` catalog scaffold** — Mac-side action above; Spark side has nothing to add until Mac CC opens a PR proposing the page chrome.
- **`fieldkit.eval.VerticalBench.from_jsonl(..., open_book=True)` upstream** — the inline `_load_finbench_open_book` in `scripts/g3_preflight_bench.py` + `scripts/g3_measure_variants.py` should lift into the package; deferred to a fieldkit point cut.
