# Destination-owned paths (Mac-authoritative — do not clobber from Spark)

> **This file is mirrored from the Mac destination repo.** Spark CC reads it before touching
> root-level paths or introducing new top-level Astro pages, to avoid stomping on
> destination-owned IA.
>
> Mac CC's `sync-field-notes` skill is responsible for keeping this file accurate. When Mac
> adds a new top-level page or override, Mac CC opens a PR back to source updating only this
> file (PR title prefix: `mirror: destination-overrides update — <date> — <summary>`).
>
> **Last reverse-sync: TODO — pending first Mac inventory.** When Mac fills in the TBD
> sections below, replace this line with the inventory date.

## Top-level pages (Mac-authoritative)

These pages live in the Mac destination repo only. Spark CC never adds files under these globs.

- `/book/**` — Ch10–11 MTBM thesis + all book chapters. Mac CC owns end-to-end.
- `/pricing/**` — commercial license tiers for G-cluster artifacts (G1 embedder licenses, G3/G4 paid-tier quants, G6 dataset commercial tier, G8 adapter licenses, G9 LoRA commissions). Mac CC owns.
- `/about/**` — marketing about page; biographical chrome only (articles themselves live on Spark).
- `/` (root landing) — marketing hero. Spark's homepage at `:4321/` is dev preview; production landing on `ainative.business` is Mac-rendered.

## Forthcoming top-level pages (Phase 2; chrome owned by Mac, data from Spark manifests)

These will appear as `fieldkit v0.4` ships artifact-publishing modules. Mac owns page chrome; data comes from `src/content/artifacts/` on Spark.

- `/artifacts/quants/` — GGUF / AWQ / EXL3 / MLX / NVFP4 quant catalog (G3 + G4).
- `/artifacts/loras/` — Civitai-shape image/video LoRAs (G9).
- `/artifacts/adapters/` — LoRA/DoRA/IA3 adapter publisher catalog (G8).
- `/artifacts/embedders/` — niche embedding model catalog (G1) + reranker (G2).
- `/artifacts/datasets/` — synthetic dataset foundry catalog (G6).
- `/artifacts/spaces/` — HF Space app catalog (G10).
- `/artifacts/benchmarks/` — eval benchmark publisher catalog (G11).
- `/skills/**` — cross-vendor SKILL.md catalog (D7 + side-effect distribution), if/when it ships.

## Style overrides

> **TBD — to be filled by Mac's one-time inventory.** Expected scope:
>
> - Design tokens (OKLCH palette, font stack overrides beyond Geist baseline)
> - `astro.config.mjs` production base = `/field-notes/` on ainative.business
> - Custom layouts or component overrides Mac maintains for marketing chrome
> - Any divergence from Spark's `src/components/` (Mac may extend, never edit Spark-owned)

## Build / deploy config (Mac-authoritative)

> **TBD by Mac:**
>
> - Deploy hooks / GitHub Pages workflow (`.github/workflows/`)
> - `_redirects` (or equivalent on whatever host serves ainative.business)
> - Custom 404 page
> - Mac-side `sync-field-notes` skill location + version pin
> - Any CI checks Mac runs on pull beyond Spark's `npm run build`

## Reverse-sync contract

Mac CC opens a PR to source (`manavsehgal/ai-field-notes`) updating only this file when:

1. A new top-level page or page family appears on the Mac side.
2. An existing design override changes in a way that would affect Spark's understanding of "do not touch."
3. The reverse-sync date is updated regardless (heartbeat — at most once per consumption cycle).

PR title prefix: `mirror: destination-overrides update — <date> — <one-line summary>`.
