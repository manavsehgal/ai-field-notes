# Changelog

All notable changes to `fieldkit` are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). While the package is on `0.x`, minor versions may include breaking changes. `1.0` will mark API stability.

## [Unreleased]

### Added

- `fieldkit.capabilities` module: typed Python facade over `spark-capabilities.json` exposing `Capabilities.load()` (cached singleton with `.hardware`, `.memory_budget_rules_of_thumb`, `.stack`, `.in_envelope_signals`, `.out_of_envelope_signals`, `.stage_routing_hints`, `.series_routing_hints`), plus the canonical math helpers `kv_cache_bytes()`, `weight_bytes()`, and `practical_inference_envelope()`. Pinned to the math from `kv-cache-arithmetic-at-inference` and `gpu-sizing-math-for-fine-tuning`. ([#capabilities])
- `samples/feasibility-math.py` — reproduces the kv-cache article's serving table via the API.
- `scripts/sync_capabilities.py` — keeps the package-bundled `spark-capabilities.json` in sync with the source-of-truth at `scripts/lib/spark-capabilities.json` (pre-commit-enforced).

### Changed

- `frontier-scout` skill (`refresh` and `eval` modes, plus `references/feasibility-prompt.md`) now teaches the typed `from fieldkit.capabilities import …` API as the preferred grounding path; raw JSON read is the documented fallback.

[#capabilities]: https://github.com/manavsehgal/ai-field-notes/tree/main/fieldkit/src/fieldkit/capabilities

