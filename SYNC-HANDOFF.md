<!--
  🆕 STATUS: NEW — awaiting Mac sweep.
  This file is one feature/release at a time, not a running log.
  At the next release prompt, **clear this entire file and start fresh** (do NOT append to existing sections).
  Last reset: 2026-05-15 (this rotation supersedes the v0.4.1 cycle scope; the v0.4.1 PyPI release commits 7f1159e..82d95ed are wholly consumed by Mac PR #5).

  Prior Mac sweep receipts (preserved here since SYNC-HANDOFF is per-release-not-running-log; Mac will sweep this 2026-05-15 cycle next):
  - 2026-05-14 v0.4.1 cycle: swept at destination commit manavsehgal/ainative-business.github.io@e1b16de (Mac PR #5 merged 2026-05-14).
  - 2026-05-14 v0.4.0 cycle: swept at destination commit manavsehgal/ainative-business.github.io@f7ea7aa (Mac PR #4 against this repo — conflicted on rotation; safe to close, receipt captured here).
  - 2026-05-14 Orionfold/finance-chat-GGUF cycle: swept at destination commit manavsehgal/ainative-business.github.io@85f9307 (Mac PR #3, merged).
  - 2026-05-12 Autoresearch→MTBM rename: swept at destination commit manavsehgal/ainative-business.github.io@71293af (Mac PR #2, merged).
-->
---
release_slug: 2026-05-15-cyber-vertical
status: NEW
source_range: 97e824d..dd81a29
articles_added:
  - becoming-a-cyber-curator-on-spark      # third vertical-curator deep-dive; status: published; series: Machine that Builds Machines; book_chapters: [10, 11]; fieldkit_modules: [quant, publish, eval, lineage]; hf_url: https://huggingface.co/Orionfold/SecurityLLM-GGUF; inline fn-diagram (hub-and-spoke: fieldkit.publish at centre, three vertical chips); no signature SVG yet — Mac sweep can leave card without thumbnail or pick up VerticalCuratorRetry as a placeholder
articles_updated:
  - becoming-a-gguf-publisher-on-spark     # evidence/lineage-SecurityLLM/results.tsv added (5 variant rows from cyber measure); published article body unchanged
artifacts_added:
  - src/content/artifacts/securityllm-gguf.yaml   # third Phase-2 manifest (after finance-chat-gguf.yaml + saul-7b-instruct-v1-gguf.yaml) — license.tier=free, license.model=apache-2.0, base_model=ZySec-AI/SecurityLLM, vertical_eval populated from CyberMetric n=50, recommended_variant=Q4_K_M (differs from finance/legal which both used Q5_K_M — Q4_K_M topped the cyber bench by 6 points within sampling noise)
artifacts_updated: []
fieldkit_modules_changed: []               # ZERO fieldkit source changes — the headline. v0.4.1 publishing surface generalized as designed for vertical 3
papers_added: []
papers_classify_count: 0                   # no frontier-scout refresh this cycle
renames_to_replay: []
removes: []
new_top_level_pages: []
breaking_changes: []
destination_overrides_to_preserve: []
hf_repos_added:
  - Orionfold/SecurityLLM-GGUF             # 5 GGUF variants of ZySec-AI/SecurityLLM (Mistral-7B + Zephyr DPO), apache-2.0, CyberMetric-scored (n=50, mcq_letter); sha d4569840, lastModified 2026-05-15T19:51:46Z; pushed live via hf_push_resilient.py (no upload_folder crash this cycle — [[feedback_hf_upload_resilient_api]] paying off)
civitai_artifacts_added: []
fieldkit_release: null                     # no PyPI release this cycle; current published version remains 0.4.1
post_rotation_commits: []                  # rotation happens at end-of-cycle; any post-rotation commits captured in next sweep
---

## Headline

Third Orionfold quant card ships: [`Orionfold/SecurityLLM-GGUF`](https://huggingface.co/Orionfold/SecurityLLM-GGUF) — cyber vertical, five-variant Spark-tested shape, CyberMetric-80 mini-eval (50 rows, `mcq_letter` scorer). The release validates the v0.4.1 publishing surface against its design promise: **zero fieldkit source changes** were needed for the third vertical. The PyPI package version on this release's commit (`dd81a29`) is the same `0.4.1` the prior legal release shipped on.

What changed for cyber lives entirely in `scripts/` — four files:

1. **`scripts/cyber_merge.py`** (new, ~110 LOC) — samples 50 rows from `tihanyin/CyberMetric-80` (apache-2.0, arxiv 2402.07688), formats each as a 4-option MCQ prompt with a "reply with only one letter" instruction, emits the `{id, text, answer, task}` JSONL shape that `VerticalBench.from_jsonl(format="legalbench")` already consumed.
2. **`scripts/g3_measure_variants.py`** — `cybermetric` added to `VERTICAL_BENCH` valid list (alongside `financebench` / `legalbench`); local `mcq_letter` scorer (regex-extract A/B/C/D with preference for "Answer: X" markers); `_wrap_zephyr` chat-template wrapper alongside the existing `_wrap_inst`; per-vertical wrapper dispatch.
3. **`scripts/g3_preflight_bench.py`** — same VERTICAL_BENCH dispatch + zephyr-template detection (extends `_detect_prompt_format` to recognize `<|user|>` + `<|assistant|>` in `tokenizer_config.json`) + the same local `mcq_letter` scorer; the V0 preflight gate now works for cyber as a five-question fast-fail before the multi-hour quantize+measure cycle.
4. **`scripts/g3_build_first_quant.sh`** — `ZySec-AI/SecurityLLM` case auto-resolves `MODEL_LICENSE=apache-2.0`, `CHAT_FORMAT=zephyr`, `VERTICAL_BENCH=cybermetric`, `ARTICLE_SLUG=becoming-a-cyber-curator-on-spark`. `CYBERBENCH_JSONL` propagated through `step_preflight_bench` and `step_measure`.

Spark-measured numbers (n=50 per variant): F16 34% · Q8_0 36% · Q6_K 36% · Q5_K_M 38% · **Q4_K_M 40%**. The smaller variants matched or beat F16 — within sampling noise but consistent with the pattern from finance and legal. Q4_K_M is the default-pick on this card (vs Q5_K_M on the prior two cards). Q8_0 throughput anomaly (slower than F16 on both prior verticals) did NOT repeat — Q8_0 = 30.3 tok/s vs F16 = 17.5 — see [[project_q8_anomaly_model_specific]].

## What Mac CC sweeps

Straight mirror across the cyber cycle — no destination-side rewrites, no schema changes, no renames. Concrete files / paths:

- **`articles/becoming-a-cyber-curator-on-spark/article.md`** — new long-form. ~1,900 words; inline `fn-diagram` (hub-and-spoke topology: `fieldkit.publish_quant` at centre, three vertical chips for finance/legal/cyber with cyber as accent); `verify_article.sh` passes all gates including SVG hard invariants. Frontmatter: `status: published`, `series: Machine that Builds Machines`, `book_chapters: [10, 11]`, `fieldkit_modules: [quant, publish, eval, lineage]`, `also_stages: [observability]`, `hf_url: https://huggingface.co/Orionfold/SecurityLLM-GGUF`.
- **`articles/becoming-a-cyber-curator-on-spark/transcript.md`** — full session provenance: model + bench picks, the three scripts deltas, preflight gate (3/5 F16 PASS), scorer smoke-test results.
- **`articles/becoming-a-gguf-publisher-on-spark/evidence/lineage-SecurityLLM/results.tsv`** — 5 variant rows with four-axis measurement. Q4_K_M = 0.40 cyber-bench (best on bench, beats F16 = 0.34 within n=50 sampling variance), F16 = 7.301 ppl / 17.5 tg, Q4_K_M = 47.7 tg (also throughput pick).
- **`src/content/artifacts/securityllm-gguf.yaml`** — third Phase-2 manifest. Catalog templates can now render three side-by-side (finance-chat-gguf · saul-7b-instruct-v1-gguf · securityllm-gguf). `license.tier=free`, `license.model=apache-2.0`, `chat_format=zephyr`, `vertical_eval_name="CyberMetric (n=50, mcq_letter)"`.
- **`scripts/cyber_merge.py`** — new helper (see Headline).
- **`scripts/g3_measure_variants.py`** + **`scripts/g3_preflight_bench.py`** + **`scripts/g3_build_first_quant.sh`** — incremental patches threading `cybermetric` through the pipeline (see Headline).

### Auto-refreshed

- **`src/data/project-stats.json`** + **`README.md`** — 41 articles total (+1 cyber), +1 product-card on `cyber` tag, LOC nudge from the ~110-line cyber-merge script. Deployment stage gains another entry.

## What Mac CC does NOT need to do

- **No rename replays.** No new entries in `SYNC-RENAMES.log` this cycle. Existing entries remain fully `complete` after the prior `orionfoldllc → Orionfold` and `Autoresearch → Machine that Builds Machines` sweeps.
- **No new top-level pages.** Article lives at `/field-notes/becoming-a-cyber-curator-on-spark/`; sorted by ordinal-desc per the existing convention. Manifest renders via the catalog template (already in place from the v0.4.0 cycle).
- **No destination-prose rewrites.** New article slots into the existing Astro article collection without schema or template changes. The `hf_url` field, the inline `fn-diagram` system, and the `artifacts` collection are all reused unchanged.
- **No fieldkit changes.** The PyPI package version on this commit (`0.4.1`) is unchanged from the prior cycle. `pip install fieldkit==0.4.1` users see no surface delta. No new SDK call sites land — the `mcq_letter` scorer is intentionally kept as a local helper in `scripts/g3_measure_variants.py` per [[feedback_keep_scorer_local_until_reuse]]; promotion candidacy waits for vertical 4 to confirm reuse.
- **No new skill IA mirroring.** `hf-publisher`, `hf-model-scout`, and `fieldkit-curator` all live in `~/.claude/skills/` (Spark CC user config), not in the source repo.
- **No signature SVG.** The article ships without a `signature:` frontmatter field — Mac can leave the home/stage card without a thumbnail (allowed by the schema) or pick up `VerticalCuratorRetry` as a placeholder, author's call. A fresh signature can be added in a future polish PR.

## Verification (Spark-side)

- HF push verified live at <https://huggingface.co/Orionfold/SecurityLLM-GGUF> — HTTP/2 200; all 7 files committed (`.gitattributes`, `README.md`, 5× `model-*.gguf`); sha `d45698400b7b02a036443b3c3cb6520bf946c9f4`; lastModified `2026-05-15T19:51:46Z`.
- Push wall time: 4h 11min via `hf_push_resilient.py` (`upload_large_folder`, `num_workers=1`). Hit transient socket retries — the WiFi signal at -78 dBm dropped associations every 90–120s during the push window (29 disconnect events in 24h, fixed mid-cycle by `sudo iw dev wlP9s9 set power_save off` + `tcp_keepalive_time=600` lowering). Resilient pusher absorbed every retry cleanly — no `=== PUSH PARTIAL` line emitted, terminal state was `=== PUSH COMPLETE`.
- `scripts/verify_article.sh becoming-a-cyber-curator-on-spark` — all gates pass: frontmatter valid, image references resolve, slug matches folder, no TODO markers, no PII patterns, SVG hard invariants pass (gradient defs, stroke hierarchy, icon clearance, role+aria-label, no hex literals).
- Preflight gate: 3/5 F16 PASS on CyberMetric (above the ≥1/5 abort threshold). Two failures were compliance failures (model wrote prose instead of a letter) — `mcq_letter` correctly returned 0; the trap-detection works as designed.
- Five-variant measure: 5 of 5 GGUFs scored cleanly on n=50 questions. Scores in 17/50–20/50 range (34–40%) — well above 25% random baseline for 4-option MCQ.

## Release-commit chain (this cycle)

- **`dd81a29`** — `Add cyber-vertical card: Orionfold/SecurityLLM-GGUF + CyberMetric mini-eval` (10 files, +697/-46).

Single commit this cycle; HF push happened after the commit landed on `origin/main`.

## What Mac CC should look for after sweep

- Three vertical-curator cards should now render side-by-side wherever Mac-side surfaces a catalog page for `artifacts/quants/`: `finance-chat-gguf` (week 1) → `saul-7b-instruct-v1-gguf` (week 2) → `securityllm-gguf` (week 3). Suggested ordering: chronological-desc.
- The cyber article should appear in the Machine that Builds Machines series listing. It's a peer to `becoming-a-gguf-publisher-on-spark` and `becoming-a-legal-curator-on-spark` in series + book_chapters; the fn-diagram visually anchors the three-vertical relationship.
- The home-page "At a glance" infographic now reads 41 articles. The cyber tag list may be new — `[cyber, security]` enters tag-stat aggregation.
- The fieldkit landing page (`/fieldkit/`) version prop continues to display `0.4.1` — no change this cycle. The page's "Articles in this release" section may already render an empty list for v0.4.1+ (the cyber article doesn't list a v0.4.1 release coupling — its `fieldkit_modules` field declares `eval` but the cycle is article-only, not a release coupling).
