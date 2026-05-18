# Transcript — patent-strategist-v1-baseline-on-spark

Provenance for the article. Drafted mid-T10 sweep (2026-05-17, session 27).

## Session source

- `HANDOFF.md` "Session 26 complete" block — bench seed, T9 reviewer, T10 scaffold + retrieval-only smoke.
- Spec: `specs/patent-strategist-v1.md` §3 (eval design), §3.3 (scorers), §3.4 (RAG retrieval), §3.5 (full eval matrix and Judge backends).
- Driver: `scripts/run_rag_baseline.py`
- Rescore helper (new this session): `scripts/rescore_predictions.py`

## Live runs referenced

| Run                                                  | Mode      | Bench rows | Wall  | Notes                                |
|------------------------------------------------------|-----------|------------|-------|--------------------------------------|
| `20260517-104509-retrieval-518c10`                   | retrieval | 5 (D-mcq)  | 183s  | First smoke after server stand-up    |
| `20260517-104908-retrieval-136ef4`                   | retrieval | 200        | 3h 22m| Full retrieval; rescored post-bugfix |
| `20260517-141203-oracle-e6885f`                      | oracle    | 200        | 2h 52m| Patched scorer in-process            |
| `20260517-170410-closed-b8cfe9` (in progress)        | closed    | 200        | TBD   | Pending; numbers fill the §What table|

## Scorer fix

- File: `fieldkit/src/fieldkit/eval/__init__.py`
- Function: `mcq_letter`
- Change: `re.search(...)` → `re.findall(...)[-1]` on `_MCQ_AFTER_ANSWER_RE`
- New test: `test_concluding_answer_wins_over_elimination` in `tests/eval/test_mcq_letter.py`
- All 25 `mcq_letter` tests pass.

## Memory references threaded into the article

- `[[feedback_chat_vs_continued_pretrain_trap]]` — informs the model-selection paragraph.
- `[[project_q8_anomaly_model_specific]]` — informs the quantization decision.
- `[[reference_hf_cache_path_on_spark]]` — explains the `HF_HUB_CACHE` override.
- `[[feedback_reasoning_model_npredict]]` — informs the truncation tradeoff discussion.

## Pending before publish

See the DRAFT STATE comment at the top of `article.md`. Five items: closed-book numbers, `signature` SVG component, inline diagram polish, scrots from llama-server console, final `verify_article.sh` pass.
