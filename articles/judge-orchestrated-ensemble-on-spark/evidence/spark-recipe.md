# Proposed Spark recipe

The repo is at `github.com/RaguTeam/ragu_mtrag_semeval` and is uv-managed. Reproduction path:

1. `git clone --depth 1 https://github.com/RaguTeam/ragu_mtrag_semeval && cd ragu_mtrag_semeval && uv sync --extra eval`
2. Clone the IBM MTRAG benchmark: `git clone https://github.com/IBM/mt-rag-benchmark` and set `MTRAG_DATA` accordingly.
3. **Local member** — replace the README's bare-vLLM call with a NIM-served Qwen3-4B endpoint per "NIM First Inference on DGX Spark" (capability map confirms NIM serves Qwen3 with paged-attention KV economics). NIM provides the OpenAI-compatible API the harness already speaks.
4. **Other six members** — keep the OpenAI-compatible endpoint indirection. Spark-local alternative: stand up a second NIM with Meno-Lite-0.1 (7B); for the rest, you can either hit a hosted API (paper's choice) or model-swap inside vLLM. Capability map's "Long-context inference economics (KV cache, paged attention)" is in-envelope for ≤ 14B models.
5. **Judge** — Replace GPT-4o-mini with a local NIM-served Qwen3-32B (or NeMo Evaluator's judge harness from "RAG Eval — Ragas + NeMo Evaluator" in the blog). Capability map: ≤ 70B inference is in-envelope; 32B fits with margin.
6. Run `python src/generation/main.py` then `scripts/generation/run_generation_task_b.py`. Aggregate metrics: `python scripts/evaluation/metrics_aggregation.py`.
7. Adapt the routing logic in `src/generation/main.py` — that's where the per-instance "two prompting variants × seven models" candidate fan-out happens, and where the judge-selection call is wired.
