# Proposed Spark recipe

The repo is at `github.com/cxcscmu/Auto-Research-Recipes` and ships a clean adapter contract (`docs/task_adapter.md`). Reproduction path:

1. `git clone --depth 1 https://github.com/cxcscmu/Auto-Research-Recipes && cd Auto-Research-Recipes && pip install -e .`
2. Set `ANTHROPIC_API_KEY` in `.env` — the agent driver is Claude Agent SDK, not a local NIM. Capability map says "Agentic systems: tool use, multi-step planning, sandboxed execution" is in-envelope; the agent is just a remote API consumer.
3. Pick **NanoChat-D12** as the first task — it's the most representative MTBM shape (LLM-on-LLM training) and runs on a single GB10 within the 90-minute trial cap. `python -m multi_agent_nc.supervisor --state-root ./magent_state_nc`
4. Reduce the parallel-trial fanout from the published 8-H100 worker default to a **single GB10 worker** (one trial at a time). The supervisor loop, blackboard, and lineage TSV accept arbitrary worker count — the bottleneck is wall-clock per-trial, not the lineage primitive itself.
5. Tap into NemoClaw (already in the capability map's `stack`) for the sandbox — the harness's "MCP tools" wrapping in `agent_core/` is the same shape NemoClaw provides natively. (See "NemoClaw vs OpenClaw on DGX Spark" in the blog for the substrate.)
6. Inspect with `dashboard.py` while the supervisor runs; `release_artifacts/` shows what a frozen run looks like (results.tsv, tree.tsv, best.json, KNOWLEDGE.md, LEADERBOARD.md, lineage_snapshots/).

Per-trial training itself is plain PyTorch — no special TRT-LLM build flags or NIM endpoint required for the *worker*; the cleverness is on the orchestration side.
