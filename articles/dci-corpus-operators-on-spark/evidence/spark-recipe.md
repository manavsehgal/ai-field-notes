# Proposed Spark recipe

The repo is at `github.com/DCI-Agent/DCI-Agent-Lite` and is uv-managed with a one-click `bash setup.sh`. It builds on **Pi** (`badlogic/pi-mono` coding-agent) with bash tools.

1. `git clone --depth 1 https://github.com/DCI-Agent/DCI-Agent-Lite && cd DCI-Agent-Lite && bash setup.sh`
2. Configure `.env` with at least one of `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` for the published path. **Spark-local path:** point the harness at a NIM endpoint via the OpenAI-compatible API the NIM container exposes — Pi already speaks OpenAI-format, so this is a base-URL swap, no code change.
3. Download the corpus + bench: `uv run python scripts/download_corpus.py` and `uv run python scripts/download_dci_bench.py`. Both come from HF: `DCI-Agent/corpus` and `DCI-Agent/dci-bench`.
4. Install ripgrep (`apt install ripgrep`) — capability map's `stack` block already presumes a Linux userspace; this is a one-line dependency.
5. Run a benchmark: the repo ships scripts for BRIGHT, BEIR, BrowseComp-Plus, and multi-hop QA — total 13 benchmarks. The full suite is hours, not days, on a single Spark with a local NIM.
6. **The extractable abstraction is the operator vocabulary** — `rg` (regex with `-A`/`-B` context), `find` (filename / mtime predicates), `sed` (slice ranges), `cat` (whole-file read), shell pipes for composition. The agent learns to compose these instead of calling `retriever.search(q)`. This is what becomes `fieldkit.rag.operators`.
