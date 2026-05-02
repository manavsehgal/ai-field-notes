# Proposed Spark recipe

The repo is `github.com/LinesHogan/tLLM` (33⭐, Python, last push 2026-04-26). Description: *"tLLM is a test-time training extension of vLLM."* That is load-bearing for the recipe — ESamp ships as a vLLM fork, **not** TRT-LLM/NIM. So the canonical NIM serving path doesn't apply directly; we run vLLM standalone.

1. **Clone the repo**: `git clone --depth 1 https://github.com/LinesHogan/tLLM`. Top-level layout is `tllm/` (the package), `starter.py`, `doc/`, `doc_zh/`, `test/`. No requirements.txt in the public listing — read `pyproject.toml` from the package or follow `doc/`.
2. **Install in a fresh container** — vLLM on Blackwell needs CUDA 12.x kernels; use the NeMo / PyTorch container as the base and remember the `/opt/venv` pip-trap from `feedback_nvidia_container_uv_venv_trap` (always `/opt/venv/bin/python3 -m pip install`).
3. **Pick the base model**: Qwen 2.5 7B Instruct (already in the capability map's NIM-supported list — same weights, just served via vLLM here). Reasoning model option: DeepSeek-R1-Distill-Qwen-7B for the Pass@k benchmarks the paper highlights.
4. **Run the baseline**: vanilla vLLM stochastic sampling, n=8 samples, on AIME / MATH / HumanEval / a creative-writing prompt set. Measure Pass@k and tok/s.
5. **Run ESamp**: enable the Distiller via tLLM's starter.py; same n=8 samples; same benchmarks. Measure Pass@k lift, tok/s degradation (paper claims ≤5% worst-case), and semantic diversity (e.g., embed each sample with `nemotron-embed-1b-v2` and measure mean pairwise cosine).
6. **Cross-link**: ties directly to *KV-Cache Arithmetic at Inference* — both are about extracting more from a fixed compute budget. The article frames ESamp as the *exploration* counterpart to KV's *capacity* analysis.

