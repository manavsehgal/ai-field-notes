# Proposed Spark recipe

The repo `github.com/CherYou/AutoResearchBench` (29⭐, Apache-2.0, Python+Shell) is mature and the benchmark dataset is on Hugging Face at `Lk123/AutoResearchBench`. Crucially, the inference entrypoint reads `OPENAI_API_KEY` and `OPENAI_API_BASE` from `.env` — the agent talks to its model via an OpenAI-compatible chat endpoint, which **NIM exposes natively**. Drop-in.

1. **Clone + install**: `git clone --depth 1 https://github.com/CherYou/AutoResearchBench && cd AutoResearchBench && /opt/venv/bin/python3 -m pip install -r requirements.txt`.
2. **Start NIM with Llama 3.3 70B fp8** (or a smaller in-envelope model first for plumbing). Note the OpenAI-compatible base URL, e.g. `http://localhost:8000/v1`.
3. **Configure `.env`**:
   ```
   MODEL=meta/llama-3.3-70b-instruct
   OPENAI_API_KEY=local
   OPENAI_API_BASE=http://localhost:8000/v1
   INPUT_FILE=input_data/academic_deepsearch_example.jsonl
   ```
4. **Download + decrypt the bench bundle** from HF (the README documents `decrypt_benchmark.py` against an `.obf.json` released bundle).
5. **Run inference**: `bash run_inference.sh`. The agent uses two ship-with-the-repo tools — `tool_deepxivsearch.py` (academic search) and `tool_websearch.py` (general web) — both of which need internet egress and likely an API key for the academic backend.
6. **Run evaluation**:
   ```bash
   bash evaluate/run_evaluate.sh deep --input-file output_data/inference_output.jsonl
   bash evaluate/run_evaluate.sh wide --input-file output_data/inference_output.jsonl --gt-file path/to/gt.jsonl
   ```
7. **Comparative table**: run the same bench against `llama-3.1-8b-instruct` (NIM), `nemotron-super-49b` (NIM), and Nemotron via NemoClaw to land a Spark-stack-internal leaderboard. Tie back to the Autoresearch arc — this *is* the eval harness for the autonomous research loop the blog is building.

