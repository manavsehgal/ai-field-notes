# Transcript — nemo-retriever-embeddings-local

Session 2026-04-22 evening. Handoff pointed at "decide embedding NIM on Spark,
install, benchmark, draft article #2."

## Catalog recon

- `build.nvidia.com/spark/` — 30 Spark playbooks, **no dedicated embedding
  playbook**. Closest: `/spark/rag-ai-workbench` (AI Workbench clone of an
  agentic RAG app — embedding model not named) and `/spark/txt2kg` (explicitly
  says "Future Enhancements: Vector embeddings and GraphRAG capabilities are
  planned enhancements").
- `build.nvidia.com/search?q=embed` — six relevant candidates:
  - `nv-embed-v1` — free endpoint, non-commercial only
  - `llama-nemotron-embed-1b-v2` — Downloadable, 1mo old
  - `llama-3.2-nemoretriever-1b-vlm-embed-v1` — deprecating in 3d (VLM)
  - `llama-nemotron-embed-vl-1b-v2` — multimodal VL
  - `llama-3_2-nemoretriever-300m-embed-v1` — 300M variant
  - `nv-embedcode-7b-v1` — code-retrieval specialist
  - `nv-embedqa-e5-v5` — legacy E5
  - **`llama-3.2-nv-embedqa-1b-v2`** — Deprecation in 27d, badge visible on deploy page
- Deploy pages show `nvcr.io/nim/nvidia/llama-nemotron-embed-1b-v2:latest` as
  the current (not-deprecated) 1B option.
- Model Card: 2048-d output, Matryoshka truncation to 384/512/768/1024/2048,
  max 8192 tokens, Blackwell + Hopper supported architectures.

## Manifest check

```
$ docker manifest inspect nvcr.io/nim/nvidia/llama-nemotron-embed-1b-v2:latest \
    | jq '.manifests[] | .platform'
{ "architecture": "arm64", "os": "linux" }
{ "architecture": "amd64", "os": "linux" }
```

Multi-arch manifest confirmed. Decision: use this NIM (branch 1 of the
prior-handoff decision tree).

## Pull + run

- `docker pull`: 63.1 s, 36 layers, 7.21 GB image, sha256:3c22c0bd8d36…
- First `docker run`: **failed** — `NGC_API_KEY` not exported into env despite
  `source ~/.nim/secrets.env`. Container logged
  "The requested operation requires an API key, but none was found" from
  `nim_sdk.py:338` before any weight download.
- Fix: `export $(grep -v '^#' ~/.nim/secrets.env | xargs)`, or inline
  `-e NGC_API_KEY="$(grep NGC_API_KEY ~/.nim/secrets.env | cut -d= -f2)"`.
- Second run (successful): **52 s cold** to `/v1/health/ready = 200`. Model
  profile: `fp16-7af2b653`. Triton 2.61, GRPC on :8001, HTTP metrics on 8080.
  Cache ends at 2.4 GB on disk.

## Benchmarks

Chunk size: ~500 tokens (reported `usage.total_tokens = 539` for the 2 KB
English prose passage). Warmup: 3 calls.

| Load | p50 | p95 | Throughput |
|---|---:|---:|---:|
| batch=1, 20 seq calls | 40.4 ms | 42.7 ms | 24.75 docs/s · 13,342 tok/s |
| batch=8, 10 reqs      | 278.9 ms | — | 28.69 docs/s · 15,463 tok/s |
| batch=32, 5 reqs      | 1118.0 ms | — | 28.62 docs/s · 15,427 tok/s |

Throughput plateaus at batch≥8. GPU-saturated — client-side batching above 8
buys latency, not throughput.

Cosine sanity (query/near/far):
- cosine(query, near) = **0.3466**
- cosine(query, far)  = **-0.0518**

`nvidia-smi` under batch=32 load: 74–78% GPU util, 33 W power, 46–47 °C.
`memory.total` / `memory.used` / `power.limit` = `[N/A]` (unified memory +
NVML gap on GB10 — same as article #1).

Host footprint under load: `nim-embed-nemotron` 3.60 GiB / 121.7 GiB.

## Raw evidence files

Under `evidence/`:
- `01-pull*.txt` — pull wall-clock
- `02-container-start.txt` — docker run outputs
- `03-ready-wallclock.txt` — 52 s to `/v1/health/ready`
- `04-models.json` — model list response
- `05-startup-tail.log` — Triton startup log
- `06-benchmark.{log,json}` — throughput and cosine data
- `07-docker-stats.txt` — idle stats
- `08-nvidia-smi.csv` — idle GPU
- `09-free.txt` — host memory idle
- `10-cache-size.txt` / `11-image-size.txt` — footprint
- `12-nvidia-smi-load.csv` — GPU under load
- `13-docker-stats-load.txt` — container under load
- `benchmark.py` — the driver script (idempotent; re-runnable)

## Narrative decisions

- Lead with "semantic space" not "embedding model" — the thesis is that
  embeddings turn text into geometry, which is what makes the Spark's
  economics flip.
- Name the deprecation gotcha in section 4 before the install — it's the
  single piece of information most likely to save a reader's afternoon.
- Keep the Matryoshka tradeoff in section 6, not earlier — the reader doesn't
  need to commit to a dim until article #3 (pgvector).
- Do NOT lead with "three arcs" as an org chart. Overlay surfaces in the
  close, per use-case-arc.md guidance.
- Signature figure: new `EmbeddingPipeline` component, teal accent to
  distinguish from NimPipeline's blue; first article authored natively in
  the new scroll-reveal motion system (three svg-reveal-d1..d8 groups on
  the input chunks, the NIM node, and the vector cloud).
- In-body diagram: query+passage dual-path converging on the NIM node,
  then outgoing to 2048-D space chip. Uses the existing fn-diagram system
  with a travelling particle on the query path. One `dashed` edge for the
  downstream "cosine" flow that other articles haven't wired yet.
- Captions: two italic-on-their-own-line captions (`*…*`) under the two
  screenshots — validates the rehype caption plugin in production.

## Stack state at session close

```
Port 8000   → NIM Llama 3.1 8B Instruct (still running, article #1)
Port 8001   → NIM Nemotron Embed 1B v2   (NEW, this article)
Port 8080   → NemoClaw clawnav sandbox
Port 11434  → Ollama (Nemotron-3-Super)
Port 4321   → Astro dev server

~/.nim/cache/llama-3.1-8b-instruct/       (8.5 GB — article #1)
~/.nim/cache/llama-nemotron-embed/        (2.4 GB — this article)
```
