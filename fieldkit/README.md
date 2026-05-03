# fieldkit

> Verified-on-Spark patterns lifted from the [ai-field-notes](https://ainative.business/field-notes/) blog into one importable Python package.

Every essay in `ai-field-notes` ends with `evidence/` — a folder of working code that produced the article's numbers. After 25+ articles the same patterns kept reappearing: the same NIM client wrapper, the same chunk-embed-store dance, the same bench harness. `fieldkit` is what those `evidence/` folders look like once the boilerplate is lifted into a real package.

The blog stays the long-form rationale. `fieldkit` is the `pip install`-able surface so you can reproduce — and extend — the work without re-pasting 80 lines of NIM-client setup per article.

## Install

```bash
pip install fieldkit
```

For the bleeding edge between releases, install from the git tag instead:

```bash
pip install "git+https://github.com/manavsehgal/ai-field-notes.git@fieldkit/v0.1.0#subdirectory=fieldkit"
```

## Quickstart

```python
from fieldkit.nim import NIMClient

client = NIMClient(base_url="http://localhost:8000/v1", model="meta/llama-3.1-8b-instruct")
print(client.chat([{"role": "user", "content": "Hello, Spark."}]))
```

## What's in v0.1.0

| Module | Purpose | Source articles |
|---|---|---|
| `fieldkit.capabilities` | Typed Python facade over `spark-capabilities.json` — KV cache math, weight bytes, inference envelope. | `kv-cache-arithmetic-at-inference`, `gpu-sizing-math-for-fine-tuning` |
| `fieldkit.nim` | OpenAI-compatible NIM client wrapper with retry, chunking, and the 8192-token context guard. | `nim-first-inference-dgx-spark` and friends |
| `fieldkit.rag` | `Pipeline(embed_url, rerank_url, pgvector_dsn, generator)` — ingest → retrieve → rerank → fuse. | `naive-rag-on-spark` and friends |
| `fieldkit.eval` | `Bench`, `Judge`, `Trajectory` — the recurring eval harness shapes. | every article with a `bench.py` or `benchmark.py` |
| `fieldkit.cli` | `fieldkit bench rag`, `fieldkit feasibility <id>`, `fieldkit envelope <size>`. | discoverability |

Modules deferred to `v0.2`: `retriever`, `ft`, `guardrails`, `agents`. To `v0.3`: `train`, `observe`.

## Hardware

`v0.1` is **Spark-only**. Every code path is verified on a DGX Spark (GB10, 128 GB unified memory, NIM 8B + embed NIM + pgvector co-resident). Portability to other CUDA 12.x boxes lands in `v0.2+` when there's demand.

## License

Apache-2.0. See [`LICENSE`](https://github.com/manavsehgal/ai-field-notes/blob/main/fieldkit/LICENSE).

## Links

- **Blog:** https://ainative.business/field-notes/
- **Docs:** https://ainative.business/fieldkit/
- **Source:** https://github.com/manavsehgal/ai-field-notes/tree/main/fieldkit
- **Changelog:** [`CHANGELOG.md`](https://github.com/manavsehgal/ai-field-notes/blob/main/fieldkit/CHANGELOG.md)
