#!/usr/bin/env python3
"""Run the same 6-question naive-RAG sweep as the original benchmark, but
drive every stage through `fieldkit.rag.Pipeline` and aggregate via
`fieldkit.eval.Bench`.

JSON shape stays identical to the committed `benchmark.json` (same
`{summary: {embed_ms, retrieve_ms, generate_total_ms, end_to_end_ms},
queries: [...]}` shape) so a side-by-side diff against the original numbers
stays meaningful. The only field the fieldkit path drops is
`generate_first_token_ms` — `fieldkit.nim.NIMClient.chat` doesn't yet
expose streaming, so TTFT can't be timed without going under the public
API. That field surfaces again with the fieldkit v0.2 streaming work.

Three in-corpus questions (answerable from the AG-News chunks already
sitting in pgvector from `pgvector-on-spark`) + three out-of-corpus
questions (post-2004 events the AG-News corpus cannot answer). The
strict-context scaffold should ground in-corpus and refuse out-of-corpus
— same shape the original article validated.

Prerequisites:
    docker start nim-llama31-8b      # ~96s cold start
    docker start nim-embed-nemotron  # bring up after the chat NIM is warm
    # pgvector should already be up with the `chunks` table populated.

Run from this directory:
    python3 benchmark.py
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import psycopg

from fieldkit.eval import Bench
from fieldkit.nim import NIMClient, wait_for_warm
from fieldkit.rag import Chunk, Pipeline


EMBED_BASE_URL = os.environ.get("EMBED_BASE_URL", "http://localhost:8001/v1")
NIM_BASE_URL = os.environ.get("NIM_BASE_URL", "http://localhost:8000/v1")
NIM_MODEL = os.environ.get("NIM_MODEL", "meta/llama-3.1-8b-instruct")
PGVECTOR_DSN = os.environ.get(
    "PGVECTOR_DSN", "postgresql://spark:spark@localhost:5432/vectors"
)

QUESTIONS = [
    ("in_corpus", "Who won the 2004 US presidential election?"),
    ("in_corpus", "What happened at the 2004 Athens Olympics in swimming?"),
    ("in_corpus", "What did Google do in 2004 related to going public?"),
    ("out_of_corpus", "Who won the 2020 US presidential election?"),
    ("out_of_corpus", "What is NVIDIA DGX Spark?"),
    ("out_of_corpus", "When was Claude 4 Opus released?"),
]


def timed_ask(pipe: Pipeline, question: str, k: int = 5) -> dict:
    """Drive `Pipeline` with per-stage timing.

    Calls `pipe._embed` directly so the embed cost is separable from the
    SQL retrieve cost — the public `pipe.retrieve` rolls them together.
    Then issues the SQL via `psycopg` (mirroring `pipe.retrieve`'s
    internals) so we can split the wall-clock into the same 4 stages
    the original `ask.py` reported.
    """
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    qvec = pipe._embed([question], input_type="query")[0]
    timings["embed"] = round((time.perf_counter() - t0) * 1000, 2)

    vec_lit = "[" + ",".join(f"{x:.6f}" for x in qvec) + "]"
    sql = (
        f"SELECT id, label, text, (embedding <=> %s::vector) AS dist "
        f"FROM {pipe.table} ORDER BY embedding <=> %s::vector LIMIT %s"
    )
    t0 = time.perf_counter()
    with psycopg.connect(pipe.pgvector_dsn) as conn, conn.cursor() as cur:
        cur.execute(sql, (vec_lit, vec_lit, int(k)))
        rows = cur.fetchall()
    timings["retrieve"] = round((time.perf_counter() - t0) * 1000, 2)

    chunks = [
        Chunk(id=int(r[0]), label=r[1] or "", text=r[2], distance=float(r[3]))
        for r in rows
    ]

    t0 = time.perf_counter()
    raw = pipe.fuse(question, chunks, max_tokens=256)
    timings["generate_total"] = round((time.perf_counter() - t0) * 1000, 2)
    timings["end_to_end"] = round(
        timings["embed"] + timings["retrieve"] + timings["generate_total"], 2
    )

    answer = raw["choices"][0]["message"]["content"].strip()
    completion_tokens = raw.get("usage", {}).get("completion_tokens")

    return {
        "question": question,
        "k": k,
        "retrieved": [
            {"id": c.id, "label": c.label, "distance": round(c.distance or 0.0, 4)}
            for c in chunks
        ],
        "answer": answer,
        "timings_ms": timings,
        "completion_tokens": completion_tokens,
    }


def main() -> int:
    print("Waiting for NIMs to warm…")
    if not wait_for_warm(EMBED_BASE_URL, timeout=180):
        raise SystemExit("embed NIM did not warm in 180s")
    if not wait_for_warm(NIM_BASE_URL, timeout=180):
        raise SystemExit("chat NIM did not warm in 180s")

    bench = Bench(
        name="naive-rag",
        metrics=["embed", "retrieve", "generate_total", "end_to_end"],
        metrics_key="timings_ms",
    )

    with NIMClient(base_url=NIM_BASE_URL, model=NIM_MODEL) as gen, Pipeline(
        embed_url=EMBED_BASE_URL,
        pgvector_dsn=PGVECTOR_DSN,
        generator=gen,
        table="chunks",
    ) as pipe:

        def runner(item: tuple[str, str]) -> dict:
            _, q = item
            return timed_ask(pipe, q, k=5)

        with bench:
            bench.run(
                runner,
                QUESTIONS,
                tag_fn=lambda item: {"kind": item[0]},
                on_error="raise",
            )

    bench_summary = bench.summary()
    article_summary = {
        "n": bench_summary["n"],
        "embed_ms": bench_summary["embed"],
        "retrieve_ms": bench_summary["retrieve"],
        "generate_total_ms": bench_summary["generate_total"],
        "end_to_end_ms": bench_summary["end_to_end"],
    }

    queries = []
    for call in bench.calls:
        rec = dict(call.output)
        rec["kind"] = call.tags.get("kind")
        rec["wall_seconds"] = round(call.latency_ms / 1000, 2)
        queries.append(rec)

    blob = {"summary": article_summary, "queries": queries}
    out_path = Path(__file__).resolve().parent / "benchmark.json"
    out_path.write_text(json.dumps(blob, indent=2))

    for rec in queries:
        t = rec["timings_ms"]
        preview = rec["answer"][:120].replace("\n", " ")
        print(
            f"[{rec['kind']:14}] embed {t['embed']:5.1f}  retrieve {t['retrieve']:5.1f}  "
            f"gen {t['generate_total']:7.1f}  | {preview}"
        )

    print(f"\nWrote {out_path} — {len(queries)} queries")
    print(json.dumps(article_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
