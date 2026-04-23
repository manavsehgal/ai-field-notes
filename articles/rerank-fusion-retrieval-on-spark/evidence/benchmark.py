#!/usr/bin/env python3
"""Run the 30-query qrels set through all four retrieval modes
(naive, bm25, rrf, rerank), compute recall@5 and recall@10, and
summarise timings. Writes benchmark.json.

Retrieval only — no generation — so we isolate the retrieval-quality
signal from the 8B model's grounding bias.

Usage:
    python3 benchmark.py
    python3 benchmark.py --modes naive,bm25,rrf
    python3 benchmark.py --qrels qrels.jsonl --out benchmark.json
"""
import argparse
import json
import statistics
import sys
import time

import hybrid_ask as h  # reuse retrieve() from the sibling script

DEFAULT_QRELS = "qrels.jsonl"
DEFAULT_MODES = ["naive", "bm25", "rrf", "rerank"]


def recall_at(retrieved_ids, relevant_ids, k):
    if not relevant_ids:
        return None
    top = set(retrieved_ids[:k])
    hit = top.intersection(relevant_ids)
    return len(hit) / min(len(relevant_ids), k)


def run_one(query, mode, k_feed, relevant_ids):
    t0 = time.perf_counter()
    hits, timings = h.retrieve(query, mode, k_feed)
    wall_ms = (time.perf_counter() - t0) * 1000
    ids = [x["id"] for x in hits]
    return {
        "retrieved_ids": ids,
        "timings_ms": {k: round(v, 2) for k, v in timings.items()},
        "wall_ms": round(wall_ms, 2),
        "recall@5": recall_at(ids, relevant_ids, 5),
        "recall@10": recall_at(ids, relevant_ids, 10),
    }


def summarise(rows):
    by_mode = {}
    for r in rows:
        by_mode.setdefault(r["mode"], []).append(r)
    out = {}
    for mode, rs in by_mode.items():
        r5 = [x["recall@5"] for x in rs if x["recall@5"] is not None]
        r10 = [x["recall@10"] for x in rs if x["recall@10"] is not None]
        wall = [x["wall_ms"] for x in rs]
        out[mode] = {
            "n": len(rs),
            "mean_recall@5": round(statistics.mean(r5), 4) if r5 else None,
            "mean_recall@10": round(statistics.mean(r10), 4) if r10 else None,
            "median_wall_ms": round(statistics.median(wall), 2),
            "p95_wall_ms": round(sorted(wall)[int(0.95 * len(wall))], 2) if wall else None,
            "max_wall_ms": round(max(wall), 2),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qrels", default=DEFAULT_QRELS)
    ap.add_argument("--modes", default=",".join(DEFAULT_MODES))
    ap.add_argument("--k", type=int, default=10,
                    help="top-K fed to recall@K (retrievers fetch >= this)")
    ap.add_argument("--out", default="benchmark.json")
    args = ap.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    qrels = [json.loads(l) for l in open(args.qrels)]

    rows = []
    print(f"Running {len(qrels)} queries × {len(modes)} modes = "
          f"{len(qrels)*len(modes)} retrievals", file=sys.stderr)
    for q in qrels:
        for mode in modes:
            try:
                r = run_one(q["query"], mode, args.k, q["relevant_ids"])
            except Exception as e:
                r = {"error": str(e), "retrieved_ids": [], "wall_ms": 0,
                     "recall@5": 0.0, "recall@10": 0.0, "timings_ms": {}}
            r.update({"id": q["id"], "query": q["query"], "mode": mode,
                      "relevant_count": len(q["relevant_ids"])})
            rows.append(r)
            r5 = r.get("recall@5")
            print(f"  {q['id']:>4}  {mode:<7}  "
                  f"recall@5={(r5 if r5 is not None else float('nan')):.2f}  "
                  f"wall={r['wall_ms']:.0f} ms", file=sys.stderr)

    summary = summarise(rows)
    out = {"summary": summary, "per_query": rows}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== summary ===")
    print(f"{'mode':<8} {'n':>3} {'recall@5':>10} {'recall@10':>10} "
          f"{'median_ms':>10} {'p95_ms':>8} {'max_ms':>8}")
    for mode in modes:
        s = summary.get(mode, {})
        if not s:
            continue
        print(f"{mode:<8} {s['n']:>3} "
              f"{(s['mean_recall@5'] or 0):>10.4f} "
              f"{(s['mean_recall@10'] or 0):>10.4f} "
              f"{s['median_wall_ms']:>10.2f} "
              f"{s['p95_wall_ms']:>8.2f} "
              f"{s['max_wall_ms']:>8.2f}")


if __name__ == "__main__":
    main()
