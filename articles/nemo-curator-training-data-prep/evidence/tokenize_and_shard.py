"""
Stage 2 of A3 — exact-dedup + tokenize + pack the Curator-cleaned JSONL
into a single packed-token memmap that the sweep harness will read from.

Why a single int32 .npy memmap instead of Megatron's IndexedDataset .bin
+ .idx pair: the goal of A3 is to *measure* the data-path overhead vs
A2's random-token harness. The two harnesses should differ in exactly
one place — where each batch's tokens come from. Megatron's
IndexedDataset adds a record-boundary index lookup per sample that the
random path doesn't pay; that would conflate kernel time with index
time. A flat memmap is the apples-to-apples comparison.

Stages here:
  1. Read all cleaned JSONL into pandas
  2. Exact dedup by SHA256 hash of normalized text
  3. Tokenize with the gpt2 BPE tokenizer (matches A2's vocab=50,257)
  4. Concatenate token streams with EOT separators
  5. Write packed.int32.npy + packed.meta.json

Run inside nemo-curator-spark:1.1.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
from transformers import GPT2TokenizerFast

EVIDENCE = os.path.dirname(os.path.abspath(__file__))
CLEANED_GLOB = os.path.join(EVIDENCE, "cleaned", "*.jsonl")
PACKED_PATH = os.path.join(EVIDENCE, "packed.int32.npy")
META_PATH = os.path.join(EVIDENCE, "packed.meta.json")
DEDUP_STATS_PATH = os.path.join(EVIDENCE, "dedup_tokenize_stats.json")

EOT_TOKEN_ID = 50256  # gpt2 endoftext


def normalized_hash(text: str) -> bytes:
    """Hash for exact-dedup: lowercase, strip, collapse whitespace."""
    norm = " ".join(text.lower().split())
    return hashlib.sha256(norm.encode("utf-8")).digest()


def main() -> None:
    # 1) Read cleaned JSONL.
    print("loading cleaned JSONL ...")
    t0 = time.perf_counter()
    files = sorted(glob.glob(CLEANED_GLOB))
    print(f"  {len(files)} jsonl files")
    rows = []
    for fp in files:
        with open(fp) as f:
            for line in f:
                rec = json.loads(line)
                rows.append(rec.get("text", ""))
    df = pd.DataFrame({"text": rows})
    print(f"  loaded {len(df):,} docs in {time.perf_counter() - t0:.1f}s")

    # 2) Exact dedup (manual hash dedup; fast at this size).
    print("\nexact-dedup ...")
    t0 = time.perf_counter()
    df["hash"] = df["text"].map(normalized_hash)
    before = len(df)
    df = df.drop_duplicates(subset="hash").reset_index(drop=True)
    after = len(df)
    print(f"  {before:,} → {after:,}  (removed {before-after:,}, "
          f"{(before-after)/before:.1%})")
    print(f"  dedup wall: {time.perf_counter() - t0:.1f}s")

    # 3) Tokenize.
    print("\nloading GPT-2 tokenizer ...")
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    print(f"  vocab_size = {tok.vocab_size:,}")

    print("\ntokenizing ...")
    t0 = time.perf_counter()
    # Process in batches of 4096 for tokenizer parallelism.
    BATCH = 4096
    streams: list[np.ndarray] = []
    total_tokens = 0
    for i in range(0, len(df), BATCH):
        chunk = df["text"].iloc[i:i + BATCH].tolist()
        enc = tok(chunk, add_special_tokens=False)["input_ids"]
        # Append EOT after each doc, then flatten.
        for ids in enc:
            ids.append(EOT_TOKEN_ID)
            streams.append(np.asarray(ids, dtype=np.int32))
            total_tokens += len(ids)
        if (i // BATCH) % 20 == 0:
            elapsed = time.perf_counter() - t0
            tps = total_tokens / max(0.001, elapsed)
            print(f"  doc {i:>7,}/{len(df):>7,}  tokens={total_tokens:>11,}  "
                  f"({tps:>10,.0f} tok/s, {elapsed:.1f}s)")
    tokenize_s = time.perf_counter() - t0
    print(f"  tokenize wall: {tokenize_s:.1f}s")
    print(f"  total tokens: {total_tokens:,}")

    # 4) Concatenate.
    print("\nconcatenating into one packed stream ...")
    t0 = time.perf_counter()
    packed = np.concatenate(streams)
    concat_s = time.perf_counter() - t0
    print(f"  packed.shape = {packed.shape}  "
          f"size = {packed.nbytes / 1024**2:.1f} MiB  ({concat_s:.1f}s)")

    # 5) Write memmap + meta.
    print("\nwriting packed memmap ...")
    t0 = time.perf_counter()
    np.save(PACKED_PATH, packed)
    write_s = time.perf_counter() - t0
    print(f"  wrote {PACKED_PATH}  ({write_s:.1f}s)")

    meta = {
        "tokenizer": "gpt2",
        "vocab_size": tok.vocab_size,
        "eot_token_id": EOT_TOKEN_ID,
        "num_docs_after_dedup": after,
        "num_docs_dropped_by_dedup": before - after,
        "total_tokens": int(total_tokens),
        "packed_dtype": "int32",
        "packed_path": os.path.basename(PACKED_PATH),
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  wrote {META_PATH}")

    stats = {
        "stage": "dedup_tokenize_pack",
        "load_jsonl_s": None,  # subsumed in early prints
        "dedup_wall_s": None,
        "tokenize_wall_s": round(tokenize_s, 1),
        "concat_wall_s": round(concat_s, 1),
        "write_wall_s": round(write_s, 1),
        "input_docs": before,
        "after_dedup_docs": after,
        "dedup_drop_ratio": round((before - after) / before, 4),
        "total_tokens": int(total_tokens),
        "packed_size_mib": round(packed.nbytes / 1024**2, 2),
        "tokenize_throughput_tok_per_s": round(total_tokens / tokenize_s, 1),
    }
    with open(DEDUP_STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  wrote {DEDUP_STATS_PATH}")


if __name__ == "__main__":
    main()
