"""
Stage 1 of A3 — clean wikitext-103 with NeMo Curator.

Reads the raw HuggingFace parquet shards (wikitext103_train_*.parquet)
into Curator's pipeline. Runs:
  1. unicode reformatter            (normalise NFC)
  2. newline normalizer             (collapse runs of \\n\\n+)
  3. WordCountFilter                (drop docs < 50 or > 100,000 words)
  4. RepeatingTopNGramsFilter       (drop docs with high n-gram repetition)
  5. SymbolsToWordsFilter           (drop docs > 20% symbols)
Writes cleaned JSONL to evidence/cleaned/.

Run inside nemo-curator-spark:1.1 (built from evidence/Dockerfile).
Exact dedup is done in tokenize_and_shard.py (it's a single hash pass at
this corpus size — Curator's GPU dedup needs cuDF orchestration that is
overkill for a 500 MB corpus).
"""
from __future__ import annotations

import json
import os
import time

import pandas as pd

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import ParquetReader
from nemo_curator.stages.text.io.writer import JsonlWriter
from nemo_curator.stages.text.modifiers import (
    NewlineNormalizer,
    UnicodeReformatter,
)
from nemo_curator.stages.text.modules import Modify
from nemo_curator.stages.text.filters import (
    RepeatingTopNGramsFilter,
    SymbolsToWordsFilter,
    WordCountFilter,
)
from nemo_curator.stages.text.modules import ScoreFilter

EVIDENCE = os.path.dirname(os.path.abspath(__file__))
RAW_PARQUETS = sorted(
    os.path.join(EVIDENCE, p) for p in os.listdir(EVIDENCE)
    if p.startswith("wikitext103_train_") and p.endswith(".parquet")
)
WORK_PARQUETS_DIR = os.path.join(EVIDENCE, "raw_parquets_for_curator")
CLEANED_DIR = os.path.join(EVIDENCE, "cleaned")
STATS_PATH = os.path.join(EVIDENCE, "prep_stats.json")


def normalize_inputs() -> int:
    """Curator's ParquetReader expects a directory of parquets each with
    a `text` column; HuggingFace's wikitext parquets are already shaped
    that way. Just symlink them into a clean subdir to give Curator a
    stable input root."""
    os.makedirs(WORK_PARQUETS_DIR, exist_ok=True)
    n = 0
    for src in RAW_PARQUETS:
        dst = os.path.join(WORK_PARQUETS_DIR, os.path.basename(src))
        if not os.path.exists(dst):
            os.symlink(src, dst)
        n += 1
    return n


def count_input_chars() -> tuple[int, int]:
    """Sum total documents and characters in the raw parquets — gives us
    a baseline to compare against the post-filter total."""
    total_docs = 0
    total_chars = 0
    for p in RAW_PARQUETS:
        df = pd.read_parquet(p, columns=["text"])
        total_docs += len(df)
        total_chars += df["text"].str.len().sum()
    return total_docs, int(total_chars)


def count_cleaned() -> tuple[int, int]:
    """Sum docs + chars in cleaned JSONL output."""
    total_docs = 0
    total_chars = 0
    if not os.path.exists(CLEANED_DIR):
        return 0, 0
    for fn in os.listdir(CLEANED_DIR):
        if not fn.endswith(".jsonl"):
            continue
        with open(os.path.join(CLEANED_DIR, fn)) as f:
            for line in f:
                rec = json.loads(line)
                total_docs += 1
                total_chars += len(rec.get("text", ""))
    return total_docs, total_chars


def main() -> None:
    n = normalize_inputs()
    print(f"input parquets: {n}")

    print("counting raw inputs ...")
    t0 = time.perf_counter()
    raw_docs, raw_chars = count_input_chars()
    print(f"  raw docs={raw_docs:,}  raw chars={raw_chars:,}  ({time.perf_counter() - t0:.1f}s)")

    os.makedirs(CLEANED_DIR, exist_ok=True)

    p = (
        Pipeline(name="wikitext_clean")
        .add_stage(ParquetReader(file_paths=WORK_PARQUETS_DIR, fields=["text"]))
        .add_stage(Modify(UnicodeReformatter()))
        .add_stage(Modify(NewlineNormalizer()))
        .add_stage(ScoreFilter(WordCountFilter(min_words=50, max_words=100_000),
                               text_field="text",
                               score_field="word_count"))
        .add_stage(ScoreFilter(RepeatingTopNGramsFilter(n=3, max_repeating_ngram_ratio=0.18),
                               text_field="text",
                               score_field="ngram_repeat_3"))
        .add_stage(ScoreFilter(SymbolsToWordsFilter(max_symbol_to_word_ratio=0.20),
                               text_field="text",
                               score_field="symbol_to_word"))
        .add_stage(JsonlWriter(path=CLEANED_DIR))
    )

    print(f"\nrunning Curator pipeline:\n{p}")
    t0 = time.perf_counter()
    p.run()
    pipeline_s = time.perf_counter() - t0
    print(f"\npipeline wall: {pipeline_s:.1f}s")

    print("\ncounting cleaned outputs ...")
    cleaned_docs, cleaned_chars = count_cleaned()
    print(f"  cleaned docs={cleaned_docs:,}  cleaned chars={cleaned_chars:,}")
    print(f"  drop ratio: docs {1 - cleaned_docs/raw_docs:.1%}  chars {1 - cleaned_chars/raw_chars:.1%}")

    stats = {
        "stage": "curator_clean",
        "raw_docs": raw_docs,
        "raw_chars": raw_chars,
        "cleaned_docs": cleaned_docs,
        "cleaned_chars": cleaned_chars,
        "pipeline_wall_s": round(pipeline_s, 1),
        "drop_ratio_docs": round(1 - cleaned_docs/raw_docs, 4),
        "drop_ratio_chars": round(1 - cleaned_chars/raw_chars, 4),
        "input_parquets": [os.path.basename(p) for p in RAW_PARQUETS],
        "filters": [
            "WordCountFilter(min=50,max=100k)",
            "RepeatingTopNGramsFilter(n=3,max_ratio=0.18)",
            "SymbolsToWordsFilter(max_ratio=0.20)",
        ],
        "modifiers": ["UnicodeReformatter", "NewlineNormalizer"],
    }
    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nwrote {STATS_PATH}")


if __name__ == "__main__":
    main()
