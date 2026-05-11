"""Slice HotpotQA test.parquet down to the first N rows for tractable eval.

The full HotpotQA dev set is 7405 examples — at multi-turn rollout ~30s/example
that's 60+ hr per eval pass. We use a fixed N=200 prefix so both baseline and
trained-model eval pass over the same examples.

Determinism: first-N (not random sample) so the comparison row-by-row is
identical between baseline and trained run.
"""

import argparse
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--n", type=int, default=200)
    args = parser.parse_args()

    table = pq.read_table(args.input)
    sliced = table.slice(0, args.n)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(sliced, args.output)
    print(f"wrote {sliced.num_rows} rows ({len(sliced.column_names)} cols) to {args.output}")


if __name__ == "__main__":
    main()
