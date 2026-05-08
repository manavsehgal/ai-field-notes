"""Build the GRPO training pool from Phase 4's SFT training tasks.

Phase 5 split was: 42 SFT-training tasks (17 PASS + 25 near-miss in Phase 3
baseline) vs 158 held-out for matched-base eval. The GRPO loop trains on
the same 42 — never the held-out — but biases sampling toward the
near-miss subset because under temp=0.8 those produce wider reward
spread (the smoke confirmed: same-task K=4 spread of 0.05–0.63).

Inputs:
  --sft-records-and-near    JSONL of 42 SFT records (has task_id +
                            task_passed). At /work/clawgym-sft/sft-records-and-near.jsonl
                            inside tllm-build, also at the matching path
                            under articles/.../runs/2026-05-04-phase3-baseline/.
  --tasks-all               Full 200-task synth corpus (the SFT records
                            don't carry the workspace seeds + assertions
                            we need at rollout time). At /work/clawgym-sft/tasks-200.jsonl
                            inside tllm-build, also at evidence/runs/2026-05-04-phase3-corpus/tasks-200.jsonl.

Output: tasks-grpo-pool.jsonl with one task per line. Each task carries
the original synth-corpus shape (intent, assertions, workspace_seed,
persona, ...) plus a `_phase4_passed` annotation for downstream filters.

Usage:
    python3 build_grpo_pool.py \\
        --sft-records-and-near .../sft-records-and-near.jsonl \\
        --tasks-all            .../tasks-200.jsonl \\
        --out                  tasks-grpo-pool.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-records-and-near", required=True)
    ap.add_argument("--tasks-all", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sft_records = load_jsonl(Path(args.sft_records_and_near))
    tasks_all = load_jsonl(Path(args.tasks_all))

    sft_meta_by_id: dict[str, dict] = {}
    for r in sft_records:
        sft_meta_by_id[r["task_id"]] = {
            "passed": bool(r.get("task_passed", False)),
            "n_passed": int(r.get("n_passed", 0)),
            "n_total": int(r.get("n_total", 0)),
        }

    tasks_by_id: dict[str, dict] = {t["task_id"]: t for t in tasks_all}

    pool: list[dict] = []
    missing: list[str] = []
    for tid, meta in sft_meta_by_id.items():
        task = tasks_by_id.get(tid)
        if task is None:
            missing.append(tid)
            continue
        annotated = dict(task)
        annotated["_phase4_passed"] = meta["passed"]
        annotated["_phase4_assertion_rate"] = (
            meta["n_passed"] / meta["n_total"] if meta["n_total"] else 0.0
        )
        pool.append(annotated)

    if missing:
        print(f"WARN: {len(missing)} SFT task_ids absent from --tasks-all: "
              f"{missing[:5]}…", file=sys.stderr)

    n_pass = sum(1 for t in pool if t["_phase4_passed"])
    n_near = len(pool) - n_pass
    print(f"loaded {len(sft_records)} SFT records, "
          f"{len(tasks_all)} tasks total → "
          f"{len(pool)} pool entries ({n_pass} PASS, {n_near} near-miss)",
          file=sys.stderr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for t in pool:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
