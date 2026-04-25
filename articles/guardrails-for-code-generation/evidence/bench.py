"""
A5 adversarial bench — runs every test case in cases.json through the
rails defined in rails.py and reports precision / recall.

A "block recall" of 1.0 means every UNSAFE case was correctly stopped
at some rail. A "clean pass" of 1.0 means every SAFE case made it
through to the diff stage with all rails green. Anything less is a
real failure mode worth analyzing.

Each case has:
  id           : human-readable
  expect       : "pass" | "block"
  raw          : the LLM's hypothetical proposal output (string)
  expected_rail: (optional) which rail SHOULD catch this if expected=block

Outputs:
  evidence/bench_results.json — full per-case verdict
  stdout — summary table
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import Counter

EVIDENCE = Path(__file__).resolve().parent
sys.path.insert(0, str(EVIDENCE))
from rails import gate, load_menu  # noqa: E402

CASES_PATH = EVIDENCE / "cases.json"
RESULTS_PATH = EVIDENCE / "bench_results.json"

BASELINE_CFG = {
    "n_layer": 24, "n_head": 16, "d_model": 1024, "d_ff": 4096,
    "lr": 3e-4, "lr_warmup": 5, "grad_clip": 1.0, "weight_decay": 0.0,
    "beta1": 0.9, "beta2": 0.95,
    "batch_size": 16, "seq_len": 1024, "precision": "fp8",
}


def main() -> None:
    menu = load_menu()
    with open(CASES_PATH) as f:
        cases = json.load(f)

    rows = []
    for c in cases:
        v = gate(c["raw"], BASELINE_CFG, menu)
        verdict = "pass" if v.ok else "block"
        correct = (verdict == c["expect"])
        rail_correct = (
            c.get("expected_rail") is None or
            v.ok or
            v.rail == c["expected_rail"]
        )
        rows.append({
            "id": c["id"],
            "expect": c["expect"],
            "got": verdict,
            "rail": v.rail,
            "reason": v.reason,
            "correct_outcome": correct,
            "correct_rail": rail_correct,
            "expected_rail": c.get("expected_rail"),
        })

    n = len(rows)
    safe = [r for r in rows if r["expect"] == "pass"]
    unsafe = [r for r in rows if r["expect"] == "block"]

    correct_outcomes = sum(1 for r in rows if r["correct_outcome"])
    block_recall = (
        sum(1 for r in unsafe if r["got"] == "block") / max(1, len(unsafe))
    )
    clean_pass = (
        sum(1 for r in safe if r["got"] == "pass") / max(1, len(safe))
    )
    rail_correct = (
        sum(1 for r in unsafe if r["correct_rail"]) / max(1, len(unsafe))
    )

    rail_counts = Counter(r["rail"] for r in unsafe if r["got"] == "block")

    print(f"=== A5 rails bench — {n} cases ({len(safe)} safe, {len(unsafe)} unsafe) ===\n")
    for r in rows:
        mark = "✓" if r["correct_outcome"] else "✗"
        rail_mark = "" if r["correct_rail"] else f" (expected {r['expected_rail']})"
        print(f"  {mark} {r['id']:34s}  expect={r['expect']:>5s}  "
              f"got={r['got']:>5s}  rail={r['rail']:<14s}{rail_mark}  "
              f"{r['reason'][:60]}")

    print(f"\nblock recall (unsafe → block) : {block_recall:.2f}  "
          f"({sum(1 for r in unsafe if r['got']=='block')}/{len(unsafe)})")
    print(f"clean pass (safe → pass)      : {clean_pass:.2f}  "
          f"({sum(1 for r in safe if r['got']=='pass')}/{len(safe)})")
    print(f"correct rail attribution      : {rail_correct:.2f}  "
          f"({sum(1 for r in unsafe if r['correct_rail'])}/{len(unsafe)})")
    print(f"overall accuracy              : {correct_outcomes/n:.2f}  ({correct_outcomes}/{n})")
    print(f"\nblock distribution by rail:")
    for rail, count in sorted(rail_counts.items()):
        print(f"  {rail:14s} {count}")

    summary = {
        "n_cases": n,
        "n_safe": len(safe),
        "n_unsafe": len(unsafe),
        "block_recall": round(block_recall, 4),
        "clean_pass_rate": round(clean_pass, 4),
        "correct_rail_attribution": round(rail_correct, 4),
        "overall_accuracy": round(correct_outcomes / n, 4),
        "block_distribution_by_rail": dict(rail_counts),
        "baseline_cfg": BASELINE_CFG,
        "rows": rows,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
