#!/usr/bin/env python3
"""Phase 5 matched-base comparison: Qwen base vs Qwen+clawgym on the held-out 158.

Reads two trajectories.jsonl files (baseline A vs SFT B) and emits:
  - overall: task-pass + per-assertion + Δ in pp
  - per-persona: task-pass A/B/Δ
  - per-assertion-kind: pass-rate A/B/Δ
  - stop-reason mix
JSON dump alongside human-readable table.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def load(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def persona_of(task_id: str) -> str:
    # task_id shape: synth-<persona>-NN  (persona may itself contain hyphens)
    parts = task_id.split("-")
    # drop "synth" prefix and trailing 2-digit index
    return "-".join(parts[1:-1])


def stats(rows):
    n = len(rows)
    passed = sum(1 for r in rows if r["final_grade"]["passed"])
    ap = sum(r["final_grade"]["n_passed"] for r in rows)
    at = sum(r["final_grade"]["n_total"] for r in rows)
    by_persona = defaultdict(lambda: {"n": 0, "p": 0, "ap": 0, "at": 0})
    by_kind = defaultdict(lambda: {"p": 0, "n": 0})
    stops = defaultdict(int)
    turns = []
    walls = []
    for r in rows:
        g = r["final_grade"]
        per = persona_of(r["task_id"])
        by_persona[per]["n"] += 1
        by_persona[per]["p"] += int(g["passed"])
        by_persona[per]["ap"] += g["n_passed"]
        by_persona[per]["at"] += g["n_total"]
        for a in g["assertions"]:
            by_kind[a["kind"]]["n"] += 1
            by_kind[a["kind"]]["p"] += int(a["passed"])
        stops[r.get("stopped", "?")] += 1
        turns.append(r.get("n_turns", 0))
        walls.append(r.get("wall_seconds", 0))
    return {
        "n": n, "passed": passed, "ap": ap, "at": at,
        "by_persona": dict(by_persona), "by_kind": dict(by_kind),
        "stops": dict(stops), "mean_turns": (sum(turns) / n if n else 0),
        "mean_wall": (sum(walls) / n if n else 0),
    }


def fmt_pct(x, d=1):
    return f"{x:.{d}f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="trajectories.jsonl for baseline (Qwen base)")
    ap.add_argument("--sft", required=True, help="trajectories.jsonl for Qwen + clawgym")
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    A = stats(load(args.base))
    B = stats(load(args.sft))

    print(f"=== Phase 5 — matched-base eval ===")
    print(f"baseline (Qwen base): {args.base}")
    print(f"sft      (Qwen+lora): {args.sft}")
    print()
    pa = 100 * A["passed"] / A["n"] if A["n"] else 0
    pb = 100 * B["passed"] / B["n"] if B["n"] else 0
    paa = 100 * A["ap"] / A["at"] if A["at"] else 0
    pbb = 100 * B["ap"] / B["at"] if B["at"] else 0
    print(f"OVERALL")
    print(f"  task-pass     base={A['passed']}/{A['n']} ({fmt_pct(pa)})  sft={B['passed']}/{B['n']} ({fmt_pct(pb)})  Δ={pb-pa:+.1f}pp")
    print(f"  per-assertion base={A['ap']}/{A['at']} ({fmt_pct(paa)})  sft={B['ap']}/{B['at']} ({fmt_pct(pbb)})  Δ={pbb-paa:+.1f}pp")
    print(f"  mean turns  base={A['mean_turns']:.2f}  sft={B['mean_turns']:.2f}")
    print(f"  mean wall   base={A['mean_wall']:.1f}s  sft={B['mean_wall']:.1f}s")
    print(f"  stop mix    base={A['stops']}  sft={B['stops']}")
    print()

    print(f"PER-PERSONA  (rows sorted by base task-pass desc)")
    print(f"  {'persona':30s} {'base task%':>11s} {'sft task%':>10s} {'Δ pp':>7s}  {'base asrt%':>11s} {'sft asrt%':>10s} {'Δ pp':>7s}")
    rows = []
    for per in sorted(set(A["by_persona"]) | set(B["by_persona"])):
        a = A["by_persona"].get(per, {"n": 0, "p": 0, "ap": 0, "at": 0})
        b = B["by_persona"].get(per, {"n": 0, "p": 0, "ap": 0, "at": 0})
        ta = 100 * a["p"] / a["n"] if a["n"] else 0
        tb = 100 * b["p"] / b["n"] if b["n"] else 0
        aa = 100 * a["ap"] / a["at"] if a["at"] else 0
        bb = 100 * b["ap"] / b["at"] if b["at"] else 0
        rows.append((per, a, b, ta, tb, aa, bb))
    rows.sort(key=lambda x: -x[3])
    for per, a, b, ta, tb, aa, bb in rows:
        print(f"  {per:30s} {a['p']:>3d}/{a['n']:<3d}={ta:>5.1f}%  {b['p']:>3d}/{b['n']:<3d}={tb:>5.1f}% {tb-ta:>+6.1f}  {aa:>10.1f}% {bb:>9.1f}% {bb-aa:>+6.1f}")
    print()

    print(f"PER-ASSERTION-KIND")
    kinds = sorted(set(A["by_kind"]) | set(B["by_kind"]))
    print(f"  {'kind':30s} {'base':>13s} {'sft':>13s} {'Δ pp':>7s}")
    for k in kinds:
        a = A["by_kind"].get(k, {"p": 0, "n": 0})
        b = B["by_kind"].get(k, {"p": 0, "n": 0})
        ra = 100 * a["p"] / a["n"] if a["n"] else 0
        rb = 100 * b["p"] / b["n"] if b["n"] else 0
        print(f"  {k:30s} {a['p']:>4d}/{a['n']:<4d}={ra:>5.1f}%  {b['p']:>4d}/{b['n']:<4d}={rb:>5.1f}% {rb-ra:>+6.1f}")

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump({"baseline": A, "sft": B,
                   "overall": {
                       "base_task_pass_pct": pa, "sft_task_pass_pct": pb, "delta_task_pp": pb - pa,
                       "base_assert_pct": paa, "sft_assert_pct": pbb, "delta_assert_pp": pbb - paa,
                   }}, f, indent=2)
    print(f"\n{args.out_json} written")


if __name__ == "__main__":
    main()
