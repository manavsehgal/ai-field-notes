"""Post-run analysis for a T²PO loop directory.

Reads every step-NNN/adapter/t2po_step_summary.json + every step-NNN/trajectory_bundle.jsonl
in --run-dir and emits:
  - a CSV of per-step training metrics + on-policy reward stats
  - a markdown summary suitable for HANDOFF.md amendment

Usage:
    python3 analyze_t2po_run.py --run-dir .../2026-05-08-t2po-grpo/

Cloned from articles/clawgym-on-spark/scripts/analyze_grpo_run.py — only diff is the
summary filename (t2po_step_summary.json vs grpo_step_summary.json) and the eval label
in the run summary, plus an extra column for mean tds_regen_attempts per step.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path


def per_step_row(step_dir: Path) -> dict | None:
    summary_path = step_dir / "adapter" / "t2po_step_summary.json"
    bundle_path = step_dir / "trajectory_bundle.jsonl"
    if not summary_path.exists() or not bundle_path.exists():
        return None
    s = json.loads(summary_path.read_text())
    rewards: list[float] = []
    n_turns_all: list[int] = []
    stops: list[str] = []
    tds_regens: list[int] = []
    n_pass = 0
    n_groups = 0
    zero_var_groups = 0
    for line in bundle_path.open():
        line = line.strip()
        if not line:
            continue
        g = json.loads(line)
        n_groups += 1
        rewards += g["rewards"]
        n_turns_all += g["n_turns"]
        stops += g["stops"]
        if g["rewards_stats"]["stdev"] < 1e-6:
            zero_var_groups += 1
        for r in g["rollouts"]:
            if r.get("passed"):
                n_pass += 1
            if "tds_regen_attempts" in r:
                tds_regens.append(r["tds_regen_attempts"])
    n_rollouts = len(rewards)
    return {
        "step": int(step_dir.name.split("-")[-1]),
        "loss": s["mean_loss"],
        "policy_loss": s["mean_policy_loss"],
        "kl_loss": s["mean_kl_loss"],
        "grad_norm": s["grad_norm_pre_clip"],
        "weight_delta_l2": s["weight_delta_l2"],
        "n_rollouts_used": s["n_rollouts_used"],
        "n_rollouts_total": n_rollouts,
        "train_wall": s["train_wall_seconds"],
        "n_groups": n_groups,
        "zero_var_groups": zero_var_groups,
        "task_pass_count": n_pass,
        "task_pass_rate": round(n_pass / n_rollouts, 3) if n_rollouts else 0,
        "reward_mean": round(statistics.mean(rewards), 3),
        "reward_stdev": round(statistics.pstdev(rewards), 3),
        "reward_min": round(min(rewards), 3),
        "reward_max": round(max(rewards), 3),
        "mean_turns": round(statistics.mean(n_turns_all), 2),
        "median_turns": statistics.median(n_turns_all),
        "task_complete_pct": round(
            sum(1 for x in stops if x == "task_complete") / n_rollouts * 100, 1
        ),
        "max_turns_pct": round(
            sum(1 for x in stops if x == "max_turns") / n_rollouts * 100, 1
        ),
        "mean_tds_regen": round(statistics.mean(tds_regens), 2) if tds_regens else 0,
    }


def eval_summary(eval_dir: Path) -> dict | None:
    cmp_path = eval_dir / "comparison.json"
    if not cmp_path.exists():
        return None
    cmp = json.loads(cmp_path.read_text())
    base = cmp.get("baseline", {})
    t2po = cmp.get("sft", {})  # the loop labels T²PO-eval as "sft"
    overall = cmp.get("overall", {})
    out = {
        "step": int(eval_dir.name.split("-")[-1]),
        "n": t2po.get("n"),
        "qwen_base_task_pass": base.get("passed"),
        "qwen_base_assertion": f"{base.get('ap')}/{base.get('at')}",
        "t2po_task_pass": t2po.get("passed"),
        "t2po_assertion": f"{t2po.get('ap')}/{t2po.get('at')}",
        "t2po_mean_turns": t2po.get("mean_turns"),
        "t2po_task_complete": (t2po.get("stops", {}).get("task_complete", 0)
                              if isinstance(t2po.get("stops"), dict) else None),
        "t2po_max_turns": (t2po.get("stops", {}).get("max_turns", 0)
                          if isinstance(t2po.get("stops"), dict) else None),
        "delta_task_pp_vs_base": overall.get("delta_task_pp"),
        "delta_assert_pp_vs_base": overall.get("delta_assert_pp"),
    }
    for label, fname in [
        ("phase5_sft", "compare_vs_phase5_sft.json"),
        ("phase6_grpo34", "compare_vs_phase6_grpo34.json"),
        ("t2po_25", "compare_vs_t2po_25.json"),
    ]:
        path = eval_dir / fname
        if path.exists():
            c = json.loads(path.read_text())
            out[f"delta_task_pp_vs_{label}"] = c.get("overall", {}).get("delta_task_pp")
            out[f"delta_assert_pp_vs_{label}"] = c.get("overall", {}).get("delta_assert_pp")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args()

    run = Path(args.run_dir)
    if not run.is_dir():
        print(f"not a dir: {run}", file=sys.stderr)
        return 2

    rows: list[dict] = []
    for step_dir in sorted(run.glob("step-*")):
        if "failed" in step_dir.name:
            continue
        row = per_step_row(step_dir)
        if row:
            rows.append(row)

    eval_rows: list[dict] = []
    for eval_dir in sorted(run.glob("eval-step-*")):
        e = eval_summary(eval_dir)
        if e:
            eval_rows.append(e)

    out_csv = Path(args.out_csv) if args.out_csv else run / "per_step_metrics.csv"
    if rows:
        with out_csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {out_csv}  ({len(rows)} steps)")

    if rows:
        first, last = rows[0], rows[-1]
        kl_max = max(r["kl_loss"] for r in rows)
        mean_turns_min = min(r["mean_turns"] for r in rows)
        mean_turns_min_step = next(r["step"] for r in rows if r["mean_turns"] == mean_turns_min)
        tc_max = max(r["task_complete_pct"] for r in rows)
        tc_max_step = next(r["step"] for r in rows if r["task_complete_pct"] == tc_max)
        pass_max = max(r["task_pass_count"] for r in rows)
        pass_max_step = next(r["step"] for r in rows if r["task_pass_count"] == pass_max)
        total_train_wall = sum(r["train_wall"] for r in rows)
        mean_tds = round(statistics.mean(r["mean_tds_regen"] for r in rows if r["mean_tds_regen"]), 2)

    out_md = Path(args.out_md) if args.out_md else run / "SUMMARY.md"
    with out_md.open("w") as fh:
        fh.write(f"# T²PO run summary — {run.name}\n\n")
        fh.write(f"Steps completed: **{len(rows)}** of (configured) 50.\n\n")
        if rows:
            fh.write("## Per-step training\n\n")
            fh.write(f"- step 001 mean_turns={first['mean_turns']}, "
                     f"task_complete%={first['task_complete_pct']}%\n")
            fh.write(f"- step {last['step']:03d} mean_turns={last['mean_turns']}, "
                     f"task_complete%={last['task_complete_pct']}%\n")
            fh.write(f"- best mean_turns: **{mean_turns_min}** at step {mean_turns_min_step:03d}\n")
            fh.write(f"- best task_complete%: **{tc_max}%** at step {tc_max_step:03d}\n")
            fh.write(f"- best task_pass: **{pass_max}/32** at step {pass_max_step:03d}\n")
            fh.write(f"- max KL: {kl_max:.4f}\n")
            fh.write(f"- mean TDS regenerations per rollout: {mean_tds}\n")
            fh.write(f"- total trainer wall: {total_train_wall/60:.1f} min\n\n")
        if eval_rows:
            fh.write("## Eval rollouts (158 held-out)\n\n")
            fh.write("| step | task_pass | per-asrt | mean turns | task_complete | Δtask vs base | Δtask vs P5 SFT | Δtask vs P6 GRPO@34 |\n")
            fh.write("|---:|---:|---:|---:|---:|---:|---:|---:|\n")
            for e in eval_rows:
                tc = e.get("t2po_task_complete")
                tc_str = f"{tc}/{e['n']}" if tc is not None and e.get('n') else "?"
                d5 = e.get("delta_task_pp_vs_phase5_sft")
                d6 = e.get("delta_task_pp_vs_phase6_grpo34")
                d5_s = f"{d5:+.1f}pp" if d5 is not None else "—"
                d6_s = f"{d6:+.1f}pp" if d6 is not None else "—"
                fh.write(
                    f"| {e['step']} | "
                    f"{e['t2po_task_pass']}/{e['n']} | "
                    f"{e['t2po_assertion']} | "
                    f"{e.get('t2po_mean_turns', '—')} | "
                    f"{tc_str} | "
                    f"{e['delta_task_pp_vs_base']:+.1f}pp | "
                    f"{d5_s} | {d6_s} |\n"
                )
        else:
            fh.write("## Eval rollouts\n\n_no eval-step-*/comparison.json files found yet_\n")
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
