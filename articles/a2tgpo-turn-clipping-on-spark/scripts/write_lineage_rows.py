"""Post-run lineage row writer for the overnight A²TGPO run.

Reads:
  - {run_dir}/train_exit_code      (124 = timeout, 0 = clean, else = crash)
  - {run_outputs}/run.log          (verl stdout; per-step IG stats lines)
  - {run_outputs}/validation/*.jsonl  (per-checkpoint validation rollouts; each row
                                       has `score` and `em_score` from compute_score)

Writes:
  - {lineage_root}/results.tsv     (LineageStore-managed; two rows)
  - {lineage_root}/lineage-rendered.txt  (deterministic prompt for next specialist)

Mapping from train_exit_code → exp_002 status:
  - 124 (timeout):           FailureLabel.TRAIN_BUDGET_OVERRUN
  - 0 (clean termination):   keep if final_em ≥ baseline_em + 0.5pp else discard
  - other (crash):           FailureLabel.CRASH
"""

import argparse
import json
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fieldkit.lineage import FailureLabel, LineageStore, Trial


KEEP_THRESHOLD_PP = 0.5  # EM% gain over baseline required to mark as keep


def parse_em_from_validation_dir(validation_dir: Path) -> List[Tuple[int, float, int]]:
    """Return [(step, em_pct, n_examples), ...] sorted by step.

    Verl writes validation rollouts as JSONL with one record per example. Each
    record carries `score` (compute_score's primary) and `em_score` (binary).
    We use em_score, average over examples → EM% × 100.
    """
    out: List[Tuple[int, float, int]] = []
    for path in sorted(validation_dir.glob("**/*.jsonl")):
        m = re.search(r"step[_-]?(\d+)", str(path))
        step = int(m.group(1)) if m else 0
        ems: List[float] = []
        with path.open() as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "em_score" in rec:
                    ems.append(float(rec["em_score"]))
                elif "reward" in rec and isinstance(rec["reward"], dict) and "em_score" in rec["reward"]:
                    ems.append(float(rec["reward"]["em_score"]))
        if ems:
            out.append((step, 100.0 * sum(ems) / len(ems), len(ems)))
    out.sort(key=lambda t: t[0])
    return out


def parse_ig_stats_from_log(run_log: Path) -> Dict[str, float]:
    """Best-effort parse of per-step IG stats from run.log.

    Verl logs scalars per step. We grep for ig_clip_scale + ig_normed_mean style
    keys; return mean of last 50 step values for each. Missing keys → empty dict.
    """
    if not run_log.exists():
        return {}
    pattern = re.compile(r"(ig_[A-Za-z_]+|igpo_[A-Za-z_]+|adv_rescale_[A-Za-z_]+)\s*[:=]\s*(-?[0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)")
    series: Dict[str, List[float]] = {}
    for line in run_log.read_text(errors="ignore").splitlines():
        for key, val in pattern.findall(line):
            try:
                series.setdefault(key, []).append(float(val))
            except ValueError:
                continue
    summary: Dict[str, float] = {}
    for key, vals in series.items():
        tail = vals[-50:] if len(vals) > 50 else vals
        if tail:
            summary[f"{key}_mean"] = statistics.mean(tail)
            summary[f"{key}_std"] = statistics.pstdev(tail) if len(tail) > 1 else 0.0
    return summary


def parse_step_count_from_log(run_log: Path) -> int:
    """Count the number of `Step N` markers reached."""
    if not run_log.exists():
        return 0
    steps = set()
    for m in re.finditer(r"(?:^|\s)step[\s_=:]+(\d+)", run_log.read_text(errors="ignore"), re.IGNORECASE):
        steps.add(int(m.group(1)))
    return max(steps) if steps else 0


def parse_train_seconds(run_log: Path) -> Optional[float]:
    """Total training wall by reading first/last timestamp markers."""
    if not run_log.exists():
        return None
    return None  # placeholder; verl's log format varies, fill in after first run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, required=True,
                        help="Directory holding scope.md + train_exit_code + the lineage/ subdir")
    parser.add_argument("--run_outputs", type=Path, required=True,
                        help="Directory verl wrote run.log + validation/ + checkpoints/ to")
    parser.add_argument("--baseline_specialist", default="grpo-baseline")
    parser.add_argument("--run_specialist", default="a2tgpo-v1d")
    parser.add_argument("--keep_threshold_pp", type=float, default=KEEP_THRESHOLD_PP)
    args = parser.parse_args()

    lineage_root = args.run_dir / "lineage"
    lineage_root.mkdir(parents=True, exist_ok=True)
    store = LineageStore(lineage_root, lower_is_better=False)

    exit_code_path = args.run_dir / "train_exit_code"
    exit_code = int(exit_code_path.read_text().strip()) if exit_code_path.exists() else -1

    val_curve = parse_em_from_validation_dir(args.run_outputs / "validation")
    if not val_curve:
        print(f"WARNING: no validation EMs found at {args.run_outputs / 'validation'}; writing crash row")
        baseline_em = 0.0
        final_em = 0.0
    else:
        baseline_em = val_curve[0][1]
        final_em = val_curve[-1][1]

    ig_stats = parse_ig_stats_from_log(args.run_outputs / "run.log")
    final_step = parse_step_count_from_log(args.run_outputs / "run.log")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # exp_001 — baseline (val_before_train)
    exp_001 = Trial(
        exp_id="exp_001",
        timestamp=now,
        specialist=args.baseline_specialist,
        parent_exp="",
        baseline_exp="exp_001",
        domain="agentic-grpo-multihop",
        hypothesis=(
            "Vanilla Qwen3-4B rolled through verl's sync_with_tool path "
            "against the local wiki-18 e5_Flat retriever on 200-sample HotpotQA "
            "dev. No PPO updates — measures the prompt-template+retriever EM "
            "ceiling before A²TGPO touches it."
        ),
        expected_delta="",
        status=FailureLabel.BASELINE.value,
        core_metric=baseline_em,
        val_bpb=None,
        delta_vs_best=None,
        train_s=0.0,
        total_s=0.0,
        job_name="a2tgpo-overnight-baseline",
        snapshot_path="snapshots/exp_001",
        notes=(
            f"baseline_em_pct={baseline_em:.3f} from val_before_train pass on "
            f"hotpotqa-multihop test-200; faiss_gpu=False; "
            f"gpu_memory_utilization=0.5; n_val_examples={val_curve[0][2] if val_curve else 0}"
        ),
    )
    store.append(exp_001)

    # exp_002 — A²TGPO run
    delta = final_em - baseline_em
    if exit_code == 124:
        status = FailureLabel.TRAIN_BUDGET_OVERRUN.value
    elif exit_code != 0:
        status = FailureLabel.CRASH.value
    else:
        status = FailureLabel.KEEP.value if delta >= args.keep_threshold_pp else FailureLabel.DISCARD.value

    ig_notes = ", ".join(f"{k}={v:.4f}" for k, v in sorted(ig_stats.items())[:10]) if ig_stats else "no IG stats parsed"

    exp_002 = Trial(
        exp_id="exp_002",
        timestamp=now,
        specialist=args.run_specialist,
        parent_exp="exp_001",
        baseline_exp="exp_001",
        domain="agentic-grpo-multihop",
        hypothesis=(
            "A²TGPO turn-group-v1d on Qwen3-4B + HotpotQA multi-hop; "
            "8h time-boxed on 1×GB10 with quartered batches (16/2/8 vs paper's "
            "64/8/16). One trial — the receipt for the MTBM Ch10 'forge → trial' beat."
        ),
        expected_delta=f"≥ +{args.keep_threshold_pp:.1f}pp EM over baseline",
        status=status,
        core_metric=final_em,
        val_bpb=None,
        delta_vs_best=delta,
        train_s=None,
        total_s=None,
        job_name="a2tgpo-overnight-run",
        snapshot_path="snapshots/exp_002",
        notes=(
            f"exit_code={exit_code} final_step={final_step} "
            f"em_curve={[round(em, 2) for _, em, _ in val_curve]} "
            f"ig_stats={{{ig_notes}}}"
        ),
    )
    store.append(exp_002)

    # Render the prompt the NEXT specialist would see
    snap = store.render_prompt(
        for_specialist="a2tgpo-overnight-next",
        top_k=3,
        recent_n=10,
        last_m_full=2,
    )
    (args.run_dir / "lineage-rendered.txt").write_text(snap.rendered_prompt)

    # Drop a summary JSON for the article-side harness
    summary = {
        "exp_001_baseline_em_pct": baseline_em,
        "exp_002_final_em_pct": final_em,
        "delta_pp": delta,
        "exit_code": exit_code,
        "final_step": final_step,
        "n_val_checkpoints": len(val_curve),
        "em_curve": [{"step": s, "em_pct": em, "n": n} for s, em, n in val_curve],
        "ig_stats_tail": ig_stats,
        "exp_002_status": status,
    }
    (args.run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"wrote {lineage_root / 'results.tsv'} (2 rows)")
    print(f"wrote {args.run_dir / 'lineage-rendered.txt'}")
    print(f"wrote {args.run_dir / 'summary.json'}")
    print(f"exp_002 status: {status} (baseline {baseline_em:.2f}% → final {final_em:.2f}%, Δ {delta:+.2f}pp)")


if __name__ == "__main__":
    main()
