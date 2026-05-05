"""Phase 6 smoke: K-sample rollouts → reward → group-relative advantages.

Picks M tasks (default 4), generates K rollouts per task at exploration
temperature against vLLM-served Qwen 2.5 7B + clawgym LoRA, grades each,
computes shaped reward + group-relative advantages. Persists a
trajectory_bundle.jsonl that's already shaped like a GRPO batch.

Goal — *plumbing validation, not signal*. We want to see:
  1. K rollouts of the same task produce reward variance most of the time.
  2. The bundle JSONL is shaped correctly for downstream gradient code.
  3. The shaped-reward turn-cost penalty bites SFT trajectories that
     run out the clock — same diagnostic the offline replay produced.

A green smoke does not validate the GRPO learning loop end-to-end. It
validates that the substrate produces non-zero advantages on most tasks.
The full GRPO run wires this to a gradient step.

Usage:
    python3 phase6_smoke.py \\
        --tasks /home/nvidia/.../tasks-heldout-158.jsonl \\
        --task-ids synth-indie-game-dev-01,synth-ml-engineer-09,synth-technical-writer-04,synth-backend-developer-00 \\
        --k 4 \\
        --temperature 0.8 \\
        --vllm-base-url http://172.17.0.3:8000/v1 \\
        --model clawgym \\
        --out-dir articles/clawgym-on-spark/evidence/runs/2026-05-05-phase6-smoke/
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from grader import grade, seed_files_from_task  # noqa: E402
from reward import compute_group_advantages, compute_reward  # noqa: E402
from rollout import LocalTempSandbox, RolloutDriver  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--task-ids", required=True,
                    help="comma-separated list of task_ids to run")
    ap.add_argument("--k", type=int, default=4, help="rollouts per task")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--per-command-timeout", type=float, default=10.0)
    ap.add_argument("--vllm-base-url", default="http://172.17.0.3:8000/v1")
    ap.add_argument("--model", default="clawgym")
    ap.add_argument("--reward-mode", choices=["binary", "shaped"], default="shaped")
    ap.add_argument("--turn-penalty", type=float, default=0.2)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = out_dir / "trajectory_bundle.jsonl"
    summary_path = out_dir / "smoke_summary.json"
    post_states_root = out_dir / "post-states"
    post_states_root.mkdir(exist_ok=True)

    tasks: dict[str, dict[str, Any]] = {}
    with open(args.tasks) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            tasks[t["task_id"]] = t

    selected_ids = [tid.strip() for tid in args.task_ids.split(",") if tid.strip()]
    missing = [tid for tid in selected_ids if tid not in tasks]
    if missing:
        print(f"task_ids not in tasks file: {missing}", file=sys.stderr)
        return 2

    from fieldkit.nim import NIMClient
    agent = NIMClient(base_url=args.vllm_base_url, model=args.model, timeout=180.0)

    # Probe model registration before starting.
    if not agent.health():
        print(f"vLLM not healthy at {args.vllm_base_url}", file=sys.stderr)
        return 3

    bundle_records: list[dict] = []
    per_task_summary: list[dict] = []
    t_total = time.time()

    for tid in selected_ids:
        task = tasks[tid]
        seeds = seed_files_from_task(task)
        rollout_records: list[dict] = []
        rewards: list[float] = []
        n_turns_list: list[int] = []
        stops: list[str] = []

        for k in range(args.k):
            post_state_root = post_states_root / f"{tid}__rollout-{k}"
            driver = RolloutDriver(
                agent_client=agent,
                model_name=args.model,
                sandbox_factory=lambda root=post_state_root: LocalTempSandbox(root=root),
                max_turns=args.max_turns,
                per_command_timeout=args.per_command_timeout,
                debug=args.debug,
                temperature=args.temperature,
            )
            t0 = time.time()
            traj, final_sb = driver.rollout(task)
            grade_result = grade(task, final_sb.root, seed_files=seeds)
            traj.final_grade = grade_result.to_dict()

            r = compute_reward(grade_result, n_turns=traj.n_turns,
                                mode=args.reward_mode,
                                max_turns=args.max_turns,
                                turn_penalty=args.turn_penalty)
            rewards.append(r)
            n_turns_list.append(traj.n_turns)
            stops.append(traj.stopped)
            rollout_records.append({
                "rollout_idx": k,
                "n_turns": traj.n_turns,
                "stopped": traj.stopped,
                "wall_seconds": round(traj.wall_seconds, 2),
                "n_passed": grade_result.n_passed,
                "n_total": grade_result.n_total,
                "passed": grade_result.passed,
                "reward": round(r, 4),
                "trajectory": json.loads(traj.to_jsonl()),
            })
            print(f"  {tid}  rollout {k}  turns={traj.n_turns:>2}  "
                  f"asrt={grade_result.n_passed}/{grade_result.n_total}  "
                  f"r={r:.3f}  wall={traj.wall_seconds:.1f}s  stop={traj.stopped}",
                  flush=True)

        advantages = compute_group_advantages(rewards)
        for rec, adv in zip(rollout_records, advantages):
            rec["advantage"] = round(adv, 4)

        rewards_stats = {
            "mean": round(statistics.mean(rewards), 4),
            "stdev": round(statistics.pstdev(rewards), 4),
            "min": round(min(rewards), 4),
            "max": round(max(rewards), 4),
            "spread": round(max(rewards) - min(rewards), 4),
        }
        bundle_records.append({
            "task_id": tid,
            "k": args.k,
            "temperature": args.temperature,
            "reward_mode": args.reward_mode,
            "rewards": [round(r, 4) for r in rewards],
            "advantages": [round(a, 4) for a in advantages],
            "n_turns": n_turns_list,
            "stops": stops,
            "rewards_stats": rewards_stats,
            "rollouts": rollout_records,
        })
        per_task_summary.append({
            "task_id": tid,
            **rewards_stats,
            "n_passed_rollouts": sum(1 for r in rollout_records if r["passed"]),
            "advantage_signal": rewards_stats["stdev"] > 1e-6,
        })
        print(f"  → {tid}  rewards={[round(r,3) for r in rewards]}  "
              f"stdev={rewards_stats['stdev']:.3f}  "
              f"signal={'yes' if rewards_stats['stdev'] > 1e-6 else 'NO'}",
              flush=True)
        print()

    with bundle_path.open("w") as fh:
        for rec in bundle_records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n_signal = sum(1 for s in per_task_summary if s["advantage_signal"])
    overall_summary = {
        "model": args.model,
        "vllm_base_url": args.vllm_base_url,
        "k": args.k,
        "temperature": args.temperature,
        "reward_mode": args.reward_mode,
        "turn_penalty": args.turn_penalty,
        "n_tasks": len(selected_ids),
        "n_tasks_with_signal": n_signal,
        "wall_seconds_total": round(time.time() - t_total, 1),
        "per_task": per_task_summary,
    }
    summary_path.write_text(json.dumps(overall_summary, indent=2) + "\n")

    print(f"\nbundle  → {bundle_path}")
    print(f"summary → {summary_path}")
    print(f"signal: {n_signal}/{len(selected_ids)} tasks have non-zero advantage variance")
    print(f"total wall: {overall_summary['wall_seconds_total']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
