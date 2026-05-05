"""Flatten clawgym rollout Trajectory → NeMo-friendly SFT records.

Joins each trajectory back to its source task (for the initial workspace
listing the agent saw) and emits a single record per trajectory containing
the full system + user + assistant + observation chain. Downstream NeMo
SFT recipes consume this either as multi-turn chat (every assistant turn
is a loss target) or by flattening to (prompt, completion) per turn.

Output shape per line:
    {
      "task_id": "synth-data-science-researcher-00",
      "persona": "data-science-researcher",
      "n_turns": 7,
      "task_passed": true,
      "n_passed": 6, "n_total": 6,
      "system": "<SYSTEM_PROMPT>",
      "messages": [
        {"role": "user",      "content": "<initial user prompt>"},
        {"role": "assistant", "content": "<turn 1 raw response>"},
        {"role": "user",      "content": "OBSERVATION (exit 0):\\n..."},
        {"role": "assistant", "content": "<turn 2 raw response>"},
        ...
      ]
    }

This is co-located with rollout.py for now; it's a Phase 4 lift candidate
for `fieldkit.agents.sft_format` in v0.2.

Usage:
    python3 to_sft_record.py \\
        --tasks ../evidence/runs/2026-05-04-phase3-corpus/tasks-200.jsonl \\
        --trajectories ../evidence/runs/2026-05-04-phase3-baseline/sft-positives.jsonl \\
        --out ../evidence/runs/2026-05-04-phase3-baseline/sft-records.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Reuse the prompts from rollout.py so the SFT records are exactly what the
# agent saw at rollout time. If rollout.py's prompts ever drift, this import
# keeps us coherent.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rollout import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, render_files_block


def workspace_listing(task: dict) -> str:
    """Reconstruct the file listing the agent saw at turn 0.

    The rollout used `sandbox.list_files()` against the freshly-materialized
    seed; that's deterministic from the seed paths, sorted depth-first. We
    reconstruct it here from the seed file list so the SFT prompt matches
    the original prompt byte-for-byte.
    """
    paths = sorted(f["path"] for f in task["workspace_seed"]["files"])
    # rollout's list_files also includes parent dirs (find -type d). Fold
    # them in so the SFT record matches rollout output.
    dirs = set()
    for p in paths:
        parts = p.split("/")
        for i in range(1, len(parts)):
            dirs.add("/".join(parts[:i]))
    full = sorted(set(paths) | dirs)
    return render_files_block(full)


def render_observation(obs: dict | None) -> str:
    """Reproduce the exact OBSERVATION user-message that rollout sent next."""
    if obs is None:
        return ""
    flag = ", TIMED OUT" if obs.get("timed_out") else ""
    return (
        f"OBSERVATION (exit {obs.get('exit_code', '?')}{flag}):\n"
        f"--- stdout ---\n{obs.get('stdout', '')}\n"
        f"--- stderr ---\n{obs.get('stderr', '')}\n"
        "Next command (one ```bash``` block) or TASK_COMPLETE."
    )


def trajectory_to_sft(task: dict, traj: dict) -> dict:
    """Build a single SFT record from a (task, trajectory) pair."""
    initial_user = USER_PROMPT_TEMPLATE.format(
        intent=task["intent"],
        file_listing=workspace_listing(task),
    )

    messages = [{"role": "user", "content": initial_user}]
    for turn in traj["turns"]:
        # Assistant turn — always present.
        messages.append({"role": "assistant", "content": turn["agent_response"]})
        # Observation only present if the action ran (kind=shell, no parse error).
        # `done` actions (TASK_COMPLETE) and parse errors don't have an
        # observation; they end or trigger a corrective hint instead.
        if turn.get("observation") is not None:
            messages.append(
                {"role": "user", "content": render_observation(turn["observation"])}
            )
        elif turn.get("parse_error"):
            messages.append({
                "role": "user",
                "content": (
                    f"PARSE ERROR: {turn['parse_error']}. Reply with ONE ```bash``` block "
                    "containing one command, or TASK_COMPLETE on a line by itself."
                ),
            })

    grade = traj["final_grade"]
    return {
        "task_id": traj["task_id"],
        "persona": task["persona"]["role"],
        "n_turns": traj["n_turns"],
        "stopped": traj["stopped"],
        "task_passed": grade["passed"],
        "n_passed": grade["n_passed"],
        "n_total": grade["n_total"],
        "system": SYSTEM_PROMPT,
        "messages": messages,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True, help="JSONL of synth tasks")
    ap.add_argument("--trajectories", required=True, help="JSONL of trajectories to flatten")
    ap.add_argument("--out", required=True, help="JSONL output path")
    args = ap.parse_args()

    tasks_by_id = {}
    with open(args.tasks) as f:
        for line in f:
            t = json.loads(line)
            tasks_by_id[t["task_id"]] = t

    n = 0
    n_missing = 0
    with open(args.trajectories) as f, open(args.out, "w") as out:
        for line in f:
            traj = json.loads(line)
            task = tasks_by_id.get(traj["task_id"])
            if task is None:
                n_missing += 1
                continue
            rec = trajectory_to_sft(task, traj)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1

    print(f"wrote {n} SFT records → {args.out}")
    if n_missing:
        print(f"WARN: {n_missing} trajectories had no matching task in {args.tasks}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
