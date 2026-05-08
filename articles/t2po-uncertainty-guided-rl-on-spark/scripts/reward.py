"""Reward shaping + advantages for the T²PO-on-Spark trainer.

Forked from `articles/clawgym-on-spark/scripts/reward.py` (Phase 6
GRPO). Adds a GiGPO-style step-level advantage helper on top of the
existing trajectory-level group-relative advantage so the trainer can
form `A_token = α · A_traj + β · A_step[turn]`.

GiGPO (the per-step head from Verl-Agent) groups turns by *anchor
state* — the raw observation a turn lands in — and normalizes turn
rewards within each anchor group. ClawGym observations are continuous
(stdout/stderr/exit_code over arbitrary file system state), so exact
anchor matching is meaningless. We adapt by grouping at the same
*turn index* across the K rollouts of the same task; the per-turn
signal is `1.0 if exit_code == 0 and not parse_error else 0.0`. This
preserves the "reward good intermediate decisions" intent of GiGPO
without requiring discrete state matching.

Where rollouts have unequal length, the step group at turn-index t
includes only the K' ≤ K rollouts that reached turn t. If K' < 2
(can't normalize), the step advantage at that turn collapses to 0
and the token weight falls back to α · A_traj.


Given a (GradeResult, n_turns) pair, compute a scalar reward suitable
for group-relative policy optimization. The Phase 5 NOTES enumerated
two failure modes the binary `passed` reward can't see:

  1. SFT pays a turn-cost it never recovers from — every trajectory
     hits `max_turns=12` because the corpus has zero clean-stop
     demonstrations. Binary reward is silent on this; the agent has
     no gradient pressure to declare done.
  2. Per-assertion progress is invisible. A 4/5-asserts trajectory
     and a 0/5-asserts trajectory get the same `passed=False` reward.
     GRPO with K=4 rollouts on a hard task will see all-zero rewards
     and produce no learning signal.

Two reward modes:

  - `binary`: 1.0 if all assertions pass, else 0.0. Matches the Phase 5
    `passed` field directly. Dense rewards are cheap but silent on
    near-misses.
  - `shaped`: per_assertion_rate − turn_penalty * (n_turns / max_turns).
    Per-assertion rate ∈ [0, 1]; turn_penalty defaults to 0.2 so a
    full-budget rollout (12/12 turns) shaves 0.2 off the assertion
    score. A trajectory that PASSES in 4 turns scores ~0.93; the same
    PASS dragged out to 12 turns scores 0.80. A trajectory that lands
    3/5 asserts in 12 turns scores 0.40 (vs. 0.0 binary).

Group-relative advantages are the standard GRPO normalization:

    a_i = (r_i − μ) / (σ + eps)

where μ, σ are the per-task mean and std across the K rollouts. When
all K rewards are equal, σ → 0 and advantages collapse to 0 — that
*is* the diagnostic GRPO needs ("no signal here, skip this prompt").

Usage:
    from reward import compute_reward, compute_group_advantages

    r = compute_reward(grade_result, n_turns=traj.n_turns, mode="shaped")
    advs = compute_group_advantages([r1, r2, r3, r4])

The CLI replays Phase 5 trajectories through these and reports the
distribution of shaped rewards for SFT vs base — sanity check that
the shaped reward separates the two regimes.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable

DEFAULT_MAX_TURNS = 12
DEFAULT_TURN_PENALTY = 0.2
DEFAULT_ADVANTAGE_EPS = 1e-8


def compute_reward(
    grade_result: Any,
    *,
    n_turns: int,
    mode: str = "shaped",
    max_turns: int = DEFAULT_MAX_TURNS,
    turn_penalty: float = DEFAULT_TURN_PENALTY,
) -> float:
    """Scalar reward for one (grade_result, trajectory) pair.

    `grade_result` may be a `GradeResult` instance or a dict with the
    same shape (`passed`, `n_passed`, `n_total`). Phase 5's
    `traj.final_grade` is the dict shape; Phase 4's grader returns the
    dataclass. Both work.
    """
    passed, n_passed, n_total = _unpack_grade(grade_result)
    if mode == "binary":
        return 1.0 if passed else 0.0
    if mode == "shaped":
        if n_total == 0:
            return 0.0
        rate = n_passed / n_total
        # n_turns can exceed max_turns if the harness is reconfigured;
        # clamp so the penalty stays in [0, turn_penalty].
        turn_frac = min(1.0, max(0.0, n_turns / max_turns))
        return rate - turn_penalty * turn_frac
    raise ValueError(f"unknown reward mode: {mode}")


def compute_group_advantages(
    rewards: list[float],
    *,
    eps: float = DEFAULT_ADVANTAGE_EPS,
) -> list[float]:
    """Group-relative advantages: (r_i − μ) / (σ + eps).

    Returns a list of zeros if all rewards are equal (σ → 0). That's
    GRPO's no-signal signal and downstream code should skip the group.
    """
    if not rewards:
        return []
    mu = sum(rewards) / len(rewards)
    if len(rewards) == 1:
        return [0.0]
    var = sum((r - mu) ** 2 for r in rewards) / len(rewards)
    sigma = math.sqrt(var)
    if sigma < eps:
        return [0.0] * len(rewards)
    return [(r - mu) / (sigma + eps) for r in rewards]


def turn_step_reward(turn: dict) -> float:
    """Per-turn binary signal for GiGPO step advantages.

    Returns 1.0 iff the turn parsed cleanly AND the shell command
    exited 0. Parse errors and non-zero exits both score 0.0. Final
    `done` turns (TASK_COMPLETE) inherit the previous turn's exit-code
    semantics — they don't run a command, so we treat them as 1.0
    (the agent successfully decided to stop, which is the behavior we
    want to reward when paired with a passing trajectory).
    """
    if turn.get("parse_error"):
        return 0.0
    action = turn.get("action") or {}
    if action.get("kind") == "done":
        return 1.0
    obs = turn.get("observation") or {}
    if obs.get("timed_out"):
        return 0.0
    return 1.0 if obs.get("exit_code", 1) == 0 else 0.0


def compute_step_advantages_within_group(
    rollouts: list[dict],
    *,
    eps: float = DEFAULT_ADVANTAGE_EPS,
) -> list[list[float]]:
    """Per-turn group-relative step advantages across the K rollouts of a task.

    Returns a list parallel to `rollouts`; each entry is a list of
    floats, one per turn in that rollout. Step normalization is at
    turn-index t across only those rollouts that reached turn t.
    Where K' < 2 (single rollout still alive), the advantage is 0.
    """
    if not rollouts:
        return []
    turn_lists = []
    for r in rollouts:
        traj = r.get("trajectory", {})
        turns = traj.get("turns", [])
        turn_lists.append([turn_step_reward(t) for t in turns])

    max_t = max((len(tl) for tl in turn_lists), default=0)
    out = [[0.0] * len(tl) for tl in turn_lists]
    for t in range(max_t):
        active = [(i, tl[t]) for i, tl in enumerate(turn_lists) if t < len(tl)]
        if len(active) < 2:
            continue
        rs = [r for _, r in active]
        mu = sum(rs) / len(rs)
        var = sum((r - mu) ** 2 for r in rs) / len(rs)
        sigma = math.sqrt(var)
        if sigma < eps:
            continue
        for i, r in active:
            out[i][t] = (r - mu) / (sigma + eps)
    return out


def _unpack_grade(g: Any) -> tuple[bool, int, int]:
    if hasattr(g, "passed") and hasattr(g, "n_passed"):
        return bool(g.passed), int(g.n_passed), int(g.n_total)
    if isinstance(g, dict):
        return bool(g.get("passed", False)), int(g.get("n_passed", 0)), int(g.get("n_total", 0))
    raise TypeError(f"cannot unpack grade_result of type {type(g)}")


def _iter_trajectories(path: Path) -> Iterable[dict]:
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _summarize(label: str, rewards: list[float]) -> dict:
    if not rewards:
        return {"label": label, "n": 0}
    return {
        "label": label,
        "n": len(rewards),
        "mean": round(statistics.mean(rewards), 4),
        "median": round(statistics.median(rewards), 4),
        "stdev": round(statistics.pstdev(rewards), 4),
        "min": round(min(rewards), 4),
        "max": round(max(rewards), 4),
        "n_zero": sum(1 for r in rewards if r == 0.0),
        "n_perfect": sum(1 for r in rewards if r >= 0.999),
    }


def _replay_one(path: Path, mode: str, max_turns: int, turn_penalty: float) -> tuple[list[float], list[int]]:
    rewards: list[float] = []
    n_turns_list: list[int] = []
    for traj in _iter_trajectories(path):
        grade = traj.get("final_grade")
        if grade is None:
            continue
        n_turns = int(traj.get("n_turns", 0))
        r = compute_reward(grade, n_turns=n_turns, mode=mode,
                            max_turns=max_turns, turn_penalty=turn_penalty)
        rewards.append(r)
        n_turns_list.append(n_turns)
    return rewards, n_turns_list


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-trajectories", help="JSONL of base-model rollouts")
    ap.add_argument("--sft-trajectories", help="JSONL of SFT-model rollouts")
    ap.add_argument("--mode", choices=["binary", "shaped", "both"], default="both")
    ap.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    ap.add_argument("--turn-penalty", type=float, default=DEFAULT_TURN_PENALTY)
    ap.add_argument("--out", default="-")
    args = ap.parse_args()

    if not (args.base_trajectories or args.sft_trajectories):
        print("supply at least one of --base-trajectories / --sft-trajectories", file=sys.stderr)
        return 2

    modes = ["binary", "shaped"] if args.mode == "both" else [args.mode]
    summary: dict[str, Any] = {}

    for label, path_arg in [("base", args.base_trajectories), ("sft", args.sft_trajectories)]:
        if not path_arg:
            continue
        path = Path(path_arg)
        for mode in modes:
            rewards, _ = _replay_one(path, mode, args.max_turns, args.turn_penalty)
            key = f"{label}_{mode}"
            summary[key] = _summarize(key, rewards)

    # Per-mode delta (sft - base)
    for mode in modes:
        b = summary.get(f"base_{mode}")
        s = summary.get(f"sft_{mode}")
        if b and s and b.get("n") and s.get("n"):
            summary[f"delta_{mode}"] = {
                "mean_delta": round(s["mean"] - b["mean"], 4),
                "median_delta": round(s["median"] - b["median"], 4),
                "n_zero_delta": s["n_zero"] - b["n_zero"],
                "n_perfect_delta": s["n_perfect"] - b["n_perfect"],
            }

    out_text = json.dumps(summary, indent=2)
    if args.out == "-":
        print(out_text)
    else:
        Path(args.out).write_text(out_text + "\n")
        print(f"wrote summary → {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
