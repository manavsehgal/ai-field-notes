"""Turn A4's trajectory.jsonl into a chat-format LoRA fine-tuning corpus.

For each of the 50 evaluated iterations we replay the prompt the 8B agent
saw (perturbation menu + running baseline cfg + last-5 history) and pair it
with the proposal it produced as the assistant target. Hold out the last
8 iterations (43..50) as a time-ordered test split — this tests the
deployment scenario "given trajectory so far, propose what's next."

Outputs:
    train.jsonl  (iters 1..42, 42 rows)
    test.jsonl   (iters 43..50, 8 rows)

Each row:
    {
      "iter": int,
      "messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}],
      "decision": "keep" | "revert",
      "val_bpb": float,
      "improvement_frac": float
    }
"""
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

EVIDENCE = Path(__file__).resolve().parent
A4_EVIDENCE = EVIDENCE.parent.parent / "autoresearch-agent-loop" / "evidence"
TRAJECTORY = A4_EVIDENCE / "trajectory.jsonl"

# Reuse A4's proposer prompt machinery so prompts are byte-identical to what
# the 8B saw at training time.
sys.path.insert(0, str(A4_EVIDENCE))
from proposer import SYSTEM_PROMPT, build_prompt  # type: ignore


def load_trajectory(path: Path) -> tuple[dict, list[dict]]:
    rows = [json.loads(line) for line in open(path)]
    baseline = rows[0]
    iters = [r for r in rows[1:] if r.get("stage") == "evaluated"]
    return baseline, iters


def reconstruct_running_baseline(baseline: dict, iters: list[dict]) -> list[dict]:
    """For each iter, return the baseline_cfg the agent saw before proposing.
    Rolls forward when an iter's decision is 'keep'."""
    running = deepcopy(baseline["baseline_cfg"])
    seen = []
    for r in iters:
        seen.append(deepcopy(running))
        if r.get("decision") == "keep":
            running = deepcopy(r["candidate_cfg"])
    return seen


def to_chat_record(iter_row: dict, baseline_at_iter: dict, history: list[dict]) -> dict:
    """Produce one chat-format JSONL row mirroring proposer.build_prompt's I/O."""
    msgs = build_prompt(history=history, baseline_cfg=baseline_at_iter, recent_k=5)
    target = json.dumps(iter_row["proposal"], separators=(", ", ": "))
    return {
        "iter": iter_row["iter"],
        "messages": [
            {"role": "system",    "content": msgs[0]["content"]},
            {"role": "user",      "content": msgs[1]["content"]},
            {"role": "assistant", "content": target},
        ],
        "decision": iter_row.get("decision"),
        "val_bpb": iter_row.get("val_bpb"),
        "improvement_frac": iter_row.get("improvement_frac"),
    }


def main() -> None:
    baseline, iters = load_trajectory(TRAJECTORY)
    print(f"loaded {len(iters)} evaluated iters from {TRAJECTORY.name}")

    running_baselines = reconstruct_running_baseline(baseline, iters)

    rows = []
    for i, (iter_row, base_at_i) in enumerate(zip(iters, running_baselines)):
        history = iters[:i]  # everything before this iter
        rows.append(to_chat_record(iter_row, base_at_i, history))

    train = [r for r in rows if r["iter"] <= 42]
    test = [r for r in rows if r["iter"] >= 43]
    assert len(train) == 42, f"expected 42 train, got {len(train)}"
    assert len(test) == 8, f"expected 8 test, got {len(test)}"

    keep_train = sum(1 for r in train if r["decision"] == "keep")
    keep_test = sum(1 for r in test if r["decision"] == "keep")
    print(f"train: {len(train)} ({keep_train} keep, {len(train)-keep_train} revert)")
    print(f"test:  {len(test)} ({keep_test} keep, {len(test)-keep_test} revert)")

    train_path = EVIDENCE / "train.jsonl"
    test_path = EVIDENCE / "test.jsonl"
    with open(train_path, "w") as f:
        for r in train:
            f.write(json.dumps(r) + "\n")
    with open(test_path, "w") as f:
        for r in test:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {train_path} ({train_path.stat().st_size} bytes)")
    print(f"wrote {test_path}  ({test_path.stat().st_size} bytes)")

    # Inspect prompt length distribution — affects max_length in trainer.
    char_lens = [len(r["messages"][1]["content"]) for r in rows]
    print(f"user-prompt chars: min={min(char_lens)} median={sorted(char_lens)[len(char_lens)//2]} max={max(char_lens)}")


if __name__ == "__main__":
    main()
