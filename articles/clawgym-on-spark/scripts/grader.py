"""Programmatic grader for ClawGym-on-Spark synthesized tasks.

Given a (task, post-state-directory) pair, runs each verifiable assertion
against the post-state directory and returns a per-assertion pass/fail
plus an overall task pass/fail (binary AND).

The grader is intentionally a pure function over the file system — no LLM,
no fuzzy matching, no scoring. The hybrid-verification "LLM-as-judge"
flavor is deferred to a v0.2 fieldkit module; the programmatic flavor
covers everything Phase 1 synthesizes.

Usage:
    python3 grader.py --task-id synth-indie-game-dev-00 \
        --tasks tasks.jsonl \
        --post-state /tmp/sandbox-rollout-out/

    # Or batch:
    python3 grader.py --tasks tasks.jsonl --post-states-dir /tmp/rollouts/

This would lift into `fieldkit.eval.AssertionGrader` for v0.2 — see
the Fieldkit-fit annotation on papers/2604.26904/eval.md.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class AssertionResult:
    kind: str
    path: str
    passed: bool
    detail: str = ""

    def to_dict(self) -> dict:
        return {"kind": self.kind, "path": self.path, "passed": self.passed, "detail": self.detail}


@dataclass
class GradeResult:
    task_id: str
    passed: bool
    n_passed: int
    n_total: int
    assertions: list[AssertionResult]

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "passed": self.passed,
            "n_passed": self.n_passed,
            "n_total": self.n_total,
            "assertions": [a.to_dict() for a in self.assertions],
        }


def grade(
    task: dict,
    post_state_root: Path,
    *,
    seed_files: dict[str, str] | None = None,
) -> GradeResult:
    """Evaluate a single task against a post-rollout sandbox directory.

    Args:
        task: A SynthTask record (decoded from JSONL).
        post_state_root: Path to the directory the agent operated on.
        seed_files: Optional pre-rollout file-content map for `file_unchanged`
            assertions. Keys are relative paths, values are text contents.
            If omitted, `file_unchanged` reports as "skipped" (counted as pass).

    Returns:
        GradeResult with per-assertion outcomes and a binary overall pass.
    """
    seed_files = seed_files or {}
    results: list[AssertionResult] = []
    for a in task["verifiable_assertions"]:
        kind = a["kind"]
        rel = a["path"]
        full = post_state_root / rel
        if kind == "file_exists":
            # Synth tasks use file_exists for both files and directories
            # ("file_exists: enemies" for the new dir). Match either.
            results.append(AssertionResult(kind, rel, full.exists(),
                                           "" if full.exists() else "path missing"))
        elif kind == "file_not_exists":
            results.append(AssertionResult(kind, rel, not full.exists(),
                                           "" if not full.exists() else "file still present"))
        elif kind == "file_contents_contain":
            if not full.is_file():
                results.append(AssertionResult(kind, rel, False, "file missing"))
                continue
            try:
                body = full.read_text(errors="replace")
            except OSError as e:
                results.append(AssertionResult(kind, rel, False, f"read error: {e}"))
                continue
            missing = [s for s in a.get("must_contain", []) if s not in body]
            ok = not missing
            detail = "" if ok else f"missing substrings: {missing}"
            results.append(AssertionResult(kind, rel, ok, detail))
        elif kind == "file_contents_match_regex":
            if not full.is_file():
                results.append(AssertionResult(kind, rel, False, "file missing"))
                continue
            try:
                body = full.read_text(errors="replace")
                ok = re.search(a["regex"], body) is not None
                detail = "" if ok else "regex not matched"
            except (OSError, re.error) as e:
                ok, detail = False, f"error: {e}"
            results.append(AssertionResult(kind, rel, ok, detail))
        elif kind == "file_unchanged":
            seed = seed_files.get(rel)
            if seed is None:
                # No seed provided — can't verify; treat as pass-through
                results.append(AssertionResult(kind, rel, True, "skipped (no seed content)"))
                continue
            if not full.is_file():
                results.append(AssertionResult(kind, rel, False, "file missing post-rollout"))
                continue
            try:
                body = full.read_text(errors="replace")
                ok = body == seed
                detail = "" if ok else "contents diverged from seed"
            except OSError as e:
                ok, detail = False, f"read error: {e}"
            results.append(AssertionResult(kind, rel, ok, detail))
        else:
            results.append(AssertionResult(kind, rel, False, f"unknown kind: {kind}"))

    n_passed = sum(1 for r in results if r.passed)
    return GradeResult(
        task_id=task["task_id"],
        passed=all(r.passed for r in results),
        n_passed=n_passed,
        n_total=len(results),
        assertions=results,
    )


def seed_files_from_task(task: dict) -> dict[str, str]:
    """Extract text-content seed map for `file_unchanged` assertions."""
    out: dict[str, str] = {}
    for f in task["workspace_seed"]["files"]:
        if f["kind"] == "text":
            out[f["path"]] = f["content"]
    return out


def materialize_seed(task: dict, root: Path) -> None:
    """Write the workspace seed to disk under root. Used for setup + dry-run grading."""
    root.mkdir(parents=True, exist_ok=True)
    for f in task["workspace_seed"]["files"]:
        full = root / f["path"]
        full.parent.mkdir(parents=True, exist_ok=True)
        if f["kind"] == "text":
            full.write_text(f["content"])
        else:
            full.write_bytes(b"\x00" * int(f.get("size_bytes", f.get("content", 0))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True, help="JSONL of synthesized tasks")
    ap.add_argument("--task-id", help="grade a single task")
    ap.add_argument("--post-state", help="path to post-rollout sandbox dir (with --task-id)")
    ap.add_argument("--post-states-dir", help="directory containing one subdir per task_id")
    ap.add_argument("--dry-run", action="store_true",
                    help="grade against the seed itself (sanity check — should fail most assertions)")
    ap.add_argument("--out", default="-", help="JSON output path or - for stdout")
    args = ap.parse_args()

    tasks: dict[str, dict[str, Any]] = {}
    with open(args.tasks) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            tasks[t["task_id"]] = t

    selected = [tasks[args.task_id]] if args.task_id else list(tasks.values())
    if args.task_id and args.task_id not in tasks:
        print(f"task_id not in tasks file: {args.task_id}", file=sys.stderr)
        return 2

    out_records: list[dict] = []
    for task in selected:
        if args.dry_run:
            tmp = Path(f"/tmp/clawgym-dryrun-{task['task_id']}")
            materialize_seed(task, tmp)
            post_state = tmp
        elif args.post_state:
            post_state = Path(args.post_state)
        elif args.post_states_dir:
            post_state = Path(args.post_states_dir) / task["task_id"]
        else:
            print("need one of --post-state, --post-states-dir, --dry-run", file=sys.stderr)
            return 2
        seeds = seed_files_from_task(task)
        result = grade(task, post_state, seed_files=seeds)
        out_records.append(result.to_dict())
        print(f"{task['task_id']}: {'PASS' if result.passed else 'FAIL'} ({result.n_passed}/{result.n_total})")

    if args.out == "-":
        json.dump(out_records, sys.stdout, indent=2)
        print()
    else:
        Path(args.out).write_text(json.dumps(out_records, indent=2))
        print(f"wrote {len(out_records)} grade records → {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
