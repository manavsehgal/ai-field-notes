"""Sandbox rollout harness for ClawGym-on-Spark Phase 2.

Pipes a synthesized task through a sandboxed shell environment, drives an
LLM-agent loop (NIM-served Llama 3.1 8B by default) for up to N turns,
captures the (action, observation) trajectory, and grades the final
sandbox state with `grader.grade`.

Usage:
    # Single task, local-tempdir backend
    python3 rollout.py \
        --tasks /tmp/tasks-8.jsonl \
        --task-id synth-indie-game-dev-00 \
        --out-dir /tmp/clawgym-rollouts/

    # All tasks
    python3 rollout.py \
        --tasks /tmp/tasks-8.jsonl \
        --out-dir /tmp/clawgym-rollouts/

    # Mock-action mode (no NIM call, replays a hand-authored action script)
    python3 rollout.py \
        --tasks /tmp/tasks-8.jsonl \
        --task-id synth-indie-game-dev-00 \
        --mock-actions actions.jsonl \
        --out-dir /tmp/clawgym-rollouts/

The class layout (Sandbox / RolloutDriver / TurnRecord / Trajectory) is
shaped to lift into `fieldkit.agents.rollout` in v0.2 — see
fieldkit_agents_v0_2_sketch.md.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from grader import grade, materialize_seed, seed_files_from_task

SCRIPTS_DIR = Path(__file__).resolve().parent

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a file-management agent operating inside a sandboxed Linux
    workspace. You have access to standard POSIX shell tools: ls, cat,
    head, tail, mv, cp, rm, mkdir, sed, grep, find, wc, echo, touch.

    Each turn, respond with EXACTLY ONE shell command wrapped in a
    ```bash code block```. After the command runs you will see its
    stdout, stderr, and exit code in the next turn.

    When you believe the task is complete, respond with the literal
    token TASK_COMPLETE on a line by itself (no code block).

    Rules:
    - One command per turn. No semicolons or && chains; use multiple turns.
    - Always use relative paths from the workspace root (which is your CWD).
    - Use `sed -i` for in-place edits.
    - Use `mkdir -p` to create parent directories.
    - Do not invoke network commands (curl, wget, apt, pip, npm).
    - Do not invoke editors (vi, nano, emacs).
    """
).strip()

USER_PROMPT_TEMPLATE = textwrap.dedent(
    """
    TASK
    {intent}

    WORKSPACE FILES (CWD = workspace root)
    {file_listing}

    Begin. Respond with one shell command in a ```bash``` block.
    """
).strip()


# ────────────────────────────────────────────────────────────────
# Sandbox abstraction
# ────────────────────────────────────────────────────────────────


@dataclass
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class Sandbox(ABC):
    """Minimal sandbox interface — materialize a seed, exec shell, list files."""

    root: Path

    @abstractmethod
    def exec(self, cmd: str, timeout: float = 10.0) -> ExecResult: ...

    @abstractmethod
    def materialize(self, task: dict) -> None: ...

    @abstractmethod
    def list_files(self) -> list[str]: ...

    @abstractmethod
    def cleanup(self) -> None: ...


class LocalTempSandbox(Sandbox):
    """Local-filesystem sandbox using a fresh tempdir + subprocess.run.

    No isolation beyond CWD discipline — this is the development backend.
    Production rollouts (Phase 4 GRPO at 8 parallel) should use the
    NemoClaw OpenShell backend; see clawnav file-transfer memory for the
    upload pattern.
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path(tempfile.mkdtemp(prefix="clawgym-rollout-"))
        self.root.mkdir(parents=True, exist_ok=True)

    def materialize(self, task: dict) -> None:
        # Wipe + reseed so reruns are reproducible.
        if self.root.exists():
            shutil.rmtree(self.root)
        materialize_seed(task, self.root)

    def exec(self, cmd: str, timeout: float = 10.0) -> ExecResult:
        try:
            proc = subprocess.run(
                ["bash", "-c", cmd],
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return ExecResult(
                exit_code=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
            )
        except subprocess.TimeoutExpired as e:
            return ExecResult(
                exit_code=124,
                stdout=(e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or ""),
                stderr=(e.stderr or b"").decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or ""),
                timed_out=True,
            )

    def list_files(self) -> list[str]:
        out: list[str] = []
        for p in sorted(self.root.rglob("*")):
            if p.is_file():
                out.append(str(p.relative_to(self.root)))
            elif p.is_dir() and not any(p.iterdir()):
                out.append(str(p.relative_to(self.root)) + "/")
        return out

    def cleanup(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)


# ────────────────────────────────────────────────────────────────
# Trajectory record
# ────────────────────────────────────────────────────────────────


@dataclass
class TurnRecord:
    turn: int
    agent_response: str
    action: dict | None  # {"kind": "shell", "cmd": "..."} or None for parse failure
    observation: dict | None  # ExecResult.to_dict() or None for parse-failure / done
    parse_error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Trajectory:
    task_id: str
    model: str
    n_turns: int
    stopped: str  # "task_complete" | "max_turns" | "agent_error" | "parse_error"
    wall_seconds: float
    turns: list[TurnRecord]
    final_grade: dict | None = None

    def to_jsonl(self) -> str:
        return json.dumps({
            "task_id": self.task_id,
            "model": self.model,
            "n_turns": self.n_turns,
            "stopped": self.stopped,
            "wall_seconds": round(self.wall_seconds, 2),
            "turns": [t.to_dict() for t in self.turns],
            "final_grade": self.final_grade,
        }, ensure_ascii=False)


# ────────────────────────────────────────────────────────────────
# Action parsing
# ────────────────────────────────────────────────────────────────

_BASH_BLOCK_RE = re.compile(r"```(?:bash|sh|shell)?\s*\n(.*?)```", re.DOTALL)


def parse_action(text: str) -> tuple[dict | None, str | None]:
    """Extract a single shell command from the agent's response.

    Returns:
        (action_dict, parse_error). Exactly one is non-None.

        action_dict is one of:
            {"kind": "shell", "cmd": "<cmd>"}
            {"kind": "done"}
    """
    if not text or not text.strip():
        return None, "empty response"

    if re.search(r"^\s*TASK_COMPLETE\s*$", text, re.MULTILINE):
        # Prefer "done" over any code block — the agent has explicitly stopped.
        return {"kind": "done"}, None

    blocks = _BASH_BLOCK_RE.findall(text)
    if not blocks:
        return None, "no ```bash``` block and no TASK_COMPLETE token"

    cmd = blocks[0].strip()
    if not cmd:
        return None, "empty bash block"
    # Take only the first non-empty line — the system prompt forbids chains.
    lines = [ln for ln in cmd.split("\n") if ln.strip() and not ln.strip().startswith("#")]
    if not lines:
        return None, "bash block had only comments"
    return {"kind": "shell", "cmd": lines[0]}, None


def truncate(s: str, limit: int = 4000) -> str:
    if len(s) <= limit:
        return s
    half = limit // 2 - 16
    return s[:half] + f"\n…[truncated {len(s) - 2 * half} chars]…\n" + s[-half:]


def render_files_block(paths: list[str]) -> str:
    if not paths:
        return "  (empty)"
    return "\n".join(f"  {p}" for p in paths)


# ────────────────────────────────────────────────────────────────
# Rollout driver
# ────────────────────────────────────────────────────────────────


class RolloutDriver:
    """Drive one task through the agent loop and produce a Trajectory.

    The agent client is duck-typed: must accept .chat(messages, ...) and
    return {"choices": [{"message": {"content": <str>}}]}. Matches both
    fieldkit.nim.NIMClient and the MockClient below.
    """

    def __init__(
        self,
        agent_client: Any,
        model_name: str,
        sandbox_factory,
        *,
        max_turns: int = 12,
        per_command_timeout: float = 10.0,
        debug: bool = False,
    ) -> None:
        self.agent = agent_client
        self.model_name = model_name
        self.sandbox_factory = sandbox_factory
        self.max_turns = max_turns
        self.per_command_timeout = per_command_timeout
        self.debug = debug

    def rollout(self, task: dict) -> tuple[Trajectory, Sandbox]:
        sandbox = self.sandbox_factory()
        sandbox.materialize(task)
        t0 = time.time()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(
                    intent=task["intent"],
                    file_listing=render_files_block(sandbox.list_files()),
                ),
            },
        ]

        turns: list[TurnRecord] = []
        stopped = "max_turns"
        for turn_idx in range(1, self.max_turns + 1):
            try:
                resp = self.agent.chat(messages=messages, temperature=0.2, max_tokens=400)
                response_text = resp["choices"][0]["message"]["content"]
            except Exception as e:
                if self.debug:
                    print(f"  turn {turn_idx}: agent error: {e}", file=sys.stderr)
                turns.append(TurnRecord(
                    turn=turn_idx,
                    agent_response="",
                    action=None,
                    observation=None,
                    parse_error=f"agent_error: {type(e).__name__}: {e}",
                ))
                stopped = "agent_error"
                break

            action, parse_error = parse_action(response_text)

            if self.debug:
                print(f"  turn {turn_idx}: response[:120]={response_text[:120]!r}", file=sys.stderr)
                if action:
                    print(f"  turn {turn_idx}: action={action}", file=sys.stderr)
                else:
                    print(f"  turn {turn_idx}: parse_error={parse_error}", file=sys.stderr)

            if action is None:
                turns.append(TurnRecord(
                    turn=turn_idx,
                    agent_response=truncate(response_text),
                    action=None,
                    observation=None,
                    parse_error=parse_error,
                ))
                # Give the agent a corrective hint and continue once.
                messages.append({"role": "assistant", "content": response_text})
                messages.append({
                    "role": "user",
                    "content": (
                        f"PARSE ERROR: {parse_error}. Reply with ONE ```bash``` block "
                        "containing one command, or TASK_COMPLETE on a line by itself."
                    ),
                })
                continue

            if action["kind"] == "done":
                turns.append(TurnRecord(
                    turn=turn_idx,
                    agent_response=truncate(response_text),
                    action=action,
                    observation=None,
                ))
                stopped = "task_complete"
                break

            obs = sandbox.exec(action["cmd"], timeout=self.per_command_timeout)
            obs_dict = obs.to_dict()
            obs_dict["stdout"] = truncate(obs_dict["stdout"])
            obs_dict["stderr"] = truncate(obs_dict["stderr"])

            turns.append(TurnRecord(
                turn=turn_idx,
                agent_response=truncate(response_text),
                action=action,
                observation=obs_dict,
            ))

            messages.append({"role": "assistant", "content": response_text})
            messages.append({
                "role": "user",
                "content": (
                    f"OBSERVATION (exit {obs.exit_code}{', TIMED OUT' if obs.timed_out else ''}):\n"
                    f"--- stdout ---\n{obs_dict['stdout']}\n"
                    f"--- stderr ---\n{obs_dict['stderr']}\n"
                    "Next command (one ```bash``` block) or TASK_COMPLETE."
                ),
            })

        wall = time.time() - t0
        traj = Trajectory(
            task_id=task["task_id"],
            model=self.model_name,
            n_turns=len(turns),
            stopped=stopped,
            wall_seconds=wall,
            turns=turns,
        )
        return traj, sandbox


# ────────────────────────────────────────────────────────────────
# Mock client (for harness validation without NIM)
# ────────────────────────────────────────────────────────────────


class MockClient:
    """Replays a JSONL of canned (turn, response) records.

    Useful for validating the harness end-to-end without a NIM running.
    The JSONL shape is one record per line:

        {"task_id": "synth-indie-game-dev-00", "responses": ["```bash\\nls\\n```", "TASK_COMPLETE"]}
    """

    def __init__(self, scripts: dict[str, list[str]]) -> None:
        self.scripts = scripts
        self._cursor: dict[str, int] = {}
        self._current_task: str | None = None

    def set_task(self, task_id: str) -> None:
        self._current_task = task_id
        self._cursor[task_id] = 0

    def chat(self, *, messages, **_kw):
        tid = self._current_task
        if tid is None or tid not in self.scripts:
            return {"choices": [{"message": {"content": "TASK_COMPLETE"}}]}
        i = self._cursor[tid]
        responses = self.scripts[tid]
        if i >= len(responses):
            return {"choices": [{"message": {"content": "TASK_COMPLETE"}}]}
        text = responses[i]
        self._cursor[tid] = i + 1
        return {"choices": [{"message": {"content": text}}]}


def load_mock_scripts(path: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[r["task_id"]] = r["responses"]
    return out


# ────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True, help="JSONL of synthesized tasks")
    ap.add_argument("--task-id", help="rollout a single task; default = all")
    ap.add_argument("--out-dir", required=True, help="output directory for trajectories + post-state dirs")
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--per-command-timeout", type=float, default=10.0)
    ap.add_argument("--nim-base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="meta/llama-3.1-8b-instruct")
    ap.add_argument("--mock-actions", help="JSONL of canned responses (overrides NIM call)")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    tasks: dict[str, dict[str, Any]] = {}
    with open(args.tasks) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            tasks[t["task_id"]] = t

    if args.task_id:
        if args.task_id not in tasks:
            print(f"task_id not found: {args.task_id}", file=sys.stderr)
            return 2
        selected = [tasks[args.task_id]]
    else:
        selected = list(tasks.values())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    post_states_dir = out_dir / "post-states"
    post_states_dir.mkdir(exist_ok=True)
    trajectories_path = out_dir / "trajectories.jsonl"

    if args.mock_actions:
        scripts = load_mock_scripts(Path(args.mock_actions))
        agent = MockClient(scripts)
        model_name = "mock"
    else:
        from fieldkit.nim import NIMClient
        agent = NIMClient(base_url=args.nim_base_url, model=args.model, timeout=120.0)
        model_name = args.model

    def sandbox_factory():
        # Each task gets its own persistent post-state dir for the grader.
        return None  # set per-task below

    n_pass = 0
    n_total = 0
    with trajectories_path.open("w") as fh:
        for task in selected:
            tid = task["task_id"]
            print(f"→ {tid}  intent: {task['intent'][:80]}")
            if isinstance(agent, MockClient):
                agent.set_task(tid)

            post_state_root = post_states_dir / tid
            sb = LocalTempSandbox(root=post_state_root)
            driver = RolloutDriver(
                agent_client=agent,
                model_name=model_name,
                sandbox_factory=lambda root=post_state_root: LocalTempSandbox(root=root),
                max_turns=args.max_turns,
                per_command_timeout=args.per_command_timeout,
                debug=args.debug,
            )
            traj, final_sb = driver.rollout(task)

            seeds = seed_files_from_task(task)
            grade_result = grade(task, final_sb.root, seed_files=seeds)
            traj.final_grade = grade_result.to_dict()
            fh.write(traj.to_jsonl() + "\n")
            fh.flush()

            n_total += 1
            if grade_result.passed:
                n_pass += 1
            print(f"   stopped={traj.stopped}  turns={traj.n_turns}  wall={traj.wall_seconds:.1f}s  grade={'PASS' if grade_result.passed else 'FAIL'} ({grade_result.n_passed}/{grade_result.n_total})")

    print(f"\n{n_pass}/{n_total} tasks passed → {trajectories_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
