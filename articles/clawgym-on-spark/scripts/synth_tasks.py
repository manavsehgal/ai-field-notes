"""Persona-driven task synthesis for the ClawGym-on-Spark proxy corpus.

Phase 1 of the article. Generates programmatically-gradable file-management
tasks by prompting an LLM (NIM-served Nemotron Nano 9B v2 by default) with
a persona + skill list + workspace template, asking for one task in
structured JSON.

Output format: JSONL, one task per line, conforming to scripts/task_schema.md.

Usage:
    python3 synth_tasks.py --personas all --per-persona 5 --out tasks.jsonl
    python3 synth_tasks.py --persona indie-game-dev --per-persona 3 --debug

The class layout (TaskAuthor / WorkspaceSeed / SynthTask / Persona) is
shaped to be lifted into `fieldkit.agents` for v0.2 — see eval.md's
Fieldkit-fit annotation. Don't import from this file directly; once the
API stabilizes it'll move into the package.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import textwrap
from dataclasses import asdict, dataclass, field
from pathlib import Path

from fieldkit.nim import NIMClient, NIMError

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = Path(__file__).resolve().parent

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a task-author for a benchmark of AI agents that operate inside a
    sandboxed file system. Given a persona and a starting workspace, your job
    is to write ONE realistic file-management task the persona would plausibly
    ask an agent to do, plus a list of programmatic assertions that will
    verify whether the agent succeeded.

    The task must be:
    - Solvable using only the provided skill list (file ops + shell utilities).
    - Self-contained — no external network, no package installs, no code execution.
    - Verifiable by a pure function over the post-task file system. Every
      "did the agent succeed?" question must be answerable by checking file
      existence, file absence, or substring/regex matching against file contents.
    - Concrete — name specific filenames, paths, and substrings, not abstractions.

    Respond with EXACTLY a JSON object. No prose, no markdown fences, no
    preamble like "Okay" or "Let me". The first character of your response
    must be `{` and the last must be `}`.
    """
).strip()

USER_PROMPT_TEMPLATE = textwrap.dedent(
    """
    PERSONA
    role: {role}
    context: {context}
    skill_focus: {skill_focus}

    AVAILABLE SKILLS
    {skills_block}

    STARTING WORKSPACE (relative paths under sandbox root)
    {workspace_block}

    OUTPUT JSON SCHEMA
    {{
      "intent": "<1–3 sentence task description as the persona would phrase it>",
      "verifiable_assertions": [
        {{"kind": "file_exists", "path": "<rel/path>"}},
        {{"kind": "file_not_exists", "path": "<rel/path>"}},
        {{"kind": "file_contents_contain", "path": "<rel/path>", "must_contain": ["<substr>", ...]}},
        {{"kind": "file_contents_match_regex", "path": "<rel/path>", "regex": "<python regex>"}},
        {{"kind": "file_unchanged", "path": "<rel/path>"}}
      ],
      "skills_required": ["<one or more skill names>"],
      "difficulty": "easy" | "medium" | "hard",
      "estimated_steps": <int>
    }}

    Rules:
    - At least 2 assertions, at most 6.
    - Every path mentioned in assertions must either be in the starting workspace,
      or be a path the task will create / move into.
    - "file_unchanged" only for files the task should explicitly NOT touch.
    - Difficulty: easy = 1–3 steps, medium = 4–7 steps, hard = 8+ steps.

    Generate the task now.
    """
).strip()


@dataclass
class Persona:
    role: str
    context: str
    skill_focus: list[str]


@dataclass
class WorkspaceFile:
    path: str
    kind: str  # "text" | "binary-stub"
    content: str | int  # text body or size_bytes int

    def to_workspace_dict(self) -> dict:
        if self.kind == "text":
            return {"path": self.path, "kind": "text", "content": self.content}
        return {"path": self.path, "kind": "binary-stub", "size_bytes": int(self.content)}


@dataclass
class WorkspaceSeed:
    files: list[WorkspaceFile]

    def to_dict(self) -> dict:
        return {"files": [f.to_workspace_dict() for f in self.files]}

    def render_for_prompt(self) -> str:
        lines = []
        for f in self.files:
            if f.kind == "text":
                preview = f.content if len(str(f.content)) <= 80 else str(f.content)[:77] + "..."
                lines.append(f"  {f.path}  (text, {len(str(f.content))} chars): {preview!r}")
            else:
                lines.append(f"  {f.path}  (binary-stub, {f.content} bytes)")
        return "\n".join(lines) if lines else "  (empty workspace)"


@dataclass
class SynthTask:
    task_id: str
    persona: dict
    intent: str
    workspace_seed: dict
    skills_required: list[str]
    verifiable_assertions: list[dict]
    difficulty: str
    estimated_steps: int

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def load_personas(path: Path) -> list[Persona]:
    raw = json.loads(path.read_text())
    return [Persona(**p) for p in raw["personas"]]


def load_skills(path: Path) -> dict[str, str]:
    return json.loads(path.read_text())["skills"]


def workspace_for_persona(persona: Persona, seed: int) -> WorkspaceSeed:
    """Hand-authored workspace templates indexed by persona role.

    Each template is parameterized by `seed` so successive calls with the
    same persona give different starting states (different filenames,
    different file counts) while staying in distribution for the role.
    """
    rng = random.Random(seed)

    if persona.role == "indie-game-dev":
        enemies = rng.sample(["goomba", "koopa", "bowser", "shyguy", "lakitu", "boo"], k=rng.randint(2, 4))
        files = [WorkspaceFile(f"assets/enemy_{n}.png", "binary-stub", rng.randint(800, 4000)) for n in enemies]
        files += [
            WorkspaceFile("assets/hero.png", "binary-stub", 2200),
            WorkspaceFile("assets/level1.json", "text", '{"level": 1, "enemies": []}'),
            WorkspaceFile("scripts/player.gd", "text", "extends KinematicBody2D\n# player controller\n"),
        ]
        return WorkspaceSeed(files=files)

    if persona.role == "data-science-researcher":
        runs = [f"run_{i:02d}" for i in range(rng.randint(3, 5))]
        files = []
        for r in runs:
            files += [
                WorkspaceFile(f"experiments/{r}/checkpoint.pt", "binary-stub", rng.randint(50_000, 200_000)),
                WorkspaceFile(f"experiments/{r}/metrics.csv", "text", "epoch,loss\n1,0.42\n2,0.31\n"),
            ]
        files.append(WorkspaceFile("README.md", "text", "# Project\n\nNotebook + experiments.\n"))
        return WorkspaceSeed(files=files)

    if persona.role == "ml-engineer":
        files = [
            WorkspaceFile("logs/train_001.log", "text", "step 100 loss 2.3\nstep 200 loss 1.8\n"),
            WorkspaceFile("logs/train_002.log", "text", "step 100 loss 2.1\n"),
            WorkspaceFile("configs/exp_001.yaml", "text", "lr: 0.001\nbatch: 32\n"),
            WorkspaceFile("configs/exp_002.yaml", "text", "lr: 0.0005\nbatch: 64\n"),
            WorkspaceFile("eval/exp_001.json", "text", '{"acc": 0.81}'),
            WorkspaceFile("eval/exp_002.json", "text", '{"acc": 0.84}'),
        ]
        return WorkspaceSeed(files=files)

    if persona.role == "technical-writer":
        files = [
            WorkspaceFile("docs/intro.md", "text", "# Intro\n\nDraft.\n"),
            WorkspaceFile("docs/guide.md", "text", "---\ntitle: guide\n---\n\nbody.\n"),
            WorkspaceFile("docs/old-name.md", "text", "# Old\n\n![diagram](images/diag.png)\n"),
            WorkspaceFile("images/diag.png", "binary-stub", 5400),
        ]
        return WorkspaceSeed(files=files)

    if persona.role == "backend-developer":
        files = [
            WorkspaceFile("app.py", "text", "from utils import helper\nfrom models import User\n"),
            WorkspaceFile("utils.py", "text", "def helper():\n    return 1\n"),
            WorkspaceFile("models.py", "text", "class User:\n    pass\n"),
            WorkspaceFile("test_app.py", "text", "import app\n"),
        ]
        return WorkspaceSeed(files=files)

    if persona.role == "devops-engineer":
        files = [
            WorkspaceFile("api/Dockerfile", "text", "FROM python:3.11\n"),
            WorkspaceFile("worker/Dockerfile", "text", "FROM python:3.11\n"),
            WorkspaceFile(".github/workflows/ci.yml", "text", "name: ci\non: [push]\n"),
            WorkspaceFile(".github/workflows/deploy.yml", "text", "name: deploy\non: [release]\n"),
        ]
        return WorkspaceSeed(files=files)

    if persona.role == "academic-author":
        files = [
            WorkspaceFile("paper.tex", "text", "\\documentclass{article}\n\\begin{document}\nbody\n\\end{document}\n"),
            WorkspaceFile("paper.aux", "text", "\\relax\n"),
            WorkspaceFile("paper.log", "text", "build log\n"),
            WorkspaceFile("paper.bbl", "text", "\\begin{thebibliography}{1}\n\\end{thebibliography}\n"),
            WorkspaceFile("refs.bib", "text", "@article{foo,title={Foo}}\n"),
            WorkspaceFile("figures/fig1.pdf", "binary-stub", 12_000),
            WorkspaceFile("figures/fig2.pdf", "binary-stub", 8_000),
        ]
        return WorkspaceSeed(files=files)

    if persona.role == "embedded-firmware-dev":
        files = [
            WorkspaceFile("dts/board-rev1.dts", "text", "/dts-v1/;\n/ {\n    model = \"rev1\";\n};\n"),
            WorkspaceFile("dts/board-rev2.dts", "text", "/dts-v1/;\n/ {\n    model = \"rev2\";\n};\n"),
            WorkspaceFile("Kconfig.rev1", "text", "config REV1\n    bool\n"),
            WorkspaceFile("logs/bringup-rev1.txt", "text", "bringup ok\n"),
        ]
        return WorkspaceSeed(files=files)

    raise ValueError(f"unknown persona role: {persona.role}")


def render_skills_block(skills: dict[str, str], focus: list[str]) -> str:
    """Render the skills section, leading with the persona's focus skills."""
    ordered = list(focus) + [s for s in skills if s not in focus]
    return "\n".join(f"  {name}: {skills[name]}" for name in ordered if name in skills)


class TaskAuthor:
    """LLM-driven task synthesizer.

    Wraps NIMClient with a structured-output retry loop. Given a persona and
    a workspace, returns a validated SynthTask or raises after N attempts.
    """

    def __init__(self, client: NIMClient, *, max_attempts: int = 3, debug: bool = False) -> None:
        self.client = client
        self.max_attempts = max_attempts
        self.debug = debug

    def author_one(
        self,
        *,
        task_id: str,
        persona: Persona,
        workspace: WorkspaceSeed,
        skills: dict[str, str],
    ) -> SynthTask:
        skills_block = render_skills_block(skills, persona.skill_focus)
        workspace_block = workspace.render_for_prompt()
        user_prompt = USER_PROMPT_TEMPLATE.format(
            role=persona.role,
            context=persona.context,
            skill_focus=", ".join(persona.skill_focus),
            skills_block=skills_block,
            workspace_block=workspace_block,
        )

        last_err: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                resp = self.client.chat(
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.4,
                    max_tokens=1500,
                    chat_template_kwargs={"thinking": False},
                )
                text = resp["choices"][0]["message"]["content"]
                if self.debug:
                    print(f"  raw[{attempt}]: {text[:400]!r}", file=sys.stderr)
                payload = _extract_json(text)
                _validate_payload(payload, workspace=workspace)
                return SynthTask(
                    task_id=task_id,
                    persona={"role": persona.role, "context": persona.context},
                    intent=payload["intent"],
                    workspace_seed=workspace.to_dict(),
                    skills_required=payload["skills_required"],
                    verifiable_assertions=payload["verifiable_assertions"],
                    difficulty=payload["difficulty"],
                    estimated_steps=int(payload["estimated_steps"]),
                )
            except (NIMError, ValueError, KeyError, json.JSONDecodeError) as e:
                last_err = e
                if self.debug:
                    print(f"  attempt {attempt} failed: {type(e).__name__}: {e}", file=sys.stderr)
        raise RuntimeError(f"task author failed after {self.max_attempts} attempts: {last_err}")


def _extract_json(text: str) -> dict:
    """Pull the first balanced JSON object out of the response.

    Tolerates: leading prose ("Okay, let's tackle this..."), ```json fences,
    `<think>...</think>` blocks (Nemotron reasoning), multiple JSON objects
    in the same response (we take the longest/last one — synth output is
    typically the final, most-complete object after any echoed example).
    """
    s = text.strip()
    if "</think>" in s:
        s = s.split("</think>", 1)[1].strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()
        if s.startswith("json"):
            s = s[4:].strip()

    # Walk the string and extract every balanced top-level JSON object.
    objs: list[dict] = []
    i = 0
    while i < len(s):
        if s[i] != "{":
            i += 1
            continue
        depth = 0
        in_str = False
        escape = False
        for j in range(i, len(s)):
            c = s[j]
            if escape:
                escape = False
                continue
            if c == "\\" and in_str:
                escape = True
                continue
            if c == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        objs.append(json.loads(s[i : j + 1]))
                    except json.JSONDecodeError:
                        pass
                    i = j + 1
                    break
        else:
            break  # unbalanced — done

    if not objs:
        raise ValueError(f"no balanced JSON object found in response: {text[:200]!r}")
    # Prefer the object with the most fields — the longer one wins over an
    # echoed example shape from the prompt.
    return max(objs, key=lambda d: len(d) if isinstance(d, dict) else 0)


_VALID_ASSERTION_KINDS = {
    "file_exists",
    "file_not_exists",
    "file_contents_contain",
    "file_contents_match_regex",
    "file_unchanged",
}


def _validate_payload(payload: dict, *, workspace: WorkspaceSeed) -> None:
    required = ("intent", "verifiable_assertions", "skills_required", "difficulty", "estimated_steps")
    for key in required:
        if key not in payload:
            raise ValueError(f"missing required field: {key}")
    if not isinstance(payload["verifiable_assertions"], list) or not (1 <= len(payload["verifiable_assertions"]) <= 8):
        raise ValueError(f"verifiable_assertions must be a 1–8 item list (got {len(payload.get('verifiable_assertions', [])) if isinstance(payload.get('verifiable_assertions'), list) else 'non-list'})")
    seed_paths = {f.path for f in workspace.files}
    for a in payload["verifiable_assertions"]:
        if a.get("kind") not in _VALID_ASSERTION_KINDS:
            raise ValueError(f"unknown assertion kind: {a.get('kind')}")
        if "path" not in a:
            raise ValueError(f"assertion missing path: {a}")
        if a["kind"] == "file_unchanged" and a["path"] not in seed_paths:
            raise ValueError(f"file_unchanged refers to non-seed path: {a['path']}")
    if payload["difficulty"] not in ("easy", "medium", "hard"):
        raise ValueError(f"bad difficulty: {payload['difficulty']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--personas", default="all", help='"all" or comma-separated list of roles')
    ap.add_argument("--persona", help='single persona role (overrides --personas)')
    ap.add_argument("--per-persona", type=int, default=4)
    ap.add_argument("--out", required=True, help="JSONL output path")
    ap.add_argument("--nim-base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="nvidia/nemotron-nano-9b-v2")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    personas_path = SCRIPTS_DIR / "personas.json"
    skills_path = SCRIPTS_DIR / "skills.json"
    personas = load_personas(personas_path)
    skills = load_skills(skills_path)

    if args.persona:
        wanted = {args.persona}
    elif args.personas == "all":
        wanted = {p.role for p in personas}
    else:
        wanted = set(args.personas.split(","))
    selected = [p for p in personas if p.role in wanted]
    if not selected:
        print(f"no personas matched {wanted!r}; available: {[p.role for p in personas]}", file=sys.stderr)
        return 2

    client = NIMClient(base_url=args.nim_base_url, model=args.model, timeout=240.0)
    author = TaskAuthor(client, debug=args.debug)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng_seed = args.seed
    n_ok = 0
    n_fail = 0
    with out_path.open("w") as fh:
        for persona in selected:
            for i in range(args.per_persona):
                tid = f"synth-{persona.role}-{i:02d}"
                workspace = workspace_for_persona(persona, seed=rng_seed)
                rng_seed += 1
                try:
                    task = author.author_one(task_id=tid, persona=persona, workspace=workspace, skills=skills)
                    fh.write(task.to_jsonl() + "\n")
                    fh.flush()
                    n_ok += 1
                    print(f"OK   {tid}  diff={task.difficulty}  asserts={len(task.verifiable_assertions)}")
                except Exception as e:
                    n_fail += 1
                    print(f"FAIL {tid}  {type(e).__name__}: {e}", file=sys.stderr)
    print(f"\n{n_ok} ok, {n_fail} failed → {out_path}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
