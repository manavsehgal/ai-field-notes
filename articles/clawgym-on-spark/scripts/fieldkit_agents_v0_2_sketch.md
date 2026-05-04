# Proposed `fieldkit.agents` v0.2 API surface

Based on the Phase 1 substrate — `synth_tasks.py` and `grader.py`. Not yet
in the package; this file is the design doc that `tech-writer extract` will
turn into `[Unreleased]` CHANGELOG entries after the article publishes.

## Module layout

```
fieldkit/src/fieldkit/agents/
├── __init__.py        # __all__, public API
├── persona.py         # Persona, WorkspaceFile, WorkspaceSeed
├── task.py            # SynthTask + JSONL helpers
├── author.py          # TaskAuthor (LLM-driven), retry + schema validation
└── rollout.py         # SandboxRollout (Phase 2 — deferred until article #2 in this arc)
```

Plus extending `fieldkit.eval`:

```
fieldkit/src/fieldkit/eval/
├── __init__.py        # ...existing exports
├── grader.py          # NEW: AssertionGrader, AssertionResult, GradeResult
```

## Public surface

```python
# Personas + workspaces (data model)
from fieldkit.agents import Persona, WorkspaceFile, WorkspaceSeed, SynthTask

p = Persona(role="indie-game-dev", context="...", skill_focus=["file.move"])
ws = WorkspaceSeed(files=[WorkspaceFile("assets/hero.png", "binary-stub", 2200)])
ws.materialize(Path("/tmp/sandbox-1/"))   # writes the seed to disk

# Task synthesis (LLM-driven)
from fieldkit.agents import TaskAuthor
from fieldkit.nim import NIMClient

with NIMClient(base_url="http://localhost:8000/v1", model="nvidia/nemotron-nano-9b-v2") as nim:
    author = TaskAuthor(nim, max_attempts=3)
    task: SynthTask = author.author_one(persona=p, workspace=ws, skills=DEFAULT_SKILLS)

# Grading
from fieldkit.eval import AssertionGrader

grader = AssertionGrader()
result = grader.grade(task, post_state_root=Path("/tmp/sandbox-1/"))
print(result.passed, result.n_passed, result.n_total)
```

## Why this is one module not three

`Persona`, `WorkspaceSeed`, `SynthTask`, `TaskAuthor` cluster around one job:
*generate verifiable training/eval tasks for sandboxed agents*. Splitting them
into separate modules would force every consumer to import from 3+ places.
Mirroring `fieldkit.rag.{Pipeline, Document, Chunk}` — one module, one job.

`SandboxRollout` belongs in the same module because it consumes a
`WorkspaceSeed` and produces a `Trajectory` that gets graded by
`AssertionGrader`. The four are a tight loop.

## Why grader goes in `fieldkit.eval`, not `fieldkit.agents`

`fieldkit.eval` already hosts `Bench`, `Judge`, `Trajectory` — the verification
primitives. `AssertionGrader` is another verification primitive that happens
to be programmatic instead of LLM-as-judge. Sibling, not subordinate.

`fieldkit.agents` produces tasks; `fieldkit.eval` grades them. The seam is
the `SynthTask` record.

## Hard non-goals for v0.2

- **No LLM-as-judge in `AssertionGrader`.** Hybrid verification is real —
  some tasks can't be assertion-graded — but mixing the two grader styles
  inside one class hides which signal is which. Track LLM-as-judge in a
  separate `fieldkit.eval.LLMJudge` (already exists in v0.1 for RAG; this
  would extend it for agent rollouts).
- **No vendored persona / workspace templates.** The article ships with a
  hand-authored `personas.json` + `workspace_for_persona()` because the
  templates are domain-specific. The package exposes the *shape*, not the
  contents. v0.3 may add a small "starter pack" registry.
- **No execution model.** `SandboxRollout` will execute *some* abstract
  action plan against a workspace, but the action language (tool calls?
  pseudo-shell? function-calling JSON?) is open. v0.2 lands the WorkspaceSeed
  + Trajectory data model; v0.3 lands a default action interpreter.

## Why this lands in v0.2 specifically

The patches article (article #28) closed test-time-distilling at three
articles. That arc's v0.2 candidates were `VLLMClient`, `PassAtK`, `AgentRun`.
The clawgym-on-spark arc adds `agents.{Persona, WorkspaceSeed, SynthTask,
TaskAuthor}` + `eval.AssertionGrader`. Both arcs land together as v0.2,
giving the package a coherent "agents + test-time-eval" theme rather than
two separate cuts.
