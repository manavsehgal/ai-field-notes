# clawgym-on-spark scripts

Phase 1 harness for the article. See `evidence/runs/2026-05-03-phase1-synth/NOTES.md`
for the session log + design choices.

## Layout

| File | What it does |
|---|---|
| `task_schema.md` | The JSON shape every synthesized task must conform to. |
| `personas.json` | 8 persona archetypes (role, context, skill_focus). |
| `skills.json` | 15 file-management skills the agent has access to. |
| `synth_tasks.py` | LLM-driven task author. Reads personas + skills + a hand-authored per-persona workspace template, calls a NIM endpoint, validates JSON output, writes JSONL. |
| `grader.py` | Pure-function grader. Walks a task's verifiable assertions against a post-rollout sandbox directory, returns per-assertion pass/fail + a binary task-level result. |
| `fieldkit_agents_v0_2_sketch.md` | Design doc for the proposed `fieldkit.agents` v0.2 module — what `tech-writer extract` will lift after the article publishes. |

## Quick start

```bash
# 1. Make sure a NIM is running (we use Nemotron Nano 9B v2 by default;
#    Llama 3.1 8B works too — pass --model nvidia/llama-3.1-8b-instruct).
docker start nim-nemotron-nano-9b-v2
curl -sf http://localhost:8000/v1/models | python3 -m json.tool | head -10

# 2. Create a venv with fieldkit installed.
python3 -m venv /tmp/fk-clawgym
/tmp/fk-clawgym/bin/pip install -q fieldkit

# 3. Synthesize a small batch.
/tmp/fk-clawgym/bin/python3 synth_tasks.py \
    --personas all \
    --per-persona 2 \
    --out /tmp/tasks-16.jsonl \
    --debug

# 4. Sanity-check the grader against the seed (should fail most assertions
#    since the agent hasn't operated on the workspace yet).
/tmp/fk-clawgym/bin/python3 grader.py \
    --tasks /tmp/tasks-16.jsonl \
    --dry-run
```

## What's NOT here yet

Phase 1 stops at task synthesis + grading. Phase 2 (next session) adds:

- `rollout.py` — pipe a task into a NemoClaw OpenShell sandbox, capture the
  agent's (action, observation) trajectory, and return a `Trajectory` record
  the grader can score.
- `sft.py` — LoRA fine-tune Llama 3.1 8B Instruct on the synthesized
  trajectories using NeMo. (Will be light wrapper code; the heavy lift is
  the NeMo container.)
- `rl.py` — lightweight GRPO loop over 8 parallel sandboxes. Reward = grader
  binary pass/fail.
- `eval.py` — held-out 200-task split + comparison vs. NIM Nemotron baseline.
