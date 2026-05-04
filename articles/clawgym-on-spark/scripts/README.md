# clawgym-on-spark scripts

Phase 1 + Phase 2 harness for the article. See
`evidence/runs/2026-05-03-phase1-synth/NOTES.md` for synth choices and
`evidence/runs/2026-05-03-phase2-rollout/NOTES.md` for rollout choices.

## Layout

| File | What it does |
|---|---|
| `task_schema.md` | The JSON shape every synthesized task must conform to. |
| `personas.json` | 8 persona archetypes (role, context, skill_focus). |
| `skills.json` | 15 file-management skills the agent has access to. |
| `synth_tasks.py` | LLM-driven task author. Reads personas + skills + a hand-authored per-persona workspace template, calls a NIM endpoint, validates JSON output, writes JSONL. |
| `grader.py` | Pure-function grader. Walks a task's verifiable assertions against a post-rollout sandbox directory, returns per-assertion pass/fail + a binary task-level result. |
| `rollout.py` | **Phase 2.** Sandbox rollout harness. Materializes a task's workspace into a fresh dir, drives an LLM-agent loop (NIM Llama 3.1 8B by default) for up to N turns with a one-bash-block-per-turn protocol, captures the (action, observation) trajectory, and grades the final state. `--mock-actions` mode replays canned action scripts for harness validation without a NIM. |
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

## Phase 2 quick start

```bash
# 1. Start the agent (Llama 3.1 8B over NIM is the default).
docker start nim-llama31-8b
curl -sf http://localhost:8000/v1/models | python3 -m json.tool | head -10

# 2. Validate the harness end-to-end with the gold-action mock (no NIM).
python3 rollout.py \
    --tasks ../evidence/runs/2026-05-03-phase1-synth/tasks-8.jsonl \
    --mock-actions ../evidence/runs/2026-05-03-phase2-rollout/gold-actions-all.jsonl \
    --out-dir /tmp/clawgym-rollout-gold/
# Expected: 7/7 PASS

# 3. Real rollout against NIM 8B.
python3 rollout.py \
    --tasks ../evidence/runs/2026-05-03-phase1-synth/tasks-8.jsonl \
    --out-dir /tmp/clawgym-rollout-nim8b/ \
    --debug
```

## Backend abstraction

`rollout.Sandbox` is the abstract base; `LocalTempSandbox` is the
development backend (tempdir + `subprocess.run`). For Phase 4 GRPO at
8-parallel, swap in a `NemoClawSandbox` that calls
`openshell sandbox exec -n clawnav --no-tty` per the
`reference_clawnav_file_transfer` memory pattern.

## What's NOT here yet

- `sft.py` — LoRA fine-tune Llama 3.1 8B Instruct on captured trajectories
  using NeMo. (Will be light wrapper; heavy lift is the NeMo container.)
- `rl.py` — lightweight GRPO loop over 8 parallel sandboxes. Reward = grader
  binary pass/fail.
- `eval.py` — held-out 200-task split + comparison vs. NIM Nemotron baseline.
