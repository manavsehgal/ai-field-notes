# Synthesized task schema (Phase 1)

A *task* is one self-contained file-management problem an agent can attempt in
a sandbox. The schema must let downstream stages — sandbox seeding, agent
rollout, and binary-grader — all consume the same record.

## JSON shape

```json
{
  "task_id": "synth-NNNN",
  "persona": {
    "role": "<one of personas.json keys>",
    "context": "one-sentence situating context"
  },
  "intent": "natural-language task description (1–3 sentences, the prompt the agent receives)",
  "workspace_seed": {
    "files": [
      {"path": "<rel/path>", "kind": "text" | "binary-stub", "content": "<text body>" | "<size_bytes int>"}
    ]
  },
  "skills_required": ["<skill.name>", ...],
  "verifiable_assertions": [
    {"kind": "file_exists", "path": "<rel/path>"},
    {"kind": "file_not_exists", "path": "<rel/path>"},
    {"kind": "file_contents_contain", "path": "<rel/path>", "must_contain": ["<substring>", ...]},
    {"kind": "file_contents_match_regex", "path": "<rel/path>", "regex": "<python regex>"},
    {"kind": "file_unchanged", "path": "<rel/path>"}
  ],
  "difficulty": "easy" | "medium" | "hard",
  "estimated_steps": <int>
}
```

## Design choices

- **`kind: binary-stub`** — for binary files (images, archives, compiled
  artifacts) we don't materialize real bytes. The sandbox creates an empty
  placeholder of the named size. Every file-management task is about
  *structure* (paths, naming, presence/absence), not pixel content.
- **Assertion primitives are deliberately small.** Five kinds cover ~95% of
  file-ops verification. The grader is a pure function that walks the
  assertions list and returns a binary pass/fail per assertion.
- **`skills_required` is informational**, not a gate. The agent has access
  to a fixed file-ops + shell skill bundle; this field tells the synth
  prompt which subset to emphasize.
- **Workspace seed is fully self-contained.** No external downloads, no
  network calls, no hidden state. A task plus its seed is enough to
  reproduce the sandbox setup byte-for-byte.

## Hybrid verification (paper alignment)

The ClawGym abstract describes "hybrid verification mechanisms." Our split:

- **Programmatic grader** (this article): the five assertion primitives above,
  evaluated by a pure Python function against the post-rollout sandbox FS.
- **LLM-as-judge** (deferred): for tasks where structural assertions can't
  capture intent (e.g., "summarize the README" — no fixed string to match).
  Phase 1 synth is restricted to programmatically-gradable tasks; LLM-judge
  tasks are a v0.2 fieldkit candidate.
