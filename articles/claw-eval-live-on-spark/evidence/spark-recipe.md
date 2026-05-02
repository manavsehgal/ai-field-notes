# Proposed Spark recipe

There is no GitHub repo at eval time — the paper references a project page at `https://claw-eval-live.github.io` but no code/dataset URL surfaced via search. Recipe assumes the release ships shortly; in the interim, the article can demo the protocol on a 5–10 task subset reconstructed from the paper's task-family descriptions.

1. **Wait or proxy** — if the 105-task release isn't out, hand-author 5 representative tasks per family (HR, multi-system business, local workspace repair) using the paper's task structure as a template.
2. **Stand up the sandbox via NemoClaw** — each task gets a fresh OpenShell container with the workspace pre-populated from a fixture tarball. Use the verified file-transfer pattern from `reference_clawnav_file_transfer` (the `cat | openshell sandbox exec` workaround, since `openshell sandbox upload` is broken on v0.0.26).
3. **Mock the business services** as Flask/FastAPI processes inside the same network namespace — HR API, ticketing API, file-workspace state. Audit-log every request to a JSONL.
4. **Serve the agent under test via NIM**. Run two side-by-side: `llama-3.1-8b-instruct` and `nemotron-super-49b` (or 70B fp8 if the box has been freshly booted). The agent uses its tool-calling protocol against the mocked services + sandbox shell.
5. **Build the grader**: deterministic checks come from the audit log + workspace diff (file-state checksums, service-state asserts). Semantic checks (e.g. "did the agent's reply summarize the resolution correctly?") go through an LLM judge — Llama 8B as judge keeps it Spark-local.
6. **Score and compare**: publish per-task-family pass rates, mirror the paper's "leaderboard rank vs overall completion" finding on the smaller scale, and call out whether the local-first models exhibit the same HR / multi-system bottleneck pattern.

