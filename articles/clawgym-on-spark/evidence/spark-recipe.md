# Proposed Spark recipe

1. **Wait or proxy the data** — the GitHub org `ClawGym` exists but only ships a `.github` profile repo as of eval time. The article either waits for the 13.5K dataset drop, or generates a 1K-task subset using the paper's persona-driven recipe (LLM as task-author, mock workspace seeded from a list of skills) so the rest of the pipeline can be exercised end-to-end.
2. **Set up sandboxes via NemoClaw**: each per-task sandbox is an OpenShell container with a writable workspace at `/sandbox/.openclaw-data/workspace/` (per the `clawnav` file-transfer memory). NemoClaw already parallelizes well at 8–16 sandboxes/host before the box gets warm.
3. **SFT on Llama 3.1 8B Instruct via NeMo**, LoRA rank=16, on the rollout trajectories. Use `/opt/venv/bin/python3 -m pip` for any extra deps (per the NeMo container pip-trap memory). Single epoch, ~2–4 hours on Spark for a 13.5K-task corpus.
4. **Lightweight RL pass** — the abstract describes "parallelizes rollouts across per-task sandboxes." On Spark the natural shape is GRPO or DPO over rollout pairs (PPO is heavier and harder to fit alongside the rollout pool). 8 parallel sandboxes × short rollouts, reward = task-grader binary pass/fail.
5. **Evaluate on ClawGym-Bench's 200 instances** — once it ships. Until then, hold out a 200-task slice of the synthesized data.
6. **Compare against a NIM-served Nemotron baseline** (the project's existing agent default) to land an apples-to-apples "did the SFT actually help" measurement.

