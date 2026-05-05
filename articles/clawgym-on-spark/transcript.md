# Provenance: clawgym-on-spark

## Origin

Promoted from Frontier Scout on 2026-05-02.

- arXiv: 2604.26904 — ClawGym: A Scalable Framework for Building Effective Claw Agents
- Repo: github.com/ClawGym — `.github` profile only at promotion time + at article publish; no upstream code or data
- Fast verdict: spark-feasible
- Deep verdict: spark-feasible (8B LoRA SFT + parallel-sandbox RL fits inside 128 GB envelope)

The full agent eval is at `evidence/feasibility-eval.md`. The proposed Spark recipe is at `evidence/spark-recipe.md`. Both are immutable provenance from the Frontier Scout pass.

## Editorial overlay

> Substrate first; numbers second. ClawGym shipped only `.github/`, so we built the substrate ourselves — persona task synth, sandbox harness, 200-task corpus, LoRA SFT, matched-base eval — and ran the only honest experiment that data permits: matched-base SFT vs base on the same held-out 158. The "doesn't give up early" diagnostic is the load-bearing finding.

## Source material map

- `evidence/paper.pdf` — arxiv PDF, 1.7 MB
- `evidence/paper-meta.json` — papers.json entry (popularity 30/100, 44 HF upvotes)
- `evidence/feasibility-eval.md` — full Frontier Scout eval (immutable)
- `evidence/spark-recipe.md` — extracted Proposed Spark recipe section as runbook
- `evidence/repo-snapshot/README.txt` — stub (no public repo)
- `evidence/runs/2026-05-03-phase1-synth/NOTES.md` — Phase 1 substrate validation (synth + grader, 7/8 personas, hand-built fake post-rollout 5/5)
- `evidence/runs/2026-05-03-phase2-rollout/NOTES.md` — Phase 2 sandbox rollout harness (gold-action mock 7/7, real Llama 3.1 8B 2/7 + 24/36 asrt, five recurring failure modes catalogued)
- `evidence/runs/2026-05-04-phase3-corpus/` — 200-task synth corpus, 25 × 8 personas, 83 min wall
- `evidence/runs/2026-05-04-phase3-baseline/NOTES.md` — Phase 3 baseline against Llama 3.1 8B (17/200 = 8.5% task / 497/980 = 50.7% per-asrt; per-persona spread 0–24%; per-assertion-kind synth-noise floors)
- `evidence/runs/2026-05-04-phase4-sft/NOTES.md` — Phase 4 LoRA SFT on Qwen 2.5 7B Instruct (42 records = 17 PASS + 25 near-miss; 11 optimizer steps; loss 1.21 → 0.39; 8/8 well-formed bash blocks across all personas in smoke eval)
- `evidence/runs/2026-05-04-phase5-eval/NOTES.md` — Phase 5 matched-base eval on 158 held-out (Qwen base 4/158 = 2.5% / 31.8% per-asrt → Qwen + clawgym 10/158 = 6.3% / 46.8% per-asrt; +3.8 pp task / +15.0 pp asrt; six personas improved, one regressed; failure-mode traces for `synth-indie-game-dev-01` recovery + `synth-data-science-researcher-03` regression)
- `evidence/runs/2026-05-04-phase5-eval/comparison.json` — full per-persona + per-assertion-kind breakdown
- `evidence/runs/2026-05-04-phase5-eval/{qwen-base,qwen-sft}/trajectories.jsonl` — 158 trajectories per side, JSONL with per-turn (action, observation) records and per-task `final_grade`

## Authoring scripts (all under `scripts/`)

- `synth_tasks.py` — persona-driven task author (NIM-served Nemotron Nano 9B v2)
- `rollout.py` — sandbox harness + agent loop + grader integration (`Sandbox` ABC + `LocalTempSandbox` + `RolloutDriver` + `Trajectory`/`TurnRecord` records + `MockClient` for offline validation)
- `grader.py` — pure-function programmatic grader over five assertion primitives
- `personas.json`, `skills.json`, `task_schema.md` — synth inputs and contract
- `to_sft_record.py` — flatten `(task, trajectory)` → SFT record (prompts byte-identical to rollout time)
- `train_lora_sft.py` — minimal HF + peft LoRA SFT loop in `tllm-build` (PyTorch 2.11 + transformers 5.7 + peft 0.19 + torchao 0.17 + accelerate 1.13)
- `smoke_eval_adapter.py` — single-task-per-persona inference smoke test
- `compare_phase5.py` — held-out-by-training-set-membership filter + B−A comparison + per-persona / per-assertion-kind breakdown + JSON dump
- `run_phase5_pipeline.sh` — Phase 5 driver: guards on rollout #1 completion → launches rollout #2 → defers comparison to manual call

## Container + service state at publish

- `tllm-build` container: PyTorch 25.11 + vLLM 0.20.0 + tLLM (writable working tree at `/tmp/tllm-rw/`); LoRA dependencies (peft 0.19.1 + accelerate 1.13.0 + torchao 0.17.0) added during Phase 4. vLLM stopped at end of Phase 5 to free GPU.
- `nim-llama31-8b`: Phase 2/3's NIM-served Llama 3.1 8B Instruct (`max_model_len=8192`). Stopped between phases.
- `nim-nemotron-nano-9b-v2`: Phase 1's reasoning model for task synthesis (`max_model_len=131072`, NVFP4 on vLLM, `chat_template_kwargs={"thinking": False}`). Stopped after Phase 1 synth.
- `pgvector`: idempotent CPU-side; not touched in this article.

## Voice deviations from house style — none

The article follows the Frontier Scout pattern established by `autoresearchbench-on-spark`, `test-time-distilling-for-exploration`, `runtime-frontier-six-patches-on-spark`, and `pass-at-k-after-the-seventh-patch`: opening lede + `## The paper, in one breath` (Thesis / Why-this-matters / Promise-vs-achieved) + `## Why this matters for a personal AI builder` + `## Architectural context` (with one inline fn-diagram) + journey + verification + tradeoffs + unlocks + closing.

The signature SVG (`ClawgymSftLift`) follows the paired-bar pattern from `BaselineTrainingEnvelope` and `LoraVoiceTransfer` — two bars + delta callout, accent on the SFT side, halo containment per the validator invariants.
