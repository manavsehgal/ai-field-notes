# Source material: clawgym-on-spark-grpo

Cleaned provenance for this article. Phase 6 of the [clawgym-on-spark](/articles/clawgym-on-spark/) work, executed across two evening sessions (2026-05-05). The article is the companion essay to the run; this transcript is the bookkeeping.

## Sessions and commits

| Date | Session | Result | Commit |
|---|---|---|---|
| 2026-05-05 | Phase 6 step 2 | `rollout.py:parse_action()` recognizes prose `Task complete.` as a stop sentinel — 19/19 inline parse_action cases + Phase 5 trajectory replay (69/158 short-circuit, 433 turns reclaimed). | `f0cc227` |
| 2026-05-05 | Phase 6 step 3 | `reward.py` ships `compute_reward(grade, n_turns, mode={binary, shaped})` + `compute_group_advantages` (group-relative). 9/9 inline tests + offline replay (binary 154+148 zeros / shaped 1+5 zeros). Smoke 4 tasks × K=4 against vLLM-served Qwen 7B + clawgym SFT adapter — 4/4 tasks have non-zero advantage variance, 5/16 rollouts stop via `task_complete` at temp=0.8. | `6e6b959` |
| 2026-05-05 | Phase 6 step 4 (scaffold + dryrun) | `build_grpo_pool.py`, `grpo_train.py` (~280 lines), `grpo_loop.sh` (~200 lines). Dryrun on 2 near-miss tasks × K=4 validates plumbing end-to-end. Two hold-pattern fixes: `--reference-adapter` (decouples KL reference from `--adapter-init`), CPU-resident snapshot/swap (392/392 tensors mapped via `.{adapter_name}.weight` ↔ `.weight` key transform). | `5f2d04f` |
| 2026-05-05 (evening) | Phase 6 step 4 (full run) | 34 GRPO steps in 8.5 hr wall, ERROR'd at step 35 with 8/8 tasks at zero advantage variance (pool converged). Eval-1 at step 25, eval-2 at step 34. GRPO@34 beats Phase 5 SFT on every metric: task pass 10→13, per-asrt 365→389, mean turns 12.0→5.0, task_complete 0/158→154/158, wall 28.3s→10.7s. | (run artifacts; uncommitted) |

## Run artifact map

`articles/clawgym-on-spark/evidence/runs/2026-05-06-phase6-grpo/` (gitignored — reproducibility aid, not committed):

```
.
├── per_step_metrics.csv          # 34 rows × 22 cols (loss, KL, mean_turns, …)
├── SUMMARY.md                    # auto-generated regenerable via analyze_grpo_run.py
├── loop.log                      # full bash + trainer + rollout logs
├── nohup.log                     # background launcher log
├── step-001/ … step-035/         # 35 step dirs, last (step-035) bundle-only
│   ├── trajectory_bundle.jsonl   # K=4 rollouts × 8 tasks per step
│   ├── adapter/                  # peft LoRA tensors (steps 1-34)
│   └── grpo_step_summary.json    # one-record summary with policy_loss, KL, weight_delta
├── eval-step-025/
│   ├── trajectories.jsonl        # 158 held-out rollouts
│   ├── comparison.json           # vs Qwen 2.5 7B base
│   ├── compare_vs_phase5_sft.json
│   └── rollout.log
└── eval-step-034/                # same shape; the article's headline anchor
```

## Headline numbers (article body's source of truth)

Verified against `eval-step-034/comparison.json` and `compare_vs_phase5_sft.json`:

```
matched-base eval @ step 34, 158 held-out tasks, vLLM Qwen 2.5 7B + adapter, T=0.2

           Qwen-base       SFT (Phase 5)    GRPO@34 (Phase 6)
task_pass  4/158 (2.5%)    10/158 (6.3%)    13/158 (8.2%)         +1.9pp vs SFT
per-asrt   248/780 (31.8%) 365/780 (46.8%)  389/780 (49.9%)       +3.1pp vs SFT
mean_turns 4.59            12.00            5.00                  −7.0
mean_wall  12.4s           28.3s            10.7s                 −62%
task_complete 147/158      0/158            154/158               +97.5pp vs SFT
```

## Per-step training trajectory (selected)

Sourced from `per_step_metrics.csv`:

```
step  mean_turns  task_complete%  KL       comment
   1        7.38           75.0   0.0000   step 1 — already shorter than SFT's fixed 12
   5        6.53           81.2   0.0001   turns dropping, stops climbing
  10        5.28           93.8   0.0002   first 90+% TC step
  16        4.41          100.0   0.0006   first 100% task_complete step
  22        4.38          100.0   0.0011   second 100% step
  29        3.69          100.0   0.0011   best mean-turns step
  32        4.09          100.0   0.0016   last 100% step
  34        4.62           96.9   0.0020   final saved adapter — eval-2 anchor
  35      (CRASH)         (n/a)   (n/a)    8/8 tasks at zero advantage variance
```

## Pool convergence diagnostic (step 35)

`loop.log` final entries before the trainer ERROR:

```
[step 35] sampled tasks: synth-data-science-researcher-23, synth-academic-author-16,
  synth-technical-writer-05, synth-backend-developer-10, synth-backend-developer-13,
  synth-data-science-researcher-01, synth-academic-author-24, synth-backend-developer-06

  → synth-backend-developer-06  rewards=[0.917, 0.917, 0.917, 0.917]  stdev=0.000  signal=NO
  (similar zero-stdev across all 8 sampled tasks)

signal: 0/8 tasks have non-zero advantage variance
ERROR: no usable rollouts (all-zero advantages, missing tasks, or too long)
```

The `0.917` reward is `1.0 − 0.2 × 5/12` — every K=4 rollout passed every assertion in 5 turns. Pool saturated; the right answer is to grow the pool. Filed in handoff as "loop hardening" for future RL-loop reuse.

## Hold-pattern fixes captured along the way

- **`--reference-adapter`** (decouples KL reference from `--adapter-init`). Pass the Phase 5 SFT-init dir on every step for classic GRPO fixed-SFT-init reference. Safetensors-key transform `.{adapter_name}.weight` ↔ `.weight` validated 392/392 tensors mapped.
- **CPU-resident snapshot/swap** for the KL reference. peft's `load_adapter(adapter_name="reference", is_trainable=False)` crashes on `device_map="auto"` whenever the GPU has anything else resident — verified with vLLM up AND with the trainer alone (peft's offload-detection over-triggers on Spark unified memory). Solution is a 30-line snapshot/swap. **Strongest fieldkit-extraction candidate from this run.**
- **Co-residence test at `--gpu-memory-utilization=0.4`** — vLLM (50 GiB) + trainer (Qwen 7B bf16 + LoRA + activations, ~28 GiB peak) co-reside cleanly. But vLLM 0.20 in `tllm-build` does not expose `/v1/load_lora_adapter` (404, openapi.json has zero LoRA endpoints), so co-residence buys zero wall in this loop architecture. Persistent-trainer refactor would need the API.
- **Unified-memory orphan**: caught and killed an orphaned `VLLM::EngineCore` (PPID=1, holding 108 GB unified memory) en route. Saved as `feedback_vllm_engine_core_orphan` memory: `pkill -f 'vllm|EngineCore'` with `free -h` verification is the corrected pattern.

## fieldkit v0.2 candidates surfaced

Three from this run, ordered by extraction value:

1. **`fieldkit.training.LoraReferenceSnapshot`** — strongest. CPU-resident snapshot of a peft adapter's LoRA tensors via safetensors with `.{adapter_name}.weight` ↔ `.weight` key transform; context manager that swaps the snapshot in for one no-grad forward pass and restores trainable weights. ~30 lines. Source: `grpo_train.py` (`--reference-adapter` + snapshot/swap blocks). Opens `fieldkit.training`.
2. **`fieldkit.training.WeightDeltaTracker`** — pre/post snapshot of trainable params with L2 + max|Δ| report. ~15 lines. Source: `grpo_train.py` (`--check-weight-delta` block).
3. **`fieldkit.agents.replay_messages_from_trajectory`** — *defer until next article supplies a second use case*. Reconstruction of (system, user, assistant, observation, …) message list from a saved Trajectory. Currently in two places that must stay byte-identical: `rollout.py:RolloutDriver.rollout()` and `grpo_train.py:reconstruct_messages()`. Right callable interface won't be obvious until a second article uses it.

A formal `tech-writer extract` pass against this article runs in the next session and lands these in `fieldkit/CHANGELOG.md` under `[Unreleased]`.

## What's *not* in the article

- The full per-step CSV (34 rows × 22 cols) lives in `evidence/runs/`. The article uses 8 illustrative rows.
- The eval-1 (step 25) numbers are mentioned in the tradeoffs section but not given a full table; eval-2 (step 34) is the anchor table because step 34 is the last successfully-saved adapter.
- The dryrun's specific weight-delta trace (`L2=0.063, max|Δ|=1e-5`). Mentioned in the trainer log block but not as a separate diagnostic.
- The smoke run's full 16-rollout reward distribution (only the cleanest single example, `synth-backend-developer-00` with `[0.05, 0.633, 0.05, 0.55]`, is quoted).
- The pre-step-1 base-load wall and adapter-load wall (124s and 2.5s respectively) appear once in the trainer log block but not in the main timing analysis. The 8.5-hour total dominates.
