---
title: 'T²PO on Spark — uncertainty-guided exploration on top of the GRPO loop'
date: 2026-05-07
author: 'Manav Sehgal'
product: 'NeMo'
stage: training
also_stages: [agentic]
difficulty: 'intermediate'
time_required: '~30 min read'
hardware: 'NVIDIA DGX Spark'
tags: [rl, grpo, gigpo, agentic, exploration, t2po, alfworld, lora, qwen]
summary: 'ICML 2026 spotlight paper layers two uncertainty-guided controls on top of GRPO for multi-turn agents. Reproducing the algorithmic deltas on a single Spark with Qwen 2.5 7B + LoRA on the Phase 6 ClawGym harness, ALFWorld benchmark.'
status: upcoming
series: 'Frontier Scout'
fieldkit_modules: [capabilities, eval, training]
---

## The paper, in one breath (ARTICLE OPENING — required at publish)

> tech-writer: this becomes a `## The paper, in one breath` section in
> the published article, placed immediately after the lede and before
> any "Why this matters for a personal AI builder" substrate framing.
> Pull thesis material from the eval's `## Hypothesis`; fill in the
> achieved beat after the experiment runs.

**Thesis.** Multi-turn agentic RL collapses or stalls because the policy spends most of its rollouts on low-information actions — moves that neither reduce uncertainty nor advance the task. T²PO (Token- and Turn-level Policy Optimization) adds two surgical uncertainty-guided controls on top of a GRPO-family backbone (the repo uses **GiGPO**): a **token-level "thinking intervention"** that caps the chain-of-thought budget per turn and triggers when marginal token uncertainty stops dropping, and a **turn-level dynamic-sampling (TDS) regeneration** that detects turns whose entropy barely shifted from the previous one and resamples them up to 2 retries. The mechanism that distinguishes T²PO from the obvious baseline is that it cuts wasted rollouts at *both* granularities — inside a single response and across turns — instead of only filtering trajectories post-hoc.

**Why this technique matters for a personal AI builder.** GRPO on a single Spark is already ~6.7 hours per training run (per the Phase 6 ClawGym arc); halving the wasted-rollout share means measurably more useful gradient signal per hour, which is the actual constraint on iteration speed for a one-Spark builder. It is also the first algorithmic refinement of the Phase 6 stack that doesn't require new memory headroom — purely additional logic on top of an already-fitting LoRA + vLLM-co-residence rig.

**Promise vs achieved.** Paper: 8-GPU full-FT of Qwen3-4B reports substantial gains in training stability and per-step performance on WebShop, ALFWorld, and Search QA (exact pp deltas to be filled in after a full read of the published tables). Spark: <fill in after run — task_pass / mean_turns / task_complete on the same 158-task ALFWorld dev split as Phase 6>. Delta: <one sentence on why the gap is what it is — likely the 8-GPU full-FT-vs-single-Spark-LoRA scale gap, not an algorithmic artefact>.

## Source paper

- arXiv: [2605.02178](https://arxiv.org/abs/2605.02178) — *T²PO: Uncertainty-Guided Exploration Control for Stable Multi-Turn Agentic Reinforcement Learning*
- Repo: [WillDreamer/T2PO](https://github.com/WillDreamer/T2PO) (9★, last commit 2026-05; ICML 2026 Spotlight)
- Popularity: 14/100 · 5 HF upvotes · citations not yet indexed

## Frontier Scout verdict

**spark-feasible** — Qwen 2.5 7B + LoRA + T²PO on top of the already-proven Phase 6 GRPO harness fits comfortably in ~38 GB of the 128 GB envelope; the only new code is ~80 LOC for GiGPO advantage and ~120 LOC for the TDS regenerate loop, both directly readable from `WillDreamer/T2PO/agent_system/multi_turn_rollout/rollout_loop.py`.

## Proposed Spark recipe

The cleanest path is to layer T²PO on top of the **already-shipped Phase 6 GRPO harness** at `articles/clawgym-on-spark/scripts/grpo_train.py` + `grpo_loop.sh`. That keeps the proven vLLM-co-residence + LoRA-reference-snapshot machinery and avoids a Ray/VeRL transplant onto aarch64 Spark.

1. **Resume the `tllm-build` container** (already has Qwen 2.5 7B + LoRA adapter from Phase 6; vLLM 0.20 with `--enable-lora`). Copy `grpo_train.py` → `t2po_train.py`.
2. **Port the GiGPO advantage estimator.** Vanilla GRPO uses per-trajectory advantages; GiGPO adds a per-step (turn-level) head: `A_total = α · A_traj + β · A_step`. Cribbed from VeRL's `core_algos.py`; ~50 LOC. Set `step_advantage_w=1.0` to match author config.
3. **Add the TDS resample loop** — the algorithm's headline. Reference impl is `agent_system/multi_turn_rollout/rollout_loop.py:T2PO_multi_turn_loop()` (line 631–760). Per-turn:
   - Compute `turn_level_entropy_t` from response logprobs (mean per-token entropy).
   - If `_step > 0`: `Δ = |turn_level_entropy_t − turn_level_entropy_{t-1}|`. Resample turn iff `0 < Δ < eta_threshold=0.3`. Cap at `max_try=2`.
4. **Add the thinking-token cap.** Set `num_think_tokens=450` as the response budget for any `<think>…</think>` block; the existing rollout supports `max_tokens` per generate call. This is the "token-level intervention" — the paper's marginal-uncertainty trigger collapses to a hard budget in practice.
5. **Pick one benchmark.** ALFWorld (text-only, 50 steps, deterministic) is the cleanest first target — drops in next to Phase 6's ClawGym harness with similar shape. Skip WebShop (live web, brittle on aarch64) and Search QA (needs the retrieval server in `examples/search_agent_trainer/retriever/`).
6. **Eval delta.** Run the same eval cadence as Phase 6 (every 10 grad steps, 158-task ALFWorld dev split). Compare against three baselines: Qwen base, Phase 6 SFT, Phase 6 GRPO@34. Report task_pass / mean_turns / task_complete deltas.
7. **Wall:** Phase 6 GRPO ran 34 steps in ~6.7 hr. T²PO adds up to 2× turn regeneration in the worst case → estimate **9–13 hr** for a full step-34 run.

## Open questions for the experiment

- (none — recipe should run as-is) — every primitive (LoRA training, vLLM co-residence with LoRA, multi-turn rollout, group advantage, KL reference snapshot) was already shipped + measured in the Phase 6 GRPO arc (article #31 `clawgym-on-spark-grpo`). T²PO is purely additive.
- ALFWorld dependency on `textworld 1.5+` is the one platform unknown — aarch64 wheel availability not yet verified, but a TextWorld-shaped Python game env is replaceable with the existing ClawGym sandbox if it doesn't drop in.
- VeRL/Ray multi-worker config in the upstream repo is **not** the model to copy on Spark — single-process trainer + single-process vLLM is the proven pattern.

## Suggested article shape

- **Would write?** yes
- **Suggested slug:** `t2po-uncertainty-guided-rl-on-spark`
- **Suggested stage:** training
- **Suggested series:** Frontier Scout
- **Suggested also_stages:** `[agentic]` (multi-stage, like the Phase 6 article)
- **Suggested tags:** `rl, grpo, gigpo, agentic, exploration, t2po, alfworld, lora, qwen`
- **Suggested summary:** ICML 2026 spotlight paper layers two uncertainty-guided controls on top of GRPO for multi-turn agents. Reproducing the algorithmic deltas on a single Spark with Qwen 2.5 7B + LoRA on the Phase 6 ClawGym harness, ALFWorld benchmark.
- **Suggested `fieldkit_modules`:** `[capabilities, eval, training]`

## Fieldkit fit (carried from eval)

- **Would import:** `fieldkit.eval` for the per-step task-pass / per-assertion / mean-turns rollups; `fieldkit.capabilities` for the 7B + LoRA budget claim. The published v0.2 surface covers it.
- **Would extend:** `fieldkit.training.LoraReferenceSnapshot` (introduced in Phase 6 GRPO) — T²PO uses the **same** frozen-reference + KL pattern; this becomes a **second consuming use case**, which graduates the snapshot from Phase-6-only utility to a documented, multi-article primitive.
- **Would propose for v0.3:** `fieldkit.agents.replay_messages_from_trajectory` — currently deferred (sketched in `articles/clawgym-on-spark/scripts/fieldkit_agents_v0_2_sketch.md`); T²PO's TDS regenerate path needs the **exact same** per-turn message reconstruction that Phase 6's `grpo_train.py:reconstruct_messages()` does. Promoting + executing this article supplies the second use case needed to lock the abstraction's parameter shape, unblocking extraction.
