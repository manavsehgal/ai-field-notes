# Proposed Spark recipe

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
