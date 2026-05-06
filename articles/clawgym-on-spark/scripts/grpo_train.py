"""One GRPO gradient step on a trajectory bundle — Phase 6 trainer.

Reads a `trajectory_bundle.jsonl` produced by `phase6_smoke.py` (or its
full-run sibling), reconstructs each rollout's exact prompt/response
sequence, computes per-token log-probs against the current policy
adapter and a frozen reference adapter, and applies one
advantage-weighted gradient step on the policy LoRA. Saves the updated
policy adapter to --out-dir.

GRPO objective (single-epoch, sequence-level advantage):
    L = -A * mean_token(log π_θ(a_t | h_t))   over assistant tokens
        + β * mean_token(KL[π_θ || π_ref])     K3 estimator

  - A is the group-relative advantage already computed in the bundle
    (compute_group_advantages in reward.py: a_i = (r_i − μ) / (σ + ε)).
  - The K3 KL estimator (Schulman 2020) is exp(−Δ) − (−Δ) − 1 with
    Δ = log π_θ − log π_ref. Stable, non-negative, and unbiased.
  - PPO clipping is a no-op here because the rollout policy IS the
    starting policy (single-epoch — ratio = 1 at step 0). The clip
    layer can be added when we add multi-epoch training.

Architecture choice (per HANDOFF.md design choice resolution):
  Log-probs are recomputed in-process via transformers + peft. We do
  NOT consume vLLM logprobs from the rollout side. This keeps Phase 4's
  proven transformers stack as the source of truth; vLLM is used only
  for fast sampling.

Memory: Qwen 7B base in bf16 (~16 GB) + LoRA-trainable + LoRA-frozen
(~330 MB total) + per-rollout activations (long sequences with grad
checkpointing should keep peak under 30 GB on Spark's unified pool).

Usage (inside tllm-build container):
    python3 grpo_train.py \\
        --bundle /work/clawgym-grpo/run-1/step-001/trajectory_bundle.jsonl \\
        --tasks-pool /work/clawgym-grpo/tasks-grpo-pool.jsonl \\
        --adapter-init /work/clawgym-sft/adapter-v1 \\
        --out-dir /work/clawgym-grpo/run-1/step-001/adapter/ \\
        --base-model Qwen/Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from rollout import (  # noqa: E402
    LocalTempSandbox,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    render_files_block,
)


def reconstruct_messages(task: dict, file_listing: str, turns: list[dict]) -> list[dict]:
    """Recreate the (system, user, assistant_1, user_obs_1, …) sequence
    that the policy actually saw at rollout time. Mirrors RolloutDriver."""
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(
                intent=task["intent"], file_listing=file_listing,
            ),
        },
    ]
    for turn in turns:
        msgs.append({"role": "assistant", "content": turn["agent_response"]})
        action = turn.get("action")
        parse_error = turn.get("parse_error")
        observation = turn.get("observation")
        if parse_error:
            msgs.append({
                "role": "user",
                "content": (
                    f"PARSE ERROR: {parse_error}. Reply with ONE ```bash``` block "
                    "containing one command, or TASK_COMPLETE on a line by itself."
                ),
            })
        elif action and action.get("kind") == "done":
            break  # last turn, no follow-up user message
        elif observation is not None:
            timed_out = observation.get("timed_out", False)
            msgs.append({
                "role": "user",
                "content": (
                    f"OBSERVATION (exit {observation['exit_code']}"
                    f"{', TIMED OUT' if timed_out else ''}):\n"
                    f"--- stdout ---\n{observation['stdout']}\n"
                    f"--- stderr ---\n{observation['stderr']}\n"
                    "Next command (one ```bash``` block) or TASK_COMPLETE."
                ),
            })
    return msgs


def build_input_and_assistant_mask(
    msgs: list[dict], tokenizer, max_length: int = 8192,
) -> tuple[list[int], list[int]]:
    """Tokenize via chat template + identify assistant-token positions
    by prefix walk. Same trick as Phase 4's train_lora_sft.py."""
    prev_len = 0
    assistant_mask: list[int] = []
    for i, m in enumerate(msgs):
        partial = tokenizer.apply_chat_template(
            msgs[: i + 1], tokenize=True, add_generation_prompt=False, return_dict=True,
        )
        cur_len = len(partial["input_ids"])
        new_tokens = cur_len - prev_len
        assistant_mask.extend([1 if m["role"] == "assistant" else 0] * new_tokens)
        prev_len = cur_len
    full = tokenizer.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=False, return_dict=True,
    )
    input_ids = full["input_ids"]
    while len(assistant_mask) < len(input_ids):
        assistant_mask.append(0)
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        assistant_mask = assistant_mask[:max_length]
    return input_ids, assistant_mask


def per_token_logp(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Per-token log-prob of the actually-emitted next token (shifted)."""
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[:, :-1, :]
    targets = input_ids[:, 1:]
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    return log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [B, N-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, help="trajectory_bundle.jsonl from phase6_smoke.py")
    ap.add_argument("--tasks-pool", required=True, help="tasks-grpo-pool.jsonl")
    ap.add_argument("--adapter-init", required=True, help="dir of adapter to load+train")
    ap.add_argument("--reference-adapter", default=None,
                    help="dir of adapter to use as the FROZEN KL reference. If "
                         "omitted, snapshot from --adapter-init at step start "
                         "(trust-region from prior step). Pass the SFT-init "
                         "adapter on every step to get classic GRPO fixed-SFT-init.")
    ap.add_argument("--out-dir", required=True, help="updated adapter output dir")
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--lr", type=float, default=1e-5,
                    help="lower than SFT lr (2e-4) — RL is jumpier")
    ap.add_argument("--kl-beta", type=float, default=0.04,
                    help="KL penalty weight against frozen reference adapter")
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--check-weight-delta", action="store_true",
                    help="Snapshot LoRA weights pre/post step and report L2 delta")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    print(f"=== loading tasks pool from {args.tasks_pool} ===", flush=True)
    tasks: dict[str, dict] = {}
    with open(args.tasks_pool) as f:
        for line in f:
            line = line.strip()
            if line:
                t = json.loads(line)
                tasks[t["task_id"]] = t
    print(f"  {len(tasks)} tasks in pool", flush=True)

    print(f"=== loading bundle from {args.bundle} ===", flush=True)
    bundle: list[dict] = []
    with open(args.bundle) as f:
        for line in f:
            line = line.strip()
            if line:
                bundle.append(json.loads(line))
    print(f"  {len(bundle)} rollout groups in bundle", flush=True)

    print(f"=== loading tokenizer + model: {args.base_model} ===", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    base.gradient_checkpointing_enable()
    if hasattr(base, "config"):
        base.config.use_cache = False
    print(f"  base loaded in {time.time()-t0:.1f}s", flush=True)

    print(f"=== loading policy adapter from {args.adapter_init} ===", flush=True)
    t0 = time.time()
    model = PeftModel.from_pretrained(base, args.adapter_init, is_trainable=True)
    print(f"  policy adapter loaded in {time.time()-t0:.1f}s", flush=True)
    model.print_trainable_parameters()

    # Reference snapshot: when --kl-beta > 0, freeze a set of LoRA weights
    # as a CPU-resident snapshot. Reference forward swaps the LoRA tensors
    # in for one forward pass, then restores the trainable policy weights.
    # This avoids peft 0.19's meta-parameter bug with multi-adapter loads
    # under device_map="auto" while letting us choose the KL reference.
    #
    # Two reference modes:
    #  - --reference-adapter omitted  → snapshot from current policy
    #    (= start-of-step adapter). Each step's KL is to that step's
    #    starting policy: trust-region from prior step (TRPO-style).
    #  - --reference-adapter <path>   → load LoRA weights from that adapter
    #    on disk into the snapshot. Pass the SFT-init dir on every step to
    #    get classic GRPO fixed-SFT-init reference.
    use_reference = args.kl_beta > 0
    ref_snapshot: dict[str, torch.Tensor] = {}
    if use_reference:
        if args.reference_adapter and args.reference_adapter != args.adapter_init:
            from safetensors.torch import load_file as load_safetensors
            ref_path = Path(args.reference_adapter) / "adapter_model.safetensors"
            if not ref_path.exists():
                print(f"ERROR: --reference-adapter has no adapter_model.safetensors at {ref_path}",
                      file=sys.stderr)
                return 2
            raw_ref = load_safetensors(str(ref_path), device="cpu")
            # peft inserts the adapter_name (default: "default") between the
            # parameter path and `.weight`. Translate the file's raw keys
            # back to model.named_parameters() names by inserting "default".
            adapter_name = "default"
            ref_suffix = f".{adapter_name}.weight"
            n_matched = 0
            for n, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                if n.endswith(ref_suffix):
                    raw_key = n[: -len(ref_suffix)] + ".weight"
                    if raw_key in raw_ref:
                        ref_snapshot[n] = raw_ref[raw_key].clone()
                        n_matched += 1
                        continue
                if n in raw_ref:
                    ref_snapshot[n] = raw_ref[n].clone()
                    n_matched += 1
                else:
                    print(f"WARN: no ref weight for {n}", file=sys.stderr)
            print(f"  ref snapshot: loaded {n_matched} LoRA tensors from "
                  f"{args.reference_adapter} (file had {len(raw_ref)} entries)",
                  flush=True)
        else:
            for n, p in model.named_parameters():
                if p.requires_grad:
                    ref_snapshot[n] = p.detach().clone().cpu()
            ref_src = args.adapter_init if args.reference_adapter is None \
                else args.reference_adapter
            print(f"  ref snapshot: {len(ref_snapshot)} LoRA tensors frozen "
                  f"from current policy (= {ref_src})", flush=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
    )

    pre_snapshot: dict[str, torch.Tensor] = {}
    if args.check_weight_delta:
        for n, p in model.named_parameters():
            if p.requires_grad:
                pre_snapshot[n] = p.detach().clone().cpu()
        print(f"  snapshotted {len(pre_snapshot)} trainable tensors for delta check", flush=True)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    n_groups_used = 0
    n_groups_zero_adv = 0
    n_groups_missing_task = 0
    n_rollouts_used = 0
    n_rollouts_too_long = 0
    n_rollouts_no_assistant = 0
    sum_loss = 0.0
    sum_pol_loss = 0.0
    sum_kl = 0.0
    n_assistant_tokens_total = 0

    t_train = time.time()
    for group in bundle:
        tid = group["task_id"]
        advantages = group["advantages"]
        if all(abs(a) < 1e-6 for a in advantages):
            n_groups_zero_adv += 1
            continue
        task = tasks.get(tid)
        if task is None:
            print(f"  WARN: task {tid} not in pool; skipping group", file=sys.stderr)
            n_groups_missing_task += 1
            continue

        sb = LocalTempSandbox()
        sb.materialize(task)
        file_listing = render_files_block(sb.list_files())
        sb.cleanup()

        any_used = False
        for rollout, adv in zip(group["rollouts"], advantages):
            if abs(adv) < 1e-6:
                continue
            traj = rollout["trajectory"]
            msgs = reconstruct_messages(task, file_listing, traj["turns"])
            input_ids, assistant_mask = build_input_and_assistant_mask(
                msgs, tokenizer, args.max_length,
            )
            n_assistant = sum(assistant_mask)
            if n_assistant == 0:
                n_rollouts_no_assistant += 1
                continue
            if len(input_ids) >= args.max_length:
                n_rollouts_too_long += 1
                continue

            input_ids_t = torch.tensor([input_ids], dtype=torch.long, device=base.device)
            attn_t = torch.ones_like(input_ids_t)
            mask_shifted = torch.tensor(
                [assistant_mask[1:]], dtype=torch.bool, device=base.device,
            )
            adv_t = torch.tensor(adv, dtype=torch.float32, device=base.device)

            pol_logp = per_token_logp(model, input_ids_t, attn_t)
            pol_logp_a = pol_logp[mask_shifted]

            policy_loss = -adv_t * pol_logp_a.mean()

            if use_reference:
                # Swap LoRA params: stash current (trainable) weights, load
                # frozen snapshot, forward under no_grad, restore trainable.
                stash: dict[str, torch.Tensor] = {}
                with torch.no_grad():
                    for n, p in model.named_parameters():
                        if n in ref_snapshot:
                            stash[n] = p.data.clone()
                            p.data.copy_(ref_snapshot[n].to(p.device, dtype=p.dtype))
                with torch.no_grad():
                    ref_logp = per_token_logp(model, input_ids_t, attn_t)
                with torch.no_grad():
                    for n, p in model.named_parameters():
                        if n in stash:
                            p.data.copy_(stash[n])
                ref_logp_a = ref_logp[mask_shifted]
                delta = pol_logp_a - ref_logp_a  # log π_pol − log π_ref
                kl_per_token = torch.exp(-delta) - (-delta) - 1.0  # K3 estimator
                kl_loss = args.kl_beta * kl_per_token.mean()
                loss = policy_loss + kl_loss
            else:
                kl_loss = torch.zeros((), device=base.device)
                loss = policy_loss

            loss.backward()

            n_rollouts_used += 1
            n_assistant_tokens_total += int(pol_logp_a.numel())
            sum_loss += float(loss.detach())
            sum_pol_loss += float(policy_loss.detach())
            sum_kl += float(kl_loss.detach())
            any_used = True
            print(f"  {tid} rollout-{rollout['rollout_idx']} "
                  f"adv={adv:+.3f} n_asrt={n_assistant} "
                  f"pol_loss={float(policy_loss.detach()):+.4f} "
                  f"kl={float(kl_loss.detach()):+.4f}", flush=True)
        if any_used:
            n_groups_used += 1

    if n_rollouts_used == 0:
        print("\nERROR: no usable rollouts (all-zero advantages, missing tasks, or too long)",
              file=sys.stderr)
        return 1

    # Average gradients across used rollouts
    for p in model.parameters():
        if p.grad is not None:
            p.grad /= n_rollouts_used

    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], 1.0,
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    train_wall = time.time() - t_train
    print(f"\n=== gradient step complete in {train_wall:.1f}s ===", flush=True)

    weight_delta_l2 = None
    weight_delta_max = None
    if args.check_weight_delta and pre_snapshot:
        sq_sum = 0.0
        max_abs = 0.0
        for n, p in model.named_parameters():
            if p.requires_grad and n in pre_snapshot:
                d = (p.detach().cpu() - pre_snapshot[n]).float()
                sq_sum += float((d * d).sum())
                max_abs = max(max_abs, float(d.abs().max()))
        weight_delta_l2 = sq_sum ** 0.5
        weight_delta_max = max_abs
        print(f"  weight delta L2 = {weight_delta_l2:.6f}, max|Δ| = {weight_delta_max:.6f}",
              flush=True)

    print(f"=== saving policy adapter → {args.out_dir} ===", flush=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    summary = {
        "bundle": str(args.bundle),
        "adapter_init": args.adapter_init,
        "reference_adapter": args.reference_adapter,
        "base_model": args.base_model,
        "lr": args.lr,
        "kl_beta": args.kl_beta,
        "n_groups_total": len(bundle),
        "n_groups_used": n_groups_used,
        "n_groups_zero_adv": n_groups_zero_adv,
        "n_groups_missing_task": n_groups_missing_task,
        "n_rollouts_used": n_rollouts_used,
        "n_rollouts_too_long": n_rollouts_too_long,
        "n_rollouts_no_assistant": n_rollouts_no_assistant,
        "n_assistant_tokens_total": n_assistant_tokens_total,
        "mean_loss": round(sum_loss / n_rollouts_used, 4),
        "mean_policy_loss": round(sum_pol_loss / n_rollouts_used, 4),
        "mean_kl_loss": round(sum_kl / n_rollouts_used, 4),
        "grad_norm_pre_clip": round(float(grad_norm), 4),
        "train_wall_seconds": round(train_wall, 1),
        "weight_delta_l2": weight_delta_l2,
        "weight_delta_max": weight_delta_max,
    }
    with (out_dir / "grpo_step_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
