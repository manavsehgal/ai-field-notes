"""
Baseline pretrain throughput sweep on a single GB10 (DGX Spark).

The model shape and training loop come straight from
articles/nemo-framework-on-spark/evidence/nemo_train.py — the A1 matched
NeMo run. The only thing that varies here is the throughput envelope:

  micro batch  ∈ {2, 4, 8, 16}
  seq length   ∈ {1024, 2048}
  precision    ∈ {bf16, fp8}

→ 16 configurations. Each config runs 30 steps; the first 5 are warmup
and excluded from the mean. OOMs are caught and recorded; the sweep
continues.

Writes evidence/sweep_results.json. Run inside nvcr.io/nvidia/nemo:26.04.00
(its container venv has TransformerEngine 2.14 with FP8 ready to go).
"""
from __future__ import annotations

import gc
import json
import math
import os
import time
import traceback
from dataclasses import dataclass, asdict
from typing import Any

import torch
import torch.nn.functional as F
import transformer_engine
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling, Format
from megatron.core import parallel_state, tensor_parallel
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.transformer import TransformerConfig


# --- model shape (frozen — same 354M GPT as A1) -----------------------------
VOCAB_SIZE = 50_257
N_LAYER = 24
N_HEAD = 16
D_MODEL = 1024
D_FF = 4096
LR = 3e-4
GRAD_CLIP = 1.0
SEED = 0
STEPS = 30
WARMUP_STEPS = 5  # excluded from mean
SEED_TIME = 0


@dataclass
class RunCfg:
    batch_size: int
    seq_len: int
    precision: str  # "bf16" | "fp8"


def init_distributed_single_gpu_once() -> None:
    if torch.distributed.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    torch.distributed.init_process_group(backend="nccl", world_size=1, rank=0)
    torch.cuda.set_device(0)
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )
    tensor_parallel.model_parallel_cuda_manual_seed(SEED)


def build_model(seq_len: int) -> GPTModel:
    tcfg = TransformerConfig(
        num_layers=N_LAYER,
        hidden_size=D_MODEL,
        num_attention_heads=N_HEAD,
        ffn_hidden_size=D_FF,
        bf16=True,
        params_dtype=torch.bfloat16,
        attention_softmax_in_fp32=True,
        pipeline_dtype=torch.bfloat16,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        sequence_parallel=False,
        use_cpu_initialization=False,
        gradient_accumulation_fusion=False,
        masked_softmax_fusion=True,
        bias_activation_fusion=False,
        bias_dropout_fusion=False,
        persist_layer_norm=False,
        normalization="LayerNorm",
        activation_func=F.gelu,
        add_bias_linear=False,
    )
    spec = get_gpt_layer_with_transformer_engine_spec()
    model = GPTModel(
        config=tcfg,
        transformer_layer_spec=spec,
        vocab_size=VOCAB_SIZE,
        max_sequence_length=seq_len,
        pre_process=True,
        post_process=True,
        parallel_output=False,
        share_embeddings_and_output_weights=True,
        position_embedding_type="learned_absolute",
    )
    return model.cuda()


def lr_at(step: int) -> float:
    if step < WARMUP_STEPS:
        return LR * (step + 1) / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, STEPS - WARMUP_STEPS)
    return LR * 0.5 * (1 + math.cos(math.pi * progress))


def random_batch(batch: int, seq: int, device: torch.device):
    g = torch.Generator(device=device).manual_seed(SEED_TIME)
    x = torch.randint(0, VOCAB_SIZE, (batch, seq), device=device, generator=g)
    y = torch.roll(x, -1, dims=1)
    return x, y


def make_attention_mask(seq: int, device: torch.device) -> torch.Tensor:
    mask = torch.tril(torch.ones((seq, seq), device=device, dtype=torch.bool))
    return ~mask.unsqueeze(0).unsqueeze(0)


def run_one(cfg: RunCfg) -> dict[str, Any]:
    """Run one configuration. Returns metrics dict (or {oom: True})."""
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    try:
        model = build_model(cfg.seq_len)
        n_params = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95),
                                fused=True)
        attn_mask = make_attention_mask(cfg.seq_len, device)
        pos_ids = torch.arange(cfg.seq_len, device=device).unsqueeze(0).expand(
            cfg.batch_size, -1)

        fp8_recipe = None
        if cfg.precision == "fp8":
            fp8_recipe = DelayedScaling(
                fp8_format=Format.HYBRID,  # E4M3 fwd, E5M2 bwd
                amax_history_len=16,
                amax_compute_algo="max",
            )

        step_times: list[float] = []
        losses: list[float] = []
        model.train()

        for step in range(STEPS):
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            x, y = random_batch(cfg.batch_size, cfg.seq_len, device)

            torch.cuda.synchronize(device)
            t0 = time.perf_counter()

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                if fp8_recipe is not None:
                    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                        logits = model(x, position_ids=pos_ids,
                                       attention_mask=attn_mask, labels=None)
                else:
                    logits = model(x, position_ids=pos_ids,
                                   attention_mask=attn_mask, labels=None)
            loss = F.cross_entropy(
                logits.float().view(-1, logits.size(-1)), y.view(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

            torch.cuda.synchronize(device)
            dt = time.perf_counter() - t0
            step_times.append(dt)
            losses.append(loss.item())

        warm = step_times[WARMUP_STEPS:]
        mean_dt = sum(warm) / len(warm)
        tok_per_step = cfg.batch_size * cfg.seq_len
        peak_alloc = torch.cuda.max_memory_allocated(device) / (1024**3)
        peak_reserved = torch.cuda.max_memory_reserved(device) / (1024**3)

        return {
            "ok": True,
            "params_m": round(n_params / 1e6, 1),
            "mean_step_ms": round(mean_dt * 1e3, 2),
            "tokens_per_s": round(tok_per_step / mean_dt, 1),
            "peak_gpu_mem_gib": round(peak_alloc, 2),
            "peak_gpu_reserved_gib": round(peak_reserved, 2),
            "loss_first": round(losses[0], 3),
            "loss_last": round(losses[-1], 3),
        }

    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        return {"ok": False, "oom": True, "error": str(e)[:200]}
    except Exception as e:
        torch.cuda.empty_cache()
        return {
            "ok": False,
            "oom": False,
            "error": str(e)[:200],
            "trace_tail": traceback.format_exc()[-400:],
        }
    finally:
        # Free model + optimizer between runs.
        for name in ("opt", "model", "logits", "loss", "x", "y", "attn_mask",
                     "pos_ids"):
            if name in dir():
                pass
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    init_distributed_single_gpu_once()
    device = torch.device("cuda")

    print(f"sweep on {torch.cuda.get_device_name(0)}  torch={torch.__version__}  "
          f"te={transformer_engine.__version__}")

    configs: list[RunCfg] = []
    for precision in ("bf16", "fp8"):
        for seq in (1024, 2048):
            for batch in (2, 4, 8, 16):
                configs.append(RunCfg(batch_size=batch, seq_len=seq,
                                      precision=precision))

    results = []
    t_sweep = time.perf_counter()
    for i, c in enumerate(configs, 1):
        label = f"[{i:2d}/{len(configs)}] batch={c.batch_size:2d} seq={c.seq_len} prec={c.precision}"
        t0 = time.perf_counter()
        m = run_one(c)
        dt = time.perf_counter() - t0
        m["wall_s"] = round(dt, 1)
        rec = {**asdict(c), **m}
        results.append(rec)
        if m.get("ok"):
            print(f"{label}  → tok/s={m['tokens_per_s']:>8.0f}  "
                  f"step={m['mean_step_ms']:>6.1f}ms  "
                  f"peak={m['peak_gpu_mem_gib']:>5.2f}GiB  "
                  f"loss {m['loss_first']:6.2f}→{m['loss_last']:6.2f}  "
                  f"({dt:.0f}s)")
        else:
            tag = "OOM" if m.get("oom") else "ERR"
            print(f"{label}  → {tag}: {m.get('error','?')[:100]}  ({dt:.0f}s)")

    sweep_total = time.perf_counter() - t_sweep
    print(f"\nsweep done in {sweep_total/60:.1f} min")

    # Pick winners.
    ok = [r for r in results if r.get("ok")]
    if ok:
        winner_throughput = max(ok, key=lambda r: r["tokens_per_s"])
        winner_efficiency = min(
            ok, key=lambda r: r["peak_gpu_mem_gib"] / r["tokens_per_s"])
        print(f"\npeak throughput  : {winner_throughput['tokens_per_s']:>8.0f} tok/s "
              f"@ batch={winner_throughput['batch_size']} seq={winner_throughput['seq_len']} "
              f"prec={winner_throughput['precision']}")
        print(f"best mem/throughput: batch={winner_efficiency['batch_size']} "
              f"seq={winner_efficiency['seq_len']} prec={winner_efficiency['precision']} "
              f"({winner_efficiency['peak_gpu_mem_gib']:.2f} GiB / "
              f"{winner_efficiency['tokens_per_s']:.0f} tok/s)")

    out = os.path.join(os.path.dirname(__file__), "sweep_results.json")
    with open(out, "w") as f:
        json.dump({
            "device": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "te_version": transformer_engine.__version__,
            "model": {
                "vocab_size": VOCAB_SIZE,
                "n_layer": N_LAYER,
                "n_head": N_HEAD,
                "d_model": D_MODEL,
                "d_ff": D_FF,
            },
            "sweep_steps": STEPS,
            "warmup_steps_excluded_from_mean": WARMUP_STEPS,
            "sweep_wall_s": round(sweep_total, 1),
            "results": results,
        }, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
