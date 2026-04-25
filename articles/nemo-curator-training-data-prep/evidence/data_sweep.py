"""
Stage 3 of A3 — measure the throughput envelope when batches come from
the prepared corpus instead of `torch.randint`.

Loads packed.int32.npy as an mmap (cold-cache start), then runs the
exact same Megatron-Core training loop as A2's sweep.py — same model
shape, same configs, same step count. The only delta vs A2 is:

    A2:  x = torch.randint(0, vocab, (batch, seq), device=device)
    A3:  x = torch.from_numpy(packed[start:start+batch*seq].reshape(...))
              .to(device, non_blocking=True)

Results land in evidence/data_sweep_results.json. Compared head-to-head
with articles/baseline-training-loop-on-spark/evidence/sweep_results.json
in the article body.

Sweep: 8 configurations (skip the largest seq=2048/batch=16 fp8 corner
which adds 80s and gives no new signal — A2's seq=1024 envelope is
already definitive).

Run inside nemo-curator-spark:1.1 (its venv has the same torch +
megatron-core versions as the base nemo container).
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

import numpy as np
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

EVIDENCE = os.path.dirname(os.path.abspath(__file__))
PACKED_PATH = os.path.join(EVIDENCE, "packed.int32.npy")
META_PATH = os.path.join(EVIDENCE, "packed.meta.json")
RESULTS_PATH = os.path.join(EVIDENCE, "data_sweep_results.json")

# --- model shape (frozen — same 354M GPT as A1/A2) --------------------------
N_LAYER = 24
N_HEAD = 16
D_MODEL = 1024
D_FF = 4096
LR = 3e-4
GRAD_CLIP = 1.0
SEED = 0
STEPS = 30
WARMUP_STEPS = 5

# Configs: subset of A2's 16 — focus on the regime A2 said matters.
CONFIGS = [
    # (batch_size, seq_len, precision)
    (4,  1024, "bf16"),
    (8,  1024, "bf16"),
    (16, 1024, "bf16"),
    (4,  2048, "bf16"),
    (4,  1024, "fp8"),
    (8,  1024, "fp8"),
    (16, 1024, "fp8"),  # A2's overall peak config
    (4,  2048, "fp8"),
]


@dataclass
class RunCfg:
    batch_size: int
    seq_len: int
    precision: str


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
    parallel_state.initialize_model_parallel(1, 1)
    tensor_parallel.model_parallel_cuda_manual_seed(SEED)


def build_model(seq_len: int, vocab_size: int) -> GPTModel:
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
        vocab_size=vocab_size,
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


def make_attention_mask(seq: int, device: torch.device) -> torch.Tensor:
    mask = torch.tril(torch.ones((seq, seq), device=device, dtype=torch.bool))
    return ~mask.unsqueeze(0).unsqueeze(0)


class CorpusBatcher:
    """Walk the packed corpus sequentially. Each batch is `batch * seq`
    contiguous tokens; we pin-page once at construction so transfer cost
    is a measured part of the step time, not a startup oddity.

    Wraps around if the corpus runs out — but at 109M tokens and 30
    steps × max(16 * 2048) tokens-per-step, no config gets close."""

    def __init__(self, packed_path: str):
        self._packed = np.load(packed_path, mmap_mode="r")
        self._cursor = 0

    @property
    def total_tokens(self) -> int:
        return int(self._packed.shape[0])

    def next_batch(self, batch: int, seq: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Need batch * (seq + 1) so y can be a +1 shift of x.
        need = batch * (seq + 1)
        if self._cursor + need > self._packed.shape[0]:
            self._cursor = 0
        view = np.asarray(
            self._packed[self._cursor:self._cursor + need], dtype=np.int64
        )
        self._cursor += batch * seq  # advance non-overlappingly per batch
        # Reshape: x = first seq tokens of each row, y = last seq tokens.
        view = view.reshape(batch, seq + 1)
        x = torch.from_numpy(view[:, :-1]).to(
            "cuda", dtype=torch.long, non_blocking=True)
        y = torch.from_numpy(view[:, 1:]).to(
            "cuda", dtype=torch.long, non_blocking=True)
        return x, y


def run_one(cfg: RunCfg, batcher: CorpusBatcher, vocab_size: int) -> dict[str, Any]:
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    try:
        model = build_model(cfg.seq_len, vocab_size)
        n_params = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95),
                                fused=True)
        attn_mask = make_attention_mask(cfg.seq_len, device)
        pos_ids = torch.arange(cfg.seq_len, device=device).unsqueeze(0).expand(
            cfg.batch_size, -1)

        fp8_recipe = None
        if cfg.precision == "fp8":
            fp8_recipe = DelayedScaling(
                fp8_format=Format.HYBRID,
                amax_history_len=16,
                amax_compute_algo="max",
            )

        step_times: list[float] = []
        data_times: list[float] = []
        losses: list[float] = []
        model.train()

        for step in range(STEPS):
            for g in opt.param_groups:
                g["lr"] = lr_at(step)

            torch.cuda.synchronize(device)
            t_data0 = time.perf_counter()
            x, y = batcher.next_batch(cfg.batch_size, cfg.seq_len)
            torch.cuda.synchronize(device)
            data_dt = time.perf_counter() - t_data0

            t_step0 = time.perf_counter()
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
            step_dt = time.perf_counter() - t_step0

            step_times.append(step_dt)
            data_times.append(data_dt)
            losses.append(loss.item())

        warm_step = step_times[WARMUP_STEPS:]
        warm_data = data_times[WARMUP_STEPS:]
        mean_step = sum(warm_step) / len(warm_step)
        mean_data = sum(warm_data) / len(warm_data)
        # Combined wall = data + step (sequential per iteration here).
        mean_total = mean_step + mean_data
        tok_per_step = cfg.batch_size * cfg.seq_len
        peak_alloc = torch.cuda.max_memory_allocated(device) / (1024**3)

        return {
            "ok": True,
            "params_m": round(n_params / 1e6, 1),
            "mean_step_ms": round(mean_step * 1e3, 2),
            "mean_data_ms": round(mean_data * 1e3, 2),
            "mean_total_ms": round(mean_total * 1e3, 2),
            "tokens_per_s_step_only": round(tok_per_step / mean_step, 1),
            "tokens_per_s_with_data": round(tok_per_step / mean_total, 1),
            "data_overhead_pct": round(mean_data / mean_step * 100, 2),
            "peak_gpu_mem_gib": round(peak_alloc, 2),
            "loss_first": round(losses[0], 3),
            "loss_last": round(losses[-1], 3),
        }

    except torch.cuda.OutOfMemoryError as e:
        return {"ok": False, "oom": True, "error": str(e)[:200]}
    except Exception as e:
        return {"ok": False, "oom": False, "error": str(e)[:200],
                "trace_tail": traceback.format_exc()[-400:]}
    finally:
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    init_distributed_single_gpu_once()
    with open(META_PATH) as f:
        meta = json.load(f)
    vocab_size = meta["vocab_size"] + 1  # gpt2 has 50,257 tokens, eot id = 50256
    print(f"sweep on {torch.cuda.get_device_name(0)}  torch={torch.__version__}  "
          f"te={transformer_engine.__version__}")
    print(f"corpus packed.int32.npy: {meta['total_tokens']:,} tokens, vocab={vocab_size}")

    batcher = CorpusBatcher(PACKED_PATH)

    results = []
    t_sweep = time.perf_counter()
    for i, (b, s, p) in enumerate(CONFIGS, 1):
        cfg = RunCfg(batch_size=b, seq_len=s, precision=p)
        label = f"[{i:2d}/{len(CONFIGS)}] batch={b:2d} seq={s} prec={p}"
        t0 = time.perf_counter()
        m = run_one(cfg, batcher, vocab_size)
        dt = time.perf_counter() - t0
        m["wall_s"] = round(dt, 1)
        rec = {**asdict(cfg), **m}
        results.append(rec)
        if m.get("ok"):
            print(f"{label}  → tok/s(step)={m['tokens_per_s_step_only']:>8.0f}  "
                  f"tok/s(+data)={m['tokens_per_s_with_data']:>8.0f}  "
                  f"data={m['mean_data_ms']:>5.2f}ms  "
                  f"step={m['mean_step_ms']:>6.1f}ms  "
                  f"data%={m['data_overhead_pct']:.2f}  "
                  f"peak={m['peak_gpu_mem_gib']:>5.2f}GiB  "
                  f"loss {m['loss_first']:6.2f}→{m['loss_last']:6.2f}  ({dt:.0f}s)")
        else:
            tag = "OOM" if m.get("oom") else "ERR"
            print(f"{label}  → {tag}: {m.get('error','?')[:100]}")

    sweep_total = time.perf_counter() - t_sweep
    print(f"\nsweep done in {sweep_total/60:.1f} min ({sweep_total:.1f}s)")

    out = {
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "te_version": transformer_engine.__version__,
        "corpus": meta,
        "model": {
            "n_layer": N_LAYER, "n_head": N_HEAD,
            "d_model": D_MODEL, "d_ff": D_FF,
            "vocab_size": vocab_size,
        },
        "sweep_steps": STEPS,
        "warmup_steps_excluded_from_mean": WARMUP_STEPS,
        "sweep_wall_s": round(sweep_total, 1),
        "results": results,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
