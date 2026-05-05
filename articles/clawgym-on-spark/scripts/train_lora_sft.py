"""LoRA SFT for clawgym agent — Phase 4.

Trains a LoRA adapter on the PASS+near-miss SFT records produced from
the Phase 3 baseline rollout. Base model = Qwen 2.5 7B Instruct (chosen
because the Spark NIM Llama is FP8-quantized via TRT-LLM ModelOpt and
not loadable by HF transformers without a re-quant step; Qwen is already
cached bf16 in the tllm-build container).

Loss is computed only on assistant tokens — system/user spans are masked
to -100 — using a prefix-based mask. Qwen's chat template doesn't carry
a `{% generation %}` annotation so we can't use the built-in
`return_assistant_tokens_mask`; the prefix-walk is what works.

Typical run on Spark (GB10, 128 GB unified):
    ~10 effective steps (42 records, batch 1, grad_accum 4)
    ~3 min wall, peak ~30 GB GPU memory

Usage (inside tllm-build container):
    python3 train_lora_sft.py \\
        --records /work/clawgym-sft/sft-records-and-near.jsonl \\
        --out-dir /work/clawgym-sft/adapter-v1/ \\
        --base-model Qwen/Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType


def build_input_and_labels(record: dict, tokenizer, max_length: int = 2048):
    """Tokenize one SFT record and produce (input_ids, labels).

    Labels are -100 everywhere except for assistant-turn tokens. We
    achieve this with a prefix walk: tokenize messages 1..i for each
    i and diff the lengths to know exactly where each message lives.
    """
    msgs = [{"role": "system", "content": record["system"]}] + record["messages"]
    prev_len = 0
    label_mask: list[int] = []  # 1 = supervised, 0 = ignored

    for i, msg in enumerate(msgs):
        partial = tokenizer.apply_chat_template(
            msgs[: i + 1],
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
        )
        cur_len = len(partial["input_ids"])
        new_tokens = cur_len - prev_len
        if msg["role"] == "assistant":
            label_mask.extend([1] * new_tokens)
        else:
            label_mask.extend([0] * new_tokens)
        prev_len = cur_len

    full = tokenizer.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=False, return_dict=True
    )
    input_ids = full["input_ids"]
    # Pad mask to full length (chat template may add trailing tokens we missed)
    while len(label_mask) < len(input_ids):
        label_mask.append(0)
    label_mask = label_mask[: len(input_ids)]

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        label_mask = label_mask[:max_length]

    labels = [tid if m else -100 for tid, m in zip(input_ids, label_mask)]
    return input_ids, labels


class SFTDataset(Dataset):
    def __init__(self, records: list[dict], tokenizer, max_length: int = 2048):
        self.examples = []
        for r in records:
            ids, labels = build_input_and_labels(r, tokenizer, max_length)
            self.examples.append({
                "input_ids": ids,
                "labels": labels,
                "n_supervised": sum(1 for l in labels if l != -100),
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def pad_collate(batch, pad_token_id: int):
    """Right-pad to longest in batch. labels pad to -100."""
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids = []
    attention_mask = []
    labels = []
    for b in batch:
        n = len(b["input_ids"])
        pad = max_len - n
        input_ids.append(b["input_ids"] + [pad_token_id] * pad)
        attention_mask.append([1] * n + [0] * pad)
        labels.append(b["labels"] + [-100] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True, help="JSONL of SFT records")
    ap.add_argument("--out-dir", required=True, help="adapter output dir")
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== loading tokenizer + model: {args.base_model} ===", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.gradient_checkpointing_enable()
    if hasattr(model, "config"):
        model.config.use_cache = False
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    # LoRA
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Data
    print(f"=== loading {args.records} ===", flush=True)
    records = []
    with open(args.records) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"  {len(records)} records", flush=True)

    dataset = SFTDataset(records, tokenizer, max_length=args.max_length)
    n_supervised = sum(ex["n_supervised"] for ex in dataset.examples)
    print(f"  total supervised tokens: {n_supervised} (mean {n_supervised/len(dataset):.0f}/record)", flush=True)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: pad_collate(b, tokenizer.pad_token_id),
    )

    n_micro = len(loader)
    n_steps_per_epoch = math.ceil(n_micro / args.grad_accum)
    total_steps = n_steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    print(f"=== training plan ===", flush=True)
    print(f"  micro-batches/epoch: {n_micro}", flush=True)
    print(f"  optimizer steps/epoch: {n_steps_per_epoch}  (grad_accum={args.grad_accum})", flush=True)
    print(f"  total steps: {total_steps}, warmup: {warmup_steps}", flush=True)
    print(f"  lr: {args.lr}, weight_decay: {args.weight_decay}", flush=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    model.train()
    step = 0
    t_train = time.time()
    losses = []
    for epoch in range(args.epochs):
        accum_loss = 0.0
        accum_count = 0
        for i, batch in enumerate(loader):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / args.grad_accum
            loss.backward()
            accum_loss += out.loss.item()
            accum_count += 1

            if (i + 1) % args.grad_accum == 0 or (i + 1) == n_micro:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                avg_loss = accum_loss / accum_count
                losses.append(avg_loss)
                lr_now = scheduler.get_last_lr()[0]
                el = time.time() - t_train
                print(f"  step {step:>3d}/{total_steps}  loss {avg_loss:.4f}  lr {lr_now:.2e}  elapsed {el:.1f}s", flush=True)
                accum_loss = 0.0
                accum_count = 0

    train_wall = time.time() - t_train
    print(f"=== training complete in {train_wall:.1f}s ===", flush=True)

    # Save adapter
    print(f"=== saving adapter → {out_dir} ===", flush=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    summary = {
        "base_model": args.base_model,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lr": args.lr,
        "epochs": args.epochs,
        "n_records": len(records),
        "n_supervised_tokens": n_supervised,
        "total_optimizer_steps": step,
        "train_wall_seconds": round(train_wall, 1),
        "loss_curve": [round(l, 4) for l in losses],
        "loss_first": round(losses[0], 4) if losses else None,
        "loss_last": round(losses[-1], 4) if losses else None,
    }
    with open(out_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"=== summary → {out_dir}/training_summary.json ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
