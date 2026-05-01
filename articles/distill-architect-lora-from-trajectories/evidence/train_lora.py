"""LoRA fine-tune of Qwen2.5-3B-Instruct on the A4 trajectory corpus.

Mirrors articles/lora-on-your-own-qa-pairs/evidence/train_lora.py but consumes
chat-format messages directly (system / user / assistant) and supervises only
the assistant span.

Runs inside nvcr.io/nvidia/tritonserver:25.12-trtllm-python-py3 with these
mounts:
    /work/base    <-  /home/nvidia/lora-work/base               (Qwen2.5-3B-Instruct)
    /work/train.jsonl, /work/test.jsonl, /work/adapter, /work/runs   <-  scratch
"""
import json
import os
import time
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

BASE = "/work/base"
OUT = "/work/adapter"
TRAIN_JSONL = "/work/train.jsonl"
EVAL_JSONL = "/work/test.jsonl"

os.makedirs(OUT, exist_ok=True)

print(f"torch {torch.__version__}  cuda {torch.cuda.is_available()}  "
      f"dev {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

tok = AutoTokenizer.from_pretrained(BASE)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

print("loading base model (bf16)...")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    BASE,
    dtype=torch.bfloat16,
    device_map="cuda:0",
    attn_implementation="sdpa",
)
print(f"  loaded in {time.time()-t0:.1f}s, "
      f"params={sum(p.numel() for p in model.parameters())/1e9:.2f}B")

lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
)
model = get_peft_model(model, lora_cfg)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"trainable: {trainable/1e6:.2f}M / total: {total/1e9:.2f}B  "
      f"({100*trainable/total:.3f}%)")


def load_jsonl(p):
    return [json.loads(l) for l in open(p)]


train_raw = load_jsonl(TRAIN_JSONL)
eval_raw = load_jsonl(EVAL_JSONL)
print(f"train={len(train_raw)}  eval={len(eval_raw)}")


MAX_LEN = 2048  # user prompts are ~600 tokens; chat template + assistant push us past 1024


def to_chat(example):
    """Tokenize the full chat and mask labels on prompt tokens (system+user)."""
    msgs = example["messages"]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    prompt_only = tok.apply_chat_template(msgs[:2], tokenize=False, add_generation_prompt=True)
    enc = tok(text, truncation=True, max_length=MAX_LEN, padding=False)
    prompt_ids = tok(prompt_only, add_special_tokens=False)["input_ids"]
    labels = list(enc["input_ids"])
    n_prompt = min(len(prompt_ids), len(labels))
    # If truncation eliminated the entire assistant span, this row is unusable.
    if n_prompt >= len(labels):
        # Keep at least one supervised token by trimming the prompt mask back by 1.
        n_prompt = max(0, len(labels) - 1)
    for i in range(n_prompt):
        labels[i] = -100
    enc["labels"] = labels
    return enc


print("tokenizing...")
keep_cols = ["messages"]
drop_cols = ["iter", "decision", "val_bpb", "improvement_frac"]
train_ds = Dataset.from_list(train_raw).map(to_chat, remove_columns=keep_cols + drop_cols)
eval_ds = Dataset.from_list(eval_raw).map(to_chat, remove_columns=keep_cols + drop_cols)

# Diagnostic: confirm every row has at least one supervised (non-masked) token.
def supervised_count(row):
    return sum(1 for x in row["labels"] if x != -100)


tr_sup = [supervised_count(r) for r in train_ds]
ev_sup = [supervised_count(r) for r in eval_ds]
print(f"train supervised tokens — min={min(tr_sup)} median={sorted(tr_sup)[len(tr_sup)//2]} max={max(tr_sup)}")
print(f"eval  supervised tokens — min={min(ev_sup)} median={sorted(ev_sup)[len(ev_sup)//2]} max={max(ev_sup)}")
assert min(tr_sup) > 0 and min(ev_sup) > 0, "some rows have zero supervised tokens — increase MAX_LEN"


class PadCollator:
    def __init__(self, tokenizer):
        self.pad = tokenizer.pad_token_id

    def __call__(self, batch):
        max_len = max(len(b["input_ids"]) for b in batch)
        out = {"input_ids": [], "attention_mask": [], "labels": []}
        for b in batch:
            pad_len = max_len - len(b["input_ids"])
            out["input_ids"].append(b["input_ids"] + [self.pad] * pad_len)
            out["attention_mask"].append(b["attention_mask"] + [0] * pad_len)
            out["labels"].append(b["labels"] + [-100] * pad_len)
        return {k: torch.tensor(v) for k, v in out.items()}


args = TrainingArguments(
    output_dir="/work/runs",
    num_train_epochs=5,                       # 42 examples — needs more passes
    per_device_train_batch_size=2,            # prompts ~2K chars, leave headroom
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,            # effective batch 8
    learning_rate=3e-4,                       # HANDOFF spec
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,                         # 5 epochs × 21 = 105 steps; 10 warmup
    weight_decay=0.01,
    logging_steps=2,
    save_strategy="no",
    eval_strategy="epoch",
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    report_to=[],
    seed=42,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=PadCollator(tok),
)

print("starting training...")
t0 = time.time()
trainer.train()
dt = time.time() - t0
print(f"training finished in {dt/60:.1f} min")

print(f"saving adapter to {OUT}")
trainer.model.save_pretrained(OUT)
tok.save_pretrained(OUT)
print("done.")
