"""LoRA fine-tune of Qwen2.5-3B-Instruct on nvidia-learn Q&A pairs.

Runs inside nvcr.io/nvidia/tritonserver:25.12-trtllm-python-py3.
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
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

BASE = "/work/base"
OUT = "/work/adapter"
TRAIN_JSONL = "/work/train.jsonl"
EVAL_JSONL = "/work/eval.jsonl"

os.makedirs(OUT, exist_ok=True)

print(f"torch {torch.__version__}  cuda {torch.cuda.is_available()}  dev {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

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
print(f"  loaded in {time.time()-t0:.1f}s, params={sum(p.numel() for p in model.parameters())/1e9:.2f}B")

# PEFT / LoRA
lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
)
model = get_peft_model(model, lora_cfg)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"trainable: {trainable/1e6:.2f}M / total: {total/1e9:.2f}B  ({100*trainable/total:.3f}%)")


# --- Dataset ---
def load_jsonl(p):
    return [json.loads(l) for l in open(p)]


train_raw = load_jsonl(TRAIN_JSONL)
eval_raw = load_jsonl(EVAL_JSONL)
print(f"train={len(train_raw)}  eval={len(eval_raw)}")

SYS = (
    "You are an assistant that answers questions about the nvidia-learn DGX Spark project "
    "(articles by Manav Sehgal on running AI locally on the NVIDIA DGX Spark). "
    "Answer concisely, grounded in the project's own content."
)


def to_chat(example):
    """Apply Qwen chat template; loss masked to assistant response only via attention."""
    msgs = [
        {"role": "system", "content": SYS},
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["answer"]},
    ]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    # build labels that only supervise the assistant response
    prompt_only = tok.apply_chat_template(msgs[:2], tokenize=False, add_generation_prompt=True)
    enc = tok(text, truncation=True, max_length=1024, padding=False)
    prompt_ids = tok(prompt_only, add_special_tokens=False)["input_ids"]
    labels = list(enc["input_ids"])
    n_prompt = min(len(prompt_ids), len(labels))
    for i in range(n_prompt):
        labels[i] = -100  # mask prompt tokens
    enc["labels"] = labels
    return enc


print("tokenizing...")
train_ds = Dataset.from_list(train_raw).map(to_chat, remove_columns=["question", "answer", "source", "chunk"])
eval_ds = Dataset.from_list(eval_raw).map(to_chat, remove_columns=["question", "answer", "source", "chunk"])


# --- Data collator that pads and keeps labels ---
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
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=0.01,
    logging_steps=5,
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
