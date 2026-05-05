"""Smoke-test the LoRA adapter from Phase 4 SFT.

Loads the base model + adapter, runs one inference per persona against
a held-out task drawn from the synth corpus (avoiding the 17 PASS +
25 near-miss training set). Prints the generated agent response so we
can eyeball whether it produces well-formed bash blocks.

Usage (inside tllm-build container):
    python3 smoke_eval_adapter.py \\
        --adapter /work/clawgym-sft/adapter-v1 \\
        --tasks /work/clawgym-sft/tasks-200.jsonl \\
        --training-records /work/clawgym-sft/sft-records-and-near.jsonl \\
        --n-per-persona 1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

SYSTEM_PROMPT = """You are a file-management agent operating inside a sandboxed Linux
workspace. You have access to standard POSIX shell tools: ls, cat,
head, tail, mv, cp, rm, mkdir, sed, grep, find, wc, echo, touch.

Each turn, respond with EXACTLY ONE shell command wrapped in a
```bash code block```. After the command runs you will see its
stdout, stderr, and exit code in the next turn.

When you believe the task is complete, respond with the literal
token TASK_COMPLETE on a line by itself (no code block).

Rules:
- One command per turn. No semicolons or && chains; use multiple turns.
- Always use relative paths from the workspace root (which is your CWD).
- Use `sed -i` for in-place edits.
- Use `mkdir -p` to create parent directories.
- Do not invoke network commands (curl, wget, apt, pip, npm).
- Do not invoke editors (vi, nano, emacs)."""


def render_files(paths):
    if not paths:
        return "  (empty)"
    return "\n".join(f"  {p}" for p in paths)


def workspace_listing(task):
    paths = sorted(f["path"] for f in task["workspace_seed"]["files"])
    dirs = set()
    for p in paths:
        parts = p.split("/")
        for i in range(1, len(parts)):
            dirs.add("/".join(parts[:i]))
    return render_files(sorted(set(paths) | dirs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="path to adapter dir")
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--tasks", required=True, help="full task corpus JSONL")
    ap.add_argument("--training-records", required=True, help="training set (to exclude)")
    ap.add_argument("--n-per-persona", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    print(f"=== loading base + adapter ===", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    # Held-out: any task NOT in the training set
    train_ids = set()
    with open(args.training_records) as f:
        for line in f:
            train_ids.add(json.loads(line)["task_id"])
    print(f"  excluding {len(train_ids)} training task IDs", flush=True)

    tasks_by_persona = defaultdict(list)
    with open(args.tasks) as f:
        for line in f:
            t = json.loads(line)
            if t["task_id"] in train_ids:
                continue
            tasks_by_persona[t["persona"]["role"]].append(t)
    print(f"  held-out personas: {len(tasks_by_persona)}", flush=True)
    for p, ts in sorted(tasks_by_persona.items()):
        print(f"    {p}: {len(ts)} tasks", flush=True)

    import random
    rng = random.Random(args.seed)

    print(f"\n=== generating {args.n_per_persona} per persona ===\n", flush=True)
    well_formed = 0
    total = 0
    for persona, tasks in sorted(tasks_by_persona.items()):
        for t in rng.sample(tasks, min(args.n_per_persona, len(tasks))):
            user_content = (
                f"TASK\n{t['intent']}\n\nWORKSPACE FILES (CWD = workspace root)\n"
                f"{workspace_listing(t)}\n\nBegin. Respond with one shell command in a ```bash``` block."
            )
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
            enc = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=True
            )
            input_ids = torch.tensor([enc["input_ids"]], device=model.device)
            attn = torch.tensor([enc["attention_mask"]], device=model.device)

            t1 = time.time()
            with torch.no_grad():
                out = model.generate(
                    input_ids=input_ids,
                    attention_mask=attn,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            new_ids = out[0, input_ids.shape[1]:]
            response = tokenizer.decode(new_ids, skip_special_tokens=True)
            wall = time.time() - t1

            total += 1
            has_bash = "```bash" in response or "```sh" in response
            has_done = "TASK_COMPLETE" in response
            well = has_bash or has_done
            if well:
                well_formed += 1

            print(f"--- {t['task_id']} ({wall:.1f}s, {len(new_ids)} toks) "
                  f"{'✓' if well else '✗'} bash={has_bash} done={has_done} ---", flush=True)
            print(f"INTENT: {t['intent'][:200]}", flush=True)
            print(f"RESPONSE:\n{response[:500]}", flush=True)
            print(flush=True)

    print(f"=== well-formed: {well_formed}/{total} = {100*well_formed/total:.0f}% ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
