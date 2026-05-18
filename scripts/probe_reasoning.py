"""Reasoning-preservation probe runner (spec §4 Layer 5).

Loads a model (HF id or local path), runs the 20-question reasoning probe
at probes/reasoning-preservation-20q.jsonl, and measures:

  - think_presence_rate   = fraction of responses containing <think>...</think>
  - think_token_length    = mean tokens between <think> and </think>
  - think_quality_score   = Claude-judge coherence score 0-5 (optional)

Output JSON shape (one record per checkpoint):
  {
    "model": "<id-or-path>",
    "lora_path": null | "<adapter-path>",
    "step": null | int,
    "n_probe": 20,
    "by_category": {
      "general-reasoning": {"think_presence_rate": ..., "think_token_length": ..., "think_quality_score": ...},
      "patent-irac":       {...},
      "patent-strategic":  {...}
    },
    "overall": {...},
    "raw_responses": [{"qid": ..., "response": ..., "has_think": bool, "think_n_tok": int|null}],
    "wall_seconds": float
  }

Usage:
  python scripts/probe_reasoning.py \
    --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
    --probe-set probes/reasoning-preservation-20q.jsonl \
    --output probes/baseline.json

  # Post-FT, optionally with an adapter:
  python scripts/probe_reasoning.py \
    --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
    --lora /work/runs/checkpoint-200 \
    --step 200 \
    --output probes/smoke-step200.json

  # Skip the LLM-judge step (faster, no Claude API call):
  python scripts/probe_reasoning.py ... --skip-judge
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_CACHE", "/root/.cache/huggingface/hub")
os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF id or local path of base model")
    p.add_argument("--lora", default=None, help="Optional peft adapter path to attach after base load")
    p.add_argument("--probe-set", default="probes/reasoning-preservation-20q.jsonl")
    p.add_argument("--output", required=True, help="JSON output path")
    p.add_argument("--step", type=int, default=None, help="Training step (for checkpoint probes)")
    p.add_argument("--max-new-tokens", type=int, default=1024,
                   help="Per [[feedback_reasoning_model_npredict]]: <1024 truncates <think> blocks")
    p.add_argument("--temperature", type=float, default=0.6, help="R1-Distill recommended")
    p.add_argument("--skip-judge", action="store_true",
                   help="Skip the Claude-judge coherence scoring (faster)")
    return p.parse_args()


def load_probe_set(path: Path) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_model(model_id: str, lora_path: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"loading tokenizer ({model_id})...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"loading base model bf16 on cuda:0...", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    print(f"  base loaded in {time.time()-t0:.1f}s", flush=True)

    if lora_path:
        from peft import PeftModel
        print(f"attaching LoRA adapter {lora_path}...", flush=True)
        model = PeftModel.from_pretrained(model, lora_path)

    model.eval()
    return tok, model


def generate(tok, model, question: str, max_new_tokens: int, temperature: float) -> str:
    import torch
    # R1-0528-Qwen3-8B supports chat template; minimal user-turn structure
    messages = [{"role": "user", "content": question}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tok.pad_token_id,
        )
    gen_ids = out[0, enc["input_ids"].shape[1]:]
    return tok.decode(gen_ids, skip_special_tokens=False)


def parse_think(response: str) -> tuple[bool, int | None, str]:
    """Returns (has_think_block, n_tokens_in_block_approx, think_text).

    Picks the longest pair when multiple `<think>...</think>` blocks are emitted —
    R1-distill models occasionally false-start with an empty `<think></think>`
    before opening a real one. The non-greedy `.*?` regex would match the empty
    pair first and undercount the real chain (caught on smoke-step-200 row 14).
    """
    matches = THINK_RE.findall(response)
    if not matches:
        return False, None, ""
    think_text = max(matches, key=len).strip()
    if not think_text:
        return True, 0, ""
    n_tok = max(1, len(think_text) // 4)
    return True, n_tok, think_text


def judge_coherence(think_text: str, question: str) -> float:
    """Score reasoning coherence 0-5 via Claude API. Returns float on success, -1 on failure."""
    try:
        import anthropic
    except ImportError:
        print("  (anthropic not installed — skip judge)", file=sys.stderr)
        return -1.0
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("  (ANTHROPIC_API_KEY not set — skip judge)", file=sys.stderr)
        return -1.0
    client = anthropic.Anthropic(api_key=api_key)
    sys_msg = (
        "You are scoring the coherence of a reasoning trace. Given a question and the "
        "reasoning chain that led to an answer, score the chain 0-5 on these criteria:\n"
        "  0 = incoherent, no chain of reasoning\n"
        "  1 = fragmented thoughts, no clear progression\n"
        "  2 = some structure but major gaps or errors\n"
        "  3 = adequate reasoning, mostly correct steps\n"
        "  4 = clear, well-structured reasoning with minor issues\n"
        "  5 = excellent step-by-step reasoning, no obvious errors\n"
        "Respond with ONLY a single integer 0-5 — no explanation."
    )
    user_msg = f"Question:\n{question}\n\nReasoning chain:\n{think_text[:4000]}"
    try:
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            system=sys_msg,
            messages=[{"role": "user", "content": user_msg}],
        )
        txt = resp.content[0].text.strip()
        m = re.search(r"[0-5]", txt)
        return float(m.group(0)) if m else -1.0
    except Exception as exc:
        print(f"  (judge call failed: {exc})", file=sys.stderr)
        return -1.0


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"think_presence_rate": 0.0, "think_token_length": 0.0, "think_quality_score": 0.0, "n": 0}
    has = [r for r in rows if r["has_think"]]
    presence = len(has) / len(rows)
    mean_len = sum(r["think_n_tok"] for r in has) / max(1, len(has))
    judged = [r for r in rows if r.get("think_quality") is not None and r["think_quality"] >= 0]
    mean_quality = sum(r["think_quality"] for r in judged) / max(1, len(judged))
    return {
        "think_presence_rate": round(presence, 4),
        "think_token_length": round(mean_len, 1),
        "think_quality_score": round(mean_quality, 2) if judged else None,
        "n": len(rows),
        "n_judged": len(judged),
    }


def main() -> int:
    args = parse_args()
    probe_set = load_probe_set(Path(args.probe_set))
    print(f"loaded {len(probe_set)} probe rows from {args.probe_set}", flush=True)

    tok, model = load_model(args.model, args.lora)

    rows = []
    t_start = time.time()
    for i, q in enumerate(probe_set, 1):
        print(f"[{i:2d}/{len(probe_set)}] qid={q['qid']} cat={q['category']}", flush=True)
        t0 = time.time()
        response = generate(tok, model, q["question"], args.max_new_tokens, args.temperature)
        has_think, n_tok, think_text = parse_think(response)
        quality = None
        if has_think and not args.skip_judge:
            quality = judge_coherence(think_text, q["question"])
        rows.append({
            "qid": q["qid"],
            "category": q["category"],
            "response": response,
            "has_think": has_think,
            "think_n_tok": n_tok,
            "think_quality": quality,
            "wall_seconds": round(time.time() - t0, 2),
        })
        print(f"     wall={time.time()-t0:.1f}s  has_think={has_think}  n_tok={n_tok}  q={quality}", flush=True)

    overall = summarize(rows)
    by_category = {}
    for cat in sorted({r["category"] for r in rows}):
        by_category[cat] = summarize([r for r in rows if r["category"] == cat])

    out = {
        "model": args.model,
        "lora_path": args.lora,
        "step": args.step,
        "n_probe": len(rows),
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "skip_judge": args.skip_judge,
        "overall": overall,
        "by_category": by_category,
        "raw_responses": rows,
        "wall_seconds": round(time.time() - t_start, 1),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote {args.output}")
    print(f"  overall: {overall}")
    print(f"  total wall: {out['wall_seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
