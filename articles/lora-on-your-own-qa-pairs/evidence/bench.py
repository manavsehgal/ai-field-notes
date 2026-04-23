"""Benchmark base vs base+adapter on the held-out eval.jsonl.

Generates answers from both; saves JSONL side-by-side predictions.
"""
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "/work/base"
ADAPTER = "/work/adapter"
EVAL = "/work/eval.jsonl"
OUT_BASE = "/work/preds_base.jsonl"
OUT_ADAP = "/work/preds_adapter.jsonl"
OUT_LAT = "/work/latency.json"

SYS = (
    "You are an assistant that answers questions about the nvidia-learn DGX Spark project "
    "(articles by Manav Sehgal on running AI locally on the NVIDIA DGX Spark). "
    "Answer concisely, grounded in the project's own content."
)


def fmt_prompt(tok, question):
    msgs = [
        {"role": "system", "content": SYS},
        {"role": "user", "content": question},
    ]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def generate_all(model, tok, items, label):
    preds = []
    tot_latency = []
    for it in items:
        prompt = fmt_prompt(tok, it["question"])
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=160,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                pad_token_id=tok.eos_token_id,
                use_cache=True,
            )
        dt = time.time() - t0
        tot_latency.append({
            "idx": it.get("idx", None),
            "wall_s": dt,
            "new_tokens": int(out.shape[1] - inputs["input_ids"].shape[1]),
        })
        resp = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        preds.append({
            "question": it["question"],
            "reference": it["answer"],
            "source": it["source"],
            "prediction": resp,
            "wall_s": dt,
            "new_tokens": int(out.shape[1] - inputs["input_ids"].shape[1]),
        })
    return preds, tot_latency


def main():
    print(f"torch {torch.__version__}  cuda {torch.cuda.is_available()}")
    eval_items = [json.loads(l) for l in open(EVAL)]
    for i, it in enumerate(eval_items):
        it["idx"] = i
    print(f"eval items: {len(eval_items)}")

    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # 1. Base
    print("loading BASE model...")
    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa"
    )
    base.eval()
    print(f"  loaded in {time.time()-t0:.1f}s")

    print("generating with BASE...")
    t0 = time.time()
    preds_base, lat_base = generate_all(base, tok, eval_items, "base")
    print(f"  done in {time.time()-t0:.1f}s")
    Path(OUT_BASE).write_text("\n".join(json.dumps(p) for p in preds_base) + "\n")

    # 2. Base + adapter
    print("attaching adapter (no reload)...")
    t0 = time.time()
    adapted = PeftModel.from_pretrained(base, ADAPTER)
    adapted.eval()
    print(f"  attached in {time.time()-t0:.1f}s")

    print("generating with BASE+ADAPTER...")
    t0 = time.time()
    preds_adap, lat_adap = generate_all(adapted, tok, eval_items, "adapter")
    print(f"  done in {time.time()-t0:.1f}s")
    Path(OUT_ADAP).write_text("\n".join(json.dumps(p) for p in preds_adap) + "\n")

    # summary
    import statistics
    lat_sum = {
        "base_mean_s": statistics.mean([x["wall_s"] for x in lat_base]),
        "base_p50_s": statistics.median([x["wall_s"] for x in lat_base]),
        "adapter_mean_s": statistics.mean([x["wall_s"] for x in lat_adap]),
        "adapter_p50_s": statistics.median([x["wall_s"] for x in lat_adap]),
        "base_mean_tokens": statistics.mean([x["new_tokens"] for x in lat_base]),
        "adapter_mean_tokens": statistics.mean([x["new_tokens"] for x in lat_adap]),
    }
    lat_sum["base_tps"] = lat_sum["base_mean_tokens"] / lat_sum["base_mean_s"]
    lat_sum["adapter_tps"] = lat_sum["adapter_mean_tokens"] / lat_sum["adapter_mean_s"]
    Path(OUT_LAT).write_text(json.dumps(lat_sum, indent=2))
    print(json.dumps(lat_sum, indent=2))


if __name__ == "__main__":
    main()
