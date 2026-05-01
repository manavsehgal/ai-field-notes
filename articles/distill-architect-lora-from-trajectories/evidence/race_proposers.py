"""Race the distilled LoRA against the 8B NIM that wrote the trajectory.

For each of the 8 held-out histories (test.jsonl), we generate one proposal
from each proposer and score:
    - validity:  does the output parse as the menu-conforming JSON?
    - knob_match:  did it pick the same knob as the ground-truth iter?
    - value_match: did it pick the same (knob, value) pair?
    - latency_s:   wall-clock per proposal
    - val_bpb_lookup: if the proposed cfg appears in trajectory.jsonl, its
                      observed val_bpb; else null.

The 8B is reached via NIM at NIM_BASE (default localhost:8000). The LoRA
runs locally inside the same container as training.

Outputs:
    race_results.json   — aggregate stats per proposer
    preds.jsonl         — per-iter prompt, both proposals, parse + match flags
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from copy import deepcopy
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

EVIDENCE = Path(__file__).resolve().parent
A4_EVIDENCE = EVIDENCE.parent.parent / "autoresearch-agent-loop" / "evidence"
A5_EVIDENCE = EVIDENCE.parent.parent / "guardrails-for-code-generation" / "evidence"

BASE = os.environ.get("BASE_PATH", "/work/base")
ADAPTER = os.environ.get("ADAPTER_PATH", "/work/adapter")
TEST_JSONL = os.environ.get("TEST_JSONL", str(EVIDENCE / "test.jsonl"))
TRAJECTORY = os.environ.get("TRAJECTORY", str(A4_EVIDENCE / "trajectory.jsonl"))
MENU_PATH = os.environ.get("MENU_PATH", str(A5_EVIDENCE / "perturbation_menu.json"))

NIM_BASE = os.environ.get("NIM_BASE", "http://localhost:8000")
NIM_MODEL = os.environ.get("NIM_MODEL", "meta/llama-3.1-8b-instruct")

OUT_RESULTS = EVIDENCE / "race_results.json"
OUT_PREDS = EVIDENCE / "preds.jsonl"


def load_menu() -> dict:
    return json.load(open(MENU_PATH))


def parse_proposal(text: str) -> dict | None:
    """Extract the first JSON object from text. None if no parse."""
    m = re.search(r"\{[^{}]*\"knob\"[^{}]*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def validate(proposal: dict | None, menu: dict) -> dict:
    """Return {valid, reason}. Mirrors A5 rails minimally."""
    if proposal is None:
        return {"valid": False, "reason": "no_parse"}
    if not isinstance(proposal, dict):
        return {"valid": False, "reason": "not_dict"}
    knob = proposal.get("knob")
    val = proposal.get("new_value", proposal.get("value"))
    if knob is None or val is None:
        return {"valid": False, "reason": "missing_keys"}
    if knob not in menu["knobs"]:
        return {"valid": False, "reason": f"knob_not_in_menu:{knob}"}
    spec = menu["knobs"][knob]
    if "choices" in spec and val not in spec["choices"]:
        return {"valid": False, "reason": f"value_not_in_choices"}
    if "range" in spec:
        lo, hi = spec["range"]
        try:
            if not (lo <= float(val) <= hi):
                return {"valid": False, "reason": "value_out_of_range"}
        except (TypeError, ValueError):
            return {"valid": False, "reason": "value_not_numeric"}
    return {"valid": True, "reason": ""}


def lookup_val_bpb(knob: str, value, traj_iters: list[dict], baseline_cfg: dict) -> float | None:
    """If a (knob, value) was tried and its candidate_cfg matches what we'd
    apply to baseline_cfg, return the observed val_bpb. Else None."""
    target_cfg = deepcopy(baseline_cfg)
    target_cfg[knob] = value
    for r in traj_iters:
        if r.get("candidate_cfg") == target_cfg:
            return r.get("val_bpb")
    return None


def call_nim_8b(messages: list[dict], temperature: float = 0.5,
                max_tokens: int = 200, timeout_s: int = 60) -> dict:
    import requests
    url = f"{NIM_BASE}/v1/chat/completions"
    payload = {"model": NIM_MODEL, "messages": messages,
               "temperature": temperature, "max_tokens": max_tokens, "stream": False}
    t0 = time.perf_counter()
    try:
        r = requests.post(url, json=payload, timeout=timeout_s)
        latency = time.perf_counter() - t0
        if r.status_code != 200:
            return {"ok": False, "text": "", "latency_s": latency,
                    "error": f"http {r.status_code}: {r.text[:200]}"}
        return {"ok": True, "text": r.json()["choices"][0]["message"]["content"],
                "latency_s": latency}
    except requests.exceptions.RequestException as e:
        return {"ok": False, "text": "", "latency_s": time.perf_counter()-t0, "error": str(e)}


def call_local(model, tok, messages: list[dict],
               max_new_tokens: int = 200, temperature: float = 0.5) -> dict:
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    latency = time.perf_counter() - t0
    gen = out[0][inputs["input_ids"].shape[1]:]
    return {"ok": True, "text": tok.decode(gen, skip_special_tokens=True),
            "latency_s": latency}


def main() -> None:
    print(f"loading base + adapter from {BASE}, {ADAPTER}")
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa")
    distilled = PeftModel.from_pretrained(base_model, ADAPTER)
    distilled.eval()
    print("base + adapter loaded.")

    menu = load_menu()
    test_rows = [json.loads(l) for l in open(TEST_JSONL)]
    traj_iters = []
    with open(TRAJECTORY) as f:
        for line in f:
            o = json.loads(line)
            if o.get("stage") == "evaluated":
                traj_iters.append(o)

    nim_up = False
    try:
        import requests
        r = requests.get(f"{NIM_BASE}/v1/models", timeout=2)
        nim_up = r.status_code == 200
    except Exception:
        pass
    print(f"NIM 8B at {NIM_BASE}: {'UP' if nim_up else 'DOWN — 8B comparisons will be skipped'}")

    preds = []
    distilled_lat, nim_lat = [], []
    distilled_correct_knob, nim_correct_knob = 0, 0
    distilled_correct_value, nim_correct_value = 0, 0
    distilled_valid, nim_valid = 0, 0

    for row in test_rows:
        msgs = row["messages"][:2]  # system + user only
        gt = json.loads(row["messages"][2]["content"])  # ground-truth proposal
        # Reconstruct baseline_cfg the agent saw at this iter from the prompt's
        # JSON block (between "## Current cfg" and "## Recent iterations").
        user_text = msgs[1]["content"]
        m_cfg = re.search(r"## Current cfg.*?\n(\{.*?\})\n", user_text, re.DOTALL)
        baseline_cfg = json.loads(m_cfg.group(1)) if m_cfg else None

        # Distilled
        d_resp = call_local(distilled, tok, msgs)
        d_prop = parse_proposal(d_resp["text"])
        d_val = validate(d_prop, menu)
        distilled_lat.append(d_resp["latency_s"])
        if d_val["valid"]:
            distilled_valid += 1
            if d_prop["knob"] == gt["knob"]:
                distilled_correct_knob += 1
                if d_prop.get("new_value", d_prop.get("value")) == gt["new_value"]:
                    distilled_correct_value += 1
        d_lookup = lookup_val_bpb(d_prop["knob"], d_prop.get("new_value", d_prop.get("value")),
                                  traj_iters, baseline_cfg) if (d_val["valid"] and baseline_cfg) else None

        # 8B NIM
        if nim_up:
            n_resp = call_nim_8b(msgs)
            n_prop = parse_proposal(n_resp["text"])
            n_val = validate(n_prop, menu)
            nim_lat.append(n_resp["latency_s"])
            if n_val["valid"]:
                nim_valid += 1
                if n_prop["knob"] == gt["knob"]:
                    nim_correct_knob += 1
                    if n_prop.get("new_value", n_prop.get("value")) == gt["new_value"]:
                        nim_correct_value += 1
            n_lookup = lookup_val_bpb(n_prop["knob"], n_prop.get("new_value", n_prop.get("value")),
                                      traj_iters, baseline_cfg) if (n_val["valid"] and baseline_cfg) else None
        else:
            n_resp = {"ok": False, "text": "", "latency_s": None}
            n_prop, n_val, n_lookup = None, {"valid": False, "reason": "nim_down"}, None

        preds.append({
            "iter": row["iter"],
            "ground_truth": gt,
            "distilled": {"text": d_resp["text"], "proposal": d_prop, "valid": d_val,
                          "latency_s": d_resp["latency_s"], "val_bpb_lookup": d_lookup},
            "nim_8b": {"text": n_resp["text"], "proposal": n_prop, "valid": n_val,
                       "latency_s": n_resp["latency_s"], "val_bpb_lookup": n_lookup},
        })
        print(f"iter {row['iter']:>2d}: gt={gt['knob']}={gt['new_value']}"
              f"  d={(d_prop.get('knob','?'),d_prop.get('new_value','?')) if d_prop else 'parse_fail'}"
              f"  n={(n_prop.get('knob','?'),n_prop.get('new_value','?')) if n_prop else ('SKIP' if not nim_up else 'parse_fail')}")

    n_test = len(test_rows)
    results = {
        "n_test": n_test,
        "distilled": {
            "validity_rate": distilled_valid / n_test,
            "knob_match_rate": distilled_correct_knob / n_test,
            "exact_match_rate": distilled_correct_value / n_test,
            "mean_latency_s": sum(distilled_lat) / len(distilled_lat) if distilled_lat else None,
            "median_latency_s": sorted(distilled_lat)[len(distilled_lat)//2] if distilled_lat else None,
        },
        "nim_8b": {
            "validity_rate": nim_valid / n_test if nim_up else None,
            "knob_match_rate": nim_correct_knob / n_test if nim_up else None,
            "exact_match_rate": nim_correct_value / n_test if nim_up else None,
            "mean_latency_s": sum(nim_lat) / len(nim_lat) if nim_lat else None,
            "median_latency_s": sorted(nim_lat)[len(nim_lat)//2] if nim_lat else None,
        } if nim_up else {"status": "skipped — NIM 8B not running"},
        "throughput_speedup": (sum(nim_lat)/len(nim_lat)) / (sum(distilled_lat)/len(distilled_lat))
            if nim_up and distilled_lat and nim_lat else None,
    }

    with open(OUT_RESULTS, "w") as f:
        json.dump(results, f, indent=2)
    with open(OUT_PREDS, "w") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")

    print("\n=== RESULTS ===")
    print(json.dumps(results, indent=2))
    print(f"\nwrote {OUT_RESULTS}")
    print(f"wrote {OUT_PREDS}")


if __name__ == "__main__":
    main()
