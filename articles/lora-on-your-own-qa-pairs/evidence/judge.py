"""Grade base vs adapter predictions against reference answers using NIM 8B as judge,
plus lightweight deterministic metrics (refusal rate, keyword overlap, length).
"""
import json
import re
import urllib.request
import concurrent.futures as cf
import sys

NIM_URL = "http://localhost:8000/v1/chat/completions"
NIM_MODEL = "meta/llama-3.1-8b-instruct"

BASE = [json.loads(l) for l in open("/home/nvidia/lora-work/preds_base.jsonl")]
ADAP = [json.loads(l) for l in open("/home/nvidia/lora-work/preds_adapter.jsonl")]
assert len(BASE) == len(ADAP)
N = len(BASE)


REFUSAL = re.compile(
    r"not (?:directly|explicitly|specifically) (?:provided|mentioned|stated|specified|detailed|present)"
    r"|not provided"
    r"|is not mentioned"
    r"|no (?:specific|direct) (?:information|detail)"
    r"|cannot be determined"
    r"|refer (?:to|directly to)"
    r"|one would need to refer"
    r"|for (?:accurate|specific) (?:information|details)",
    re.I,
)


def is_refusal(ans: str) -> bool:
    return bool(REFUSAL.search(ans))


def tokens(s: str):
    return set(re.findall(r"[A-Za-z0-9._%:/-]+", s.lower()))


def overlap(pred: str, ref: str) -> float:
    ref_t = {t for t in tokens(ref) if len(t) >= 3 and not t.isalpha() or len(t) >= 5}
    if not ref_t:
        ref_t = tokens(ref)
    pred_t = tokens(pred)
    if not ref_t:
        return 0.0
    return len(ref_t & pred_t) / len(ref_t)


JUDGE_PROMPT = """You are grading an assistant's answer against a reference answer from an article.

Scale 0-5:
  0 = refuses / says "not provided" / no content
  1 = attempts an answer but is wrong in its key fact
  2 = partially correct but misses the key fact
  3 = gets the key fact directionally right but imprecise
  4 = key fact correct
  5 = key fact correct AND phrased concisely (matches reference style)

Return ONLY a JSON object: {{"score": <int 0-5>, "rationale": "<one short sentence>"}}.
No other text.

QUESTION: {q}
REFERENCE: {r}
ANSWER:    {a}

JSON:"""


def judge(q, r, a, retries=3):
    prompt = JUDGE_PROMPT.format(q=q, r=r, a=a)
    body = {
        "model": NIM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 160,
    }
    req = urllib.request.Request(
        NIM_URL,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
                raw = data["choices"][0]["message"]["content"].strip()
                if raw.startswith("```"):
                    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
                s = raw.find("{")
                e = raw.rfind("}")
                if s != -1 and e > s:
                    raw = raw[s : e + 1]
                obj = json.loads(raw)
                return int(obj["score"]), obj.get("rationale", "")
        except Exception as exc:
            if attempt == retries - 1:
                print(f"judge fail: {exc}", file=sys.stderr)
                return None, f"judge-failed: {exc}"
    return None, "exhausted"


def score_side(preds, label):
    print(f"--- scoring {label} ---", file=sys.stderr)
    results = [None] * N
    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        futs = {
            ex.submit(judge, p["question"], p["reference"], p["prediction"]): i
            for i, p in enumerate(preds)
        }
        for fut in cf.as_completed(futs):
            i = futs[fut]
            score, rationale = fut.result()
            results[i] = (score, rationale)
    return results


def main():
    base_scores = score_side(BASE, "base")
    adap_scores = score_side(ADAP, "adapter")

    rows = []
    for i in range(N):
        b, a = BASE[i], ADAP[i]
        rows.append({
            "question": b["question"],
            "reference": b["reference"],
            "source": b["source"],
            "base_prediction": b["prediction"],
            "adapter_prediction": a["prediction"],
            "base_refusal": is_refusal(b["prediction"]),
            "adapter_refusal": is_refusal(a["prediction"]),
            "base_overlap": overlap(b["prediction"], b["reference"]),
            "adapter_overlap": overlap(a["prediction"], b["reference"]),
            "base_judge_score": base_scores[i][0],
            "base_judge_rationale": base_scores[i][1],
            "adapter_judge_score": adap_scores[i][0],
            "adapter_judge_rationale": adap_scores[i][1],
            "base_tokens": b["new_tokens"],
            "adapter_tokens": a["new_tokens"],
            "base_wall_s": b["wall_s"],
            "adapter_wall_s": a["wall_s"],
        })

    with open("/home/nvidia/lora-work/graded.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    import statistics

    def agg(field, filt=None):
        if filt is None:
            _field = field
            filt = lambda r, _f=_field: r[_f] is not None
        vals = [r[field] for r in rows if filt(r)]
        return {
            "n": len(vals),
            "mean": statistics.mean(vals) if vals else None,
            "median": statistics.median(vals) if vals else None,
        }

    def rate(field):
        return sum(1 for r in rows if r[field]) / len(rows)

    summary = {
        "n_items": N,
        "refusal_rate": {
            "base": rate("base_refusal"),
            "adapter": rate("adapter_refusal"),
        },
        "judge_score": {
            "base": agg("base_judge_score", lambda r: r["base_judge_score"] is not None),
            "adapter": agg("adapter_judge_score", lambda r: r["adapter_judge_score"] is not None),
        },
        "keyword_overlap": {
            "base": agg("base_overlap"),
            "adapter": agg("adapter_overlap"),
        },
        "answer_tokens": {
            "base_mean": statistics.mean([r["base_tokens"] for r in rows]),
            "adapter_mean": statistics.mean([r["adapter_tokens"] for r in rows]),
        },
        "wall_seconds": {
            "base_mean": statistics.mean([r["base_wall_s"] for r in rows]),
            "adapter_mean": statistics.mean([r["adapter_wall_s"] for r in rows]),
        },
    }
    # per-score distribution
    from collections import Counter
    for side in ("base", "adapter"):
        c = Counter(r[f"{side}_judge_score"] for r in rows)
        summary[f"{side}_judge_distribution"] = {
            str(k): c.get(k, 0) for k in sorted(set(list(c.keys()) + [0, 1, 2, 3, 4, 5]))
        }

    with open("/home/nvidia/lora-work/grade_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
