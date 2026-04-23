"""Split qa.jsonl into train/eval, stratified by source article."""
import json
import random
from collections import defaultdict
from pathlib import Path

random.seed(1337)
pairs = [json.loads(l) for l in open("/home/nvidia/lora-work/qa.jsonl")]

by_src = defaultdict(list)
for p in pairs:
    by_src[p["source"]].append(p)

train, eval_ = [], []
for src, items in by_src.items():
    random.shuffle(items)
    n_eval = max(3, len(items) // 6)  # ~15% held-out per source
    eval_.extend(items[:n_eval])
    train.extend(items[n_eval:])

random.shuffle(train)
random.shuffle(eval_)

Path("/home/nvidia/lora-work/train.jsonl").write_text("\n".join(json.dumps(p) for p in train) + "\n")
Path("/home/nvidia/lora-work/eval.jsonl").write_text("\n".join(json.dumps(p) for p in eval_) + "\n")
print(f"train: {len(train)}  eval: {len(eval_)}  total: {len(pairs)}")
print("per-source eval counts:")
from collections import Counter
for s, n in Counter(p["source"] for p in eval_).most_common():
    print(f"  {n:3d}  {s}")
