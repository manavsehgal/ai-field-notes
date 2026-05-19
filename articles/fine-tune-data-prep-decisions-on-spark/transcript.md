# Transcript — fine-tune-data-prep-decisions-on-spark

Provenance for the article. Captures the cleaned source material from
session 39 + session 40 of the patent-strategist W3 work.

---

## Session 39 (2026-05-19, first train)

### What was run

```bash
DATASET=/home/nvidia/data/corpus/patent-prod-2026-05-19.train.jsonl \
RUN_NAME=patent-strategist-v1-2026-05-19 \
EPOCHS=2 CHECKPOINT_EVERY=200 \
./scripts/g3_train_first_lora.sh
```

Conversion script (`scripts/build_train_jsonl.py` first cut) wrote
`text = f"<｜User｜>{prompt}<｜Assistant｜>{response}"` — no BOS, no EOS.

### What happened

- Train: 128 minutes wall. Loss step 5: 3.17 → step 100: 1.57 → step 200: 1.40 → step 626: 1.17–1.25 stable. `mean_token_accuracy` 0.48 → 0.72. `grad_norm` 0.4–0.9.
- Merge: 47 s. Output 16.38 GB BF16.
- Layer-1 isolation verified.
- Probe row 1 (AIME-style): `wall=369s has_think=False`. Hit full MNT=4096.
- Probe row 2: same shape.
- Stopped at row 4 of 20 to write a durable handoff for s40 cold-start.

### Diagnosis (round 1)

Read: tokenizer.encode does NOT auto-add BOS or EOS for this model.
Verified: first/last tokens of training row 1 were `(151669, 13)` — `<｜User｜>` then `.`. No `<｜begin▁of▁sentence｜>` (151643), no `<｜end▁of▁sentence｜>` (151645).

Fix: one-line patch in `scripts/build_train_jsonl.py`:

```python
BOS = "<｜begin▁of▁sentence｜>"  # 151643
EOS = "<｜end▁of▁sentence｜>"    # 151645
text = f"{BOS}<｜User｜>{prompt}<｜Assistant｜>{response}{EOS}"
```

Memory saved: `feedback_sft_eos_bos_explicit`.

### Cost

128 min train + 14 min partial probe + 16 GB disk on a model that was thrown away.

---

## Session 40 (2026-05-19, retrain + diagnosis)

### Preflight

```
docker ps --filter "name=^ps-train$" --format '{{.Names}} uptime={{.RunningFor}}'
ps-train uptime=35 hours ago

python3 -c "import json; r=json.loads(open('/home/nvidia/data/corpus/patent-prod-2026-05-19.train.jsonl').readline()); assert r['text'].startswith('<｜begin▁of▁sentence｜><｜User｜>'); assert r['text'].endswith('<｜end▁of▁sentence｜>'); print('OK BOS+EOS bookends present')"
OK BOS+EOS bookends present

docker exec -i ps-train python3 -c "
from transformers import AutoTokenizer; import json
tok=AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-R1-0528-Qwen3-8B', trust_remote_code=True)
r=json.loads(open('/home/nvidia/data/corpus/patent-prod-2026-05-19.train.jsonl').readline())
ids=tok.encode(r['text'])
assert ids[0]==151643 and ids[-1]==151645, f'BOS/EOS missing: first={ids[0]} last={ids[-1]}'
print('OK tokenizer sees BOS+EOS')"
OK tokenizer sees BOS+EOS — first=151643 last=151645 len=609 tok
```

Free memory: 115 GiB / 121 GiB total. No competing services.

### Train

Same invocation as s39. Wall: 131 min (vs 128 min s39 reference — within ±5 min).
Loss curve essentially identical to s39's. Layer-1 isolation verified again.
Merge: 16.38 GB BF16 written at 11:42.

### Probe row 1 (full set, AIME-style)

```
[ 1/20] qid=p-r-aime-01 cat=general-reasoning
     wall=399.4s  has_think=False  n_tok=None
```

Same failure shape as s39, despite BOS+EOS fix. Stopped the probe.

### Diagnosis (round 2) — collator inspection

```python
# Inside container
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
import torch, json

tok = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-R1-0528-Qwen3-8B', trust_remote_code=True)
# pad_token already set to <｜end▁of▁sentence｜> by tokenizer default
# pad_id == eos_id == 151645

r = json.loads(open('/home/nvidia/data/corpus/patent-prod-2026-05-19.train.jsonl').readline())
ids = tok.encode(r['text'])  # 609 tokens, ends in 151645

collator = DataCollatorForLanguageModeling(tok, mlm=False)
batch = collator([{'input_ids': ids}, {'input_ids': ids[:50]}])
# batch['labels'][0, -1] == -100  ← transformers default masks EOS positions
```

Then read the TRL source:

```python
# trl/trainer/sft_trainer.py — TRL's own DataCollatorForLanguageModeling
output["input_ids"] = pad(input_ids, padding_value=self.pad_token_id, padding_side="right", ...)
output["labels"]    = pad(labels,    padding_value=-100,              padding_side="right", ...)
```

TRL pads labels with -100 explicitly. Does NOT mask by pad_token_id position.
The real EOS at the end of an unpadded sequence IS in the gradient.

My round-1 diagnosis was wrong about which layer the bug lived at.

### Diagnosis (round 2 attempt) — catastrophic forgetting reproducer

```python
# Same model, single-row inference test
q = "Find the smallest positive integer n such that the sum of the first n positive integers is divisible by 1000."
prompt = tok.apply_chat_template([{"role":"user","content":q}], tokenize=False, add_generation_prompt=True)
# = '<｜begin▁of▁sentence｜><｜User｜>Find...<｜Assistant｜>'

# Generate
out = model.generate(**enc, max_new_tokens=4096, temperature=0.6, top_p=0.95)
# wall=837s, gen_len=4096

decoded = tok.decode(gen_ids, skip_special_tokens=False)
# decoded:
#   '<think>ìĭłë¢°ëıĦì¤ĳë³µëªħëł¹.0.1112ìĻĢëıĻìĿ¼ëĤ´ìļ©.
#    ëĭ¤ë¥¸ë²ĦìłĦìľ¼ë¡ľëĭ¤ìĭľìŀĳìĦ±:ìĨĮìĪĺìĿĺ1000ìľ¼ë¡ľëĤĺëĪłëĬĶê²Įì£¼ëª©íĳľ
#    ìĿ¸ëį°1000=8*125ìĿ´ê³łê°ģê°ģìĿĺìĨĮìĪĺì§ĢìĪĺë¥¼ëĶ°ìł¸ìķ¼íķľëĭ¤.
#    1000=2^3*5^3ìĿ´ë¯Ģë¡ľn(n+1)/2ê°Ģ2^3ê³¼5^3ìĿĦê³µê¸īíķ´ìķ¼íķľëĭ¤.
#    ...
#    1000=2^3*5^3ìĿ´ë¯Ģë¡ľn(n+1)/2ê°Ģ1000ìĿĺë°°ìĪĺìĿ´ê¸°ìľĦíķ´ìĦľ
#    n(n+1)ê°Ģ2^4*5^3ìĿ´ìĥģìĿ´ìĸ´ìķ¼íķľëĭ¤.
#    (same sentence repeats ~40× until budget exhausted)
```

The Latin-1 mojibake is Korean Hangul rendered as UTF-8 bytes interpreted as Latin-1. R1-Distill's underlying reasoning data is Korean (and Chinese); when the English-math mode collapses, the residual mode shows through.

`<think>` opens (token 151667). Never closes. No EOS. Falls into repetition loop.

So at least part of the diagnosis was right: catastrophic forgetting on out-of-distribution input is real. But I needed to check whether the patent side was actually working before concluding.

### Patent-only probe (10 rows)

Filtered probe to patent-irac + patent-strategic rows only:

```
[ 1/10] qid=p-p-irac-01 cat=patent-irac
     wall=64.3s has_think=True n_tok=20
[ 2/10] qid=p-p-irac-02 cat=patent-irac
     wall=84.9s has_think=True n_tok=99
[ 3/10] qid=p-p-irac-03 cat=patent-irac
     wall=122.7s has_think=True n_tok=155
[ 4/10] qid=p-p-irac-04 cat=patent-irac
     wall=130.8s has_think=True n_tok=123
[ 5/10] qid=p-p-irac-05 cat=patent-irac
     wall=66.1s has_think=True n_tok=90
[ 6/10] qid=p-p-strat-01 cat=patent-strategic
     wall=77.6s has_think=True n_tok=57
[ 7/10] qid=p-p-strat-02 cat=patent-strategic
     wall=189.0s has_think=True n_tok=276
[ 8/10] qid=p-p-strat-03 cat=patent-strategic
     wall=110.9s has_think=True n_tok=331
[ 9/10] qid=p-p-strat-04 cat=patent-strategic
     wall=52.4s has_think=True n_tok=25
[10/10] qid=p-p-strat-05 cat=patent-strategic
     wall=68.8s has_think=True n_tok=88

overall: {'think_presence_rate': 1.0, 'think_token_length': 126.4, 'n': 10}
```

10/10 think presence. Mean think length 126 tok (vs baseline 1252 tok — 10× shorter).
Wall per row 52–189s (vs ~94s baseline) — model stops naturally on patent prompts.

### But the content was wrong

Reading per-row output:

```
qid=p-p-irac-01:
  think: A1duplicateof3886.Diversifybyemphasizingthe§103reasoningstructureandtheKSRanalysis.
  answer: The§103rejectionisproperunderKSRv.Teleflex,550U.S.398(2007),andMPEP2143...

qid=p-p-strat-04:
  think: Duplicateof501.Diversifybyleadingwiththeacquirer'sdiligenceframinginsteadoftheacquirer'sperspective.
  answer: Theacquirer'sIPdiligenceteamwillfirstevaluateyourpatentportfolio...

qid=p-p-irac-03:
  answer: Claim1recitesamethodof"receivingdata,applyinganalgorithmtocomputearesult..."
          Under35USC§101,examinermustfirstdeterminewhethertheclaimisdirectedto
          aweunder...AliceCorp.v.CLSBank,573U.S.208(2014)...MayoClinicv.KleigElectric,564U.S.638(2011)...
```

Three defects visible:
1. NO SPACES between words
2. Synth-pipeline meta-state ("A1 duplicate of N", "Diversify by ...") leaks at start of think
3. Hallucinated case ("Mayo Clinic v. Klein Electric" doesn't exist — real case is Mayo v. Prometheus)

### Corpus audit (round 3 — the real diagnosis)

```python
import json, re
rows = [json.loads(l) for l in open('/home/nvidia/data/corpus/patent-prod-2026-05-19.jsonl')]

fam_prefix_count = 0
dup_count = 0
diversify_count = 0
for r in rows:
    think_text = re.search(r'<think>(.*?)</think>', r['response'], re.DOTALL)
    if not think_text: continue
    t = think_text.group(1).strip()
    if re.match(r'^(A[124]|E[12])(\s|:|\.|duplicate|spice)', t[:30]):
        fam_prefix_count += 1
    if 'duplicate of' in t.lower():
        dup_count += 1
    if 'diversify' in t.lower():
        diversify_count += 1

# Result:
#   total rows: 5000
#   rows with family-prefix in think: 2797  (55.94%)
#   rows with 'duplicate of' in think: 311  (6.22%)
#   rows with 'diversify'    in think: 1012 (20.24%)
```

Sample contaminated rows:

```
row 152 fam=A4: A4 traversal. Rejection: §103 over WO-2020/098765 against a method of edge-deploying a quantized transformer; cited reference characterized as teaching an "air-cooled embodiment." Mismatch between the …
row 159 fam=A4: A4 spice combinator: rejection is framed as "112(a) enablement ... citing JP-H10-123456 as anticipating." That conflates two statutory grounds — §112(a) enablement is a disclosure-side requirement on …
row 168 fam=A4: A4 spice combinator: "§112(b) indefiniteness rejection citing Smith et al. (2018) IEEE Trans. as anticipating" — §112(b) is a definiteness/scope problem in the applicant's claim, not a prior-art doctr…
```

The corpus source prose has normal spacing. The no-space token generation in the model output is a separate model-side artifact correlated with the meta-prefix pattern training the model into a specific token-variant preference.

### Pivot

Per user direction: do NOT clean+retrain. The model is shelved.
Pivot to article-only publish. The article IS the customer-facing deliverable from W3.

---

## Key artifacts referenced

- `scripts/build_train_jsonl.py` — corpus → text-shape conversion (BOS/EOS fix that turned out unnecessary)
- `scripts/g3_train_first_lora.{sh,py}` — TRL `SFTTrainer` wrapper around `DeepSeek-R1-0528-Qwen3-8B`
- `scripts/probe_reasoning.py` — 20-row reasoning-preservation probe runner
- `scripts/compare_probes.py` — gate evaluator (presence ≥ 0.54, length ≥ 964)
- `probes/reasoning-preservation-20q.jsonl` — probe questions (10 general + 5 IRAC + 5 strategic)
- `probes/baseline-4096.json` — baseline R1-Distill, MNT=4096 probe (presence=0.60, length=1285)
- `/home/nvidia/data/corpus/patent-prod-2026-05-19.jsonl` — 5000-row synth corpus (canonical)
- `/home/nvidia/data/aifn-train-lora/patent-strategist-v1-2026-05-19/` — s40 v2 train output (shelved)

## Memory entries cross-linked from the article

- `feedback_sft_eos_bos_explicit` — saved during s39's round-1 misdiagnosis
- (new this session, implicit) — the TRL collator doesn't mask EOS correction
- `feedback_preflight_bench_before_quant` — 5-row preflight pattern
- `feedback_keep_scorer_local_until_reuse` — content judge as fieldkit promotion candidate
- `project_spark_unified_memory_oom` — Spark unified-memory accounting
- `reference_pytorch_2511_image_contents` — ps-train container baseline
- `feedback_refresh_stats_on_publish` — stats-refresh discipline before commit

---

The notes.md scaffold (~31k of pre-existing planning) covered the
data-prep decision tree forks 1–6 — corpus gap, base model, synthesis paths,
throughput, quality, orchestration. The published article reframes around
the s39+s40 failure story; forks 4–6 informed the "intended pipeline"
backdrop but the article body focuses on the three rounds of misdiagnosis
and the corpus-quality lesson.
