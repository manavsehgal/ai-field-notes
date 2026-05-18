# Notes — fine-tune data-prep journey: model & path selection tradeoffs

Reference / research notes for the planned article. The tech-writer skill should consume this file when drafting `article.md` in this directory.

**Slug rationale:** `fine-tune-data-prep-decisions-on-spark` — captures both halves (base-model selection + synthesis-path routing). Companion piece to the LoRA-stack mechanics article (also in HANDOFF backlog).

**Author voice:** Manav deep-dive — fork-by-fork, measured numbers not opinions, every decision tied to a specific incident. Pairs with the LoRA-mechanics article as the *why* (decisions) to its *what* (mechanics).

**Editorial arc:** belongs in the "Machine that Builds Machines" thread (per `[[project_nvidia_learn_editorial]]` — the renamed Autoresearch arc, broadened to /book/ Ch10–11 thesis on 2026-05-08).

---

## Article frame

**Hook:** Patent-strategist W3 needs ~25k synthetic `<think>chain</think>answer` patent-law training examples. There's no public corpus of this shape (per spec §4 Layer 2 — patent reasoning data isn't on Hugging Face; it's locked in attorney workflows). So the data has to be synthesized. The question every fine-tuner faces — how do you build training data when the corpus doesn't exist? — splits into five forks. This article walks each fork with the measured number that decided it.

**Reader:** somebody who's thinking about fine-tuning a small reasoning model on a domain corpus they don't yet have, on hardware they actually own (Spark / 1× consumer GPU / single-node cluster), and wants the *decision math*, not the recipe.

**Throughline:** every fork forced a sharp choice between cap-dense/wall-cheap vs cap-free/wall-heavy paths. The Spark constrains both axes (128 GB unified memory caps local model size; a single human's CC weekly cap caps in-session generation rate). The two axes have to be traded against each other consciously, and the article is about teaching the trade.

---

## The five forks (from HANDOFF backlog, in chronological order)

### Fork 1: The corpus gap — why synthetic at all

**Situation (sessions 18–19):** spec §4 Layer 2 mandates 100% `<think>chain</think>answer` structure across the patent training mix. Layer 1 (continued pre-training) uses MPEP + BIGPATENT + PatentMatch (real text, no chains). Layer 2 (chain-of-thought) and Layer 3 (reasoning anchor) require chain-bearing examples. **There is no public source.** USPTO publishes rejections but not the attorney response chains. OARD has office-action data but not the structured reasoning. Patent-MCQ exam banks are proprietary.

**Decision:** synthesize Layer 2. Spec §6.1 sets the share at 10% of the total training mix ≈ 25,000 examples across families A1 (claim drafting, 30%), A2 (indefiniteness, 25%), A4 (office-action traversal, 20%), E1 (plain-English explanations, 15%), E2 (MCQ generation, 10%).

**What the article teaches:** when public data doesn't exist for the structure you need, synthesis is the only path — but synthesis isn't free, and the *path* of synthesis is itself a multi-fork decision (forks 3–5 below).

**Data point worth quoting:** the spec table at §6.1 mapping family → row count → why-synthetic-only.

**References:**
- `specs/patent-strategist-v1.md` §4 Layer 2, §5.3, §6.1
- Memory: `[[project_artifact_manifests_phase2]]` for the broader patent-strategist project shape

---

### Fork 2: Base model selection — chat-tuned vs continued-pretrain trap

**Situation (sessions 26–28):** several "patent-friendly" 8B models were considered. The trap caught us on the first attempt — picking a Hugging Face namespace by name without reading the chat-template field.

**Decision pipeline that emerged:**
1. **Preflight bench gate** — score 5 vertical-bench questions on FP weights via `transformers` before sinking multi-hour quant+measure cycles. Saved ~5h on the 2026-05-13 finance-Llama3-8B attempt that scored 0/250.
2. **Chat-template check** — read the model card's "How to use" section, look for SFT/DPO/Hermes/Tulu/Zephyr in the lineage. `instruction-pretrain/*` namespace is **NOT** chat-tuned (Microsoft's continued-pretraining methodology). Skipped a planned attempt that would have produced bare-completion outputs on a chain-of-thought target.
3. **MNT (max-new-tokens) re-baseline** — default `--num-predict 256` truncates `<think>` blocks before the answer token lands. II-Medical-8B scored 2/5 at 256, clean sweep at 1024. The fix went into memory `[[feedback_reasoning_model_npredict]]` (and got extended in session 30 with the chain+answer 5–8K budget rule).

**Final base-model pick:** still in flight — the W3 production LoRA train is gated on the corpus build, which is now unblocked by the Track A decision.

**What the article teaches:** base-model selection has at least 3 silent failure modes (no chat template → bare completions; wrong MNT → truncated chains; wrong license → can't ship). All 3 caught us in real time on patent-strategist W3.

**Data points worth quoting:**
- Preflight bench: 0/5 → abort, save 5h
- MNT=256 vs MNT=1024: 2/5 → 5/5 on II-Medical-8B
- License trap: NC-licensed bases block commercial Orionfold tier

**References:**
- Memory: `[[feedback_preflight_bench_before_quant]]`, `[[feedback_chat_vs_continued_pretrain_trap]]`, `[[feedback_reasoning_model_npredict]]`
- Sessions 26–28 of HANDOFF
- `scripts/g3_preflight_bench.py`

---

### Fork 3: Synthesis-path elimination — what couldn't work and why

**Situation (sessions 28–29):** four candidate paths for generating the 25k Layer 2 rows. Three got eliminated; one survived.

**Path A (eliminated): `anthropic.Anthropic(api_key=…)`.** Fails the auth check — no auto-OAuth fallback, no fall-through to the user's existing Claude Code OAuth credentials at `~/.claude/.credentials.json`. Also bills the API console wallet, not the Max 20x subscription that's already paid for. Practical failure: the import works, the request returns 401 unless an API console key is provisioned. We documented the OAuth-fallback workaround for `claude-agent-sdk` at `[[reference_claude_agent_sdk_oauth]]` — but for the bulk synthesis use case, even with OAuth working, the cost shape is wrong because every call is a fresh non-session round-trip with no cache reuse.

**Path B (eliminated): `claude-agent-sdk` subprocess.** Works (the OAuth fallback patch lands cleanly), but each call spawns a `claude` CLI subprocess that does its own auth round-trip, its own context-build, and pays the no-cache penalty on each row. Measured cost: **~1976 tokens/row** on Sonnet 4.6 (session 29 dry run). For 25k rows: ~49M tokens of session billing. Too expensive at production scale.

**Path C (eliminated for production, survives as bench tool): local SLM via Ollama.** Cap-free, but quality decision pending (resolved in fork 5 below).

**Path D (chosen): in-CC-session orchestration.** Claude generates rows directly in the active CC session via the Edit-append pattern. No subprocess, no SDK, no API key. The `claude-corpus-synth` skill (session 29) wraps this: deterministic helper scripts for queue prep + preflight + merge; Claude writes rows live via tool calls. Measured cost in production-incremental cadence: **~500 tokens/row** (session 29 dry run, confirmed at ~477 in session 31's 20-row CC leg). Roughly 4× cheaper than the SDK route.

**What the article teaches:** the three eliminations are different *kinds* of failure (auth, cost-shape, quality). Only the in-CC-session path threads all three needles. The pattern is general: when you have a paid LLM subscription and want to bulk-synthesize artifacts, doing it *inside* a session is structurally cheaper than calling out *from* a script.

**Data points worth quoting:**
- SDK route: 1976 tok/row × 25k = 49.4M tokens (Sonnet 4.6) ≈ huge weekly-cap fraction
- In-CC-session: 477 tok/row × 25k = 11.9M tokens (Opus 4.7) ≈ 27% of weekly Max 20x cap
- Ratio: ~4× cheaper than SDK subprocess

**References:**
- Memory: `[[feedback_llm_skill_pattern]]` (the canonical pattern + ban on both anthropic SDK and claude-agent-sdk routes)
- `[[reference_claude_agent_sdk_oauth]]` for the OAuth-fallback workaround (lives for non-corpus use cases)
- `.claude/skills/claude-corpus-synth/` (the canonical implementation)
- Session 29 of HANDOFF

---

### Fork 4: The cost-shape flip — cap-dense/wall-cheap vs cap-free/wall-heavy

**Situation (sessions 30–31):** the surviving in-CC-session path has a binding cap (Max 20x weekly tokens). The local-SLM path is cap-free but Spark-wall-heavy. **They're not on the same axis.** The article needs to make this explicit because most fine-tuning teachers conflate them.

**The flip, with measured numbers from session 30–31's 20-row routing bench:**

| Path | Per-row wall | 25k continuous wall | 25k realistic wall | Cost |
|---|---|---|---|---|
| Ollama Qwen3.5:9b on Spark (np=8192) | **148 s** (measured, mean of 20 rows) | **~1026 h ≈ 43 days continuous Spark** | ~6+ weeks (1–2 parallel streams max) | $0 cap, Spark fully occupied |
| CC in-session (Opus 4.7) | **~13–18 s** (measured one E1 row at incremental cadence; A4 likely 20–30 s) | **~90 h continuous** if uncapped | **~36 h active across ~125 sessions in 2–4 weekly Max 20x cycles** | 27% of weekly cap mid-estimate, Spark free |

**Why throughput flips even though Spark "should" be fast:** Qwen3.5:9b on Spark runs at **33.5 tok/s output** (very tight band, 32–34 across 20 rows). Opus 4.7 effective rate is ~50–80 tok/s. Speed ratio is only ~2×. But Qwen writes **4951 mean output tokens/row** vs CC's **477 mean output tokens/row** — a **10× verbosity gap**. The verbosity gap dominates the throughput math: even capping `num_predict=1024` wouldn't fix it because the verbosity is a sampling-shape problem, not a budget problem. Qwen wants to produce 500–800 word multi-section markdown analyses with disclaimers and sub-headers for every prompt; CC wants to produce 1–3 sentence final answers.

**What the article teaches:** the cost axes don't combine into a single number. You have to decide *which scarce resource* you're optimizing — weekly LLM cap or weeks-of-Spark — based on what else needs that resource. The Spark is the user's *only* development hardware; sacrificing 6 weeks of it to save 27% of a weekly cap is the wrong trade unless cap consumption is the binding personal constraint. For Manav at the Max 20x tier on a personal AI-builder project, the cap renews every week; the Spark renews never.

**Data points worth quoting:**
- 33.5 vs ~60 tok/s — only 2× speed delta
- 4951 vs 477 mean out_tok — 10× verbosity delta
- 1026 h vs 36 h gated wall — 28× practical throughput delta
- Cap math: 27% weekly cap × 2–4 cycles = full corpus

**References:**
- Memory: `[[feedback_llm_skill_pattern]]`, `[[feedback_stop_unneeded_services]]`, `[[feedback_reasoning_model_npredict]]`
- `/tmp/aifn-bench-local-vs-cc/scores.md` "Throughput axis" section
- Session 31 HANDOFF — routing decision table with per-row wall column

---

### Fork 5: The quality decision — when local fails, *how* matters more than *whether*

**Situation (session 31):** the 20-row local-vs-CC bench scored under the README's rubric (5 = production-quality with concrete MPEP cites + 1–3 sentence answer; 1 = wrong / hallucinated). Rule: ≥16/20 → pure local; 10–15/20 → hybrid (local generates 3 candidates, CC picks best); <10/20 → CC only.

**Result:** CC 20/20 ≥4/5 (mean 5.00). Ollama Qwen3.5:9b 6/20 ≥4/5 (mean 2.95). Decision: **CC-only (Track A).**

**But the count isn't the interesting finding — the failure-mode taxonomy is.** Five systematic failures explain why hybrid doesn't save anything either:

1. **Fabricated MPEP subsections** (rows 4, 8, 16, 17): cites `2173.05(q)`, `2173.05(a)(1)`, `2173.05(a)(2)`, `2181.04`, `2173.03 "Apparatus Claims with Functional Limitations"`. MPEP 2173.05 actually subdivides through (h); (q) doesn't exist. MPEP 2173.03 is "Correction of Inaccuracies," not functional-language guidance. A patent attorney catches these on first read.
2. **Fabricated claim subject matter** on mismatched-prompt rows (rows 2, 9): the queue's spice combinator produced incoherent fact patterns (PEG drug carrier prompt with air-cooled prior art). Qwen invented bridging limitations rather than reaching for the analytically correct move (non-analogous art under MPEP 2141.01(a) + In re Klein). CC recognized both mismatches and used them as the traversal hook — the *correct* legal move.
3. **Mischaracterized core concepts** in plain-language E1 (row 10): terminal disclaimer described as "agreement to share patent rights so the USPTO sees you as a single owner." A terminal disclaimer surrenders term and binds in common ownership; it doesn't "share rights."
4. **Missing §112(f) trigger** on `means for displaying results` (row 8) and `configured to ...` apparatus claims with functional language at point of novelty (rows 4, 17). MPEP 2181 was not invoked.
5. **Verbosity off-spec**: 500–800 word multi-section markdown when the format wanted 1–3 sentences.

Each failure is the kind a hybrid judge-filter would reject. But rejection costs nearly as much as regeneration (the judge has to read the chain to score it). So the hybrid math collapses: filtering systematic failures costs CC time without saving CC time.

**What the article teaches:** "is local good enough" decomposes into *count* (X/20) and *failure-mode taxonomy*. The same 6/20 score from random small failures vs systematic structural failures implies different routing decisions. Hybrid is the right pick only when the local generator's errors are random/independent (judge filters them); it's wrong when errors are systematic (judge replicates the regeneration cost).

**Data points worth quoting:**
- Decision rule: ≥16/20 / 10–15/20 / <10/20 thresholds
- CC: 20/20, mean 5.00
- Ollama Qwen3.5:9b: 6/20, mean 2.95
- Six rows where Ollama scored ≥4: rows 1, 3, 6, 7, 11, 12 — disproportionately the E1 + simple-A4 + simple-A1 rows where citation accuracy matters less. The A2 indefiniteness analyses (where MPEP-citation accuracy is the whole game) cratered.
- Specific hallucinated MPEP cites for sidebars: `2173.05(q)`, `2181.04`, `2173.03 "Apparatus Claims with Functional Limitations"`

**References:**
- `/tmp/aifn-bench-local-vs-cc/scores.md` (full per-row 1–5 scoring + failure-mode taxonomy)
- `/tmp/aifn-bench-local-vs-cc/comparison.md` (side-by-side generator output, 768 KB)
- `/tmp/aifn-bench-local-vs-cc/queue.jsonl` (20-row balanced bench, seed=42, reproducible)
- `scripts/bench_local_vs_cc_*` + `scripts/bench_local_vs_cc_README.md` (the harness)

---

## Key tables for the article

### Table 1: Throughput axis (the load-bearing table)

| Generator | Per-row wall | Output tok/s | Mean out_tok/row | 25k continuous | 25k realistic | Binding constraint |
|---|---|---|---|---|---|---|
| Ollama Qwen3.5:9b on Spark | 148 s | 33.5 (32–34 tight) | 4951 | 1026 h | ~6 weeks (Spark-bound) | Single-Spark hardware |
| CC Opus 4.7 in-session | ~13–18 s (E1); est. 20–30 s (A4) | ~50–80 effective | 477 | ~90 h | ~36 h across 2–4 weekly cycles | Max 20x weekly cap |

### Table 2: Quality axis (the deciding axis)

| Generator | Rows ≥4/5 | Mean score | Routing implication |
|---|---|---|---|
| CC Opus 4.7 in-session | 20 / 20 | 5.00 | Reference ceiling |
| Ollama Qwen3.5:9b | 6 / 20 | 2.95 | Below 10/20 hybrid floor — Track A |

### Table 3: The four synthesis paths, eliminated one by one

| Path | Mechanism | Tokens/row | Status | Why eliminated |
|---|---|---|---|---|
| `anthropic.Anthropic(api_key=…)` | Direct API call | N/A | ✗ | No OAuth fallback; bills wrong wallet; no cache reuse |
| `claude-agent-sdk` subprocess | CLI subprocess per call | 1976 (Sonnet 4.6) | ✗ | ~4× more tokens than in-session (no cache, fresh context per call) |
| Local SLM (Qwen3.5:9b on Ollama) | HTTP to localhost:11434 | 4951 (Qwen output) | ✗ | 6/20 quality below hybrid floor; systematic MPEP fabrication |
| In-CC-session (`claude-corpus-synth`) | Edit-append in active session | 477 (Opus 4.7) | ✓ | 4× cheaper than SDK; production-quality; cap-fits |

---

## Specific anecdotes worth dramatizing

1. **The MPEP 2173.05(q) moment.** When you read Qwen's analysis of row 16, it cites MPEP § 2173.05(q) (Terms of relative nature) with such authoritative formatting — bolded section number, italicized title, "Why it is problematic" sub-bullets — that for a moment it sounds correct. Then you look up MPEP 2173.05 and the subdivisions stop at (h). The (q) is fabricated whole-cloth. This is the canonical "small-model bullshit-with-confidence" pattern, and showing the exact passage side-by-side with the correct MPEP would be the article's strongest single visual.

2. **Row 2 / row 9 fact-pattern mismatch.** The bench's spice combinator generated some incoherent prompts by accident: "fluid-cooled CPU package" paired with "non-quantized model in cloud" prior art; "PEG drug carrier" paired with "air-cooled embodiment" prior art. CC saw these and pivoted to *non-analogous art* (MPEP 2141.01(a)) — the legally correct move. Qwen invented bridging limitations to make the prompt make sense ("quantization state of CPU package"; "liquid-cooled cryogenic PEG coating process"). The contrast captures the difference between a model that *reasons about what's been asked* vs a model that *completes the prompt pattern*.

3. **The 18.7-second wall-clock measurement.** Real, in-session timing on this Spark: Bash `date` mark, Write tool with one full `<think>` + answer row, Bash `date` mark again — 18.7 s total, with ~5 s tool round-trip overhead, so ~13 s content-generating turn. Reproducible. Single sample (won't generalize to A4 family without more data), but anchors the throughput math empirically.

4. **The CIP-to-MBA explanation that almost works.** Qwen row 7's plain-English continuation-in-part explanation reads fine on first pass. Then you notice it misses the 20-year-term caveat — which is the *one* fact an MBA student needs to make the strategic-value calculation. The explanation isn't wrong; it's audience-mismatched. That's a subtler failure than the MPEP fabrications, and important for the article because it shows that "quality" isn't binary even on plain-language tasks.

---

## What this article does NOT cover

- The actual W3 production LoRA train (separate article in the HANDOFF backlog: "Article: W3 LoRA stack on Spark + in-CC-session pattern + local-SLM routing"). This article is the *decisions before* the train; that one is the *mechanics of* the train.
- The G3 quantize/publish pipeline. That's `becoming-a-gguf-publisher-on-spark` territory and orthogonal to data-prep.
- Other verticals' corpus builds. Each vertical (finance, legal, cyber, medical) had its own corpus-prep flavor; this article is patent-strategist specific to keep numbers concrete. A future "cross-vertical synthesis patterns" article would consolidate.

---

## Open questions / things to verify before publishing

- **Verify Opus 4.7 effective tok/s rate.** The 50–80 tok/s figure is an industry-typical estimate, not a measurement. Could do a more rigorous 5-row timing pass at incremental cadence to get a tighter number for the throughput table.
- **A4 per-row timing.** The 13 s figure is from one E1 row. A4 rows are longer (4–5× the think text); per-row time will be higher. Worth measuring before publishing the "~13–20 s" range.
- **Single-rater scoring on the bench.** Section 5 (the quality verdict) was self-rated by Claude in-session, with conflict-of-interest disclosed. The article should either (a) get one human spot-check pass before publishing, or (b) call out the self-scoring caveat prominently and let readers judge.
- **Whether to commit the bench scaffold.** Per HANDOFF: bench scripts uncommitted because they're "decision-supporting tooling, not project deliverable — commit only if routing decision is 'pure local' (then scripts become part of corpus pipeline) or 'hybrid' (then recurring infra)." Decision is CC-only, so technically the scripts can be deprecated. But the *bench pattern* is reusable for future "is local good enough" questions on other domains; arguably worth committing as an evergreen reference even though this specific project doesn't need them. Decide before publishing.

---

## Suggested article structure (first-draft outline)

1. **Cold open:** the MPEP 2173.05(q) screenshot moment. Two-paragraph hook on the bullshit-with-confidence pattern.
2. **The corpus gap.** Why synthesis. Spec §4 Layer 2 mandate. (Fork 1.)
3. **The base-model trap.** Three silent failure modes. Preflight bench saved us. (Fork 2.)
4. **Four synthesis paths walk into a Spark.** Eliminate the SDK route, the API route, defer the local route to a separate bench. Introduce in-CC-session as the pattern. (Fork 3.)
5. **The bench: 20 rows, three generators, two axes.** Quality and throughput. (Bridge to Forks 4–5.)
6. **The throughput flip.** Why 33 tok/s on Spark is slower than expected vs ~60 tok/s on the cloud. The verbosity gap dominates. (Fork 4 with Table 1.)
7. **The quality decision and what it cost.** 20/20 vs 6/20. The five failure modes. Why hybrid doesn't save anything. (Fork 5 with Tables 2 + the failure taxonomy.)
8. **The verdict and what it teaches the next builder.** Cap-dense vs wall-heavy is a *resource-allocation* decision, not a *which-is-better* decision. The right answer depends on which scarce resource you're optimizing.
9. **What's next:** the W3 LoRA train. Pointer to the mechanics article when it lands.

Target length: ~3000–4000 words. Voice: deep-dive, fork-by-fork, every claim backed by a measured number. Pairs with the LoRA-mechanics article as the *what* (decisions) and the *why* (mechanics).
