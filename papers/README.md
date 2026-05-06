# Frontier Scout — paper triage

_Last refresh: 2026-05-06 · 36 papers tracked · [run history](runs/index.md)_

## Recommended dive-deep candidates

These are the papers most worth running through `/frontier-scout eval <id>` next, ranked by combined relevance × popularity × verdict-feasibility:

1. **[From Context to Skills: Can Language Models Learn from Context Skillfully?](2604.27660/paper.md)** · 132 upv · spark-feasible · LLM Wiki
   _Inference-time skill extraction from long context — a NIM-hostable LLM workflow, easy to stand up on Spark for an 'LLM Wiki' study._
2. **[ARIS: Autonomous Research via Adversarial Multi-Agent Collaboration](2605.03042/paper.md)** · 65 upv · spark-feasible · Autoresearch
   _Open-source autonomous-research harness with adversarial review — direct Autoresearch-arc material; runs over hosted NIM endpoints._
3. **[OpenSeeker-v2: Pushing the Limits of Search Agents with Informative and High-Difficulty Trajectories](2605.04036/paper.md)** · 36 upv · spark-feasible · Autoresearch
   _SFT-only recipe for deep-search agents at small scale — high-signal trajectories + LoRA on a 7B base fits the Spark._
4. **[Beyond SFT-to-RL: Pre-alignment via Black-Box On-Policy Distillation for Multimodal RL](2604.28123/paper.md)** · 34 upv · spark-feasible · LLM Wiki
   _Black-box on-policy distillation as pre-alignment for multimodal RL — small-student distillation is in the Spark envelope._
5. **[Nemotron 3 Nano Omni: Efficient and Open Multimodal Intelligence](2604.24954/paper.md)** · 12 upv · spark-feasible · Foundations
   _Native NVIDIA Nemotron 3 Nano Omni 30B-A3B MoE shipped with BF16/FP8/FP4 — FP4 weights (~15 GB) leave ample 128 GB headroom for KV + multimodal context._

## What's new this run

See [runs/2026-05-06/refresh-summary.md](runs/2026-05-06/refresh-summary.md) for new + dropped + distributions.

## Full listing

### Foundations (1)

#### spark-feasible (1)
- [2604.24954 Nemotron 3 Nano Omni: Efficient and Open Multimodal Intellig](2604.24954/paper.md) · 18 · _Native NVIDIA Nemotron 3 Nano Omni 30B-A3B MoE shipped with BF16/FP8/FP4 — FP4 weights (~15 GB) leave ample 128 GB headroom for KV + multimodal context._


### Second Brain (1)

#### spark-feasible (1)
- [2605.00529 Hierarchical Abstract Tree for Cross-Document Retrieval-Augm](2605.00529/paper.md) · 10 · _Hierarchical cross-document Tree-RAG — Second Brain F-arc fit; pgvector + NIM-Embed already in place._


### LLM Wiki (8)

#### spark-feasible (8)
- [2604.27660 From Context to Skills: Can Language Models Learn from Conte](2604.27660/paper.md) · 38 · _Inference-time skill extraction from long context — a NIM-hostable LLM workflow, easy to stand up on Spark for an 'LLM Wiki' study._
- [2604.24927 Large Language Models Explore by Latent Distilling](2604.24927/paper.md) · 29 · _Lightweight test-time distiller plus reweighted sampling on existing open-weight reasoning models fits comfortably within Spark's 128 GB inference envelope._ · [eval](2604.24927/eval.md) · → `articles/test-time-distilling-for-exploration/`
- [2604.28123 Beyond SFT-to-RL: Pre-alignment via Black-Box On-Policy Dist](2604.28123/paper.md) · 27 · _Black-box on-policy distillation as pre-alignment for multimodal RL — small-student distillation is in the Spark envelope._
- [2604.27039 Length Value Model: Scalable Value Pretraining for Token-Lev](2604.27039/paper.md) · 21 · _Token-level length value head over a 7B model is a small auxiliary network atop standard inference — comfortably inside Spark's envelope._
- [2605.00553 Stable-GFlowNet: Toward Diverse and Robust LLM Red-Teaming v](2605.00553/paper.md) · 21 · _Stable GFlowNet for diverse LLM red-team prompt generation — small-scale RL on small attacker LM, safety-arc material._
- [2605.01428 Hallucinations Undermine Trust; Metacognition is a Way Forwa](2605.01428/paper.md) · 20 · _Metacognitive-uncertainty layer over a hosted LLM — pure inference-time wrapper, easy NIM-side experiment._
- [2604.26779 Accelerating RL Post-Training Rollouts via System-Integrated](2604.26779/paper.md) · 14 · _Speculative decoding inside NeMo-RL with vLLM at 8B scale is exactly Spark-class — 8B target + small draft model fit comfortably in 128 GB._
- [2604.27251 Compliance versus Sensibility: On the Reasoning Controllabil](2604.27251/paper.md) · 13 · _Behavioral study of induction/deduction/abduction conflicts in LLMs is a pure-inference replication runnable against any NIM-hosted model._


### Autoresearch (20)

#### spark-feasible (16)
- [2604.27351 Heterogeneous Scientific Foundation Model Collaboration](2604.27351/paper.md) · 39 · _Lightweight LLM-orchestrator over domain foundation models is software glue that fits NemoClaw/NIM; underlying scientific FMs would be hosted as endpoints._ · [eval](2604.27351/eval.md) · → `articles/scientific-foundation-models-as-tools/`
- [2605.03042 ARIS: Autonomous Research via Adversarial Multi-Agent Collab](2605.03042/paper.md) · 33 · _Open-source autonomous-research harness with adversarial review — direct Autoresearch-arc material; runs over hosted NIM endpoints._
- [2604.26904 ClawGym: A Scalable Framework for Building Effective Claw Ag](2604.26904/paper.md) · 28 · _Claw-style sandboxed agent SFT + lightweight RL on per-task sandboxes maps directly onto NemoClaw + NeMo fine-tuning within the 128 GB envelope._ · [eval](2604.26904/eval.md) · → `articles/clawgym-on-spark/`
- [2605.04036 OpenSeeker-v2: Pushing the Limits of Search Agents with Info](2605.04036/paper.md) · 28 · _SFT-only recipe for deep-search agents at small scale — high-signal trajectories + LoRA on a 7B base fits the Spark._
- [2604.25256 AutoResearchBench: Benchmarking AI Agents on Complex Scienti](2604.25256/paper.md) · 24 · _Agent-driven literature discovery benchmark fits Autoresearch arc; runnable on Spark via NemoClaw + NIM + NeMo Retriever with pgvector, no training needed._ · [eval](2604.25256/eval.md) · → `articles/autoresearchbench-on-spark/`
- [2604.28139 Claw-Eval-Live: A Live Agent Benchmark for Evolving Real-Wor](2604.28139/paper.md) · 23 · _Live agent benchmark with execution traces and graders maps cleanly onto NemoClaw/OpenClaw sandboxed agents on Spark for local workflow eval._ · [eval](2604.28139/eval.md) · → `articles/claw-eval-live-on-spark/`
- [2604.28158 Intern-Atlas: A Methodological Evolution Graph as Research I](2604.28158/paper.md) · 20 · _Method-evolution graph extraction is an LLM-over-corpus pipeline that maps directly onto NIM + NeMo Retriever + pgvector on Spark, on a subset._
- [2605.02396 HeavySkill: Heavy Thinking as the Inner Skill in Agentic Har](2605.02396/paper.md) · 18 · _Internalizing 'heavy thinking' into the model rather than the harness — directly testable on Spark's NIM-hosted thinking models._
- [2604.27419 InteractWeb-Bench: Can Multimodal Agent Escape Blind Executi](2604.27419/paper.md) · 17 · _Multimodal agent benchmark with persona-driven instructions runs on Spark via NemoClaw + NIM-served MLLMs without training._
- [2604.28181 Synthetic Computers at Scale for Long-Horizon Productivity S](2604.28181/paper.md) · 17 · _Synthetic-computer long-horizon agent sims map directly onto OpenShell sandboxes inside NemoClaw, with NIM-hosted Nemotron driving the loop._
- [2604.27776 WindowsWorld: A Process-Centric Benchmark of Autonomous GUI ](2604.27776/paper.md) · 17 · _Cross-application Windows GUI benchmark — natural ClawGym/OpenClaw ablation target, runnable as Spark-side harness._
- [2604.27151 Step-level Optimization for Efficient Computer-use Agents](2604.27151/paper.md) · 16 · _Routing routine GUI steps to a small policy and reserving the big MLLM for high-risk steps is exactly the kind of single-Spark optimization the blog studies._
- [2604.25135 FAMA: Failure-Aware Meta-Agentic Framework for Open-Source L](2604.25135/paper.md) · 16 · _Failure-trajectory analysis + targeted helper agents over open-source LLMs runs cleanly atop NemoClaw with a Spark-hosted Nemotron._
- [2604.24658 The Last Human-Written Paper: Agent-Native Research Artifact](2604.24658/paper.md) · 15 · _Agent-native research artifact protocol is process + tooling — implementable as a NemoClaw skill atop NIM-hosted reasoning models on Spark._
- [2605.02178 T^2PO: Uncertainty-Guided Exploration Control for Stable Mul](2605.02178/paper.md) · 14 · _Uncertainty-guided exploration for multi-turn agentic RL — direct sequel to the GRPO-on-ClawGym arc._
- [2605.03596 Workspace-Bench 1.0: Benchmarking AI Agents on Workspace Tas](2605.03596/paper.md) · 9 · _Workspace-Bench: agents over real file-dependency graphs — close cousin of ClawGym, ready Spark harness._

#### borderline (4)
- [2604.26752 GLM-5V-Turbo: Toward a Native Foundation Model for Multimoda](2604.26752/paper.md) · 33 · _GLM-5V-Turbo is a frontier multimodal agent model; only borderline for Spark inference depending on released parameter count and quantization._
- [2605.00347 Odysseus: Scaling VLMs to 100+ Turn Decision-Making in Games](2605.00347/paper.md) · 19 · _100+ turn VLM RL on Mario — small-VLM fits, but 100+ turn rollouts × RL may push the unified-memory budget; verify in eval._
- [2605.02240 PhysicianBench: Evaluating LLM Agents in Real-World EHR Envi](2605.02240/paper.md) · 15 · _Long-horizon EHR agent benchmark — domain-specific but the harness shape mirrors ClawGym; runnable against a hosted NIM._
- [2605.02801 Reinforcement Learning for LLM-based Multi-Agent Systems thr](2605.02801/paper.md) · 9 · _RL over multi-agent orchestration traces — Spark can train one agent at a time, multi-agent rollouts may need careful memory choreography._


### Looking Beyond Spark (1)

#### borderline (1)
- [2604.27085 Efficient Training on Multiple Consumer GPUs with RoundPipe](2604.27085/paper.md) · 24 · _Multi-consumer-GPU pipeline-parallel scheduler doesn't run on Spark's single GB10, but the throughput math extrapolates cleanly into Looking Beyond Spark._


### Frontier Scout (5)

#### spark-feasible (1)
- [2605.02661 AcademiClaw: When Students Set Challenges for AI Agents](2605.02661/paper.md) · 17 · _Direct OpenClaw academic-benchmark — drop-in for the ClawGym arc, 80 long-horizon student-curated tasks._

#### borderline (4)
- [2604.26951 Turning the TIDE: Cross-Architecture Distillation for Diffus](2604.26951/paper.md) · 27 · _8B dense and 16B MoE teachers distilling into 0.6B student fits the 128 GB envelope, but full TIDE training pipeline is heavy and dLLM tooling on NeMo is unproven._
- [2604.27083 Co-Evolving Policy Distillation](2604.27083/paper.md) · 26 · _Co-evolving multiple experts in parallel during RLVR is memory-heavy; whether it fits the 128 GB unified pool depends on expert sizes._
- [2604.27505 Leveraging Verifier-Based Reinforcement Learning in Image Ed](2604.27505/paper.md) · 22 · _CoT reasoning verifier RM + RLHF for image editing is multi-stage and image-domain; whether it fits depends on backbone scale._
- [2604.25719 Step-Audio-R1.5 Technical Report](2604.25719/paper.md) · 19 · _Audio reasoning post-training fits NeMo's speech surface but RLVR-on-audio at scale is a non-trivial Spark workload._


## Stats

| Metric | Value |
|--------|------:|
| Total tracked | 36 |
| Classified this run | 15 |
| Dropped under threshold | 15 |
| spark-feasible | 27 |
| borderline | 9 |
| out-of-envelope | 0 |
| Deep-eval'd | 5 |
| Promoted to article | 5 |

## Run history

[Append-only refresh log →](runs/index.md)
