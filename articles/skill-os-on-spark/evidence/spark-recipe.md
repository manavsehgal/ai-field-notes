# Proposed Spark recipe

No public code release found in the paper or trivial GitHub search — this is the dominant blocker (see below). Plausible Spark reconstruction once code lands (or as a from-scratch build):

1. Pull Qwen3-8B from NGC or HF: `huggingface-cli download Qwen/Qwen3-8B-Instruct`.
2. Stand up the executor as a NIM endpoint — capability map confirms NIM serves Qwen3-class models with paged-attention KV economics (see "NIM First Inference on DGX Spark" in the blog).
3. Build the SkillRepo as a flat directory of markdown files: `skills/<skill_name>.md` with YAML frontmatter (`name`, `usage`) + body (workflow, constraints). Retrieval: BM25 over the YAML+body via `rank_bm25` (no embedding model needed — directly mirrors the paper's choice and aligns with DCI-style "no vector index" thinking).
4. Wire the curator policy as a separate Qwen3-8B with a small action head emitting one of `insert_skill | update_skill | delete_skill` + the target file path; train with `verl` (paper's framework) or NeMo-Aligner GRPO. Capability map says fine-tuning ≤ 70B with LoRA is in-envelope; do LoRA on the curator.
5. Composite reward: task_outcome (judge model = Qwen3-32B served on a second NIM, or use the local NeMo Evaluator pattern from "RAG Eval — Ragas + NeMo Evaluator" in the blog) + λf · validity + λu · content_quality + λc · compression. Weights from the paper: λf=1.0, λu=0.1, λc=0.05.
6. Eval on **ALFWorld** subsets (Pick=35, Look=13, Clean=27, Heat=16, Cool=25, Pick2=24 — small enough to run in a few hours on Spark) before scaling to WebShop or DeepMath-103k.
