# Proposed Spark recipe

The repo is at `github.com/CuSO4-Chen/A-TGPO` and uses **verl** for RL. Reproduction path:

1. `git clone --depth 1 https://github.com/CuSO4-Chen/A-TGPO && cd A-TGPO`
2. Two conda envs as the README prescribes — one for the retriever (`pyserini` + `faiss-gpu=1.8.0`), one for training (`torch==2.6.0` + `flash-attn`). The `flash-attn` build needs CUDA 12.4; capability map confirms Spark ships CUDA 12.x in the NeMo / PyTorch containers, so this works inside `nvcr.io/nvidia/pytorch:25.x` (avoid the venv-trap from memory note `feedback_nvidia_container_uv_venv_trap`).
3. Stand up the local retriever: `python rag_server/download.py` then `bash rag_server/launch.sh`. Wiki-18 + e5_Flat fits the Spark NVMe budget (~50 GB).
4. Process datasets: `python data_process/hotpotqa_multihop_train.py` + `python data_process/multihop_test_merge.py` for multi-hop, plus the single-hop pair.
5. Run `bash ATGPO/scripts/ATGPO_multihop_qwen3_4B.sh`. The script's batch sizes will need a halve-or-quarter pass for single-GPU verl (the published 8×H20 schedule won't map 1:1), but the algorithm is the same.
6. Eval on the seven QA datasets the paper uses: HotpotQA, 2WikiMultihopQA, MuSiQue, Bamboogle (multi-hop) + NaturalQuestions, TriviaQA, PopQA (single-hop).

The IG forward + adaptive-clipping logic lives in `verl_atgpo/` and is the actual extractable abstraction — three small overrides on top of verl's GRPO loss.
