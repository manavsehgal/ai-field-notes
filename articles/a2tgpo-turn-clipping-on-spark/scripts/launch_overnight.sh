#!/bin/bash
# A²TGPO overnight run launcher — Spark single-GPU adaptation
#
# Scope: articles/a2tgpo-turn-clipping-on-spark/evidence/runs/2026-05-11-overnight/scope.md
# Run from inside the a2tgpo-spark container (or `docker exec a2tgpo-spark bash -c '...'`).
# Adapts ATGPO/scripts/ATGPO_multihop_qwen3_4B.sh for 1×GB10 (vs 8×H20).

set -euo pipefail

# ============================ Paths ============================
A2TGPO_REPO=/a2tgpo/evidence/repo-snapshot/ATGPO
VERL_PATH=${A2TGPO_REPO}/verl_atgpo
DATA_DIR=/work/data/hotpotqa-multihop
MODEL_PATH=/work/models/Qwen3-4B
RUN_DIR=/a2tgpo/evidence/runs/2026-05-11-overnight
RUN_DIR_OUTPUTS=/work/runs/2026-05-11-overnight    # large artifacts; gitignored
LINEAGE_DIR=${RUN_DIR}/lineage

mkdir -p ${RUN_DIR_OUTPUTS} ${LINEAGE_DIR}

# ============================ Environment ============================
export HYDRA_FULL_ERROR=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VERL_LOGGING_LEVEL=WARN
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU
export RAY_memory_usage_threshold=0.85
export RAY_memory_monitor_refresh_ms=0
export PYTHONPATH=${VERL_PATH}:/fieldkit/src:${PYTHONPATH:-}
export WANDB_MODE=disabled

# ============================ Experiment ============================
PROJECT_NAME="agentic_rl_search"
EXPERIMENT_NAME="a2tgpo-v1d-hotpotqa-spark"
CONFIG_PATH="${A2TGPO_REPO}/scripts/config"
CONFIG_NAME="ppo_trainer_dr.yaml"

# 1×GB10 vs paper's 8×H20
NNODES=1
N_GPUS_PER_NODE=1

# Quartered batches (paper 64/8/16 → 16/2/8)
TRAIN_BATCH_SIZE=16
PPO_MINI_BATCH_SIZE=2
ROLLOUT_N=8
BEAM_SIZE=2

# Context (paper-identical)
MAX_PROMPT_LENGTH=2000
MAX_RESPONSE_LENGTH=6192

# Data
PROMPT_KEY="prompt"
TRAIN_FILES="${DATA_DIR}/train.parquet"
VALID_FILES="[${DATA_DIR}/test-200.parquet]"   # tractable val cost on single GPU; full test.parquet is 7405 ex → multi-day eval

# A²TGPO algorithm knobs (paper's best config)
ENABLE_DYNAMIC_ROLLOUTS=False
ENABLE_POLICY_LOSS_GSPO=False
ENABLE_POLICY_LOSS_GSPO_TURN=True
CLIP_RATIO_LOW=3e-3
CLIP_RATIO_HIGH=4e-3
DYNAMIC_TURN_CLIP=True
IGPO_GAMMA=1.0
IGPO_NORM_MODE="turn-group-v1d"
IGPO_ADV_RESCALE_ALPHA=0.3

# Rollout
ROLLOUT_NAME="vllm"
ROLLOUT_MODE="sync_with_tool"
ENABLE_MULTI_TURN=False

# Reward
REWARD_MANAGER="naive"
CUSTOM_REWARD_FUNCTION_PATH="${VERL_PATH}/verl/utils/reward_score/deep_research_em.py"
CUSTOM_REWARD_FUNCTION_NAME="compute_score"

# Search tool (misnamed BingSearchTool — actually local faiss at 127.0.0.1:5003/retrieve)
SEARCH_CACHE_PATH="${RUN_DIR_OUTPUTS}/search_cache.json"
API_KEY="local-faiss-not-used"

# Training schedule
TOTAL_EPOCHS=1                # we time-box, not epoch-box
SAVE_FREQ=60
TEST_FREQ=20

# Output dirs
SAVE_PATH=${RUN_DIR_OUTPUTS}
ROLLOUT_SAVE_PATH="${SAVE_PATH}/rollout"
VALIDATION_SAVE_PATH="${SAVE_PATH}/validation"
mkdir -p ${ROLLOUT_SAVE_PATH} ${VALIDATION_SAVE_PATH}

# ============================ 8h time-box ============================
# verl doesn't have a built-in wall-clock limit. Wrap with `timeout` so the run
# stops cleanly at 8h regardless of where in the step loop we are. The lineage
# row is then written by the post-run harness reading the final checkpoint.
#
# Override for smoke runs: WALL_LIMIT_SECONDS=600 bash launch_overnight.sh
WALL_LIMIT_SECONDS=${WALL_LIMIT_SECONDS:-$((8 * 3600))}

# ============================ Launch ============================
echo "==> A²TGPO overnight run kickoff at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "==> Wall limit: ${WALL_LIMIT_SECONDS}s ($((WALL_LIMIT_SECONDS / 3600))h)"
echo "==> Run dir: ${RUN_DIR_OUTPUTS}"
echo "==> Lineage: ${LINEAGE_DIR}"

cd ${A2TGPO_REPO}

# Note: verl's main_ppo reads N_GPUS_PER_NODE from the trainer config; the
# rollout's vLLM engine reads tensor_model_parallel_size separately. Both set
# to 1 below.

timeout --signal=SIGTERM --kill-after=300s ${WALL_LIMIT_SECONDS}s \
python3 -m verl.trainer.main_ppo \
    --config-path=${CONFIG_PATH} \
    --config-name=${CONFIG_NAME} \
    algorithm.adv_estimator=igpo \
    algorithm.kl_ctrl.kl_coef=0 \
    data.train_files=${TRAIN_FILES} \
    data.val_files=${VALID_FILES} \
    data.prompt_key=${PROMPT_KEY} \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.enable_policy_loss_gspo_turn=${ENABLE_POLICY_LOSS_GSPO_TURN} \
    actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW} \
    actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH} \
    actor_rollout_ref.actor.dynamic_turn_clip=${DYNAMIC_TURN_CLIP} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((2*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.enable_dynamic_rollouts=${ENABLE_DYNAMIC_ROLLOUTS} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((4*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=${ROLLOUT_NAME} \
    actor_rollout_ref.rollout.mode=${ROLLOUT_MODE} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.initial_rollouts=${ROLLOUT_N} \
    actor_rollout_ref.rollout.beam_size=${BEAM_SIZE} \
    ++actor_rollout_ref.rollout.tools.tool_instances.search.params.cache_file=${SEARCH_CACHE_PATH} \
    ++actor_rollout_ref.rollout.tools.tool_instances.search.params.api_key=${API_KEY} \
    actor_rollout_ref.rollout.multi_turn.enable=${ENABLE_MULTI_TURN} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((4*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager=${REWARD_MANAGER} \
    custom_reward_function.path=${CUSTOM_REWARD_FUNCTION_PATH} \
    custom_reward_function.name=${CUSTOM_REWARD_FUNCTION_NAME} \
    trainer.critic_warmup=0 \
    trainer.balance_batch=False \
    trainer.logger="[console]" \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.default_local_dir=${SAVE_PATH} \
    trainer.val_before_train=True \
    trainer.rollout_data_dir=${ROLLOUT_SAVE_PATH} \
    trainer.validation_data_dir=${VALIDATION_SAVE_PATH} \
    algorithm.igpo_gamma=${IGPO_GAMMA} \
    algorithm.igpo_norm_mode=${IGPO_NORM_MODE} \
    algorithm.igpo_adv_rescale_alpha=${IGPO_ADV_RESCALE_ALPHA} \
    hydra.run.dir=${SAVE_PATH}/outputs 2>&1 | tee ${SAVE_PATH}/run.log

TRAIN_EXIT=$?
echo "==> Training exited with code ${TRAIN_EXIT} at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "${TRAIN_EXIT}" > ${RUN_DIR}/train_exit_code

# 124 = timeout fired (= train_budget_overrun)
# 0   = clean termination (= keep if EM gain ≥ 0.5pp, else discard)
# *   = crash (= crash)
