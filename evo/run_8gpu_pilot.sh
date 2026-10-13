#!/usr/bin/env bash
set -euo pipefail
export RSI_NAV_ROOT="${RSI_NAV_ROOT:-xxx/RSI-Nav}"
export PYTHONPATH="${RSI_NAV_ROOT}:${RSI_NAV_ROOT}/habitat-lab/habitat-lab:${RSI_NAV_ROOT}/habitat-lab/habitat-baselines:${RSI_NAV_ROOT}/eval/Nav:${PYTHONPATH:-}"
# 改bsz需要改candidate accepedtarget spatialqa数量 文件夹名称
#8-GPU single-node pilot for RSI-Nav evo.
# Rollout/annotation follows eval/Nav/r2r_multiworker.py: torchrun starts one process
# per GPU, and each process loads one full Qwen3/Qwen3.5-VL replica for both
# VA2L instruction annotation and online navigation verification.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/eval:${REPO_ROOT}/eval/Nav:${PYTHONPATH:-}"
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
export RSI_NAV_EVO_DISABLE_DIST_BARRIER="${RSI_NAV_EVO_DISABLE_DIST_BARRIER:-1}"

CHECKPOINT="${CHECKPOINT:-}"
EPISODES_JSONL="${EPISODES_JSONL:-${REPO_ROOT}/rollout/continuous_srdf/srdf_episodes_merged.jsonl}"
TRAJECTORY_JSONL="${TRAJECTORY_JSONL:-${REPO_ROOT}/rollout/continuous_srdf/srdf_annotations_merged.jsonl}"
HABITAT_CONFIG="${HABITAT_CONFIG:-${REPO_ROOT}/evo/vln_r2r.yaml}"
IMAGE_ROOT="${IMAGE_ROOT:-${REPO_ROOT}/data/trajectory_data/evo_traj}"
SCENE_ROOT="${SCENE_ROOT:-${REPO_ROOT}/data/scene}"
RUNS_DIR="${RUNS_DIR:-xx/evo-vln/evo_traj/65536update_211_full/evo_runs}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-xx/evo-vln/evo_traj/65536update_211_full/evo_ckpts}"

ACCEPTED_TARGET="${ACCEPTED_TARGET:-24576}"
MAX_CANDIDATES_PER_RANK="${MAX_CANDIDATES_PER_RANK:-256}"
SPATIAL_QA_PER_UPDATE="${SPATIAL_QA_PER_UPDATE:-8192}"
STOP_QA_PER_UPDATE="${STOP_QA_PER_UPDATE:-0}"
ROLLOUT_NNODES="${ROLLOUT_NNODES:-${NODE_COUNT:-1}}"
ROLLOUT_NODE_RANK="${ROLLOUT_NODE_RANK:-${NODE_RANK:-0}}"
ROLLOUT_MASTER_ADDR="${ROLLOUT_MASTER_ADDR:-${MASTER_ADDR:-127.0.0.1}}"
ROLLOUT_NPROC_PER_NODE="${ROLLOUT_NPROC_PER_NODE:-${PROC_PER_NODE:-8}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${ROLLOUT_NPROC_PER_NODE}}"
ROLLOUT_WORLD_SIZE="$((ROLLOUT_NNODES * ROLLOUT_NPROC_PER_NODE))"
CANDIDATE_STRIDE="${CANDIDATE_STRIDE:-$((ROLLOUT_WORLD_SIZE * MAX_CANDIDATES_PER_RANK))}"
MAX_STEPS="${MAX_STEPS:-300}"
VL2A_MIN_SR="${VL2A_MIN_SR:-1.0}"
VL2A_MIN_SPL="${VL2A_MIN_SPL:-0.4}"
VA2L_MIN_SR="${VA2L_MIN_SR:-1.0}"
VA2L_MIN_SPL="${VA2L_MIN_SPL:-0.4}"
METRIC_SUCCESS_DISTANCE="${METRIC_SUCCESS_DISTANCE:-3.0}"
MAX_ROLLOUT_BATCHES="${MAX_ROLLOUT_BATCHES:-1000000}"
TOTAL_SOURCE_CANDIDATES="${TOTAL_SOURCE_CANDIDATES:-2366929}"
SPATIAL_QA_DATASETS="${SPATIAL_QA_DATASETS:-}"
SPATIAL_QA_SEED="${SPATIAL_QA_SEED:-42}"
SPATIAL_QA_IMAGE_ROOT="${SPATIAL_QA_IMAGE_ROOT:-/}"
STOP_QA_DATASETS="${STOP_QA_DATASETS:-}"
STOP_QA_SEED="${STOP_QA_SEED:-42}"
STOP_QA_IMAGE_ROOT="${STOP_QA_IMAGE_ROOT:-/}"
SAVE_BATCH="${SAVE_BATCH:-10}"
MAX_TRAIN_UPDATES_PER_WINDOW="${MAX_TRAIN_UPDATES_PER_WINDOW:-2}"
LOG_PREFIX="${LOG_PREFIX:-${HOSTNAME:-RSI-Nav-evo}}"
MAX_ROUNDS="${MAX_ROUNDS:-1}"
MASTER_PORT="${MASTER_PORT:-20500}"
ENABLE_POST_UPDATE_EVAL="${ENABLE_POST_UPDATE_EVAL:-1}"
POST_UPDATE_EVAL_OUTPUT_ROOT="${POST_UPDATE_EVAL_OUTPUT_ROOT:-xx/evo-vln/evo_traj/65536update_211_full/evo_evals}"
POST_UPDATE_EVAL_CONFIG="${POST_UPDATE_EVAL_CONFIG:-${REPO_ROOT}/eval/Nav/config/vln_r2r.yaml}"
POST_UPDATE_EVAL_SPLIT="${POST_UPDATE_EVAL_SPLIT:-val_unseen}"
POST_UPDATE_EVAL_MASTER_PORT="${POST_UPDATE_EVAL_MASTER_PORT:-$((MASTER_PORT + 1))}"
POST_UPDATE_EVAL_MAX_STEPS="${POST_UPDATE_EVAL_MAX_STEPS:-300}"
POST_UPDATE_EVAL_MODEL_MAX_LENGTH="${POST_UPDATE_EVAL_MODEL_MAX_LENGTH:-16384}"
POST_UPDATE_EVAL_PIXELS="${POST_UPDATE_EVAL_PIXELS:-262144}"
POST_UPDATE_EVAL_NODE_WAIT_SECONDS="${POST_UPDATE_EVAL_NODE_WAIT_SECONDS:-86400}"
echo "[run_8gpu_pilot][rollout_dist] DISTRIBUTED_JOB=${DISTRIBUTED_JOB:-} ROLLOUT_NNODES=${ROLLOUT_NNODES} ROLLOUT_NODE_RANK=${ROLLOUT_NODE_RANK} ROLLOUT_MASTER_ADDR=${ROLLOUT_MASTER_ADDR} MASTER_PORT=${MASTER_PORT} ROLLOUT_NPROC_PER_NODE=${ROLLOUT_NPROC_PER_NODE} ROLLOUT_WORLD_SIZE=${ROLLOUT_WORLD_SIZE} CANDIDATE_STRIDE=${CANDIDATE_STRIDE}"
# Rollout uses pure data parallelism: every torchrun rank loads one full model
# replica, and evo_rollout.py shards candidates by global RANK/WORLD_SIZE.
ROLLOUT_COMMAND="torchrun --nnodes=${ROLLOUT_NNODES} --node_rank=${ROLLOUT_NODE_RANK} --master_addr=${ROLLOUT_MASTER_ADDR} --master_port ${MASTER_PORT} --nproc_per_node=${ROLLOUT_NPROC_PER_NODE} ${SCRIPT_DIR}/evo_rollout.py --worker-mode torchrun --model-path {current_ckpt} --checkpoint-generation {current_ckpt_generation} --episodes-jsonl {filtered_episodes_jsonl} --trajectory-jsonl {filtered_trajectory_jsonl} --habitat-config-path ${HABITAT_CONFIG} --runs-dir {runs_dir} --round-id {round_id} --batch-id {batch_id} --candidate-offset {candidate_offset} --retry-jsonl '{retry_jsonl}' --accepted-target ${ACCEPTED_TARGET} --image-root ${IMAGE_ROOT} --max-candidates-per-rank ${MAX_CANDIDATES_PER_RANK} --max-steps ${MAX_STEPS} --vl2a-min-sr ${VL2A_MIN_SR} --vl2a-min-spl ${VL2A_MIN_SPL} --va2l-min-sr ${VA2L_MIN_SR} --va2l-min-spl ${VA2L_MIN_SPL} --metric-success-distance ${METRIC_SUCCESS_DISTANCE}"

if [[ -n "${SCENE_ROOT}" ]]; then
  ROLLOUT_COMMAND="${ROLLOUT_COMMAND} --scene-root ${SCENE_ROOT}"
fi

POST_UPDATE_EVAL_COMMAND="${POST_UPDATE_EVAL_COMMAND:-}"
if [[ "${ENABLE_POST_UPDATE_EVAL}" == "1" && -z "${POST_UPDATE_EVAL_COMMAND}" ]]; then
  echo "[run_8gpu_pilot][post_update_eval] enabled=1 output_root=${POST_UPDATE_EVAL_OUTPUT_ROOT} split=${POST_UPDATE_EVAL_SPLIT} config=${POST_UPDATE_EVAL_CONFIG} master_port=${POST_UPDATE_EVAL_MASTER_PORT}"
  POST_UPDATE_EVAL_COMMAND="DISTRIBUTED_JOB=1 NODE_COUNT=${ROLLOUT_NNODES} NODE_RANK=${ROLLOUT_NODE_RANK} PROC_PER_NODE=${ROLLOUT_NPROC_PER_NODE} torchrun --nproc_per_node=${ROLLOUT_NPROC_PER_NODE} --master_port ${POST_UPDATE_EVAL_MASTER_PORT} ${SCRIPT_DIR}/r2r_single.py --model_path {current_ckpt} --habitat_config_path ${POST_UPDATE_EVAL_CONFIG} --eval_split ${POST_UPDATE_EVAL_SPLIT} --output_path {eval_output_dir} --model_max_length ${POST_UPDATE_EVAL_MODEL_MAX_LENGTH} --min_pixels ${POST_UPDATE_EVAL_PIXELS} --max_pixels ${POST_UPDATE_EVAL_PIXELS} --max_steps ${POST_UPDATE_EVAL_MAX_STEPS}"
fi

# Distributed training command. By default this script builds one for evo_train.py;
# set ENABLE_TRAIN=0 for rollout-only pilots, or override TRAIN_COMMAND manually.
# When set, evo_controller.py runs it after enough accepted rollout samples have
# been aggregated. The command should be a real training launcher, e.g. torchrun
# or DeepSpeed, and may use controller placeholders like {current_ckpt},
# {next_ckpt_tmp}, {accepted_vl2a_jsonl}, {accepted_va2l_jsonl}, {spatial_qa_jsonl}, {stop_qa_jsonl}.
# The accepted/spatial/stop placeholders resolve to the current train_update_xxxxxx files,
# produced from the pending queue after each rollout/eval batch.
# Use --logging_steps 1 in TRAIN_COMMAND for trainer loss lines. Self-evo
# updates current_ckpt after every rollout batch; SAVE_BATCH controls whether
# historical checkpoint snapshots are kept for later inspection.
ENABLE_TRAIN="${ENABLE_TRAIN:-1}"
TRAIN_COMMAND="${TRAIN_COMMAND:-}"
ROTATE_CHECKPOINT="${ROTATE_CHECKPOINT:-1}"

if [[ "${ENABLE_TRAIN}" == "1" && -z "${TRAIN_COMMAND}" ]]; then
  TRAIN_NNODES="${TRAIN_NNODES:-${NODE_COUNT:-1}}"
  TRAIN_NODE_RANK="${TRAIN_NODE_RANK:-${NODE_RANK:-0}}"
  TRAIN_MASTER_ADDR="${TRAIN_MASTER_ADDR:-${MASTER_ADDR:-127.0.0.1}}"
  TRAIN_MASTER_PORT="${TRAIN_MASTER_PORT:-${MASTER_PORT:-20500}}"
  TRAIN_NPROC_PER_NODE="${TRAIN_NPROC_PER_NODE:-${PROC_PER_NODE:-${NPROC_PER_NODE}}}"

  echo "[run_8gpu_pilot][train_dist] DISTRIBUTED_JOB=${DISTRIBUTED_JOB:-} NODE_COUNT=${NODE_COUNT:-} NODE_RANK=${NODE_RANK:-} MASTER_ADDR=${MASTER_ADDR:-} MASTER_PORT=${MASTER_PORT:-} PROC_PER_NODE=${PROC_PER_NODE:-} TRAIN_NNODES=${TRAIN_NNODES} TRAIN_NODE_RANK=${TRAIN_NODE_RANK} TRAIN_MASTER_ADDR=${TRAIN_MASTER_ADDR} TRAIN_MASTER_PORT=${TRAIN_MASTER_PORT} TRAIN_NPROC_PER_NODE=${TRAIN_NPROC_PER_NODE}"
  echo "[run_8gpu_pilot][train_lr] TRAIN_LR=${TRAIN_LR:-5e-7} TRAIN_WARMUP_RATIO=${TRAIN_WARMUP_RATIO:-0.0} TRAIN_LR_SCHEDULER=${TRAIN_LR_SCHEDULER:-constant}"

  TRAIN_DEEPSPEED_CONFIG="${TRAIN_DEEPSPEED_CONFIG:-${REPO_ROOT}/scripts/zero3.json}"
  TRAIN_LR="${TRAIN_LR:-5e-7}"
  TRAIN_WARMUP_RATIO="${TRAIN_WARMUP_RATIO:-0.0}"
  TRAIN_LR_SCHEDULER="${TRAIN_LR_SCHEDULER:-constant}"
  TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
  TRAIN_GRAD_ACCUM="${TRAIN_GRAD_ACCUM:-1}"
  TRAIN_EMPTY_CACHE_STEPS="${TRAIN_EMPTY_CACHE_STEPS:-1}"
  TRAIN_DATA_SEED="${TRAIN_DATA_SEED:-42}"
  export NAV_DATA_SAMPLE_FRACTION="${NAV_DATA_SAMPLE_FRACTION:-1}"
  TRAIN_DATA_FLATTEN="${TRAIN_DATA_FLATTEN:-True}"
  TRAIN_DATA_PACKING="${TRAIN_DATA_PACKING:-False}"
  TRAIN_RUN_NAME="${TRAIN_RUN_NAME:-qwen3_5_9b_nav_32gpu}"
  TRAIN_EPOCHS="${TRAIN_EPOCHS:-1}"
  TRAIN_MODEL_MAX_LENGTH="${TRAIN_MODEL_MAX_LENGTH:-16384}"
  TRAIN_MAX_PIXELS="${TRAIN_MAX_PIXELS:-262144}"
  TRAIN_VIDEO_FRAMES="${TRAIN_VIDEO_FRAMES:-32}"
  TRAIN_SAVE_STEPS="${TRAIN_SAVE_STEPS:-10000}"
  TRAIN_SAVE_TOTAL_LIMIT="${TRAIN_SAVE_TOTAL_LIMIT:-4}"
  TRAIN_DATALOADER_WORKERS="${TRAIN_DATALOADER_WORKERS:-2}"

  train_dataset_use="evo_nav,evo_va2l,evo_spatial"
  train_args="--evo_nav_jsonl {accepted_vl2a_jsonl}"
  train_args+=" --evo_va2l_jsonl {accepted_va2l_jsonl}"
  train_args+=" --evo_spatial_jsonl {spatial_qa_jsonl}"
  train_args+=" --evo_nav_image_root ${IMAGE_ROOT}"
  train_args+=" --evo_va2l_image_root ${IMAGE_ROOT}"
  train_args+=" --evo_spatial_image_root ${SPATIAL_QA_IMAGE_ROOT}"
  if [[ "${STOP_QA_PER_UPDATE}" != "0" ]]; then
    train_args+=" --evo_stop_jsonl {stop_qa_jsonl}"
    train_args+=" --evo_stop_image_root ${STOP_QA_IMAGE_ROOT}"
    train_dataset_use+=",evo_stop"
  fi
  train_args+=" --deepspeed ${TRAIN_DEEPSPEED_CONFIG}"
  train_args+=" --model_name_or_path {current_ckpt}"
  train_args+=" --dataset_use ${train_dataset_use}"
  train_args+=" --data_flatten ${TRAIN_DATA_FLATTEN}"
  train_args+=" --data_packing ${TRAIN_DATA_PACKING}"
  train_args+=" --tune_mm_vision False"
  train_args+=" --tune_mm_mlp True"
  train_args+=" --tune_mm_llm True"
  train_args+=" --max_pixels ${TRAIN_MAX_PIXELS}"
  train_args+=" --min_pixels ${TRAIN_MAX_PIXELS}"
  train_args+=" --bf16"
  train_args+=" --output_dir {next_ckpt_tmp}"
  train_args+=" --num_train_epochs ${TRAIN_EPOCHS}"
  train_args+=" --per_device_train_batch_size ${TRAIN_BATCH_SIZE}"
  train_args+=" --per_device_eval_batch_size $((TRAIN_BATCH_SIZE * 2))"
  train_args+=" --gradient_accumulation_steps ${TRAIN_GRAD_ACCUM}"
  train_args+=" --video_max_frames ${TRAIN_VIDEO_FRAMES}"
  train_args+=" --video_min_frames ${TRAIN_VIDEO_FRAMES}"
  train_args+=" --eval_strategy no"
  train_args+=" --save_strategy steps"
  train_args+=" --save_steps ${TRAIN_SAVE_STEPS}"
  train_args+=" --save_total_limit ${TRAIN_SAVE_TOTAL_LIMIT}"
  train_args+=" --learning_rate ${TRAIN_LR}"
  train_args+=" --weight_decay 0"
  train_args+=" --warmup_ratio ${TRAIN_WARMUP_RATIO}"
  train_args+=" --max_grad_norm 1"
  train_args+=" --lr_scheduler_type ${TRAIN_LR_SCHEDULER}"
  train_args+=" --logging_steps 1"
  train_args+=" --torch_empty_cache_steps ${TRAIN_EMPTY_CACHE_STEPS}"
  train_args+=" --group_by_length True"
  train_args+=" --seed ${TRAIN_DATA_SEED}"
  train_args+=" --data_seed ${TRAIN_DATA_SEED}"
  train_args+=" --model_max_length ${TRAIN_MODEL_MAX_LENGTH}"
  train_args+=" --gradient_checkpointing True"
  train_args+=" --dataloader_num_workers ${TRAIN_DATALOADER_WORKERS}"
  train_args+=" --run_name ${TRAIN_RUN_NAME}"
  train_args+=" --report_to none"

  TRAIN_COMMAND="torchrun \
    --nnodes=\"${TRAIN_NNODES}\" \
    --node_rank=\"${TRAIN_NODE_RANK}\" \
    --master_addr=\"${TRAIN_MASTER_ADDR}\" \
    --master_port=\"${TRAIN_MASTER_PORT}\" \
    --nproc_per_node=\"${TRAIN_NPROC_PER_NODE}\" \
    ${SCRIPT_DIR}/evo_train.py ${train_args}"
fi

CONTROLLER_EXTRA_ARGS=()
if [[ -n "${TRAIN_COMMAND}" ]]; then
  CONTROLLER_EXTRA_ARGS+=(--train-command "${TRAIN_COMMAND}")
fi
if [[ -n "${POST_UPDATE_EVAL_COMMAND}" ]]; then
  CONTROLLER_EXTRA_ARGS+=(--post-update-eval-command "${POST_UPDATE_EVAL_COMMAND}")
  CONTROLLER_EXTRA_ARGS+=(--post-update-eval-output-root "${POST_UPDATE_EVAL_OUTPUT_ROOT}")
  CONTROLLER_EXTRA_ARGS+=(--post-update-eval-node-wait-seconds "${POST_UPDATE_EVAL_NODE_WAIT_SECONDS}")
fi
if [[ "${ROTATE_CHECKPOINT}" == "1" ]]; then
  CONTROLLER_EXTRA_ARGS+=(--rotate-checkpoint)
fi

python "${SCRIPT_DIR}/evo_controller.py" \
  --runs-dir "${RUNS_DIR}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --initial-ckpt "${CHECKPOINT}" \
  --accepted-target "${ACCEPTED_TARGET}" \
  --source-episodes-jsonl "${EPISODES_JSONL}" \
  --source-trajectory-jsonl "${TRAJECTORY_JSONL}" \
  --spatial-qa-per-update "${SPATIAL_QA_PER_UPDATE}" \
  --spatial-qa-datasets "${SPATIAL_QA_DATASETS}" \
  --spatial-qa-seed "${SPATIAL_QA_SEED}" \
  --stop-qa-per-update "${STOP_QA_PER_UPDATE}" \
  --stop-qa-datasets "${STOP_QA_DATASETS}" \
  --stop-qa-seed "${STOP_QA_SEED}" \
  --save-batch "${SAVE_BATCH}" \
  --max-train-updates-per-window "${MAX_TRAIN_UPDATES_PER_WINDOW}" \
  --log-prefix "${LOG_PREFIX}" \
  --candidate-stride "${CANDIDATE_STRIDE}" \
  --total-source-candidates "${TOTAL_SOURCE_CANDIDATES}" \
  --expected-rollout-ranks "${ROLLOUT_WORLD_SIZE}" \
  --max-rollout-batches-per-round "${MAX_ROLLOUT_BATCHES}" \
  --max-rounds "${MAX_ROUNDS}" \
  --rollout-command "${ROLLOUT_COMMAND}" \
  "${CONTROLLER_EXTRA_ARGS[@]}"
