#!/usr/bin/env bash
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export NCCL_TIMEOUT=10800
export TORCH_DISTRIBUTED_TIMEOUT=10800
export TOKENIZERS_PARALLELISM=false

MASTER_PORT=${MASTER_PORT:-20500}

NODE_COUNT=${NODE_COUNT:-8}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
NPROC_PER_NODE=${PROC_PER_NODE:-8}

deepspeed=${NAV_DEEPSPEED_CONFIG:-"${REPO_ROOT}/scripts/zero3.json"}
llm=${LLM_PATH:-"xxx/Qwen3.5-9B/"}
entry_file="${REPO_ROOT}/qwenvl/train/train_qwen.py"

lr=5e-6
batch_size=${NAV_BATCH_SIZE:-2}

grad_accum_steps=${NAV_GRAD_ACCUM:-1}
empty_cache_steps=${NAV_EMPTY_CACHE_STEPS:-1}
data_seed=${NAV_DATA_SEED:-42}
export NAV_DATA_SAMPLE_FRACTION=${NAV_DATA_SAMPLE_FRACTION:-1}
data_flatten=${NAV_DATA_FLATTEN:-True}
data_packing=${NAV_DATA_PACKING:-False}

datasets=${NAV_DATASETS:-"r2r_va2l,srdf_va2l,srdf,scalevln,envdrop,r2r32,rxr32,vsi_590k_video,llava_3d_546k_scannet,mindcube"}

run_name=${NAV_RUN_NAME:-"qwen3_5_9b_nav_32gpu"}
output_dir=${NAV_OUTPUT_DIR:-"xx/evo-vln/outputs/navigation/data_ablation/final"}

args="
    --deepspeed ${deepspeed} \
    --model_name_or_path ${llm} \
    --dataset_use ${datasets} \
    --data_flatten ${data_flatten} \
    --data_packing ${data_packing} \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --max_pixels 262144 \
    --min_pixels 262144 \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size * 2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --video_max_frames 32 \
    --video_min_frames 32 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 10000 \
    --save_total_limit 4 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --torch_empty_cache_steps ${empty_cache_steps} \
    --group_by_length True \
    --seed ${data_seed} \
    --data_seed ${data_seed} \
    --model_max_length 16384 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --run_name ${run_name} \
    --report_to none"

torchrun \
    --nnodes="${NODE_COUNT}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    "${entry_file}" ${args}
