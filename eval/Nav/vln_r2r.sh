#!/bin/bash
export RSI_NAV_ROOT="${RSI_NAV_ROOT:-xxx/RSI-Nav}"
export PYTHONPATH="${RSI_NAV_ROOT}:${RSI_NAV_ROOT}/habitat-lab/habitat-lab:${RSI_NAV_ROOT}/habitat-lab/habitat-baselines:${RSI_NAV_ROOT}/eval/Nav:${PYTHONPATH:-}"
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export ATTN_IMPLEMENTATION=flash_attention_2
MASTER_PORT=$((RANDOM % 101 + 20000))

# CHECKPOINT='xxx/RSI-Nav54.5'
CHECKPOINT='xx/evo-vln/weights/data_ablation/final-35023/'
OUTPUT_PATH='./results/r2r/evo/final-35023/'
echo "CHECKPOINT: ${CHECKPOINT}"
echo "OUTPUT_PATH: ${OUTPUT_PATH}"

torchrun --nproc_per_node=8 --master_port "$MASTER_PORT" "$RSI_NAV_ROOT/eval/Nav/r2r_multiworker.py" \
        --habitat_config_path "$RSI_NAV_ROOT/eval/Nav/config/vln_r2r.yaml" \
        --model_path "$CHECKPOINT" \
        --output_path "$OUTPUT_PATH"
