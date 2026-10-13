#!/usr/bin/env bash
set -euo pipefail

rjob submit \
    --name=xx \
    --gpu=8 \
    --memory=1000000 \
    --cpu=100 \
    --private-machine=group \
    --charged-group=xx_gpu \
    --image=xxx:habitat0.3.3_qwen3.5_new \
    --mount=gpfs://xxx:xxx \
    --custom-resources brainpp.cn/fuse=1 \
    -P 16 \
    --host-network=true \
    -e DISTRIBUTED_JOB=true \
    -e MASTER_PORT=20500 \
    -- bash -c '
      set -e
      export RSI_NAV_ROOT=xxx/RSI-Nav
      cd "$RSI_NAV_ROOT/evo"
      bash mount.sh
      cd "$RSI_NAV_ROOT"
      bash scripts/navigation/qwen3_5_9b_nav_32.sh'
