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
      cd xxx/RSI-Nav/evo
      bash mount.sh
      bash run_8gpu_pilot.sh'