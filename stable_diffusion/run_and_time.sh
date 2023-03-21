#!/bin/bash
set -e

HF_DATASETS_OFFLINE=1
TRANSFORMERS_OFFLINE=1
DIFFUSERS_OFFLINE=1

start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

# CLEAR YOUR CACHE HERE
  python -c "
from mlperf_logging.mllog import constants
from runtime.logging import mllog_event
mllog_event(key=constants.CACHE_CLEAR, value=True)"

# TODO(ahmadki): add validation threshold, data folder and number of GPUs as params
python main.py \
    -m train \
    --ckpt /checkpoints/sd/512-base-ema.ckpt \
    -b ./configs/train_512.yaml

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# runtime
runtime=$(( $end - $start ))
result_name="stable_diffusion"

echo "RESULT,$result_name,$runtime,$USER,$start_fmt"
