#!/bin/bash
# runs benchmark and reports time to convergence
# to use the script:
#   run_and_time.sh <random seed> <threshold>

SEED=${1:-1}
THRESHOLD=${2:-0.212}
DATADIR=../coco

time stdbuf -o 0 \
  python3 train.py --data $DATADIR --seed $SEED --threshold $THRESHOLD | tee run.log.$SEED
