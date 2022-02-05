#!/usr/bin/env bash

## DL params
export BATCHSIZE=32
export NUMEPOCHS=${NUMEPOCHS:-8}
export DATASET_DIR="/datasets/open-images-v6-mlperf"
export EXTRA_PARAMS='--lr 0.0001 --output-dir=/results'

## System run parms
export DGXNNODES=1
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=08:00:00

## System config params
export DGXNGPU=8
export DGXSOCKETCORES=64
export DGXNSOCKET=2
export DGXHT=2  # HT is on is 2, HT off is 1
