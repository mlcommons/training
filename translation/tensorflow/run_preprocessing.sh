#!/bin/bash

set -e

SEED=$1

cd /research/transformer

# TODO: Add SEED to process_data.py since this uses a random generator (future PR)
export PYTHONPATH=/research/transformer/transformer:${PYTHONPATH}
# Add compliance to PYTHONPATH
# export PYTHONPATH=/mlperf/training/compliance:${PYTHONPATH}

python3 process_data.py --raw_dir /raw_data/ --data_dir processed_data
