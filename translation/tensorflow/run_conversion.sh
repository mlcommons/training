#!/bin/bash

set -e

SEED=$1

cd /research/transformer

# TODO: Add SEED to process_data.py since this uses a random generator (future PR)
export PYTHONPATH=/research/transformer/transformer:${PYTHONPATH}
# Add compliance to PYTHONPATH
# export PYTHONPATH=/mlperf/training/compliance:${PYTHONPATH}

python3 convert_utf8_to_tfrecord.py --data_dir /research/transformer/processed_data/utf8
