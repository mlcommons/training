#!/bin/bash

set -e

SEED=$1

cd /research/transformer

# TODO: Add SEED to process_data.py since this uses a random generator (future PR)
export PYTHONPATH=/research/transformer/transformer:${PYTHONPATH}
# Add compliance to PYTHONPATH
# export PYTHONPATH=/mlperf/training/compliance:${PYTHONPATH}

mkdir -p /research/transformer/processed_data/
mkdir -p /research/transformer/processed_data/utf8

cp /research/transformer/transformer/vocab/vocab.translate_ende_wmt32k.32768.subwords /research/transformer/processed_data/vocab.ende.32768

python3 process_data.py --raw_dir /raw_data/ --data_dir processed_data && python3 convert_utf8_to_tfrecord.py --data_dir /research/transformer/processed_data/utf8
