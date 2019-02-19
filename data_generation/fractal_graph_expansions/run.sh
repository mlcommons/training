#!/bin/bash

blaze run -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both \
  third_party/tensorflow_models/mlperf/data_generation/fractal_graph_expansions:run_expansion -- \
  --alsologtostderr --input_csv_file=/cns/oi-d/home/sim-research/datasets/recommendations/ml-20m/ratings.csv \
  --num_row_multiplier=16 --num_col_multiplier=32 \
  --output_prefix=/cns/oi-d/home/sim-research/datasets/recommendations/ml-20m/16_32_correct --gfs_user=sim-research
