#!/bin/bash

input_file="/data/ml-20m/ratings.csv"
output_prefix=""
num_row_multiplier=16
num_col_multiplier=32

python run_expansion.py --input_csv_file=$input_file --output_prefix=$output_prefix --num_row_multiplier=$num_row_multiplier --num_col_multiplier=$num_col_multiplier
# python post_process.py --output_prefix=$output_prefix --num_shards=$num_row_multiplier
