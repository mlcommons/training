#!/bin/bash

data_dir=${DATA_DIR:-./}
wiki_dir=$data_dir/wiki/
results_dir=$data_dir/results/
tfrecord_dir=$data_dir/tfrecord/

mkdir -p $tfrecord_dir

echo "Processing train data"
# Generate one TFRecord for each results_dir/part-00XXX-of-00500 file.
for file in $results_dir/*
do
  if [[ $file == *"part"* ]]; then
    echo "Processing file: $file"
    python create_pretraining_data.py \
    --input_file=$file \
    --output_file=$tfrecord_dir/${file##*/} \
    --vocab_file=$wiki_dir/vocab.txt \
    --do_lower_case=True \
    --max_seq_length=512 \
    --max_predictions_per_seq=76 \
    --masked_lm_prob=0.15 \
    --random_seed=12345 \
    --dupe_factor=10
  fi
done

echo "Processing eval data"
python create_pretraining_data.py \
  --input_file=$results_dir/eval.txt \
  --output_file=$tfrecord_dir/eval_intermediate \
  --vocab_file=$wiki_dir/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=512 \
  --max_predictions_per_seq=76 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=10

python3 pick_eval_samples.py \
  --input_tfrecord=$tfrecord_dir/eval_intermediate \
  --output_tfrecord=$tfrecord_dir/eval_10k \
  --num_examples_to_pick=10000