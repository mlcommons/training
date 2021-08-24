#!/bin/bash

data_dir=${DATA_DIR:-./cleanup_scripts}
output_dir=${OUTPUT_DIR:-/tmp/output/}
wiki_dir=$data_dir/wiki/
results_dir=$data_dir/results/
tfrecord_dir=$data_dir/tfrecord/

TF_XLA_FLAGS='--tf_xla_auto_jit=2' \
time python3 run_pretraining.py \
  --bert_config_file=$wiki_dir/bert_config.json \
  --output_dir=$output_dir \
  --input_file="${tfrecord_dir}/part*" \
  --do_train \
  --do_eval \
  --eval_batch_size=8 \
  --init_checkpoint=./checkpoint/model.ckpt-28252 \
  --iterations_per_loop=1000 \
  --learning_rate=0.0001 \
  --max_eval_steps=1250 \
  --max_predictions_per_seq=76 \
  --max_seq_length=512 \
  --num_gpus=1 \
  --num_train_steps=107538 \
  --num_warmup_steps=1562 \
  --optimizer=lamb \
  --save_checkpoints_steps=1562 \
  --start_warmup_step=0 \
  --train_batch_size=24 \
  --nouse_tpu