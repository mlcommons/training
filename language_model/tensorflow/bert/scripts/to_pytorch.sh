#!/bin/bash

# Script to import tensorflow bert checkpoint to pytorch/huggingface.
# See https://huggingface.co/transformers/converting_tensorflow_models.html for details.
# 

if [ $# != 1 ]; then 
  echo "usage: ./to_pytorch.sh bert-checkpoint"
  exit 1
fi

BERT_BASE_DIR=$1
MODEL="bs64k_32k_ckpt_model.ckpt-7037"

transformers-cli convert --model_type bert \
  --tf_checkpoint $BERT_BASE_DIR/$MODEL \
  --config $BERT_BASE_DIR/bs64k_32k_ckpt_bert_config.json \
  --pytorch_dump_output $BERT_BASE_DIR/pytorch_model.bin

