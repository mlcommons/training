#!/bin/bash

# Script to import tensorflow bert checkpoint to onnx.
# You need to install tf2onnx with 'pip install tf2onnx==1.5.5
# 
# We first take the bert model and add placeholders as inputs, than freeze the model.
# If you'd want to train this model via onnx we assume the runtime will add the backprop part of the
# training graph.


if [ $# != 1 ]; then 
  echo "usage: ./to_onnx.sh bert-checkpoint-dir"
  exit 1
fi

BERT_BASE_DIR=$1
MODEL="bs64k_32k_ckpt_model.ckpt-7037"

python freeze_bert.py  \
  --config $BERT_BASE_DIR/bs64k_32k_ckpt_bert_config.json \
  --checkpoint $BERT_BASE_DIR/$MODEL

python -m tf2onnx.convert --opset 11 --fold_const --input $BERT_BASE_DIR/$MODEL.pb \
  --inputs input_ids:0,input_mask:0,input_type_ids:0 \
  --outputs final_encodes:0 \
  --output $BERT_BASE_DIR/$MODEL.onnx 

