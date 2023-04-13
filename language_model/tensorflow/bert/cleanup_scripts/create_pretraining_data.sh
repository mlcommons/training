#!/bin/bash

: "${INPUT_PATH:=/workspace/data/dataset/processed_dataset/results4}"
: "${VOCAB_PATH:=/workspace/data/vocab.txt}"
: "${OUTPUT_PATH:=/workspace/tf_data}"
: "${OUTPUT_EVAL_PATH:=/workspace/output_eval_data}"
: "${EVAL_TXT:=/workspace/data/dataset/processed_dataset/results4/eval.txt}"

while [ "$1" != "" ]; do
    case $1 in
    --input_path=*)
        INPUT_PATH="${1#*=}"
        ;;
    --vocab_path=*)
        VOCAB_PATH="${1#*=}"
        ;;
    --output_path=*)
        OUTPUT_PATH="${1#*=}"
        ;;
    --eval_txt=*)
        EVAL_TXT="${1#*=}"
        ;;
    --output_eval_path=*)
        OUTPUT_EVAL_PATH="${1#*=}"
        ;;
    esac
    shift
done

echo "INPUT_PATH:" $INPUT_PATH
echo "VOCAB_PATH:" $VOCAB_PATH
echo "OUTPUT_PATH:" $OUTPUT_PATH

cd cleanup_scripts

#for FILE in $INPUT_PATH/part*; do
#    echo "file: " $FILE
#    NEW_FILE="$(basename -- $FILE)"
#    echo "*Processing: " $NEW_FILE
#    python3 create_pretraining_data.py \
#        --input_file=$FILE \
#        --vocab_file=$VOCAB_PATH \
#        --output_file=$OUTPUT_PATH/$NEW_FILE \
#        --do_lower_case=True \
#        --max_seq_length=512 \
#        --max_predictions_per_seq=76 \
#        --masked_lm_prob=0.15 \
#        --random_seed=12345 \
#        --dupe_factor=10
#done

TEMP_FILE=$OUTPUT_EVAL_PATH/eval_temp
echo "AQUI"
echo $TEMP_FILE

python3 create_pretraining_data.py \
  --input_file=$EVAL_TXT \
  --output_file=$TEMP_FILE \
  --vocab_file=$VOCAB_PATH \
  --do_lower_case=True \
  --max_seq_length=512 \
  --max_predictions_per_seq=76 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=10

python3 pick_eval_samples.py \
  --input_tfrecord=$TEMP_FILE \
  --output_tfrecord=$OUTPUT_EVAL_PATH/eval_10k \
  --num_examples_to_pick=10000
