#!/bin/bash

: "${INPUT_PATH:=/1/downloads/coco2017}"
: "${VOCAB_PATH:=/2/coco2017}"
: "${OUTPUT_PATH:=/3/coco2017}"

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
    esac
    shift
done

echo "INPUT_PATH:" $INPUT_PATH
echo "VOCAB_PATH:" $VOCAB_PATH
echo "OUTPUT_PATH:" $OUTPUT_PATH

cd cleanup_scripts

for FILE in $INPUT_PATH/part*; do
    echo "file: " $FILE
    NEW_FILE="$(basename -- $FILE)"
    echo "*Processing: " $NEW_FILE
    python3 create_pretraining_data.py \
    --input_file=$FILE \
    --vocab_file=$VOCAB_PATH \
    --output_file=$OUTPUT_PATH/$NEW_FILE \
    --do_lower_case=True \
    --max_seq_length=512 \
    --max_predictions_per_seq=76 \
    --masked_lm_prob=0.15 \
    --random_seed=12345 \
    --dupe_factor=10
done