#!/usr/bin/env bash

: "${INPUT_IMAGES_PATH:=/datasets/coco2014/val2014}"
: "${INPUT_COCO_CAPTIONS:=/datasets/coco2014/annotations/captions_val2014.json}"
: "${OUTPUT_IMAGES_PATH:=/datasets/coco2014/val2014_512x512_30k}"
: "${OUTPUT_TSV_FILE:=/datasets/coco2014/val2014_30k.tsv}"
: "${NUM_SAMPLES:=30000}"

while [ "$1" != "" ]; do
    case $1 in
        -i | --input-images-path )      shift
                                        INPUT_IMAGES_PATH=$1
                                        ;;
        -c | --input-coco-captions )    shift
                                        INPUT_COCO_CAPTIONS=$1
                                        ;;
        -i | --output-images-path )     shift
                                        OUTPUT_IMAGES_PATH=$1
                                        ;;
        -t | --output-tsv-file )        shift
                                        OUTPUT_TSV_FILE=$1
                                        ;;
        -n | --num-samples )            shift
                                        NUM_SAMPLES=$1
                                        ;;
    esac
    shift
done

python scripts/datasets/coco-split-resize.py \
    --input-images-dir ${INPUT_IMAGES_PATH} \
    --input-captions-file ${INPUT_COCO_CAPTIONS} \
    --output-images-dir ${OUTPUT_IMAGES_PATH} \
    --output-tsv-file ${OUTPUT_TSV_FILE} \
    --num-samples ${NUM_SAMPLES} \
    --seed 2023 \
    --width 512 \
    --height 512 \
    --num-workers 8
