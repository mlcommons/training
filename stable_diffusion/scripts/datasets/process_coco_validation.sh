#!/usr/bin/env bash

python scripts/datasets/process_coco_validation.py \
    --input-images-dir /datasets/coco2014/val2014/ \
    --input-captions-file /datasets/coco2014/annotations/captions_val2014.json \
    --output-images-dir /datasets/coco2014/val2014_512x512_30k/ \
    --output-tsv-file /datasets/coco2014/val2014_30k.tsv \
    --num-samples 30000 \
    --seed 2023 \
    --width 512 \
    --height 512 \
    --num-workers 8
