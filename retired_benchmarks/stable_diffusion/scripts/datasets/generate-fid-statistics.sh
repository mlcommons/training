#!/usr/bin/env bash

: "${DATASET_DIR:=/datasets/coco2014/val2014_30k}"
: "${OUTPUT_FILE:=/datasets/coco2014/val2014_30k_stats.npz}"

while [ "$1" != "" ]; do
    case $1 in
        -d | --dataset-dir )      shift
                                  DATASET_DIR=$1
                                  ;;
        -o | --output-file  )     shift
                                  OUTPUT_FILE=$1
                                  ;;
    esac
    shift
done

python ldm/modules/fid/fid_score.py --save-stats ${DATASET_DIR} ${OUTPUT_FILE}
