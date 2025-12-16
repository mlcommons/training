#!/bin/bash

: "${DATASET_PATH:=/datasets/open-images-v6}"

while [ "$1" != "" ]; do
    case $1 in
        -d | --dataset-path  )        shift
                                      DATASET_PATH=$1
                                      ;;
    esac
    shift
done

python fiftyone_openimages.py \
    --dataset-dir=${DATASET_PATH}
