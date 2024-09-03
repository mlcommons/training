#!/bin/bash

: "${DATASET_PATH:=/}"

while [ "$1" != "" ]; do
    case $1 in
    --data_dir=*)
        DATASET_PATH="${1#*=}"
        ;;
    esac
    shift
done

wget https://mlcube.mlcommons-storage.org/minibenchmarks/image_classification.zip -P $DATASET_PATH
unzip $DATASET_PATH/image_classification.zip -d $DATASET_PATH
rm $DATASET_PATH/image_classification.zip
