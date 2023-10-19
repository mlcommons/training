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

wget http://cs231n.stanford.edu/tiny-imagenet-200.zip -P $DATASET_PATH
unzip $DATASET_PATH/tiny-imagenet-200.zip -d $DATASET_PATH
rm $DATASET_PATH/tiny-imagenet-200.zip
