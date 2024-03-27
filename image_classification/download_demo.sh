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

wget https://storage.googleapis.com/bert_tf_data/image_net/demo/tfdata.zip -P $DATASET_PATH
unzip $DATASET_PATH/tfdata.zip -d $DATASET_PATH
rm $DATASET_PATH/tfdata.zip
