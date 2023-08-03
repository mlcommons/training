#!/usr/bin/env bash

: "${OUTPUT_DIR:=/datasets/coco2014}"

while [ "$1" != "" ]; do
    case $1 in
        -o | --output-dir )     shift
                                OUTPUT_DIR=$1
                                ;;
    esac
    shift
done

mkdir -p ${OUTPUT_DIR}
wget -O ${OUTPUT_DIR}/val2014_30k.tsv -c "https://cloud.mlcommons.org/index.php/s/training_stable_diffusion/download?path=/datasets/coco2014&files=val2014_30k.tsv"
