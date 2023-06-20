#!/usr/bin/env bash

: "${OUTPUT_DIR:=/datasets/laion2B-en-aesthetic/metadata}"

while [ "$1" != "" ]; do
    case $1 in
        -o | --output-dir )     shift
                                OUTPUT_DIR=$1
                                ;;
    esac
    shift
done

mkdir -p ${OUTPUT_DIR}

for i in {00000..00127}; do wget -N -P ${OUTPUT_DIR} https://huggingface.co/datasets/laion/laion2B-en-aesthetic/resolve/main/part-$i-9230b837-b1e0-4254-8b88-ed2976e9cee9-c000.snappy.parquet; done
