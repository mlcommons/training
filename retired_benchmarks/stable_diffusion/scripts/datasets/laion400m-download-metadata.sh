#!/usr/bin/env bash

: "${OUTPUT_DIR:=/datasets/laion-400m/metadata}"

while [ "$1" != "" ]; do
    case $1 in
        -o | --output-dir )     shift
                                OUTPUT_DIR=$1
                                ;;
    esac
    shift
done

mkdir -p ${OUTPUT_DIR}

for i in {00000..00031}; do wget -N -P ${OUTPUT_DIR} https://the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-meta/part-$i-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet; done
