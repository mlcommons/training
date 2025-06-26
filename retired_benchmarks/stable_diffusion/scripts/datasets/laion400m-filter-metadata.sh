#!/usr/bin/env bash

: "${INPUT_METADATA_DIR:=/datasets/laion-400m/metadata}"
: "${OUTPUT_METADATA_DIR:=/datasets/laion-400m/metadata-filtered}"

while [ "$1" != "" ]; do
    case $1 in
        -i | --input-metadata-dir )     shift
                                        INPUT_METADATA_DIR=$1
                                        ;;
        -o | --output-metadata-dir )    shift
                                        OUTPUT_METADATA_DIR=$1
                                        ;;
    esac
    shift
done

mkdir -p ${OUTPUT_METADATA_DIR}

python scripts/datasets/filter-metadata.py \
    --input-folder ${INPUT_METADATA_DIR} \
    --output-folder ${OUTPUT_METADATA_DIR}
