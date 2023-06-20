#!/usr/bin/env bash

: "${NPROCS:=16}"
: "${NTHREADS:=64}"
: "${METADATA_DIR:=/datasets/laion2B-en-aesthetic/metadata}"
: "${OUTPUT_DIR:=/datasets/laion2B-en-aesthetic/webdataset}"

while [ "$1" != "" ]; do
    case $1 in
        -j | --processes )        shift
                                  NPROCS=$1
                                  ;;
        -t | --threads  )         shift
                                  NTHREADS=$1
                                  ;;
        -m | --metadata-dir )     shift
                                  METADATA_DIR=$1
                                  ;;
        -o | --output-dir )       shift
                                  OUTPUT_DIR=$1
                                  ;;
    esac
    shift
done

mkdir -p ${OUTPUT_DIR}

img2dataset \
  --url_list ${METADATA_DIR} \
  --input_format "parquet" \
  --url_col "URL" \
  --caption_col "TEXT" \
  --output_format webdataset \
  --output_folder ${OUTPUT_DIR} \
  --processes_count ${NPROCS} \
  --thread_count ${NTHREADS} \
  --incremental_mode "incremental" \
  --resize_mode "no" \
  --save_additional_columns '["similarity","hash","punsafe","pwatermark","aesthetic"]' \
  --enable_wandb False
