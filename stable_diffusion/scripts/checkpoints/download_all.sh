#!/usr/bin/env bash

: "${OUTPUT_SD:=/checkpoints/sd}"
: "${OUTPUT_INCEPTION:=/checkpoints/inception}"
: "${OUTPUT_CLIP:=/checkpoints/clip}"

while [ $# -gt 0 ]; do
    case "$1" in
    --output_sd=*)
        OUTPUT_SD="${1#*=}"
        ;;
    --output_inception=*)
        OUTPUT_INCEPTION="${1#*=}"
        ;;
    --output_clip=*)
        OUTPUT_CLIP="${1#*=}"
        ;;
    *) ;;
    esac
    shift
done

cd "$(dirname "$0")"
bash download_sd.sh --output-dir $OUTPUT_SD
bash download_inception.sh --output-dir $OUTPUT_INCEPTION
bash download_clip.sh --output-dir $OUTPUT_CLIP
