#!/usr/bin/env bash

: "${OUTPUT_DIR:=/checkpoints/clip}"

while [ "$1" != "" ]; do
    case $1 in
        -o | --output-dir )       shift
                                  OUTPUT_DIR=$1
                                  ;;
    esac
    shift
done

CLIP_WEIGHTS_URL="https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/main/open_clip_pytorch_model.bin"
CLIP_WEIGHTS_SHA256="9a78ef8e8c73fd0df621682e7a8e8eb36c6916cb3c16b291a082ecd52ab79cc4"

CLIP_CONFIG_URL="https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/raw/main/open_clip_config.json"

wget -N -P ${OUTPUT_DIR} ${CLIP_WEIGHTS_URL}
wget -N -P ${OUTPUT_DIR} ${CLIP_CONFIG_URL}
echo "${CLIP_WEIGHTS_SHA256}  ${OUTPUT_DIR}/open_clip_pytorch_model.bin"                    | sha256sum -c
