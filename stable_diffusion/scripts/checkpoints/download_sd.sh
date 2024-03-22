#!/usr/bin/env bash

: "${OUTPUT_DIR:=/checkpoints/sd}"

while [ "$1" != "" ]; do
    case $1 in
        -o | --output-dir )       shift
                                  OUTPUT_DIR=$1
                                  ;;
    esac
    shift
done

SD_WEIGHTS_URL='https://huggingface.co/stabilityai/stable-diffusion-2-base/resolve/main/512-base-ema.ckpt'
SD_WEIGHTS_SHA256="d635794c1fedfdfa261e065370bea59c651fc9bfa65dc6d67ad29e11869a1824"

wget -N -P ${OUTPUT_DIR} ${SD_WEIGHTS_URL}
echo "${SD_WEIGHTS_SHA256}  ${OUTPUT_DIR}/512-base-ema.ckpt"                    | sha256sum -c
