#!/usr/bin/env bash

: "${DST_IMG:=mlperf_sd:22.12-py3}"

while [ "$1" != "" ]; do
    case $1 in
        -d | --dst-img  )       shift
                                DST_IMG=$1
                                ;;
    esac
    shift
done

docker run --rm -it --gpus=all --ipc=host \
    --workdir /pwd \
    -v ${PWD}:/pwd \
    -v /datasets/laion2B-en-aesthetic:/datasets/laion2B-en-aesthetic \
    -v /datasets/laion-400m:/datasets/laion-400m \
    -v /datasets/coco/coco2014:/datasets/coco2014 \
    -v /lfs/stable-diffusion/checkpoints:/checkpoints \
    -v /lfs/stable-diffusion/results:/results \
    -e PYTHONPYCACHEPREFIX=/tmp/.pycache \
    ${DST_IMG} bash
