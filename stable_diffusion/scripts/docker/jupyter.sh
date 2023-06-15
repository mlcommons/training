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
    -v /datasets/laion2B-en-aesthetic/webdataset:/datasets/laion2B-en-aesthetic \
    -v /datasets/coco/coco2014:/datasets/coco2014 \
    -v /lfs/stable-diffusion/cache/huggingface:/root/.cache/huggingface \
    -v /lfs/stable-diffusion/results:/results \
    -v /lfs/stable-diffusion/ckpts:/ckpts \
    -e PYTHONPYCACHEPREFIX=/tmp/.pycache \
    -p 8888:8888 \
    ${DST_IMG} jupyter notebook
