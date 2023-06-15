#!/usr/bin/env bash

: "${SRC_IMG:=nvcr.io/nvidia/pytorch:22.12-py3}"
: "${DST_IMG:=mlperf_sd:22.12-py3}"

while [ "$1" != "" ]; do
    case $1 in
        -s | --src-img )        shift
                                SRC_IMG=$1
                                ;;
        -d | --dst-img  )       shift
                                DST_IMG=$1
                                ;;
    esac
    shift
done

docker build -f Dockerfile . --rm -t ${DST_IMG} --build-arg FROM_IMAGE_NAME=${SRC_IMG}
