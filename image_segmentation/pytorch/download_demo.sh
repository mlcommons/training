#!/bin/bash
set -e

: "${DATASET_PATH:=/}"

while [ "$1" != "" ]; do
    case $1 in
    --data_dir=*)
        DATASET_PATH="${1#*=}"
        ;;
    esac
    shift
done

wget https://mlcube.mlcommons-storage.org/minibenchmarks/3d_unet.zip
unzip -o 3d_unet.zip -d $DATASET_PATH
rm 3d_unet.zip