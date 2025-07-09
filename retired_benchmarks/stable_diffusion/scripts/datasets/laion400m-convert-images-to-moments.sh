#!/bin/bash

: "${INPUT_FOLDER:=/datasets/laion-400m/webdataset-filtered}"
: "${OUTPUT_FOLDER:=/datasets/laion-400m/webdataset-latents-filtered}"

while [ "$1" != "" ]; do
    case $1 in
        -i | --input-folder )       shift
                                    INPUT_FOLDER=$1
                                    ;;
        -o | --output-folder )      shift
                                    OUTPUT_FOLDER=$1
                                    ;;
    esac
    shift
done

mkdir -p ${OUTPUT_FOLDER}

# Loop over each tar file in the input directory
for tar_file in ${INPUT_FOLDER}/*.tar; do
    file_name=$(basename "$tar_file")
    base_name="${file_name%.*}"
    python webdataset_images2latents.py \
        --input-tar ${tar_file} \
        --output-tar ${OUTPUT_FOLDER}/${base_name}.tar \
        --config configs/train_512.yaml \
        --ckpt /checkpoints/sd/512-base-ema.ckpt
done
