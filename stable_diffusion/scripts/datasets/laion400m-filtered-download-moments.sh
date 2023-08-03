#!/usr/bin/env bash

: "${OUTPUT_DIR:=/datasets/laion-400m/webdataset-moments-filtered}"

while [ "$1" != "" ]; do
    case $1 in
        -o | --output-dir )     shift
                                OUTPUT_DIR=$1
                                ;;
    esac
    shift
done

mkdir -p ${OUTPUT_DIR}
cd ${OUTPUT_DIR}


for i in {00000..00831}; do wget -O ${OUTPUT_DIR}/${i}.tar -c "https://cloud.mlcommons.org/index.php/s/training_stable_diffusion/download?path=/datasets/laion-400m/moments-webdataset-filtered&files=${i}.tar"; done

wget -O ${OUTPUT_DIR}/sha512sums.txt -c "https://cloud.mlcommons.org/index.php/s/training_stable_diffusion/download?path=/datasets/laion-400m/moments-webdataset-filtered&files=sha512sums.txt"

sha512sum --quiet -c sha512sums.txt
