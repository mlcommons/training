#!/usr/bin/env bash

: "${LAION_OUTPUT_DIR:=/datasets/laion-400m/webdataset-moments-filtered}"
: "${COCO_OUTPUT_DIR:=/datasets/coco2014}"

while [ $# -gt 0 ]; do
  case "$1" in
  --laion_output_dir=*)
    LAION_OUTPUT_DIR="${1#*=}"
    ;;
  --coco_output_dir=*)
    COCO_OUTPUT_DIR="${1#*=}"
    ;;
  *) ;;
  esac
  shift
done

mkdir -p ${LAION_OUTPUT_DIR}
cd ${LAION_OUTPUT_DIR}


for i in {00000..00003}; do wget -O ${LAION_OUTPUT_DIR}/${i}.tar -c "https://cloud.mlcommons.org/index.php/s/training_stable_diffusion/download?path=/datasets/laion-400m/moments-webdataset-filtered&files=${i}.tar"; done

wget -O ${LAION_OUTPUT_DIR}/sha512sums.txt -c "https://cloud.mlcommons.org/index.php/s/training_stable_diffusion/download?path=/datasets/laion-400m/moments-webdataset-filtered&files=sha512sums.txt"

sha512sum --quiet -c sha512sums.txt

mkdir -p ${COCO_OUTPUT_DIR}
wget -O ${COCO_OUTPUT_DIR}/val2014_30k.tsv -c "https://cloud.mlcommons.org/index.php/s/training_stable_diffusion/download?path=/datasets/coco2014&files=val2014_30k.tsv"
wget -O ${COCO_OUTPUT_DIR}/val2014_30k_stats.npz -c "https://cloud.mlcommons.org/index.php/s/training_stable_diffusion/download?path=/datasets/coco2014&files=val2014_30k_stats.npz"
