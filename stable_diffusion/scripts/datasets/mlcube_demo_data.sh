#!/usr/bin/env bash

: "${DEMO_OUTPUT_DIR:=/demo_data}"

while [ $# -gt 0 ]; do
  case "$1" in
  --demo_output_dir=*)
    DEMO_OUTPUT_DIR="${1#*=}"
    ;;
  *) ;;
  esac
  shift
done

mkdir -p ${DEMO_OUTPUT_DIR}
cd ${DEMO_OUTPUT_DIR}

wget -O demo_data.zip -c https://storage.googleapis.com/mlperf_training_demo/stable_diffusion/demo_data.zip
unzip -o demo_data.zip
rm demo_data.zip