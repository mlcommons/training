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

wget -O demo_data.zip -c https://mlcube.mlcommons-storage.org/minibenchmarks/stable_diffusion.zip
unzip -o stable_diffusion.zip
rm stable_diffusion.zip
