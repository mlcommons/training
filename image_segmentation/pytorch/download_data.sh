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

git clone https://github.com/neheller/kits19
cd kits19
cp -r data/* $DATASET_PATH
rm -r data/
ln -s $DATASET_PATH data
pip install -r requirements.txt
python -m starter_code.get_imaging