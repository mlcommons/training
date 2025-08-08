#!/bin/bash

DATA_DIR="./igbh/full/processed"

# Capture MLCube parameter
while [ $# -gt 0 ]; do
    case "$1" in
    --data_dir=*)
        DATA_DIR="${1#*=}"
        ;;
    *) ;;
    esac
    shift
done

echo "Minified dataset download starting ..."
mkdir -p $DATA_DIR
cd $DATA_DIR

wget https://mlcube.mlcommons-storage.org/minibenchmarks/gnn.zip
unzip -o gnn.zip
rm gnn.zip
echo "completed!"