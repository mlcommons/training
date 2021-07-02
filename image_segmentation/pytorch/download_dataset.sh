#!/bin/bash
set -e

DATA_SET="kits19"
DATA_ROOT_DIR="${DATA_ROOT_DIR:-/raw-data-dir}"
DATA_DIR="${DATA_ROOT_DIR}/${DATA_SET}"
if [ ! -d "$DATA_DIR" ]
then
    mkdir $DATA_DIR
    git clone https://github.com/neheller/kits19 "$DATA_DIR"
    cd "$DATA_DIR"
    pip3 install -r requirements.txt
    python3 "${DATA_DIR}/starter_code/get_imaging.py"
else
    echo "Directory $DATA_DIR already exists."
fi
