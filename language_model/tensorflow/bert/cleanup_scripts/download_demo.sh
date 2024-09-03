#!/bin/bash

# pip install --user gdown

DATA_DIR="./wiki"

# Capture MLCube parameter
while [ "$1" != "" ]; do
    case $1 in
    --data_dir=*)
        DATA_DIR="${1#*=}"
        ;;
    esac
    shift
done

mkdir -p $DATA_DIR

cd $DATA_DIR

# Downloading files from Google Drive location: https://drive.google.com/drive/folders/1oQF4diVHNPCclykwdvQJw8n_VIWwV0PT

# bert_config.json
gdown https://drive.google.com/uc?id=1fbGClQMi2CoMv7fwrwTC5YYPooQBdcFW

#License.txt
gdown https://drive.google.com/uc?id=1SYfj3zsFPvXwo4nUVkAS54JVczBFLCWI

# vocab.txt
gdown https://drive.google.com/uc?id=1USK108J6hMM_d27xCHi738qBL8_BT1u1

# Download TF-2 checkpoints
mkdir tf2_ckpt

cd tf2_ckpt

gdown https://drive.google.com/uc?id=1pJhVkACK3p_7Uc-1pAzRaOXodNeeHZ7F

gdown https://drive.google.com/uc?id=1oVBgtSxkXC9rH2SXJv85RXR9-WrMPy-Q

cd ..

# Download dummy data in TFRecord format
wget https://mlcube.mlcommons-storage.org/minibenchmarks/bert.zip
unzip bert.zip
rm bert.zip
