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

#enwiki-20200101-pages-articles-multistream.xml.bz2.md5sum
gdown https://drive.google.com/uc?id=14_A6gQ0NJ7Pay1X0xFq9rCKUuFJcKLF-

# enwiki-20200101-pages-articles-multistream.xml.bz2
gdown https://drive.google.com/uc?id=18K1rrNJ_0lSR9bsLaoP3PkQeSFO-9LE7

bzip2 -dv enwiki-20200101-pages-articles-multistream.xml.bz2

# Download TF-1 checkpoints
mkdir tf1_ckpt

cd tf1_ckpt

gdown https://drive.google.com/uc?id=1chiTBljF0Eh1U5pKs6ureVHgSbtU8OG_

gdown https://drive.google.com/uc?id=1Q47V3K3jFRkbJ2zGCrKkKk-n0fvMZsa0

gdown https://drive.google.com/uc?id=1vAcVmXSLsLeQ1q7gvHnQUSth5W_f_pwv

cd ..

# Download TF-2 checkpoints
mkdir tf2_ckpt

cd tf2_ckpt

gdown https://drive.google.com/uc?id=1pJhVkACK3p_7Uc-1pAzRaOXodNeeHZ7F

gdown https://drive.google.com/uc?id=1oVBgtSxkXC9rH2SXJv85RXR9-WrMPy-Q

cd ..

mkdir dataset

cd dataset

gdown https://drive.google.com/corp/drive/u/0/folders/1cywmDnAsrP5-2vsr8GDc6QUc7VWe-M3v -O ./ --folder

mv 'Processed dataset' processed_dataset

cd processed_dataset

tar -xvf results_text.tar.gz

unzip -u results_text.zip

# Back to bert/cleanup_scripts
cd ../../..
