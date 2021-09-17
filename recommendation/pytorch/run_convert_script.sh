#!/bin/bash

DATA_DIR="${DATA_DIR:-./}"
DIR_NAME="ml-20mx16x32"
FULL_DATA_DIR=$DATA_DIR$DIR_NAME
echo "PATH" $FULL_DATA_DIR
python3 convert.py $FULL_DATA_DIR
