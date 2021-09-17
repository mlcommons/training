#!/bin/bash

function download_20m {
	echo "Download ml-20m"
	curl -O http://files.grouplens.org/datasets/movielens/ml-20m.zip
}

function download_1m {
	echo "Downloading ml-1m"
	curl -O http://files.grouplens.org/datasets/movielens/ml-1m.zip
}

DATA_DIR="${DATA_DIR:-./}"
pushd $DATA_DIR
curl -O https://files.grouplens.org/datasets/movielens/ml-20mx16x32.tar
tar  -xvf ml-20mx16x32.tar
if [[ $1 == "ml-1m" ]]
then
	download_1m
	FILE_NAME="ml-1m.zip"
else
	download_20m
	FILE_NAME="ml-20m.zip"
fi
popd

echo "Verifying:" $DATA_DIR/$FILE_NAME
bash ./verify_dataset.sh $1

echo "Uncompressing:" $DATA_DIR/$FILE_NAME
unzip -n $DATA_DIR/$FILE_NAME -d $DATA_DIR
