#!/bin/bash
: "${DATA_ROOT_DIR:=./pytorch/datasets/coco}"

while [ $# -gt 0 ]; do
  case "$1" in
  --data_dir=*)
    DATA_ROOT_DIR="${1#*=}"
    ;;
  *) ;;
  esac
  shift
done

# Get COCO 2017 data sets
echo "Downloaading to folder: $DATA_ROOT_DIR"
mkdir -p $DATA_ROOT_DIR
pushd $DATA_ROOT_DIR

echo "Downloading coco_annotations_minival.tgz:"
curl -O https://dl.fbaipublicfiles.com/detectron/coco/coco_annotations_minival.tgz
echo "Extracting coco_annotations_minival.tgz:"
tar -xzf coco_annotations_minival.tgz &>/dev/null

echo "Downloading annotations_trainval2017.zip:"
curl -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip
echo "Extracting annotations_trainval2017.zip:"
n_files=$(unzip -l annotations_trainval2017.zip | grep .json | wc -l)
unzip annotations_trainval2017.zip | {
  I=-1
  while read; do printf "Progress: $((++I * 100 / $n_files))%%\r"; done
  echo ""
}

echo "Downloading val2017.zip:"
curl -O http://images.cocodataset.org/zips/val2017.zip
echo "Extracting val2017.zip:"
n_files=$(unzip -l val2017.zip | grep .jpg | wc -l)
unzip val2017.zip | {
  I=-1
  while read; do printf "Progress: $((++I * 100 / $n_files))%%\r"; done
  echo ""
}

echo "Downloading train2017.zip:"
curl -O http://images.cocodataset.org/zips/train2017.zip
echo "Extracting train2017.zip:"
n_files=$(unzip -l train2017.zip | grep .jpg | wc -l)
unzip train2017.zip | {
  I=-1
  while read; do printf "Progress: $((++I * 100 / $n_files))%%\r"; done
  echo ""
}

# MD5 verification
echo "Running MD5 verification ... this might take a while"
checkMD5() {
  if [ $(pv -f $1 | md5sum | cut -d' ' -f1) = $2 ]; then
    echo "$1 MD5 is valid"
  else
    echo "*ERROR* $1 MD5 is NOT valid"
  fi
}

echo "validating annotations_trainval2017.zip:"
checkMD5 "annotations_trainval2017.zip" "f4bbac642086de4f52a3fdda2de5fa2c"
echo "validating coco_annotations_minival.tgz:"
checkMD5 "coco_annotations_minival.tgz" "2d2b9d2283adb5e3b8d25eec88e65064"
echo "validating val2017.zip:"
checkMD5 "val2017.zip" "442b8da7639aecaf257c1dceb8ba8c80"
echo "validating train2017.zip:"
checkMD5 "train2017.zip" "cced6f7f71b7629ddf16f17bbcfab6b2"

popd
