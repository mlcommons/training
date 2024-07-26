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

echo "Downloaading demo to folder: $DATA_ROOT_DIR"
mkdir -p $DATA_ROOT_DIR
pushd $DATA_ROOT_DIR

echo "Downloading annotations_trainval2017.zip:"
curl -O https://mlcube.mlcommons-storage.org/minibenchmarks/object_detection.zip
echo "Extracting demo_data.zip:"
unzip -o -q object_detection.zip
rm object_detection.zip
echo "Done!"

# MD5 verification
echo "Running MD5 verification ..."
checkMD5() {
  if [ $(pv -f $1 | md5sum | cut -d' ' -f1) = $2 ]; then
    echo "$1 MD5 is valid"
  else
    echo "*ERROR* $1 MD5 is NOT valid"
  fi
}

echo "validating demo_data.zip:"
checkMD5 "demo_data.zip" "1b50202a21b0d8c3235d0a6f39b6f40c"

popd
