#!/bin/sh
set -ex

cd ./data/coco

if [ -f train2017.zip ]; then
ACTUAL_TEST=`cat train2017.zip | md5sum`
EXPECTED_TEST='cced6f7f71b7629ddf16f17bbcfab6b2  -'
if [[ "$ACTUAL_TEST" = "$EXPECTED_TEST" ]]; then
  echo "OK: correct train2017.zip"
else
  echo "ERROR: incorrect train2017.zip"
  echo "ERROR: expected $EXPECTED_TEST"
  echo "ERROR: found $ACTUAL_TEST"
fi
fi

if [ -f val2017.zip ]; then
ACTUAL_TEST=`cat val2017.zip | md5sum`
EXPECTED_TEST='442b8da7639aecaf257c1dceb8ba8c80  -'
if [[ "$ACTUAL_TEST" = "$EXPECTED_TEST" ]]; then
  echo "OK: correct val2017.zip"
else
  echo "ERROR: incorrect val2017.zip"
  echo "ERROR: expected $EXPECTED_TEST"
  echo "ERROR: found $ACTUAL_TEST"
fi
fi

if [ -f annotations_trainval2017.zip ]; then
ACTUAL_TEST=`cat annotations_trainval2017.zip | md5sum`
EXPECTED_TEST='f4bbac642086de4f52a3fdda2de5fa2c  -'
if [[ "$ACTUAL_TEST" = "$EXPECTED_TEST" ]]; then
  echo "OK: correct annotations_trainval2017.zip"
else
  echo "ERROR: incorrect annotations_trainval2017.zip"
  echo "ERROR: expected $EXPECTED_TEST"
  echo "ERROR: found $ACTUAL_TEST"
fi
fi
