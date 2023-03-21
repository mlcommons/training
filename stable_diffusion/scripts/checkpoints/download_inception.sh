#!/usr/bin/env bash

: "${OUTPUT_DIR:=/checkpoints/inception}"

while [ "$1" != "" ]; do
    case $1 in
        -o | --output-dir )       shift
                                  OUTPUT_DIR=$1
                                  ;;
    esac
    shift
done

FID_WEIGHTS_URL='https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth'
FID_WEIGHTS_SHA1="bd836944fd6db519dfd8d924aa457f5b3c8357ff"

wget -N -P ${OUTPUT_DIR} ${FID_WEIGHTS_URL}
echo "${FID_WEIGHTS_SHA1}  ${OUTPUT_DIR}/pt_inception-2015-12-05-6726825d.pth"                    | sha1sum -c
