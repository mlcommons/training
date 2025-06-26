#!/usr/bin/env bash

: "${OUTPUT_DIR:=/datasets/laion-400m/webdataset-filtered}"

while [ "$1" != "" ]; do
    case $1 in
        -o | --output-dir )     shift
                                OUTPUT_DIR=$1
                                ;;
    esac
    shift
done

mkdir -p ${OUTPUT_DIR}
cd ${OUTPUT_DIR}


rclone config create mlc-training s3 provider=Cloudflare access_key_id=76ea42eadb867e854061a1806220ee1e secret_access_key=a53625c4d45e3ca8ac0df8a353ea3a41ffc3292aa25259addd8b7dc5a6ce2936 endpoint=c2686074cb2caf5cbaf6d134bdba8b47.r2.cloudflarestorage.com

rclone copy mlc-training:mlcommons-training-wg-public/stable_diffusion/datasets/laion-400m/images-webdataset-filtered/ ${OUTPUT_DIR} --include="*.tar" -P

rclone copy mlc-training:mlcommons-training-wg-public/stable_diffusion/datasets/laion-400m/images-webdataset-filtered/sha512sums.txt ${OUTPUT_DIR} -P

sha512sum --quiet -c sha512sums.txt
