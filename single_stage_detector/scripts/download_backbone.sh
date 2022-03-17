#!/usr/bin/env bash

DOWNLOAD_LINK='https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth'
SHA512='15c9f0bc1c8d64750712f86ffaded3b0bc6a87e77a395dcda3013d8af65b7ebf3ca1c24dd3aae60c0d83e510b4d27731f0526b6f9392c0a85ffc18e5fecd8a13'
FILENAME='resnext50_32x4d-7cdf4587.pth'

wget -c $DOWNLOAD_LINK
echo "${SHA512}  ./${FILENAME}" | sha512sum -c
