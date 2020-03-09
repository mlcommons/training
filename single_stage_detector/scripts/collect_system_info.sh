#!/bin/sh
set -ex

mkdir -p ./results/closed/systems

facter --json > ./results/closed/systems/system_info.json
