#!/bin/bash
#SBATCH -N 1
#SBATCH --gpus-per-node 1
#SBATCH -t 02:00:00
#SBATCH --mem=0

set -e

: "${CONT_IMAGE_URL:?CONT_IMAGE_URL not set}"
: "${SRC_PATH:?SRC_PATH not set}"
: "${DST_PATH:?DST_PATH not set}"

working_dir=$(dirname -- ${BASH_SOURCE[0]})

if [ ! -d $DST_PATH ]; then
    mkdir -p $DST_PATH
fi

container_maps="${SRC_PATH}:/source,${DST_PATH}:/destination,${working_dir}:/workspace/utils"

srun --nodes=1 --ntasks-per-node=1 \
--container-image=$CONT_IMAGE_URL --container-mounts $container_maps --no-container-entrypoint \
python3 /workspace/utils/convert.py --source /source --destination /destination
