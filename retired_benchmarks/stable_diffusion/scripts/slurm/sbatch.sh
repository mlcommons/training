#!/bin/bash

: "${NUM_NODES:=8}"
: "${GPUS_PER_NODE:=8}"
: "${WALLTIME:=04:00:00}"
: "${CONFIG:=./configs/train_512_latents.yaml}"
: "${BASE_LOG_DIR:=./nogit/logs}"
: "${BASE_RESULTS_DIR:=/results}"
: "${JOB_NAME:=job}"
: "${CONTAINER_IMAGE:=mlperf_sd:22.12-py3}"
: "${CHECKPOINT:=/checkpoints/sd/512-base-ema.ckpt}"
: "${ACCOUNT:=account}"
: "${PARTITION:=partition}"

while [ "$1" != "" ]; do
    case $1 in
        -a | --account )        shift
                                ACCOUNT=$1
                                ;;
        -p | --partition )      shift
                                PARTITION=$1
                                ;;
        -n | --num-nodes )      shift
                                NUM_NODES=$1
                                ;;
        -g | --gpus-per-node )  shift
                                GPUS_PER_NODE=$1
                                ;;
        -t | --walltime )       shift
                                WALLTIME=$1
                                ;;
        -c | --config )         shift
                                CONFIG=$1
                                ;;
        -k | --checkpoint )     shift
                                CHECKPOINT=$1
                                ;;
        -l | --log-dir )        shift
                                BASE_LOG_DIR=$1
                                ;;
        -r | --results-dir )    shift
                                BASE_RESULTS_DIR=$1
                                ;;
        -d | --container )      shift
                                CONTAINER_IMAGE=$1
                                ;;
    esac
    shift
done

# Misc
SUFFIX=`date +%s`
CONFIG_NAME=`basename ${CONFIG} .yaml`
WORKDIR_MNT=/workdir

# Job config
JOB_NAME=train_${CONFIG_NAME}_${SUFFIX}

# Laion 400m
LAION_400M=/datasets/laion-400m
LAION_400M_MOUNT=/datasets/laion-400m

# COCO
COCO=/datasets/coco2014
COCO_MNT=/datasets/coco2014

# checkpoints
CKPT_DIR=/checkpoints
CKPT_MOUNT=/checkpoints

# Hugging face home
HF_HOME_DIR=/hf_home
HF_HOME_MOUNT=/hf_home

# exp
RESULTS_DIR=${BASE_RESULTS_DIR} # no need to append job name, pytorch appends datetime automatically
RESULTS_MNT=/results
mkdir -p ${RESULTS_DIR}

# logdir
LOG_DIR="${BASE_LOG_DIR}"
mkdir -p ${LOG_DIR}

# Mounts
MOUNTS="${PWD}:${WORKDIR_MNT},${LAION_400M}:${LAION_400M_MOUNT},${COCO}:${COCO_MNT},${RESULTS_DIR}:${RESULTS_MNT},${CKPT_DIR}:${CKPT_MOUNT},${HF_HOME_DIR}:${HF_HOME_MOUNT}"

sbatch \
    --account=${ACCOUNT} \
    --partition=${PARTITION} \
    --job-name="mlperf-ssd:${JOB_NAME}" \
    --nodes="${NUM_NODES}" \
    --ntasks-per-node="${GPUS_PER_NODE}" \
    --time="${WALLTIME}" \
    --output="${LOG_DIR}/%A_${JOB_NAME}.out" \
    ./scripts/slurm/srun.sh \
        --num-nodes ${NUM_NODES} \
        --gpus-per-node ${GPUS_PER_NODE} \
        --config ${CONFIG} \
        --workdir ${WORKDIR_MNT} \
        --results-dir ${RESULTS_MNT} \
        --mounts ${MOUNTS} \
        --container ${CONTAINER_IMAGE} \
        --checkpoint ${CHECKPOINT}
