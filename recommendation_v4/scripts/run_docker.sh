#!/bin/bash
# Launch a yambda_8gpu container from rocm/mlperf:dlrm_v3_mi355 with the repo
# and data directories bind-mounted at matching host/container paths.
#
# Usage:
#   bash scripts/run_docker.sh                                      # interactive shell
#   bash scripts/run_docker.sh -- bash scripts/launch_slurm.sh      # one-shot single-node train
#
# Inside the container /.dockerenv exists, so launch_slurm.sh auto-selects its
# SLURM-free `worker` phase (NNODES=1) — identical to the old launch_smoke_8gpu.sh.
#
# Overrides (export before invoking):
#   IMAGE          docker image                 (default: rocm/mlperf:dlrm_v3_mi355)
#   CONTAINER_NAME container name               (default: mlperf-recommendation-v4)
#   REPO_HOST      host path to repo            (default: this script's parent)
#   DATA_HOST      host path to dataset root    (default: /data/mlperf_dlrm_v4)
#   LOG            in-container train log path  (default: /workspace/recommendation_v4/mlperf_dlrm_v4.log)
#   MODE           launch_slurm.sh mode         (default: launcher default = streaming-train-eval; set train-eval for classic)
#   MAX_SEQ_LEN / HISTORY_LENGTH                seq shape; set 2048 / 2039 for the previous 2k shape
#   NCCL_SOCKET_IFNAME  NCCL bootstrap NIC      (default: launch_slurm picks lo single-node / fenic0 multi-node; override per host)

set -euo pipefail

IMAGE=${IMAGE:-rocm/mlperf:dlrm_v3_mi355}
CONTAINER_NAME=${CONTAINER_NAME:-mlperf-recommendation-v4}
REPO_HOST=${REPO_HOST:-$(cd "$(dirname "$0")/.." && pwd)}
DATA_HOST=${DATA_HOST:-/data/mlperf_dlrm_v4}

# Mount host paths at the same string inside the container so DLRM_DATA_PATH
# can be set from either side and resolve identically (env_path() in
# dlrm_v3/utils.py:641-653 does a literal os.environ.get).
REPO_CONT=/workspace/recommendation_v4
DATA_CONT=${DATA_HOST}

if [ ! -d "${DATA_HOST}" ]; then
  echo "warning: ${DATA_HOST} does not exist on host. Run preprocess_public_data first or override DATA_HOST." >&2
fi

# Drop an optional `--` separating this script's invocation from the in-container
# command (the documented `run_docker.sh -- bash scripts/launch_slurm.sh` form).
# Without this, `--` is forwarded verbatim to `docker run` as the command and
# fails with: exec: "--": executable file not found.
if [ "${1:-}" = "--" ]; then shift; fi

# If a container with this name is already running, exec into it instead of
# starting a new one. Matches the `docker exec yambda_8gpu ...` pattern in
# README.MD:9-12.
if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "container ${CONTAINER_NAME} already running; exec'ing in" >&2
  exec docker exec -it "${CONTAINER_NAME}" "${@:-bash}"
fi

# Remove a stopped container with the same name so --name doesn't collide.
if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  docker rm "${CONTAINER_NAME}" >/dev/null
fi

exec docker run --rm -it \
  --name "${CONTAINER_NAME}" \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --ipc=host --network=host \
  --shm-size=64g --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "${REPO_HOST}:${REPO_CONT}" \
  -v "${DATA_HOST}:${DATA_CONT}" \
  -e DLRM_DATA_PATH="${DATA_CONT}" \
  -e HSTU_HAMMER_KERNEL="${HSTU_HAMMER_KERNEL:-TRITON}" \
  -e RUN_NAME="${RUN_NAME:-default}" \
  -e LOG="${LOG:-/workspace/recommendation_v4/mlperf_dlrm_v4.log}" \
  ${MODE:+-e MODE="${MODE}"} \
  ${MAX_SEQ_LEN:+-e MAX_SEQ_LEN="${MAX_SEQ_LEN}"} \
  ${HISTORY_LENGTH:+-e HISTORY_LENGTH="${HISTORY_LENGTH}"} \
  ${NCCL_SOCKET_IFNAME:+-e NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}"} \
  -w "${REPO_CONT}" \
  "${IMAGE}" \
  "${@:-bash}"
