#!/bin/bash
# Launch a yambda_8gpu container from rocm/mlperf:dlrm_v3_mi355 with the repo
# and data directories bind-mounted at matching host/container paths.
#
# Usage:
#   bash scripts/run_docker.sh                   # interactive shell
#   bash scripts/run_docker.sh -- bash scripts/launch_smoke_8gpu.sh   # one-shot
#
# Overrides (export before invoking):
#   IMAGE          docker image                 (default: rocm/mlperf:dlrm_v3_mi355)
#   CONTAINER_NAME container name               (default: yambda_8gpu)
#   REPO_HOST      host path to repo            (default: this script's parent)
#   DATA_HOST      host path to dataset root    (default: /data/mlperf_dlrm_v4)

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
  -w "${REPO_CONT}" \
  "${IMAGE}" \
  "${@:-bash}"
