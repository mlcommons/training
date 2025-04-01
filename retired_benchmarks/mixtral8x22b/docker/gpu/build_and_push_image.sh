set -euox pipefail
SCRIPTS_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" && pwd )"
DATE=$(date +%Y%m%d)
: ${PROJECT_ID:=cloud-tpu-multipod-dev}
: ${IMAGE:=gcr.io/${PROJECT_ID}/${USER}-pytorch-nemo-moe-${DATE}}
: ${DOCKER_BUILD_ARGS:=""}

pushd ${SCRIPTS_DIR}

docker build --network host \
  --file Dockerfile \
  --tag ${IMAGE}-base \
  ${DOCKER_BUILD_ARGS} \
  .

docker build --network host \
  --file Dockerfile.GCP \
  --tag ${IMAGE} \
  --build-arg FROM_BASE_IMAGE=${IMAGE}-base \
  .

popd

docker push ${IMAGE}
