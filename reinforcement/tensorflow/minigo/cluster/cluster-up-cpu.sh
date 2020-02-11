#!/bin/bash
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

source ${SCRIPT_DIR}/common.sh
source ${SCRIPT_DIR}/utils.sh

echo "CPU Cluster Creation"
echo "--------------------------------------"
echo "Using Project:      ${PROJECT}"
echo "Using Zone:         ${ZONE}"
echo "Using Cluster Name: ${CLUSTER_NAME}"
echo "Using K8S Version:  ${K8S_VERSION}"
echo "Number of Nodes:    ${NUM_NODES}"
echo "Bucket name:        ${BUCKET_NAME}"
echo "Bucket location:    ${BUCKET_LOCATION}"

export PARALLELISM="$((4 * ${NUM_NODES}))"

check_gcloud_exists

# Create a Kubernetes cluster
# Note, we require Intel Broadwells since they are a bit newer, and can provide
# up to a 30% speedup, since we're so CPU bound.
gcloud beta container clusters create \
  --num-nodes $NUM_NODES \
  --machine-type n1-standard-4 \
  --min-cpu-platform "Intel Broadwell" \
  --disk-size 30 \
  --zone $ZONE \
  --project $PROJECT \
  --cluster-version=$K8S_VERSION \
  $CLUSTER_NAME

# Fetch its credentials so we can use kubectl locally
gcloud container clusters get-credentials $CLUSTER_NAME --project $PROJECT --zone $ZONE

create_gcs_bucket
create_service_account_key

# Import the credentials into the cluster as a secret
kubectl create secret generic ${SERVICE_ACCOUNT}-creds --from-file=service-account.json=${SERVICE_ACCOUNT_KEY_LOCATION}
