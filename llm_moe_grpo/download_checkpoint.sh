#!/bin/bash

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

readonly MODEL_ID="Qwen/Qwen3.5-397B-A17B"
readonly MODEL_REVISION="8472618112abcbd45acbcdc58436aff4233c23f7"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Usage: HF_HOME=<cache-directory> [HF_TOKEN=<token>] ./download_checkpoint.sh

Download the exact Qwen3.5-397B-A17B snapshot used by the reference.
EOF
    exit 0
fi

if (($# != 0)); then
    echo "ERROR: unexpected argument: $1" >&2
    exit 2
fi

: "${HF_HOME:?HF_HOME must name the Hugging Face cache directory}"
command -v uvx >/dev/null 2>&1 || {
    echo "ERROR: 'uvx' is required" >&2
    exit 1
}

snapshot_path="$(
    uvx --from huggingface-hub==1.14.0 hf download "${MODEL_ID}" \
        --revision "${MODEL_REVISION}"
)"

case "${snapshot_path}" in
    */snapshots/"${MODEL_REVISION}") ;;
    *)
        echo "ERROR: download resolved to an unexpected snapshot: ${snapshot_path}" >&2
        exit 1
        ;;
esac

echo "Downloaded ${MODEL_ID}@${MODEL_REVISION}"
echo "Set HF_CKPT_PATH=${snapshot_path}"
