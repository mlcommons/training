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

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly RL_DIR="${SCRIPT_DIR}/RL"
readonly DATASET_ID="R2E-Gym/R2E-Gym-Subset"
readonly DATASET_REVISION="e8b9fcbce43eaca0dc2c0d4798ee6f3e965f590a"
readonly TRAIN_FILENAME="benchmark_r2e_gym_easy_train.filtered.curriculum-v2-classic-cycles2-seed20260710.jsonl"
readonly VAL_FILENAME="benchmark_r2e_gym_easy_val.jsonl"

usage() {
    cat <<'EOF'
Usage: download_dataset.sh [options]

Required:
  --dataset-dir PATH       Destination for the downloaded Hugging Face dataset.
  --output-dir PATH        Destination for the qualified train/validation JSONL.
  --cache-dir PATH         Persistent Git cache used by the converter.

Optional:
  --container-image-dir PATH
                           SIF path encoded in the JSONL
                           (default: /inputs/nemo_gym/sif).
  -h, --help               Show this help.
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 2
}

require_option_value() {
    local option="$1"
    local count="$2"
    ((count >= 2)) || die "${option} requires a value"
}

dataset_dir=""
output_dir=""
cache_dir=""
container_image_dir="/inputs/nemo_gym/sif"

while (($# > 0)); do
    case "$1" in
        --dataset-dir)
            require_option_value "$1" "$#"
            dataset_dir="$2"
            shift 2
            ;;
        --dataset-dir=*)
            dataset_dir="${1#*=}"
            shift
            ;;
        --output-dir)
            require_option_value "$1" "$#"
            output_dir="$2"
            shift 2
            ;;
        --output-dir=*)
            output_dir="${1#*=}"
            shift
            ;;
        --cache-dir)
            require_option_value "$1" "$#"
            cache_dir="$2"
            shift 2
            ;;
        --cache-dir=*)
            cache_dir="${1#*=}"
            shift
            ;;
        --container-image-dir)
            require_option_value "$1" "$#"
            container_image_dir="$2"
            shift 2
            ;;
        --container-image-dir=*)
            container_image_dir="${1#*=}"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

[[ -n "${dataset_dir}" ]] || die "--dataset-dir is required"
[[ -n "${output_dir}" ]] || die "--output-dir is required"
[[ -n "${cache_dir}" ]] || die "--cache-dir is required"

command -v uv >/dev/null 2>&1 || die "'uv' is required"
command -v uvx >/dev/null 2>&1 || die "'uvx' is required"
command -v git >/dev/null 2>&1 || die "'git' is required"

readonly converter="${RL_DIR}/tools/create_r2e_gym_easy_subset_jsonl.py"
readonly train_ids="${RL_DIR}/tools/train-instance-ids.txt"
readonly val_ids="${RL_DIR}/tools/val-instance-ids.txt"

[[ -f "${converter}" ]] || die "initialize the llm_moe_grpo/RL submodule first"
[[ -f "${train_ids}" ]] || die "missing qualified training ID list: ${train_ids}"
[[ -f "${val_ids}" ]] || die "missing qualified validation ID list: ${val_ids}"

mkdir -p "${dataset_dir}" "${output_dir}" "${cache_dir}"

uvx --from huggingface-hub==1.14.0 hf download "${DATASET_ID}" \
    --repo-type dataset \
    --revision "${DATASET_REVISION}" \
    --local-dir "${dataset_dir}"

staging_dir="$(mktemp -d "${output_dir}/.qwen35-dataset.XXXXXX")"
trap 'rm -rf -- "${staging_dir}"' EXIT

(
    cd "${RL_DIR}"
    uv run --with pyarrow==23.0.1 python \
        tools/create_r2e_gym_easy_subset_jsonl.py \
        --dataset-dir "${dataset_dir}" \
        --output-dir "${staging_dir}" \
        --cache-dir "${cache_dir}" \
        --container-image-dir "${container_image_dir}" \
        --train-ids "${train_ids}" \
        --val-ids "${val_ids}"
)

mv -f \
    "${staging_dir}/benchmark_r2e_gym_easy_train.jsonl" \
    "${output_dir}/${TRAIN_FILENAME}"
mv -f \
    "${staging_dir}/benchmark_r2e_gym_easy_val.jsonl" \
    "${output_dir}/${VAL_FILENAME}"
mv -f \
    "${staging_dir}/r2e_gym_subset_full_conversion_metrics.json" \
    "${output_dir}/r2e_gym_subset_full_conversion_metrics.json"

"${SCRIPT_DIR}/verify_dataset.sh" "${output_dir}"

echo "Qualified dataset written to ${output_dir}"
