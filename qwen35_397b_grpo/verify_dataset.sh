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

readonly TRAIN_FILENAME="benchmark_r2e_gym_easy_train.jsonl"
readonly TRAIN_SHA256="e772af09599270ab16a04ccbbcef395bcb2929607f050478033a84b087a606fc"
readonly TRAIN_ROWS="700"
readonly TRAIN_BYTES="446364412"

readonly VAL_FILENAME="benchmark_r2e_gym_easy_val.jsonl"
readonly VAL_SHA256="2d5bddf717afb4389510a7d197f26ce5eb7f610aab4b20d6af6e0150de6f2c79"
readonly VAL_ROWS="251"
readonly VAL_BYTES="172078629"

usage() {
    cat <<'EOF'
Usage: verify_dataset.sh <dataset-directory>

Verify the row count, byte size, and SHA-256 identity of the qualified
Qwen3.5 GRPO training and validation JSONL files.
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

sha256_file() {
    local path="$1"
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "${path}" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "${path}" | awk '{print $1}'
    else
        die "sha256sum or shasum is required"
    fi
}

verify_file() {
    local path="$1"
    local expected_sha256="$2"
    local expected_rows="$3"
    local expected_bytes="$4"
    local actual_sha256
    local actual_rows
    local actual_bytes

    [[ -f "${path}" ]] || die "missing dataset file: ${path}"

    actual_sha256="$(sha256_file "${path}")"
    actual_rows="$(wc -l < "${path}" | tr -d '[:space:]')"
    actual_bytes="$(wc -c < "${path}" | tr -d '[:space:]')"

    [[ "${actual_sha256}" == "${expected_sha256}" ]] || \
        die "${path}: SHA-256 ${actual_sha256}, expected ${expected_sha256}"
    [[ "${actual_rows}" == "${expected_rows}" ]] || \
        die "${path}: ${actual_rows} rows, expected ${expected_rows}"
    [[ "${actual_bytes}" == "${expected_bytes}" ]] || \
        die "${path}: ${actual_bytes} bytes, expected ${expected_bytes}"

    echo "OK: ${path}"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi
if (($# != 1)); then
    usage >&2
    exit 2
fi

readonly dataset_dir="$1"

verify_file \
    "${dataset_dir}/${TRAIN_FILENAME}" \
    "${TRAIN_SHA256}" \
    "${TRAIN_ROWS}" \
    "${TRAIN_BYTES}"
verify_file \
    "${dataset_dir}/${VAL_FILENAME}" \
    "${VAL_SHA256}" \
    "${VAL_ROWS}" \
    "${VAL_BYTES}"
