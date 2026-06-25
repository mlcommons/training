#!/usr/bin/env bash
# MLPerf Training reference script: verify the preprocessed dataset.
#
# Checks the integrity of the preprocessed dataset under
#   ${DLRM_DATA_PATH}/${PROCESSED_SUBDIR}
# against md5sums_yambda_5b_processed.txt (standard `md5sum -c` format).
#
# If the checksum file still contains placeholder hashes (TODO_GENERATE_HASH),
# the script falls back to an existence/layout check and warns that the
# canonical checksums have not been pinned yet.
#
# Usage:
#   DLRM_DATA_PATH=/path/to/dlrm_data ./verify_dataset.sh
#
# Env:
#   DLRM_DATA_PATH    data root (required).
#   PROCESSED_SUBDIR  processed subdir under the data root (default: processed_5b).
set -euo pipefail

: "${DLRM_DATA_PATH:?Set DLRM_DATA_PATH to the data root}"
PROCESSED_SUBDIR="${PROCESSED_SUBDIR:-processed_5b}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKSUM_FILE="${REPO_ROOT}/md5sums_yambda_5b_processed.txt"
PROCESSED_DIR="${DLRM_DATA_PATH}/${PROCESSED_SUBDIR}"

echo "[verify_dataset] processed dir: ${PROCESSED_DIR}"

if [[ ! -d "${PROCESSED_DIR}" ]]; then
    echo "[verify_dataset] ERROR: ${PROCESSED_DIR} does not exist. Run ./download_dataset.sh first." >&2
    exit 1
fi

EXPECTED_FILES=(
    train_sessions.parquet
    test_events.parquet
    session_index.parquet
    item_popularity.npy
    split_meta.json
)

# Detect whether the checksum file has real (32 hex char) hashes or placeholders.
if grep -qiE '^[0-9a-f]{32}[[:space:]]' "${CHECKSUM_FILE}"; then
    echo "[verify_dataset] checking md5 checksums from ${CHECKSUM_FILE}"
    (cd "${PROCESSED_DIR}" && md5sum -c "${CHECKSUM_FILE}")
    echo "[verify_dataset] OK: all checksums match."
else
    echo "[verify_dataset] WARNING: ${CHECKSUM_FILE} contains placeholder hashes;" >&2
    echo "[verify_dataset]          falling back to existence/layout check only." >&2
    missing=0
    for f in "${EXPECTED_FILES[@]}"; do
        if [[ -s "${PROCESSED_DIR}/${f}" ]]; then
            echo "  OK   ${f}"
        else
            echo "  MISS ${f}" >&2
            missing=1
        fi
    done
    if [[ "${missing}" -ne 0 ]]; then
        echo "[verify_dataset] ERROR: one or more expected files are missing/empty." >&2
        exit 1
    fi
    echo "[verify_dataset] layout OK (checksums NOT yet pinned -- see TODO in ${CHECKSUM_FILE})."
fi
