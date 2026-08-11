#!/usr/bin/env bash
# MLPerf Training reference script: verify the dataset.
#
# There are two supported layouts under ${DLRM_DATA_PATH}, and this script
# verifies whichever is present:
#
#   A. Prepared download (MLCommons). The derived flat-event cache plus the two
#      item mappings and item_popularity.npy. No parquet: the cache already
#      contains everything the trainer reads.
#
#   B. Local build (./download_dataset.sh). The five preprocessing outputs plus
#      the two item mappings. The cache is built by the first training run, and
#      is checked here too once it exists.
#
# Both layouts must carry the two shared_metadata/ item mappings, since
# DLRMv4YambdaDataset reads them at init to size the embedding tables. Beyond
# that, at least one trainable source is required -- either the cache or
# train_sessions.parquet to build it from.
#
# Checksums live in two files, both keyed relative to the processed dir:
#   md5sums_yambda_5b_processed.txt     preprocessing outputs + item mappings
#   sha256sums_yambda_5b_cache.txt      the sixteen cache files
#
# Usage:
#   DLRM_DATA_PATH=/path/to/dlrm_data ./verify_dataset.sh
#
# Env:
#   DLRM_DATA_PATH    data root (required).
#   PROCESSED_SUBDIR  processed subdir under the data root (default: processed_5b).
#   CACHE_SUBDIR      cache subdir under the processed dir (default: hstu_cache_L4086).
set -euo pipefail

: "${DLRM_DATA_PATH:?Set DLRM_DATA_PATH to the data root}"
PROCESSED_SUBDIR="${PROCESSED_SUBDIR:-processed_5b}"
CACHE_SUBDIR="${CACHE_SUBDIR:-hstu_cache_L4086}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MD5_FILE="${REPO_ROOT}/md5sums_yambda_5b_processed.txt"
CACHE_FILE="${REPO_ROOT}/sha256sums_yambda_5b_cache.txt"
PROCESSED_DIR="${DLRM_DATA_PATH}/${PROCESSED_SUBDIR}"

echo "[verify_dataset] data root:     ${DLRM_DATA_PATH}"
echo "[verify_dataset] processed dir: ${PROCESSED_DIR}"

if [[ ! -d "${PROCESSED_DIR}" ]]; then
    echo "[verify_dataset] ERROR: ${PROCESSED_DIR} does not exist." >&2
    echo "[verify_dataset]        Download the prepared dataset or run ./download_dataset.sh." >&2
    exit 1
fi

WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

# Verify the subset of a checksum file whose paths match a grep pattern, so a
# layout that legitimately omits some files is not reported as corrupt.
# $1 checksum file, $2 tool (md5sum|sha256sum), $3 grep -E pattern, $4 label
verify_subset() {
    local file="$1" tool="$2" pattern="$3" label="$4"
    local subset="${WORK}/subset.$$"
    grep -E "^[0-9a-f]{32,64}[[:space:]]" "${file}" | grep -E "${pattern}" > "${subset}" || true
    if [[ ! -s "${subset}" ]]; then
        echo "[verify_dataset] ERROR: no checksums matched ${pattern} in ${file}" >&2
        return 1
    fi
    echo "[verify_dataset] ${label}: checking $(wc -l < "${subset}") file(s) with ${tool}"
    (cd "${PROCESSED_DIR}" && "${tool}" -c "${subset}")
}

# --- always required: the two item mappings ----------------------------------
verify_subset "${MD5_FILE}" md5sum 'shared_metadata/' "item mappings"

# --- optional: the length guard ----------------------------------------------
if [[ -s "${PROCESSED_DIR}/item_popularity.npy" ]]; then
    verify_subset "${MD5_FILE}" md5sum 'item_popularity\.npy' "item popularity"
else
    echo "[verify_dataset] NOTE: item_popularity.npy absent (optional; it only" \
         "cross-checks the vocabulary size against the mappings)."
fi

# --- at least one trainable source -------------------------------------------
have_cache=0
have_parquet=0
[[ -f "${PROCESSED_DIR}/${CACHE_SUBDIR}/_READY" ]] && have_cache=1
[[ -s "${PROCESSED_DIR}/train_sessions.parquet" ]] && have_parquet=1

if [[ "${have_cache}" -eq 0 && "${have_parquet}" -eq 0 ]]; then
    echo "[verify_dataset] ERROR: neither ${CACHE_SUBDIR}/_READY nor train_sessions.parquet found." >&2
    echo "[verify_dataset]        One of them is required; the cache is built from the parquet." >&2
    exit 1
fi

if [[ "${have_parquet}" -eq 1 ]]; then
    verify_subset "${MD5_FILE}" md5sum '^[0-9a-f]{32}[[:space:]]+[a-z_]+\.(parquet|json)$' \
        "preprocessing outputs"
fi

if [[ "${have_cache}" -eq 1 ]]; then
    # Sound for a locally built cache as well as a downloaded one: the build is
    # deterministic (see the header of sha256sums_yambda_5b_cache.txt).
    verify_subset "${CACHE_FILE}" sha256sum "${CACHE_SUBDIR}/" "flat-event cache"
else
    echo "[verify_dataset] NOTE: ${CACHE_SUBDIR}/ not present; the first training run" \
         "will build it (~190 GB peak RAM)."
fi

echo
if [[ "${have_cache}" -eq 1 && "${have_parquet}" -eq 1 ]]; then
    layout="local build with cache present"
elif [[ "${have_cache}" -eq 1 ]]; then
    layout="prepared download (cache, no parquet)"
else
    layout="local build (cache not built yet)"
fi
echo "[verify_dataset] OK: all present files match. Layout: ${layout}."
