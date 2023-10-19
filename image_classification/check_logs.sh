#!/bin/bash

set +x
set -e

LOG_DIR=${LOG_DIR:-"./workspace/logs/"}
CHECKER_LOG_DIR=${CHECKER_LOG_DIR:-"./workspace/checker_logs/"}

# Handle MLCube parameters
while [ $# -gt 0 ]; do
    case "$1" in
    --log_dir=*)
        LOG_DIR="${1#*=}"
        ;;
    --checker_logs_dir=*)
        CHECKER_LOG_DIR="${1#*=}"
        ;;
    *) ;;
    esac
    shift
done

for filename in $LOG_DIR/bert_*.log; do
    log_file=${filename##*/}
    python -m mlperf_logging.compliance_checker $filename --log_output $CHECKER_LOG_DIR/$log_file || true
done
