#!/bin/bash
set -e

: "${C4_PATH:?C4_PATH not set}"

echo "Starting parallel compression in $C4_PATH..."

# Use 50% of available CPU cores (adjust -j as needed)
find "$C4_PATH" -maxdepth 1 -name '*.json' | \
  parallel -j$(nproc) '
    echo "Compressing {}"
    gzip "{}"
'

echo "Parallel compression complete!"

