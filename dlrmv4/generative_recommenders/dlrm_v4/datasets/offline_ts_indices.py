# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

"""Bucketize Yambda anchors by time window and save TS indices and offsets."""

import argparse
import fcntl
import os
import tempfile
from pathlib import Path

import numpy as np


def build_offline_ts_indices(
    cache_dir, history_length, min_history=None, window_seconds=86400,
    chunk_size=1_000_000,
):
    """Bucketize anchors by time window, preserving their order within each bucket."""
    if history_length <= 0 or window_seconds <= 0 or chunk_size <= 0:
        raise ValueError("history_length, window_seconds and chunk_size must be positive")
    if min_history is not None and min_history < 0:
        raise ValueError("min_history must be nonnegative")
    cache_dir = Path(cache_dir)
    tag = f"L{history_length}"
    if min_history is not None and min_history != history_length:
        tag += f"_m{min_history}"
    # Keep derived files with their source cache so separate caches cannot collide.
    indices_path = cache_dir / f"ts_indices_{tag}_W{window_seconds}.npy"
    offsets_path = cache_dir / f"ts_offsets_{tag}_W{window_seconds}.npy"
    # Complete caches bypass the build lock, as do training-time index reads.
    if indices_path.exists() and offsets_path.exists():
        return indices_path, offsets_path

    # Each rank calls this during dataset initialization. Reuse the positions
    # lock so only one rank builds missing output, avoiding duplicate work and
    # concurrent writes to the same files without adding a separate TS lock.
    with open(cache_dir / "_positions_lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        # Another rank may have finished building while this rank waited.
        if indices_path.exists() and offsets_path.exists():
            return indices_path, offsets_path
        positions = np.load(cache_dir / f"positions_{tag}.npy", mmap_mode="r")
        anchor_path = cache_dir / f"anchor_ts_{tag}.npy"
        has_anchor_ts = anchor_path.exists()
        timestamps = np.load(
            anchor_path if has_anchor_ts else cache_dir / "flat_timestamps.npy", mmap_mode="r"
        )
        if any(array.ndim != 1 or array.dtype != np.int64 for array in (positions, timestamps)):
            raise ValueError("positions and timestamps must be one-dimensional int64 arrays")
        count = len(positions)
        if has_anchor_ts and len(timestamps) != count:
            raise ValueError("Anchor timestamps and positions have different lengths")

        def timestamp_chunk(start):
            stop = min(start + chunk_size, count)
            return timestamps[start:stop] if has_anchor_ts else timestamps[positions[start:stop]]

        t_min, t_max = 0, -1
        for start in range(0, count, chunk_size):
            chunk = timestamp_chunk(start)
            t_min = int(chunk.min()) if start == 0 else min(t_min, int(chunk.min()))
            t_max = int(chunk.max()) if start == 0 else max(t_max, int(chunk.max()))
        if t_max - t_min > np.iinfo(np.int64).max:
            raise ValueError("Timestamp span exceeds int64 capacity")
        windows = (t_max - t_min) // window_seconds + 1 if count else 0
        counts = np.zeros(windows, dtype=np.int64)
        for start in range(0, count, chunk_size):
            buckets = (timestamp_chunk(start) - t_min) // window_seconds
            counts += np.bincount(buckets, minlength=windows)
        offsets = np.concatenate(([0], np.cumsum(counts)))
        cursors = offsets[:-1].copy()
        with tempfile.TemporaryDirectory(prefix=".ts-indices-", dir=cache_dir) as tmp:
            staging = Path(tmp)
            indices = np.lib.format.open_memmap(
                staging / indices_path.name, mode="w+", dtype=np.int64, shape=(count,)
            )
            try:
                for start in range(0, count, chunk_size):
                    buckets = (timestamp_chunk(start) - t_min) // window_seconds
                    order = np.argsort(buckets, kind="stable")
                    sizes = np.bincount(buckets, minlength=windows)
                    begin = 0
                    for bucket in np.flatnonzero(sizes):
                        end = begin + int(sizes[bucket])
                        cursor = int(cursors[bucket])
                        indices[cursor:cursor + end - begin] = order[begin:end] + start
                        cursors[bucket] += end - begin
                        begin = end
                indices.flush()
            finally:
                indices._mmap.close()
            np.save(staging / offsets_path.name, offsets)
            # Publish offsets last so an interrupted build cannot appear complete.
            offsets_path.unlink(missing_ok=True)
            os.replace(staging / indices_path.name, indices_path)
            os.replace(staging / offsets_path.name, offsets_path)
    return indices_path, offsets_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--history-length", type=int, required=True)
    parser.add_argument("--min-history", type=int)
    parser.add_argument("--window-seconds", type=int, default=86400)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    print(*build_offline_ts_indices(**vars(parser.parse_args())), sep="\n")
