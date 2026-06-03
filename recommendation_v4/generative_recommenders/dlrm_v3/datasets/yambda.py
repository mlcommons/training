# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

# pyre-unsafe
"""
Yambda dataset for the DLRMv3 HSTU `modules/` path.

Reads the parquets produced by `dlrm_v3/preprocess_public_data.py
--dataset yambda-<size>`. Each sample is one anchor LISTEN event with:
  * label = (played_ratio >= LISTEN_PLUS_THRESHOLD) — the listen_plus bit
  * a chronologically interleaved 3-pool history (listen+/like/skip), with
    pool identity tagged per-position in `action_weight` (bits 1/2/4)
  * 7 pre-hashed cross-feature ids exposed as length-1 contextual entries
"""

import logging
import mmap as _mmap_mod
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import polars as pl
import torch
from generative_recommenders.dlrm_v3.datasets.dataset import DLRMv3RandomDataset
from generative_recommenders.dlrm_v3.datasets.utils import xxhash_cross
from generative_recommenders.modules.dlrm_hstu import DlrmHSTUConfig
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

logger = logging.getLogger(__name__)


def _load_npy_readonly(path: Union[str, Path]) -> np.ndarray:
    # MAP_SHARED + PROT_READ so the kernel does not charge the mapping against
    # vm.overcommit_memory=2 limits. numpy's mmap_mode='r' uses MAP_PRIVATE and
    # reserves per-process commit; at 8 ranks × ~190 GB store, that OOMs.
    path = Path(path)
    with open(path, "rb") as f:
        version = np.lib.format.read_magic(f)
        if version[0] == 1:
            shape, _, dtype = np.lib.format.read_array_header_1_0(f)
        else:
            shape, _, dtype = np.lib.format.read_array_header_2_0(f)
        offset = f.tell()
    fd = os.open(str(path), os.O_RDONLY)
    try:
        buf = _mmap_mod.mmap(fd, 0, access=_mmap_mod.ACCESS_READ)
    finally:
        os.close(fd)
    arr = np.ndarray(shape, dtype=dtype, buffer=buf, offset=offset)
    arr.flags.writeable = False
    return arr

# Yambda event-type encoding written by preprocess_public_data.py.
LISTEN_TYPE = 0
LIKE_TYPE = 1
LISTEN_PLUS_THRESHOLD = 50

# Action-weight bits (must match hstu_config.action_weights = [1, 2, 4]).
LP_BIT = 1
LIKE_BIT = 2
SKIP_BIT = 4


class _FlatEventStore:
    """Per-user flat event index built from the preprocessed sessions parquet.

    Reads `train_sessions.parquet` and explodes per-session arrays into flat
    numpy columns + per-user `(start, end)` index arrays. Cache-compatible
    layout, but writes nothing (rebuilds from parquet each construction).
    """

    # On-disk column layout.
    _MMAP_COLS = (
        "flat_uid", "flat_item_ids", "flat_timestamps",
        "flat_event_types", "flat_played_ratio",
        "flat_is_listen_plus", "flat_is_like", "flat_is_skip",
        "flat_is_organic",
        "user_start", "user_end", "unique_uids",
    )

    def __init__(self, sessions_df: pl.DataFrame) -> None:
        logger.info("Building flat event store from sessions...")
        sorted_sessions = sessions_df.sort(["uid", "session_id"])
        exploded = sorted_sessions.explode(
            ["item_ids", "timestamps", "event_types", "is_organic", "played_ratio_pct"]
        )

        self.flat_uid: np.ndarray = exploded["uid"].to_numpy().astype(np.int64)
        self.flat_item_ids: np.ndarray = exploded["item_ids"].to_numpy().astype(np.int64)
        self.flat_timestamps: np.ndarray = exploded["timestamps"].to_numpy().astype(np.int64)
        self.flat_event_types: np.ndarray = exploded["event_types"].to_numpy().astype(np.int64)
        self.flat_played_ratio: np.ndarray = exploded["played_ratio_pct"].to_numpy().astype(np.float32)
        self.flat_is_organic: np.ndarray = exploded["is_organic"].to_numpy().astype(np.int8)
        np.nan_to_num(self.flat_played_ratio, copy=False, nan=0.0)

        is_listen = self.flat_event_types == LISTEN_TYPE
        self.flat_is_listen_plus: np.ndarray = is_listen & (
            self.flat_played_ratio >= LISTEN_PLUS_THRESHOLD
        )
        self.flat_is_like: np.ndarray = self.flat_event_types == LIKE_TYPE
        self.flat_is_skip: np.ndarray = is_listen & (
            self.flat_played_ratio < LISTEN_PLUS_THRESHOLD
        )

        uid_changes = np.where(np.diff(self.flat_uid) != 0)[0] + 1
        starts = np.concatenate([[0], uid_changes])
        ends = np.concatenate([uid_changes, [len(self.flat_uid)]])
        uid_vals = self.flat_uid[starts]
        max_uid = int(uid_vals.max()) + 1
        self.user_start: np.ndarray = np.full(max_uid, -1, dtype=np.int64)
        self.user_end: np.ndarray = np.full(max_uid, -1, dtype=np.int64)
        self.user_start[uid_vals] = starts
        self.user_end[uid_vals] = ends
        self.unique_uids: np.ndarray = uid_vals
        self.num_users: int = len(uid_vals)
        self.total_events: int = len(self.flat_item_ids)
        logger.info(
            f"FlatEventStore: {self.total_events:,} events, {self.num_users:,} users"
        )

    @classmethod
    def load_mmap(cls, cache_dir: Union[str, Path]) -> "_FlatEventStore":
        """Load flat columns by MAP_SHARED+PROT_READ from a prebuilt cache.
        All ranks on a node share the same physical pages."""
        import json as _json
        cache_dir = Path(cache_dir)
        with open(cache_dir / "store_meta.json") as f:
            meta = _json.load(f)
        store = object.__new__(cls)
        for name in cls._MMAP_COLS:
            setattr(store, name, _load_npy_readonly(cache_dir / f"{name}.npy"))
        store.num_users = int(meta["num_users"])
        store.total_events = int(meta["total_events"])
        logger.info(
            f"FlatEventStore mmap from {cache_dir}: "
            f"{store.total_events:,} events, {store.num_users:,} users"
        )
        return store

    def save_mmap(self, cache_dir: Union[str, Path]) -> None:
        """Persist flat columns to disk as .npy, then write a sentinel.
        Subsequent runs (any rank, any node sharing the FS) load via mmap."""
        import json as _json
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        for name in self._MMAP_COLS:
            np.save(cache_dir / f"{name}.npy", getattr(self, name))
        with open(cache_dir / "store_meta.json", "w") as f:
            _json.dump(
                {"num_users": self.num_users, "total_events": self.total_events}, f
            )
        # Sentinel — readers check this before mmap'ing to avoid partial files.
        (cache_dir / "_READY").touch()
        logger.info(f"FlatEventStore saved to {cache_dir}")


class DLRMv3YambdaDataset(DLRMv3RandomDataset):
    """Yambda-5b dataset for the DLRMv3 HSTU modules/ path.

    Args:
        hstu_config: DlrmHSTUConfig (must come from `get_hstu_configs("yambda-5b")`).
        processed_dir: directory with `train_sessions.parquet` + `item_popularity.npy`.
        metadata_dir: directory with `{artist,album}_item_mapping.parquet`.
        history_length: per-pool truncation cap (total interleaved ≤ 3 * this).
        scan_window: how far back to scan when filling each pool.
        cross_specs: list of (name, keys, num_embeddings, salt). Source of truth
            in `dlrm_v3/configs.py:YAMBDA_5B_CROSS_SPECS`.
        is_inference: passed through to base class.
    """

    def __init__(
        self,
        hstu_config: DlrmHSTUConfig,
        processed_dir: str,
        metadata_dir: str,
        history_length: int = 2048,
        scan_window: int = 20000,
        cross_specs: Optional[Sequence[Tuple[str, Sequence[str], int, int]]] = None,
        cache_dir: Optional[str] = None,
        is_inference: bool = False,
        streaming_window_seconds: int = 86400,
        streaming_sort_within_window: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(hstu_config=hstu_config, is_inference=is_inference)
        self._processed_dir: str = processed_dir
        self._metadata_dir: str = metadata_dir
        self._history_length: int = history_length
        self._scan_window: int = scan_window
        # Streaming/temporal-order state. Everything here is LAZY: nothing is
        # built or read until the first set_ts()/num_windows() call (only the
        # streaming-train-eval loop does that), so the default train-eval path
        # is byte-for-byte unaffected.
        self._streaming_window_seconds: int = streaming_window_seconds
        self._streaming_sort_within_window: bool = streaming_sort_within_window
        self._active: Optional[np.ndarray] = None
        self.is_eval: bool = False
        self._anchor_ts: Optional[np.ndarray] = None
        self._t_min: Optional[int] = None
        self._t_max: Optional[int] = None
        self._cache_dir: Optional[str] = cache_dir
        self._cross_specs: List[Tuple[str, Tuple[str, ...], int, int]] = [
            (name, tuple(keys), n, s) for (name, keys, n, s) in (cross_specs or [])
        ]
        assert hstu_config.action_weights is not None
        self._action_weights: List[int] = hstu_config.action_weights

        self._load_metadata(metadata_dir)
        # Build-once-mmap-many: first rank to arrive acquires the build lock
        # and explodes the parquet (one ~190 GB in-memory pass), then writes
        # flat .npy columns + _READY sentinel. All ranks (including the
        # builder, after dropping its in-memory copy) reload via MAP_SHARED+
        # PROT_READ — kernel shares physical pages across ranks so the steady-
        # state per-rank RSS for the dataset is ~0.
        if cache_dir is None:
            cache_dir = os.path.join(processed_dir, f"hstu_cache_L{history_length}")
        self._cache_dir = cache_dir
        self._ensure_cache_built(cache_dir, processed_dir, history_length)
        self.store: _FlatEventStore = _FlatEventStore.load_mmap(cache_dir)
        # Mmap the positions file built alongside the flat columns.
        self._positions: np.ndarray = _load_npy_readonly(
            os.path.join(cache_dir, f"positions_L{history_length}.npy")
        )
        logger.info(
            f"Yambda dataset ready: {self.store.total_events:,} events, "
            f"{len(self._positions):,} training positions"
        )

    @staticmethod
    def _ensure_cache_built(
        cache_dir: str, processed_dir: str, history_length: int
    ) -> None:
        """File-locked one-shot build with column-at-a-time explode.

        A naive `pl.read_parquet(...).explode([5 list cols])` peaks at ~1.6 TB
        on the 5b dataset (polars holds input list-columns + dense output +
        parallel-worker scratch all together). Instead we:
          1) Read parquet + sort once (sorted list-column DF, ~80 GB).
          2) For each output column: select that single list, explode, write
             .npy, drop. Bounds incremental peak to one column (~38 GB).
          3) Derive bool flags and indices from the on-disk mmaps.

        Peak RAM: ~150 GB. Steady state across all ranks afterward: ~0
        incremental thanks to MAP_SHARED in load_mmap.
        """
        import fcntl
        import gc
        import json as _json

        ready = os.path.join(cache_dir, "_READY")
        if os.path.exists(ready):
            return
        os.makedirs(cache_dir, exist_ok=True)
        lock_path = os.path.join(cache_dir, "_lock")
        with open(lock_path, "w") as lf:
            logger.info(f"Acquiring build lock for {cache_dir}...")
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                if os.path.exists(ready):
                    return
                parquet_path = os.path.join(processed_dir, "train_sessions.parquet")
                logger.info(
                    f"Building flat-event cache from {parquet_path} "
                    f"(column-at-a-time, ~150 GB peak RAM)"
                )

                # Step 1: read + sort. List columns stay nested at this stage.
                sessions = pl.read_parquet(parquet_path).sort(["uid", "session_id"])
                logger.info(f"Sessions sorted: {sessions.shape}")

                # Per-session lengths + uids — used to derive flat_uid via
                # np.repeat (cheap) without exploding the whole DF at once.
                lengths = (
                    sessions.select(pl.col("item_ids").list.len())
                    .to_numpy()
                    .reshape(-1)
                    .astype(np.int64)
                )
                session_uids = sessions["uid"].to_numpy().astype(np.int64)
                N = int(lengths.sum())
                num_users = int(np.unique(session_uids).shape[0])
                logger.info(f"Total events: {N:,}, users: {num_users:,}")

                # Step 2: column-at-a-time explode → save → drop.
                # uid is per-session scalar; expand via np.repeat.
                flat_uid = np.repeat(session_uids, lengths).astype(np.int64)
                np.save(os.path.join(cache_dir, "flat_uid.npy"), flat_uid)
                del flat_uid, session_uids, lengths
                gc.collect()
                logger.info("Wrote flat_uid.npy")

                # Derived columns flat_is_listen_plus/like/skip depend on
                # event_types + played_ratio. Save those two first, then
                # derive the bools from the mmaps.
                _list_cols = [
                    ("item_ids", "flat_item_ids", np.int64),
                    ("timestamps", "flat_timestamps", np.int64),
                    ("event_types", "flat_event_types", np.int64),
                    ("is_organic", "flat_is_organic", np.int8),
                    ("played_ratio_pct", "flat_played_ratio", np.float32),
                ]
                for src_col, dst_name, dtype in _list_cols:
                    exploded = sessions.select(pl.col(src_col).explode())
                    arr = exploded[src_col].to_numpy().astype(dtype, copy=False)
                    if dtype == np.float32:
                        np.nan_to_num(arr, copy=False, nan=0.0)
                    np.save(os.path.join(cache_dir, f"{dst_name}.npy"), arr)
                    del exploded, arr
                    gc.collect()
                    logger.info(f"Wrote {dst_name}.npy")

                # Drop the sessions DF now that all source columns are on disk.
                del sessions
                gc.collect()

                # Step 3: derive bool flags from the just-written mmaps.
                event_types = _load_npy_readonly(
                    os.path.join(cache_dir, "flat_event_types.npy")
                )
                played_ratio = _load_npy_readonly(
                    os.path.join(cache_dir, "flat_played_ratio.npy")
                )
                is_listen = event_types == LISTEN_TYPE
                np.save(
                    os.path.join(cache_dir, "flat_is_listen_plus.npy"),
                    is_listen & (played_ratio >= LISTEN_PLUS_THRESHOLD),
                )
                np.save(
                    os.path.join(cache_dir, "flat_is_like.npy"),
                    event_types == LIKE_TYPE,
                )
                np.save(
                    os.path.join(cache_dir, "flat_is_skip.npy"),
                    is_listen & (played_ratio < LISTEN_PLUS_THRESHOLD),
                )
                del is_listen, played_ratio
                gc.collect()
                logger.info("Wrote flat_is_listen_plus/like/skip.npy")

                # user_start / user_end / unique_uids from flat_uid mmap.
                flat_uid = _load_npy_readonly(
                    os.path.join(cache_dir, "flat_uid.npy")
                )
                uid_changes = np.where(np.diff(flat_uid) != 0)[0] + 1
                starts = np.concatenate([[0], uid_changes])
                ends = np.concatenate([uid_changes, [len(flat_uid)]])
                uid_vals = flat_uid[starts]
                max_uid = int(uid_vals.max()) + 1
                user_start = np.full(max_uid, -1, dtype=np.int64)
                user_end = np.full(max_uid, -1, dtype=np.int64)
                user_start[uid_vals] = starts
                user_end[uid_vals] = ends
                np.save(os.path.join(cache_dir, "user_start.npy"), user_start)
                np.save(os.path.join(cache_dir, "user_end.npy"), user_end)
                np.save(os.path.join(cache_dir, "unique_uids.npy"), uid_vals)
                logger.info("Wrote user_start/end/unique_uids.npy")

                # Positions: LISTEN events with ≥history_length prior history.
                # Done now (before dropping user_start) so all sibling ranks
                # just mmap the result instead of each running a 75 GB build.
                user_start_per_event = user_start[flat_uid]
                idx = np.arange(len(flat_uid), dtype=np.int64)
                keep = (idx - user_start_per_event >= history_length) & (
                    event_types == LISTEN_TYPE
                )
                positions = np.where(keep)[0].astype(np.int64)
                np.save(
                    os.path.join(cache_dir, f"positions_L{history_length}.npy"),
                    positions,
                )
                logger.info(
                    f"Wrote positions_L{history_length}.npy: {len(positions):,}"
                )
                del (
                    flat_uid, event_types, user_start, user_end, uid_vals,
                    starts, ends, uid_changes, idx, user_start_per_event,
                    keep, positions,
                )
                gc.collect()

                # Meta + sentinel — written last; readers gate on _READY.
                with open(os.path.join(cache_dir, "store_meta.json"), "w") as f:
                    _json.dump(
                        {"num_users": num_users, "total_events": N}, f
                    )
                open(os.path.join(cache_dir, "_READY"), "w").close()
                logger.info(f"Cache build complete: {cache_dir}")
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)

    def _load_metadata(self, metadata_dir: str) -> None:
        item_pop_path = os.path.join(metadata_dir, "item_popularity.npy")
        if os.path.exists(item_pop_path):
            item_popularity = np.load(item_pop_path)
        else:
            # Fallback: derive vocab size from the artist+album maps.
            item_popularity = None

        artist_map = pl.read_parquet(os.path.join(metadata_dir, "artist_item_mapping.parquet"))
        album_map = pl.read_parquet(os.path.join(metadata_dir, "album_item_mapping.parquet"))
        n_items = int(
            max(
                int(artist_map["item_id"].max()) + 1,
                int(album_map["item_id"].max()) + 1,
                len(item_popularity) if item_popularity is not None else 0,
            )
        )
        self.item_to_artist: np.ndarray = np.zeros(n_items, dtype=np.int64)
        valid = artist_map.filter(pl.col("item_id") < n_items)
        self.item_to_artist[valid["item_id"].to_numpy()] = valid["artist_id"].to_numpy()
        self.item_to_album: np.ndarray = np.zeros(n_items, dtype=np.int64)
        valid = album_map.filter(pl.col("item_id") < n_items)
        self.item_to_album[valid["item_id"].to_numpy()] = valid["album_id"].to_numpy()
        self.num_items: int = n_items

    def get_item_count(self) -> int:
        # Streaming mode restricts the active set to the current time window;
        # otherwise the full (user-major) anchor list is used (train-eval).
        if self._active is not None:
            return int(len(self._active))
        return int(len(self._positions))

    def iloc(self, idx: int) -> int:
        if self._active is not None:
            return int(self._positions[self._active[idx]])
        return int(self._positions[idx])

    def _ensure_streaming_index(self) -> None:
        """Lazily build + mmap the per-anchor target-timestamp array used for
        time-windowed streaming.

        Built only on the first ``set_ts()``/``num_windows()`` call, so the
        default train-eval path never reads timestamps or writes a new file.
        Multi-rank safe via an exclusive file lock + atomic rename; all ranks
        then mmap the result read-only (shared physical pages, ~0 anon).
        """
        if self._anchor_ts is not None:
            return
        import fcntl

        assert self._cache_dir is not None
        anchor_path = os.path.join(
            self._cache_dir, f"anchor_ts_L{self._history_length}.npy"
        )
        if not os.path.exists(anchor_path):
            lock_path = os.path.join(self._cache_dir, "_anchor_ts_lock")
            with open(lock_path, "w") as lf:
                logger.info(f"Acquiring anchor-ts build lock for {anchor_path}...")
                fcntl.flock(lf, fcntl.LOCK_EX)
                if not os.path.exists(anchor_path):
                    logger.info(
                        f"Building {anchor_path}: target ts for "
                        f"{len(self._positions):,} anchors"
                    )
                    anchor_ts = self.store.flat_timestamps[self._positions]
                    tmp = anchor_path + ".tmp.npy"
                    np.save(tmp, anchor_ts)
                    os.replace(tmp, anchor_path)
                    del anchor_ts
        self._anchor_ts = _load_npy_readonly(anchor_path)
        self._t_min = int(self._anchor_ts.min())
        self._t_max = int(self._anchor_ts.max())

    def num_windows(self) -> int:
        """Number of fixed-duration windows spanning [t_min, t_max]."""
        self._ensure_streaming_index()
        assert self._t_min is not None and self._t_max is not None
        span = self._t_max - self._t_min + 1
        w = self._streaming_window_seconds
        return int((span + w - 1) // w)

    def window_indices(
        self, ts: int, sort_by_time: Optional[bool] = None
    ) -> np.ndarray:
        """Global anchor indices (into ``_positions``) whose target timestamp is
        in window ``ts``: ``[t_min + ts*W, t_min + (ts+1)*W)``.

        Returned in ascending global-index order (user-major), which keeps the
        per-sample history scans page-local in the mmap'd event arrays. Used by
        the per-window path (via ``set_ts``) and the persistent path (shipped to
        workers through the sampler). ``sort_by_time`` defaults to
        ``streaming_sort_within_window``.

        Note: an O(log N) variant using a cached argsort of the timestamps was
        evaluated but rejected — it doubles resident mmap (sorted-ts + order
        permutation, ~52 GB) and that extra residency evicts the event-array
        page cache, stalling dataloader workers (NCCL watchdog timeouts). The
        O(N) mask here keeps only one ~26 GB array resident and is robust.
        """
        self._ensure_streaming_index()
        assert self._anchor_ts is not None and self._t_min is not None
        w = self._streaming_window_seconds
        lo = self._t_min + ts * w
        hi = lo + w
        idx = np.where((self._anchor_ts >= lo) & (self._anchor_ts < hi))[0]
        do_sort = (
            self._streaming_sort_within_window if sort_by_time is None else sort_by_time
        )
        if do_sort and idx.size > 0:
            idx = idx[np.argsort(self._anchor_ts[idx], kind="stable")]
        logger.warning(f"window_indices({ts}): [{lo}, {hi}) -> {idx.size:,} anchors")
        return idx.astype(np.int64)

    def set_ts(self, ts: int) -> None:
        """Restrict the active sample set to anchors in window ``ts`` (used by
        the per-window-DataLoader path, where ``iloc``/``get_item_count`` index
        through ``_active``).

        Forward-only temporal slicing for streaming train/eval. History for any
        anchor is still gathered causally (``scan_start:flat_pos``) and may span
        earlier windows, so there is no feature leakage from future events.
        """
        self._active = self.window_indices(ts)

    def load_query_samples(self, sample_list) -> None:
        max_num_candidates = (
            self._max_num_candidates_inference
            if self._is_inference
            else self._max_num_candidates
        )
        self.items_in_memory = {}
        for idx in sample_list:
            flat_pos = self.iloc(idx)
            self.items_in_memory[idx] = self._build_sample(flat_pos, max_num_candidates)
        self.last_loaded = time.time()

    def get_sample(self, idx: int) -> Tuple[KeyedJaggedTensor, KeyedJaggedTensor]:
        if idx in self.items_in_memory:
            return self.items_in_memory[idx]
        max_num_candidates = (
            self._max_num_candidates_inference
            if self._is_inference
            else self._max_num_candidates
        )
        flat_pos = self.iloc(idx)
        return self._build_sample(flat_pos, max_num_candidates)

    def _gather_interleaved_history(
        self, flat_pos: int, user_start: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build a single chronologically-ordered history sequence from the 3
        behavior pools. Each event's `action_weight` carries the pool bitmask
        (LP_BIT/LIKE_BIT/SKIP_BIT). Per-pool cap = history_length // 3."""
        L = self._history_length
        per_pool = max(1, L // 3)
        scan_start = max(int(user_start), int(flat_pos) - self._scan_window)
        scan_end = int(flat_pos)
        if scan_end <= scan_start:
            empty = np.empty(0, dtype=np.int64)
            return empty, empty, empty, empty, empty

        item_ids = self.store.flat_item_ids[scan_start:scan_end]
        timestamps = self.store.flat_timestamps[scan_start:scan_end]
        is_lp = self.store.flat_is_listen_plus[scan_start:scan_end]
        is_like = self.store.flat_is_like[scan_start:scan_end]
        is_skip = self.store.flat_is_skip[scan_start:scan_end]

        # Local indices into the scan window — preserves chronological order
        # within each pool and lets us interleave by re-sorting.
        idx_all = np.arange(item_ids.shape[0], dtype=np.int64)
        lp_idx = idx_all[is_lp][-per_pool:]
        like_idx = idx_all[is_like][-per_pool:]
        skip_idx = idx_all[is_skip][-per_pool:]

        keep_local = np.concatenate([lp_idx, like_idx, skip_idx])
        if keep_local.size == 0:
            empty = np.empty(0, dtype=np.int64)
            return empty, empty, empty, empty, empty

        order = np.argsort(keep_local, kind="stable")
        keep_local = keep_local[order]

        items = item_ids[keep_local]
        ts = timestamps[keep_local]
        artists = self.item_to_artist[np.clip(items, 0, self.item_to_artist.shape[0] - 1)]
        albums = self.item_to_album[np.clip(items, 0, self.item_to_album.shape[0] - 1)]

        # Pool bitmask per kept event (LP/LIKE/SKIP are mutually exclusive in
        # the source data, but OR is safe and forward-compatible).
        weight = np.zeros(keep_local.shape[0], dtype=np.int64)
        weight[is_lp[keep_local]] |= LP_BIT
        weight[is_like[keep_local]] |= LIKE_BIT
        weight[is_skip[keep_local]] |= SKIP_BIT

        return items, artists, albums, ts, weight

    def _build_sample(
        self, flat_pos: int, max_num_candidates: int
    ) -> Tuple[KeyedJaggedTensor, KeyedJaggedTensor]:
        uid = int(self.store.flat_uid[flat_pos])
        user_start = int(self.store.user_start[uid])

        items, artists, albums, ts, weight = self._gather_interleaved_history(
            flat_pos, user_start
        )

        target_item = int(self.store.flat_item_ids[flat_pos])
        target_artist = int(
            self.item_to_artist[target_item]
            if target_item < self.item_to_artist.shape[0]
            else 0
        )
        target_album = int(
            self.item_to_album[target_item]
            if target_item < self.item_to_album.shape[0]
            else 0
        )
        target_ts = int(self.store.flat_timestamps[flat_pos])

        played_ratio = float(self.store.flat_played_ratio[flat_pos])
        is_lp = (
            int(self.store.flat_event_types[flat_pos]) == LISTEN_TYPE
            and played_ratio >= LISTEN_PLUS_THRESHOLD
        )
        # Label encoded into the candidate's action_weight via the LP bit, so
        # _get_supervision_labels_and_weights sees the right supervision.
        candidate_action_weight = LP_BIT if is_lp else 0

        cross_id_anchor: Dict[str, int] = {
            "uid": uid,
            "item_id": target_item,
            "artist_id": target_artist,
            "album_id": target_album,
            "hour_of_day": int((target_ts // 3600) % 24),
            "is_organic": int(self.store.flat_is_organic[flat_pos]),
        }
        cross_ids: Dict[str, int] = {
            name: xxhash_cross(cross_id_anchor, list(keys), n, salt)
            for (name, keys, n, salt) in self._cross_specs
        }

        # ---- Truncate UIH to fit max_seq_len budget ----
        uih_seq_len_budget = (
            self._max_seq_len
            - max_num_candidates
            - len(self._contextual_feature_to_max_length or {})
        )
        if items.shape[0] > uih_seq_len_budget:
            items = items[-uih_seq_len_budget:]
            artists = artists[-uih_seq_len_budget:]
            albums = albums[-uih_seq_len_budget:]
            ts = ts[-uih_seq_len_budget:]
            weight = weight[-uih_seq_len_budget:]
        uih_seq_len = int(items.shape[0])
        dummy_watch_time = np.zeros(uih_seq_len, dtype=np.int64)

        # ---- Build UIH KJT ----
        # Contextual features (length-1 each) iterated in the same order as
        # `_contextual_feature_to_max_length` (matches movielens reference).
        uih_kjt_values: List[int] = []
        uih_kjt_lengths: List[int] = []
        for name, length in (self._contextual_feature_to_max_length or {}).items():
            assert length == 1, f"yambda contextuals are length-1, got {name}={length}"
            if name == "uid":
                uih_kjt_values.append(uid)
            else:
                uih_kjt_values.append(int(cross_ids[name]))
            uih_kjt_lengths.append(1)

        # Sequential features — order must match the trailing entries of
        # hstu_uih_feature_names in configs.py:
        #   item_id, artist_id, album_id, action_weight, action_timestamp, dummy_watch_time
        uih_kjt_values.extend(items.tolist())
        uih_kjt_values.extend(artists.tolist())
        uih_kjt_values.extend(albums.tolist())
        uih_kjt_values.extend(weight.tolist())
        uih_kjt_values.extend(ts.tolist())
        uih_kjt_values.extend(dummy_watch_time.tolist())
        n_sequential = len(self._uih_keys) - len(self._contextual_feature_to_max_length or {})
        uih_kjt_lengths.extend([uih_seq_len] * n_sequential)

        dummy_query_time = int(ts[-1]) if uih_seq_len > 0 else target_ts
        uih_kjt_values.append(dummy_query_time)
        uih_kjt_lengths.append(1)

        uih_features_kjt = KeyedJaggedTensor(
            keys=self._uih_keys + ["dummy_query_time"],
            lengths=torch.tensor(uih_kjt_lengths, dtype=torch.long),
            values=torch.tensor(uih_kjt_values, dtype=torch.long),
        )

        # ---- Build candidates KJT ----
        # Order must match configs.py:hstu_candidate_feature_names exactly:
        #   item_candidate_id, item_candidate_artist_id, item_candidate_album_id,
        #   item_query_time, item_action_weight, item_dummy_watchtime
        candidates_kjt_lengths = max_num_candidates * torch.ones(
            len(self._candidates_keys), dtype=torch.long
        )
        candidates_kjt_values: List[int] = (
            [target_item] * max_num_candidates
            + [target_artist] * max_num_candidates
            + [target_album] * max_num_candidates
            + [dummy_query_time] * max_num_candidates
            + [candidate_action_weight] * max_num_candidates
            + [1] * max_num_candidates  # item_dummy_watchtime
        )
        candidates_features_kjt = KeyedJaggedTensor(
            keys=self._candidates_keys,
            lengths=candidates_kjt_lengths,
            values=torch.tensor(candidates_kjt_values, dtype=torch.long),
        )
        return uih_features_kjt, candidates_features_kjt
