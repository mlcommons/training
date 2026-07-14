# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe
import argparse
import json
import logging
import os
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import urlretrieve

import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main")

"""
Usage:
    mkdir -p data/ && python3 preprocess_public_data.py --dataset kuairand-1k
    python3 preprocess_public_data.py --dataset yambda-5b   --data-path <root>
    python3 preprocess_public_data.py --dataset yambda-500m --data-path <root>
    python3 preprocess_public_data.py --dataset yambda-50m  --data-path <root>
"""

SUPPORTED_DATASETS = [
    "kuairand-1k",
    "kuairand-27k",
    "yambda-50m",
    "yambda-500m",
    "yambda-5b",
]


def get_feature_merge_weights(dataset: str = "debug") -> Dict[str, int]:
    if "kuairand" in dataset:
        return {
            "is_click": 1,
            "is_like": 2,
            "is_follow": 4,
            "is_comment": 8,
            "is_forward": 16,
            "is_hate": 32,
            "long_view": 64,
            "is_profile_enter": 128,
        }
    else:
        return {"dummy": 1}


class DataProcessor:
    def __init__(
        self,
        download_url: str,
        data_path: str,
        file_name: str,
        prefix: str,
    ) -> None:
        self._download_url = download_url
        self._data_path = data_path
        self._file_name = file_name
        self._prefix = prefix

    def download(self) -> None:
        return

    def preprocess(self) -> None:
        return

    def file_exists(self, name: str) -> bool:
        return os.path.isfile("%s/%s" % (os.getcwd(), name))


class DLRMKuaiRandProcessor(DataProcessor):
    def __init__(
        self,
        download_url: str,
        data_path: str,
        file_name: str,
        prefix: str,
    ) -> None:
        super().__init__(download_url, data_path, file_name, prefix)
        if prefix == "KuaiRand-1K":
            self._log_files: List[str] = [
                f"{data_path}{prefix}/data/log_standard_4_08_to_4_21_1k.csv",
                f"{data_path}{prefix}/data/log_standard_4_22_to_5_08_1k.csv",
            ]
            self._user_features_file: str = (
                f"{data_path}{prefix}/data/user_features_1k.csv"
            )
        elif prefix == "KuaiRand-27K":
            self._log_files: List[str] = [
                f"{data_path}{prefix}/data/log_standard_4_08_to_4_21_27k_part1.csv",
                f"{data_path}{prefix}/data/log_standard_4_08_to_4_21_27k_part2.csv",
                f"{data_path}{prefix}/data/log_standard_4_22_to_5_08_27k_part1.csv",
                f"{data_path}{prefix}/data/log_standard_4_22_to_5_08_27k_part2.csv",
            ]
            self._user_features_file: str = (
                f"{data_path}{prefix}/data/user_features_27k.csv"
            )
        self._output_file: str = f"{data_path}{prefix}/data/processed_seqs.csv"
        self._event_merge_weight: Dict[str, int] = get_feature_merge_weights(
            prefix.lower()
        )

    def download(self) -> None:
        file_path = f"{self._data_path}{self._file_name}"
        if not self.file_exists(file_path):
            log.info(f"Downloading {self._download_url}")
            urlretrieve(self._download_url, file_path)
            log.info(f"Downloaded to {file_path}")
            with tarfile.open(file_path, "r:*") as tar_ref:
                tar_ref.extractall(path=self._data_path)
                log.info("Data files extracted")
            os.remove(file_path)
            log.info("Tar file removed")

    def preprocess(self) -> None:
        self.download()
        log.info("Preprocessing data...")
        seq_cols = [
            "video_id",
            "time_ms",
            "action_weights",
            "play_time_ms",
            "duration_ms",
        ]
        df = None
        for idx, log_file in enumerate(self._log_files):
            log.info(f"Processing {log_file}...")
            log_df = pd.read_csv(
                log_file,
                delimiter=",",
            )
            df_grouped_by_user = log_df.groupby("user_id").agg(list).reset_index()

            for event, weight in self._event_merge_weight.items():
                df_grouped_by_user[event] = df_grouped_by_user[event].apply(
                    lambda seq: np.where(np.array(seq) == 0, 0, weight)
                )

            events = list(self._event_merge_weight.keys())
            df_grouped_by_user["action_weights"] = df_grouped_by_user.apply(
                lambda row: [int(sum(x)) for x in zip(*[row[col] for col in events])],
                axis=1,
            )
            df_grouped_by_user = df_grouped_by_user[["user_id"] + seq_cols]

            if idx == 0:
                df = df_grouped_by_user
            else:
                df = df.merge(df_grouped_by_user, on="user_id", suffixes=("_x", "_y"))
                for col in seq_cols:
                    df[col] = df.apply(
                        lambda row: row[col + "_x"] + row[col + "_y"], axis=1
                    )
                    df = df.drop(columns=[col + "_x", col + "_y"])

        max_seq_len = df["video_id"].apply(len).max()
        min_seq_len = df["video_id"].apply(len).min()
        average_seq_len = df["video_id"].apply(len).mean()
        log.info(f"{max_seq_len=}, {min_seq_len=}, {average_seq_len=}")

        log.info("Merging user features...")
        user_features_df = pd.read_csv(self._user_features_file, delimiter=",")

        def _one_hot_encode(row):
            mapping = {category: i + 1 for i, category in enumerate(row.unique())}
            row = row.map(mapping)
            return row

        for col in [
            "user_active_degree",
            "follow_user_num_range",
            "fans_user_num_range",
            "friend_user_num_range",
            "register_days_range",
        ]:
            user_features_df[col] = _one_hot_encode(user_features_df[col])

        final_df = pd.merge(df, user_features_df, on="user_id")
        final_df.to_csv(self._output_file, index=False, sep=",")
        log.info(f"Processed file saved to {self._output_file}")


# ----------------------------------------------------------------------------
# Yambda processor
# ----------------------------------------------------------------------------
#
# Yambda is hosted on HuggingFace at `yandex/yambda` and comes in three sizes:
# 50m, 500m, 5b. Each size shares the same catalog metadata (embeddings,
# artist/album mappings); only the interaction stream differs.
#
# This processor:
#   1) Downloads `multi_event.parquet` for the chosen size + the catalog
#      metadata files via the `datasets` library.
#   2) Encodes event_type strings into uint8.
#   3) Splits temporally into train + test (Global Temporal Split, GTS).
#   4) Builds per-user sessions by inactivity gap.
#   5) Computes item popularity counts.
#   6) Writes the layout expected by `DLRMv3YambdaDataset`:
#
#      <data_path>/processed_<size>/
#          train_sessions.parquet
#          test_events.parquet
#          session_index.parquet
#          item_popularity.npy
#          split_meta.json
#
#      <data_path>/shared_metadata/
#          artist_item_mapping.parquet
#          album_item_mapping.parquet
#          embeddings.parquet           (optional; not used by HSTU training)
#
# The HSTU training path then auto-builds an `hstu_cache_L<N>/` mmap under
# `processed_<size>/` on first use.
# ----------------------------------------------------------------------------

YAMBDA_HF_REPO = "yandex/yambda"
YAMBDA_SIZES = {"yambda-50m": "50m", "yambda-500m": "500m", "yambda-5b": "5b"}
YAMBDA_METADATA_FILES = (
    "artist_item_mapping",
    "album_item_mapping",
    "embeddings",
)

# Yambda timestamps are seconds (rounded to 5s boundaries).
SECONDS_PER_DAY = 86400
# Polars chunk size for streaming the 5b parquet (~150 GB on disk).
YAMBDA_CHUNK_SIZE = 10_000_000
EVENT_TYPE_MAP = {"listen": 0, "like": 1, "dislike": 2, "unlike": 3, "undislike": 4}


class DLRMYambdaProcessor(DataProcessor):
    """Download + preprocess Yambda (50m / 500m / 5b) for DLRMv3YambdaDataset."""

    def __init__(
        self,
        data_path: str,
        size: str,
        session_gap_seconds: int = 1800,
        train_days: int = 300,
        gap_minutes: int = 30,
        test_days: int = 1,
    ) -> None:
        assert size in {"50m", "500m", "5b"}, f"unknown yambda size {size}"
        super().__init__(
            download_url="",  # download is via HuggingFace `datasets` lib
            data_path=data_path.rstrip("/") + "/",
            file_name=f"{size}/multi_event.parquet",
            prefix=f"yambda-{size}",
        )
        self._size: str = size
        self._raw_dir: Path = Path(self._data_path) / "raw"
        self._processed_dir: Path = Path(self._data_path) / f"processed_{size}"
        self._shared_dir: Path = Path(self._data_path) / "shared_metadata"
        self._session_gap_seconds: int = session_gap_seconds
        self._train_days: int = train_days
        self._gap_minutes: int = gap_minutes
        self._test_days: int = test_days

    def download(self) -> None:
        try:
            from datasets import DatasetDict, load_dataset
        except ImportError as e:
            raise ImportError(
                "Downloading Yambda requires the `datasets` package "
                "(`pip install datasets`)."
            ) from e

        self._raw_dir.mkdir(parents=True, exist_ok=True)
        self._shared_dir.mkdir(parents=True, exist_ok=True)

        # Size-specific interaction stream.
        event_path = self._raw_dir / self._size / "multi_event.parquet"
        if not event_path.exists():
            event_path.parent.mkdir(parents=True, exist_ok=True)
            log.info(
                f"Downloading multi_event.parquet for {self._size} "
                f"from {YAMBDA_HF_REPO} ..."
            )
            ds = load_dataset(
                YAMBDA_HF_REPO,
                data_dir=f"flat/{self._size}",
                data_files="multi_event.parquet",
            )
            assert isinstance(ds, DatasetDict)
            ds["train"].to_parquet(str(event_path))
            log.info(f"Saved {event_path}")
        else:
            log.info(f"Already exists: {event_path}")

        # Catalog metadata files (shared across sizes).
        for name in YAMBDA_METADATA_FILES:
            shared_path = self._shared_dir / f"{name}.parquet"
            if shared_path.exists():
                log.info(f"Already exists: {shared_path}")
                continue
            log.info(f"Downloading {name}.parquet from {YAMBDA_HF_REPO} ...")
            ds = load_dataset(YAMBDA_HF_REPO, data_files=f"{name}.parquet")
            assert isinstance(ds, DatasetDict)
            ds["train"].to_parquet(str(shared_path))
            log.info(f"Saved {shared_path}")

    def preprocess(self) -> None:
        self.download()
        try:
            import polars as pl
        except ImportError as e:
            raise ImportError(
                "Yambda preprocessing requires polars "
                "(`pip install polars-u64-idx` is recommended for the 5b "
                "variant — stock polars overflows its 32-bit row index)."
            ) from e

        self._processed_dir.mkdir(parents=True, exist_ok=True)
        event_path = self._raw_dir / self._size / "multi_event.parquet"

        log.info(f"Loading multi_event from {event_path} ...")
        events = self._load_events(pl, event_path)
        log.info(f"Loaded {len(events):,} events")

        events = self._encode_event_types(pl, events)
        t_min = int(events["timestamp"].min())
        t_max = int(events["timestamp"].max())
        log.info(
            f"Timestamp range: {t_min}..{t_max} "
            f"({(t_max - t_min) / SECONDS_PER_DAY:.1f} days)"
        )

        train_start, train_end, test_start, test_end = self._split_boundaries(t_max)
        log.info(
            f"GTS train=[{train_start},{train_end}) gap=[{train_end},{test_start}) "
            f"test=[{test_start},{test_end})"
        )
        train_events, test_events = self._temporal_split(
            pl, events, train_start, train_end, test_start, test_end
        )
        log.info(
            f"Train: {len(train_events):,} events, Test: {len(test_events):,} events"
        )

        gap_units = self._session_gap_seconds  # 1 unit = 1 second
        sessions = self._build_sessions(pl, train_events, gap_units)
        log.info(f"Built {len(sessions):,} sessions")

        session_index = self._build_session_index(pl, sessions)
        log.info(f"Session index covers {len(session_index):,} users")

        item_popularity = self._compute_item_popularity(train_events)

        sessions.write_parquet(str(self._processed_dir / "train_sessions.parquet"))
        test_events.write_parquet(str(self._processed_dir / "test_events.parquet"))
        session_index.write_parquet(str(self._processed_dir / "session_index.parquet"))
        np.save(self._processed_dir / "item_popularity.npy", item_popularity)

        with open(self._processed_dir / "split_meta.json", "w") as f:
            json.dump(
                {
                    "size": self._size,
                    "t_min": t_min,
                    "t_max": t_max,
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end,
                    "train_days": self._train_days,
                    "gap_minutes": self._gap_minutes,
                    "test_days": self._test_days,
                    "session_gap_seconds": self._session_gap_seconds,
                    "num_train_events": int(len(train_events)),
                    "num_test_events": int(len(test_events)),
                    "num_sessions": int(len(sessions)),
                    "num_users": int(len(session_index)),
                },
                f,
                indent=2,
            )
        log.info(f"Preprocessing complete: {self._processed_dir}")

    # ------- helpers --------

    def _load_events(self, pl, parquet_path: Path):
        # 5b is too large to load in one polars pass on most boxes (~150 GB
        # peak in-RAM with eager read). Stream in 10M-row chunks for safety.
        if self._size == "5b":
            log.info(f"Streaming load (chunk_size={YAMBDA_CHUNK_SIZE:,})...")
            lf = pl.scan_parquet(parquet_path)
            n = lf.select(pl.len()).collect().item()
            log.info(f"Total rows: {n:,}")
            chunks = []
            for off in range(0, n, YAMBDA_CHUNK_SIZE):
                chunk = lf.slice(off, YAMBDA_CHUNK_SIZE).collect()
                chunks.append(chunk)
                log.info(f"  loaded {off:,}..{off + len(chunk):,}")
            return pl.concat(chunks)
        return pl.read_parquet(parquet_path)

    def _encode_event_types(self, pl, events):
        dt = events["event_type"].dtype
        if dt == pl.Utf8 or isinstance(dt, (pl.Categorical, pl.Enum)):
            events = events.with_columns(
                pl.col("event_type")
                .cast(pl.Utf8)
                .replace_strict(EVENT_TYPE_MAP)
                .cast(pl.UInt8)
                .alias("event_type")
            )
        return events

    def _split_boundaries(self, t_max: int) -> Tuple[int, int, int, int]:
        test_end = t_max
        test_start = test_end - self._test_days * SECONDS_PER_DAY
        train_end = test_start - self._gap_minutes * 60
        train_start = train_end - self._train_days * SECONDS_PER_DAY
        return train_start, train_end, test_start, test_end

    def _temporal_split(self, pl, events, train_start, train_end, test_start, test_end):
        train = events.filter(
            (pl.col("timestamp") >= train_start) & (pl.col("timestamp") < train_end)
        )
        test_all = events.filter(
            (pl.col("timestamp") >= test_start) & (pl.col("timestamp") < test_end)
        )
        # Test users must also appear in train (next-item prediction setup).
        train_users = train.select("uid").unique()
        test = test_all.join(train_users, on="uid", how="inner")
        return train, test

    def _build_sessions(self, pl, events, session_gap_units: int):
        sorted_events = events.sort(["uid", "timestamp"])
        return (
            sorted_events
            .with_columns(
                (
                    (pl.col("timestamp").diff().fill_null(0) > session_gap_units)
                    .cast(pl.UInt32)
                    .cum_sum()
                )
                .over("uid")
                .alias("session_id")
            )
            .group_by(["uid", "session_id"])
            .agg(
                pl.col("item_id").alias("item_ids"),
                pl.col("timestamp").alias("timestamps"),
                pl.col("event_type").alias("event_types"),
                pl.col("is_organic").alias("is_organic"),
                pl.col("played_ratio_pct").alias("played_ratio_pct"),
                pl.col("track_length_seconds").alias("track_length_seconds"),
            )
            .sort(["uid", "session_id"])
        )

    def _build_session_index(self, pl, sessions):
        return (
            sessions
            .with_columns(pl.col("item_ids").list.len().alias("session_len"))
            .group_by("uid")
            .agg(
                pl.col("session_id").alias("session_ids"),
                pl.col("session_len").alias("session_lens"),
                pl.col("session_len").cum_sum().alias("session_offsets"),
            )
            .sort("uid")
        )

    def _compute_item_popularity(self, train_events) -> np.ndarray:
        counts = (
            train_events
            .group_by("item_id")
            .len()
            .sort("item_id")
        )
        max_item = int(counts["item_id"].max())
        popularity = np.zeros(max_item + 1, dtype=np.int64)
        popularity[counts["item_id"].to_numpy()] = counts["len"].to_numpy()
        return popularity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=SUPPORTED_DATASETS,
        required=True,
        help="dataset",
    )
    parser.add_argument(
        "--data-path",
        default="data/",
        help=(
            "Root directory for raw + processed data. KuaiRand defaults to "
            "the existing `data/` convention; Yambda defaults to `data/` too "
            "but is commonly overridden to a shared filesystem location with "
            "enough space for the 5b variant (~500 GB)."
        ),
    )
    args = parser.parse_args()

    data_path = args.data_path.rstrip("/") + "/"

    if args.dataset == "kuairand-1k":
        DLRMKuaiRandProcessor(
            download_url="https://zenodo.org/records/10439422/files/KuaiRand-1K.tar.gz",
            data_path=data_path,
            file_name="KuaiRand-1K.tar.gz",
            prefix="KuaiRand-1K",
        ).preprocess()
    elif args.dataset == "kuairand-27k":
        DLRMKuaiRandProcessor(
            download_url="https://zenodo.org/records/10439422/files/KuaiRand-27K.tar.gz",
            data_path=data_path,
            file_name="KuaiRand-27K.tar.gz",
            prefix="KuaiRand-27K",
        ).preprocess()
    elif args.dataset in YAMBDA_SIZES:
        DLRMYambdaProcessor(
            data_path=data_path,
            size=YAMBDA_SIZES[args.dataset],
        ).preprocess()


if __name__ == "__main__":
    main()
