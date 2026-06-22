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

# pyre-strict
"""
Configuration module for DLRMv3 model.

This module provides configuration functions for the HSTU model architecture and embedding table configurations.
"""

import hashlib
import math
import os
from typing import Callable, Dict, Optional, Tuple

import gin
import torch

from generative_recommenders.modules.dlrm_hstu import DlrmHSTUConfig
from generative_recommenders.modules.multitask_module import (
    MultitaskTaskType,
    TaskConfig,
)
from torchrec.modules.embedding_configs import DataType, EmbeddingConfig

HSTU_EMBEDDING_DIM = 512  # final DLRMv3 model
HASH_SIZE = 10_000_000
HASH_SIZE_1B = 1_000_000_000

# (name, keys, num_embeddings, salt) — single source of truth for both
# get_embedding_table_config("yambda-5b") and the dataset's cross-hash inputs.
# Sizes mirror Primus-DLRM/configs/bench_onetrans_large_5b_cross_feat_shampoo.yaml.
YAMBDA_5B_CROSS_SPECS = [
    ("user_x_artist",        ("uid", "artist_id"),                  100_000_000, 0),
    ("user_x_album",         ("uid", "album_id"),                    40_000_000, 0),
    ("user_x_hour",          ("uid", "hour_of_day"),                 24_000_000, 0),
    ("item_x_hour",          ("item_id", "hour_of_day"),             40_000_000, 0),
    ("artist_x_hour",        ("artist_id", "hour_of_day"),           32_000_000, 0),
    ("user_x_is_organic",    ("uid", "is_organic"),                   2_000_000, 0),
    ("user_x_artist_x_hour", ("uid", "artist_id", "hour_of_day"),    40_000_000, 0),
]


@gin.configurable
def get_hstu_configs(
    dataset: str = "debug",
    max_seq_len: Optional[int] = None,
    max_num_candidates: Optional[int] = None,
    hstu_embedding_table_dim: Optional[int] = None,
    hstu_transducer_embedding_dim: Optional[int] = None,
    hstu_num_heads: Optional[int] = None,
    hstu_attn_num_layers: Optional[int] = None,
    hstu_attn_linear_dim: Optional[int] = None,
    hstu_attn_qk_dim: Optional[int] = None,
    hstu_input_dropout_ratio: Optional[float] = None,
    hstu_linear_dropout_rate: Optional[float] = None,
) -> DlrmHSTUConfig:
    """
    Create and return HSTU model configuration.

    Builds a complete DlrmHSTUConfig with default hyperparameters for the HSTU
    architecture including attention settings, embedding dimensions, dropout rates,
    and feature name mappings.

    Args:
        dataset: Dataset identifier (currently unused, reserved for dataset-specific configs).

    Returns:
        DlrmHSTUConfig: Complete configuration object for the HSTU model.
    """
    hstu_config = DlrmHSTUConfig(
        hstu_num_heads=4,
        hstu_attn_linear_dim=128,
        hstu_attn_qk_dim=128,
        hstu_attn_num_layers=5,
        hstu_embedding_table_dim=HSTU_EMBEDDING_DIM,
        hstu_preprocessor_hidden_dim=256,
        hstu_transducer_embedding_dim=512,
        hstu_group_norm=False,
        hstu_input_dropout_ratio=0.2,
        hstu_linear_dropout_rate=0.1,
        causal_multitask_weights=0.2,
    )
    if "movielens" in dataset:
        assert dataset in [
            "movielens-1m",
            "movielens-20m",
            "movielens-13b",
            "movielens-18b",
        ]
        hstu_config.user_embedding_feature_names = (
            [
                "movie_id",
                "user_id",
                "sex",
                "age_group",
                "occupation",
                "zip_code",
            ]
            if dataset == "movielens-1m"
            else [
                "movie_id",
                "user_id",
            ]
        )
        hstu_config.item_embedding_feature_names = [
            "item_movie_id",
        ]
        hstu_config.uih_post_id_feature_name = "movie_id"
        hstu_config.uih_action_time_feature_name = "action_timestamp"
        hstu_config.candidates_querytime_feature_name = "item_query_time"
        hstu_config.candidates_weight_feature_name = "item_action_weights"
        hstu_config.uih_weight_feature_name = "item_weights"
        hstu_config.candidates_watchtime_feature_name = "item_movie_rating"
        hstu_config.action_weights = [1, 2, 4, 8, 16]
        hstu_config.contextual_feature_to_max_length = (
            {
                "user_id": 1,
                "sex": 1,
                "age_group": 1,
                "occupation": 1,
                "zip_code": 1,
            }
            if dataset == "movielens-1m"
            else {
                "user_id": 1,
            }
        )
        hstu_config.contextual_feature_to_min_uih_length = (
            {
                "user_id": 20,
                "sex": 20,
                "age_group": 20,
                "occupation": 20,
                "zip_code": 20,
            }
            if dataset == "movielens-1m"
            else {
                "user_id": 20,
            }
        )
        hstu_config.merge_uih_candidate_feature_mapping = [
            ("movie_id", "item_movie_id"),
            ("movie_rating", "item_movie_rating"),
            ("action_timestamp", "item_query_time"),
            ("item_weights", "item_action_weights"),
            ("dummy_watch_time", "item_dummy_watchtime"),
        ]
        hstu_config.hstu_uih_feature_names = (
            [
                "user_id",
                "sex",
                "age_group",
                "occupation",
                "zip_code",
                "movie_id",
                "movie_rating",
                "action_timestamp",
                "item_weights",
                "dummy_watch_time",
            ]
            if dataset == "movielens-1m"
            else [
                "user_id",
                "movie_id",
                "movie_rating",
                "action_timestamp",
                "item_weights",
                "dummy_watch_time",
            ]
        )
        hstu_config.hstu_candidate_feature_names = [
            "item_movie_id",
            "item_movie_rating",
            "item_query_time",
            "item_action_weights",
            "item_dummy_watchtime",
        ]
        hstu_config.max_num_candidates = 10
        hstu_config.max_num_candidates_inference = (
            5 if dataset not in ["movielens-13b", "movielens-18b"] else 2048
        )
        hstu_config.multitask_configs = [
            TaskConfig(
                task_name="rating",
                task_weight=1,
                task_type=MultitaskTaskType.BINARY_CLASSIFICATION,
            )
        ]
    elif "streaming" in dataset:
        hstu_config.user_embedding_feature_names = [
            "item_id",
            "user_id",
            "item_category_id",
        ]
        hstu_config.item_embedding_feature_names = [
            "item_candidate_id",
            "item_candidate_category_id",
        ]
        hstu_config.uih_post_id_feature_name = "item_id"
        hstu_config.uih_action_time_feature_name = "action_timestamp"
        hstu_config.candidates_querytime_feature_name = "item_query_time"
        hstu_config.candidates_weight_feature_name = "item_action_weights"
        hstu_config.uih_weight_feature_name = "item_weights"
        hstu_config.candidates_watchtime_feature_name = "item_rating"
        hstu_config.action_weights = [1, 2, 4, 8, 16]
        hstu_config.action_embedding_init_std = 5.0
        hstu_config.contextual_feature_to_max_length = {"user_id": 1}
        hstu_config.contextual_feature_to_min_uih_length = {"user_id": 20}
        hstu_config.merge_uih_candidate_feature_mapping = [
            ("item_id", "item_candidate_id"),
            ("item_rating", "item_candidate_rating"),
            ("action_timestamp", "item_query_time"),
            ("item_weights", "item_action_weights"),
            ("dummy_watch_time", "item_dummy_watchtime"),
            ("item_category_id", "item_candidate_category_id"),
        ]
        hstu_config.hstu_uih_feature_names = [
            "user_id",
            "item_id",
            "item_rating",
            "action_timestamp",
            "item_weights",
            "dummy_watch_time",
            "item_category_id",
        ]
        hstu_config.hstu_candidate_feature_names = [
            "item_candidate_id",
            "item_candidate_rating",
            "item_query_time",
            "item_action_weights",
            "item_dummy_watchtime",
            "item_candidate_category_id",
        ]
        hstu_config.max_num_candidates = 32
        hstu_config.max_num_candidates_inference = 2048
        hstu_config.multitask_configs = [
            TaskConfig(
                task_name="rating",
                task_weight=1,
                task_type=MultitaskTaskType.BINARY_CLASSIFICATION,
            )
        ]
    elif "kuairand" in dataset:
        hstu_config.user_embedding_feature_names = [
            "video_id",
            "user_id",
            "user_active_degree",
            "follow_user_num_range",
            "fans_user_num_range",
            "friend_user_num_range",
            "register_days_range",
        ]
        hstu_config.item_embedding_feature_names = [
            "item_video_id",
        ]
        hstu_config.uih_post_id_feature_name = "video_id"
        hstu_config.uih_action_time_feature_name = "action_timestamp"
        hstu_config.candidates_querytime_feature_name = "item_query_time"
        hstu_config.uih_weight_feature_name = "action_weight"
        hstu_config.candidates_weight_feature_name = "item_action_weight"
        hstu_config.candidates_watchtime_feature_name = "item_target_watchtime"
        # There are more contextual features in the dataset, see https://kuairand.com/ for details
        hstu_config.contextual_feature_to_max_length = {
            "user_id": 1,
            "user_active_degree": 1,
            "follow_user_num_range": 1,
            "fans_user_num_range": 1,
            "friend_user_num_range": 1,
            "register_days_range": 1,
        }
        hstu_config.merge_uih_candidate_feature_mapping = [
            ("video_id", "item_video_id"),
            ("action_timestamp", "item_query_time"),
            ("action_weight", "item_action_weight"),
            ("watch_time", "item_target_watchtime"),
        ]
        hstu_config.hstu_uih_feature_names = [
            "user_id",
            "user_active_degree",
            "follow_user_num_range",
            "fans_user_num_range",
            "friend_user_num_range",
            "register_days_range",
            "video_id",
            "action_timestamp",
            "action_weight",
            "watch_time",
        ]
        hstu_config.hstu_candidate_feature_names = [
            "item_video_id",
            "item_action_weight",
            "item_target_watchtime",
            "item_query_time",
        ]
        hstu_config.multitask_configs = [
            TaskConfig(
                task_name="is_click",
                task_weight=1,
                task_type=MultitaskTaskType.BINARY_CLASSIFICATION,
            ),
            TaskConfig(
                task_name="is_like",
                task_weight=2,
                task_type=MultitaskTaskType.BINARY_CLASSIFICATION,
            ),
            TaskConfig(
                task_name="is_follow",
                task_weight=4,
                task_type=MultitaskTaskType.BINARY_CLASSIFICATION,
            ),
            TaskConfig(
                task_name="is_comment",
                task_weight=8,
                task_type=MultitaskTaskType.BINARY_CLASSIFICATION,
            ),
            TaskConfig(
                task_name="is_forward",
                task_weight=16,
                task_type=MultitaskTaskType.BINARY_CLASSIFICATION,
            ),
            TaskConfig(
                task_name="is_hate",
                task_weight=32,
                task_type=MultitaskTaskType.BINARY_CLASSIFICATION,
            ),
            TaskConfig(
                task_name="long_view",
                task_weight=64,
                task_type=MultitaskTaskType.BINARY_CLASSIFICATION,
            ),
            TaskConfig(
                task_name="is_profile_enter",
                task_weight=128,
                task_type=MultitaskTaskType.BINARY_CLASSIFICATION,
            ),
        ]
        hstu_config.action_weights = [1, 2, 4, 8, 16, 32, 64, 128]
    elif "yambda" in dataset:
        assert dataset in ["yambda-5b"]
        cross_names = [name for (name, _k, _n, _s) in YAMBDA_5B_CROSS_SPECS]
        # Per-table dim defaults to HSTU_EMBEDDING_DIM (512); override via the
        # `get_hstu_configs.hstu_embedding_table_dim = N` gin binding if needed.
        # Note: the embedding tables in get_embedding_table_config also use
        # HSTU_EMBEDDING_DIM and must stay aligned with this value.
        hstu_config.hstu_embedding_table_dim = HSTU_EMBEDDING_DIM
        hstu_config.hstu_transducer_embedding_dim = 512
        hstu_config.max_seq_len = 8192
        hstu_config.max_num_candidates = 1
        hstu_config.max_num_candidates_inference = 1
        # Per dlrm_hstu convention (see streaming-100b/movielens):
        #  - user_embedding_feature_names = UIH-side post-id features + contextual features.
        #    After main_forward merges UIH + candidate, only these entries hold the merged
        #    sequence (used by user-side transducer).
        #  - item_embedding_feature_names = candidate-side names only. _item_forward
        #    concats these along dim=-1 to feed the item MLP (per-candidate, not per-position).
        hstu_config.user_embedding_feature_names = (
            ["uid"]
            + cross_names
            + ["item_id", "artist_id", "album_id"]
        )
        hstu_config.item_embedding_feature_names = [
            "item_candidate_id",
            "item_candidate_artist_id",
            "item_candidate_album_id",
        ]
        hstu_config.uih_post_id_feature_name = "item_id"
        hstu_config.uih_action_time_feature_name = "action_timestamp"
        hstu_config.uih_weight_feature_name = "action_weight"
        hstu_config.candidates_querytime_feature_name = "item_query_time"
        hstu_config.candidates_weight_feature_name = "item_action_weight"
        hstu_config.candidates_watchtime_feature_name = "item_dummy_watchtime"
        hstu_config.action_weights = [1, 2, 4]  # lp, like, skip bits
        hstu_config.contextual_feature_to_max_length = {
            "uid": 1,
            **{name: 1 for name in cross_names},
        }
        hstu_config.contextual_feature_to_min_uih_length = {
            "uid": 0,
            **{name: 0 for name in cross_names},
        }
        # uih names map to candidate names (no name collisions allowed):
        # item_id/artist_id/album_id appear with prefix "item_" on candidate side.
        hstu_config.merge_uih_candidate_feature_mapping = [
            ("item_id", "item_candidate_id"),
            ("artist_id", "item_candidate_artist_id"),
            ("album_id", "item_candidate_album_id"),
            ("action_weight", "item_action_weight"),
            ("action_timestamp", "item_query_time"),
            ("dummy_watch_time", "item_dummy_watchtime"),
        ]
        hstu_config.hstu_uih_feature_names = (
            ["uid"]
            + cross_names
            + [
                "item_id",
                "artist_id",
                "album_id",
                "action_weight",
                "action_timestamp",
                "dummy_watch_time",
            ]
        )
        hstu_config.hstu_candidate_feature_names = [
            "item_candidate_id",
            "item_candidate_artist_id",
            "item_candidate_album_id",
            "item_query_time",
            "item_action_weight",
            "item_dummy_watchtime",
        ]
        hstu_config.multitask_configs = [
            TaskConfig(
                task_name="listen_plus",
                task_weight=1,  # matches action_weights[0] (lp bit)
                task_type=MultitaskTaskType.BINARY_CLASSIFICATION,
            )
        ]
    else:
        hstu_config.user_embedding_feature_names = [
            "uih_post_id",
            "uih_owner_id",
            "viewer_id",
            "dummy_contexual",
        ]
        hstu_config.item_embedding_feature_names = [
            "item_post_id",
            "item_owner_id",
        ]
        hstu_config.uih_post_id_feature_name = "uih_post_id"
        hstu_config.uih_action_time_feature_name = "uih_action_time"
        hstu_config.candidates_querytime_feature_name = "item_query_time"
        hstu_config.candidates_weight_feature_name = "item_action_weight"
        hstu_config.candidates_watchtime_feature_name = "item_target_watchtime"
        hstu_config.contextual_feature_to_max_length = {
            "viewer_id": 1,
            "dummy_contexual": 1,
        }
        hstu_config.contextual_feature_to_min_uih_length = {
            "viewer_id": 128,
            "dummy_contexual": 128,
        }
        hstu_config.merge_uih_candidate_feature_mapping = [
            ("uih_post_id", "item_post_id"),
            ("uih_owner_id", "item_owner_id"),
            ("uih_action_time", "item_query_time"),
            ("uih_weight", "item_action_weight"),
            ("uih_watchtime", "item_target_watchtime"),
            ("uih_video_length", "item_video_length"),
            ("uih_surface_type", "item_surface_type"),
        ]
        hstu_config.hstu_uih_feature_names = [
            "uih_post_id",
            "uih_action_time",
            "uih_weight",
            "uih_owner_id",
            "uih_watchtime",
            "uih_surface_type",
            "uih_video_length",
            "viewer_id",
            "dummy_contexual",
        ]
        hstu_config.hstu_candidate_feature_names = [
            "item_post_id",
            "item_owner_id",
            "item_surface_type",
            "item_video_length",
            "item_action_weight",
            "item_target_watchtime",
            "item_query_time",
        ]
        hstu_config.multitask_configs = [
            TaskConfig(
                task_name="vvp100",
                task_weight=1,
                task_type=MultitaskTaskType.BINARY_CLASSIFICATION,
            )
        ]

    # Apply gin overrides last so a value set in the gin file wins over the
    # per-dataset defaults above. Anything left as None inherits the default
    # the dataset branch (or DlrmHSTUConfig) chose. Example in a gin file:
    #   get_hstu_configs.max_seq_len = 4096
    #   get_hstu_configs.hstu_embedding_table_dim = 256
    _gin_overrides = {
        "max_seq_len": max_seq_len,
        "max_num_candidates": max_num_candidates,
        "max_num_candidates_inference": max_num_candidates,
        "hstu_embedding_table_dim": hstu_embedding_table_dim,
        "hstu_transducer_embedding_dim": hstu_transducer_embedding_dim,
        "hstu_num_heads": hstu_num_heads,
        "hstu_attn_num_layers": hstu_attn_num_layers,
        "hstu_attn_linear_dim": hstu_attn_linear_dim,
        "hstu_attn_qk_dim": hstu_attn_qk_dim,
        "hstu_input_dropout_ratio": hstu_input_dropout_ratio,
        "hstu_linear_dropout_rate": hstu_linear_dropout_rate,
    }
    for _name, _val in _gin_overrides.items():
        if _val is not None:
            setattr(hstu_config, _name, _val)

    return hstu_config


def _stable_table_seed(init_seed: int, table_name: str) -> int:
    """Deterministic 63-bit seed from (init_seed, table_name).

    Uses sha256 (not Python's salted built-in ``hash()``) so the per-table seed
    is identical across processes/ranks/runs for a given ``$SEED`` + table name.
    """
    digest = hashlib.sha256(f"{init_seed}:{table_name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF


def _uniform_init_bounds(cfg: EmbeddingConfig) -> Tuple[float, float]:
    """Mirror TorchREC's default per-table init bounds.

    TorchREC falls back to ``uniform_(-1/sqrt(N), +1/sqrt(N))`` when a table does
    not set ``weight_init_min/max``; honor any explicit bounds the config carries.
    """
    bound = math.sqrt(1.0 / cfg.num_embeddings)
    lo = -bound if cfg.weight_init_min is None else cfg.weight_init_min
    hi = bound if cfg.weight_init_max is None else cfg.weight_init_max
    return lo, hi


def _make_seeded_uniform_init(
    table_seed: int, lo: float, hi: float
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build a seeded in-place uniform initializer for one table's weight.

    TorchREC/FBGEMM calls ``init_fn`` with the (per-rank) local shard tensor on
    its compute device, so we seed a generator on that same device. For a fixed
    sharding plan (world size + plan unchanged) this makes embedding init
    byte-reproducible run-to-run.
    """

    def _init(weight: torch.Tensor) -> torch.Tensor:
        # TorchREC builds the unsharded EmbeddingCollection on the META device
        # first (DMP materializes real storage on the compute device later).
        # Meta tensors have no storage and torch.Generator(device="meta") is
        # invalid ("META device type not an accelerator"), so skip them: the
        # seeded init for the sharded/fused TBE path is provided by the RNG
        # re-seed right before DMP in make_optimizer_and_shard. On a real
        # device (eager/non-meta path) we still apply the per-table seeded fill.
        if weight.device.type == "meta":
            return weight
        gen = torch.Generator(device=weight.device)
        gen.manual_seed(table_seed)
        with torch.no_grad():
            weight.uniform_(lo, hi, generator=gen)
        return weight

    return _init


@gin.configurable
def get_embedding_table_config(
    dataset: str = "debug",
    embedding_dim: Optional[int] = None,
    init_seed: Optional[int] = None,
) -> Dict[str, EmbeddingConfig]:
    """
    Create and return embedding table configurations.

    Defines the embedding table configurations for item IDs, category IDs, and user IDs
    with their respective dimensions and data types.

    Args:
        dataset: Dataset identifier (currently unused, reserved for dataset-specific configs).
        embedding_dim: Per-table embedding width override. When set via gin
            (e.g. `get_embedding_table_config.embedding_dim = 256`), wins over
            `HSTU_EMBEDDING_DIM`. Keep in sync with the matching gin override on
            `get_hstu_configs.hstu_embedding_table_dim` — the model and the
            tables must agree on dim or sharding will reject the plan.
        init_seed: Base seed for the per-table seeded `init_fn` (Tier 1
            reproducible embedding init). When None, falls back to `$SEED`
            (default 1), matching `seed_everything`. Each table draws from a
            generator seeded by `sha256(init_seed, table_name)` so init is
            reproducible run-to-run for a fixed sharding plan.

    Returns:
        Dict mapping table names to their EmbeddingConfig objects.
    """
    tables = _build_embedding_table_config(dataset=dataset, embedding_dim=embedding_dim)

    if init_seed is None:
        init_seed = int(os.environ.get("SEED", "1"))
    for name, cfg in tables.items():
        lo, hi = _uniform_init_bounds(cfg)
        cfg.init_fn = _make_seeded_uniform_init(
            _stable_table_seed(init_seed, name), lo, hi
        )
    return tables


def _build_embedding_table_config(
    dataset: str = "debug",
    embedding_dim: Optional[int] = None,
) -> Dict[str, EmbeddingConfig]:
    DIM = embedding_dim if embedding_dim is not None else HSTU_EMBEDDING_DIM
    if "movielens" in dataset:
        assert dataset in [
            "movielens-1m",
            "movielens-20m",
            "movielens-13b",
            "movielens-18b",
        ]
        return (
            {
                "movie_id": EmbeddingConfig(
                    num_embeddings=HASH_SIZE,
                    embedding_dim=DIM,
                    name="movie_id",
                    data_type=DataType.FP16,
                    feature_names=["movie_id", "item_movie_id"],
                ),
                "user_id": EmbeddingConfig(
                    num_embeddings=HASH_SIZE,
                    embedding_dim=DIM,
                    name="user_id",
                    data_type=DataType.FP16,
                    feature_names=["user_id"],
                ),
                "sex": EmbeddingConfig(
                    num_embeddings=HASH_SIZE,
                    embedding_dim=DIM,
                    name="sex",
                    data_type=DataType.FP16,
                    feature_names=["sex"],
                ),
                "age_group": EmbeddingConfig(
                    num_embeddings=HASH_SIZE,
                    embedding_dim=DIM,
                    name="age_group",
                    data_type=DataType.FP16,
                    feature_names=["age_group"],
                ),
                "occupation": EmbeddingConfig(
                    num_embeddings=HASH_SIZE,
                    embedding_dim=DIM,
                    name="occupation",
                    data_type=DataType.FP16,
                    feature_names=["occupation"],
                ),
                "zip_code": EmbeddingConfig(
                    num_embeddings=HASH_SIZE,
                    embedding_dim=DIM,
                    name="zip_code",
                    data_type=DataType.FP16,
                    feature_names=["zip_code"],
                ),
            }
            if dataset == "movielens-1m"
            else {
                "movie_id": EmbeddingConfig(
                    num_embeddings=HASH_SIZE_1B,
                    embedding_dim=DIM,
                    name="movie_id",
                    data_type=DataType.FP16,
                    feature_names=["movie_id", "item_movie_id"],
                ),
                "user_id": EmbeddingConfig(
                    num_embeddings=3_000_000,
                    embedding_dim=DIM,
                    name="user_id",
                    data_type=DataType.FP16,
                    feature_names=["user_id"],
                ),
            }
        )
    elif "streaming" in dataset:
        return {
            "item_id": EmbeddingConfig(
                num_embeddings=HASH_SIZE_1B,
                embedding_dim=DIM,
                name="item_id",
                data_type=DataType.FP16,
                feature_names=["item_id", "item_candidate_id"],
            ),
            "item_category_id": EmbeddingConfig(
                num_embeddings=128,
                embedding_dim=DIM,
                name="item_category_id",
                data_type=DataType.FP16,
                weight_init_max=1.0,
                weight_init_min=-1.0,
                feature_names=["item_category_id", "item_candidate_category_id"],
            ),
            "user_id": EmbeddingConfig(
                num_embeddings=10_000_000,
                embedding_dim=DIM,
                name="user_id",
                data_type=DataType.FP16,
                feature_names=["user_id"],
            ),
        }
    elif "kuairand" in dataset:
        return {
            "video_id": EmbeddingConfig(
                num_embeddings=HASH_SIZE,
                embedding_dim=DIM,
                name="video_id",
                data_type=DataType.FP16,
                feature_names=["video_id", "item_video_id"],
            ),
            "user_id": EmbeddingConfig(
                num_embeddings=HASH_SIZE,
                embedding_dim=DIM,
                name="user_id",
                data_type=DataType.FP16,
                feature_names=["user_id"],
            ),
            "user_active_degree": EmbeddingConfig(
                num_embeddings=8,
                embedding_dim=DIM,
                name="user_active_degree",
                data_type=DataType.FP16,
                feature_names=["user_active_degree"],
            ),
            "follow_user_num_range": EmbeddingConfig(
                num_embeddings=9,
                embedding_dim=DIM,
                name="follow_user_num_range",
                data_type=DataType.FP16,
                feature_names=["follow_user_num_range"],
            ),
            "fans_user_num_range": EmbeddingConfig(
                num_embeddings=9,
                embedding_dim=DIM,
                name="fans_user_num_range",
                data_type=DataType.FP16,
                feature_names=["fans_user_num_range"],
            ),
            "friend_user_num_range": EmbeddingConfig(
                num_embeddings=8,
                embedding_dim=DIM,
                name="friend_user_num_range",
                data_type=DataType.FP16,
                feature_names=["friend_user_num_range"],
            ),
            "register_days_range": EmbeddingConfig(
                num_embeddings=8,
                embedding_dim=DIM,
                name="register_days_range",
                data_type=DataType.FP16,
                feature_names=["register_days_range"],
            ),
        }
    elif "yambda" in dataset:
        assert dataset in ["yambda-5b"]
        tables: Dict[str, EmbeddingConfig] = {
            "item_id": EmbeddingConfig(
                num_embeddings=9_390_624,
                embedding_dim=DIM,
                name="item_id",
                data_type=DataType.FP32,
                feature_names=["item_id", "item_candidate_id"],
            ),
            "artist_id": EmbeddingConfig(
                num_embeddings=1_293_395,
                embedding_dim=DIM,
                name="artist_id",
                data_type=DataType.FP32,
                feature_names=["artist_id", "item_candidate_artist_id"],
            ),
            "album_id": EmbeddingConfig(
                num_embeddings=3_367_692,
                embedding_dim=DIM,
                name="album_id",
                data_type=DataType.FP32,
                feature_names=["album_id", "item_candidate_album_id"],
            ),
            "uid": EmbeddingConfig(
                num_embeddings=1_000_001,
                embedding_dim=DIM,
                name="uid",
                data_type=DataType.FP32,
                feature_names=["uid"],
            ),
        }
        for name, _keys, num_embeddings, _salt in YAMBDA_5B_CROSS_SPECS:
            tables[name] = EmbeddingConfig(
                num_embeddings=num_embeddings,
                embedding_dim=DIM,
                name=name,
                data_type=DataType.FP32,
                feature_names=[name],
            )
        return tables
    else:
        return {
            "post_id": EmbeddingConfig(
                num_embeddings=HASH_SIZE,
                embedding_dim=DIM,
                name="post_id",
                data_type=DataType.FP16,
                feature_names=[
                    "uih_post_id",
                    "item_post_id",
                    "uih_owner_id",
                    "item_owner_id",
                ],
            ),
            "viewer_id": EmbeddingConfig(
                num_embeddings=HASH_SIZE,
                embedding_dim=DIM,
                name="viewer_id",
                data_type=DataType.FP16,
                feature_names=["viewer_id"],
            ),
            "dummy_contexual": EmbeddingConfig(
                num_embeddings=HASH_SIZE,
                embedding_dim=DIM,
                name="dummy_contexual",
                data_type=DataType.FP16,
                feature_names=["dummy_contexual"],
            ),
        }
