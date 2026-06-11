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
Checkpoint utilities for saving and loading DLRMv3 model checkpoints.

This module provides functions for saving and loading distributed model checkpoints,
including both sparse (embedding) and dense (non-embedding) components.
"""

import gc
import logging
import os
import random
import shutil
from datetime import datetime
from typing import Any, Dict, Optional, Set, Tuple

import gin
import numpy as np
import torch
from generative_recommenders.dlrm_v3.utils import (
    BinnedCumulativeAUC,
    LifetimeAUCMetricComputation,
    MetricsLogger,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim.optimizer import Optimizer
from torchrec.distributed.types import ShardedTensor

logger: logging.Logger = logging.getLogger(__name__)

# Sentinel meaning "the saved window completed in full" — when the loop reads
# this back it advances start_ts past the saved train_ts. Anything >=0 means the
# saved checkpoint stopped mid-window after K batches; resume continues that
# window at batch K.
WINDOW_COMPLETE: int = -1

# Filename (per-rank) holding the lifetime-AUC trailing buffers, mirroring the
# rng_rank{rank}.pt pattern. The buffers are per-rank-local, so a single
# rank-0 copy in non_sparse.ckpt would (wrongly) restore 1/world_size of the
# true history to every rank — hence a dedicated per-rank artifact.
METRICBUF_FILE_FMT: str = "metricbuf_rank{rank}.pt"


def _metric_blob_state_dict(m: torch.nn.Module) -> Dict[str, Any]:
    """State dict for the shared (rank-0) non_sparse.ckpt metric blob.

    Both lifetime-AUC backends carry per-rank-local state that is persisted
    authoritatively per-rank in ``metricbuf_rank{rank}.pt``; we must keep it out
    of the shared blob so a rank's load doesn't inherit rank-0's counts:

    - ``LifetimeAUCMetricComputation``: drop the explicitly-serialized trailing
      buffer keys (the rest of the blob keys are the parent's persistent state).
    - ``BinnedCumulativeAUC``: zero the histogram buffers (they are persistent so
      the keys must remain for a strict load, but the values are neutralized).

    All other metrics serialize normally. In both cases the per-rank file is
    loaded afterward and is authoritative.
    """
    sd = m.state_dict()
    if isinstance(m, LifetimeAUCMetricComputation):
        prefix = LifetimeAUCMetricComputation._LIFETIME_KEY_PREFIX
        sd = {k: v for k, v in sd.items() if not k.startswith(prefix)}
    elif isinstance(m, BinnedCumulativeAUC):
        sd = {
            k: (torch.zeros_like(v) if torch.is_tensor(v) else v)
            for k, v in sd.items()
        }
    return sd


def _collect_perrank_metric_state(
    metric_logger: "MetricsLogger",
) -> Dict[str, Dict[str, Any]]:
    """Map "<collection>|<mode>|<idx>" -> state_dict for every metric whose
    cumulative state is per-rank-local and must be restored per-rank:

    - lifetime-AUC instances (`LifetimeAUCMetricComputation` trailing buffer, or
      `BinnedCumulativeAUC` histograms) in class_metrics train/eval. Covers the
      train lifetime AUC and, in legacy single-set eval, the eval lifetime AUC,
      under either configured backend.
    - the ENTIRE cumulative eval set (`eval_cum`, both class + regression) used
      by the streaming dual-set eval: the lifetime-AUC backend state plus the
      persistent cumulative scalar sums of NE/Accuracy/GAUC/MSE/MAE.

    Selected by structure/isinstance (not a hard index) since metric positions
    depend on the configured tasks/mode.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for mode in ("train", "eval"):
        for idx, m in enumerate(metric_logger.class_metrics.get(mode, [])):
            if isinstance(m, (LifetimeAUCMetricComputation, BinnedCumulativeAUC)):
                out[f"class_metrics|{mode}|{idx}"] = m.state_dict()
    for coll in ("class_metrics", "regression_metrics"):
        for idx, m in enumerate(getattr(metric_logger, coll).get("eval_cum", [])):
            out[f"{coll}|eval_cum|{idx}"] = m.state_dict()
    return out


def _restore_perrank_metric_state(
    metric_logger: "MetricsLogger", state: Dict[str, Dict[str, Any]]
) -> None:
    for key, sd in state.items():
        coll, mode, idx_str = key.split("|")
        getattr(metric_logger, coll)[mode][int(idx_str)].load_state_dict(sd)


def _perrank_sample_counts(metric_logger: "MetricsLogger") -> Dict[str, int]:
    out: Dict[str, int] = {}

    def _count(m: torch.nn.Module) -> Optional[int]:
        if isinstance(m, LifetimeAUCMetricComputation):
            return m.lifetime_sample_count()
        if isinstance(m, BinnedCumulativeAUC):
            return m.cumulative_sample_count()
        return None

    for mode in ("train", "eval", "eval_cum"):
        for idx, m in enumerate(metric_logger.class_metrics.get(mode, [])):
            n = _count(m)
            if n is not None:
                out[f"class|{mode}|{idx}"] = n
    return out


class SparseState(Stateful):
    """
    Stateful wrapper for sparse (embedding) tensors in a model.

    This class implements the Stateful interface for distributed checkpointing,
    allowing sparse tensors to be saved and loaded separately from dense tensors.

    Args:
        model: The PyTorch model containing sparse tensors.
        sparse_tensor_keys: Set of keys identifying sparse tensors in the model's state dict.
    """

    def __init__(self, model: torch.nn.Module, sparse_tensor_keys: Set[str]) -> None:
        self.model = model
        self.sparse_tensor_keys = sparse_tensor_keys

    def state_dict(self) -> Dict[str, torch.Tensor]:
        out_dict: Dict[str, torch.Tensor] = {}
        is_sharded_tensor: Optional[bool] = None
        for k, v in self.model.state_dict().items():
            if k in self.sparse_tensor_keys:
                if is_sharded_tensor is None:
                    is_sharded_tensor = isinstance(v, ShardedTensor)
                assert is_sharded_tensor == isinstance(v, ShardedTensor)
                out_dict[k] = v
        return out_dict

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        incompatible_keys = self.model.load_state_dict(state_dict, strict=False)
        assert not incompatible_keys.unexpected_keys


def is_sparse_key(k: str, v: torch.Tensor) -> bool:
    return isinstance(v, ShardedTensor) or "embedding_collection" in k


def load_dense_state_dict(model: torch.nn.Module, state_dict: Dict[str, Any]) -> None:
    own_state = model.state_dict()
    own_state_dense_keys = {k for k, v in own_state.items() if not is_sparse_key(k, v)}
    state_dict_dense_keys = {
        k for k, v in state_dict.items() if not is_sparse_key(k, v)
    }
    assert own_state_dense_keys == state_dict_dense_keys, (
        f"expects {own_state_dense_keys} but gets {state_dict_dense_keys}"
    )
    for name in state_dict_dense_keys:
        param = state_dict[name]
        if isinstance(param, torch.nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        own_state[name].copy_(param)


def _rng_state(device: torch.device) -> Dict[str, Any]:
    """Snapshot every RNG source bit-equal training depends on.

    HSTU has stochastic dropout (input_dropout=0.2, linear_dropout_rate=0.1)
    consuming the per-device CUDA RNG cycle each step. Without round-tripping
    these, a resumed run draws different dropout masks and the resumed AUC
    trajectory diverges from the uninterrupted run within a few steps.
    """
    return {
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device),
        "numpy": np.random.get_state(),
        "random": random.getstate(),
    }


def _restore_rng_state(state: Dict[str, Any], device: torch.device) -> None:
    torch.set_rng_state(state["cpu"])
    torch.cuda.set_rng_state(state["cuda"], device)
    np.random.set_state(state["numpy"])
    random.setstate(state["random"])


def _list_numeric_subdirs(base_path: str) -> list[str]:
    """Return subdir names of `base_path` that look like an int, sorted ascending.

    Filters out `*.tmp` (orphaned in-progress saves), `*.sparse/` and any other
    non-numeric entries.
    """
    if not os.path.isdir(base_path):
        return []
    out: list[str] = []
    for name in os.listdir(base_path):
        if name.isdigit():
            out.append(name)
    return sorted(out, key=int)


def _resolve_latest_subdir(path: str) -> str:
    """Map a base ckpt dir → its highest-numbered numeric subdir.

    Used so users can set `load_dmp_checkpoint.path = "<base>"` (or
    `CKPT_PATH=<base>`) and automatically pick up the most recent save without
    needing to know which step number to point at.     If `path` already names a leaf save (numeric basename) it's returned
    unchanged. If the base dir has no numeric subdirs yet — the cold-start case
    where ``CKPT_PATH`` is configured but nothing has been saved (e.g. the
    interrupt phase of the resume test starts from a freshly-cleaned dir) — we
    return ``""`` so ``load_*_checkpoint`` no-ops instead of asserting on a
    missing ``sparse/.metadata``.
    """
    if not path:
        return path
    base = path.rstrip("/")
    leaf = os.path.basename(base)
    if leaf.isdigit():
        return base  # already a leaf, caller knows what it wants
    subs = _list_numeric_subdirs(base)
    if not subs:
        logger.info("No checkpoint subdirs under %s — cold start (no load).", base)
        return ""  # nothing to load → load_*_checkpoint short-circuits
    resolved = os.path.join(base, subs[-1])
    logger.info("Auto-latest checkpoint: %s → %s", base, resolved)
    return resolved


def _prune_old_checkpoints(base_path: str, keep_last_n: int, just_saved_subdir: str) -> None:
    """Delete numeric subdirs older than the keep_last_n most recent.

    Defensive: never prune `just_saved_subdir` even if it would be evicted by
    the keep_last_n window (shouldn't happen since we just wrote it, but
    catches off-by-one bugs). Skipped entirely when keep_last_n<=0.
    """
    if keep_last_n <= 0:
        return
    subs = _list_numeric_subdirs(base_path)
    if len(subs) <= keep_last_n:
        return
    to_prune = subs[:-keep_last_n]
    for name in to_prune:
        full = os.path.join(base_path, name)
        if os.path.realpath(full) == os.path.realpath(just_saved_subdir):
            continue
        try:
            shutil.rmtree(full)
            logger.info("Pruned old checkpoint: %s", full)
        except OSError as e:
            logger.warning("Failed to prune %s: %s", full, e)


def _cleanup_stale_tmps(base_path: str) -> None:
    """Remove `*.tmp`/`*.old` subdirs left by a crashed prior save attempt.

    `*.tmp` = an interrupted write; `*.old` = an interrupted atomic-overwrite
    swap (see the promotion step in save_dmp_checkpoint). Both are non-numeric
    so `_resolve_latest_subdir` already ignores them; this just reclaims disk.
    """
    if not os.path.isdir(base_path):
        return
    for name in os.listdir(base_path):
        if name.endswith(".tmp") or name.endswith(".old"):
            full = os.path.join(base_path, name)
            try:
                shutil.rmtree(full)
                logger.warning("Removed stale checkpoint dir: %s", full)
            except OSError as e:
                logger.warning("Failed to remove stale dir %s: %s", full, e)


@gin.configurable
def save_dmp_checkpoint(
    model: torch.nn.Module,
    optimizer: Optimizer,
    metric_logger: MetricsLogger,
    rank: int,
    batch_idx: int,
    path: str = "",
    keep_last_n: int = 1,
    train_ts: Optional[int] = None,
    batch_idx_in_window: int = WINDOW_COMPLETE,
    device: Optional[torch.device] = None,
    split_contract: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save a distributed model checkpoint including sparse and dense components.

    Writes into a per-rank-coordinated atomic layout:
        <path>/<batch_idx>.tmp/   ← directory written into during save
        <path>/<batch_idx>/       ← atomically renamed from .tmp on success

    A crash mid-save leaves the `.tmp/` orphan, which `_cleanup_stale_tmps`
    sweeps on the next save attempt and which `_resolve_latest_subdir` ignores
    (non-numeric basename). The previous successful `<N-K>/` remains valid.

    Args:
        model: The model to checkpoint.
        optimizer: The optimizer whose state should be saved.
        metric_logger: The metrics logger containing training/eval metrics.
        rank: The current process rank in distributed training.
        batch_idx: Subdir name (for streaming we set this == train_ts so the
            on-disk layout monotonically increases).
        path: Base path for saving the checkpoint. If empty, no checkpoint is saved.
        keep_last_n: Number of most-recent numeric subdirs to retain after a
            successful save. Set 1 (default) for disk-bounded long runs;
            <=0 disables pruning.
        train_ts: For streaming-train-eval, the current train timestamp.
            Stored in non_sparse.ckpt so resume knows which window to enter.
        batch_idx_in_window: For streaming-train-eval, batches completed within
            train_ts. WINDOW_COMPLETE (-1) means the window finished; resume
            advances to train_ts+1. >=0 means crash happened mid-window; resume
            re-enters train_ts at batch_idx_in_window.
        device: CUDA device for the per-rank RNG snapshot. Required for
            bit-equal trajectories across resume (HSTU dropout consumes the
            per-device RNG cycle).
    """
    if path == "":
        return
    base_path = path
    # Atomic-save layout: write to .tmp, rename to final, prune older.
    tmp_subdir = f"{base_path}/{batch_idx}.tmp"
    final_subdir = f"{base_path}/{batch_idx}"

    if rank == 0:
        _cleanup_stale_tmps(base_path)
        # Always (re)write into a fresh .tmp. An existing `final_subdir` with the
        # same batch_idx (e.g. a later in-window save for the same train_ts, or a
        # deterministic re-run at the same step) is overwritten atomically at the
        # promotion step below — NOT skipped here. Skipping would desync ranks:
        # the collective barrier/checkpoint.save calls below run on *every* rank,
        # so a rank-0-only early return deadlocks ranks 1..N on the next barrier.
        shutil.rmtree(tmp_subdir, ignore_errors=True)
        os.makedirs(tmp_subdir, exist_ok=True)
        os.makedirs(f"{tmp_subdir}/sparse/", exist_ok=True)
    torch.distributed.barrier()
    sparse_path = f"{tmp_subdir}/sparse/"
    non_sparse_ckpt = f"{tmp_subdir}/non_sparse.ckpt"

    sparse_tensor_keys = {
        k for k, v in model.state_dict().items() if isinstance(v, ShardedTensor)
    }
    if rank == 0:
        dense_state_dict = {
            k: v
            for k, v in model.state_dict().items()
            if not isinstance(v, ShardedTensor)
        }
        class_metric_state_dict = {
            "train": [
                _metric_blob_state_dict(m)
                for m in metric_logger.class_metrics["train"]
            ],
            "eval": [
                _metric_blob_state_dict(m)
                for m in metric_logger.class_metrics["eval"]
            ],
        }
        regression_metric_state_dict = {
            "train": [
                m.state_dict() for m in metric_logger.regression_metrics["train"]
            ],
            "eval": [m.state_dict() for m in metric_logger.regression_metrics["eval"]],
        }
        torch.save(
            {
                "dense_dict": dense_state_dict,
                "optimizer_dict": optimizer.state_dict(),
                "class_metrics": class_metric_state_dict,
                "reg_metrics": regression_metric_state_dict,
                "global_step": metric_logger.global_step,
                # MLPerf progress counter (global trained samples). Defaulted on
                # load so pre-existing checkpoints restore as 0 and resume the
                # count from there.
                "cumulative_train_samples": metric_logger.cumulative_train_samples,
                "sparse_tensor_keys": sparse_tensor_keys,
                # Streaming resume fields. Defaulted on load so old checkpoints
                # (pre-streaming-resume) still load as a normal restart.
                "train_ts": train_ts,
                "batch_idx_in_window": batch_idx_in_window,
                # Immutable train:eval split + resume-determinism contract
                # (train_split_percentage, split_salt, eval holdout window,
                # batch_size, world_size). Validated on resume so a relaunch
                # cannot silently change the split (which would desync the skip
                # offset and/or train on held-out eval users). None for
                # non-holdout / legacy runs.
                "split_contract": split_contract,
            },
            non_sparse_ckpt,
        )

    # Per-rank RNG snapshot. Written even on a single rank because dropout's
    # randomness comes from the CUDA generator which differs across devices.
    if device is not None:
        rng_path = f"{tmp_subdir}/rng_rank{rank}.pt"
        torch.save(_rng_state(device), rng_path)

    # Per-rank cumulative metric state (lifetime-AUC buffers + cumulative-eval
    # histograms/scalar sums). Written by EVERY rank (outside the rank-0 block)
    # because this state is per-rank-local; restoring rank-0's copy to all ranks
    # would lose (world_size-1)/world_size of the history.
    if metric_logger is not None:
        perrank_state = _collect_perrank_metric_state(metric_logger)
        if perrank_state:
            torch.save(
                perrank_state,
                f"{tmp_subdir}/{METRICBUF_FILE_FMT.format(rank=rank)}",
            )
            logger.info(
                "checkpoint save: cumulative metric state rank=%d samples=%s",
                rank,
                _perrank_sample_counts(metric_logger),
            )

    torch.distributed.barrier()
    sparse_dict = {"sparse_dict": SparseState(model, sparse_tensor_keys)}
    torch.distributed.checkpoint.save(
        sparse_dict,
        storage_writer=torch.distributed.checkpoint.FileSystemWriter(sparse_path),
    )
    torch.distributed.barrier()
    # Promote .tmp → final, then prune. Done on rank 0 only since the directory
    # operations are global filesystem state.
    if rank == 0:
        if os.path.exists(final_subdir):
            # POSIX rename() refuses to replace a non-empty directory, so we
            # can't os.replace(tmp, final) directly. Swap the old snapshot aside
            # (instant rename), move the new one into place, then delete the old.
            # The `.old` name is non-numeric → ignored by _resolve_latest_subdir
            # and swept by _cleanup_stale_tmps on the next save if we crash mid-swap.
            old_aside = f"{final_subdir}.old"
            shutil.rmtree(old_aside, ignore_errors=True)
            os.replace(final_subdir, old_aside)
            os.replace(tmp_subdir, final_subdir)
            shutil.rmtree(old_aside, ignore_errors=True)
        else:
            os.replace(tmp_subdir, final_subdir)
        _prune_old_checkpoints(base_path, keep_last_n, final_subdir)
        logger.info("checkpoint successfully saved → %s", final_subdir)
    torch.distributed.barrier()


@gin.configurable
def load_sparse_checkpoint(
    model: torch.nn.Module,
    path: str = "",
) -> None:
    if path == "":
        return
    sparse_path = f"{path}/sparse/"

    sparse_tensor_keys = {
        k for k, v in model.state_dict().items() if is_sparse_key(k, v)
    }
    sparse_dict = {"sparse_dict": SparseState(model, sparse_tensor_keys)}
    gc.collect()
    torch.distributed.checkpoint.load(
        sparse_dict,
        storage_reader=torch.distributed.checkpoint.FileSystemReader(sparse_path),
    )
    gc.collect()
    print("sparse checkpoint successfully loaded")


@gin.configurable
def load_nonsparse_checkpoint(
    model: torch.nn.Module,
    device: torch.device,
    optimizer: Optional[Optimizer] = None,
    metric_logger: Optional[MetricsLogger] = None,
    path: str = "",
    rank: int = 0,
) -> Tuple[Optional[int], int, Optional[Dict[str, Any]]]:
    """
    Load non-sparse (dense) components from a checkpoint.

    Loads dense model parameters, and optionally optimizer state and metrics.
    Also restores per-rank RNG state if a matching `rng_rank{rank}.pt` is found
    next to `non_sparse.ckpt`.

    Returns:
        (train_ts, batch_idx_in_window, split_contract) — the streaming resume
        hint and the saved train:eval split contract (None for legacy / non-
        holdout checkpoints). `(None, WINDOW_COMPLETE, None)` if not a streaming
        checkpoint or no path supplied.
    """
    if path == "":
        return None, WINDOW_COMPLETE, None
    non_sparse_ckpt = f"{path}/non_sparse.ckpt"

    # weights_only=False: these are our own trusted checkpoints, and they hold
    # non-tensor objects (optimizer/metric state dicts, numpy-backed RNG state)
    # that PyTorch>=2.6's weights_only=True default refuses to unpickle.
    non_sparse_state_dict = torch.load(
        non_sparse_ckpt, map_location=device, weights_only=False
    )
    load_dense_state_dict(model, non_sparse_state_dict["dense_dict"])
    print("dense checkpoint successfully loaded")
    if optimizer is not None:
        optimizer.load_state_dict(non_sparse_state_dict["optimizer_dict"])
        print("optimizer checkpoint successfully loaded")
    if metric_logger is not None:
        metric_logger.global_step = non_sparse_state_dict["global_step"]
        # Defaulted for legacy checkpoints written before the counter existed.
        metric_logger.cumulative_train_samples = non_sparse_state_dict.get(
            "cumulative_train_samples", 0
        )
        class_metric_state_dict = non_sparse_state_dict["class_metrics"]
        regression_metric_state_dict = non_sparse_state_dict["reg_metrics"]
        # Length-safe positional restore: if a checkpoint was written with a
        # different metric set (e.g. tasks added/removed since), restore the
        # overlap instead of crashing with an IndexError at run end.
        def _restore_metric_list(
            live: list, saved: Optional[list], label: str
        ) -> None:
            saved = saved or []
            if len(live) != len(saved):
                logger.warning(
                    "metric count mismatch for %s: live=%d saved=%d; "
                    "restoring overlapping %d",
                    label,
                    len(live),
                    len(saved),
                    min(len(live), len(saved)),
                )
            for i in range(min(len(live), len(saved))):
                live[i].load_state_dict(saved[i])

        _restore_metric_list(
            metric_logger.class_metrics["train"],
            class_metric_state_dict.get("train"),
            "class/train",
        )
        _restore_metric_list(
            metric_logger.class_metrics["eval"],
            class_metric_state_dict.get("eval"),
            "class/eval",
        )
        _restore_metric_list(
            metric_logger.regression_metrics["train"],
            regression_metric_state_dict.get("train"),
            "reg/train",
        )
        _restore_metric_list(
            metric_logger.regression_metrics["eval"],
            regression_metric_state_dict.get("eval"),
            "reg/eval",
        )

        # Per-rank cumulative metric state restore. This runs AFTER the generic
        # load above so it is authoritative: the shared blob carries no lifetime
        # buffers (stripped at save) nor any eval_cum state, and each rank
        # restores its OWN cumulative state here. Missing file = legacy/pre-fix
        # checkpoint; cumulative metrics self-heal (lifetime AUC refills; the
        # binned-AUC histograms / scalar sums restart from zero).
        mb_path = f"{path}/{METRICBUF_FILE_FMT.format(rank=rank)}"
        if os.path.exists(mb_path):
            perrank_state = torch.load(
                mb_path, map_location=device, weights_only=False
            )
            _restore_perrank_metric_state(metric_logger, perrank_state)
            logger.info(
                "checkpoint load: cumulative metric state rank=%d samples=%s",
                rank,
                _perrank_sample_counts(metric_logger),
            )
        else:
            logger.info(
                "checkpoint load: no per-rank cumulative metric state at %s "
                "(legacy/pre-fix checkpoint); cumulative metrics will refill",
                mb_path,
            )

    # Per-rank RNG restore. Missing file = bit-equal trajectory not requested at
    # save time; we silently continue (the test harness checks for both).
    rng_path = f"{path}/rng_rank{rank}.pt"
    if os.path.exists(rng_path):
        # weights_only=False: RNG state is numpy/Python tuples, not tensors.
        rng_state = torch.load(rng_path, map_location="cpu", weights_only=False)
        _restore_rng_state(rng_state, device)
        logger.info("RNG state restored from %s", rng_path)

    train_ts = non_sparse_state_dict.get("train_ts")
    batch_idx_in_window = non_sparse_state_dict.get(
        "batch_idx_in_window", WINDOW_COMPLETE
    )
    split_contract = non_sparse_state_dict.get("split_contract")
    return train_ts, batch_idx_in_window, split_contract


@gin.configurable
def load_dmp_checkpoint(
    model: torch.nn.Module,
    optimizer: Optimizer,
    metric_logger: MetricsLogger,
    device: torch.device,
    path: str = "",
    rank: int = 0,
) -> Tuple[Optional[int], int, Optional[Dict[str, Any]], bool]:
    """
    Load a complete distributed model checkpoint (both sparse and dense components).

    `path` is auto-resolved: if it points at a directory containing numeric
    subdirs (e.g. CKPT_PATH=<base>/), the highest-numbered subdir is used. If it
    already names a leaf save (e.g. <base>/300), it's used as-is. Empty string =
    no load.

    Returns:
        (train_ts, batch_idx_in_window, split_contract, cold_start) — streaming
        resume hint plus the saved split contract, and `cold_start` which is True
        iff there was nothing to load (no checkpoint resolved). `cold_start`
        distinguishes a genuine fresh run (no weights loaded) from a resume that
        merely lacks a split contract (e.g. a legacy/non-streaming checkpoint),
        which the caller's split-contract guard must still reject.
    """
    resolved = _resolve_latest_subdir(path)
    cold_start = resolved == ""
    load_sparse_checkpoint(model=model, path=resolved)
    train_ts, batch_idx_in_window, split_contract = load_nonsparse_checkpoint(
        model=model,
        optimizer=optimizer,
        metric_logger=metric_logger,
        path=resolved,
        device=device,
        rank=rank,
    )
    return train_ts, batch_idx_in_window, split_contract, cold_start
