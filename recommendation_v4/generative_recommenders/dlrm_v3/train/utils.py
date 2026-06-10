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
import logging
import os
import threading
import time
from collections.abc import Iterator
from datetime import timedelta
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

import gin
import torch
import torchrec
from generative_recommenders.dlrm_v3.checkpoint import save_dmp_checkpoint, WINDOW_COMPLETE
from generative_recommenders.dlrm_v3.configs import (
    get_embedding_table_config,
    get_hstu_configs,
)
from generative_recommenders.dlrm_v3.datasets.dataset import (
    collate_fn,
    Dataset,
    Samples,
)
from generative_recommenders.dlrm_v3.utils import get_dataset, MetricsLogger, Profiler
from generative_recommenders.common import HammerKernel
from generative_recommenders.modules.dlrm_hstu import DlrmHSTU, DlrmHSTUConfig
from torch import distributed as dist
from torch.distributed.optim import (
    _apply_optimizer_in_backward as apply_optimizer_in_backward,
)
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torch.utils.data.distributed import _T_co, DistributedSampler
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.sharding_plan import get_default_sharders
from torchrec.distributed.types import ShardedTensor, ShardingEnv
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingCollection,
)
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

logger: logging.Logger = logging.getLogger(__name__)

TORCHREC_TYPES: Set[Type[Union[EmbeddingBagCollection, EmbeddingCollection]]] = {
    EmbeddingBagCollection,
    EmbeddingCollection,
}


def setup(
    rank: int,
    world_size: int,
    master_port: int,
    device: torch.device,
    master_addr: str = "localhost",
) -> dist.ProcessGroup:
    # Default "localhost" keeps the single-node path unchanged; multi-node
    # launches pass the rank-0 host so every node rendezvouses at the same addr.
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)

    BACKEND = dist.Backend.NCCL
    TIMEOUT = 1800

    # set device BEFORE init_process_group so NCCL binds this rank to its
    # own GPU; otherwise every rank's first CUDA context lands on GPU 0,
    # leaving stale allocations and triggering OOMs on rank 0.
    torch.cuda.set_device(device)

    # Seed all RNGs so weight init (make_model, called after setup) is
    # reproducible across runs. Same seed on every rank → dense params are
    # initialized identically across ranks; sharded embeddings are init'd from
    # the meta device by DMP. Fixed seed makes pipeline-vs-non-pipeline an
    # init-matched A/B (data order is already deterministic via the sampler).
    import random

    import numpy as np

    _SEED = 1
    random.seed(_SEED)
    np.random.seed(_SEED)
    torch.manual_seed(_SEED)
    torch.cuda.manual_seed_all(_SEED)

    # initialize the process group
    if not dist.is_initialized():
        dist.init_process_group(
            "nccl", rank=rank, world_size=world_size, device_id=device
        )

    pg = dist.new_group(
        backend=BACKEND,
        timeout=timedelta(seconds=TIMEOUT),
    )

    return pg


def cleanup() -> None:
    dist.destroy_process_group()


class HammerToTorchDataset(TorchDataset):
    def __init__(
        self,
        dataset: Dataset,
    ) -> None:
        self.dataset: Dataset = dataset

    def __getitem__(self, idx: int) -> Tuple[KeyedJaggedTensor, KeyedJaggedTensor]:
        self.dataset.load_query_samples([idx])
        sample = self.dataset.get_sample(idx)
        self.dataset.unload_query_samples([idx])
        return sample

    def __getitems__(
        self, indices: List[int]
    ) -> List[Tuple[KeyedJaggedTensor, KeyedJaggedTensor]]:
        self.dataset.load_query_samples(indices)
        samples = [self.dataset.get_sample(i) for i in indices]
        self.dataset.unload_query_samples(indices)
        return samples


class _ChainedRanges:
    """O(1) __len__ + O(log K) __getitem__ over a sequence of `range`s.

    Lets `torch.utils.data.Subset(dataset, _ChainedRanges([r1, r2, ...]))`
    avoid materializing a Python list of all per-block indices (which at
    multi-billion totals is ~28 B/int and dominates host RAM).
    """

    def __init__(self, ranges: List[range]) -> None:
        self._ranges: List[range] = list(ranges)
        offsets = [0]
        for r in self._ranges:
            offsets.append(offsets[-1] + len(r))
        self._offsets: List[int] = offsets

    def __len__(self) -> int:
        return self._offsets[-1]

    def __getitem__(self, idx: int) -> int:
        import bisect
        if idx < 0:
            idx += self._offsets[-1]
        if idx < 0 or idx >= self._offsets[-1]:
            raise IndexError(idx)
        bucket = bisect.bisect_right(self._offsets, idx) - 1
        return self._ranges[bucket][idx - self._offsets[bucket]]


class ChunkDistributedSampler(DistributedSampler[_T_co]):
    """
    Each rank reads a contiguous chunk (trunk) of the input data
    """

    def __init__(
        self,
        dataset: TorchDataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 1,
        drop_last: bool = False,
    ) -> None:
        super().__init__(
            dataset=dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )

    def __iter__(self) -> Iterator[_T_co]:
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch * 1001 + int(self.rank))
            indices_t = torch.randperm(self.num_samples, generator=g)
        else:
            indices_t = torch.arange(self.num_samples)
        assert self.drop_last is True, (
            "drop_last must be True for ChunkDistributedSampler"
        )
        indices_t = indices_t + (self.num_samples * int(self.rank))
        assert indices_t.numel() == self.num_samples
        # Iterate via the numpy view, NOT directly over the tensor: iter(Tensor)
        # calls Tensor.unbind(0) which eagerly materializes one zero-dim Tensor
        # object per element (~600 B each). For 40 M eval / 525 M train samples
        # that's 24 GB / 315 GB of [heap] growth per rank, blowing host RAM
        # before the first batch. numpy's iter yields one Python int at a time
        # with O(1) extra memory.
        indices_np = indices_t.numpy()
        return (int(x) for x in indices_np)

    def set_epoch(self, epoch: int) -> None:
        logger.warning(f"Setting epoch to {epoch}")
        self.epoch = epoch


@gin.configurable
def make_model(
    dataset: str,
    bf16_training: bool = False,
    hammer_kernel: Optional[str] = None,
) -> Tuple[torch.nn.Module, DlrmHSTUConfig, Dict[str, EmbeddingConfig]]:
    hstu_config = get_hstu_configs(dataset)
    table_config = get_embedding_table_config(dataset)

    # bf16 autocast is off by default: on the PYTORCH attn backend the
    # pt_hstu_attention QK einsum backward overflows in bf16 at long
    # sequences (NaN at step 1 when N>1k). Safe with TRITON; flip via
    # `make_model.bf16_training = True` in the gin.
    model = DlrmHSTU(
        hstu_configs=hstu_config,
        embedding_tables=table_config,
        is_inference=False,
        bf16_training=bf16_training,
    )

    # HSTU attention/compute kernel backend. Precedence:
    #   HSTU_HAMMER_KERNEL env var  >  make_model.hammer_kernel gin  >  model default.
    # The env var stays as an ad-hoc override (e.g. forcing PYTORCH for a one-off
    # debug run) without editing the gin. Note: the fused TRITON path avoids
    # materializing the dense [B, H, N, N] attention-score tensor that the PYTORCH
    # path allocates (~32 GiB at N=2048, bs=1024), so TRITON is both faster and
    # far lighter on HBM. On older ROCm, TRITON could hit PassManager errors at
    # some shapes (make_ttgir) — fall back to PYTORCH via the gin/env if so.
    kernel_choice = (
        os.environ.get("HSTU_HAMMER_KERNEL", "").upper()
        or (hammer_kernel.upper() if hammer_kernel else "")
    )
    if kernel_choice:
        model.set_hammer_kernel(HammerKernel[kernel_choice])
        logger.warning(f"HSTU hammer kernel set to: {kernel_choice}")

    return (
        model,
        hstu_config,
        table_config,
    )


@gin.configurable()
def dense_optimizer_factory_and_class(
    optimizer_name: str,
    betas: Tuple[float, float],
    eps: float,
    weight_decay: float,
    momentum: float,
    learning_rate: float,
) -> Tuple[
    Type[Optimizer], Dict[str, Any], Callable[[Iterable[torch.Tensor]], Optimizer]
]:
    kwargs: Dict[str, Any] = {"lr": learning_rate}
    if optimizer_name == "Adam":
        optimizer_cls = torch.optim.Adam
        kwargs.update({"betas": betas, "eps": eps, "weight_decay": weight_decay})
    elif optimizer_name == "SGD":
        optimizer_cls = torch.optim.SGD
        kwargs.update({"weight_decay": weight_decay, "momentum": momentum})
    elif optimizer_name == "AdamW":
        optimizer_cls = torch.optim.AdamW
        kwargs.update({"betas": betas, "eps": eps, "weight_decay": weight_decay})
    else:
        raise Exception("Unsupported optimizer!")

    optimizer_factory = lambda params: optimizer_cls(params, **kwargs)

    return optimizer_cls, kwargs, optimizer_factory


@gin.configurable()
def sparse_optimizer_factory_and_class(
    optimizer_name: str,
    betas: Tuple[float, float],
    eps: float,
    weight_decay: float,
    momentum: float,
    learning_rate: float,
) -> Tuple[
    Type[Optimizer], Dict[str, Any], Callable[[Iterable[torch.Tensor]], Optimizer]
]:
    kwargs: Dict[str, Any] = {"lr": learning_rate}
    if optimizer_name == "Adam":
        optimizer_cls = torch.optim.Adam
        beta1, beta2 = betas
        kwargs.update(
            {"beta1": beta1, "beta2": beta2, "eps": eps, "weight_decay": weight_decay}
        )
    elif optimizer_name == "SGD":
        optimizer_cls = torchrec.optim.SGD
        kwargs.update({"weight_decay": weight_decay, "momentum": momentum})
    elif optimizer_name == "RowWiseAdagrad":
        optimizer_cls = torchrec.optim.RowWiseAdagrad
        beta1, beta2 = betas
        kwargs.update(
            {
                "eps": eps,
                "beta1": beta1,
                "beta2": beta2,
                "weight_decay": weight_decay,
            }
        )
    else:
        raise Exception("Unsupported optimizer!")

    optimizer_factory = lambda params: optimizer_cls(params, **kwargs)

    return optimizer_cls, kwargs, optimizer_factory


@gin.configurable
def make_optimizer_and_shard(
    model: torch.nn.Module,
    device: torch.device,
    world_size: int,
    local_world_size: Optional[int] = None,
    hbm_cap_gb: int = 260,
) -> Tuple[DistributedModelParallel, torch.optim.Optimizer]:
    dense_opt_cls, dense_opt_args, dense_opt_factory = (
        dense_optimizer_factory_and_class()
    )

    sparse_opt_cls, sparse_opt_args, sparse_opt_factory = (
        sparse_optimizer_factory_and_class()
    )
    # Fuse sparse optimizer to backward step
    for k, module in model.named_modules():
        if type(module) in TORCHREC_TYPES:
            for _, param in module.named_parameters(prefix=k):
                if param.requires_grad:
                    apply_optimizer_in_backward(
                        sparse_opt_cls, [param], sparse_opt_args
                    )
    sharders = get_default_sharders()
    # local_world_size = GPUs per node so the planner respects the intra-node
    # (xGMI/NVLink) vs inter-node hierarchy when placing shards. Defaults to
    # world_size for the single-node case (no behavior change).
    planner = EmbeddingShardingPlanner(
        topology=Topology(
            local_world_size=local_world_size or world_size,
            world_size=world_size,
            compute_device="cuda",
            hbm_cap=hbm_cap_gb * 1024 * 1024 * 1024,
            ddr_cap=0,
        )
    )
    pg = dist.GroupMember.WORLD
    env = ShardingEnv.from_process_group(pg)  # pyre-ignore [6]
    pg = env.process_group

    plan = planner.collective_plan(model, sharders, pg)

    # Shard model
    model = DistributedModelParallel(
        module=model,
        device=device,
        plan=plan,
        sharders=sharders,
    )
    # Create keyed optimizer
    all_optimizers = []
    all_params = {}
    non_fused_sparse_params = {}
    for k, v in in_backward_optimizer_filter(model.named_parameters()):
        if v.requires_grad:
            if isinstance(v, ShardedTensor):
                non_fused_sparse_params[k] = v
            else:
                all_params[k] = v

    if non_fused_sparse_params:
        all_optimizers.append(
            (
                "sparse_non_fused",
                KeyedOptimizerWrapper(
                    params=non_fused_sparse_params, optim_factory=sparse_opt_factory
                ),
            )
        )

    if all_params:
        all_optimizers.append(
            (
                "dense",
                KeyedOptimizerWrapper(
                    params=all_params,
                    optim_factory=dense_opt_factory,
                ),
            )
        )
    output_optimizer = CombinedOptimizer(all_optimizers)
    output_optimizer.init_state(set(model.sparse_grad_parameter_names()))
    return model, output_optimizer


@gin.configurable
def make_streaming_dataloader(
    dataset: HammerToTorchDataset,
    ts: Optional[int] = None,
    batch_size: int = 0,
    num_workers: int = 0,
    prefetch_factor: int = 0,
    train_only: bool = False,
    indices: Optional["np.ndarray"] = None,
) -> DataLoader:
    # `indices` (explicit anchor index array) is used by the eval path to
    # iterate the FIXED user-holdout set, which spans a window range rather than
    # a single ts. Otherwise restrict to window `ts`; train_only=True drops
    # held-out eval users so the non-persistent TRAIN loader never trains on
    # them (no-leakage guarantee).
    if indices is not None:
        dataset.dataset.set_active_indices(indices)  # pyre-ignore [16]
    else:
        assert ts is not None, "make_streaming_dataloader needs ts or indices"
        dataset.dataset.set_ts(ts, train_only=train_only)  # pyre-ignore [16]
    total_items = dataset.dataset.get_item_count()
    subset = torch.utils.data.Subset(dataset, range(total_items))
    # shuffle=False keeps temporal order within the window: a non-shuffling
    # DistributedSampler hands rank r the strided slice indices[r::num_replicas]
    # (round-robin), so all ranks stay on the same time front and consume the
    # window in index order. Fork ctx mirrors the train path (COW-share the
    # mmap'd store instead of pickling it into every worker).
    mp_ctx = "fork" if num_workers and num_workers > 0 else None
    dataloader = DataLoader(
        dataset=subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        sampler=DistributedSampler(subset, shuffle=False, drop_last=True),
        multiprocessing_context=mp_ctx,
    )
    return dataloader


class StreamingWindowSampler(torch.utils.data.Sampler):
    """Per-rank sampler whose index list is swapped each window.

    Yields this rank's round-robin slice of the active window's GLOBAL anchor
    indices (into the dataset's ``_positions``). Because indices are global, a
    single DataLoader with ``persistent_workers=True`` can be reused across all
    windows: the main process re-iterates this sampler each window and ships the
    new indices to the already-forked workers, which map any global index via
    the shared mmap. No per-window worker respawn / dataset re-pickle.

    Round-robin striding (rank r gets ``indices[r::world_size]``) over the
    time-sorted window keeps every rank on the same time front; the window is
    truncated to a multiple of ``world_size`` so all ranks get equal counts
    (required for DDP collective lockstep).
    """

    def __init__(self, rank: int, world_size: int) -> None:
        self._rank: int = rank
        self._world_size: int = world_size
        self._indices: List[int] = []

    def set_window(self, global_indices, skip_samples: int = 0) -> None:
        """Install this window's per-rank index list, optionally fast-forwarding.

        ``skip_samples`` drops the first N per-rank samples from the list so the
        next ``__iter__`` starts at sample N+1 in this rank's slice. Used on
        resume to skip batches that were already trained: pass
        ``skip_samples = batch_size * batches_completed`` and the dataloader
        emits batches starting at exactly the next unseen batch.

        The skip is safe because the sample order is fully deterministic given
        (global_indices, rank, world_size): we re-derive the same per-rank list
        as the pre-crash run, just hand back a tail slice of it.
        """
        n = (len(global_indices) // self._world_size) * self._world_size
        per_rank = global_indices[:n][self._rank :: self._world_size].tolist()
        if skip_samples < 0 or skip_samples > len(per_rank):
            raise ValueError(
                f"skip_samples={skip_samples} out of [0, {len(per_rank)}] "
                f"for rank={self._rank} world_size={self._world_size}"
            )
        self._indices = per_rank[skip_samples:]

    def __iter__(self):
        return iter(self._indices)

    def __len__(self) -> int:
        return len(self._indices)


@gin.configurable
def make_persistent_streaming_dataloader(
    dataset: HammerToTorchDataset,
    sampler: StreamingWindowSampler,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
) -> DataLoader:
    """One reusable DataLoader for the whole streaming run. ``sampler`` is
    mutated per window via ``set_window``; workers persist across windows."""
    use_workers = bool(num_workers and num_workers > 0)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if use_workers else None,
        sampler=sampler,
        persistent_workers=use_workers,
        multiprocessing_context="fork" if use_workers else None,
    )


class _PrefetchingWindowLoader:
    """Double-buffered window loader for the persistent streaming path.

    Holds ``n_buffers`` pre-forked persistent worker pools that ping-pong: while
    the current window trains on one pool, the *next* window's index selection
    (``window_indices``) and first-batch prefetch are prepared on another pool
    in a background thread. By the time training advances, that window is warm,
    so the per-window reset (mask + first-batch stall) is hidden behind GPU
    compute (~0 dead time at the boundary).

    Worker pools are forked once on the main thread at the start of ``stream``;
    afterwards only iterator resets happen (no forks), so background-thread
    preparation cannot fork while other threads hold locks.
    """

    def __init__(
        self,
        dataset: "HammerToTorchDataset",
        sampler_factory,
        dl_factory,
        n_buffers: int = 2,
    ) -> None:
        self._dataset = dataset
        self._n = n_buffers
        self._samplers = [sampler_factory() for _ in range(n_buffers)]
        self._dls = [dl_factory(s) for s in self._samplers]
        self._iters: List[Optional[object]] = [None] * n_buffers

    def _prepare(self, buf: int, ts: int, skip_samples: int = 0) -> None:
        # train_window_indices() is the O(N) mask (+ uid-hash filter for the
        # holdout); numpy releases the GIL for it, so it overlaps the main
        # thread's GPU dispatch. iter() then kicks off this pool's background
        # prefetch. This is a TRAIN-only loader, so held-out eval users are
        # excluded here. `skip_samples` is non-zero only for the very first
        # window after a mid-window resume; subsequent windows always start at 0.
        self._samplers[buf].set_window(
            self._dataset.dataset.train_window_indices(ts), skip_samples=skip_samples
        )
        self._iters[buf] = iter(self._dls[buf])

    def stream(self, ts_list: List[int], first_skip_samples: int = 0):
        """Stream (ts, iterator) pairs. `first_skip_samples` is applied ONLY to
        the first ts in ``ts_list`` (the mid-window-resumed window); every
        subsequent window starts at sample 0 of its own per-rank list."""
        n = len(ts_list)
        if n == 0:
            return
        threads: List[Optional[threading.Thread]] = [None] * self._n
        # Prime the first n_buffers windows on the main thread (forks all pools).
        for b in range(min(self._n, n)):
            skip = first_skip_samples if b == 0 else 0
            self._prepare(b, ts_list[b], skip_samples=skip)
        for i in range(n):
            buf = i % self._n
            if threads[buf] is not None:
                threads[buf].join()
                threads[buf] = None
            yield ts_list[i], self._iters[buf]
            # This pool is now free; prefetch the window n_buffers ahead.
            # No skip on subsequent windows — only the first prepared window
            # carries `first_skip_samples`.
            j = i + self._n
            if j < n:
                th = threading.Thread(
                    target=self._prepare, args=(buf, ts_list[j]), daemon=True
                )
                th.start()
                threads[buf] = th


@gin.configurable
def make_train_test_dataloaders(
    batch_size: int,
    dataset_type: str,
    hstu_config: DlrmHSTUConfig,
    train_split_percentage: float,
    embedding_table_configs: Dict[str, EmbeddingConfig],
    new_path_prefix: str = "",
    num_workers: int = 0,
    num_blocks: int = 1,
    prefetch_factor: Optional[int] = None,
    eval_batch_size: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    dataset_class, kwargs = get_dataset(
        name=dataset_type, new_path_prefix=new_path_prefix
    )
    kwargs["embedding_config"] = embedding_table_configs

    # Create dataset
    dataset = HammerToTorchDataset(
        dataset=dataset_class(hstu_config=hstu_config, is_inference=False, **kwargs)
    )
    total_items = dataset.dataset.get_item_count()
    items_per_block = total_items // num_blocks
    train_size_per_block = round(train_split_percentage * items_per_block)
    # Avoid `extend(range(...))` which materializes a Python list of all sample
    # indices — at 3.2B yambda samples × 28 bytes/int ≈ 90 GB/rank just for
    # train_inds. Subset accepts any sequence with O(1) __len__ and __getitem__,
    # so pass range objects (or a tiny chained view) directly.
    if num_blocks == 1:
        train_inds = range(0, train_size_per_block)
        test_inds = range(train_size_per_block, items_per_block)
    else:
        train_inds = _ChainedRanges([
            range(i * items_per_block, i * items_per_block + train_size_per_block)
            for i in range(num_blocks)
        ])
        test_inds = _ChainedRanges([
            range(i * items_per_block + train_size_per_block, (i + 1) * items_per_block)
            for i in range(num_blocks)
        ])
    train_set = torch.utils.data.Subset(dataset, train_inds)
    test_set = torch.utils.data.Subset(dataset, test_inds)

    # When the parent rank is started via mp.start_processes(start_method="spawn"),
    # torch.multiprocessing's default Process context is also "spawn". DataLoader
    # then pickles `self._dataset` to send to each worker — which for our mmap'd
    # 211 GB yambda store materializes the entire dataset into the parent's anon
    # memory (~230 GB/rank). Forcing "fork" lets workers inherit the parent's
    # mmap'd pages via COW with zero extra anon.
    mp_ctx = "fork" if num_workers and num_workers > 0 else None
    train_dataloader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        sampler=ChunkDistributedSampler(train_set, drop_last=True, shuffle=True),
        multiprocessing_context=mp_ctx,
    )
    test_dataloader = DataLoader(
        dataset=test_set,
        batch_size=eval_batch_size if eval_batch_size is not None else batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        sampler=ChunkDistributedSampler(test_set, drop_last=True, shuffle=True),
        multiprocessing_context=mp_ctx,
    )
    return train_dataloader, test_dataloader


@gin.configurable
def train_loop(
    rank: int,
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: Optimizer,
    metric_logger: MetricsLogger,
    device: torch.device,
    num_epochs: int,
    num_batches: Optional[int] = None,
    output_trace: bool = False,
    metric_log_frequency: int = 1,
    checkpoint_frequency: int = 100,
    start_batch_idx: int = 0,
    # lr_scheduler: to-do: Add a scheduler
) -> None:
    model.train()
    batch_idx: int = start_batch_idx
    profiler = Profiler(rank) if output_trace else None

    for epoch in range(num_epochs):
        dataloader.sampler.set_epoch(epoch)  # pyre-ignore [16]
        for sample in dataloader:
            optimizer.zero_grad()
            sample.to(device)
            (
                _,
                _,
                aux_losses,
                mt_target_preds,
                mt_target_labels,
                mt_target_weights,
            ) = model.forward(
                sample.uih_features_kjt,
                sample.candidates_features_kjt,
            )
            # pyre-ignore
            sum(aux_losses.values()).backward()
            optimizer.step()
            metric_logger.update(
                mode="train",
                predictions=mt_target_preds,
                labels=mt_target_labels,
                weights=mt_target_weights,
                num_candidates=sample.candidates_features_kjt.lengths().view(
                    len(sample.candidates_features_kjt.keys()), -1
                )[0],
            )
            if batch_idx % metric_log_frequency != 0:
                metric_logger.compute_and_log(
                    mode="train",
                    additional_logs={
                        "losses": aux_losses,
                    },
                )
            if batch_idx % checkpoint_frequency == 0 and batch_idx > 0:
                save_dmp_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    metric_logger=metric_logger,
                    rank=rank,
                    batch_idx=batch_idx,
                )
            batch_idx += 1
            if output_trace:
                assert profiler is not None
                profiler.step()
            if num_batches is not None and batch_idx >= num_batches:
                break
        if num_batches is not None and batch_idx >= num_batches:
            break


@gin.configurable
def eval_loop(
    rank: int,
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    metric_logger: MetricsLogger,
    device: torch.device,
    metric_log_frequency: int = 1,
    num_batches: Optional[int] = None,
    output_trace: bool = False,
    # lr_scheduler: to-do: Add a scheduler
) -> None:
    model.eval()
    batch_idx: int = 0
    profiler = Profiler(rank) if output_trace else None
    metric_logger.reset(mode="eval")
    with torch.no_grad():
        for sample in dataloader:
            sample.to(device)
            (
                _,
                _,
                _,
                mt_target_preds,
                mt_target_labels,
                mt_target_weights,
            ) = model.forward(
                sample.uih_features_kjt,
                sample.candidates_features_kjt,
            )
            metric_logger.update(
                mode="eval",
                predictions=mt_target_preds,
                labels=mt_target_labels,
                weights=mt_target_weights,
                num_candidates=sample.candidates_features_kjt.lengths().view(
                    len(sample.candidates_features_kjt.keys()), -1
                )[0],
            )
            if batch_idx % metric_log_frequency != 0:
                metric_logger.compute_and_log(mode="eval")
            batch_idx += 1
            if output_trace:
                assert profiler is not None
                profiler.step()
            if num_batches is not None and batch_idx >= num_batches:
                break
    metric_logger.compute_and_log(mode="eval")
    for k, v in metric_logger.compute(mode="eval").items():
        print(f"{k}: {v}")


class _PipelineModelWrapper(torch.nn.Module):
    """Adapt ``DlrmHSTU.forward`` to the ``(loss, output)`` contract that
    ``TrainPipelineSparseDist`` expects.

    The wrapped ``model`` is the same DMP instance handed to the pipeline as
    ``model=``; the pipeline rewrites its sharded ``EmbeddingCollection`` in
    place, so calling it here is what lets the embedding all-to-all overlap the
    dense forward/backward compute.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self._model = model

    def forward(
        self, batch: Samples
    ) -> Tuple[torch.Tensor, Tuple[Any, ...]]:
        # The model runs in `_pipeline_mode`: it takes the whole batch as its
        # single arg and reads the pre-merged sparse KJT off it. This keeps the
        # EmbeddingCollection input a plain getattr on the batch placeholder so
        # TorchRec pipelines its input_dist (instead of skipping it for "input
        # modifications").
        (
            _,
            _,
            aux_losses,
            mt_target_preds,
            mt_target_labels,
            mt_target_weights,
        ) = self._model(batch)
        loss = sum(aux_losses.values())
        num_candidates = batch.candidates_features_kjt.lengths().view(
            len(batch.candidates_features_kjt.keys()), -1
        )[0]
        output = (
            aux_losses,
            mt_target_preds,
            mt_target_labels,
            mt_target_weights,
            num_candidates,
        )
        return loss, output


def build_train_pipeline(
    model: torch.nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    grad_clip_norm: float = 1.0,
) -> Any:
    """Build a ``TrainPipelineSparseDist`` for the DMP-wrapped HSTU model.

    The 3-stage pipeline overlaps (1) H2D transfer of batch N+2, (2) the sparse
    data-dist all-to-all of batch N+1's embedding lookup, and (3) dense fwd/bwd
    of batch N, on separate CUDA streams. Requires the model to be wrapped with
    ``DistributedModelParallel`` (see ``make_optimizer_and_shard``).
    """
    # Lazy import: keeps module import working on torchrec builds that move or
    # rename the pipeline, and matches the reference Primus-DLRM setup.
    from torchrec.distributed.train_pipeline import TrainPipelineSparseDist

    # Switch the (DMP-wrapped) HSTU model into pipeline mode so both the fx trace
    # and the live forward consume the batch as a single arg and read the
    # pre-merged sparse KJT off it — required for the embedding input_dist to be
    # pipelined. Eval call sites pass the batch the same way (see train_eval_loop).
    underlying = model.module if hasattr(model, "module") else model
    underlying._pipeline_mode = True

    # The pipeline calls backward()+optimizer.step() internally inside
    # progress(), leaving no in-loop hook point for gradient clipping. Clip via
    # a full-backward hook (fires after autograd populates dense grads, before
    # the optimizer step) to preserve parity with the sequential path's
    # clip_grad_norm_(model.parameters(), max_norm=1.0).
    if grad_clip_norm and grad_clip_norm > 0:

        def _clip_grads(_m: torch.nn.Module, _gi: Any, _go: Any) -> None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=grad_clip_norm
            )

        model.register_full_backward_hook(_clip_grads)

    return TrainPipelineSparseDist(
        model=model,
        optimizer=optimizer,
        device=device,
        execute_all_batches=True,
        custom_model_fwd=_PipelineModelWrapper(model),
    )


@gin.configurable
def train_eval_loop(
    rank: int,
    model: torch.nn.Module,
    optimizer: Optimizer,
    metric_logger: MetricsLogger,
    device: torch.device,
    num_epochs: int,
    num_train_batches: Optional[int] = None,
    num_eval_batches: Optional[int] = None,
    train_dataloader: Optional[torch.utils.data.DataLoader] = None,
    eval_dataloader: Optional[torch.utils.data.DataLoader] = None,
    output_trace: bool = False,
    metric_log_frequency: int = 1,
    checkpoint_frequency: int = 100,
    eval_frequency: int = 1,
    start_train_batch_idx: int = 0,
    start_eval_batch_idx: int = 0,
    use_pipeline: bool = False,
    # lr_scheduler: to-do: Add a scheduler
) -> None:
    train_batch_idx: int = start_train_batch_idx
    eval_batch_idx: int = start_eval_batch_idx
    profiler = Profiler(rank) if output_trace else None
    assert train_dataloader is not None and eval_dataloader is not None

    eval_data_iterator = iter(eval_dataloader)
    train_data_iterator = iter(train_dataloader)

    # 3-stage TorchRec pipeline (overlaps embedding a2a with dense compute).
    # When enabled, progress() owns H2D copy, sparse-dist, fwd/bwd and the
    # optimizer step; grad clipping moves to a full-backward hook (see builder).
    train_pipeline = (
        build_train_pipeline(model, optimizer, device) if use_pipeline else None
    )

    for epoch in range(num_epochs):
        train_dataloader.sampler.set_epoch(epoch)  # pyre-ignore [16]
        while True:
            model.train()
            if train_pipeline is not None:
                try:
                    (
                        aux_losses,
                        mt_target_preds,
                        mt_target_labels,
                        mt_target_weights,
                        num_candidates,
                    ) = train_pipeline.progress(train_data_iterator)
                except StopIteration:
                    train_data_iterator = iter(train_dataloader)
                    break
            else:
                try:
                    sample = next(train_data_iterator)
                except StopIteration:
                    train_data_iterator = iter(train_dataloader)
                    break
                optimizer.zero_grad()
                sample.to(device)
                (
                    _,
                    _,
                    aux_losses,
                    mt_target_preds,
                    mt_target_labels,
                    mt_target_weights,
                ) = model.forward(
                    sample.uih_features_kjt,
                    sample.candidates_features_kjt,
                )
                # pyre-ignore
                sum(aux_losses.values()).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                num_candidates = sample.candidates_features_kjt.lengths().view(
                    len(sample.candidates_features_kjt.keys()), -1
                )[0]
            metric_logger.update(
                mode="train",
                predictions=mt_target_preds,
                labels=mt_target_labels,
                weights=mt_target_weights,
                num_candidates=num_candidates,
            )
            if train_batch_idx % metric_log_frequency == 0:
                metric_logger.compute_and_log(
                    mode="train",
                    additional_logs={
                        "losses": aux_losses,
                    },
                )
            if train_batch_idx % checkpoint_frequency == 0 and train_batch_idx > 0:
                save_dmp_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    metric_logger=metric_logger,
                    rank=rank,
                    batch_idx=train_batch_idx,
                )
            train_batch_idx += 1
            if output_trace:
                assert profiler is not None
                profiler.step()
            if train_batch_idx % eval_frequency == 0:
                model.eval()
                eval_batch_idx: int = 0
                with torch.no_grad():
                    while True:
                        try:
                            sample = next(eval_data_iterator)
                        except StopIteration:
                            eval_data_iterator = iter(eval_dataloader)
                            sample = next(eval_data_iterator)
                        sample.to(device)
                        (
                            _,
                            _,
                            _,
                            mt_target_preds,
                            mt_target_labels,
                            mt_target_weights,
                        ) = (
                            # In pipeline mode the model takes the batch as one
                            # arg (see _PipelineModelWrapper / DlrmHSTU.forward).
                            model.forward(sample)
                            if use_pipeline
                            else model.forward(
                                sample.uih_features_kjt,
                                sample.candidates_features_kjt,
                            )
                        )
                        metric_logger.update(
                            mode="eval",
                            predictions=mt_target_preds,
                            labels=mt_target_labels,
                            weights=mt_target_weights,
                            num_candidates=sample.candidates_features_kjt.lengths().view(
                                len(sample.candidates_features_kjt.keys()), -1
                            )[0],
                        )
                        eval_batch_idx += 1
                        if output_trace:
                            assert profiler is not None
                            profiler.step()
                        if eval_batch_idx % metric_log_frequency == 0:
                            metric_logger.compute_and_log(mode="eval")
                        if (
                            num_eval_batches is not None
                            and eval_batch_idx >= num_eval_batches
                        ):
                            break
                    for k, v in metric_logger.compute(mode="eval").items():
                        print(f"{k}: {v}")
                model.train()
            # `num_train_batches` cap: None or 0 = run the whole window. >0 caps
            # batches per window (mostly the streaming-resume test driver uses
            # this to keep test windows short).
            if num_train_batches and train_batch_idx >= num_train_batches:
                break


def select_in_window_checkpoint_reason(
    *,
    train_batch_idx: int,
    global_step: int,
    elapsed_since_last_save: float,
    in_window_checkpoint_frequency: int,
    checkpoint_step_frequency: int,
    checkpoint_time_interval_s: float,
) -> Optional[str]:
    """Decide which (if any) in-window checkpoint cadence fires this batch.

    Pure / distributed-agnostic so it can be unit-tested without a real run.
    The caller computes `elapsed_since_last_save` (broadcast from rank 0 in the
    streaming loop) so all ranks pass the same value and reach the same verdict.

    Precedence (at most one save per batch): per-window-local batch count >
    monotonic global step > wall-clock interval. Returns the trigger reason
    string, or None when no cadence fires. A cadence is disabled when its
    frequency/interval is 0 / 0.0.

    Counter conventions match the loop: `train_batch_idx` is already
    post-incremented (>=1 on the first batch), and `global_step` is guarded
    >0 so step 0 doesn't trivially satisfy `% N == 0`.
    """
    if (
        in_window_checkpoint_frequency > 0
        and train_batch_idx % in_window_checkpoint_frequency == 0
    ):
        return "in_window_batch"
    if (
        checkpoint_step_frequency > 0
        and global_step > 0
        and global_step % checkpoint_step_frequency == 0
    ):
        return "global_step"
    if (
        checkpoint_time_interval_s > 0
        and elapsed_since_last_save >= checkpoint_time_interval_s
    ):
        return "time_interval"
    return None


def _validate_split_contract(
    saved: Optional[Dict[str, Any]],
    live: Dict[str, Any],
    rank: int,
) -> None:
    """Guarantee the train:eval split (and the inputs the resume skip-offset
    depends on) are unchanged across a crash/resume.

    `saved` is the contract recovered from the checkpoint (None on cold start or
    legacy pre-holdout checkpoints). Any mismatch is fatal: continuing would
    either desync the mid-window skip (duplicate/skip batches) or reassign users
    so that previously held-out eval users get trained (leakage). Set
    ALLOW_SPLIT_MISMATCH=1 to override (e.g. intentionally resuming a legacy
    checkpoint into a holdout run, accepting the risk).
    """
    allow = os.environ.get("ALLOW_SPLIT_MISMATCH", "0") == "1"
    if saved is None:
        # Legacy / cold-start checkpoint with no recorded contract. Only a
        # problem if this run actually holds users out (tsp < 1.0): we cannot
        # prove the earlier run used the same split.
        if live.get("train_split_percentage", 1.0) < 1.0 and not allow:
            raise RuntimeError(
                "Resuming a checkpoint with NO saved split contract into a "
                f"user-holdout run (train_split_percentage="
                f"{live['train_split_percentage']}). The earlier run's split "
                "cannot be verified, so held-out eval users may have been "
                "trained. Set ALLOW_SPLIT_MISMATCH=1 to override."
            )
        return
    mismatches = {
        k: (saved.get(k), live.get(k))
        for k in live
        if saved.get(k) != live.get(k)
    }
    if mismatches:
        msg = (
            "Split/resume contract mismatch between checkpoint and current run: "
            + ", ".join(
                f"{k}: checkpoint={s!r} current={c!r}" for k, (s, c) in mismatches.items()
            )
            + ". Resuming would desync the skip offset and/or leak held-out "
            "users into training."
        )
        if allow:
            if rank == 0:
                logger.warning("%s ALLOW_SPLIT_MISMATCH=1 set — continuing anyway.", msg)
        else:
            raise RuntimeError(msg + " Set ALLOW_SPLIT_MISMATCH=1 to override.")
    elif rank == 0:
        logger.info("Split/resume contract verified against checkpoint: %s", live)


@gin.configurable
def streaming_train_eval_loop(
    rank: int,
    model: torch.nn.Module,
    optimizer: Optimizer,
    metric_logger: MetricsLogger,
    device: torch.device,
    num_train_ts: int,
    hstu_config: DlrmHSTUConfig,
    embedding_table_configs: Dict[str, EmbeddingConfig],
    num_train_batches: Optional[int] = None,
    num_eval_batches: Optional[int] = None,
    output_trace: bool = False,
    metric_log_frequency: int = 1,
    checkpoint_frequency: int = 100,
    start_ts: int = 0,
    persistent_loader: bool = False,
    eval_each_window: bool = True,
    eval_every_n_windows: int = 1,
    double_buffer: bool = False,
    # --- fixed user-holdout eval set ---
    # Window range the fixed eval set is drawn from. None -> default to
    # original_end_ts (start_ts + num_train_ts), the window just past training.
    eval_holdout_ts: Optional[int] = None,
    eval_holdout_num_windows: int = 1,
    # --- resume / mid-window-exact-once knobs ---
    resume_train_ts: Optional[int] = None,
    resume_batch_idx_in_window: int = WINDOW_COMPLETE,
    # Split contract recovered from the checkpoint (None on cold start or
    # legacy checkpoints). Validated below against the live split so a resumed
    # run cannot silently train a different user-split (would leak).
    resume_split_contract: Optional[Dict[str, Any]] = None,
    # True iff no checkpoint was loaded (genuine fresh run). Distinguishes a
    # cold start (safe to establish a new split) from a resume that merely lacks
    # a contract (legacy/non-streaming checkpoint), which the guard must reject.
    resume_cold_start: bool = False,
    in_window_checkpoint_frequency: int = 0,
    # --- global step / wall-clock checkpoint cadences ---
    checkpoint_step_frequency: int = 0,
    checkpoint_time_interval_s: float = 0.0,
    # --- test-only failure injection knob ---
    die_at_step: int = -1,
) -> None:
    """Streaming train+eval loop with per-window (and optionally mid-window)
    checkpoints.

    Resume semantics (set by train_ranker after `load_dmp_checkpoint` returns):
      - resume_train_ts=None: cold start; honor `start_ts` as-is.
      - resume_train_ts=N, resume_batch_idx_in_window=WINDOW_COMPLETE(-1):
        previous run finished window N cleanly. Start at N+1 from sample 0.
      - resume_train_ts=N, resume_batch_idx_in_window=K (K>=0): previous run
        crashed mid-window after K completed batches. Re-enter window N and
        skip the first K batches of THIS rank's per-rank sample list (deterministic
        slice since `window_indices(N)` is a pure function of the anchor_ts cache).

    Checkpoint cadences (all independent; any combination may be enabled):
      - `checkpoint_frequency`: window-granularity. End-of-window save every
        Nth train_ts (and always on the final window). Uses WINDOW_COMPLETE.
      - `in_window_checkpoint_frequency`: per-window-local batch count. Fires
        every N batches *within* a window (counter resets each window).
      - `checkpoint_step_frequency`: global-step granularity. Fires whenever
        the monotonic `metric_logger.global_step['train']` hits a multiple of
        N — i.e. a true "every 1000 steps" trigger that spans windows and
        survives resume (global_step is restored from the checkpoint).
      - `checkpoint_time_interval_s`: wall-clock granularity. Fires when at
        least this many seconds have elapsed since the last save (e.g. 3600
        for hourly). Rank 0 owns the clock and broadcasts the decision so all
        ranks save together (avoids the collective barrier in
        `save_dmp_checkpoint` deadlocking on a split decision).

    All in-window triggers (`in_window_checkpoint_frequency`,
    `checkpoint_step_frequency`, `checkpoint_time_interval_s`) route through
    `_save_mid_window`, which stamps `batch_idx_in_window=K` so a crash leaves
    a resumable partial-window checkpoint. End-of-window saves
    (`checkpoint_frequency`) always use the WINDOW_COMPLETE sentinel. 0 / 0.0
    disables a given cadence (the default for all three fine-grained ones).

    `die_at_step` is a test-only hook: when `metric_logger.global_step['train']`
    reaches this value, the process exits with code 42 right after the in-window
    save fires. Used by the failure-injection test to crash at a deterministic
    boundary and then resume.
    """
    profiler = Profiler(rank) if output_trace else None
    # Normalize the per-window caps: <=0 (the env-binding default) means "no cap
    # = consume the full window". The eval-break check below is `is not None and
    # eval_batch_idx >= num_eval_batches`, so a literal 0 would (wrongly) break
    # after the first batch — map it to None instead for the full-holdout eval.
    if num_eval_batches is not None and num_eval_batches <= 0:
        num_eval_batches = None
    if num_train_batches is not None and num_train_batches <= 0:
        num_train_batches = None
    dataset_class, kwargs = get_dataset()
    kwargs["embedding_config"] = embedding_table_configs
    dataset = HammerToTorchDataset(
        dataset=dataset_class(hstu_config=hstu_config, is_inference=False, **kwargs)
    )
    # Persistent path: build ONE DataLoader + a stateful sampler whose indices
    # are swapped per window, so workers fork once and are reused across all
    # windows (eliminates the per-window dataloader respawn + first-batch
    # warmup). The non-persistent path recreates a DataLoader per window.
    window_sampler: Optional[StreamingWindowSampler] = None
    persistent_dl: Optional[DataLoader] = None
    world_size = (
        torch.distributed.get_world_size()
        if torch.distributed.is_initialized()
        else 1
    )
    if persistent_loader:
        window_sampler = StreamingWindowSampler(rank=rank, world_size=world_size)
        persistent_dl = make_persistent_streaming_dataloader(
            dataset=dataset, sampler=window_sampler
        )

    # The fixed user-holdout eval is yambda-specific (needs window_indices +
    # the split API). Other streaming datasets (synthetic) keep the legacy
    # per-window eval. Detect support once.
    supports_holdout = hasattr(dataset.dataset, "eval_holdout_indices")

    # Fixed eval-holdout window range. Captured from the REQUESTED (start_ts,
    # num_train_ts) BEFORE the resume block mutates them, so it is identical on
    # cold start and on every resume (the supervisor relaunches with the same
    # START_TS / NUM_TRAIN_TS). Defaults to the window just past training.
    requested_end_ts = start_ts + num_train_ts
    # Eval-cadence anchor: the ORIGINAL requested start_ts, captured BEFORE the
    # resume block rebases start_ts. `_should_eval` keys the every-N-windows
    # cadence off the absolute window ts relative to THIS anchor, so the eval
    # grid (e.g. 150,160,170,...) is identical on cold start and on every resume.
    # (Keying off the per-call loop index instead would re-anchor the grid to
    # whatever window a mid-run resume happens to restart from.)
    eval_anchor_ts = start_ts
    # None (Python default) or <0 (the env-binding default) both mean "use the
    # window just past training", which is stable across resume.
    eval_holdout_ts_resolved = (
        eval_holdout_ts
        if (eval_holdout_ts is not None and eval_holdout_ts >= 0)
        else requested_end_ts
    )

    # The split is an immutable run contract: a silent change across resume
    # would both desync the mid-window skip offset AND turn held-out eval users
    # into trained users (leakage). Build the live contract and validate the
    # one recovered from the checkpoint against it; abort on any mismatch unless
    # ALLOW_SPLIT_MISMATCH=1 is set (e.g. deliberately resuming a legacy run).
    live_split_contract: Optional[Dict[str, Any]] = None
    if supports_holdout:
        live_split_contract = {
            "train_split_percentage": dataset.dataset._train_split_percentage,  # pyre-ignore[16]
            "split_salt": dataset.dataset._split_salt,  # pyre-ignore[16]
            "eval_holdout_ts": eval_holdout_ts_resolved,
            "eval_holdout_num_windows": eval_holdout_num_windows,
            "batch_size": persistent_dl.batch_size if persistent_dl is not None else None,
            "world_size": world_size,
        }
        # Only validate on an actual resume. On a genuine cold start there is no
        # prior split to verify and establishing this run's split is always safe;
        # validating there would wrongly reject every fresh holdout run. A resume
        # that lacks a contract (legacy/non-streaming checkpoint) is NOT a cold
        # start and is still validated (and rejected) below.
        if not resume_cold_start:
            _validate_split_contract(resume_split_contract, live_split_contract, rank)

    # Apply resume hint: advance start_ts past the last completed window, or
    # re-enter the partial window with a per-rank skip on its first iter.
    # Shrink num_train_ts by the same amount so the resumed run finishes at
    # the same final timestamp (start_ts + num_train_ts) as a fresh run would
    # — i.e. resumed and uninterrupted produce identical total work.
    first_skip_samples = 0
    if resume_train_ts is not None:
        original_end_ts = start_ts + num_train_ts
        if resume_batch_idx_in_window == WINDOW_COMPLETE:
            new_start = resume_train_ts + 1
            if rank == 0:
                logger.info(
                    "Resuming from completed train_ts=%d → start_ts=%d "
                    "(num_train_ts %d → %d)",
                    resume_train_ts, new_start,
                    num_train_ts, max(0, original_end_ts - new_start),
                )
            start_ts = new_start
        else:
            if rank == 0:
                logger.info(
                    "Resuming mid-window at train_ts=%d batch_idx_in_window=%d "
                    "(skipping batches already trained)",
                    resume_train_ts,
                    resume_batch_idx_in_window,
                )
            start_ts = resume_train_ts
            # `batch_size` is per-rank from the persistent dataloader (set via
            # gin `make_persistent_streaming_dataloader.batch_size`). The
            # skip-samples-per-rank below maps "K batches done" → "K * bs
            # samples in this rank's index list", since each batch draws bs
            # samples from this rank's deterministic round-robin slice.
            assert persistent_dl is not None, (
                "Mid-window resume requires persistent_loader=True"
            )
            first_skip_samples = resume_batch_idx_in_window * persistent_dl.batch_size
        num_train_ts = max(0, original_end_ts - start_ts)
        if num_train_ts == 0 and rank == 0:
            logger.info(
                "Resume target already reached (end_ts=%d, start_ts=%d) — "
                "no further training windows; skipping straight to final eval.",
                original_end_ts, start_ts,
            )

    def _window_iter(ts: int, skip_samples: int = 0):
        # TRAIN-only iterator: both branches exclude held-out eval users via
        # train_window_indices / set_ts(train_only=True). (Eval uses the fixed
        # holdout set, never this helper.)
        if persistent_loader:
            assert window_sampler is not None and persistent_dl is not None
            window_sampler.set_window(
                dataset.dataset.train_window_indices(ts),  # pyre-ignore [16]
                skip_samples=skip_samples,
            )
            return iter(persistent_dl)
        if skip_samples != 0:
            raise NotImplementedError(
                "skip_samples>0 requires persistent_loader=True"
            )
        return iter(
            make_streaming_dataloader(dataset=dataset, ts=ts, train_only=True)
        )
    # Windows are [start_ts, start_ts + num_train_ts); each step trains window T
    # then evals window T+1, so the last eval window is start_ts + num_train_ts,
    # which must be < num_windows(). Anchors require >= history_length prior
    # events, so the earliest windows are near-empty warm-up — use start_ts to
    # begin at a dense window. Clamp instead of failing.
    if hasattr(dataset.dataset, "num_windows"):
        available = dataset.dataset.num_windows()  # pyre-ignore [16]
        max_count = max(0, available - 1 - start_ts)
        if num_train_ts > max_count:
            logger.warning(
                f"start_ts={start_ts} + num_train_ts={num_train_ts} exceeds "
                f"available windows ({available}); clamping num_train_ts to {max_count}."
            )
            num_train_ts = max_count
    # Wall-clock anchor for time-based checkpointing. Mutable single-element
    # list so the nested train loop can reset it after each save. Starts at
    # loop entry so the first time-trigger fires ~interval seconds in.
    last_ckpt_time = [time.time()]

    def _broadcast_elapsed() -> float:
        """Seconds since the last save, owned by rank 0 and broadcast to all
        ranks. save_dmp_checkpoint runs a collective barrier, so every rank must
        feed the same wall-clock value into the cadence decision — otherwise a
        split verdict (rank 0 saves, rank 1 doesn't) would deadlock. Broadcasting
        rank 0's elapsed keeps the (pure) decision identical everywhere."""
        elapsed = time.time() - last_ckpt_time[0]
        if torch.distributed.is_initialized() and world_size > 1:
            t = torch.tensor([elapsed], device=device, dtype=torch.float64)
            torch.distributed.broadcast(t, src=0)
            elapsed = float(t.item())
        return elapsed

    def _save_mid_window(train_ts: int, batch_idx_in_window: int) -> None:
        """In-window checkpoint helper. Snapshots the same state as the
        end-of-window save but stamps `batch_idx_in_window=K` instead of
        WINDOW_COMPLETE so the resume path knows to skip K batches.
        Uses train_ts as the numeric subdir name — every save into the same
        train_ts overwrites the previous in-window snapshot (via atomic
        replace), so disk stays bounded to keep_last_n train_ts dirs."""
        save_dmp_checkpoint(
            model=model,
            optimizer=optimizer,
            metric_logger=metric_logger,
            rank=rank,
            batch_idx=train_ts,
            train_ts=train_ts,
            batch_idx_in_window=batch_idx_in_window,
            device=device,
            split_contract=live_split_contract,
        )

    def _run_train_window(
        train_data_iterator,
        train_ts: int,
        start_batch_idx: int = 0,
        label: Optional[str] = None,
    ) -> None:
        # `start_batch_idx` is set when we're re-entering a window that was
        # interrupted mid-way (in_window resume); the dataloader iterator was
        # already advanced past those batches via the sampler skip, and we
        # account for them in the local counter so in-window saves and the
        # die_at_step hook fire at the right relative offsets.
        train_batch_idx = start_batch_idx
        first_wait: Optional[float] = None
        while True:
            model.train()
            _t_next = time.perf_counter() if (label and rank == 0) else None
            try:
                sample = next(train_data_iterator)
            except StopIteration:
                break
            if _t_next is not None and first_wait is None:
                first_wait = time.perf_counter() - _t_next
            optimizer.zero_grad()
            sample.to(device)
            (
                _,
                _,
                aux_losses,
                mt_target_preds,
                mt_target_labels,
                mt_target_weights,
            ) = model.forward(
                sample.uih_features_kjt,
                sample.candidates_features_kjt,
            )
            # pyre-ignore
            sum(aux_losses.values()).backward()
            optimizer.step()
            metric_logger.update(
                mode="train",
                predictions=mt_target_preds,
                labels=mt_target_labels,
                weights=mt_target_weights,
                num_candidates=sample.candidates_features_kjt.lengths().view(
                    len(sample.candidates_features_kjt.keys()), -1
                )[0],
            )
            if train_batch_idx % metric_log_frequency == 0:
                metric_logger.compute_and_log(
                    mode="train",
                    additional_logs={
                        "losses": aux_losses,
                    },
                )
            train_batch_idx += 1
            if output_trace:
                assert profiler is not None
                profiler.step()
            # Fine-grained in-window checkpoint triggers. All stamp
            # batch_idx_in_window so a crash here leaves a resumable partial
            # checkpoint, and all fire AFTER the metric update so restored
            # state reflects the just-completed batch. Triggers are mutually
            # short-circuited (one save per batch max) but evaluated on the
            # same deterministic counters across all ranks, so the collective
            # inside save_dmp_checkpoint stays in lockstep.
            gstep = metric_logger.global_step["train"]
            # Wall-clock elapsed is broadcast from rank 0 so every rank feeds
            # the same value into the (otherwise pure) cadence decision.
            elapsed = (
                _broadcast_elapsed() if checkpoint_time_interval_s > 0 else 0.0
            )
            save_reason = select_in_window_checkpoint_reason(
                train_batch_idx=train_batch_idx,
                global_step=gstep,
                elapsed_since_last_save=elapsed,
                in_window_checkpoint_frequency=in_window_checkpoint_frequency,
                checkpoint_step_frequency=checkpoint_step_frequency,
                checkpoint_time_interval_s=checkpoint_time_interval_s,
            )
            if save_reason is not None:
                if rank == 0:
                    logger.info(
                        "checkpoint trigger=%s train_ts=%d batch=%d global_step=%d",
                        save_reason,
                        train_ts,
                        train_batch_idx,
                        gstep,
                    )
                _save_mid_window(train_ts, train_batch_idx)
                # Reset the wall-clock anchor on ANY save so the next time
                # trigger is measured from the most recent checkpoint.
                last_ckpt_time[0] = time.time()
            # Test-only: deterministic crash for the failure-injection test.
            # Triggered AFTER the save above, so on resume we re-enter at
            # batch_idx_in_window=train_batch_idx and emit batches [K+1, end).
            if (
                die_at_step >= 0
                and metric_logger.global_step["train"] >= die_at_step
            ):
                if rank == 0:
                    logger.warning(
                        "die_at_step=%d hit at train_ts=%d batch=%d global_step=%d "
                        "→ sys.exit(42)",
                        die_at_step,
                        train_ts,
                        train_batch_idx,
                        metric_logger.global_step["train"],
                    )
                # Distributed barrier so all ranks exit together rather than
                # leaving a few ranks hanging on NCCL ops.
                torch.distributed.barrier()
                import sys
                sys.exit(42)
            # `num_train_batches` cap: None or 0 = run the whole window. >0 caps
            # batches per window (mostly the streaming-resume test driver uses
            # this to keep test windows short).
            if num_train_batches and train_batch_idx >= num_train_batches:
                break
        if label and rank == 0 and first_wait is not None:
            logger.info(
                f"[boundary] {label} train first-batch data-wait={first_wait * 1000:.1f}ms"
            )

    def _run_eval_window(eval_data_iterator, label: Optional[str] = None) -> None:
        # DO NOT add a checkpoint trigger anywhere inside this function. The eval
        # data iterator's position is not serializable, so a checkpoint taken
        # mid-eval could not be resumed deterministically. `_maybe_checkpoint`
        # only fires after a completed eval window or mid-train-window, so any
        # restored state always sits on a completed-eval boundary -- which is
        # also why the eval reset below is safe across resume.
        model.eval()
        # Reset eval metrics so each pass reports a clean number over the FIXED
        # holdout set. Without this, lifetime/window eval metrics would keep
        # accumulating across eval steps (the old behavior, made worse now that
        # every step sees the identical set), making the eval-AUC trajectory
        # uninterpretable. With the reset, each eval point == AUC over the whole
        # fixed holdout at that train step -> directly comparable across steps.
        metric_logger.reset(mode="eval")
        eval_batch_idx = 0
        first_wait: Optional[float] = None
        _t_enter = time.perf_counter() if (label and rank == 0) else None
        with torch.no_grad():
            while True:
                _t_next = time.perf_counter() if (label and rank == 0) else None
                try:
                    sample = next(eval_data_iterator)
                except StopIteration:
                    break
                if _t_next is not None and first_wait is None:
                    first_wait = time.perf_counter() - _t_next
                sample.to(device)
                (
                    _,
                    _,
                    _,
                    mt_target_preds,
                    mt_target_labels,
                    mt_target_weights,
                ) = model.forward(
                    sample.uih_features_kjt,
                    sample.candidates_features_kjt,
                )
                metric_logger.update(
                    mode="eval",
                    predictions=mt_target_preds,
                    labels=mt_target_labels,
                    weights=mt_target_weights,
                    num_candidates=sample.candidates_features_kjt.lengths().view(
                        len(sample.candidates_features_kjt.keys()), -1
                    )[0],
                )
                eval_batch_idx += 1
                if output_trace:
                    assert profiler is not None
                    profiler.step()
                if eval_batch_idx % metric_log_frequency == 0:
                    metric_logger.compute_and_log(mode="eval")
                if num_eval_batches is not None and eval_batch_idx >= num_eval_batches:
                    break
            for k, v in metric_logger.compute(mode="eval").items():
                print(f"{k}: {v}")
        if label and rank == 0 and _t_enter is not None:
            _eval_total = time.perf_counter() - _t_enter
            _fw = (first_wait * 1000) if first_wait is not None else float("nan")
            logger.info(
                f"[boundary] {label} eval first-batch data-wait={_fw:.1f}ms "
                f"total_eval={_eval_total * 1000:.1f}ms batches={eval_batch_idx}"
            )

    def _maybe_checkpoint(train_ts: int) -> None:
        if (
            train_ts % checkpoint_frequency == 0 and train_ts > 0
        ) or train_ts == start_ts + num_train_ts - 1:
            # End-of-window save: stamp WINDOW_COMPLETE so resume advances past
            # this train_ts. `device` enables per-rank RNG snapshot for
            # bit-equal resume of dropout-bearing modules.
            save_dmp_checkpoint(
                model=model,
                optimizer=optimizer,
                metric_logger=metric_logger,
                rank=rank,
                batch_idx=train_ts,
                train_ts=train_ts,
                batch_idx_in_window=WINDOW_COMPLETE,
                device=device,
                split_contract=live_split_contract,
            )
            last_ckpt_time[0] = time.time()

    # Apply start_ts shift from resume (may have moved past the original start).
    # num_train_ts is the requested *count*; preserve it so the loop runs for
    # the same total number of windows post-resume as a fresh run would have.
    train_ts_list = list(range(start_ts, start_ts + num_train_ts))
    n_train = len(train_ts_list)

    def _should_eval(i: int) -> bool:
        """Whether to run the full-holdout eval after training window index `i`.

        `eval_every_n_windows<=1` (default) preserves the per-window cadence.
        For K>1 we eval when the ABSOLUTE window ts is on the grid anchored at
        `eval_anchor_ts` (the original start_ts), i.e. ts in {anchor, anchor+K,
        anchor+2K, ...}, and ALWAYS on the final window so the trajectory ends
        with an eval point. Anchoring to the absolute ts (not the per-call loop
        index `i`) keeps the eval grid (e.g. 150,160,170,...) stable across a
        mid-run resume, which rebases start_ts/`train_ts_list` to the resume
        window. Gated by `eval_each_window`.
        """
        if not eval_each_window:
            return False
        if eval_every_n_windows <= 1:
            return True
        return (train_ts_list[i] - eval_anchor_ts) % eval_every_n_windows == 0 or i == n_train - 1

    # Fixed eval set: held-out users' anchors over the resolved holdout window
    # range, computed ONCE and reused at every eval step. Same anchors every
    # step -> stable, comparable eval-AUC curve, and bounded eval time
    # (~(1 - train_split_percentage) of a window). Cached inside the dataset so
    # re-deriving it (e.g. on resume) returns the identical set. None for
    # datasets without holdout support (synthetic) -> legacy per-window eval.
    eval_global_indices: Optional["np.ndarray"] = None
    if supports_holdout:
        eval_global_indices = dataset.dataset.eval_holdout_indices(  # pyre-ignore [16]
            eval_holdout_ts_resolved, eval_holdout_num_windows
        )
        if rank == 0:
            logger.info(
                "Fixed eval holdout: ts=[%d, %d) -> %d anchors (train_split_percentage=%s)",
                eval_holdout_ts_resolved,
                eval_holdout_ts_resolved + eval_holdout_num_windows,
                len(eval_global_indices),
                dataset.dataset._train_split_percentage,  # pyre-ignore[16]
            )

    if persistent_loader and double_buffer:
        # Double-buffered: next window prepared in the background during the
        # current window's compute. Eval (if enabled) uses its own pre-forked
        # pool, primed up front on the main thread so no fork races a bg thread.
        prefetcher = _PrefetchingWindowLoader(
            dataset=dataset,
            sampler_factory=lambda: StreamingWindowSampler(rank, world_size),
            dl_factory=lambda s: make_persistent_streaming_dataloader(
                dataset=dataset, sampler=s
            ),
        )
        eval_sampler: Optional[StreamingWindowSampler] = None
        eval_dl: Optional[DataLoader] = None
        # Eval iterator is built one window ahead: the eval pool (idle while the
        # current train window runs) prefetches the next eval window's first
        # batches concurrently with train compute, so eval starts warm. yambda's
        # sample content depends only on the sampler window, not is_eval, so
        # prefetching during train is safe.
        eval_iter: Optional[Iterator] = None
        if eval_each_window and len(train_ts_list) > 0:
            eval_sampler = StreamingWindowSampler(rank, world_size)
            eval_dl = make_persistent_streaming_dataloader(
                dataset=dataset, sampler=eval_sampler
            )
            # CRITICAL: fork the eval worker pool HERE, on the main thread,
            # BEFORE prefetcher.stream() below spins up its background prep
            # thread. The pool is persistent_workers=True, so this first iter()
            # is the ONLY fork; every later iter() merely resets and reuses these
            # workers (no fork), so it can never deadlock against the background
            # thread holding an allocator/GIL-released lock. (Deferring this
            # first fork into the loop — as a sparse-eval cadence naively might —
            # hangs the run.) _should_eval(0) is always True when eval is enabled
            # (0 % K == 0). The eval set is the FIXED holdout (same every step),
            # so we install it on the sampler ONCE here; later evals just call
            # iter() again to replay the identical set (no set_window churn).
            eval_sampler.set_window(eval_global_indices)
            eval_iter = iter(eval_dl)
        for i, (train_ts, train_data_iterator) in enumerate(
            # Only the FIRST window after a mid-window resume needs the skip
            # (handed via prefetcher.stream's first_skip_samples). The skip is
            # zero on cold start (resume_train_ts is None → first_skip_samples=0)
            # and on completed-window resume (mid-window slice is 0 too).
            prefetcher.stream(train_ts_list, first_skip_samples=first_skip_samples)
        ):
            dataset.dataset.is_eval = False  # pyre-ignore [16]
            # First iteration after a mid-window resume carries
            # resume_batch_idx_in_window so in-window saves and the die_at_step
            # hook keep accurate counters; otherwise count from 0.
            start_batch = (
                resume_batch_idx_in_window
                if i == 0 and resume_batch_idx_in_window > 0
                else 0
            )
            _run_train_window(
                train_data_iterator,
                train_ts=train_ts,
                start_batch_idx=start_batch,
                label=f"train_ts={train_ts}",
            )
            if _should_eval(i):
                dataset.dataset.is_eval = True  # pyre-ignore [16]
                assert eval_sampler is not None and eval_dl is not None
                _run_eval_window(eval_iter, label=f"eval_holdout@train_ts={train_ts}")
                # Re-arm the (already-forked) eval pool for the NEXT eval. The
                # holdout set is fixed, so the sampler window is unchanged; we
                # only need a fresh iter() to replay it. iter() reuses the
                # persistent workers — no fork, safe alongside the bg thread.
                next_eval_i = next(
                    (j for j in range(i + 1, n_train) if _should_eval(j)), None
                )
                if next_eval_i is not None:
                    eval_iter = iter(eval_dl)
            _maybe_checkpoint(train_ts)
    else:
        for i, train_ts in enumerate(train_ts_list):
            dataset.dataset.is_eval = False  # pyre-ignore [16]
            skip = first_skip_samples if i == 0 else 0
            start_batch = (
                resume_batch_idx_in_window
                if i == 0 and resume_batch_idx_in_window > 0
                else 0
            )
            _run_train_window(
                _window_iter(train_ts, skip_samples=skip),
                train_ts=train_ts,
                start_batch_idx=start_batch,
            )
            if _should_eval(i):
                dataset.dataset.is_eval = True  # pyre-ignore [16]
                if eval_global_indices is not None:
                    _run_eval_window(
                        iter(
                            make_streaming_dataloader(
                                dataset=dataset, indices=eval_global_indices
                            )
                        ),
                        label=f"eval_holdout@train_ts={train_ts}",
                    )
                else:
                    # Legacy per-window eval (datasets without user holdout).
                    _run_eval_window(
                        iter(make_streaming_dataloader(dataset=dataset, ts=train_ts + 1))
                    )
            _maybe_checkpoint(train_ts)

    # Final eval over the SAME fixed user-holdout set (consistent with the
    # per-window evals above). Reuses _run_eval_window so metrics are reset and
    # reported the same way. Falls back to the legacy final-window eval for
    # datasets without user holdout.
    dataset.dataset.is_eval = True  # pyre-ignore [16]
    if eval_global_indices is not None:
        _run_eval_window(
            iter(make_streaming_dataloader(dataset=dataset, indices=eval_global_indices)),
            label="eval_holdout@final",
        )
    else:
        _run_eval_window(
            iter(make_streaming_dataloader(dataset=dataset, ts=num_train_ts)),
            label="eval@final",
        )
    if rank == 0:
        for k, v in metric_logger.compute(mode="eval").items():
            print(f"{k}: {v}")
