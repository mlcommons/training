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
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.planner.types import ParameterConstraints
from torchrec.distributed.sharding_plan import get_default_sharders
from torchrec.distributed.types import ShardedTensor, ShardingEnv, ShardingType
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

# Embedding placement vocabulary -> torchrec compute kernel. Used by
# make_optimizer_and_shard to translate the gin/env placement strings
# ("hbm"/"uvm"/"uvm_caching") into ParameterConstraints. "auto" (or anything not
# in this map) means "no constraint": the planner decides from the HBM cap.
_PLACEMENT_TO_KERNEL: Dict[str, EmbeddingComputeKernel] = {
    "hbm": EmbeddingComputeKernel.FUSED,
    "uvm": EmbeddingComputeKernel.FUSED_UVM,
    "uvm_caching": EmbeddingComputeKernel.FUSED_UVM_CACHING,
}

# Per-table sharding-type vocabulary -> torchrec ShardingType. Used by
# make_optimizer_and_shard to pin a table's shard layout via ParameterConstraints
# (e.g. move a hot, data-skewed table off ROW_WISE to COLUMN_WISE so its
# embedding all-to-all load is balanced by rank instead of routed by row/value).
# "auto" (or anything not in this map) means "no constraint": the planner decides.
# Short aliases (rw/cw/tw/twrw) are accepted alongside the canonical names.
_SHARDING_TO_TYPE: Dict[str, ShardingType] = {
    "row_wise": ShardingType.ROW_WISE,
    "column_wise": ShardingType.COLUMN_WISE,
    "table_wise": ShardingType.TABLE_WISE,
    "table_row_wise": ShardingType.TABLE_ROW_WISE,
    "table_column_wise": ShardingType.TABLE_COLUMN_WISE,
    "data_parallel": ShardingType.DATA_PARALLEL,
    "rw": ShardingType.ROW_WISE,
    "cw": ShardingType.COLUMN_WISE,
    "tw": ShardingType.TABLE_WISE,
    "twrw": ShardingType.TABLE_ROW_WISE,
}


@gin.configurable
def seed_everything(seed: int = -1, rank: int = 0) -> None:
    """Seed all RNGs (same value on every rank) for reproducible dense weight init.

    Call right before make_model(), after setup() (process group needed for the
    broadcast) and the gin parse. seed < 0 ($SEED unset) draws a fresh random seed
    per run (rank 0 broadcasts; exported to $SEED); seed >= 0 reproduces a run.
    Data order/split are independent of this seed (StreamingWindowSampler/$SPLIT_SALT).
    """
    import random

    import numpy as np

    pinned = seed >= 0
    if not pinned:
        # rank 0 draws a random seed and broadcasts it so all ranks agree (an
        # identical seed on every rank is REQUIRED, else dense weight init
        # diverges across ranks and DDP/AllReduce trains garbage).
        if dist.is_available() and dist.is_initialized():
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
            drawn = int.from_bytes(os.urandom(4), "little") if rank == 0 else 0
            _seed_t = torch.tensor([drawn], dtype=torch.int64, device=device)
            dist.broadcast(_seed_t, src=0)
            seed = int(_seed_t.item())
        else:
            seed = int.from_bytes(os.urandom(4), "little")
    # Export the resolved value so the run is reproducible after the fact.
    os.environ["SEED"] = str(seed)

    logger.info(
        f"[rank {rank}] seeding all RNGs with SEED={seed} "
        f"({'pinned via $SEED' if pinned else 'random per-run; set $SEED to reproduce'})"
    )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@gin.configurable
def decorrelate_runtime_rng(rank: int = 0, enabled: bool = True) -> None:
    """Re-seed torch/cuda with $SEED + rank so HSTU dropout draws different masks
    per data-parallel rank (seed_everything's identical seed would draw the same).

    MUST run after make_model() + make_optimizer_and_shard() so init stays
    identical across ranks; it perturbs only forward-time stochasticity.
    Reproducible (pure fn of $SEED + rank; RNG state checkpointed). enabled=False
    keeps the legacy identical-mask behavior.
    """
    if not enabled:
        logger.info(
            f"[rank {rank}] decorrelate_runtime_rng disabled; dropout masks "
            f"identical across ranks"
        )
        return
    base = int(os.environ.get("SEED", "1"))
    offset_seed = base + int(rank)
    torch.manual_seed(offset_seed)
    torch.cuda.manual_seed_all(offset_seed)
    logger.info(
        f"[rank {rank}] decorrelated runtime RNG: SEED={base} + rank={rank} "
        f"=> {offset_seed} (per-rank dropout masks)"
    )


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
    # Process-group / NCCL watchdog timeout (seconds). Env-overridable so a
    # diagnostic run can use a short, finite timeout that trips the NCCL flight
    # recorder dump (TORCH_NCCL_TRACE_BUFFER_SIZE + TORCH_NCCL_DUMP_ON_TIMEOUT)
    # on a collective desync instead of hanging for the full default.
    TIMEOUT = int(os.environ.get("PG_TIMEOUT_S", "1800"))

    # set device BEFORE init_process_group so NCCL binds this rank to its
    # own GPU; otherwise every rank's first CUDA context lands on GPU 0,
    # leaving stale allocations and triggering OOMs on rank 0.
    torch.cuda.set_device(device)

    # NOTE: RNG seeding for reproducible weight init lives in seed_everything(),
    # which train_ranker calls right before make_model() (after setup() so the
    # process group is initialized for the cross-rank seed broadcast, and after
    # the full gin parse so the gin-configurable $SEED is bound). Seeding here
    # would be too early to be gin-configurable.

    # initialize the process group
    #
    # The default PG timeout must match TIMEOUT (not the 600s NCCL default):
    # checkpoint saves go through DCP collectives on *this* default PG, and the
    # 560GB sparse-embedding write is both slow on shared NFS and badly skewed
    # across ranks (shards range ~37GB..~95GB), so the fastest rank can sit in
    # the post-write allgather/barrier well past 600s waiting for the slowest
    # rank. The stock 600s watchdog then SIGABRTs an otherwise-healthy job.
    if not dist.is_initialized():
        dist.init_process_group(
            "nccl",
            rank=rank,
            world_size=world_size,
            device_id=device,
            timeout=timedelta(seconds=TIMEOUT),
        )

    pg = dist.new_group(
        backend=BACKEND,
        timeout=timedelta(seconds=TIMEOUT),
    )

    return pg


def cleanup() -> None:
    dist.destroy_process_group()


def _window_boundary_barrier(
    device: torch.device, world_size: int, train_ts: int
) -> None:
    """Collective rendezvous at a streaming window boundary.

    The per-window data prep (``window_indices``: an O(N) mask over the ~18 GB
    mmap'd ``anchor_ts`` array) can complete at very different times across
    ranks. The embedding input-dist all-to-all that follows is a collective, so
    if a fast rank reaches it while a slow rank is still in prep, the NCCL
    stream desyncs and the job deadlocks (one rank a collective behind the
    rest). Synchronizing here makes prep-time skew harmless: every rank waits
    until all ranks have a ready window before any issues the first forward.

    Cost is one near-zero-payload barrier per window (299 total over a full
    run). In the healthy case prep already overlapped the previous window's
    compute via the prefetcher, so the barrier returns immediately; it only
    blocks for the real prep skew it is there to absorb.
    """
    if not (dist.is_available() and dist.is_initialized()) or world_size <= 1:
        return
    t0 = time.time()
    if device.type == "cuda":
        dist.barrier(device_ids=[device.index])
    else:
        dist.barrier()
    waited = time.time() - t0
    # Surface non-trivial skew (the thing this barrier exists to absorb) so a
    # node with a slow rank is visible without trawling the flight recorder.
    if waited > 5.0:
        logger.warning(
            "[window-barrier] train_ts=%d: waited %.1fs at boundary "
            "rendezvous (per-rank data-prep skew)",
            train_ts,
            waited,
        )
    # Test/debug observability: the healthy-path barrier is otherwise SILENT
    # (the skew warning above only fires on >5s waits), so the resume e2e test
    # has no signal that the boundary rendezvous actually executed. When
    # WINDOW_BARRIER_DEBUG=1, rank 0 emits exactly one line per crossed window
    # so the test can assert the barrier ran at EVERY boundary (regression guard
    # for the desync deadlock the barrier fixes). Off by default — zero prod cost.
    if os.environ.get("WINDOW_BARRIER_DEBUG") == "1" and dist.get_rank() == 0:
        logger.info(
            "[window-barrier] train_ts=%d rendezvous complete (waited %.3fs)",
            train_ts,
            waited,
        )


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

    logger.info(
        f"[dense optimizer] {optimizer_name} learning_rate={learning_rate} "
        f"(resolved from gin; override via $DENSE_LR)"
    )
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

    logger.info(
        f"[sparse optimizer] {optimizer_name} learning_rate={learning_rate} "
        f"(resolved from gin; override via $SPARSE_LR)"
    )
    return optimizer_cls, kwargs, optimizer_factory


_FBGEMM_LOWMEM_PATCHED = False


def _patch_fbgemm_lowmem_clamp_cast(enabled: bool = True, rank0: bool = False) -> None:
    """Replace fbgemm's quant clamp+cast with a memory-frugal equivalent.

    ``enabled`` is the gin/env-driven kill switch (see
    ``make_optimizer_and_shard.qcomm_lowmem_clamp_cast`` /
    ``$QCOMM_LOWMEM_CODEC``). Default ON; pass ``enabled=False`` to fall back to
    stock fbgemm (e.g. to reproduce the pre-patch OOM, or if a future fbgemm
    version changes the codec and the patch needs revalidation).

    fbgemm's ``fp32_to_fp16_with_clamp`` (and the bf16 variant) does
    ``torch.clamp(tensor, MIN, MAX).half()``. ``torch.clamp(...)`` allocates a
    SECOND full-size fp32 tensor (same numel as the input) *before* the cast, so
    the transient peak is input(fp32) + clamp_temp(fp32) + output(fp16) ~= 2.5x
    the input. On a skewed row-wise-sharded batch the hottest shard's embedding
    tensor is huge (observed 81.5 GiB clamp temp), and that extra fp32 copy is
    exactly the allocation that OOMs the rank — which then exits the train loop
    while peers block forever in the a2a (a 30-min NCCL-watchdog hang). See
    HANG_ROOTCAUSE.md / flight-recorder dump for the full diagnosis.

    Cast FIRST then clamp IN PLACE: ``tensor.half().clamp_(MIN, MAX)``. This
    allocates only the fp16 output (no full-size fp32 temp), cutting the peak by
    the size of the input tensor, while being numerically identical: an fp32
    value above HALF_MAX casts to +inf, which clamp_ maps back to HALF_MAX (and
    NaNs pass through unchanged), matching clamp-then-cast bit for bit. Safe to
    do in place because the codec ``encode()`` runs inside the qcomm autograd
    ``Function.forward`` (grad disabled), so there is no graph to corrupt.
    """
    global _FBGEMM_LOWMEM_PATCHED
    if not enabled:
        if rank0:
            logger.warning(
                "[qcomm-lowmem] DISABLED (qcomm_lowmem_clamp_cast=False / "
                "QCOMM_LOWMEM_CODEC=0) — running stock fbgemm clamp+cast, which "
                "allocates a full-size fp32 clamp temp and can OOM->hang the "
                "hottest row-wise embedding shard on skewed batches."
            )
        return
    if _FBGEMM_LOWMEM_PATCHED:
        return
    try:
        from fbgemm_gpu import quantize_comm, quantize_utils

        _HMIN = quantize_utils.TORCH_HALF_MIN
        _HMAX = quantize_utils.TORCH_HALF_MAX
        _BMIN = quantize_utils.TORCH_BFLOAT16_MIN
        _BMAX = quantize_utils.TORCH_BFLOAT16_MAX

        def _lowmem_fp16(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.half().clamp_(_HMIN, _HMAX)

        def _lowmem_bf16(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.bfloat16().clamp_(_BMIN, _BMAX)

        # Patch BOTH the definition module and quantize_comm, which imported the
        # names directly (``from .quantize_utils import fp32_to_fp16_with_clamp``)
        # so its module-level reference must be overridden too.
        for _mod in (quantize_utils, quantize_comm):
            if hasattr(_mod, "fp32_to_fp16_with_clamp"):
                _mod.fp32_to_fp16_with_clamp = _lowmem_fp16
            if hasattr(_mod, "fp32_to_bf16_with_clamp"):
                _mod.fp32_to_bf16_with_clamp = _lowmem_bf16

        _FBGEMM_LOWMEM_PATCHED = True
        if rank0:
            logger.info(
                "[qcomm-lowmem] patched fbgemm fp32->fp16/bf16 clamp+cast to "
                "cast-then-clamp_ (drops the full-size fp32 clamp temp; avoids "
                "OOM on skewed row-wise embedding a2a)"
            )
    except Exception as e:  # noqa: BLE001 — patch is best-effort, never fatal
        if rank0:
            logger.warning(
                "[qcomm-lowmem] could not patch fbgemm clamp+cast (%s: %s); "
                "running with stock (higher-peak) quantizer",
                type(e).__name__,
                e,
            )


def _maybe_apply_qcomm_a2a(
    sharders: List[Any],
    device: torch.device,
    forward_precision: str = "fp32",
    backward_precision: str = "fp32",
    lowmem_clamp_cast: bool = True,
) -> List[Any]:
    """Optionally quantize the embedding all-to-all payload via TorchRec qcomm.

    The yambda-5b embedding shuffle is the dominant, bandwidth-bound (multi-node)
    collective (~14.5 GB/rank fp32); a bf16/fp16 wire dtype halves it. Quant/
    dequant happen inside the comm op, transparent to the lookup consumer. Ported
    from the DLRMv2 R2 lever, retargeted from ``EmbeddingBagCollectionSharder`` to
    the sequence ``EmbeddingCollectionSharder`` this model uses.

    Forward and backward are configured independently because they have different
    numerical needs (TorchRec golden_training/train_dlrm.py recommends
    forward=fp16, backward=bf16): the forward carries bounded embedding
    activations where fp16's extra mantissa helps, while gradients have a wider
    range that can overflow fp16, so bf16 (fp32 exponent range) is safer there.
    bf16 and fp16 are both 2 bytes, so the wire volume / perf is identical — the
    choice is purely numerical.

    Args (set via gin on ``make_optimizer_and_shard``, env-overridable). Each is
    one of ``fp32`` (that direction unquantized) | ``bf16`` | ``fp16``. If BOTH
    are fp32 the sharders are returned untouched (identical to baseline trunk).
    """
    _COMM = {"bf16": "BF16", "fp16": "FP16", "fp32": "FP32"}
    fwd = (forward_precision or "fp32").strip().lower()
    bwd = (backward_precision or "fp32").strip().lower()
    rank0 = (not dist.is_initialized()) or dist.get_rank() == 0
    for name, p in (("forward", fwd), ("backward", bwd)):
        if p not in _COMM:
            # Misconfigured precision: fail loudly rather than silently running
            # fp32. A typo in SPARSE_A2A_{FWD,BWD} must not pass as "no quant".
            raise ValueError(
                f"DLRMV4 qcomm a2a: unknown {name} precision {p!r} "
                f"(want one of fp32|bf16|fp16)"
            )
    if fwd == "fp32" and bwd == "fp32":
        return sharders
    # Before building the codec, swap fbgemm's clamp+cast for a memory-frugal
    # equivalent — see _patch_fbgemm_lowmem_clamp_cast for why (avoids a full
    # extra fp32 temp that OOMs the hottest row-wise shard on skewed batches).
    # Gated by `lowmem_clamp_cast` (gin/env); ON by default.
    _patch_fbgemm_lowmem_clamp_cast(enabled=lowmem_clamp_cast, rank0=rank0)
    try:
        from torchrec.distributed.embedding import EmbeddingCollectionSharder
        from torchrec.distributed.fbgemm_qcomm_codec import (
            CommType,
            get_qcomm_codecs_registry,
            QCommsConfig,
        )

        qcfg = QCommsConfig(
            forward_precision=getattr(CommType, _COMM[fwd]),
            backward_precision=getattr(CommType, _COMM[bwd]),
        )
        registry = get_qcomm_codecs_registry(qcfg, device=device)
    except Exception as e:  # noqa: BLE001
        # A configured quantized a2a that fails to build is a hard error. Silently
        # downgrading to fp32 would change numerics/throughput with no signal, and
        # a partial failure (one rank fp32, others fp16) would also desync the
        # collectives. Raise on every rank so the whole job aborts consistently.
        raise RuntimeError(
            f"DLRMV4 qcomm a2a: failed to enable configured quantization "
            f"(forward={fwd} backward={bwd}): {type(e).__name__}: {e}"
        ) from e

    new_sharders = []
    replaced = False
    for s in sharders:
        if type(s).__name__ == "EmbeddingCollectionSharder" and not replaced:
            new_sharders.append(
                EmbeddingCollectionSharder(qcomm_codecs_registry=registry)
            )
            replaced = True
        else:
            new_sharders.append(s)
    if not replaced:
        # Codec registry built fine, but there was no EmbeddingCollectionSharder to
        # bind it to, so the quantized a2a would be silently inert. Treat this as a
        # hard failure too — "configured but not applied" is the bug we want caught.
        raise RuntimeError(
            f"DLRMV4 qcomm a2a: quantization configured (forward={fwd} "
            f"backward={bwd}) but no EmbeddingCollectionSharder was found to attach "
            f"the qcomm codec registry to; refusing to run with quantization "
            f"silently disabled"
        )
    if rank0:
        logger.info(
            "DLRMV4 qcomm a2a ENABLED: forward=%s backward=%s "
            "replaced_ec_sharder=%s",
            fwd,
            bwd,
            replaced,
        )
    return new_sharders


def _maybe_apply_ec_index_dedup(
    sharders: List[Any],
    enabled: bool,
) -> List[Any]:
    """Optionally enable TorchRec's in-batch index dedup on the EC sharder.

    Runs as a FINAL pass AFTER get_default_sharders() and _maybe_apply_qcomm_a2a,
    so it covers BOTH sharder construction paths with a single knob:
      * qcomm ON  (SPARSE_A2A_* != fp32, the default): the EC sharder was already
        rebuilt with a qcomm codec registry inside _maybe_apply_qcomm_a2a.
      * qcomm OFF (SPARSE_A2A_* == fp32): that function early-returns the bare
        get_default_sharders() list untouched.
    In either case we rebuild the one EmbeddingCollectionSharder in place,
    PRESERVING its existing fused_params + qcomm codec registry, and only flip
    use_index_dedup. That ordering guarantees dedup composes with (never clobbers)
    the fp16/bf16 a2a instead of depending on which qcomm branch ran.

    use_index_dedup collapses duplicate ids within a batch to unique rows before
    the input all-to-all (torch.ops.fbgemm.jagged_unique_indices), does the
    embedding lookup once, then scatters back via reverse_indices. For a
    non-pooled EmbeddingCollection (this sequence model) that reconstruction is
    exact, so the forward is numerically lossless — unlike the fp16 a2a it does
    not perturb AUC/NE. The buffers it registers (_hash_size_cumsum_tensor_*,
    _hash_size_offset_tensor_*) are persistent=False, so it does NOT change the
    on-disk shard layout and can be toggled on an existing checkpoint with no
    reshard. Payoff scales with the in-batch duplicate rate: large under
    user-major batching (STREAMING_SHUFFLE_FRACTION=0, the default, where one
    user's UIH re-reads the same item/artist/album ids), shrinking toward
    neutral/negative as batch diversity rises.

    The knob only exists on EmbeddingCollectionSharder, which is the sole default
    sharder this model actually binds (a pure sequence EmbeddingCollection), so
    no other sharder needs touching. Configured via gin on
    ``make_optimizer_and_shard`` (env-overridable $EC_INDEX_DEDUP).
    """
    if not enabled:
        return sharders

    from torchrec.distributed.embedding import EmbeddingCollectionSharder

    rank0 = (not dist.is_initialized()) or dist.get_rank() == 0
    new_sharders = []
    replaced = False
    for s in sharders:
        if type(s).__name__ == "EmbeddingCollectionSharder" and not replaced:
            new_sharders.append(
                EmbeddingCollectionSharder(
                    fused_params=s.fused_params,
                    qcomm_codecs_registry=s.qcomm_codecs_registry,
                    use_index_dedup=True,
                )
            )
            replaced = True
        else:
            new_sharders.append(s)
    if not replaced:
        # Dedup configured but no EC sharder to bind it to would mean it is
        # silently inert — the same "configured but not applied" bug class the
        # qcomm path guards against. Fail loudly on every rank instead.
        raise RuntimeError(
            "DLRMV4 EC index dedup configured (use_index_dedup=True) but no "
            "EmbeddingCollectionSharder was found to enable it on; refusing to "
            "run with dedup silently disabled"
        )
    if rank0:
        logger.info("DLRMV4 EC index dedup ENABLED (use_index_dedup=True)")
    return new_sharders


def _embedding_table_names(
    model: torch.nn.Module,
    embedding_table_configs: Optional[Dict[str, EmbeddingConfig]],
) -> List[str]:
    """All embedding table names the planner will place.

    Prefers the authoritative `embedding_table_configs` (keys == table names ==
    planner parameter names). Falls back to walking the model's EBC/EC modules
    when the configs are not passed in.
    """
    if embedding_table_configs:
        return list(embedding_table_configs.keys())
    names: List[str] = []
    for _, module in model.named_modules():
        if type(module) in TORCHREC_TYPES:
            if isinstance(module, EmbeddingBagCollection):
                names.extend(c.name for c in module.embedding_bag_configs())
            elif isinstance(module, EmbeddingCollection):
                names.extend(c.name for c in module.embedding_configs())
    return names


def _build_placement_constraints(
    model: torch.nn.Module,
    embedding_placement: str,
    embedding_placement_overrides: Dict[str, str],
    embedding_table_configs: Optional[Dict[str, EmbeddingConfig]],
    embedding_sharding_overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, ParameterConstraints]:
    """Translate gin/env placement + sharding strings into ParameterConstraints.

    Two orthogonal per-table knobs are merged into one constraint per table:

    * Placement (compute kernel / memory tier):
      ``embedding_placement_overrides.get(name, embedding_placement)``.
      ``auto``/empty -> no compute-kernel constraint (planner decides from HBM).
    * Sharding type (shard layout): ``embedding_sharding_overrides.get(name)``.
      ``auto``/empty (or absent) -> no sharding-type constraint (planner decides,
      which is ROW_WISE for the large yambda tables). Use e.g. ``column_wise``
      to move a hot, data-skewed table off ROW_WISE so its embedding all-to-all
      is balanced by rank instead of routed by (hot) row.

    A table is added to the returned dict only if at least one knob is set for it
    (so with everything ``auto`` we return {} and the plan is byte-identical to
    the legacy path). Unknown values raise ValueError.
    """
    embedding_sharding_overrides = embedding_sharding_overrides or {}
    valid_place = set(_PLACEMENT_TO_KERNEL) | {"auto", ""}
    for where, val in [
        ("embedding_placement", embedding_placement),
        *[
            (f"embedding_placement_overrides[{k}]", v)
            for k, v in embedding_placement_overrides.items()
        ],
    ]:
        if val not in valid_place:
            raise ValueError(
                f"Invalid embedding placement {val!r} for {where}; "
                f"expected one of {sorted(valid_place - {''})}."
            )
    valid_shard = set(_SHARDING_TO_TYPE) | {"auto", ""}
    for k, v in embedding_sharding_overrides.items():
        if v not in valid_shard:
            raise ValueError(
                f"Invalid embedding sharding {v!r} for "
                f"embedding_sharding_overrides[{k}]; "
                f"expected one of {sorted(valid_shard - {''})}."
            )

    names = _embedding_table_names(model, embedding_table_configs)
    unknown = (
        set(embedding_placement_overrides) | set(embedding_sharding_overrides)
    ) - set(names)
    if unknown:
        logger.warning(
            "[emb-placement] override(s) for unknown table(s) %s ignored; "
            "known tables: %s",
            sorted(unknown),
            sorted(names),
        )

    constraints: Dict[str, ParameterConstraints] = {}
    resolved_place: Dict[str, str] = {}
    resolved_shard: Dict[str, str] = {}
    for name in names:
        placement = embedding_placement_overrides.get(name, embedding_placement)
        sharding = embedding_sharding_overrides.get(name, "auto")
        resolved_place[name] = placement or "auto"
        resolved_shard[name] = sharding or "auto"
        kernel = _PLACEMENT_TO_KERNEL.get(placement)
        stype = _SHARDING_TO_TYPE.get(sharding)
        kwargs: Dict[str, Any] = {}
        if kernel is not None:
            kwargs["compute_kernels"] = [kernel.value]
        if stype is not None:
            kwargs["sharding_types"] = [stype.value]
        if kwargs:
            constraints[name] = ParameterConstraints(**kwargs)

    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        logger.info(
            "[emb-placement] placement(global=%r overrides=%s) sharding(overrides=%s) "
            "-> resolved_placement=%s resolved_sharding=%s "
            "(constrained=%d/%d tables; the rest are planner-auto)",
            embedding_placement,
            embedding_placement_overrides or {},
            embedding_sharding_overrides or {},
            resolved_place,
            resolved_shard,
            len(constraints),
            len(names),
        )
    return constraints


@gin.configurable
def make_optimizer_and_shard(
    model: torch.nn.Module,
    device: torch.device,
    world_size: int,
    local_world_size: Optional[int] = None,
    hbm_cap_gb: int = 260,
    sparse_a2a_forward_precision: str = "fp32",
    sparse_a2a_backward_precision: str = "fp32",
    qcomm_lowmem_clamp_cast: bool = True,
    ec_index_dedup: bool = False,
    embedding_placement: str = "auto",
    embedding_placement_overrides: Optional[Dict[str, str]] = None,
    embedding_sharding_overrides: Optional[Dict[str, str]] = None,
    embedding_table_configs: Optional[Dict[str, EmbeddingConfig]] = None,
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
    sharders = _maybe_apply_qcomm_a2a(
        sharders,
        device,
        forward_precision=sparse_a2a_forward_precision,
        backward_precision=sparse_a2a_backward_precision,
        lowmem_clamp_cast=qcomm_lowmem_clamp_cast,
    )
    # Final pass: enable in-batch index dedup on the EC sharder regardless of
    # whether the qcomm branch above rebuilt it (fp16 default) or returned the
    # bare get_default_sharders() list (fp32). Preserves the qcomm registry.
    sharders = _maybe_apply_ec_index_dedup(sharders, enabled=ec_index_dedup)
    # local_world_size = GPUs per node so the planner respects the intra-node
    # (xGMI/NVLink) vs inter-node hierarchy when placing shards. Defaults to
    # world_size for the single-node case (no behavior change).
    logger.info(
        "[hbm-cap] make_optimizer_and_shard: hbm_cap_gb=%s (planner Topology hbm_cap=%d bytes), "
        "world_size=%s local_world_size=%s",
        hbm_cap_gb,
        hbm_cap_gb * 1024 * 1024 * 1024,
        world_size,
        local_world_size or world_size,
    )
    # Resolve per-table embedding placement (gin/env-driven). Global default
    # `embedding_placement` applies to every table; `embedding_placement_overrides`
    # (table name -> placement) wins per table. Tables resolving to "auto" carry
    # no constraint (planner decides from hbm_cap). When nothing is constrained we
    # pass constraints=None so the plan is byte-identical to the legacy path.
    constraints = _build_placement_constraints(
        model=model,
        embedding_placement=embedding_placement,
        embedding_placement_overrides=embedding_placement_overrides or {},
        embedding_sharding_overrides=embedding_sharding_overrides or {},
        embedding_table_configs=embedding_table_configs,
    )
    planner = EmbeddingShardingPlanner(
        topology=Topology(
            local_world_size=local_world_size or world_size,
            world_size=world_size,
            compute_device="cuda",
            hbm_cap=hbm_cap_gb * 1024 * 1024 * 1024,
            ddr_cap=0,
        ),
        constraints=constraints or None,
    )
    pg = dist.GroupMember.WORLD
    env = ShardingEnv.from_process_group(pg)  # pyre-ignore [6]
    pg = env.process_group

    plan = planner.collective_plan(model, sharders, pg)

    # Authoritative placement log: report the compute kernel the planner ACTUALLY
    # assigned to each table (vs the [emb-placement] line above, which reports what
    # was requested). "fused" = HBM; "fused_uvm"/"fused_uvm_caching" = UVM-backed.
    # Rank 0 only, best-effort (never break the build over a logging shape change).
    if (dist.get_rank() if dist.is_initialized() else 0) == 0:
        try:
            for module_path, param_plans in plan.plan.items():
                for param_name, ps in param_plans.items():
                    logger.info(
                        "[emb-placement] plan: %s.%s -> compute_kernel=%s "
                        "sharding_type=%s",
                        module_path,
                        param_name,
                        getattr(ps, "compute_kernel", "?"),
                        getattr(ps, "sharding_type", "?"),
                    )
        except Exception as e:  # logging only; must never fail the build
            logger.warning("[emb-placement] could not dump plan kernels: %s", e)

    # Re-seed right before DMP materializes/inits the sharded embedding tables.
    # The per-table seeded init_fn (configs.get_embedding_table_config) handles
    # the eager path, but the fused FBGEMM TBE path inits weights on-device and
    # may bypass init_fn, drawing from the global RNG instead. Re-seeding here
    # (same value on every rank) makes embedding init reproducible run-to-run for
    # a fixed sharding plan (Tier 1). Dense params are already initialized in
    # make_model, so this does not perturb them.
    _emb_seed = int(os.environ.get("SEED", "1"))
    torch.manual_seed(_emb_seed)
    torch.cuda.manual_seed_all(_emb_seed)
    logger.info(f"[emb-init] re-seeded RNGs before DMP with SEED={_emb_seed}")

    # Shard model
    model = DistributedModelParallel(
        module=model,
        device=device,
        plan=plan,
        sharders=sharders,
    )

    # --- startup init checksum (reproducibility probe) -------------------------
    # Right after DMP materializes real weights, log a deterministic fingerprint
    # of every parameter so two builds with the same $SEED + sharding plan can be
    # diffed for byte-level init reproducibility. For sharded embeddings we
    # all-reduce the per-shard (count, sum, sumsq) so the fingerprint covers the
    # WHOLE table independent of how rows split across ranks; replicated dense
    # params use rank 0's local copy.
    # OFF BY DEFAULT: the fp64 reductions below (.sum(dtype=float64) /
    # vector_norm(dtype=float64)) materialize a full fp64 copy of each local
    # embedding shard (~2x the fp32 shard, i.e. >150 GiB for the big tables),
    # which leaves almost no HBM headroom after sharding and will OOM the build on
    # any node with residual memory. Only enable for explicit reproducibility
    # checks, ideally with a smaller batch / on a clean node. Enable with
    # INIT_CHECKSUM=1.
    if os.environ.get("INIT_CHECKSUM", "0") == "1":
        import hashlib

        _rank = dist.get_rank() if dist.is_initialized() else 0
        _fps: List[str] = []
        for _name, _p in sorted(model.named_parameters(), key=lambda kv: kv[0]):
            _sharded = isinstance(_p, ShardedTensor)
            if _sharded:
                _shards = _p.local_shards()
                _loc = _shards[0].tensor if _shards else None
            else:
                _loc = _p
            if _loc is None or _loc.numel() == 0:
                _cnt, _sm, _sq = 0.0, 0.0, 0.0
            else:
                _det = _loc.detach()
                _cnt = float(_det.numel())
                _sm = _det.sum(dtype=torch.float64).item()
                _nrm = torch.linalg.vector_norm(
                    _det, ord=2, dtype=torch.float64
                ).item()
                _sq = _nrm * _nrm
            if _sharded and dist.is_initialized():
                _stat = torch.tensor(
                    [_cnt, _sm, _sq], dtype=torch.float64, device=device
                )
                dist.all_reduce(_stat, op=dist.ReduceOp.SUM)
                _cnt, _sm, _sq = _stat.tolist()
            _fps.append(f"{_name}|{int(_cnt)}|{_sm:.6f}|{_sq:.6f}")
            if _rank == 0:
                logger.info(
                    f"[init-checksum] {'sharded' if _sharded else 'dense'} "
                    f"{_name} n={int(_cnt)} sum={_sm:.6f} sumsq={_sq:.6f}"
                )
        if _rank == 0:
            _digest = hashlib.sha256("\n".join(_fps).encode()).hexdigest()[:16]
            logger.info(
                f"[init-checksum] SEED={os.environ.get('SEED', '?')} "
                f"params={len(_fps)} digest={_digest}"
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


# THROWAWAY DIAG state: per-embedding-table lookup stats accumulated across a
# fixed window of steps (DIAG_EMB_STEPS, default 100) so the reported numbers are
# averaged/aggregated rather than a single noisy batch. Rank-0 only.
_EMB_DIAG_ACC: Dict[str, Dict[str, Any]] = {}
_EMB_DIAG_NBATCH: int = 0
# Cap (per-batch lookups) below which we also track a TRUE global-unique set
# across the whole window (cheap for contextual/cross tables, total == batch
# size). Sequential tables (item/artist/album) blow past this and only get the
# per-batch averages.
_EMB_DIAG_GLOBAL_CAP: int = 1 << 17  # 131072


def _log_unique_embedding_diag(
    sample, rank: int, step: int, max_steps: int = 100, log_every: int = 50
) -> None:
    """Diagnostic: aggregate per-embedding-table lookup stats over a step window.

    Quantifies the user-major batching concern — when consecutive sliding-window
    anchors come from the same few users, a batch reads very few UNIQUE embedding
    rows (low unique/total), so embedding lookups are highly redundant. Covers the
    base id tables AND the cross-feature tables (user_x_* / *_x_hour hashed
    combos); the user_x_* tables should be the most redundant under shuffle OFF.

    Accumulates over ``max_steps`` batches and emits an aggregate summary (mean
    per-batch unique%/hot%/top10%, plus a true global-unique over the whole
    window for the small contextual/cross tables). Rank-0 only, non-fatal.
    """
    if rank != 0:
        return
    global _EMB_DIAG_NBATCH
    try:
        from generative_recommenders.dlrm_v3.configs import YAMBDA_5B_CROSS_SPECS

        cross_caps = {name: n for (name, _k, n, _s) in YAMBDA_5B_CROSS_SPECS}

        def _table_of(key: str):
            # cross tables match by exact name; resolve BEFORE substring fallbacks
            # so e.g. 'user_x_artist' isn't misread as the artist_id table.
            if key in cross_caps:
                return key, cross_caps[key]
            if key == "uid" or key.endswith("_uid") or key.endswith(".uid"):
                return "uid", 0
            if "artist" in key:
                return "artist_id", 0
            if "album" in key:
                return "album_id", 0
            if "item" in key and key.endswith("id"):
                return "item_id", 0
            return None, 0

        for tag, kjt in (
            ("uih", sample.uih_features_kjt),
            ("cand", sample.candidates_features_kjt),
        ):
            for key in kjt.keys():
                table, cap = _table_of(key)
                if table is None:
                    continue
                vals = kjt[key].values()
                total = int(vals.numel())
                if total == 0:
                    continue
                u, counts = torch.unique(vals, return_counts=True)
                uniq = int(u.numel())
                hot1 = int(counts.max().item())
                k = min(10, uniq)
                topk = int(torch.topk(counts, k).values.sum().item())

                slot = _EMB_DIAG_ACC.setdefault(
                    f"{tag}.{key}",
                    {
                        "table": table,
                        "cap": cap,
                        "n": 0,
                        "tot": 0,
                        "uniq": 0,
                        "upct": 0.0,
                        "upct_min": 100.0,
                        "upct_max": 0.0,
                        "hot1pct": 0.0,
                        "topkpct": 0.0,
                        "glob": None,  # running global-unique id tensor (small tables)
                    },
                )
                upct = 100.0 * uniq / total
                slot["n"] += 1
                slot["tot"] += total
                slot["uniq"] += uniq
                slot["upct"] += upct
                slot["upct_min"] = min(slot["upct_min"], upct)
                slot["upct_max"] = max(slot["upct_max"], upct)
                slot["hot1pct"] += 100.0 * hot1 / total
                slot["topkpct"] += 100.0 * topk / total
                if total <= _EMB_DIAG_GLOBAL_CAP:
                    prev = slot["glob"]
                    merged = u if prev is None else torch.cat([prev, u])
                    slot["glob"] = torch.unique(merged)

        _EMB_DIAG_NBATCH += 1
        n = _EMB_DIAG_NBATCH
        if n % log_every == 0 or n >= max_steps:
            lines = [f"emb-diag AGGREGATE over {n} batches (step<= {step}):"]
            for name in sorted(_EMB_DIAG_ACC):
                s = _EMB_DIAG_ACC[name]
                c = max(1, s["n"])
                cap_s = f" cap={s['cap']/1e6:.0f}M" if s["cap"] else ""
                glob_s = ""
                if s["glob"] is not None:
                    g = int(s["glob"].numel())
                    glob_s = (
                        f" | global_uniq={g} over {s['tot']} seen "
                        f"({s['tot']/max(1,g):.1f}x reuse)"
                    )
                lines.append(
                    f"  {name}[{s['table']}]{cap_s}: "
                    f"avg_tot={s['tot']/c:.0f} "
                    f"avg_uniq%={s['upct']/c:.1f} "
                    f"(min={s['upct_min']:.1f} max={s['upct_max']:.1f}) "
                    f"avg_hot1%={s['hot1pct']/c:.1f} "
                    f"avg_top10%={s['topkpct']/c:.1f}"
                    f"{glob_s}"
                )
            logger.info("\n".join(lines))
    except Exception as e:  # diagnostic must never break training
        logger.warning(f"emb-diag failed: {e}")


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
    streaming_diag_unique_emb: bool = False,
    # lr_scheduler: to-do: Add a scheduler
) -> None:
    model.train()
    batch_idx: int = start_batch_idx
    profiler = Profiler(rank) if output_trace else None

    for epoch in range(num_epochs):
        dataloader.sampler.set_epoch(epoch)  # pyre-ignore [16]
        for sample in dataloader:
            if streaming_diag_unique_emb and batch_idx < int(
                os.environ.get("DIAG_EMB_STEPS", "100")
            ):
                _log_unique_embedding_diag(
                    sample,
                    rank,
                    batch_idx,
                    max_steps=int(os.environ.get("DIAG_EMB_STEPS", "100")),
                    log_every=metric_log_frequency,
                )
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
    # Exclude eval wall-time from the train step-time window (see _run_eval_window).
    metric_logger.pause_perf("eval")
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
    metric_logger.resume_perf("eval")


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
    # MLPerf `train_loss` event cadence, in global train steps, INDEPENDENT of
    # metric_log_frequency (the console/TB cadence). 0 = fall back to
    # metric_log_frequency (preserves prior coupled behavior). Wired to
    # $MLPERF_TRAIN_LOSS_LOG_FREQ via gin.
    mlperf_train_loss_log_frequency: int = 0,
    checkpoint_frequency: int = 100,
    start_ts: int = 0,
    persistent_loader: bool = False,
    eval_every_n_windows: int = 1,
    # Data-fraction eval cadence (mutually exclusive with eval_every_n_windows).
    # >0 = eval every this FRACTION of the run's total training data, converted
    # once to a global train-step interval. 0.0 = OFF (use the per-window
    # cadence). Wired to $EVAL_EVERY_DATA_PCT via gin.
    eval_every_data_pct: float = 0.0,
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
    # --- gradient clipping (streaming path, dense params). 0.0 = OFF, which
    #     preserves legacy streaming behavior. Wired to $GRAD_CLIP_NORM via gin. ---
    grad_clip_norm: float = 0.0,
    # --- LR warmup (streaming path). Linearly ramp LR 0 -> base over the first
    #     N GLOBAL train steps, then hold. Scales BOTH the dense optimizer and
    #     the in-backward fused sparse-embedding optimizer. 0 = OFF (byte-
    #     identical). Wired to $LR_WARMUP_STEPS via gin. ---
    lr_warmup_steps: int = 0,
    # --- LR warmup floor (streaming path). Absolute LR the ramp STARTS from at
    #     gstep 0 (per group, clamped to <= that group's base LR). The ramp then
    #     goes start -> base over `lr_warmup_steps`. 0.0 (default) reproduces the
    #     0 -> base ramp exactly (byte-identical). A small nonzero floor turns the
    #     warmup into a "partial warmup" that reclaims the near-dead head of the
    #     ramp (trading a little seed-divergence damping for faster time-to-target).
    #     Only has any effect when lr_warmup_steps > 0. Wired to
    #     $LR_WARMUP_START_LR via gin. ---
    lr_warmup_start_lr: float = 0.0,
    # --- diagnostic: log per-batch unique/total embedding-id counts ---
    streaming_diag_unique_emb: bool = False,
    # --- test-only failure injection knob ---
    die_at_step: int = -1,
    # MLPerf logger (rank-0-gated); None disables all MLPerf event emission.
    mlperf_logger: Optional[Any] = None,
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
    # Exactly one eval cadence may be active. eval_every_n_windows defaults to 1
    # (eval every window), so enabling the data-fraction cadence REQUIRES
    # explicitly disabling the per-window one (EVAL_EVERY_N_WINDOWS=0). Fail fast
    # on a contradictory config rather than silently picking one.
    if (eval_every_data_pct and eval_every_data_pct > 0) and eval_every_n_windows > 0:
        raise ValueError(
            "Conflicting eval cadences: eval_every_data_pct="
            f"{eval_every_data_pct} (>0) AND eval_every_n_windows="
            f"{eval_every_n_windows} (>0). They are mutually exclusive. To use "
            "the data-fraction cadence set EVAL_EVERY_N_WINDOWS=0; to use the "
            "per-window cadence set EVAL_EVERY_DATA_PCT=0."
        )
    # MLPerf train_loss cadence: independent of metric_log_frequency. 0 (the
    # env-binding default) falls back to metric_log_frequency so unset behavior
    # matches the prior coupled implementation.
    mlperf_loss_every = (
        mlperf_train_loss_log_frequency
        if mlperf_train_loss_log_frequency and mlperf_train_loss_log_frequency > 0
        else metric_log_frequency
    )
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

    # Data-fraction eval cadence: convert eval_every_data_pct into a global
    # train-step interval ONCE, over the ORIGINAL requested window range
    # [eval_anchor_ts, requested_end_ts). Keying the later trigger off
    # `global_step % eval_interval_steps` (global_step is monotonic and
    # checkpoint-restored) makes the eval grid identical on cold start and on
    # every resume, exactly like checkpoint_step_frequency. 0 => disabled.
    eval_interval_steps = 0
    if eval_every_data_pct and eval_every_data_pct > 0:
        # Per-rank batch size: the persistent loader carries it directly; the
        # per-window path uses the same gin %batch_size (env BATCH_SIZE,
        # default 1024 — matches make_streaming_dataloader.batch_size).
        bs = (
            persistent_dl.batch_size
            if persistent_dl is not None
            else int(os.environ.get("BATCH_SIZE", "1024"))
        )
        if hasattr(dataset.dataset, "total_train_anchors"):
            # total_train_anchors does a full-range gather over the mmap'd uid
            # array for ~billions of positions + a uid hash. It is slow
            # (minutes, single-threaded) AND, run on every rank independently,
            # a large per-rank skew source: a fast rank finishes and races into
            # the first embedding all-to-all while slow ranks are still hashing,
            # desyncing the NCCL collective stream and hanging the job. The
            # result is a pure function of the (identical) dataset + split, so
            # compute it ONCE on rank 0 and broadcast the scalar; ranks 1..N
            # skip the gather entirely (no 8x mmap/CPU contention, no skew).
            if world_size > 1 and torch.distributed.is_initialized():
                _tta = (
                    dataset.dataset.total_train_anchors(  # pyre-ignore[16]
                        eval_anchor_ts, requested_end_ts - eval_anchor_ts
                    )
                    if rank == 0
                    else 0
                )
                _tta_t = torch.tensor([_tta], dtype=torch.int64, device=device)
                torch.distributed.broadcast(_tta_t, src=0)
                total_train_anchors = int(_tta_t.item())
            else:
                total_train_anchors = dataset.dataset.total_train_anchors(  # pyre-ignore[16]
                    eval_anchor_ts, requested_end_ts - eval_anchor_ts
                )
            total_train_steps = total_train_anchors // max(1, bs * world_size)
            eval_interval_steps = max(
                1, round(eval_every_data_pct * total_train_steps)
            )
            if rank == 0:
                logger.info(
                    "[data-pct-eval] eval_every_data_pct=%.6g -> "
                    "eval_interval_steps=%d (total_train_anchors=%d bs=%d "
                    "world_size=%d total_train_steps=%d over windows [%d, %d))",
                    eval_every_data_pct,
                    eval_interval_steps,
                    total_train_anchors,
                    bs,
                    world_size,
                    total_train_steps,
                    eval_anchor_ts,
                    requested_end_ts,
                )
        elif rank == 0:
            logger.warning(
                "[data-pct-eval] dataset %s has no total_train_anchors(); "
                "data-fraction eval is DISABLED (no per-window eval either, "
                "since EVAL_EVERY_N_WINDOWS must be 0 to reach here) — only the "
                "final eval will run.",
                type(dataset.dataset).__name__,
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

    if rank == 0:
        logger.info(
            "[grad-clip] streaming path gradient clipping %s (max_norm=%.4g via $GRAD_CLIP_NORM)",
            "ENABLED" if (grad_clip_norm and grad_clip_norm > 0) else "OFF",
            grad_clip_norm,
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

    # --- LR warmup (streaming path) ------------------------------------------
    # Linearly ramp LR from 0 -> base over the first `lr_warmup_steps` GLOBAL
    # train steps, then hold at base. Keyed off the monotonic, checkpoint-
    # restored global_step so the ramp is identical on cold start and on every
    # resume (a run that crashes mid-warmup resumes at the correct multiplier;
    # a run already past warmup is a no-op). Scales BOTH LR channels: the
    # dense/non-fused optimizer passed in as `optimizer`, AND the in-backward
    # fused sparse-embedding optimizer (model.fused_optimizer), whose per-table
    # LR dominates this model. `lr_warmup_steps=0` (default) => OFF and the base
    # LRs are never touched (byte-identical to the pre-warmup path).
    _warmup_groups = list(optimizer.param_groups)
    _warmup_fused_opt = getattr(model, "fused_optimizer", None)
    if _warmup_fused_opt is not None:
        _warmup_groups += list(_warmup_fused_opt.param_groups)
    _warmup_base_lrs = [pg["lr"] for pg in _warmup_groups]
    # Per-group start LR for the ramp: the configured floor, clamped so it can
    # never exceed a group's base (a floor >= base would mean "no ramp" for that
    # group, which we express by starting it AT base). 0.0 => classic 0 -> base.
    _warmup_start_lrs = [min(float(lr_warmup_start_lr), b) for b in _warmup_base_lrs]
    if lr_warmup_steps and lr_warmup_steps > 0 and rank == 0:
        logger.info(
            "[lr-warmup] linear warmup over %d global steps across %d param "
            "group(s); start LRs: %s -> base LRs: %s",
            lr_warmup_steps,
            len(_warmup_groups),
            ", ".join(f"{s:.3e}" for s in _warmup_start_lrs),
            ", ".join(f"{b:.3e}" for b in _warmup_base_lrs),
        )

    def _apply_lr_warmup(gstep: int) -> None:
        # OFF, or ramp already finished (kernel + param_groups already hold the
        # base LR from the last sync at gstep==lr_warmup_steps): nothing to do.
        if not (lr_warmup_steps and lr_warmup_steps > 0) or gstep > lr_warmup_steps:
            return
        progress = min(1.0, float(gstep) / float(lr_warmup_steps))
        # Ramp each group from its (clamped) start LR up to its base LR. With
        # start=0 this collapses to `base * progress` (the original schedule).
        for pg, base, start in zip(
            _warmup_groups, _warmup_base_lrs, _warmup_start_lrs
        ):
            pg["lr"] = start + (base - start) * progress
        # Push the warmed LR into the FBGEMM TBE backward kernel. The in-backward
        # fused embedding optimizer only calls
        # `emb_module.set_learning_rate(param_groups[0]["lr"])` from its
        # step()/zero_grad() (torchrec/distributed/batched_embedding_kernel.py),
        # and this loop steps the dense `optimizer` — NOT model.fused_optimizer —
        # so WITHOUT this explicit call the embedding tables would keep training
        # at the construction-time base LR and warmup would silently apply to the
        # dense params only. `.step()` on the fused (in-backward) optimizer does
        # no parameter update; it only re-syncs the LR to the kernel.
        if _warmup_fused_opt is not None:
            _warmup_fused_opt.step()
        if rank == 0 and gstep % max(1, metric_log_frequency) == 0:
            _fused_lr = ""
            if _warmup_fused_opt is not None:
                _fused_lr = (
                    f" fused_pg_lr={_warmup_fused_opt.param_groups[0]['lr']:.3e}"
                )
                # Best-effort readback of the ACTUAL kernel LR to confirm
                # set_learning_rate() propagated. model.fused_optimizer is a
                # (possibly nested) CombinedOptimizer whose leaves carry the TBE
                # module as `_emb_module`, so recurse to the first leaf. Purely
                # diagnostic; wrapped so it can never break training.
                def _first_tbe_kernel_lr(opt, depth=0):
                    if depth > 6 or opt is None:
                        return None
                    _em = getattr(opt, "_emb_module", None)
                    if _em is not None and hasattr(_em, "get_learning_rate"):
                        return float(_em.get_learning_rate())
                    for _sub in getattr(opt, "_optims", []):
                        _child = _sub[1] if isinstance(_sub, tuple) else _sub
                        _lr = _first_tbe_kernel_lr(_child, depth + 1)
                        if _lr is not None:
                            return _lr
                    _wrapped = getattr(opt, "_optimizer", None)
                    if _wrapped is not None:
                        return _first_tbe_kernel_lr(_wrapped, depth + 1)
                    return None

                try:
                    _klr = _first_tbe_kernel_lr(_warmup_fused_opt)
                    if _klr is not None:
                        _fused_lr += f" tbe_kernel_lr={_klr:.3e}"
                except Exception:
                    pass
            logger.info(
                "[lr-warmup] gstep=%d progress=%.4f dense_lr=%.3e%s",
                gstep,
                progress,
                _warmup_groups[0]["lr"],
                _fused_lr,
            )

    def _run_train_window(
        train_data_iterator,
        train_ts: int,
        start_batch_idx: int = 0,
        label: Optional[str] = None,
        do_eval: Optional[Callable[[int, int], None]] = None,
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
            if streaming_diag_unique_emb and train_batch_idx < int(
                os.environ.get("DIAG_EMB_STEPS", "100")
            ):
                _log_unique_embedding_diag(
                    sample,
                    rank,
                    train_batch_idx,
                    max_steps=int(os.environ.get("DIAG_EMB_STEPS", "100")),
                    log_every=metric_log_frequency,
                )
            # LR warmup: set the scaled LR on BOTH channels (dense param_groups
            # + the fused embedding TBE via set_learning_rate) BEFORE the
            # forward/backward, so the in-backward embedding update THIS step
            # uses the warmed LR (the dense optimizer reads it at .step() below).
            # No-op when lr_warmup_steps=0.
            _apply_lr_warmup(metric_logger.global_step["train"])
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
            # Gradient clipping for the streaming path. Clips dense params (the
            # sparse embedding tables use a fused optimizer and are unaffected,
            # same as the non-streaming path's clip_grad_norm_). OFF by default
            # (grad_clip_norm=0.0 via $GRAD_CLIP_NORM) so legacy streaming runs
            # are byte-for-byte unchanged; set >0 to enable.
            if grad_clip_norm and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=grad_clip_norm
                )
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
            # MLPerf train_loss event on its own cadence (decoupled from the
            # console/TB metric cadence below). Called every step; the cross-rank
            # all-reduce only fires on the cadence, gated by the rank-identical
            # global_step inside the method, so it stays in lockstep.
            metric_logger.maybe_log_mlperf_train_loss(aux_losses, every=mlperf_loss_every)
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
            # Data-fraction eval cadence: run the full-holdout eval whenever the
            # monotonic global step crosses a multiple of eval_interval_steps
            # (i.e. every eval_every_data_pct of the training data). Keyed off
            # global_step (checkpoint-restored) so the eval grid is identical
            # across resume. Mid-window-safe: eval sets model.eval(), so restore
            # train mode + dataset.is_eval afterward. do_eval is None unless the
            # data-pct cadence is enabled.
            if (
                do_eval is not None
                and eval_interval_steps > 0
                and gstep > 0
                and gstep % eval_interval_steps == 0
            ):
                if rank == 0:
                    logger.info(
                        "[data-pct-eval] trigger eval train_ts=%d global_step=%d "
                        "(interval=%d)",
                        train_ts,
                        gstep,
                        eval_interval_steps,
                    )
                do_eval(train_ts, gstep)
                model.train()
                dataset.dataset.is_eval = False  # pyre-ignore [16]
                # Data-fraction eval may hit the MLPerf target and emit RUN_STOP
                # (via _do_eval_*). Stop the window immediately so we don't train
                # past the convergence point; the outer window loop checks the
                # same flag and breaks too.
                if mlt.run_stopped:
                    break
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

    def _run_eval_window(
        eval_data_iterator, label: Optional[str] = None
    ) -> Dict[str, float]:
        # DO NOT add a checkpoint trigger anywhere inside this function. The eval
        # data iterator's position is not serializable, so a checkpoint taken
        # mid-eval could not be resumed deterministically. `_maybe_checkpoint`
        # only fires after a completed eval window or mid-train-window, so any
        # restored state always sits on a completed-eval boundary -- which is
        # also why the eval reset below is safe across resume.
        #
        # Exclude this eval pass's wall-time from the train step-time window so
        # step_ms stays canonical even when eval coincides with a train interval;
        # the duration is reported separately (window_eval_time_ms + total_eval
        # below). Resumed unconditionally at the end of this function.
        metric_logger.pause_perf("eval")
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
            _eval_metrics = metric_logger.compute(mode="eval")
            for k, v in _eval_metrics.items():
                print(f"{k}: {v}")
        if label and rank == 0 and _t_enter is not None:
            _eval_total = time.perf_counter() - _t_enter
            _fw = (first_wait * 1000) if first_wait is not None else float("nan")
            logger.info(
                f"[boundary] {label} eval first-batch data-wait={_fw:.1f}ms "
                f"total_eval={_eval_total * 1000:.1f}ms batches={eval_batch_idx}"
            )
            # Dedicated per-eval metrics sink. One JSON line per eval boundary
            # capturing the END-OF-PASS metric over the FIXED holdout -- the single
            # correct value for that eval point (no interim/averaging ambiguity).
            # Rank 0 only; append-only so it survives restarts and the trajectory
            # accumulates across resumes. Written next to the main run log
            # ("<LOG>.metrics.jsonl"), falling back to cwd if LOG is unset.
            import json
            import re as _re

            _log = os.environ.get("LOG")
            if _log:
                _base = _log[:-4] if _log.endswith(".log") else _log
                _metrics_path = f"{_base}.metrics.jsonl"
            else:
                _metrics_path = "streaming_eval_metrics.jsonl"
            _ts_m = _re.search(r"train_ts=(\d+)", label)
            _rec = {
                "label": label,
                "train_ts": int(_ts_m.group(1)) if _ts_m else None,
                "global_step": int(metric_logger.global_step.get("train", -1)),
                "eval_batches": eval_batch_idx,
                "total_eval_ms": round(_eval_total * 1000, 1),
                "wall_time": time.time(),
            }
            for _k, _v in _eval_metrics.items():
                try:
                    _rec[_k] = float(_v)
                except (TypeError, ValueError):
                    pass
            try:
                with open(_metrics_path, "a") as _f:
                    _f.write(json.dumps(_rec) + "\n")
            except OSError as _e:
                logger.warning("failed to write metrics sink %s: %s", _metrics_path, _e)
        metric_logger.resume_perf("eval")
        # Return metrics (on every rank) so the MLPerf eval hooks can consume them.
        return _eval_metrics

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

        Single cadence knob `eval_every_n_windows`:
          * <=0 -> eval disabled entirely (train-only; e.g. perf benchmarking or
            the resume test). The eval dataloader is not even built.
          * 1 (default) -> eval after every window.
          * K>1 -> eval when the ABSOLUTE window ts is on the grid anchored at
            `eval_anchor_ts` (the original start_ts), i.e. ts in {anchor,
            anchor+K, anchor+2K, ...}, and ALWAYS on the final window so the
            trajectory ends with an eval point. Anchoring to the absolute ts
            (not the per-call loop index `i`) keeps the eval grid (e.g.
            150,160,170,...) stable across a mid-run resume, which rebases
            start_ts/`train_ts_list` to the resume window.
        """
        if eval_every_n_windows <= 0:
            return False
        if eval_every_n_windows == 1:
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

    # --- MLPerf run tracking --------------------------------------------------
    # total_train_samples = epoch_num denominator (global trainable samples over
    # the window range), computed once and logged as TRAIN_SAMPLES.
    total_train_samples = 0
    if mlperf_logger is not None:
        _idx_fn = getattr(
            dataset.dataset, "train_window_indices", None
        ) or getattr(dataset.dataset, "window_indices", None)
        if _idx_fn is not None:
            for _ts in train_ts_list:
                total_train_samples += int(_idx_fn(_ts).size)
        if rank == 0:
            logger.info(
                "MLPerf: total_train_samples=%d over %d windows",
                total_train_samples,
                n_train,
            )
        # Wire the logger + LR getter so MetricsLogger.compute emits train_loss.
        metric_logger.mlperf_logger = mlperf_logger

        def _current_lr() -> float:
            return float(optimizer.param_groups[0]["lr"])

        metric_logger.lr_getter = _current_lr

    # Centralized MLPerf run-boundary state machine: owns block/eval/run markers,
    # SAMPLES_COUNT/EPOCH_NUM progress metadata, and the per-window-AUC vs
    # auc_threshold convergence decision. Every method no-ops when mlperf_logger
    # is None, so the loop below calls them unconditionally.
    from generative_recommenders.dlrm_v3.train.mlperf_logging_utils import (
        MLPerfRunTracker,
    )

    mlt = MLPerfRunTracker(
        logger=mlperf_logger,
        metric_logger=metric_logger,
        total_train_samples=total_train_samples,
        rank=rank,
        device=device,
    )
    mlt.log_dataset_sizes(
        eval_samples=eval_global_indices.size
        if eval_global_indices is not None
        else None
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
        # Build/fork the eval pool when EITHER cadence needs it: the per-window
        # cadence (eval_every_n_windows>0) or the data-fraction cadence
        # (eval_interval_steps>0). Both are never simultaneously on (validated
        # at entry), so this is "eval is enabled at all".
        if (eval_every_n_windows > 0 or eval_interval_steps > 0) and len(
            train_ts_list
        ) > 0:
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

        # Data-fraction eval callback (double-buffer path). Fired mid-window by
        # _run_train_window on the global-step cadence. Reuses the already-forked
        # persistent eval pool: iter(eval_dl) here runs on the MAIN thread (a
        # reset, not a fork — the only fork was the up-front iter() above), so it
        # stays safe alongside the background window-prefetch thread.
        def _do_eval_db(train_ts: int, gstep: int) -> None:
            # Data-fraction eval boundary: this closes the current MLPerf block,
            # runs the holdout eval with full EVAL_START/EVAL_STOP + EVAL_ACCURACY
            # + convergence, then opens the next block. The block thus brackets
            # exactly one eval_interval_steps of training (MLPerf block == work
            # between two evals), instead of one timestamp window.
            dataset.dataset.is_eval = True  # pyre-ignore [16]
            assert eval_dl is not None
            mlt.block_stop()
            mlt.eval_start()
            eval_metrics = _run_eval_window(
                iter(eval_dl),
                label=f"eval_holdout@train_ts={train_ts}@step={gstep}",
            )
            # Emits RUN_STOP (sets mlt.run_stopped) if the target is met;
            # _run_train_window / the window loop break on that flag.
            mlt.eval_stop(eval_metrics)
            if not mlt.run_stopped:
                mlt.block_start()

        _db_do_eval = _do_eval_db if eval_interval_steps > 0 else None

        # Block placement depends on the eval cadence. Per-window cadence
        # (eval_every_n_windows>0): one block per timestamp window. Otherwise
        # (data-fraction cadence, or no eval): a single block spans the whole
        # run, split at each data-fraction eval boundary by _do_eval_db. Open
        # the first block here for the latter; the boundary helper + the
        # post-loop stop handle the rest.
        _per_window_blocks = eval_every_n_windows > 0
        if not _per_window_blocks:
            mlt.block_start()

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
            # Rendezvous all ranks at the window boundary BEFORE the first
            # forward of this window. The prefetcher has already handed back a
            # ready iterator (this window's window_indices mmap scan is done),
            # but that O(N) scan over the ~18 GB anchor_ts array can finish at
            # very different times across ranks. Without this barrier a fast
            # rank issues the first embedding all-to-all while a slow rank is
            # still in prep, desyncing the NCCL collective stream and hanging
            # the job (observed at a window boundary via the flight recorder:
            # ranks split across consecutive collective seq ids). This only
            # absorbs prep skew (one near-zero sync per window); it does not
            # serialize the background prefetch of future windows.
            _window_boundary_barrier(device, world_size, train_ts)
            if _per_window_blocks:
                mlt.block_start()
            _run_train_window(
                train_data_iterator,
                train_ts=train_ts,
                start_batch_idx=start_batch,
                label=f"train_ts={train_ts}",
                do_eval=_db_do_eval,
            )
            if _per_window_blocks:
                mlt.block_stop()
            should_stop = False
            if _should_eval(i):
                dataset.dataset.is_eval = True  # pyre-ignore [16]
                assert eval_sampler is not None and eval_dl is not None
                mlt.eval_start()
                eval_metrics = _run_eval_window(
                    eval_iter, label=f"eval_holdout@train_ts={train_ts}"
                )
                should_stop = mlt.eval_stop(eval_metrics)
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
            # should_stop: per-window convergence. mlt.run_stopped:
            # data-fraction convergence (RUN_STOP fired mid-window by _do_eval_db).
            if should_stop or mlt.run_stopped:
                # MLPerf target reached: RUN_STOP already emitted; stop training.
                break

        # Close the run-spanning block for the data-fraction / no-eval case.
        # Idempotent: a no-op if the last eval boundary already closed it (i.e.
        # convergence stopped the run) or if per-window blocks were used.
        if not _per_window_blocks:
            mlt.block_stop()
    else:
        # Data-fraction eval callback (non-double-buffer path). Builds a fresh
        # eval dataloader per call over the FIXED holdout set (or the legacy
        # next-window eval when the dataset has no holdout support).
        def _do_eval_nb(train_ts: int, gstep: int) -> None:
            # Data-fraction eval boundary (non-double-buffer path). See _do_eval_db:
            # close the current MLPerf block, run the eval with full markers +
            # convergence, then open the next block so a block brackets one
            # eval_interval_steps of training rather than a timestamp window.
            dataset.dataset.is_eval = True  # pyre-ignore [16]
            mlt.block_stop()
            mlt.eval_start()
            if eval_global_indices is not None:
                eval_metrics = _run_eval_window(
                    iter(
                        make_streaming_dataloader(
                            dataset=dataset, indices=eval_global_indices
                        )
                    ),
                    label=f"eval_holdout@train_ts={train_ts}@step={gstep}",
                )
            else:
                eval_metrics = _run_eval_window(
                    iter(make_streaming_dataloader(dataset=dataset, ts=train_ts + 1)),
                    label=f"eval@train_ts={train_ts}@step={gstep}",
                )
            mlt.eval_stop(eval_metrics)
            if not mlt.run_stopped:
                mlt.block_start()

        _nb_do_eval = _do_eval_nb if eval_interval_steps > 0 else None

        # See the double-buffer branch: per-window blocks for the per-window
        # cadence, else a single run-spanning block split at data-fraction eval
        # boundaries by _do_eval_nb.
        _per_window_blocks = eval_every_n_windows > 0
        if not _per_window_blocks:
            mlt.block_start()

        for i, train_ts in enumerate(train_ts_list):
            dataset.dataset.is_eval = False  # pyre-ignore [16]
            skip = first_skip_samples if i == 0 else 0
            start_batch = (
                resume_batch_idx_in_window
                if i == 0 and resume_batch_idx_in_window > 0
                else 0
            )
            # See the double-buffer path: rendezvous all ranks at the window
            # boundary before the first forward so per-rank data-prep skew
            # cannot desync the NCCL collective stream and hang the job.
            _window_boundary_barrier(device, world_size, train_ts)
            if _per_window_blocks:
                mlt.block_start()
            _run_train_window(
                _window_iter(train_ts, skip_samples=skip),
                train_ts=train_ts,
                start_batch_idx=start_batch,
                do_eval=_nb_do_eval,
            )
            if _per_window_blocks:
                mlt.block_stop()
            should_stop = False
            if _should_eval(i):
                dataset.dataset.is_eval = True  # pyre-ignore [16]
                mlt.eval_start()
                if eval_global_indices is not None:
                    eval_metrics = _run_eval_window(
                        iter(
                            make_streaming_dataloader(
                                dataset=dataset, indices=eval_global_indices
                            )
                        ),
                        label=f"eval_holdout@train_ts={train_ts}",
                    )
                else:
                    # Legacy per-window eval (datasets without user holdout).
                    eval_metrics = _run_eval_window(
                        iter(make_streaming_dataloader(dataset=dataset, ts=train_ts + 1))
                    )
                should_stop = mlt.eval_stop(eval_metrics)
            _maybe_checkpoint(train_ts)
            # should_stop: per-window convergence. mlt.run_stopped:
            # data-fraction convergence (RUN_STOP fired mid-window by _do_eval_nb).
            if should_stop or mlt.run_stopped:
                # MLPerf target reached: RUN_STOP already emitted; stop training.
                break

        # Close the run-spanning block for the data-fraction / no-eval case
        # (idempotent; no-op under per-window blocks or after a convergence stop).
        if not _per_window_blocks:
            mlt.block_stop()

    # Final eval over the fixed user-holdout set (legacy final-window eval
    # otherwise). Skipped if the MLPerf target already stopped the run mid-run.
    if not mlt.run_stopped:
        dataset.dataset.is_eval = True  # pyre-ignore [16]
        mlt.eval_start()
        if eval_global_indices is not None:
            final_metrics = _run_eval_window(
                iter(
                    make_streaming_dataloader(
                        dataset=dataset, indices=eval_global_indices
                    )
                ),
                label="eval_holdout@final",
            )
        else:
            final_metrics = _run_eval_window(
                iter(make_streaming_dataloader(dataset=dataset, ts=num_train_ts)),
                label="eval@final",
            )
        mlt.eval_stop(final_metrics)
        if rank == 0:
            for k, v in final_metrics.items():
                print(f"{k}: {v}")
        # End-of-run RUN_STOP: SUCCESS if final metric met target, else ABORTED.
        mlt.finalize(final_metrics)
