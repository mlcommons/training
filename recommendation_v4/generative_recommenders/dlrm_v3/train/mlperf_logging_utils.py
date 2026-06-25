# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
"""MLPerf Training compliance logging for the DLRMv3 streaming-train-eval path.

Rank-0-gated wrapper around ``mlperf_logging.mllog`` so the streaming loop emits
the MLPerf event stream without every call site re-checking rank or the dep.
"""

import logging
import os
from typing import Any, Dict, Optional

import gin
import torch

logger: logging.Logger = logging.getLogger(__name__)

try:
    from mlperf_logging import mllog
    from mlperf_logging.mllog import constants as mllog_constants

    _MLLOG_AVAILABLE = True
except Exception as e:  # pragma: no cover - import-time guard
    mllog = None  # type: ignore[assignment]
    mllog_constants = None  # type: ignore[assignment]
    _MLLOG_AVAILABLE = False
    logger.warning(
        "mlperf_logging not importable (%s); MLPerf logging disabled. "
        "Install via `pip install git+https://github.com/mlcommons/logging.git`.",
        e,
    )


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def _barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


class MLPerfLogger:
    """Rank-0-gated facade over ``mllog``.

    Event methods no-op on non-zero ranks and when mlperf_logging is absent.
    ``sync=True`` barriers before emit so the timestamp reflects the slowest rank
    (required for INIT_STOP/RUN_START/RUN_STOP).
    """

    def __init__(
        self,
        rank: Optional[int] = None,
        log_path: Optional[str] = None,
        default_stack_offset: int = 2,
        benchmark_name: str = "hstu",
        submitter_name: str = "AMD",
        submission_platform: str = "MI355X",
        fresh: bool = True,
    ):
        self.enabled: bool = _MLLOG_AVAILABLE
        # Use the EXPLICIT caller rank: this is built before init_process_group,
        # when dist.get_rank() would return 0 on every rank (all would log).
        self.rank: int = rank if rank is not None else _rank()
        self.benchmark_name: str = benchmark_name
        self.submitter_name: str = submitter_name
        self.submission_platform: str = submission_platform
        self._logger = None
        if not self.enabled:
            return
        # Only rank 0 emits, so only rank 0 needs the file handler.
        if log_path and self.rank == 0:
            log_dir = os.path.dirname(log_path)
            if log_dir:  # guard: os.makedirs("") raises for a bare filename
                os.makedirs(log_dir, exist_ok=True)
            # mllog's FileHandler APPENDS (mode "a"), which is what a resume needs
            # so the single run's event stream accumulates across relaunches into
            # one file. On a genuine cold start, truncate first so a re-used run
            # dir / a previous crashed-cold-start's orphaned stream can't leave a
            # second run_start in the file (the compliance checker requires
            # EXACTLY_ONE). Resume (fresh=False) appends to continue the stream.
            if fresh:
                open(log_path, "w").close()
            mllog.config(filename=log_path, default_stack_offset=default_stack_offset)
        else:
            mllog.config(default_stack_offset=default_stack_offset)
        self._logger = mllog.get_mllogger()

    @property
    def constants(self):  # pyre-ignore[3]
        return mllog_constants

    def event(
        self,
        key: str,
        value: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        sync: bool = False,
    ) -> None:
        if sync:
            _barrier()
        if self.enabled and self.rank == 0:
            self._logger.event(key=key, value=value, metadata=metadata or {})

    def start(
        self,
        key: str,
        value: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        sync: bool = False,
    ) -> None:
        if sync:
            _barrier()
        if self.enabled and self.rank == 0:
            self._logger.start(key=key, value=value, metadata=metadata or {})

    def end(
        self,
        key: str,
        value: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        sync: bool = False,
    ) -> None:
        if sync:
            _barrier()
        if self.enabled and self.rank == 0:
            self._logger.end(key=key, value=value, metadata=metadata or {})

    def submission_info(self) -> None:
        """Emit the five SUBMISSION_* events required for a valid submission."""
        if not (self.enabled and self.rank == 0):
            return
        c = mllog_constants
        self.event(key=c.SUBMISSION_BENCHMARK, value=self.benchmark_name)
        self.event(key=c.SUBMISSION_ORG, value=self.submitter_name)
        self.event(key=c.SUBMISSION_DIVISION, value=c.CLOSED)
        self.event(key=c.SUBMISSION_STATUS, value=c.ONPREM)
        self.event(key=c.SUBMISSION_PLATFORM, value=self.submission_platform)

    def log_run_start(
        self,
        global_batch_size: int,
        seed: int,
        gradient_accumulation_steps: int = 1,
    ) -> None:
        """Emit submission info + core hyperparameters, then INIT_STOP + RUN_START.

        Optimizer names/LRs are read from gin (dense Adam + sparse RowWiseAdagrad),
        resolving env-macro refs to concrete values. Call once on a genuine cold
        start, after the model is built. INIT_STOP/RUN_START barrier so the
        timestamp reflects the slowest rank, so ALL ranks must call this together
        (non-rank-0 / disabled calls no-op the emit but still hit the barrier).
        """
        c = self.constants
        self.submission_info()
        self.event(key=c.GLOBAL_BATCH_SIZE, value=int(global_batch_size))
        self.event(
            key=c.GRADIENT_ACCUMULATION_STEPS, value=int(gradient_accumulation_steps)
        )
        self.event(key=c.SEED, value=int(seed))
        self.event(
            key=c.OPT_NAME,
            value=_gin_param("dense_optimizer_factory_and_class.optimizer_name", "Adam"),
        )
        self.event(
            key=c.OPT_BASE_LR,
            value=_gin_param("dense_optimizer_factory_and_class.learning_rate", None),
        )
        self.event(
            key="opt_sparse_name",
            value=_gin_param(
                "sparse_optimizer_factory_and_class.optimizer_name", "RowWiseAdagrad"
            ),
        )
        self.event(
            key="opt_sparse_base_learning_rate",
            value=_gin_param(
                "sparse_optimizer_factory_and_class.learning_rate", None
            ),
        )
        self.end(key=c.INIT_STOP, sync=True)
        self.start(key=c.RUN_START, sync=True)


def _gin_param(name: str, default: Any) -> Any:
    """Read a gin-bound parameter, resolving env-macro refs to concrete values.

    Returns ``default`` if the parameter is unbound or a macro ref cannot be
    resolved (so env-overridden LRs log as numbers, not unencodable objects).
    """
    try:
        value = gin.query_parameter(name)
    except (ValueError, KeyError):
        return default
    if hasattr(value, "scoped_configurable_fn"):
        try:
            return value.scoped_configurable_fn()
        except Exception:
            return default
    return value


class MLPerfRunTracker:
    """Centralized MLPerf run-boundary state machine for the streaming loop.

    Owns the block/eval/run markers, the SAMPLES_COUNT/EPOCH_NUM progress
    metadata, and the convergence decision (per-window AUC vs the configured
    ``auc_threshold``). Every method no-ops when ``logger`` is None, so the
    streaming loop can call them unconditionally. The convergence metric is
    fixed to per-window AUC (higher-is-better).
    """

    # MetricsLogger.compute key short name for per-window AUC.
    _EVAL_METRIC_SHORT = "window_auc"

    def __init__(
        self,
        logger: Optional[MLPerfLogger],
        metric_logger: Any,
        total_train_samples: int,
        rank: int,
        device: Any,
    ):
        self.logger = logger
        self.metric_logger = metric_logger
        self.total_train_samples = int(total_train_samples)
        self.rank = int(rank)
        self.device = device
        self.run_stopped: bool = False
        # Idempotency flag so the boundary helpers and the outer loop can both
        # call start/stop without risking a double BLOCK_START/STOP.
        self._block_open: bool = False

    @property
    def enabled(self) -> bool:
        return self.logger is not None

    def _progress(self) -> Dict[str, Any]:
        c = self.logger.constants
        samples = self.metric_logger.cumulative_train_samples
        epoch = (
            samples / self.total_train_samples if self.total_train_samples > 0 else 0.0
        )
        return {c.SAMPLES_COUNT: samples, c.EPOCH_NUM: epoch}

    def log_dataset_sizes(self, eval_samples: Optional[int] = None) -> None:
        if not self.enabled:
            return
        c = self.logger.constants
        self.logger.event(key=c.TRAIN_SAMPLES, value=self.total_train_samples)
        if eval_samples is not None:
            self.logger.event(key=c.EVAL_SAMPLES, value=int(eval_samples))

    def block_start(self) -> None:
        if self.enabled and not self._block_open:
            self.logger.start(
                key=self.logger.constants.BLOCK_START, metadata=self._progress()
            )
            self._block_open = True

    def block_stop(self) -> None:
        if self.enabled and self._block_open:
            self.logger.end(
                key=self.logger.constants.BLOCK_STOP, metadata=self._progress()
            )
            self._block_open = False

    def eval_start(self) -> None:
        if self.enabled:
            self.logger.start(
                key=self.logger.constants.EVAL_START, metadata=self._progress()
            )

    def _target_metric(self, metrics: Dict[str, float]) -> Optional[float]:
        # Key format `metric/{prefix}_{name}/{task}` (see MetricsLogger.compute);
        # match the per-window AUC short name.
        for key, val in metrics.items():
            short = key.split("/")[-2] if "/" in key else key
            if short == self._EVAL_METRIC_SHORT:
                return float(val)
        return None

    def _meets_target(self, value: Optional[float]) -> bool:
        thr = self.metric_logger.auc_threshold
        if value is None or thr is None:
            return False
        return value >= thr

    def run_stop(self, status: object) -> None:
        # Emit RUN_STOP exactly once, with an all-rank barrier so the timestamp
        # reflects the slowest rank (MLPerf requirement).
        if not self.enabled or self.run_stopped:
            return
        c = self.logger.constants
        self.logger.end(
            key=c.RUN_STOP,
            metadata={c.STATUS: status, **self._progress()},
            sync=True,
        )
        self.run_stopped = True

    def eval_stop(self, eval_metrics: Dict[str, float]) -> bool:
        # Emit EVAL_ACCURACY + EVAL_STOP, early SUCCESS RUN_STOP on target.
        # Rank 0 decides + broadcasts the stop bool so all ranks break in lockstep
        # (a per-rank test could diverge and hang the next all-to-all).
        if not self.enabled:
            return False
        c = self.logger.constants
        eval_value = self._target_metric(eval_metrics)
        if eval_value is not None:
            self.logger.event(
                key=c.EVAL_ACCURACY, value=eval_value, metadata=self._progress()
            )
        self.logger.end(key=c.EVAL_STOP, metadata=self._progress())
        decision = torch.zeros(1, device=self.device)
        if self.rank == 0 and not self.run_stopped and self._meets_target(eval_value):
            decision[0] = 1.0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast(decision, src=0)
        should_stop = bool(decision.item() > 0.5)
        if should_stop:
            # All ranks agree -> all reach the RUN_STOP barrier together.
            self.run_stop(c.SUCCESS)
        return should_stop

    def finalize(self, final_metrics: Dict[str, float]) -> None:
        # End-of-run RUN_STOP when the target was never crossed: SUCCESS iff the
        # final eval metric meets the target, else ABORTED.
        if not self.enabled or self.run_stopped:
            return
        c = self.logger.constants
        success = self._meets_target(self._target_metric(final_metrics))
        self.run_stop(c.SUCCESS if success else c.ABORTED)


def mlperf_checkpoint_present(ckpt_path: str) -> bool:
    """True iff ``ckpt_path`` resolves to an existing checkpoint (i.e. a resume).

    A dependency-light mirror of ``checkpoint._resolve_latest_subdir`` so
    ``train_ranker`` can decide cold-start vs resume BEFORE the heavy checkpoint
    import + ``setup()``. This gates the one-time INIT_START/RUN_START markers:
    emit them on a genuine cold start only, and never re-emit on a resume
    relaunch (the MLPerf run spans the resume). Matches the loader's resolution:
    empty path or a base dir with no numeric subdirs => cold start.
    """
    if not ckpt_path:
        return False
    base = ckpt_path.rstrip("/")
    # A leaf save (numeric basename) is a resume iff that dir actually exists.
    if os.path.basename(base).isdigit():
        return os.path.isdir(base)
    if not os.path.isdir(base):
        return False
    for name in os.listdir(base):
        if name.isdigit() and os.path.isdir(os.path.join(base, name)):
            return True
    return False


@gin.configurable
def get_mlperf_logger(
    rank: int = 0,
    log_path: str = "",
    benchmark_name: str = "hstu",
    submitter_name: str = "AMD",
    submission_platform: str = "MI355X",
    fresh: bool = True,
) -> Optional[MLPerfLogger]:
    """Build a configured :class:`MLPerfLogger`, or ``None`` if unavailable.

    Path defaults to ``$MLPERF_LOG_PATH``. Returns ``None`` (not a disabled
    logger) so callers' ``is not None`` guards cleanly skip logging.

    Disable knob: set ``$MLPERF_LOGGING=0`` (or false/no/off) to turn the whole
    MLPerf event stream off — returns ``None`` on EVERY rank, so the train loop's
    ``is not None`` guards skip emission AND the cross-rank train-loss all-reduce
    in lockstep. Default (unset / "1") = enabled, preserving prior behavior.
    """
    if not _MLLOG_AVAILABLE:
        return None
    if os.environ.get("MLPERF_LOGGING", "1").strip().lower() in (
        "0", "false", "no", "off",
    ):
        logger.info("MLPerf logging disabled via $MLPERF_LOGGING=0")
        return None
    resolved_path = os.environ.get("MLPERF_LOG_PATH", log_path)
    # SUBMISSION_PLATFORM defaults to "MI355X"; override per-submitter via env.
    resolved_platform = os.environ.get(
        "MLPERF_SUBMISSION_PLATFORM", submission_platform
    )
    return MLPerfLogger(
        rank=rank,
        log_path=resolved_path,
        benchmark_name=benchmark_name,
        submitter_name=submitter_name,
        submission_platform=resolved_platform,
        fresh=fresh,
    )
