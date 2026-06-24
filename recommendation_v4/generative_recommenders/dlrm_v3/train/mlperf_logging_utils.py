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
        submitter_name: str = "reference_implementation",
    ):
        self.enabled: bool = _MLLOG_AVAILABLE
        # Use the EXPLICIT caller rank: this is built before init_process_group,
        # when dist.get_rank() would return 0 on every rank (all would log).
        self.rank: int = rank if rank is not None else _rank()
        self.benchmark_name: str = benchmark_name
        self.submitter_name: str = submitter_name
        self._logger = None
        if not self.enabled:
            return
        # Only rank 0 emits, so only rank 0 needs the file handler.
        if log_path and self.rank == 0:
            log_dir = os.path.dirname(log_path)
            if log_dir:  # guard: os.makedirs("") raises for a bare filename
                os.makedirs(log_dir, exist_ok=True)
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

    def submission_info(self, benchmark_name: str, submitter_name: str) -> None:
        """Emit the five SUBMISSION_* events required for a valid submission."""
        if not (self.enabled and self.rank == 0):
            return
        c = mllog_constants
        self.event(key=c.SUBMISSION_BENCHMARK, value=benchmark_name)
        self.event(key=c.SUBMISSION_ORG, value=submitter_name)
        self.event(key=c.SUBMISSION_DIVISION, value=c.CLOSED)
        self.event(key=c.SUBMISSION_STATUS, value=c.ONPREM)
        self.event(key=c.SUBMISSION_PLATFORM, value=submitter_name)


@gin.configurable
def get_mlperf_logger(
    rank: int = 0,
    log_path: str = "",
    benchmark_name: str = "hstu",
    submitter_name: str = "reference_implementation",
) -> Optional[MLPerfLogger]:
    """Build a configured :class:`MLPerfLogger`, or ``None`` if unavailable.

    Path defaults to ``$MLPERF_LOG_PATH``. Returns ``None`` (not a disabled
    logger) so callers' ``is not None`` guards cleanly skip logging.
    """
    if not _MLLOG_AVAILABLE:
        return None
    resolved_path = os.environ.get("MLPERF_LOG_PATH", log_path)
    return MLPerfLogger(
        rank=rank,
        log_path=resolved_path,
        benchmark_name=benchmark_name,
        submitter_name=submitter_name,
    )
