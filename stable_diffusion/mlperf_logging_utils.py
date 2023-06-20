import os
from typing import Any, Dict, Optional, Type

from mlperf_logging import mllog
from mlperf_logging.mllog import constants
import mlperf_logging.mllog.constants as mllog_constants

import torch
import torch.distributed as dist

try:
    import lightning.pytorch as pl
    from lightning.pytorch.utilities.types import STEP_OUTPUT
except:
    import pytorch_lightning as pl
    from pytorch_lightning.utilities.types import STEP_OUTPUT


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def barrier():
    if not is_dist_avail_and_initialized():
        return
    torch.distributed.barrier()

class SDLogger:
    def __init__(self, filename=None, default_stack_offset=2):
        self.mllogger = mllog.get_mllogger()
        mllog.config(default_stack_offset=default_stack_offset,
                     filename=(filename or os.getenv("COMPLIANCE_FILE") or "mlperf_compliance.log"),
                     root_dir=os.path.normpath(os.path.dirname(os.path.realpath(__file__))))

    @property
    def rank(self):
        return get_rank()

    def event(self, key, value=None, metadata=None, sync=False, log_rank=None):
        log_rank = self.rank==0 if log_rank is None else log_rank
        if sync:
            barrier()
        if log_rank:
            self.mllogger.event(key=key, value=value, metadata=metadata)

    def start(self, key, value=None, metadata=None, sync=False, log_rank=None):
        log_rank = self.rank==0 if log_rank is None else log_rank
        if sync:
            barrier()
        if log_rank:
            self.mllogger.start(key=key, value=value, metadata=metadata)

    def end(self, key, value=None, metadata=None, sync=False, log_rank=None):
        log_rank = self.rank==0 if log_rank is None else log_rank
        if sync:
            barrier()
        if log_rank:
            self.mllogger.end(key=key, value=value, metadata=metadata)

def submission_info(mllogger: SDLogger,
                    submission_benchmark: str, submission_division: str, submission_org: str, submission_platform: str,
                    submission_poc_name: str, submission_poc_email: str, submission_status: str):
    """Logs required for a valid MLPerf submission."""
    mllogger.event(key=constants.SUBMISSION_BENCHMARK, value=submission_benchmark)
    mllogger.event(key=constants.SUBMISSION_DIVISION, value=submission_division)
    mllogger.event(key=constants.SUBMISSION_ORG, value=submission_org)
    mllogger.event(key=constants.SUBMISSION_PLATFORM, value=submission_platform)
    mllogger.event(key=constants.SUBMISSION_POC_NAME, value=submission_poc_name)
    mllogger.event(key=constants.SUBMISSION_POC_EMAIL, value=submission_poc_email)
    mllogger.event(key=constants.SUBMISSION_STATUS, value=submission_status)

class MLPerfLoggingCallback(pl.callbacks.Callback):
    def __init__(self, logger, train_log_interval=5, validation_log_interval=1):
        super().__init__()
        self.logger = mllogger
        self.train_log_interval = train_log_interval
        self.validation_log_interval = validation_log_interval

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.logger.start(mllog_constants.RUN_START)

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.logger.end(mllog_constants.RUN_STOP)

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass

    def on_train_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule",
                             batch: Any, batch_idx: int) -> None:
        if trainer.global_step % self.train_log_interval == 0:
            self.logger.start(key=mllog_constants.BLOCK_START, value="training_step", metadata={mllog_constants.STEP_NUM: trainer.global_step})

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule",
                           outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        if trainer.global_step % self.train_log_interval == 0:
            logs = trainer.callback_metrics
            self.logger.event(key="loss", value=logs["train/loss"].item(), metadata={mllog_constants.STEP_NUM: trainer.global_step})
            self.logger.event(key="lr_abs", value=logs["lr_abs"].item(), metadata={mllog_constants.STEP_NUM: trainer.global_step})
            self.logger.end(key=mllog_constants.BLOCK_STOP, value="training_step", metadata={mllog_constants.STEP_NUM: trainer.global_step})

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.logger.start(key=mllog_constants.EVAL_START, value=trainer.global_step)

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        logs = trainer.callback_metrics
        if "validation/fid" in logs:
            self.logger.event(key=mllog_constants.EVAL_ACCURACY,
                              value=logs["validation/fid"].item(),
                              metadata={mllog_constants.STEP_NUM: trainer.global_step, "metric": "FID"})
        if "validation/clip" in logs:
            self.logger.event(key=mllog_constants.EVAL_ACCURACY,
                              value=logs["validation/clip"].item(),
                              metadata={mllog_constants.STEP_NUM: trainer.global_step, "metric": "CLIP"})
        self.logger.end(key=mllog_constants.EVAL_STOP, value=trainer.global_step)

    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass

    def on_validation_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule",
                                  batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        if batch_idx % self.validation_log_interval == 0:
            self.logger.start(key=mllog_constants.BLOCK_START, value="validation_step", metadata={mllog_constants.STEP_NUM: batch_idx})

    def on_validation_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Optional[STEP_OUTPUT],
                                batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        if batch_idx % self.validation_log_interval == 0:
            self.logger.end(key=mllog_constants.BLOCK_STOP, value="validation_step", metadata={mllog_constants.STEP_NUM: batch_idx})


mllogger = SDLogger()
