from typing import Any, Dict, Optional, Type
from lightning.pytorch.utilities.types import STEP_OUTPUT

from mlperf_logging.mllog import constants
from mlperf_logging.mllog.mllog import MLLogger
import mlperf_logging.mllog.constants as mllog_constants

try:
    import lightning.pytorch as pl
except:
    import pytorch_lightning as pl


def submission_info(mllogger: MLLogger,
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
    def __init__(self, mllogger):
        super().__init__()
        self.logger = mllogger

    # Misc
    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass


    # Validation logging
    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.logger.start(key=mllog_constants.EVAL_START, value=trainer.global_step) # TODO(ahmadki): is this the right place ?
        
    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.logger.end(key=mllog_constants.EVAL_STOP, value=trainer.global_step) # TODO(ahmadki): is this the right place ?

    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass
    
    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if "validation/fid" in trainer.callback_metrics:
            self.logger.event(key=mllog_constants.EVAL_ACCURACY,
                              value=trainer.callback_metrics["validation/fid"].item(),
                              metadata={mllog_constants.STEP_NUM: trainer.global_step, "metric": "FID"})
        if "validation/clip" in trainer.callback_metrics:
            self.logger.event(key=mllog_constants.EVAL_ACCURACY,
                              value=trainer.callback_metrics["validation/clip"].item(),
                              metadata={mllog_constants.STEP_NUM: trainer.global_step, "metric": "CLIP"})


    def on_validation_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule",
                                  batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        pass
 
    def on_validation_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Optional[STEP_OUTPUT],
                                batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        pass


    # Train logging
    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.logger.start(mllog_constants.RUN_START) # TODO(ahmadki): is this the right place ?

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.logger.end(mllog_constants.RUN_STOP) # TODO(ahmadki): is this the right place ?

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass

    def on_train_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule",
                             batch: Any, batch_idx: int) -> None:
        pass

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule",
                           outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        pass
