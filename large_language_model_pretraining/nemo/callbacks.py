# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


### MLLogger
from mlperf_logging import mllog
from mlperf_logging.mllog import constants
import torch.distributed as dist

def is_dist_avail_and_initialized():
    return (dist.is_available() and dist.is_initialized())

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def barrier():
    if not is_dist_avail_and_initialized():
        return
    
    dist.barrier()

class MLLogger:
    def __init__(self, filepath="/mlperf-outputs/mlperf_llama31_405b.log", default_stack_offset=2):
        self.logger = mllog.get_mllogger()
        mllog.config(default_stack_offset=default_stack_offset, filename=filepath)

    def start(self, **kwargs):
        if get_rank() == 0:
            self.logger.start(**kwargs)

    def end(self, **kwargs):
        if get_rank() == 0:
            self.logger.end(**kwargs)

    def event(self, **kwargs):
        if get_rank() == 0:
            self.logger.event(**kwargs)

    def submission_info(self):
        self.event(key=constants.SUBMISSION_BENCHMARK, value="llama31_405b")
        self.event(key=constants.SUBMISSION_ORG, value="reference_implementation")
        self.event(key=constants.SUBMISSION_DIVISION, value=constants.CLOSED)
        self.event(key=constants.SUBMISSION_STATUS, value=constants.ONPREM)
        self.event(key=constants.SUBMISSION_PLATFORM, value="DGX-H100")
        self.event(key=constants.SUBMISSION_POC_NAME, value="Yunzhou Liu")
        self.event(key=constants.SUBMISSION_POC_EMAIL, value="yunzhoul@nvidia.com")

mllogger = MLLogger()

### Preemptive checkpoint callbacks
import lightning.pytorch as pl
from nemo.utils import logging

class PreemptiveStop(pl.Callback):
    """Preemptively stop training at a given global step. Allows stopping training before reaching
    the max steps. Useful for testing checkpoint save and resume.

    Args:
        stop_on_step (int): Stop training when trainer.global_step reaches this value.
            Checked at the start of every step.
    """

    def __init__(self, stop_on_step: int):
        self.stop_on_step = stop_on_step

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ) -> None:
        if trainer.global_step >= self.stop_on_step:
            logging.info(f"Global step {trainer.global_step} >= {self.stop_on_step}, signaling Trainer to stop.")
            trainer.should_stop = True
            # skip EarlyStopping validation unless val_check_interval met
            if trainer.global_step % trainer.val_check_interval != 0:
                trainer.limit_val_batches = 0


### Metrics Logger
from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities import rank_zero_only

class MetricsLogger(Logger):
    def __init__(
        self, 
        init_global_step, global_batch_size, seq_length,
        target_log_ppl, 
        train_loss_key = "reduced_train_loss",
        val_loss_key = "val_loss"
    ):
        super().__init__()

        self.init_global_step = init_global_step
        self.gbs = global_batch_size
        self.seq_len = seq_length

        self.target = target_log_ppl
        self.train_loss_key = train_loss_key
        self.val_loss_key = val_loss_key
        self.is_target_reached = False

    def log_metrics(self, metrics, step):
        if self.val_loss_key in metrics:
            self.log_validation_loss(metrics, step)

    def log_validation_loss(self, metrics, step):
        consumed_tokens = (step - self.init_global_step) * self.gbs * self.seq_len

        loss = metrics[self.val_loss_key]

        mllogger.event(key=constants.EVAL_ACCURACY, value=loss, metadata={'epoch_num': consumed_tokens})

        if not self.is_target_reached and loss <= self.target:
            self.is_target_reached = True

    @rank_zero_only
    def log_hyperparams(self, params, *args, **kwargs):
        pass

    @property
    def name(self):
        return 'mlperf-metrics'

    @property
    def version(self):
        return 1

### MLPerf callbacks
def compute_consumed_mllog_tokens(trainer, init_global_step, global_batch_size, seq_length):
    steps_since_resume = trainer.global_step + 1 - init_global_step # global steps are 0-indexed
    consumed_samples = (
        steps_since_resume * global_batch_size
    )
    return int(consumed_samples) * seq_length

class MLPerfCallback(pl.Callback):
    def __init__(
        self, 
        global_batch_size, 
        micro_batch_size,
        sequence_length,
        init_global_step,
        configs={}
    ):
        mllogger.event(key=constants.CACHE_CLEAR, value=True)
        mllogger.start(key=constants.INIT_START)
        super().__init__()

        self.init_global_step = init_global_step
        self.gbs = global_batch_size
        self.mbs = micro_batch_size
        self.seq_len = sequence_length

        self.is_target_reached = False
        self.status = constants.ABORTED
        self.configs = configs

    def consumed_tokens(self, trainer):
        return compute_consumed_mllog_tokens(trainer, self.init_global_step, self.gbs, self.seq_len)

    def set_success_status(self):
        self.status = constants.SUCCESS
        self.is_target_reached = True

    @rank_zero_only
    def on_train_epoch_start(self, trainer, pl_module):
        mllogger.start(key=constants.EPOCH_START, metadata={'epoch_num': self.consumed_tokens(trainer)})
        mllogger.start(key=constants.BLOCK_START, metadata={"epoch_num": self.consumed_tokens(trainer)})

        return super().on_train_epoch_start(trainer, pl_module)
    
    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        mllogger.end(key=constants.EPOCH_STOP, metadata={'epoch_num': self.consumed_tokens(trainer)})
        return super().on_train_epoch_end(trainer, pl_module)

    def on_train_end(self, trainer, pl_module):
        # for every occurrences, run on all ranks to allow sync
        barrier()
        mllogger.end(key=constants.RUN_STOP, metadata={"status": self.status})
        mllogger.event(key="trained_samples", value=self.consumed_tokens(trainer))
        return super().on_train_end(trainer, pl_module)
    
    @rank_zero_only
    def on_validation_start(self, trainer, pl_module):
        mllogger.end(key=constants.BLOCK_STOP, metadata={'epoch_num': self.consumed_tokens(trainer)})
        mllogger.start(key=constants.EVAL_START, metadata={'epoch_num': self.consumed_tokens(trainer)})
        return super().on_validation_start(trainer, pl_module)

    def on_validation_end(self, trainer, pl_module):
        mllogger.end(key=constants.EVAL_STOP, metadata={'epoch_num': self.consumed_tokens(trainer)})

        for logger in trainer.loggers:
            if isinstance(logger, MetricsLogger):
                if logger.is_target_reached:
                    trainer.should_stop = True
                    self.set_success_status()

        if not trainer.should_stop:
            mllogger.start(key=constants.BLOCK_START, metadata={"epoch_num": self.consumed_tokens(trainer)})

        return super().on_validation_end(trainer, pl_module)

    @rank_zero_only
    def load_state_dict(self, state_dict):
        print(f":::MLLOG Weight initialization: {state_dict.keys()}")
        return super().load_state_dict(state_dict)
    
    def on_train_start(self, trainer, pl_module):
        # run on all ranks to allow synchronization
        barrier()
        mllogger.submission_info()

        for key, value in self.configs.items():
            mllogger.event(key=key, value=value)

        mllogger.end(key=constants.INIT_STOP)
        mllogger.start(key=constants.RUN_START)