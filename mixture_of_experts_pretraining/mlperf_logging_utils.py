"""
Copyright 2024 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os

import torch
import torch.distributed as dist
from mlperf_logging import mllog
from mlperf_logging.mllog import constants
from pytorch_lightning import Callback
from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities import rank_zero_only
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    is_torch_xla_available,
)

if is_torch_xla_available():
    import torch_xla.runtime as xr


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if is_torch_xla_available():
        return xr.global_ordinal()
    else:
        if not is_dist_avail_and_initialized():
            return 0
        return dist.get_rank()


def barrier():
    if not is_dist_avail_and_initialized():
        return
    torch.distributed.barrier()


class ClmLogger:
    def __init__(self, config, filename=None, default_stack_offset=2):
        self.mllogger = mllog.get_mllogger()
        mllog.config(
            default_stack_offset=default_stack_offset,
            filename=(
                filename
                or os.getenv("COMPLIANCE_FILE")
                or os.path.join(config.run_dir, "mlperf_compliance.log")
            ),
        )
        self.target_eval_loss = config.target_eval_loss

    def event(self, key, value=None, metadata=None, sync=False, log_rank=None):
        if get_rank() == 0:
            self.mllogger.event(key=key, value=value, metadata=metadata)

    def start(self, key, value=None, metadata=None, sync=False, log_rank=None):
        if get_rank() == 0:
            self.mllogger.start(key=key, value=value, metadata=metadata)

    def end(self, key, value=None, metadata=None, sync=False, log_rank=None):
        if get_rank() == 0:
            self.mllogger.end(key=key, value=value, metadata=metadata)


class MLPerfCallback(TrainerCallback):
    "A callback that prints a message at the beginning of training"

    def __init__(self, config):
        super().__init__()
        self.mllogger = ClmLogger(config)
        self.submission_info = {
            "submission_benchmark": "mixture-of-expert",  # TODO change task name
            "submission_division": "closed",
            "submission_org": "Google",
            "submission_platform": "reference",
            "submission_status": "reference",
        }
        self.mllogger.event(
            key=constants.CACHE_CLEAR,
            value="True",
        )
        self.mllogger.start(key=constants.INIT_START, value="")
        self.config = config
        self.global_batch_tokens = config.global_train_batch_size * config.max_length

    def on_train_begin(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            num_devices = int(os.getenv("WORLD_SIZE", 1))
        elif is_torch_xla_available():
            num_devices = xr.global_runtime_device_count()
        else:
            raise ValueError("The pipeline should be either cuda or xla backend.")

        self.global_batch_size = int(
            args.per_device_train_batch_size
            * args.gradient_accumulation_steps
            * num_devices
        )

        self.mllogger.event(
            key=constants.SUBMISSION_BENCHMARK,
            value=self.submission_info["submission_benchmark"],
        )
        self.mllogger.event(
            key=constants.SUBMISSION_DIVISION,
            value=self.submission_info["submission_division"],
        )
        self.mllogger.event(
            key=constants.SUBMISSION_ORG, value=self.submission_info["submission_org"]
        )
        self.mllogger.event(
            key=constants.SUBMISSION_PLATFORM,
            value=self.submission_info["submission_platform"],
        )
        self.mllogger.event(
            key=constants.SUBMISSION_STATUS,
            value=self.submission_info["submission_status"],
        )
        self.mllogger.event(
            key=constants.GLOBAL_BATCH_SIZE,
            value=self.config.global_train_batch_size,
        )
        self.mllogger.event(
            key=constants.EVAL_SAMPLES,
            value=12694503,
        )
        self.mllogger.event(key=constants.SEED, value=args.seed)
        self.mllogger.event(
            key=constants.OPT_LR_WARMUP_FACTOR, value=args.sched.warmup_ratio
        )
        self.mllogger.event(key=constants.OPT_LR_TRAINING_STEPS, value=args.max_steps)
        self.mllogger.event(
            key=constants.OPT_ADAMW_WEIGHT_DECAY, value=args.weight_decay
        )
        self.mllogger.event(
            key=constants.OPT_GRADIENT_CLIP_NORM, value=args.max_grad_norm
        )
        self.mllogger.event(key=constants.OPT_BASE_LR, value=args.lr)
        self.mllogger.event(
            key=constants.GRADIENT_ACCUMULATION_STEPS,
            value=args.gradient_accumulation_steps,
        )
        # device warmup should be done here
        self.mllogger.end(key=constants.INIT_STOP, value="")
        self.mllogger.start(constants.RUN_START, value="")
        self.mllogger.start(
            constants.BLOCK_START,
            value="",
            metadata={
                "samples_count": 0,
            },
        )

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the end of a training step.
        """
        if state.global_step % state.eval_steps == 0 and state.global_step > 0:
            self.mllogger.event(
                "train_loss",
                value=state.log_history[-1]["train/loss"] if state.log_history else -1,
                metadata={
                    "samples_count": (
                        state.global_step * self.global_batch_tokens
                        if state.log_history
                        else -1
                    )
                },
            )
            control.should_log = True

        if state.global_step % state.eval_steps == 0:
            self.mllogger.end(
                constants.BLOCK_STOP,
                value="",
                metadata={
                    "samples_count": state.global_step * self.global_batch_tokens,
                },
            )
            self.mllogger.event(
                constants.EVAL_ACCURACY,
                value=state.log_history[-1]["eval/loss"],
                metadata={
                    "samples_count": state.global_step * self.global_batch_tokens,
                },
            )
            latest_eval_loss = float("nan")
            if state.log_history and "eval/loss" in state.log_history[-1]:
                latest_eval_loss = state.log_history[-1]["eval/loss"]
            if latest_eval_loss <= self.mllogger.target_eval_loss:
                control.should_training_stop = True
                self.mllogger.end(
                    constants.RUN_STOP,
                    value=latest_eval_loss,
                    metadata={
                        "samples_count": state.global_step * self.global_batch_tokens,
                        "status": "success",
                    },
                )
            if state.global_step >= state.max_steps:
                control.should_training_stop = True
                self.mllogger.end(
                    constants.RUN_STOP,
                    value=latest_eval_loss,
                    metadata={
                        "samples_count": state.global_step * self.global_batch_tokens,
                        "status": "fail",
                    },
                )

            if not control.should_training_stop:
                self.mllogger.start(
                    constants.BLOCK_START,
                    value="",
                    metadata={
                        "samples_count": state.global_step * self.global_batch_tokens
                    },
                )

        return control


class MLPerfLightningCallback(Callback):
    def __init__(self, logger, global_batch_size: int, sequence_length: int):
        super().__init__()
        self.gbs = global_batch_size
        self.seq = sequence_length
        self.mllogger = logger
        self.force_success = False

    def __deepcopy__(self, memo):
        return MLPerfLightningCallback(self.mllogger, self.gbs, self.seq)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        return super().on_train_batch_start(trainer, pl_module, batch, batch_idx)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    @rank_zero_only
    def on_validation_start(self, trainer, pl_module):
        self.mllogger.end(
            constants.BLOCK_STOP,
            metadata={"samples_count": trainer.global_step * self.gbs * self.seq},
            sync=False,
        )
        self.mllogger.start(
            key=constants.EVAL_START,
            metadata={"samples_count": trainer.global_step * self.gbs * self.seq},
            sync=False,
        )
        return super().on_validation_start(trainer, pl_module)

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        if not trainer.should_stop:
            self.mllogger.start(
                constants.BLOCK_START,
                metadata={"samples_count": trainer.global_step * self.gbs * self.seq},
                sync=False,
            )
        return super().on_validation_end(trainer, pl_module)

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        self.mllogger.start(
            constants.BLOCK_START, metadata={"samples_count": 0}, sync=False
        )

    @rank_zero_only
    def on_train_end(self, trainer, pl_module):
        if hasattr(trainer, "run_stop_logged") and not trainer.run_stop_logged:
            self.mllogger.end(
                constants.RUN_STOP,
                metadata={
                    "samples_count": trainer.global_step * self.gbs * self.seq,
                    "status": "aborted" if not self.force_success else "success",
                },
            )
        return super().on_train_end(trainer, pl_module)


class MetricsLogger(Logger):
    def __init__(
        self,
        logger,
        nodes: int,
        global_batch_size: int,
        learning_rate: float,
        sequence_length: int,
    ):
        super().__init__()
        self.nodes = nodes
        self.gbs = global_batch_size
        self.seq = sequence_length
        self.lr = learning_rate
        self.mllogger = logger
        self.experiment = None

    def __deepcopy__(self, memo):
        output = MetricsLogger(self.mllogger, self.nodes, self.gbs, self.lr, self.seq)
        if hasattr(self, "trainer"):
            output.trainer = self.trainer
        return output

    def set_trainer(self, trainer):
        self.trainer = trainer
        trainer.run_stop_logged = False

    @rank_zero_only
    def log_metrics(self, metrics, step):
        if "reduced_train_loss" in metrics:
            self.mllogger.event(
                "train_loss_update",
                value=metrics["reduced_train_loss"],
                metadata={
                    "samples_count": self.trainer.global_step * self.gbs * self.seq,
                },
            )

        if "val_loss" in metrics:
            val_loss = metrics["val_loss"]
            self.mllogger.event(
                constants.EVAL_ACCURACY,
                value=val_loss,
                metadata={
                    "samples_count": self.trainer.global_step * self.gbs * self.seq,
                },
            )
            self.mllogger.end(
                key=constants.EVAL_STOP,
                metadata={
                    "samples_count": self.trainer.global_step * self.gbs * self.seq
                },
                sync=False,
            )

    @rank_zero_only
    def log_hyperparams(self, params, *args, **kwargs):
        self.mllogger.event(key=constants.CACHE_CLEAR, value=True)
        self.mllogger.start(key=constants.INIT_START)
        # self.mllogger.mlperf_submission_log(
        #    benchmark="mixtral_8x22b",
        #    num_nodes=self.nodes,
        # )
        # self.mllogger.event(
        #     key=constants.SEED,
        #     value=self.cfg.model.seed,
        #     sync=False,
        #     unique=True,
        # )
        self.mllogger.event(
            key=constants.GLOBAL_BATCH_SIZE,
            value=self.gbs,
            sync=False,
        )
        # self.mllogger.event(
        #     key=constants.TRAIN_SAMPLES,
        #     value=0,
        # )
        # self.mllogger.event(
        #     key=constants.EVAL_SAMPLES,
        #     value=0,
        # )
        # self.mllogger.event(
        #     key=constants.OPT_LR_WARMUP_FACTOR,
        #     value=self.cfg.model.optim.sched.warmup_ratio,
        # )
        # self.mllogger.event(
        #     key=constants.OPT_ADAMW_WEIGHT_DECAY,
        #     value=self.cfg.model.optim.weight_decay,
        # )
        # self.mllogger.event(
        #     key=constants.OPT_GRADIENT_CLIP_NORM,
        #     value=self.cfg.trainer.gradient_clip_val,
        # )
        # ga = int(os.getenv("MINIBS", "1")) // self.cfg.model.micro_batch_size
        # self.mllogger.event(key=constants.GRADIENT_ACCUMULATION_STEPS, value=ga)
        # self.mllogger.event(
        #    key=constants.OPT_LR_TRAINING_STEPS, value=self.cfg.trainer.max_steps
        # )
        self.mllogger.event(key=constants.OPT_BASE_LR, value=self.lr)

    @property
    def name(self):
        return "mlperf-metrics"

    @property
    def version(self):
        return 1
