# Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
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

import logging
import os
import time

import torch
from megatron.bridge.training.callbacks import Callback, CallbackContext
from megatron.core import parallel_state as mpu
from megatron.core.full_cuda_graph import FullCudaGraphWrapper
from megatron.core.inference.communication_utils import broadcast_from_last_pipeline_stage
from mlperf_common.frameworks.pyt import PyTCommunicationHandler
from mlperf_common.logging import MLLoggerWrapper


logger = logging.getLogger(__name__)


def get_last_pp_rank():
    """Check if current rank is the last pipeline parallel rank."""
    is_last_pp = mpu.is_pipeline_last_stage(ignore_virtual=True)
    is_first_dp = mpu.get_data_parallel_rank() == 0
    is_first_tp = mpu.get_tensor_model_parallel_rank() == 0
    is_first_cp = mpu.get_context_parallel_rank() == 0
    return is_last_pp and is_first_dp and is_first_tp and is_first_cp


def broadcast_loss(loss_reduced):
    """Broadcast loss from last pipeline stage to all ranks."""
    if "lm loss" in loss_reduced:
        loss_tensor = loss_reduced["lm loss"]
    else:
        loss_tensor = None

    loss_synced = broadcast_from_last_pipeline_stage(
        size=[1],
        dtype=torch.float32,
        tensor=loss_tensor.unsqueeze(0) if loss_tensor is not None else None,
    )

    return loss_synced.item()


mllogger = MLLoggerWrapper(PyTCommunicationHandler())


class MLPerfLoggingCallback(Callback):
    """MLPerf logging callback for compliance logging."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.global_batch_size = self.cfg.model.global_batch_size
        self.train_block_started = True
        self.train_current_block = 0

    def on_train_start(self, context: CallbackContext):
        mllogger.log_init_stop_run_start()
        context.state.should_stop = False
        mllogger.start(
            mllogger.constants.BLOCK_START,
            metadata={
                mllogger.constants.SAMPLES_COUNT: self.cfg.trainer.val_check_interval * self.global_batch_size,
                "step": context.state.train_state.step,
            },
        )

    def on_train_end(self, context: CallbackContext):
        if self.train_block_started:
            self._end_train_block(context.state)

        FullCudaGraphWrapper.cuda_graph = None

    def on_eval_start(self, context: CallbackContext):
        """Log validation start."""
        if hasattr(context.state, "warmup") and context.state.warmup:
            return
        if self.train_block_started:
            self._end_train_block(context.state)

        mllogger.start(
            mllogger.constants.EVAL_START,
            metadata={
                mllogger.constants.SAMPLES_COUNT: self._get_samples_count(context.state),
                "step": self._get_step(context.state),
            },
        )

    def on_eval_end(self, context: CallbackContext):
        if hasattr(context.state, "warmup") and context.state.warmup:
            return
        samples_count = self._get_samples_count(context.state)
        if self.cfg.model.pipeline_model_parallel_size > 1:
            loss = broadcast_loss(context.total_loss_dict)
        else:
            loss = context.total_loss_dict["lm loss"].item()

        mllogger.event(
            key=mllogger.constants.EVAL_ACCURACY,
            metadata={mllogger.constants.SAMPLES_COUNT: samples_count},
            value=loss,
        )
        mllogger.end(
            mllogger.constants.EVAL_STOP,
            metadata={
                mllogger.constants.SAMPLES_COUNT: samples_count,
                "step": self._get_step(context.state),
            },
        )

        if loss < self.cfg.custom.target_log_ppl:
            context.state.should_stop = True
            mllogger.end(
                mllogger.constants.RUN_STOP,
                metadata={mllogger.constants.SAMPLES_COUNT: samples_count, "status": "success"},
            )
        elif context.state.train_state.step >= self.cfg.trainer.max_steps:
            context.state.should_stop = True
            mllogger.end(
                mllogger.constants.RUN_STOP,
                metadata={mllogger.constants.SAMPLES_COUNT: samples_count, "status": "aborted"},
            )
        if not os.environ.get("VAL_CHECK_INTERVAL"):
            context.state.cfg.validate.eval_interval = self.cfg.trainer.val_check_interval

        if not context.state.should_stop:
            self._start_train_block(context.state)
        else:
            context.state.train_state.step = self.cfg.trainer.max_steps + 1
            context.state.train_state.do_valid = False
            context.state.train_state.do_test = False

    def on_train_step_end(self, context: CallbackContext):
        step = context.state.train_state.step + 1
        last_step = step >= self.cfg.trainer.max_steps
        eval_after_this_step = step % context.state.cfg.validate.eval_interval == 0
        if last_step and not eval_after_this_step:
            samples_count = self._get_samples_count(context.state)
            self._end_train_block(context.state)
            mllogger.end(
                mllogger.constants.RUN_STOP,
                metadata={mllogger.constants.SAMPLES_COUNT: samples_count, "status": "aborted"},
            )
            self.train_block_started = False
            context.state.should_stop = True
            context.state.train_state.do_valid = False
            context.state.train_state.do_test = False

    def _start_train_block(self, global_state) -> None:
        self.train_block_started = True
        mllogger.start(
            mllogger.constants.BLOCK_START,
            metadata={
                mllogger.constants.SAMPLES_COUNT: global_state.cfg.validate.eval_interval * self.global_batch_size,
                "step": self._get_step(global_state),
            },
        )

    def _end_train_block(self, global_state) -> None:
        mllogger.end(
            mllogger.constants.BLOCK_STOP,
            metadata={
                mllogger.constants.SAMPLES_COUNT: global_state.cfg.validate.eval_interval * self.global_batch_size,
                "step": self._get_step(global_state),
            },
        )
        self.train_block_started = False

    def _get_step(self, global_state):
        return global_state.train_state.step

    def _get_samples_count(self, global_state):
        return self._get_step(global_state) * self.global_batch_size


class DeltaTimingCallback(Callback):
    """Callback for tracking training step timing."""

    def __init__(self, cfg):
        self.t0 = 0
        self.total_train_step_time = [0, 0]
        self.global_batch_size = cfg.model.global_batch_size
        self.log_every_n_steps = cfg.trainer.log_every_n_steps

    def on_train_start(self, context: CallbackContext):
        self.t0 = time.time()

    def on_train_step_end(self, context: CallbackContext):
        t1 = time.time()
        d = t1 - self.t0
        self.total_train_step_time[0] += d
        self.total_train_step_time[1] += 1
        self.t0 = t1

        if context.state.train_state.step % self.log_every_n_steps == 0 and get_last_pp_rank():
            mllogger.event(
                key="tracked_stats",
                metadata={mllogger.constants.SAMPLES_COUNT: self.global_batch_size * context.state.train_state.step},
                value={
                    "train_step_time": d,
                    "reduced_train_loss": context.loss_dict["lm loss"].item(),
                },
                unique=False,
            )

    def on_eval_end(self, context: CallbackContext):
        """Reset timer after validation to avoid including validation time in first train step."""
        self.t0 = time.time()
