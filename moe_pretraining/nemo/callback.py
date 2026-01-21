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
from functools import wraps
from typing import Any, Callable, List, Optional, Protocol, Union

import torch
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.forward_step_func_types import ForwardStepCallable
from megatron.bridge.training.state import GlobalState
from megatron.core import parallel_state as mpu
from megatron.core.full_cuda_graph import FullCudaGraphWrapper
from megatron.core.inference.communication_utils import broadcast_from_last_pipeline_stage
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.optimizer import MegatronOptimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.rerun_state_machine import RerunDataIterator
from megatron.core.transformer import MegatronModule
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


class DeltaTimer:
    """Timer for measuring time deltas."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.start_time = time.perf_counter()
        return self.start_time

    def get_delta(self):
        prev_time = self.start_time
        return self.reset() - prev_time


# =============================================================================
# Callback Protocol and Manager
# =============================================================================

class TrainingCallback(Protocol):
    """Protocol defining all available callback hooks."""

    def on_train_start(
        self,
        global_state: GlobalState,
        forward_step_func: ForwardStepCallable,
        model: list[MegatronModule],
        optimizer: MegatronOptimizer,
        scheduler: OptimizerParamScheduler,
    ):
        """Called once at the start of training."""
        pass

    def on_train_end(
        self,
        global_state: GlobalState,
        forward_step_func: ForwardStepCallable,
        model: list[MegatronModule],
        optimizer: MegatronOptimizer,
        scheduler: OptimizerParamScheduler,
    ):
        """Called once at the end of training."""
        pass

    def on_train_batch_start(
        self,
        global_state: GlobalState,
        forward_step_func: ForwardStepCallable,
        model: list[MegatronModule],
        optimizer: MegatronOptimizer,
        scheduler: OptimizerParamScheduler,
    ):
        """Called before each training step."""
        pass

    def on_train_batch_end(
        self,
        global_state: GlobalState,
        forward_step_func: ForwardStepCallable,
        model: list[MegatronModule],
        optimizer: MegatronOptimizer,
        scheduler: OptimizerParamScheduler,
        ret: Any,
    ):
        """Called after each training step."""
        pass

    def on_validation_start(
        self,
        global_state: GlobalState,
        forward_step_func: ForwardStepCallable,
        model: list[MegatronModule],
    ):
        """Called before validation begins."""
        pass

    def on_validation_end(
        self,
        global_state: GlobalState,
        forward_step_func: ForwardStepCallable,
        model: list[MegatronModule],
        ret: Any,
    ):
        """Called after validation ends."""
        pass


class CallbackManager:
    """Manages callbacks and provides function wrappers."""

    def __init__(self):
        self.callbacks: List[TrainingCallback] = []

    def register(self, callback: TrainingCallback) -> None:
        self.callbacks.append(callback)

    def trigger_on_train_start(
        self,
        global_state: GlobalState,
        forward_step_func: ForwardStepCallable,
        model: list[MegatronModule],
        optimizer: MegatronOptimizer,
        scheduler: OptimizerParamScheduler,
    ) -> None:
        for callback in self.callbacks:
            if hasattr(callback, "on_train_start"):
                callback.on_train_start(
                    global_state,
                    forward_step_func,
                    model,
                    optimizer,
                    scheduler,
                )

    def trigger_on_train_end(
        self,
        global_state: GlobalState,
        forward_step_func: ForwardStepCallable,
        model: list[MegatronModule],
        optimizer: MegatronOptimizer,
        scheduler: OptimizerParamScheduler,
    ):
        for callback in self.callbacks:
            if hasattr(callback, "on_train_end"):
                callback.on_train_end(global_state, forward_step_func, model, optimizer, scheduler)

    def trigger_on_train_batch_start(
        self,
        global_state: GlobalState,
        forward_step_func: ForwardStepCallable,
        model: list[MegatronModule],
        optimizer: MegatronOptimizer,
        scheduler: OptimizerParamScheduler,
    ) -> None:
        for callback in self.callbacks:
            if hasattr(callback, "on_train_batch_start"):
                callback.on_train_batch_start(global_state, forward_step_func, model, optimizer, scheduler)

    def trigger_on_train_batch_end(
        self,
        global_state: GlobalState,
        iteration: int,
        loss_dict: dict[str, torch.Tensor],
        optimizer: MegatronOptimizer,
        model: list[MegatronModule],
        ret: Any,
    ) -> None:
        for callback in self.callbacks:
            if hasattr(callback, "on_train_batch_end"):
                callback.on_train_batch_end(global_state, iteration, loss_dict, optimizer, model, ret)

    def trigger_on_validation_start(
        self,
        global_state: GlobalState,
        forward_step_func: ForwardStepCallable,
        model: list[MegatronModule],
    ) -> None:
        for callback in self.callbacks:
            if hasattr(callback, "on_validation_start"):
                callback.on_validation_start(global_state, forward_step_func, model)

    def trigger_on_validation_end(
        self,
        global_state: GlobalState,
        forward_step_func: ForwardStepCallable,
        model: list[MegatronModule],
        ret: Any,
    ) -> None:
        for callback in self.callbacks:
            if hasattr(callback, "on_validation_end"):
                callback.on_validation_end(global_state, forward_step_func, model, ret)

    def wrap_train(self, train_func: Callable) -> Callable:
        """Wrap the train() function to add on_train_start and on_train_end hooks."""

        @wraps(train_func)
        def wrapped_train(
            forward_step_func: ForwardStepCallable,
            model: list[MegatronModule],
            optimizer: MegatronOptimizer,
            scheduler: OptimizerParamScheduler,
            train_data_iterator: Optional[Union[RerunDataIterator, list[RerunDataIterator]]],
            valid_data_iterator: Optional[Union[RerunDataIterator, list[RerunDataIterator]]],
            global_state: GlobalState,
            checkpointing_context: dict[str, Any],
            pg_collection: ProcessGroupCollection,
            process_non_loss_data_func: Optional[Callable] = None,
            non_loss_data_func: Optional[Callable] = None,
        ) -> None:
            self.trigger_on_train_start(global_state, forward_step_func, model, optimizer, scheduler)
            train_func(
                forward_step_func,
                model,
                optimizer,
                scheduler,
                train_data_iterator,
                valid_data_iterator,
                global_state,
                checkpointing_context,
                pg_collection,
                process_non_loss_data_func,
                non_loss_data_func,
            )
            self.trigger_on_train_end(global_state, forward_step_func, model, optimizer, scheduler)

        return wrapped_train

    def wrap_train_step(self, train_step_func: Callable) -> Callable:
        """Wrap the train_step() function to add batch-level hooks."""

        @wraps(train_step_func)
        def wrapped_train_step(
            forward_step_func: ForwardStepCallable,
            data_iterator: Optional[Union[RerunDataIterator, list[RerunDataIterator]]],
            model: list[MegatronModule],
            optimizer: MegatronOptimizer,
            scheduler: OptimizerParamScheduler,
            global_state: GlobalState,
            pg_collection: ProcessGroupCollection,
            forward_backward_func: Callable,
        ):
            self.trigger_on_train_batch_start(global_state, forward_step_func, model, optimizer, scheduler)
            ret = train_step_func(
                forward_step_func,
                data_iterator,
                model,
                optimizer,
                scheduler,
                global_state,
                pg_collection,
                forward_backward_func,
            )
            self.trigger_on_train_batch_end(global_state, forward_step_func, model, optimizer, scheduler, ret)
            return ret

        return wrapped_train_step

    def wrap_evaluate(self, evaluate_func: Callable) -> Callable:
        """Wrap the evaluate() function to add validation hooks."""

        @wraps(evaluate_func)
        def wrapped_evaluate(
            state: GlobalState,
            forward_step_func: ForwardStepCallable,
            data_iterator: Optional[Union[RerunDataIterator, list[RerunDataIterator]]],
            model: list[MegatronModule],
            process_non_loss_data_func: Optional[Callable],
            config: ConfigContainer,
            verbose: bool = False,
            non_loss_data_func: Optional[Callable] = None,
        ):
            self.trigger_on_validation_start(state, forward_step_func, model)
            ret = evaluate_func(
                state,
                forward_step_func,
                data_iterator,
                model,
                process_non_loss_data_func,
                config,
                verbose,
                non_loss_data_func,
            )
            self.trigger_on_validation_end(state, forward_step_func, model, ret)

            return ret

        return wrapped_evaluate


_callback_manager = CallbackManager()


def register_callback(callback: TrainingCallback) -> None:
    """Register a callback globally."""
    _callback_manager.register(callback)


def install_callbacks() -> None:
    """Install callbacks by wrapping the train, train_step, and evaluate functions."""
    import sys

    from megatron.bridge.training import eval as eval_module
    from megatron.bridge.training import train as train_module

    train_module.train = _callback_manager.wrap_train(train_module.train)
    train_module.train_step = _callback_manager.wrap_train_step(train_module.train_step)
    eval_module.evaluate = _callback_manager.wrap_evaluate(eval_module.evaluate)

    if "megatron.bridge.training.pretrain" in sys.modules:
        pretrain_module = sys.modules["megatron.bridge.training.pretrain"]
        pretrain_module.train = train_module.train


class MLPerfLoggingCallback:
    """MLPerf logging callback for compliance logging."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.global_batch_size = self.cfg.model.global_batch_size
        self.train_block_started = True
        self.train_current_block = 0
        self.previous_step = 0

    def on_train_start(
        self,
        global_state: GlobalState,
        forward_step_func: ForwardStepCallable,
        model: list[MegatronModule],
        optimizer: MegatronOptimizer,
        scheduler: OptimizerParamScheduler,
    ):
        mllogger.log_init_stop_run_start()
        global_state.should_stop = False
        mllogger.start(
            mllogger.constants.BLOCK_START,
            metadata={
                mllogger.constants.SAMPLES_COUNT: self.cfg.trainer.val_check_interval * self.global_batch_size,
                "step": global_state.train_state.step,
            },
        )
        self.timer = DeltaTimer()

    def on_train_end(
        self,
        global_state: GlobalState,
        forward_step_func: ForwardStepCallable,
        model: list[MegatronModule],
        optimizer: MegatronOptimizer,
        scheduler: OptimizerParamScheduler,
    ):
        if self.train_block_started:
            self._end_train_block(global_state)

        FullCudaGraphWrapper.cuda_graph = None

    def on_validation_start(
        self,
        global_state: GlobalState,
        forward_step_func: ForwardStepCallable,
        model: list[MegatronModule],
    ):
        """Log validation start."""
        if hasattr(global_state, "warmup") and global_state.warmup:
            return
        self._log_train_step_time(global_state)
        if self.train_block_started:
            self._end_train_block(global_state)

        mllogger.start(
            mllogger.constants.EVAL_START,
            metadata={
                mllogger.constants.SAMPLES_COUNT: self._get_samples_count(global_state),
                "step": self._get_step(global_state),
            },
        )

    def on_validation_end(
        self,
        global_state: GlobalState,
        forward_step_func: ForwardStepCallable,
        model: list[MegatronModule],
        ret: Any,
    ):
        if hasattr(global_state, "warmup") and global_state.warmup:
            return
        self._log_custom_timedelta("validation_time", self._get_step(global_state))

        samples_count = self._get_samples_count(global_state)
        if self.cfg.model.pipeline_model_parallel_size > 1:
            loss = broadcast_loss(ret[0])
        else:
            loss = ret[0]["lm loss"].item()

        mllogger.event(
            key=mllogger.constants.EVAL_ACCURACY,
            metadata={mllogger.constants.SAMPLES_COUNT: samples_count},
            value=loss,
        )
        mllogger.end(
            mllogger.constants.EVAL_STOP,
            metadata={
                mllogger.constants.SAMPLES_COUNT: samples_count,
                "step": self._get_step(global_state),
            },
        )

        if loss < self.cfg.custom.target_log_ppl:
            global_state.should_stop = True
            mllogger.end(
                mllogger.constants.RUN_STOP,
                metadata={mllogger.constants.SAMPLES_COUNT: samples_count, "status": "success"},
            )
        elif global_state.train_state.step >= self.cfg.trainer.max_steps:
            global_state.should_stop = True
            mllogger.end(
                mllogger.constants.RUN_STOP,
                metadata={mllogger.constants.SAMPLES_COUNT: samples_count, "status": "aborted"},
            )
        if not os.environ.get("VAL_CHECK_INTERVAL"):
            global_state.cfg.train.eval_interval = self.cfg.default_val_check_interval

        if not global_state.should_stop:
            self._start_train_block(global_state)
        else:
            global_state.train_state.step = self.cfg.trainer.max_steps + 1
            global_state.train_state.do_valid = False
            global_state.train_state.do_test = False

    def on_train_batch_end(
        self,
        global_state: GlobalState,
        forward_step_func: ForwardStepCallable,
        model: list[MegatronModule],
        optimizer: MegatronOptimizer,
        scheduler: OptimizerParamScheduler,
        ret: Any,
    ):
        step = global_state.train_state.step + 1
        last_step = step >= self.cfg.trainer.max_steps
        eval_after_this_step = step % global_state.cfg.train.eval_interval == 0
        if last_step and not eval_after_this_step:
            samples_count = self._get_samples_count(global_state)
            self._end_train_block(global_state)
            mllogger.end(
                mllogger.constants.RUN_STOP,
                metadata={mllogger.constants.SAMPLES_COUNT: samples_count, "status": "aborted"},
            )
            self.train_block_started = False
            global_state.should_stop = True
            global_state.train_state.do_valid = False
            global_state.train_state.do_test = False

    def _start_train_block(self, global_state: GlobalState) -> None:
        self.train_block_started = True
        mllogger.start(
            mllogger.constants.BLOCK_START,
            metadata={
                mllogger.constants.SAMPLES_COUNT: global_state.cfg.train.eval_interval * self.global_batch_size,
                "step": self._get_step(global_state),
            },
        )

    def _end_train_block(self, global_state: GlobalState) -> None:
        mllogger.end(
            mllogger.constants.BLOCK_STOP,
            metadata={
                mllogger.constants.SAMPLES_COUNT: global_state.cfg.train.eval_interval * self.global_batch_size,
                "step": self._get_step(global_state),
            },
        )
        self.train_block_started = False

    def _log_train_step_time(
        self,
        global_state: GlobalState,
    ) -> None:
        delta_t = self.timer.get_delta()
        global_step = self._get_step(global_state)
        delta_step = global_step - self.previous_step
        mllogger.event(
            key="tracked_stats",
            metadata={mllogger.constants.SAMPLES_COUNT: delta_step * self.global_batch_size},
            value={
                "train_step_time": delta_t / (delta_step + 1e-8),
            },
        )

        self.previous_step = global_step

    def _log_custom_timedelta(self, value_key, step: int = 0):
        mllogger.event(
            key="tracked_stats",
            metadata={"step": step},
            value={value_key: self.timer.get_delta()},
        )

    def _get_step(self, global_state):
        return global_state.train_state.step

    def _get_samples_count(self, global_state):
        return self._get_step(global_state) * self.global_batch_size


class DeltaTimingCallback:
    """Callback for tracking training step timing."""

    def __init__(self, cfg):
        self.t0 = 0
        self.total_train_step_time = [0, 0]
        self.global_batch_size = cfg.model.global_batch_size
        self.log_every_n_steps = cfg.trainer.log_every_n_steps

    def on_train_start(
        self,
        global_state: GlobalState,
        forward_step_func: ForwardStepCallable,
        model: list[MegatronModule],
        optimizer: MegatronOptimizer,
        scheduler: OptimizerParamScheduler,
    ):
        self.t0 = time.time()

    def on_train_batch_end(
        self,
        global_state: GlobalState,
        forward_step_func: ForwardStepCallable,
        model: list[MegatronModule],
        optimizer: MegatronOptimizer,
        scheduler: OptimizerParamScheduler,
        ret: Any,
    ):
        t1 = time.time()
        d = t1 - self.t0
        self.total_train_step_time[0] += d
        self.total_train_step_time[1] += 1
        self.t0 = t1

        if global_state.train_state.step % self.log_every_n_steps == 0 and get_last_pp_rank():
            mllogger.event(
                key="tracked_stats",
                metadata={mllogger.constants.SAMPLES_COUNT: self.global_batch_size * global_state.train_state.step},
                value={
                    "train_step_time": d,
                    "reduced_train_loss": ret[0]["lm loss"].item(),
                },
                unique=False,
            )

    def on_validation_end(
        self,
        global_state: GlobalState,
        forward_step_func: ForwardStepCallable,
        model: list[MegatronModule],
        ret: Any,
    ):
        """Reset timer after validation to avoid including validation time in first train step."""
        self.t0 = time.time()
