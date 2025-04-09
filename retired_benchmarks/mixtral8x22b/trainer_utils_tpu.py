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
from torch.nn import CrossEntropyLoss
from transformers.trainer_utils import EvalLoopOutput
from transformers.trainer_pt_utils import find_batch_size
from transformers import default_data_collator
from torch.utils.data import DataLoader
from typing import List
from tqdm.auto import tqdm

from transformers import logging, TrainerState, TrainerControl
import torch_xla.runtime as xr
from typing import Dict
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
import torch_xla.distributed.parallel_loader as pl
import torch_xla
import numpy as np
import datetime
import torch_xla.debug.profiler as xp
from transformers.models.mixtral.modeling_mixtral import load_balancing_loss_func

logger = logging.get_logger(__name__)
PROFILE_PORT = 9012
server = xp.start_server(PROFILE_PORT)
logger.info(f"Profiling server started: {server=}")


def calculate_tflops_training_per_device(model, config):
    n_params = sum({name: p.numel() for name, p in model.named_parameters()}.values())
    logger.info(f"Total size={n_params/1e9:.3f}B params")
    n_active_params = sum(
        {
            name: p.numel() for name, p in model.named_parameters() if p.requires_grad
        }.values()
    )
    logger.info(f"Active size={n_active_params/1e9:.3f}B params")

    # effective param
    if hasattr(model.config, "num_experts_per_tok") and hasattr(
        model.config, "num_local_experts"
    ):
        effective_n_params = (
            n_params * model.config.num_experts_per_tok / model.config.num_local_experts
        )
    else:
        effective_n_params = n_params

    # estimated tflops i.e. 6 * B * P, where B means number of tokens in batch
    tflops_training_per_device = (
        6
        * config.per_device_train_batch_size
        * config.max_length
        * effective_n_params
        / 1e12
    )

    logger.info(
        f"Estimated {tflops_training_per_device=} with dtype as {config.model.dtype}"
    )
    return tflops_training_per_device


class Trainer:
    def __init__(
        self,
        config,
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
        optimizer,
        scheduler,
        data_collator=default_data_collator,
        callbacks: List = None,
    ):
        self.config = config
        self.model = model
        mesh = xs.get_global_mesh()

        assert (
            config.global_train_batch_size % config.gradient_accumulation_steps == 0
        ), f"{config.global_train_batch_size=} is not divisable by {config.gradient_accumulation_steps=}"
        self.global_train_micro_batch_size = (
            config.global_train_batch_size // config.gradient_accumulation_steps
        )
        self.train_dataloader = pl.MpDeviceLoader(
            DataLoader(
                train_dataset,
                collate_fn=data_collator,
                batch_size=self.global_train_micro_batch_size,
            ),
            torch_xla.device(),
            input_sharding=xs.ShardingSpec(mesh, ("fsdp", None)),
        )

        self.eval_dataloader = pl.MpDeviceLoader(
            DataLoader(
                eval_dataset,
                collate_fn=data_collator,
                batch_size=config.global_eval_batch_size,
            ),
            torch_xla.device(),
            input_sharding=xs.ShardingSpec(mesh, ("fsdp", None)),
        )
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.callbacks = callbacks

        self.state = TrainerState()
        self.state.global_step = 0
        self.state.max_steps = config.max_steps
        self.state.eval_steps = config.eval_frequency
        self.control = TrainerControl()
        self.per_device_tflops = calculate_tflops_training_per_device(model, config)

    def compute_loss(self, batch, add_load_balancing_loss: bool = False):
        labels = batch.pop("labels")
        outputs = self.model(**batch)
        logits = outputs.logits
        # Flatten the tokens
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss(ignore_index=self.config.pad_token_id)
        # flatten
        shift_logits = shift_logits.view(-1, logits.shape[-1])
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
        num_tokens = (labels != self.config.pad_token_id).sum()
        loss_weight = (shift_labels != self.config.pad_token_id).sum()
        metrics = {
            "num_tokens": num_tokens,
            "loss_weight": loss_weight,
        }
        if add_load_balancing_loss:
            assert self.model.training
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.model.num_experts,
                self.model.num_experts_per_tok,
                attention_mask=batch["attention_mask"],
            )
            loss += self.model.router_aux_loss_coef * aux_loss
        return loss, metrics

    def eval_loop(self):
        self.model.eval()
        group_eval_loss_sum: List = []
        group_eval_loss_weight: List = []
        group_eval_num_tokens: List = []
        for eval_batch in self.eval_dataloader:
            with torch.no_grad():
                eval_loss_mean, eval_metrics = self.compute_loss(
                    eval_batch, add_load_balancing_loss=False
                )
                eval_num_tokens = eval_metrics["num_tokens"]
                eval_loss_weight = eval_metrics["loss_weight"]
                eval_loss_sum = eval_loss_mean * eval_loss_weight
                group_eval_loss_sum.append(eval_loss_sum)
                group_eval_loss_weight.append(eval_loss_weight)
                group_eval_num_tokens.append(eval_num_tokens)

        total_eval_loss_sum = sum(group_eval_loss_sum)
        total_eval_loss_weight = sum(group_eval_loss_weight)
        total_eval_num_tokens = sum(group_eval_num_tokens)
        group_eval_metrics = {
            "eval/loss": (total_eval_loss_sum / total_eval_loss_weight),
            "eval/num_tokens": total_eval_num_tokens,
            "eval/total_weights": total_eval_loss_weight,
        }
        return group_eval_metrics

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        logs = {
            k: v.cpu().item() if isinstance(v, torch.Tensor) else v
            for k, v in logs.items()
        }
        logs = {**logs, **{"step": self.state.global_step}}
        logger.info(f"{logs}")
        for callback in self.callbacks:
            callback.on_log(self.config, self.state, self.control, logs=logs)
        self.state.log_history.append(logs)

    def update_step(self):
        self.state.global_step += 1

    def train(self):
        # Train!
        for callback in self.callbacks:
            callback.on_train_begin(self.config, self.state, self.control)

        logger.info("***** Running training *****")
        logger.info(f"  Num Epochs = 1")
        logger.info(
            f"  Instantaneous batch size per device = {self.config.per_device_train_batch_size}"
        )
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {self.config.global_train_batch_size}"
        )
        logger.info(
            f"  Gradient Accumulation steps = {self.config.gradient_accumulation_steps}"
        )
        logger.info(f"  Total optimization steps = {self.config.max_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(
            range(self.config.max_steps), disable=xr.process_index() > 0
        )

        train_loss_list = []
        train_num_tokens_list = []
        eval_first = self.config.do_first_eval
        last_step_completion = datetime.datetime.now()
        for batch_idx, batch in enumerate(self.train_dataloader):
            if eval_first:
                eval_metrics = self.eval_loop()
                xm.add_step_closure(self.log, args=(eval_metrics,))
                eval_first = False

            if (
                self.control.should_training_stop
                or self.state.global_step >= self.config.max_steps
            ):
                xm.mark_step()
                break

            self.model.train()
            train_loss_step, train_metrics_step = self.compute_loss(
                batch, add_load_balancing_loss=True
            )
            train_num_tokens_step = train_metrics_step["num_tokens"]

            train_loss_step /= self.config.gradient_accumulation_steps
            train_loss_step.backward()
            train_loss_list.append(train_loss_step)
            train_num_tokens_list.append(train_num_tokens_step)
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # ensure wrap updating global step to avoid async in lazy printing
                logs: Dict[str, float] = {}
                if self.config.max_grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    logs["train/grad_norm"] = grad_norm.detach()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()

                train_loss = sum(train_loss_list)
                train_num_tokens = sum(train_num_tokens_list)
                logs["train/loss"] = train_loss.detach()
                logs["train/num_tokens"] = train_num_tokens.detach()
                logs["train/lr"] = self.scheduler.get_last_lr()[0]
                if (self.state.global_step + 1) % self.state.eval_steps == 0:
                    eval_metrics = self.eval_loop()
                    logs.update(eval_metrics)

                # add tflops per second
                new_time = datetime.datetime.now()
                step_time_delta = (new_time - last_step_completion).total_seconds()
                logs["perf/step_time_seconds"] = step_time_delta
                logs["perf/per_device_tflops"] = self.per_device_tflops
                logs["perf/per_device_tflops_per_sec"] = (
                    self.per_device_tflops / step_time_delta
                )
                logs["perf/per_device_tokens_per_sec"] = (
                    logs["train/num_tokens"] / step_time_delta
                )
                last_step_completion = new_time

                xm.add_step_closure(self.update_step)
                if (self.state.global_step + 1) % self.config.log_frequency == 0:
                    xm.add_step_closure(self.log, args=(logs,))
                    for callback in self.callbacks:
                        xm.add_step_closure(
                            callback.on_step_end,
                            args=(self.config, self.state, self.control),
                        )

                train_loss_list = []
                train_num_tokens_list = []
                progress_bar.update(1)

                if self.state.global_step == self.config.xla_profile_step:
                    xm.wait_device_ops()
                    duration_ms = 20000
                    xp.trace_detached(
                        f"localhost:{PROFILE_PORT}",
                        os.path.join(self.config.run_dir, "profile"),
                        duration_ms=duration_ms,
                    )

        logger.info("train finished.")
