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

import functools
import gc
import os
from omegaconf import OmegaConf
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
from transformers import logging
from torch_xla.experimental.distributed_checkpoint import (
    CheckpointManager,
    prime_optimizer,
)
import numpy as np
from torch_xla.experimental.spmd_fully_sharded_data_parallel import (
    SpmdFullyShardedDataParallel as FSDPv2,
)

from torch_xla.distributed.fsdp import checkpoint_module

from torch_xla.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)

from transformers.trainer_pt_utils import (
    get_module_class_from_name,
)
from psutil import Process
from transformers import AutoModelForCausalLM, AutoConfig, TrainerCallback
from nemo.core.optim.lr_scheduler import CosineAnnealing, WarmupHoldPolicy
from torch.utils.tensorboard import SummaryWriter
import json


logger = logging.get_logger(__name__)


def prepare_model(model, config):
    if config.tensor_parallelism == 1:

        def shard_output(output, mesh):
            real_output = None
            if isinstance(output, torch.Tensor):
                real_output = output
            elif isinstance(output, tuple):
                real_output = output[0]
            elif hasattr(output, "logits"):
                real_output = output.logits

            if real_output is None:
                raise ValueError(
                    "Something went wrong, the output of the model shouldn't be `None`"
                )
            xs.mark_sharding(real_output, mesh, ("fsdp", None, None))

        auto_wrap_policy = None
        auto_wrapper_callable = None

        default_transformer_cls_names_to_wrap = getattr(
            model, "_no_split_modules", None
        )
        fsdp_transformer_layer_cls_to_wrap = config.model.fsdp_config.get(
            "transformer_layer_cls_to_wrap", default_transformer_cls_names_to_wrap
        )

        if config.model.fsdp_config["min_num_params"] > 0:
            auto_wrap_policy = functools.partial(
                size_based_auto_wrap_policy,
                min_num_params=config.model.fsdp_config["min_num_params"],
            )
        elif fsdp_transformer_layer_cls_to_wrap is not None:
            transformer_cls_to_wrap = set()
            for layer_class in fsdp_transformer_layer_cls_to_wrap:
                transformer_cls = get_module_class_from_name(model, layer_class)
                if transformer_cls is None:
                    raise Exception(
                        "Could not find the transformer layer class to wrap in the model."
                    )
                else:
                    transformer_cls_to_wrap.add(transformer_cls)

            auto_wrap_policy = functools.partial(
                transformer_auto_wrap_policy,
                # Transformer layer class to wrap
                transformer_layer_cls=transformer_cls_to_wrap,
            )

        if config.model.fsdp_config["xla_fsdp_grad_ckpt"]:
            if model.config.use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
                )
                model.config.use_cache = False

            # Apply gradient checkpointing to auto-wrapped sub-modules if specified
            def auto_wrapper_callable(m, *args, **kwargs):
                target_cls = FSDPv2
                return target_cls(checkpoint_module(m), *args, **kwargs)

        model = FSDPv2(
            model,
            shard_output=shard_output,
            auto_wrap_policy=auto_wrap_policy,
            auto_wrapper_callable=auto_wrapper_callable,
        )

        return model
    else:
        model.to("xla")
        mesh = xs.get_global_mesh()
        for name, param in model.named_parameters():
            logger.debug(f"> [2D] Sharding tensor {name}, {param.shape}")

            # Here we intentionally skip layernorm and moe.gate weights given they are small.
            if "embed_tokens" in name:
                xs.mark_sharding(
                    param, mesh, ("fsdp", "tensor")
                )  # needed to have activations fully sharded.
            elif "q_proj" in name or "k_proj" in name or "v_proj" in name:
                xs.mark_sharding(param, mesh, ("tensor", "fsdp"))
            elif "o_proj" in name:
                xs.mark_sharding(param, mesh, ("fsdp", "tensor"))
            elif "w1" in name or "w3" in name:
                xs.mark_sharding(param, mesh, ("tensor", "fsdp"))
            elif "w2" in name:
                xs.mark_sharding(param, mesh, ("fsdp", "tensor"))
            elif "lm_head" in name:
                xs.mark_sharding(param, mesh, ("tensor", "fsdp"))

            logger.info(f"{name} {torch_xla._XLAC._get_xla_sharding_spec(param)}")

        for i, block in enumerate(model.model.layers):
            xs.apply_backward_optimization_barrier(model.model.layers[i])
        logger.info("Applying gradient checkpointing")
        if model.config.use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            model.config.use_cache = False

        for i, block in enumerate(model.model.layers):
            model.model.layers[i] = checkpoint_module(block)

    return model


def print_param_sharding(model):
    for name, param in model.named_parameters():
        logger.debug(
            f"{name}: {param.shape} {param.dtype} {torch_xla._XLAC._get_xla_sharding_spec(param)}"
        )


def setup_xla(config):
    if config.local_compile_cache_dir:
        xr.initialize_cache(config.local_compile_cache_dir)
    if config.full_precision:
        import jax

        assert config.model.dtype == "float32", "model dtype need to be float32"
        torch_xla._XLAC._xla_set_use_full_mat_mul_precision(
            use_full_mat_mul_precision=True
        )
        jax.config.update("jax_default_matmul_precision", "highest")

    num_devices = xr.global_runtime_device_count()
    mesh_shape = (num_devices // config.tensor_parallelism, config.tensor_parallelism)
    device_ids = np.array(range(num_devices))
    mesh = xs.Mesh(device_ids, mesh_shape, axis_names=("fsdp", "tensor"))
    xs.set_global_mesh(mesh)


def fmt_size(num_bytes: int) -> str:
    assert num_bytes > 0
    for unit in ["B", "KiB", "MiB", "GiB"]:
        if num_bytes < 1024.0:
            break
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} {unit}"


def get_cpu_memory() -> str:
    """print out cpu/tpu memory."""
    cpu_bytes = Process().memory_info().rss
    return fmt_size(cpu_bytes)


def setup_model_optimizer(config):
    dtype = getattr(torch, config.model.dtype)

    logger.debug(f"cpu memory usage: {get_cpu_memory()}")

    logger.info("loading model")
    if config.model.config_path:
        model_config = AutoConfig.from_pretrained(config.model.config_path)
        model_config.static = True
        model_config.flash_attention = config.model.flash_attention
        model_config.gmm = False
        model_config.gmm_stack = False
        model_config.capacity_factor = config.model.capacity_factor
        model_config.output_router_logits = True
        with torch.device("meta"):
            model = (
                AutoModelForCausalLM.from_config(model_config)
                .to_empty(device=xm.xla_device())
                .to(torch.bfloat16)
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.model.name_or_path,
            cache_dir=config.cache_local_dir,
            low_cpu_mem_usage=True,
            torch_dtype=dtype,
        )

    if model.config.architectures == ["MixtralForCausalLM"]:
        for layer in model.model.layers:
            layer.self_attn.rotary_emb._set_buffer(device=xm.xla_device())
    logger.info("model loaded")
    model = prepare_model(model, config)
    model = model.to(dtype)
    logger.info("model prepared")
    gc.collect()
    xm.mark_step()
    logger.debug(f"cpu memory usage: {get_cpu_memory()}")

    print_param_sharding(model)

    if config.checkpoint_manager_path:
        torch.distributed.init_process_group("gloo", init_method="xla://")
        logger.info(f"checkpoint found from {config.checkpoint_manager_path=}")

        ckpt_manager = CheckpointManager(
            path=config.checkpoint_manager_path,
            save_interval=float("inf"),
            max_to_keep=0,
        )

        state_dict = {
            "model": model.state_dict(),
        }
        ckpt_manager.restore(0, state_dict)
        model.load_state_dict(state_dict["model"])
        del state_dict
        xm.mark_step()
        logger.info("checkpoint loaded")
    else:
        if config.model.config_path:
            model.apply(model._init_weights)

        no_decay = ["bias", "layer_norm.weight"]

    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    if config.optimizer == "ADAMW_TORCH_XLA":
        from torch_xla.amp.syncfree import AdamW

        optimizer = AdamW(
            optimizer_grouped_parameters, lr=config.lr,
        )
    else:
        optimizer = getattr(torch.optim, config.optimizer)(
            optimizer_grouped_parameters, lr=config.lr
        )

    # initialize optimizer states and scheduler
    optimizer = prime_optimizer(optimizer)
    sched_config = OmegaConf.to_container(config.sched, resolve=True)
    scheduler_name = sched_config.pop("name")
    if scheduler_name == "WarmupHoldPolicy":
        scheduler = WarmupHoldPolicy(optimizer=optimizer, **sched_config)
    elif scheduler_name == "CosineAnnealing":
        assert (
            config.lr >= sched_config["min_lr"]
        ), f"{config.lr=} should be larger than {config.sched.min_lr=}"
        scheduler = CosineAnnealing(optimizer=optimizer, **sched_config)
    else:
        raise ValueError(
            f"{config.sched.name=} should be one of valid schedulers (WarmupHoldPolicy, CosineAnnealing)"
        )

    return model, optimizer, scheduler


def get_global_batch_size(per_device_batch_size):
    num_devices = xr.global_runtime_device_count()
    global_batch_size = int(per_device_batch_size * num_devices)
    return global_batch_size


def flatten(dictionary, parent_key="", separator="_"):
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, dict):
            items.extend(flatten(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)


class TensorBoardCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that sends the logs to [TensorBoard](https://www.tensorflow.org/tensorboard).

    Args:
        tb_writer (`SummaryWriter`, *optional*):
            The writer to use. Will instantiate one if not set.
    """

    def __init__(self, config):
        if xr.process_index() == 0:
            exp_config = {}
            for k, v in flatten(OmegaConf.to_container(config)).items():
                if isinstance(v, (str, int, float, str, bool, torch.Tensor)):
                    exp_config[k] = v
                else:
                    exp_config[k] = str(v)
            self.tb_writer = SummaryWriter(
                log_dir=os.path.join(config.run_dir, "tensorboard")
            )
            self.tb_writer.add_text("model_config", json.dumps(exp_config, indent=2))

    def on_log(self, args, state, control, logs=None, **kwargs):
        if xr.process_index() == 0:
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(k, v, state.global_step)
                else:
                    logger.warning(
                        "Trainer is attempting to log a value of "
                        f'"{v}" of type {type(v)} for key "{k}" as a scalar. '
                        "This invocation of Tensorboard's writer.add_scalar() "
                        "is incorrect so we dropped this attribute."
                    )
            self.tb_writer.flush()

    def on_train_end(self, args, state, control, **kwargs):
        if xr.process_index() == 0 and self.tb_writer:
            self.tb_writer.close()
            self.tb_writer = None
