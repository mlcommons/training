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
from megatron.core.optimizer import OptimizerConfig
from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.utils import logging


def setup_distributed(config):
    """Initialize torch.distributed."""
    # Get rank and world size.
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    logging.info(
        f"Initializing torch.distributed with local_rank: {local_rank}, rank: {rank}, world_size: {world_size}"
    )

    # Set the device id.
    device = rank % torch.cuda.device_count()
    if local_rank is not None:
        device = local_rank
    torch.cuda.set_device(device)

    # Call the init process.
    init_method = "tcp://"
    master_ip = os.getenv("MASTER_ADDR", "localhost")
    master_port = os.getenv("MASTER_PORT", "6000")
    import datetime

    DEFAULT_TIMEOUT = datetime.timedelta(minutes=60)
    init_method += master_ip + ":" + master_port
    torch.distributed.init_process_group(
        backend="nccl",
        timeout=DEFAULT_TIMEOUT,
        world_size=world_size,
        rank=rank,
        init_method=init_method,
    )
    return local_rank, rank, world_size


def setup_model_and_trainer(
    model_name_or_path: str,
    input_sequence_length: int,
    global_batch_size: int,
    nodes: int,
    tp_size: int,
    pp_size: int,
    vpp_size: int,
    cp_size: int,
    learning_rate: float,
    weight_decay: float,
    optimizer_name: str,
    tokenizer_name_or_path: str,
    scheduler,
    max_grad_norm: float,
    eval_frequency: int,
    log_frequency: int,
    max_steps: int,
    *,
    logger,
    callbacks: list,
):
    logging.info("loading model")

    if "mixtral-8x7b" in model_name_or_path.lower():
        mixtral_config = llm.MixtralConfig8x7B()
    elif "mixtral-8x22b" in model_name_or_path.lower():
        mixtral_config = llm.MixtralConfig8x22B(
            moe_aux_loss_coeff=0.001,
        )
    else:
        raise ValueError(f"Unknown model specified: {model_name_or_path}")

    resume = nl.AutoResume(resume_from_path="/app/checkpoints/")
    tokenizer = AutoTokenizer(pretrained_model_name=tokenizer_name_or_path)
    model = llm.MixtralModel(mixtral_config, tokenizer=tokenizer)

    ## initialize the strategy
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        virtual_pipeline_model_parallel_size=vpp_size,
        sequence_parallel=True,
        context_parallel_size=cp_size,
        pipeline_dtype=torch.bfloat16,
        ckpt_load_optimizer=False,
    )

    precision = nl.MegatronMixedPrecision(
        precision="bf16-mixed",
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        autocast_enabled=False,
        grad_reduce_in_fp32=True,
    )

    ## setup the optimizer
    opt_config = OptimizerConfig(
        optimizer=optimizer_name,
        lr=learning_rate,
        weight_decay=weight_decay,
        bf16=True,
        fp16=False,
        params_dtype=torch.bfloat16,
        clip_grad=max_grad_norm,
    )

    if scheduler.name == "CosineAnnealing":
        opt_sched = nl.lr_scheduler.CosineAnnealingScheduler(
            warmup_steps=scheduler.warmup_steps
            if "warmup_steps" in scheduler
            else None,
            warmup_ratio=scheduler.warmup_ratio
            if "warmup_steps" not in scheduler
            else None,
            max_steps=scheduler.max_steps,
            min_lr=scheduler.min_lr,
        )
    elif scheduler.name == "WarmupHoldPolicy":
        opt_sched = nl.lr_scheduler.WarmupHoldPolicyScheduler(
            warmup_steps=scheduler.warmup_steps
            if "warmup_steps" in scheduler
            else None,
            warmup_ratio=scheduler.warmup_ratio
            if "warmup_steps" not in scheduler
            else None,
            hold_steps=scheduler.hold_steps,
            max_steps=scheduler.max_steps,
        )

    opt = nl.MegatronOptimizerModule(config=opt_config, lr_scheduler=opt_sched)
    trainer = nl.Trainer(
        devices=torch.cuda.device_count(),
        num_nodes=nodes,
        max_steps=max_steps,
        accelerator="gpu",
        strategy=strategy,
        plugins=precision,
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=False,
        val_check_interval=eval_frequency,
        log_every_n_steps=log_frequency,
    )

    logger.set_trainer(trainer)
    logger.log_hyperparams(None)

    return (
        model,
        trainer,
        opt,
        resume,
    )
