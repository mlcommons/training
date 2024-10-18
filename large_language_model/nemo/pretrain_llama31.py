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

import argparse
from typing import Optional
from nemo.collections import llm
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo import lightning as nl
from megatron.core.distributed import DistributedDataParallelConfig
from nemo.collections.llm.recipes.precision.mixed_precision import bf16_mixed
from nemo.collections.llm.gpt.data.mock import MockDataModule
from nemo.collections.llm.recipes.log.default import default_log, default_resume
from nemo.collections.llm.recipes.optim.adam import distributed_fused_adam_with_cosine_annealing
import nemo_run as run
from nemo.utils.exp_manager import TimingCallback
import torch
import os

def slurm_executor(
    user: str,
    host: str,
    remote_job_dir: str,
    account: str, 
    partition: str, 
    nodes: int,
    devices: int, 
    time: str = "01:00:00",
    custom_mounts: Optional[list[str]] = None, 
    custom_env_vars: Optional[dict[str, str]] = None, 
    container_image: str = "nvcr.io/nvidia/nemo:dev",
    dependencies: list[str] = [],
    retries: int = 0,
) -> run.SlurmExecutor:
    if not (user and host and remote_job_dir and account and partition and nodes and devices):
        raise RuntimeError(
            "Please set user, host, remote_job_dir, account, partition, nodes and devices args for using this function."
        )

    mounts = []
    if custom_mounts:
        mounts.extend(custom_mounts)

    env_vars = {
        "TRANSFORMERS_OFFLINE": "1",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
        "NVTE_ASYNC_AMAX_REDUCTION": "1",
        "NVTE_FUSED_ATTN": "0",
    }
    if custom_env_vars:
        env_vars |= custom_env_vars

    executor = run.SlurmExecutor(
        account=account,
        partition=partition,
        tunnel=run.SSHTunnel(
            user=user,
            host=host,
            job_dir=remote_job_dir,
        ),
        nodes=nodes,
        ntasks_per_node=devices,
        gpus_per_node=devices,
        mem="0",
        exclusive=True,
        gres="gpu:8",
        packager=run.GitArchivePackager(),
        dependencies=dependencies,
    )

    executor.launcher = None
    executor.container_image = container_image
    executor.container_mounts = mounts
    executor.env_vars = env_vars
    executor.retries = retries
    executor.time = time

    return executor

def get_pretrain(
    size: str, 
    nnodes: int, 
    ngpus_per_node: int,
    data_module: run.Config,
    ckpt: Optional[str]=None,
    eval_every: Optional[int]=None, 
    eval_batches: Optional[int]=None,
    max_steps: Optional[int]=None,
) -> run.Partial:
    
    exp_name = size
    
    if size == "8b":
        pretrain = llm.llama3_8b.pretrain_recipe(
            dir="/outputs",
            name=exp_name,
            num_nodes=nnodes,
            num_gpus_per_node=ngpus_per_node
        )

        llama31_config = run.Config(llm.gpt.model.llama.Llama31Config8B)
        llama31_config.seq_length = 8192
        pretrain.model.config = llama31_config
        pretrain.optim = distributed_fused_adam_with_cosine_annealing(max_lr=3e-4)
    elif size == "70b":
        pretrain = llm.llama3_70b.pretrain_recipe(
            dir="/outputs",
            name=exp_name,
            num_nodes=nnodes,
            num_gpus_per_node=ngpus_per_node
        )

        llama31_config = run.Config(llm.gpt.model.llama.Llama31Config70B)
        llama31_config.seq_length = 8192
        pretrain.model.config = llama31_config
        pretrain.optim = distributed_fused_adam_with_cosine_annealing(max_lr=1.5e-4)
    elif size == "405b":
        pretrain = llm.llama31_405b.pretrain_recipe(
            dir="/outputs",
            name=exp_name,
            num_nodes=nnodes,
            num_gpus_per_node=ngpus_per_node
        )

        pretrain.trainer.strategy.ddp = run.Config(
            DistributedDataParallelConfig,
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
        )

        pretrain.optim = distributed_fused_adam_with_cosine_annealing(max_lr=8e-5)

    # sets up everything else
    pretrain.data = data_module
    pretrain.trainer.val_check_interval = eval_every
    pretrain.trainer.limit_val_batches = eval_batches
    pretrain.trainer.limit_test_batches = eval_batches
    if max_steps is not None:
        pretrain.trainer.max_steps = max_steps

    if ckpt is not None:
        pretrain.resume = run.Config(nl.AutoResume, restore_config=run.Config(nl.RestoreConfig, path=ckpt))

    return exp_name, pretrain

def get_data(
    gbs: int = 288,
    mbs: int = 4,
    seq_length: Optional[int] = 8192,
    tokenizer_path: Optional[str] = "",
    seed: Optional[int] = 1234,
) -> run.Config:
    tokenizer = run.Config(AutoTokenizer, pretrained_model_name=tokenizer_path)

    train_datasets = sum([["12.5", f"/preproc_data/c4-train.en_{idx}_text_document"] for idx in range(8)], [])
    data_paths = {
        "train": train_datasets,
        "validation": [
            "/preproc_data/c4-validation.en_text_document"
        ],
        "test": [
            "/preproc_data/c4-validation.en_text_document"
        ],
    }

    return run.Config(
        llm.PreTrainingDataModule,
        tokenizer=tokenizer,
        paths=data_paths,
        num_workers=2, # TODO: make it configurable
        seq_length=seq_length,
        global_batch_size=gbs,
        micro_batch_size=mbs,
        index_mapping_dir="/npy_index",
        seed=seed,

        # Option to reset the position IDs in the dataset at an interval.
        reset_position_ids=False,
        # Option to reset the attention mask from the dataset.
        reset_attention_mask=False,
        # Option to enable the EOD mask loss.
        eod_mask_loss=False,
        # Rampup batch size, should be in format of [start_global_batch_size, batch_size_increment, ramup_samples].
        rampup_batch_size=None,
    )

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Llama3.1 Pretraining")
    parser.add_argument("--tag", type=str, help="Optional experiment tag", required=False, default="")

    # Slurm and executor related
    slurm_group = parser.add_argument_group("Slurm executor arguments")
    slurm_group.add_argument('--user', type=str, required=True, help="Remote cluster SSH user name")
    slurm_group.add_argument("--host", type=str, required=True, help="Remote cluster host address")
    slurm_group.add_argument("--job_dir", type=str, required=True, help="Remote job directory")

    slurm_group.add_argument("--account", type=str, required=True, help="Account to be used for Slurm job submission")
    slurm_group.add_argument("--partition", type=str, required=True, help="Partition to be used for Slurm job submission")
    slurm_group.add_argument("--nodes", type=int, required=True, help="Number of nodes to be used")
    slurm_group.add_argument("--gpus_per_node", type=int, required=True, help="Number of GPUs per node")
    slurm_group.add_argument("--time", type=str, required=True, help="Time limit for the job")
    slurm_group.add_argument("--dependencies", nargs="*", help="list of dependencies for the job, dependency type as 'afterok'") # not useful for now

    slurm_group.add_argument(
        "--mounts", 
        type=str, 
        required=True, 
        help=(
            "Custom mount paths, formatted as a string of <original>:<mapped>[,<original>:<mapped>], " + 
            "and should contain " + 
            "one path for /output, " + 
            "NeMo mounted on /opt/NeMo, " + 
            "dataset path: /workspace/llm/tokenizer.model, /preproc_data, /npy_index" 
        ))
    slurm_group.add_argument("--envvars", type=str, help="Environment variables to be added", default=None)
    slurm_group.add_argument("--image", type=str, required=True, help="Container image path, either remote or local")

    model_group = parser.add_argument_group("Model arguments")
    model_group.add_argument(
        "--size", 
        type=str, 
        default="8b", 
        help="Choose the model to be trained", 
        choices=[
            "8b", # Llama 3 8B config for debugging
            "70b", # Llama 3 70B config for debugging
            "405b", # Llama 3.1 405B config
        ])

    model_group.add_argument("--ckpt_path", type=str, default=None)
    model_group.add_argument("--use_ckpt", action="store_true")
    
    data_group = parser.add_argument_group("Dataset arguments")
    
    data_group.add_argument("--gbs", type=int, default=288, help="Global batch size, should be divisible by PP")
    data_group.add_argument("--mbs", type=int, default=1, help="Micro batch size")
    data_group.add_argument("--eval_every", type=int, default=10)
    data_group.add_argument("--eval_batches", type=int, default=None)
    data_group.add_argument('--max_steps', type=int, default=None)
    data_group.add_argument("--tokenizer_path", type=str, help="Tokenizer path that's used to tokenize the dataset")

    experiment_group = parser.add_argument_group("Experiment management arguments")
    experiment_group.add_argument("--dryrun", action="store_true", help="Whether we are launching dryrun or actual runs")
    experiment_group.add_argument("--seed", type=int, default=1234, help="random seed")
    experiment_group.add_argument("--num_exps", type=int, default=1)

    # TODO: add a checkpoint loading path here
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    if args.tag and not args.tag.startswith("-"):
        args.tag = "-" + args.tag

    executor = slurm_executor(
        user=args.user, 
        host=args.host, 
        remote_job_dir=args.job_dir,
        account=args.account,
        partition=args.partition,
        nodes=args.nodes,
        devices=args.gpus_per_node, 
        time = args.time,
        custom_mounts=list(args.mounts.split(",")),
        custom_env_vars=({envvar.split("=")[0]: envvar.split("=")[1] for envvar in args.envvars.split(",")} if args.envvars is not None else None),
        container_image=args.image,
        dependencies=args.dependencies,
    )

    seq_length = 8192

    data = get_data(
        gbs=args.gbs,
        mbs=args.mbs,
        seq_length=seq_length,
        tokenizer_path=args.tokenizer_path,
        seed=args.seed,
    )

    exp_prefix, pretrain = get_pretrain(
        size=args.size,
        nnodes=args.nodes, 
        ngpus_per_node=args.gpus_per_node,
        data_module=data,
        ckpt=args.ckpt_path if args.use_ckpt else None,
        eval_every=args.eval_every,
        eval_batches=args.eval_batches,
        max_steps=args.max_steps,
    )

    assert args.gbs % pretrain.trainer.strategy.pipeline_model_parallel_size == 0, "GBS should be divisible by PP"

    # Override config for MLPerf
    pretrain.trainer.num_sanity_val_steps = 0

    # insert plugins and callbacks here
    # pretrain.trainer.callbacks.append(...)

    exp_prefix = f"{exp_prefix}{args.tag}"

    for i in range(args.num_exps):
        exp_name = f"{exp_prefix}_{i}"
        with run.Experiment(exp_name) as exp:
            exp.add(
                pretrain, executor=executor, 
                name=exp_name,
                plugins=[]
            )

            if args.dryrun:
                exp.dryrun()
            else:
                exp.run(sequential=True, detach=True)
