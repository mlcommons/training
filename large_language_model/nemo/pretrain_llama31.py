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
from nemo.collections.common.tokenizers import SentencePieceTokenizer
import nemo_run as run

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
) -> run.Partial:
    
    if size == "8b":
        exp_name = "llama3-8b"
        pretrain_fn = llm.llama3_8b.pretrain_recipe
    elif size == "70b":
        exp_name = "llama3-70b"
        pretrain_fn = llm.llama3_70b.pretrain_recipe
    elif size == "405b":
        exp_name = "llama31-405b"
        pretrain_fn = llm.llama31_405b.pretrain_recipe

    pretrain = pretrain_fn(
        dir="/outputs",
        name=exp_name,
        num_nodes=nnodes, 
        num_gpus_per_node=ngpus_per_node
    )

    return exp_name, pretrain

def get_data(
    gbs: int = 288,
    mbs: int = 4,
    seq_length: int = 8192,
) -> run.Config:
    tokenizer = run.Config(SentencePieceTokenizer, model_path="/workspace/llm/tokenizer.model")
    data_paths = {
        "train": [
            0.5,
            "/preproc_data/c4_en_6_c4_spm_text_document",
            0.5,
            "/preproc_data/c4_en_7_c4_spm_text_document",
        ],
        "validation": [
            "/preproc_data/c4_en_validation_subset_c4_spm_text_document"
        ],
        "test": [
            "/preproc_data/c4_en_validation_subset_c4_spm_text_document"
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
        seed=1234, # TODO: make seed configurable here

        # The following options are not set in e2e_example but are present in pretrain_llama3
        # reset_position_ids=False,
        # reset_attention_mask=False,
        # eod_mask_loss=False,
        # rampup_batch_size=None,
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
    
    data_group = parser.add_argument_group("Dataset arguments")
    
    data_group.add_argument("--gbs", type=int, default=288, help="Global batch size, should be divisible by PP")
    data_group.add_argument("--mbs", type=int, default=1, help="Micro batch size")

    experiment_group = parser.add_argument_group("Experiment management arguments")
    experiment_group.add_argument("--dryrun", action="store_true", help="Whether we are launching dryrun or actual runs")
    experiment_group.add_argument("--seed", type=int, default=1234, help="random seed")

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

    exp_name, pretrain = get_pretrain(
        size=args.size,
        nnodes=args.nodes, 
        ngpus_per_node=args.gpus_per_node,
    )

    assert args.gbs % pretrain.trainer.strategy.pipeline_model_parallel_size == 0, "GBS should be divisible by PP"
    seq_length = pretrain.model.config.seq_length

    data = get_data(
        gbs=args.gbs,
        mbs=args.mbs,
        seq_length=seq_length,
    )

    pretrain.data = data

    # Override config for MLPerf
    pretrain.trainer.num_sanity_val_steps = 0

    # insert plugins and callbacks here
    # pretrain.trainer.callbacks.append(...)

    with run.Experiment(f"{exp_name}{args.tag}") as exp:
        for i in range(1):
            exp.add(
                pretrain, 
                executor=executor,
                name=exp_name,
                plugins=[]
            )

        if args.dryrun: 
            exp.dryrun()
        else:
            exp.run(sequential=True, detach=True)
    
