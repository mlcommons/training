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
import math
from nemo.collections import llm
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo import lightning as nl
from megatron.core.distributed import DistributedDataParallelConfig
from nemo.collections.llm.recipes.optim.adam import distributed_fused_adam_with_cosine_annealing
import nemo_run as run
from nemo.lightning.run import plugins
from nemo.collections.llm.gpt.data import build_pretraining_datamodule
from callbacks import PreemptiveStop, MLPerfCallback, MetricsLogger

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
        "TOKENIZERS_PARALLELISM": "false",
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
    eval_every: Optional[int]=None, 
    eval_batches: Optional[int]=None,
) -> run.Partial:
    
    exp_name = size
    base_gbs = 1152
    gbs = data_module.global_batch_size
    
    # Providing 8B and 70B here for debugging purpose
    # Actual benchmark should use 405B
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

        pretrain.trainer.strategy.virtual_pipeline_model_parallel_size = 7

        base_lr = 8e-5
        warmup_tokens = 8000 * base_gbs * 8192

        max_lr = (gbs / base_gbs) * base_lr
        max_lr = round(max_lr, 8) # rounds to the nearest 8th digit.

        # Code tracing shows that this is AdamW
        pretrain.optim = distributed_fused_adam_with_cosine_annealing(
            max_lr = max_lr, 
            warmup_steps = math.ceil(warmup_tokens / 8192 / gbs),
            min_lr = 8e-7
        )

        from nemo.collections.llm.recipes.tp_overlap_configs.userbuffers import (
            userbuffers_bf16_h100_h16384_tp8_cp2_mbs1_seqlen8192,
        )
        from nemo.lightning.pytorch.callbacks.megatron_comm_overlap import MegatronCommOverlapCallback

        pretrain.trainer.callbacks.append(
            run.Config(
                MegatronCommOverlapCallback,
                tp_comm_overlap=True,
                tp_comm_overlap_cfg=userbuffers_bf16_h100_h16384_tp8_cp2_mbs1_seqlen8192,
                defer_embedding_wgrad_compute=True,
                wgrad_deferral_limit=50,
                overlap_param_gather_with_optimizer_step=False, 
                align_param_gather=True,
            )
        )

    # sets up everything else
    max_tokens = 1_200_000 * 8192 * base_gbs # Llama 3.1 paper section 3.4.1 - decays LR to 8e10-7 over 1,200,000 steps
    pretrain.trainer.max_steps = math.ceil(max_tokens / 8192 / gbs)

    pretrain.data = data_module
    pretrain.trainer.val_check_interval = eval_every
    pretrain.trainer.limit_val_batches = eval_batches
    pretrain.trainer.limit_test_batches = eval_batches

    pretrain.log.tensorboard = None
    pretrain.log.ckpt.every_n_train_steps = None
    pretrain.log.ckpt.save_top_k = 1
    pretrain.log.ckpt.save_last = False
    pretrain.log.ckpt.always_save_context = True
    pretrain.log.ckpt.save_weights_only = False
    pretrain.log.ckpt.save_optim_on_train_end = True
    pretrain.log.ckpt.save_on_train_epoch_end = True
    pretrain.log.ckpt.monitor = "consumed_samples"
    pretrain.log.ckpt.mode = "max"

    return exp_name, pretrain

def get_data(
    gbs: int = 288,
    mbs: int = 4,
    seq_length: Optional[int] = 8192,
    tokenizer_path: Optional[str] = "",
    seed: Optional[int] = 1234,
    use_full_dataset: Optional[bool] = False,
) -> run.Config:
    tokenizer = run.Config(AutoTokenizer, pretrained_model_name=tokenizer_path)

    train_datasets = None

    if use_full_dataset:
        train_datasets = sum([["12.5", f"/preproc_data/c4-train.en_{idx}_text_document"] for idx in range(8)], [])
    else:
        train_datasets = sum([["50", f"/preproc_data/c4-train.en_{idx}_text_document"] for idx in range(6, 8)], [])

    data_paths = {
        "train": train_datasets,
        "validation": [
            "/preproc_data/c4-validation-91205-samples.en_text_document"
        ],
        "test": [
            "/preproc_data/c4-validation-91205-samples.en_text_document"
        ],
    }

    return run.Config(
        llm.PreTrainingDataModule,
        tokenizer=tokenizer,
        paths=data_paths,
        num_workers=8, # TODO: make it configurable
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
    slurm_group.add_argument("--max_retries", type=int, default=0)

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
        default="405b", 
        help="Choose the model to be trained", 
        choices=[
            "8b", # Llama 3 8B config for debugging
            "70b", # Llama 3 70B config for debugging
            "405b", # Llama 3.1 405B config
        ])

    model_group.add_argument("--initial_ckpt_path", type=str, default=None)
    model_group.add_argument("--use_ckpt", action="store_true", help="If set, then resume from the initial checkpoint path")
    model_group.add_argument("--resume_from_hf", action="store_true", help="Setting this knob indicates that we are resuming from a weight-only checkpoint")
    model_group.add_argument("--ckpt_start_step", type=int, default=0, help="Sets this value to how many steps the resumed checkpoint is already trained on")
    model_group.add_argument("--continual_ckpt_path", type=str, default=None, help="Sets this to the path that saves the checkpoint")
    model_group.add_argument("--save_ckpt", action="store_true", help="If set, then we save the checkpoint at the end of the experiment")
    
    data_group = parser.add_argument_group("Dataset arguments")
    
    data_group.add_argument("--gbs", type=int, default=1152, help="Global batch size, should be divisible by PP")
    data_group.add_argument("--mbs", type=int, default=1, help="Micro batch size")
    data_group.add_argument("--eval_every", type=int, default=46080, help="Evaluate at least every N training sequences")
    data_group.add_argument("--eval_tokens", type=int, default=5760, help="Evaluate using at least N evaluation sequences")
    data_group.add_argument('--max_steps', type=int, default=None, help="Maximum number of steps that each experiment partition will train on. None means no restriction on max steps. ")
    data_group.add_argument("--use_full_dataset", action="store_true", help="If set, then we use the full dataset, instead of the last 256/1024 shards")
    data_group.add_argument("--tokenizer_path", type=str, help="Tokenizer path that's used to tokenize the dataset")

    experiment_group = parser.add_argument_group("Experiment management arguments")
    experiment_group.add_argument("--dryrun", action="store_true", help="Whether we are launching dryrun or actual runs")
    experiment_group.add_argument("--seeds", type=int, nargs="*", default=[], help="random seeds")
    experiment_group.add_argument("--num_exps", type=int, default=1)
    experiment_group.add_argument("--num_pars", type=int, default=1)
    experiment_group.add_argument("--target_log_ppl", type=float, default=5.6)
    experiment_group.add_argument("--step_time_atol", type=int, default=1600, help="train step time atol")

    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    if args.tag and not args.tag.startswith("-"):
        args.tag = "-" + args.tag

    assert not (args.num_pars == 1 and args.continual_ckpt_path is None), "NPar > 1 but a shared checkpoint path is not found"
    assert not (not args.save_ckpt and args.num_pars > 1), "multiple experiments are specified but checkpoint is not saved"

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
        retries = args.max_retries,
    )

    seq_length = 8192

    data = get_data(
        gbs=args.gbs,
        mbs=args.mbs,
        seq_length=seq_length,
        tokenizer_path=args.tokenizer_path,
        seed=1234, # overwritten in each experiments
        use_full_dataset=args.use_full_dataset,
    )

    eval_every_n_batches = math.ceil(args.eval_every / (args.gbs))
    eval_batches = math.ceil(args.eval_tokens / (args.gbs))

    exp_prefix, pretrain = get_pretrain(
        size=args.size,
        nnodes=args.nodes, 
        ngpus_per_node=args.gpus_per_node,
        data_module=data,
        eval_every=eval_every_n_batches,
        eval_batches=eval_batches,
    )

    assert args.gbs % pretrain.trainer.strategy.pipeline_model_parallel_size == 0, "GBS should be divisible by PP"
    
    # Collect all HP configs
    from mlperf_logging.mllog import constants
    
    tp = pretrain.trainer.strategy.tensor_model_parallel_size
    pp = pretrain.trainer.strategy.pipeline_model_parallel_size
    cp = pretrain.trainer.strategy.context_parallel_size
    dp = (pretrain.trainer.num_nodes * pretrain.trainer.devices) // (tp * pp * cp)
    mini_batch_size = (args.gbs // dp)
    grad_accumulation_steps = mini_batch_size // args.mbs

    configs = {
        # HPs
        constants.GLOBAL_BATCH_SIZE: args.gbs,
        constants.GRADIENT_ACCUMULATION_STEPS: grad_accumulation_steps,
        constants.MAX_SEQUENCE_LENGTH: 8192,
        constants.EVAL_SAMPLES: args.eval_tokens,

        # Optimizers
        constants.OPT_NAME: "adamw", 
        constants.OPT_BASE_LR: pretrain.optim.config.lr,
        constants.OPT_ADAMW_BETA_1: pretrain.optim.config.adam_beta1,
        constants.OPT_ADAMW_BETA_2: pretrain.optim.config.adam_beta2,
        constants.OPT_ADAMW_EPSILON: pretrain.optim.config.adam_eps,
        constants.OPT_ADAMW_WEIGHT_DECAY: pretrain.optim.config.weight_decay,
        constants.OPT_GRADIENT_CLIP_NORM: pretrain.optim.config.clip_grad,

        # Schedulers
        constants.OPT_END_LR: pretrain.optim.lr_scheduler.min_lr,
        constants.OPT_LR_WARMUP_STEPS: pretrain.optim.lr_scheduler.warmup_steps,
        constants.OPT_LR_DECAY_STEPS: pretrain.trainer.max_steps - pretrain.optim.lr_scheduler.warmup_steps,
        constants.OPT_LR_DECAY_SCHEDULE: "cosine with linear warmup",
    }

    # Override config for MLPerf
    pretrain.trainer.num_sanity_val_steps = 0

    run_plugins = [
        plugins.PerfEnvPlugin(),
    ]

    exp_prefix = f"{exp_prefix}{args.tag}"

    # Pretrain data index builder
    # max steps
    pretrain.data.num_train_samples = pretrain.trainer.max_steps * pretrain.data.global_batch_size
    datamodule = pretrain.data.clone()
    datamodule.num_dataset_builder_threads = 32
    build_data_index = run.Partial(
        build_pretraining_datamodule,
        datamodule=datamodule,
        trainer_max_steps=pretrain.trainer.max_steps,
        trainer_val_check_interval=pretrain.trainer.val_check_interval,
        trainer_limit_val_batches=pretrain.trainer.limit_val_batches,
        trainer_limit_test_batches=pretrain.trainer.limit_test_batches,
    )
    data_index_executor = executor.clone()
    data_index_executor.launcher = None
    data_index_executor.nodes = 1
    data_index_executor.ntasks_per_node = 1
    data_index_executor.retries = 1
    data_index_executor.time = "01:00:00"

    static_read_from_path = args.initial_ckpt_path if args.use_ckpt else None
    static_write_to_path = args.continual_ckpt_path
    static_max_steps = args.max_steps if args.max_steps is not None else None

    if not args.save_ckpt:
        pretrain.trainer.enable_checkpointing = False

    original_callbacks = pretrain.trainer.callbacks

    random_seeds = args.seeds
    if len(random_seeds) < args.num_exps:
        import random
        random_seeds = random_seeds + [random.randint(0, 32767) for _ in range(args.num_exps - len(random_seeds))]
        print(f"Missing {args.num_exps - len(random_seeds)} seeds, padding the random seeds to {random_seeds}")

    random_seeds = random_seeds[:args.num_exps]

    for index, seed in enumerate(random_seeds):
        # sets the seeds
        pretrain.data.seed = seed
        build_data_index.datamodule.seed = seed
        configs[constants.SEED] = seed

        exp_name = f"{exp_prefix}_{index}_seed_{seed}"
        experiment_read_from_path = static_read_from_path
        experiment_write_to_path = static_write_to_path
        experiment_max_steps = args.ckpt_start_step

        with run.Experiment(exp_name) as exp:
            exp.add(build_data_index, executor=data_index_executor, name=f"build_data_index")
            
            for j in range(args.num_pars):
                ending_steps = ""
                starting_steps = f"{experiment_max_steps}"
                if static_max_steps is not None:
                    ending_steps = f"-{experiment_max_steps + static_max_steps}-steps"

                checkpoint_name = "checkpoint" + f"-seed-{seed}-par-{j}{ending_steps}"
                experiment_write_to_path = static_write_to_path + "/" + checkpoint_name
                
                if not (args.resume_from_hf and j == 0):
                    pretrain.resume = run.Config(
                        nl.AutoResume,
                        resume_if_exists=True,
                        resume_ignore_no_checkpoint=True,
                        resume_from_path = experiment_read_from_path,
                        resume_from_directory = experiment_read_from_path,
                    )
                else:
                    pretrain.resume = run.Config(nl.AutoResume, restore_config = run.Config(nl.RestoreConfig, path=experiment_read_from_path))
                pretrain.log.ckpt.train_time_interval = None

                if args.save_ckpt:
                    pretrain.log.ckpt.dirpath = experiment_write_to_path
                    pretrain.log.ckpt.filename = "checkpoint"

                if static_max_steps is not None:
                    start_step = experiment_max_steps
                    experiment_max_steps += static_max_steps
                    configs[constants.INIT_CHECKPOINT_STEP] = start_step
                    pretrain.trainer.callbacks = (
                        original_callbacks + [
                            run.Config(PreemptiveStop, stop_on_step=experiment_max_steps),
                            run.Config(
                                MLPerfCallback, 
                                global_batch_size=args.gbs,
                                micro_batch_size=args.mbs, 
                                sequence_length=8192,
                                init_global_step=start_step,
                                configs=configs,
                            ),
                        ]
                    )

                    pretrain.log.extra_loggers = [
                        run.Config(
                            MetricsLogger, 
                            init_global_step=start_step,
                            global_batch_size=args.gbs,
                            seq_length=8192,
                            target_log_ppl=args.target_log_ppl,
                            train_step_time_atol=args.step_time_atol,
                        ), 
                    ]

                    if args.save_ckpt:
                        pretrain.log.ckpt.every_n_train_steps = experiment_max_steps
                        pretrain.log.ckpt.save_on_train_epoch_end = False

                experiment_read_from_path = experiment_write_to_path + "/checkpoint"

                exp.add(
                    pretrain, executor=executor, 
                    name=f"{exp_name}_{j}_{starting_steps}{ending_steps}",
                    plugins=run_plugins
                )

            if args.dryrun:
                exp.dryrun()
            else:
                exp.run(sequential=True, detach=True)
