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

import argparse
import math
import os
from typing import Optional

import torch

from megatron.bridge.recipes.deepseek import deepseek_v3_pretrain_config
from megatron.bridge.training.config import GPTDatasetConfig, ConfigContainer
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain

from callback import (
    MLPerfLoggingCallback,
    DeltaTimingCallback,
    mllogger,
    install_callbacks,
    register_callback,
)


def get_rank():
    """Get the current process rank."""
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def init_logging():
    """Initialize logging configuration."""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def get_data(seq_length: int, seed):
    """Configure dataset paths and parameters."""
    dataset_path = os.getenv("PREPROCESSED_PATH", "/preproc_data")
    val_test_path = f"{dataset_path}/c4-validation-91205-samples.en_text_document"
    train_datasets = [f"{dataset_path}/c4-train.en_{idx}_text_document" for idx in [6, 7]]
    train_datasets_weights = [50] * 2

    data_paths = [
        (train_datasets, train_datasets_weights),
        ([val_test_path], None),
        ([val_test_path], None)
    ]

    return GPTDatasetConfig(
        dataloader_type="single",
        blend_per_split=data_paths,
        sequence_length=seq_length,
        random_seed=seed,
        num_workers=8,
        path_to_cache="/npy_index",
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
    )


def log_hyperparams(args, mbridge_config: ConfigContainer):
    """Log hyperparameters for MLPerf compliance."""
    bmark = mllogger.constants.DEEPSEEK_V3
    opt_lr_decay_steps = args.max_steps - args.warmup_steps
    mllogger.mlperf_submission_log(bmark)

    # Compute gradient accumulation steps
    tp = args.tensor_parallel_size
    pp = args.pipeline_parallel_size
    cp = args.context_parallel_size
    dp = (args.nodes * args.gpus_per_node) // (tp * pp * cp)
    mini_batch_size = args.gbs // dp
    grad_accumulation_steps = mini_batch_size // args.mbs

    logging_configs = {
        mllogger.constants.SEED: args.seed,
        mllogger.constants.GLOBAL_BATCH_SIZE: args.gbs,
        mllogger.constants.GRADIENT_ACCUMULATION_STEPS: grad_accumulation_steps,
        mllogger.constants.MAX_SEQUENCE_LENGTH: args.sequence_length,
        mllogger.constants.EVAL_SAMPLES: args.gbs * args.eval_batches,
        mllogger.constants.TRAIN_SAMPLES: 1574207408,
        mllogger.constants.INIT_CHECKPOINT_STEP: 0,
        mllogger.constants.OPT_NAME: mllogger.constants.ADAMW,
        mllogger.constants.OPT_BASE_LR: mbridge_config.optimizer.lr,
        mllogger.constants.OPT_ADAMW_BETA_1: mbridge_config.optimizer.adam_beta1,
        mllogger.constants.OPT_ADAMW_BETA_2: mbridge_config.optimizer.adam_beta2,
        mllogger.constants.OPT_ADAMW_EPSILON: mbridge_config.optimizer.adam_eps,
        mllogger.constants.OPT_ADAMW_WEIGHT_DECAY: mbridge_config.optimizer.weight_decay,
        mllogger.constants.OPT_GRADIENT_CLIP_NORM: mbridge_config.optimizer.clip_grad,
        mllogger.constants.OPT_END_LR: args.min_lr,
        mllogger.constants.OPT_LR_WARMUP_STEPS: mbridge_config.scheduler.lr_warmup_iters,
        mllogger.constants.OPT_LR_DECAY_STEPS: opt_lr_decay_steps,
        mllogger.constants.MAX_STEPS: args.max_steps,
        mllogger.constants.OPT_LR_DECAY_SCHEDULE: "cosine with linear warmup",
        "target_accuracy": args.target_log_ppl,
    }

    for key, value in logging_configs.items():
        mllogger.event(key=key, value=value)


def create_config(args):
    """Create the training configuration from arguments."""
    config = deepseek_v3_pretrain_config(
        pipeline_model_parallel_size=args.pipeline_parallel_size,
        virtual_pipeline_parallel_size=args.virtual_pipeline_parallel_size,
    )

    # Model parallelism configuration (hardcoded for DeepSeek V3)
    model_cfg = config.model
    model_cfg.tensor_model_parallel_size = args.tensor_parallel_size
    model_cfg.context_parallel_size = args.context_parallel_size
    model_cfg.expert_model_parallel_size = args.expert_model_parallel_size
    model_cfg.expert_tensor_parallel_size = args.expert_tensor_parallel_size
    model_cfg.sequence_parallel = args.tensor_parallel_size > 1
    model_cfg.seq_length = args.sequence_length
    model_cfg.recompute_modules = args.recompute_modules.split(",") if args.recompute_modules else []
    model_cfg.cuda_graph_implementation = args.cuda_graph_implementation
    model_cfg.cuda_graph_scope = args.cuda_graph_scope.split(",") if args.cuda_graph_scope else []

    # MoE parameters (hardcoded for DeepSeek V3)
    model_cfg.moe_token_dispatcher_type = args.moe_token_dispatcher_type
    model_cfg.moe_grouped_gemm = args.moe_grouped_gemm
    model_cfg.moe_permute_fusion = args.moe_permute_fusion
    model_cfg.moe_router_fusion = args.moe_router_fusion
    model_cfg.moe_router_force_load_balancing = False

    # Training configuration
    train_cfg = config.train
    train_cfg.micro_batch_size = args.mbs
    train_cfg.global_batch_size = args.gbs
    train_cfg.train_iters = args.max_steps

    # Compute eval intervals
    eval_every_n_batches = math.ceil(args.eval_every / args.gbs)
    eval_batches = math.ceil(args.eval_tokens / args.gbs)
    train_cfg.eval_interval = eval_every_n_batches
    train_cfg.eval_iters = eval_batches

    # Optimizer configuration
    optimizer_cfg = config.optimizer
    optimizer_cfg.lr = args.lr
    optimizer_cfg.min_lr = args.min_lr

    # Scheduler configuration
    scheduler_cfg = config.scheduler
    scheduler_cfg.lr_warmup_iters = args.warmup_steps

    # RNG configuration
    rng_cfg = config.rng
    rng_cfg.seed = args.seed

    # Dataset configuration
    config.dataset = get_data(
        seq_length=args.sequence_length,
        seed=args.seed,
    )

    # Checkpoint configuration
    checkpoint_cfg = config.checkpoint
    checkpoint_cfg.load = "/checkpoint"
    checkpoint_cfg.load_optim = False
    checkpoint_cfg.load_rng = False
    checkpoint_cfg.exit_on_missing_checkpoint = True

    # Logger configuration
    logger_cfg = config.logger
    logger_cfg.log_interval = 1

    return config


def get_parser() -> argparse.ArgumentParser:
    """Create argument parser with same structure as llama31 pretrain script."""
    parser = argparse.ArgumentParser(description="DeepSeek V3 Pretraining")
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
    slurm_group.add_argument("--dependencies", nargs="*", help="list of dependencies for the job")
    slurm_group.add_argument("--max_retries", type=int, default=0)
    slurm_group.add_argument("--run_slurm", action="store_true", help="run in slurm executor instead of locally")
    slurm_group.add_argument(
        "--mounts",
        type=str,
        required=True,
        help=(
            "Custom mount paths, formatted as a string of <original>:<mapped>[,<original>:<mapped>], "
            "and should contain one path for /output, dataset path: /preproc_data, /npy_index"
        ))
    slurm_group.add_argument("--envvars", type=str, help="Environment variables to be added", default=None)
    slurm_group.add_argument("--image", type=str, required=True, help="Container image path, either remote or local")

    # Model arguments
    model_group = parser.add_argument_group("Model arguments")
    model_group.add_argument("--tensor_parallel_size", type=int, required=True, help="Tensor parallel size")
    model_group.add_argument("--pipeline_parallel_size", type=int, required=True, help="Pipeline parallel size")
    model_group.add_argument("--virtual_pipeline_parallel_size", type=int, default=None, help="Virtual pipeline parallel size")
    model_group.add_argument("--context_parallel_size", type=int, required=True, help="Context parallel size")
    model_group.add_argument("--expert_model_parallel_size", type=int, required=True, help="Expert model parallel size")
    model_group.add_argument("--expert_tensor_parallel_size", type=int, required=True, help="Expert tensor parallel size")
    model_group.add_argument("--recompute_modules", type=str, help="Recompute modules")
    model_group.add_argument("--cuda_graph_implementation", type=str, help="CUDA graph implementation")
    model_group.add_argument("--cuda_graph_scope", type=str, help="CUDA graph scope")
    model_group.add_argument("--moe_token_dispatcher_type", type=str, help="MoE token dispatcher type", default="alltoall")
    model_group.add_argument("--moe_grouped_gemm", type=bool, help="MoE grouped GEMM", default=True)
    model_group.add_argument("--moe_permute_fusion", type=bool, help="MoE permute fusion", default=False)
    model_group.add_argument("--moe_router_fusion", type=bool, help="MoE router fusion", default=False)
    model_group.add_argument("--moe_router_force_load_balancing", type=bool, help="MoE router force load balancing", default=False)
    model_group.add_argument("--sequence_length", type=int, help="Sequence length", default=4096)


    # Training arguments
    training_group = parser.add_argument_group("Training arguments")
    training_group.add_argument("--gbs", type=int, default=1024, help="Global batch size")
    training_group.add_argument("--mbs", type=int, default=1, help="Micro batch size")
    training_group.add_argument("--lr", type=float, default=2e-4, help="Initial learning rate after warmup.")
    training_group.add_argument("--min_lr", type=float, default=5e-6, help="Minimum learning rate")
    training_group.add_argument('--max_steps', type=int, default=1000, help="Maximum number of steps")
    training_group.add_argument('--warmup_steps', type=int, default=0, help="Number of steps for LR warmup")
    training_group.add_argument("--seed", type=int, default=1234, help="Random seed")
    training_group.add_argument("--eval_check_interval", type=int, default=10, help="Evaluate every N steps")
    training_group.add_argument("--eval_batches", type=int, default=1, help="Evaluate N batches")


    # Experiment management arguments
    experiment_group = parser.add_argument_group("Experiment management arguments")
    experiment_group.add_argument("--dryrun", action="store_true", help="Whether we are launching dryrun or actual runs")
    experiment_group.add_argument("--seeds", type=int, nargs="*", default=[], help="random seeds")
    experiment_group.add_argument("--num_exps", type=int, default=1)
    experiment_group.add_argument("--num_pars", type=int, default=1)
    experiment_group.add_argument("--target_log_ppl", type=float, default=1.0, help="Target log perplexity")
    experiment_group.add_argument("--step_time_atol", type=int, default=1600, help="train step time atol")

    return parser


class ArgsConfig:
    """Configuration object that wraps args to match expected interface."""
    def __init__(self, args):
        self.args = args

        # Create nested config structure matching what callbacks expect
        self.model = type('ModelConfig', (), {
            'global_batch_size': args.gbs,
            'micro_batch_size': args.mbs,
            'pipeline_model_parallel_size': args.pipeline_parallel_size,
            'encoder_seq_length': args.sequence_length,
            'seed': args.seed,
        })()

        self.trainer = type('TrainerConfig', (), {
            'max_steps': args.max_steps,
            'val_check_interval': args.eval_check_interval,
            'limit_val_batches': args.eval_batches,
            'log_every_n_steps': 1,
            'warmup_train_steps': 0,
        })()

        self.custom = type('CustomConfig', (), {
            'target_log_ppl': args.target_log_ppl,
        })()

        self.default_val_check_interval = self.trainer.val_check_interval


def main():
    """Main entry point for DeepSeek V3 pretraining."""
    args = get_parser().parse_args()

    if args.tag and not args.tag.startswith("-"):
        args.tag = "-" + args.tag

    init_logging()
    config = create_config(args)
    cfg = ArgsConfig(args)

    if get_rank() == 0:
        log_hyperparams(args, config)
        mllogger.start(key=mllogger.constants.INIT_START)

    register_callback(DeltaTimingCallback(cfg))
    register_callback(MLPerfLoggingCallback(cfg))
    install_callbacks()

    pretrain(config, forward_step_func=forward_step)


if __name__ == "__main__":
    main()
