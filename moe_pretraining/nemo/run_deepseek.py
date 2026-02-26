#!/usr/bin/env python3

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

"""
NemoRun launcher script for DeepSeek V3 pretraining.

This script uses NemoRun's Slurm executor to launch pretrain_deepseek_v3_671b.py
on a Slurm cluster.

Example usage:
    python run_deepseek.py \
        --account my_account \
        --partition my_partition \
        --nodes 64 \
        --gpus_per_node 4 \
        --time_limit 04:00:00 \
        --container_image nvcr.io/nvidia/nemo:dev \
        --log_dir /path/to/logs \
        --mounts /data:/data,/checkpoints:/checkpoints \
        -- \
        --tensor_parallel_size 1 \
        --pipeline_parallel_size 4 \
        --context_parallel_size 1 \
        --expert_model_parallel_size 64 \
        --expert_tensor_parallel_size 1 \
        --gbs 2048 \
        --max_steps 1000
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import nemo_run as run
from nemo_run.config import get_nemorun_home, set_nemorun_home
from nemo_run.core.execution.launcher import SlurmTemplate


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent.resolve()
PRETRAIN_SCRIPT = "pretrain_deepseek_v3_671b.py"

# Inline bash template for Slurm
INLINE_TEMPLATE = r"""
#!/usr/bin/env bash
set -euo pipefail

bash -c '{{ pre_cmds }} {{ command }}'
"""

# Default environment variables for performance
PERF_ENV_VARS = {
    "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
    "TRANSFORMERS_OFFLINE": "0",
    "TOKENIZERS_PARALLELISM": "False",
    "NCCL_NVLS_ENABLE": "0",
    "NVTE_NORM_FWD_USE_CUDNN": "1",
    "NVTE_NORM_BWD_USE_CUDNN": "1",
    "TORCH_NCCL_HIGH_PRIORITY": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}


def slurm_executor(
    gpu: str,
    account: str,
    partition: str,
    log_dir: str,
    nodes: int,
    num_gpus_per_node: int,
    time_limit: str = "04:00:00",
    container_image: str = "nvcr.io/nvidia/nemo:dev",
    custom_mounts: List[str] = None,
    custom_env_vars: Dict[str, str] = None,
    custom_srun_args: List[str] = None,
    gres: Optional[str] = None,
    segment: Optional[int] = None,
) -> run.SlurmExecutor:
    """
    Create a Slurm executor for DeepSeek V3 pretraining.

    Args:
        gpu: GPU type (e.g., "gb200", "gb300", "h100")
        account: Slurm account
        partition: Slurm partition
        log_dir: Directory for logs
        nodes: Number of nodes
        num_gpus_per_node: GPUs per node
        time_limit: Job time limit
        container_image: Container image to use
        custom_mounts: Additional container mounts
        custom_env_vars: Additional environment variables
        custom_srun_args: Additional srun arguments
        gres: GPU resource specification
        segment: Slurm segment size (auto-computed for GB200/GB300 if not provided)
    """
    custom_mounts = custom_mounts or []
    custom_env_vars = custom_env_vars or {}
    custom_srun_args = custom_srun_args or []
    custom_bash_cmds = []

    mounts = []
    srun_args = custom_srun_args.copy() + [
        "--mpi=pmix",
        "--no-container-mount-home",
    ]

    if log_dir is not None:
        set_nemorun_home(log_dir)
    else:
        if os.environ.get("NEMORUN_HOME") is None:
            logger.warning(
                f"Logs will be written to {get_nemorun_home()}. "
                "Set NEMORUN_HOME or use --log_dir to change this."
            )

    env_vars = PERF_ENV_VARS.copy()

    # GPU-specific settings
    if gpu.lower() in ["gb200", "gb300"]:
        env_vars["NCCL_NET_GDR_LEVEL"] = "PHB"
        env_vars["NCCL_NET_GDR_C2C"] = "1"

    env_vars.update(custom_env_vars)
    mounts.extend(custom_mounts)

    # Mount the script directory
    mounts.append(f"{SCRIPT_DIR}:{SCRIPT_DIR}")

    # Compute segment for GB200/GB300 if not explicitly provided
    if segment is None and num_gpus_per_node == 4:
        if nodes <= 18:
            segment = nodes
        else:
            for segment_candidate in range(18, 0, -1):
                if nodes % segment_candidate == 0:
                    segment = segment_candidate
                    break

    # NUMA binding
    numa_divisor = 2 if gpu.lower() in ["gb200", "gb300"] else 4
    numa_cmd = f"numactl --cpunodebind=$((SLURM_LOCALID/{numa_divisor})) --membind=$((SLURM_LOCALID/{numa_divisor}))"
    custom_bash_cmds.append(numa_cmd)

    launcher = SlurmTemplate(
        template_inline=INLINE_TEMPLATE,
        template_vars={"pre_cmds": " ; ".join(custom_bash_cmds)},
    )

    executor = run.SlurmExecutor(
        account=account,
        partition=partition,
        tunnel=run.LocalTunnel(job_dir=os.path.join(get_nemorun_home(), "experiments")),
        nodes=nodes,
        ntasks_per_node=num_gpus_per_node,
        gres=gres,
        container_image=container_image,
        container_mounts=mounts,
        env_vars=env_vars,
        srun_args=srun_args,
        time=time_limit,
        mem="0",
        exclusive=True,
        packager=run.GitArchivePackager(),
        segment=segment,
        launcher=launcher,
    )

    return executor


def get_parser() -> argparse.ArgumentParser:
    """Create argument parser for the launcher."""
    parser = argparse.ArgumentParser(
        description="NemoRun launcher for DeepSeek V3 pretraining",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Slurm configuration
    slurm_group = parser.add_argument_group("Slurm configuration")
    slurm_group.add_argument("--account", type=str, required=True, help="Slurm account")
    slurm_group.add_argument("--partition", type=str, required=True, help="Slurm partition")
    slurm_group.add_argument("--nodes", type=int, required=True, help="Number of nodes")
    slurm_group.add_argument("--gpus_per_node", type=int, default=4, help="GPUs per node")
    slurm_group.add_argument("--time_limit", type=str, default="04:00:00", help="Job time limit")
    slurm_group.add_argument("--gpu", type=str, default="gb300", help="GPU type (gb200, gb300, h100)")
    slurm_group.add_argument("--gres", type=str, default=None, help="GPU resource specification")
    slurm_group.add_argument("--segment", type=int, default=None, help="Slurm segment size (auto-computed for GB200/GB300 if not set)")

    # Container configuration
    container_group = parser.add_argument_group("Container configuration")
    container_group.add_argument(
        "--container_image",
        type=str,
        default="nvcr.io/nvidia/nemo:dev",
        help="Container image"
    )
    container_group.add_argument(
        "--mounts",
        type=str,
        default="",
        help="Container mounts (comma-separated, format: src:dst)"
    )
    container_group.add_argument(
        "--envvars",
        type=str,
        default="",
        help="Environment variables (comma-separated, format: KEY=VALUE)"
    )

    # Logging configuration
    log_group = parser.add_argument_group("Logging configuration")
    log_group.add_argument("--log_dir", type=str, default=None, help="Log directory")
    log_group.add_argument("--exp_name", type=str, default=None, help="Experiment name")

    # Execution control
    exec_group = parser.add_argument_group("Execution control")
    exec_group.add_argument("--dryrun", action="store_true", help="Print command without executing")
    exec_group.add_argument("--detach", action="store_true", help="Detach after submitting job")

    return parser


def parse_mounts(mounts_str: str) -> List[str]:
    """Parse comma-separated mounts string."""
    if not mounts_str:
        return []
    return [m.strip() for m in mounts_str.split(",") if m.strip()]


def parse_envvars(envvars_str: str) -> Dict[str, str]:
    """Parse comma-separated environment variables string."""
    if not envvars_str:
        return {}
    result = {}
    for item in envvars_str.split(","):
        if "=" in item:
            key, value = item.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def main():
    """Main entry point."""
    # Split arguments: before '--' are for launcher, after are for pretrain script
    if "--" in sys.argv:
        split_idx = sys.argv.index("--")
        launcher_args = sys.argv[1:split_idx]
        pretrain_args = sys.argv[split_idx + 1:]
    else:
        launcher_args = sys.argv[1:]
        pretrain_args = []

    parser = get_parser()
    args = parser.parse_args(launcher_args)

    # Parse mounts and env vars
    custom_mounts = parse_mounts(args.mounts)
    custom_env_vars = parse_envvars(args.envvars)

    # Create executor
    executor = slurm_executor(
        gpu=args.gpu,
        account=args.account,
        partition=args.partition,
        log_dir=args.log_dir,
        nodes=args.nodes,
        num_gpus_per_node=args.gpus_per_node,
        time_limit=args.time_limit,
        container_image=args.container_image,
        custom_mounts=custom_mounts,
        custom_env_vars=custom_env_vars,
        gres=args.gres,
        segment=args.segment,
    )

    # Build the pretrain script path
    pretrain_script_path = SCRIPT_DIR / PRETRAIN_SCRIPT
    if not pretrain_script_path.is_file():
        logger.error(f"Pretrain script not found: {pretrain_script_path}")
        sys.exit(1)

    # Create NemoRun script
    nemorun_script = run.Script(
        path=str(pretrain_script_path),
        entrypoint="python",
        env={"PYTHONPATH": f"{SCRIPT_DIR}:$PYTHONPATH"},
        args=pretrain_args,
    )

    # Generate experiment name
    exp_name = args.exp_name or f"deepseek_v3_{args.nodes}x{args.gpus_per_node}gpu"

    logger.info(f"Launching: {' '.join(nemorun_script.to_command())}")

    # Run the experiment
    run.run(
        nemorun_script,
        executor=executor,
        dryrun=args.dryrun,
        detach=args.detach,
        name=exp_name,
    )

    if args.dryrun:
        logger.info("Dryrun complete. No job submitted.")
    elif args.detach:
        logger.info(f"Job submitted. Experiment name: {exp_name}")
    else:
        logger.info("Job completed.")


if __name__ == "__main__":
    main()
