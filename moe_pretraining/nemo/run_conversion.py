#!/usr/bin/env python3

# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
NemoRun launcher script for converting DeepSeek V3 671B checkpoint
from HuggingFace format to Megatron-LM format.

Example usage:
    source config_conversion.sh
    bash run_conversion.sh
"""

import logging
import sys

from run_deepseek import get_parser, parse_mounts, parse_envvars, slurm_executor

import nemo_run as run

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONVERSION_SCRIPT = "/workspace/Megatron-Bridge/examples/conversion/hf_megatron_roundtrip_multi_gpu.py"

CONVERSION_ENV_VARS = {
    "HF_HOME": "/tmp/hf_home",
    "NCCL_MNNVL_ENABLE": "0",
}


def main():
    # Split arguments: before '--' are for launcher, after are for conversion script
    if "--" in sys.argv:
        split_idx = sys.argv.index("--")
        launcher_args = sys.argv[1:split_idx]
        convert_args = sys.argv[split_idx + 1:]
    else:
        launcher_args = sys.argv[1:]
        convert_args = []

    parser = get_parser()
    args = parser.parse_args(launcher_args)

    custom_mounts = parse_mounts(args.mounts)
    custom_env_vars = {**CONVERSION_ENV_VARS, **parse_envvars(args.envvars)}

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
    )

    convert_args_str = " ".join(convert_args)
    inline_script = f"python {CONVERSION_SCRIPT} {convert_args_str}"

    nemorun_script = run.Script(inline=inline_script)

    exp_name = args.exp_name or "deepseek_v3_hf_to_megatron_conversion"

    logger.info(f"Launching conversion: {inline_script}")

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
        logger.info("Conversion completed.")


if __name__ == "__main__":
    main()
