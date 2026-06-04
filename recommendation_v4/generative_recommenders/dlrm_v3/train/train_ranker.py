# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-strict
import argparse
import logging

logging.basicConfig(level=logging.INFO)
import os
import sys
import traceback

import gin
import torch
from torch import multiprocessing as mp
from torchrec.test_utils import get_free_port

# NOTE: heavy imports of generative_recommenders.dlrm_v3.* are deferred to
# inside _main_func so that gin-driven env-var bootstrap (see
# _env_bootstrap.apply_env_bootstrap) can run BEFORE the triton kernel
# modules evaluate their `@triton.autotune` decorators at module-load time.

logger: logging.Logger = logging.getLogger(__name__)


SUPPORTED_CONFIGS = {
    "debug": "debug.gin",
    "kuairand-1k": "kuairand_1k.gin",
    "movielens-1m": "movielens_1m.gin",
    "movielens-20m": "movielens_20m.gin",
    "movielens-13b": "movielens_13b.gin",
    "movielens-18b": "movielens_18b.gin",
    "streaming-400m": "streaming_400m.gin",
    "streaming-200b": "streaming_200b.gin",
    "streaming-100b": "streaming_100b.gin",
    "yambda-5b": "yambda_5b.gin",
}


def _main_func(
    rank: int,
    world_size: int,
    master_port: int,
    gin_file: str,
    mode: str,
) -> None:
    device = torch.device(f"cuda:{rank}")
    logger.info(f"rank: {rank}, world_size: {world_size}, device: {device}")
    # Phase 1: parse gin early with skip_unknown=True so env-bootstrap
    # bindings take effect BEFORE any module-level @gin.configurable
    # discovers itself. This is required because triton @triton.autotune
    # decorators in generative_recommenders.ops.triton.* read env vars at
    # module import time, and the heavy imports below pull those in.
    from generative_recommenders.dlrm_v3.train._env_bootstrap import apply_env_bootstrap

    gin.parse_config_file(gin_file, skip_unknown=True)
    apply_env_bootstrap()

    # Phase 2: heavy imports. Triton kernel modules evaluate their autotune
    # decorators here, using the env vars set above.
    from generative_recommenders.dlrm_v3.checkpoint import load_dmp_checkpoint
    from generative_recommenders.dlrm_v3.train.utils import (
        cleanup,
        eval_loop,
        make_model,
        make_optimizer_and_shard,
        make_train_test_dataloaders,
        setup,
        streaming_train_eval_loop,
        train_eval_loop,
        train_loop,
    )
    from generative_recommenders.dlrm_v3.utils import MetricsLogger

    setup(
        rank=rank,
        world_size=world_size,
        master_port=master_port,
        device=device,
    )
    # Phase 3: re-parse to bind the @gin.configurables now that they are
    # registered. The earlier skip_unknown pass already consumed the
    # env-bootstrap binding, but bindings are idempotent so re-applying is
    # fine, and this pass is the one that actually wires up make_model,
    # make_train_test_dataloaders, etc.
    gin.parse_config_file(gin_file)

    model, model_configs, embedding_table_configs = make_model()
    model, optimizer = make_optimizer_and_shard(
        model=model, device=device, world_size=world_size
    )
    train_dataloader, test_dataloader = make_train_test_dataloaders(
        hstu_config=model_configs,
        embedding_table_configs=embedding_table_configs,
    )
    metrics = MetricsLogger(
        multitask_configs=model_configs.multitask_configs,
        batch_size=train_dataloader.batch_size,
        window_size=2500,
        device=device,
        rank=rank,
    )
    # Capture streaming resume hint (None for cold start / non-streaming
    # checkpoints). For the streaming-train-eval mode, we forward this into
    # streaming_train_eval_loop so it can advance past the last completed
    # window OR re-enter the partial window and skip already-trained batches.
    resume_train_ts, resume_batch_idx_in_window = load_dmp_checkpoint(
        model=model,
        optimizer=optimizer,
        metric_logger=metrics,
        device=device,
        rank=rank,
    )

    # train loop
    try:
        if mode == "train":
            train_loop(
                rank=rank,
                model=model,
                dataloader=train_dataloader,
                optimizer=optimizer,
                metric_logger=metrics,
                device=device,
            )
        elif mode == "eval":
            # reinit metrics logger for eval
            metrics = MetricsLogger(
                multitask_configs=model_configs.multitask_configs,
                batch_size=train_dataloader.batch_size,
                window_size=1000,
                device=device,
                rank=rank,
            )
            eval_loop(
                rank=rank,
                model=model,
                dataloader=test_dataloader,
                metric_logger=metrics,
                device=device,
            )
        elif mode == "train-eval":
            train_eval_loop(
                rank=rank,
                model=model,
                train_dataloader=train_dataloader,
                eval_dataloader=test_dataloader,
                optimizer=optimizer,
                metric_logger=metrics,
                device=device,
            )
        elif mode == "streaming-train-eval":
            streaming_train_eval_loop(
                rank=rank,
                model=model,
                optimizer=optimizer,
                metric_logger=metrics,
                device=device,
                hstu_config=model_configs,
                embedding_table_configs=embedding_table_configs,
                resume_train_ts=resume_train_ts,
                resume_batch_idx_in_window=resume_batch_idx_in_window,
            )
    except Exception as e:
        logger.info(traceback.format_exc())
        cleanup()
        raise Exception(e)


def get_args():  # pyre-ignore [3]
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="debug", choices=SUPPORTED_CONFIGS.keys(), help="dataset"
    )
    parser.add_argument(
        "--mode",
        default="train",
        choices=["train", "eval", "train-eval", "streaming-train-eval"],
        help="mode",
    )
    args, unknown_args = parser.parse_known_args()
    logger.warning(f"unknown_args: {unknown_args}")
    return args


def main() -> None:
    args = get_args()
    logger.info(args)
    assert args.dataset in SUPPORTED_CONFIGS, f"Unsupported dataset: {args.dataset}"
    assert args.mode in [
        "train",
        "eval",
        "train-eval",
        "streaming-train-eval",
    ], f"Unsupported mode: {args.mode}"
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    MASTER_PORT = str(get_free_port())
    gin_path = f"{os.path.dirname(__file__)}/gin/{SUPPORTED_CONFIGS[args.dataset]}"

    mp.start_processes(
        _main_func,
        args=(WORLD_SIZE, MASTER_PORT, gin_path, args.mode),
        nprocs=WORLD_SIZE,
        join=True,
        start_method="spawn",
    )


if __name__ == "__main__":
    main()
