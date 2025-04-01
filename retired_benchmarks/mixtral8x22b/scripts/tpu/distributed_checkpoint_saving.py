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

"""Load and save huggingface model into torch_xla distributed checkpoint

Test cmd for gpt2:
   python -m clm.scripts.tpu.distributed_checkpoint_saving model.name_or_path=gpt2 checkpoint_manager_path=/tmp/save/

True cmd for mixtral-8x22b:
   export LOCAL_DIR=/tmp/save
   python -m clm.scripts.tpu.distributed_checkpoint_saving model.name_or_path=mistralai/Mixtral-8x22B-v0.1 checkpoint_manager_path=$LOCAL_DIR
   gsutil -m cp -r $LOCAL_DIR gs://some_bucket/path/to/dir
"""
import torch
import os
import hydra
from omegaconf import OmegaConf, DictConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

from transformers import logging

from transformers import set_seed
import torch_xla.core.xla_model as xm

from torch_xla.experimental.distributed_checkpoint import CheckpointManager
from ...model_utils_tpu import (
    setup_xla,
    prepare_model,
    get_global_batch_size,
)

OmegaConf.register_new_resolver(
    "path_join", lambda output_dir, exp_name: os.path.join(output_dir, exp_name)
)
OmegaConf.register_new_resolver(
    "get_global_batch_size",
    lambda per_device_batch_size: get_global_batch_size(per_device_batch_size),
)

logger = logging.get_logger(__name__)


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(config: DictConfig):
    OmegaConf.resolve(config)
    set_seed(config.seed)

    logger.info("\n\n************** Experiment configuration ***********")
    logger.info(OmegaConf.to_yaml(config))

    setup_xla(config)

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name_or_path,
        add_eos_token=False,
        add_bos_token=False,
        use_fast=False,
    )
    logger.info("model loaded")
    dtype = getattr(torch, config.model.dtype)

    model = AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path,
        cache_dir=config.cache_local_dir,
        torch_dtype=dtype,
    )
    model = prepare_model(model, config)
    model = model.to(dtype)

    torch.distributed.init_process_group("gloo", init_method="xla://")
    if config.checkpoint_manager_path:
        ckpt_manager = CheckpointManager(
            path=config.checkpoint_manager_path,
            save_interval=1,
            max_to_keep=1,
        )

        state_dict = {
            "model": model.state_dict(),
        }
        logger.info("saved model.state_dict:")
        for k, v in state_dict["model"].items():
            logger.info(f"{k}: {v.dtype} {v.mean()}")

        ckpt_manager.save(0, state_dict)
    else:
        raise ValueError("need valid {config.checkpoint_manager_path=}")

    logger.info("checkpoing saving finished.")


if __name__ == "__main__":
    main()
