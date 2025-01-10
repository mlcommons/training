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

import hydra
import numpy as np
import torch
from clm_datasets import get_dataset_cuda, get_datasets, process_datasets
from file_utils import get_file
from mlperf_logging_utils import (
    ClmLogger,
    MetricsLogger,
    MLPerfCallback,
    MLPerfLightningCallback,
)
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer, default_data_collator, logging, set_seed

USE_CUDA = torch.cuda.is_available()  # os.environ.get('USE_CUDA', False)

OmegaConf.register_new_resolver("multiply", lambda x, y: x * y, replace=True)

if not USE_CUDA:
    from model_utils_tpu import (
        TensorBoardCallback,
        get_global_batch_size,
        setup_model_optimizer,
        setup_xla,
    )
    from trainer_utils_tpu import Trainer

    OmegaConf.register_new_resolver(
        "get_global_batch_size",
        lambda per_device_batch_size: get_global_batch_size(per_device_batch_size),
    )
    OmegaConf.register_new_resolver(
        "path_join", lambda output_dir, exp_name: os.path.join(output_dir, exp_name)
    )
else:
    import torch.multiprocessing as mp
    from model_utils_gpu import setup_distributed, setup_model_and_trainer
    from nemo import lightning as nl
    from nemo.collections import llm

    OmegaConf.register_new_resolver("int_div", lambda x, y: x // y, replace=True)
    OmegaConf.register_new_resolver(
        "path_join", lambda output_dir, exp_name: os.path.join(output_dir, exp_name)
    )
    OmegaConf.register_new_resolver(
        "get_global_batch_size",
        lambda per_device_batch_size: per_device_batch_size,
    )

    mp.set_start_method("spawn", force=True)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    logger = logging.get_logger(__name__)

    OmegaConf.resolve(config)
    set_seed(config.seed)
    logger.info("\n\n************** Experiment configuration ***********")
    logger.info(OmegaConf.to_yaml(config))
    if USE_CUDA:
        setup_distributed(config)

    if config.eval_frequency == -1:
        config.eval_frequency = int(
            np.ceil(24576 * 2048 / config.max_length / config.global_train_batch_size)
        )
    logger.info(f"{config.eval_frequency=}")

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name_or_path,
        add_eos_token=False,
        add_bos_token=False,
        use_fast=False,
    )

    clmlogger = ClmLogger(config, filename="output.txt")

    if not USE_CUDA:
        config_path = os.path.join(config.run_dir, "config.yaml")
        with get_file(config_path, "w") as f:
            OmegaConf.save(config, f)

        logger.info(f"log tensorboard to {os.path.join(config.run_dir, 'tensorboard')}")
        setup_xla(config)
        model, optimizer, scheduler = setup_model_optimizer(config)

        if tokenizer.vocab_size != model.config.vocab_size:
            logger.warning(
                f"Found mismatch between {tokenizer.vocab_size=} and {model.config.vocab_size}"
            )
        raw_datasets = get_datasets(config)
        datasets = process_datasets(raw_datasets, tokenizer, config)
        train_dataset, eval_dataset = datasets["train"], datasets["validation"]

        # Initialize our Trainer
        trainer = Trainer(
            model=model,
            config=config,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            optimizer=optimizer,
            scheduler=scheduler,
            # Data collator will default to DataCollatorWithPadding, so we change it.
            data_collator=default_data_collator,
            callbacks=[MLPerfCallback(config), TensorBoardCallback(config)],
        )
        trainer.train()

    else:
        if "adam" in config.optimizer.lower():
            optimizer_name = "adam"
        else:
            raise ValueError("Unsupported optimizer for GPU run")

        data_parallel_size = torch.distributed.get_world_size() // (
            config.tensor_parallelism
            * config.pipeline_parallelism
            * config.context_parallelism
        )

        config.global_train_batch_size = int(
            config.per_device_train_batch_size
            * config.gradient_accumulation_steps
            * data_parallel_size
        )

        config.global_eval_batch_size = config.global_train_batch_size
        number_of_nodes = (
            torch.distributed.get_world_size() // torch.cuda.device_count()
        )

        metrics_logger = MetricsLogger(
            clmlogger,
            number_of_nodes,
            config.global_train_batch_size,
            config.lr,
            config.max_length,
        )

        callbacks = [
            MLPerfLightningCallback(
                clmlogger,
                config.global_train_batch_size,
                config.max_length,
            )
        ]

        if (
            "capacity_factor" in config.model
            and config.model.capacity_factor is not None
            and config.model.capacity_factor > 0
        ):
            from nemo.lightning.pytorch.callbacks.moe_token_drop import (
                MegatronTokenDropCallback,
            )

            callbacks.append(
                MegatronTokenDropCallback(
                    moe_expert_capacity_factor=config.model.capacity_factor
                )
            )

        number_of_nodes = max(
            1, torch.distributed.get_world_size() // torch.cuda.device_count()
        )

        model, trainer, optimizer, resume = setup_model_and_trainer(
            model_name_or_path=config.model.name_or_path,
            input_sequence_length=config.max_length,
            global_batch_size=config.global_train_batch_size,
            nodes=number_of_nodes,
            tp_size=config.tensor_parallelism,
            pp_size=config.pipeline_parallelism,
            vpp_size=None,  # config.virtual_pipeline_parallelism,
            cp_size=config.context_parallelism,
            learning_rate=config.lr,
            weight_decay=config.weight_decay,
            optimizer_name=optimizer_name,
            tokenizer_name_or_path=config.model.name_or_path,
            scheduler=config.sched,
            max_grad_norm=config.max_grad_norm,
            eval_frequency=config.eval_frequency,
            log_frequency=config.log_frequency,
            max_steps=config.max_steps,
            logger=metrics_logger,
            callbacks=callbacks,
        )
        ckpt = nl.ModelCheckpoint(
            save_last=False,
            save_top_k=False,
            every_n_train_steps=0,
            always_save_context=False,
            save_context_on_train_end=False,
        )

        nemo_logger = nl.NeMoLogger(
            ckpt=ckpt,
            name="mixtral-reference",
            tensorboard=None,
            wandb=None,
            log_dir="/results",
        )

        dataset = get_dataset_cuda(config)

        llm.train(
            model=model,
            data=dataset,
            trainer=trainer,
            tokenizer="data",
            optim=optimizer,
            log=nemo_logger,
            # log=None,
            resume=resume,
        )


if __name__ == "__main__":
    main()
