# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass, field
from typing import Optional
import os
from datasets import load_dataset
from mlperf_logging_utils import LoraLogger, MLPerfCallback
from transformers import HfArgumentParser, Trainer, TrainingArguments
from utils import create_and_prepare_model, peft_module_casting_to_bf16


@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """

    local_rank: Optional[int] = field(
        default=-1, metadata={"help": "Used for multi-gpu"}
    )
    per_device_train_batch_size: Optional[int] = field(default=1)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=1)
    learning_rate: Optional[float] = field(default=2e-4)
    max_grad_norm: Optional[float] = field(default=0.0)
    weight_decay: Optional[float] = field(default=0.001)
    lora_alpha: Optional[int] = field(default=32)
    lora_dropout: Optional[float] = field(default=0.1, metadata={"help": "lora dropout is a fixed to 0.1 in closed submission"})
    lora_r: Optional[int] = field(default=16, metadata={"help": "lora rank is a fixed to 16 in closed submission"})
    lora_target_modules: Optional[str] = field(
        default=None,
        metadata={
            "help": "comma separated list of target modules to apply LoRA layers to"
        },
    )
    max_seq_length: Optional[int] = field(default=8192)
    model_path: Optional[str] = field(
        default="./llama-v2-fused-qkv",
        metadata={"help": "Path to the model directory."},
    )
    dataset_path: Optional[str] = field(
        default="./dataset.npy",
        metadata={"help": "The path to the downloaded dataset."},
    )
    config_path: Optional[str] = field(
        default="./configs/default_config.yaml",
        metadata={"help": "path to model config"},
    )
    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    fp16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables fp16 training."},
    )
    bf16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables bf16 training."},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        default="adamw_torch",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: str = field(
        default="cosine",
        metadata={
            "help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"
        },
    )
    max_steps: int = field(
        default=-1, metadata={"help": "How many optimizer update steps to take"}
    )
    warmup_ratio: float = field(
        default=0.03, metadata={"help": "Fraction of steps to do a warmup for"}
    )
    save_steps: int = field(
        default=10, metadata={"help": "Save checkpoint every X updates steps."}
    )
    eval_steps: int = field(default=22, metadata={"help": "Eval model every X steps."})
    logging_steps: int = field(
        default=10, metadata={"help": "Log every X updates steps."}
    )
    target_eval_loss: float = field(
        default=0.92, metadata={"help": "target eval loss - NOT FINAL."}
    )
    output_dir: str = field(
        default="results", metadata={"help": "Where to store the final model."}
    )
    use_flash_attn: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables Flash attention for training."},
    )
    use_peft_lora: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables PEFT LoRA for training."},
    )
    use_gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables Gradient Checkpointing."},
    )
    push_to_hub: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, pushes the model to the HF Hub"},
    )
    num_workers: int = field(
        default=4, metadata={"help": "Number of dataset workers to use."}
    )
    debug: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If True, tests things like proper saving/loading/logging of model"
        },
    )
    dataset_config_name: Optional[str] = field(default="gov_report")
    hub_model_id: Optional[str] = field(default=None)
    seed: Optional[int] = field(default=42)


def main(args):
    loralogger = LoraLogger(target_eval_loss=args.target_eval_loss)
    gbs=args.per_device_train_batch_size * args.gradient_accumulation_steps * int(os.getenv("WORLD_SIZE", 1))
    training_arguments = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim=args.optim,
        learning_rate=args.learning_rate,
        fp16=args.fp16,
        bf16=args.bf16,
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        num_train_epochs=args.num_train_epochs,
        evaluation_strategy="steps",
        save_strategy="no",
        max_steps=args.max_steps,
        eval_steps=args.eval_steps,
        eval_delay=int(0.125*gbs+2)*args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        push_to_hub=args.push_to_hub,
        gradient_checkpointing=args.use_gradient_checkpointing,
        hub_model_id=args.hub_model_id,
        report_to="tensorboard",
        seed=args.seed,
    )

    model = create_and_prepare_model(args)
    model.config.use_cache = False

    # datasets
    ## ToDo uncomment once drive goes public
    # train_url = "https://drive.google.com/file/d/1-JgY1mEafcJ7qhggt6UR3OEKAciIPd5s/view?usp=sharing"
    # eval_url =  "https://drive.google.com/file/d/1jrm6Lacrq49AYv0uB_Qy22xRmfPixQvs/view?usp=sharing"
    # dataset = load_dataset("parquet", data_files={'train': train_url, 'validation': eval_url})
    dataset = load_dataset(
        "parquet",
        data_files={
            "train": f"{args.dataset_path}/train-00000-of-00001.parquet",
            "validation": f"{args.dataset_path}/validation-00000-of-00001.parquet",
        },
    )
    train_dataset, eval_dataset = dataset["train"], dataset["validation"]

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[MLPerfCallback(loralogger, len(train_dataset), len(eval_dataset),args.lora_alpha)],
    )
    trainer.accelerator.print(f"{trainer.model}")
    if args.use_peft_lora:
        trainer.model.print_trainable_parameters()

    if args.use_peft_lora:
        peft_module_casting_to_bf16(trainer.model, args)

    trainer.train()


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    main(args)
