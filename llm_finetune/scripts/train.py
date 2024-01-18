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
from pathlib import Path
import os
import subprocess
from typing import Optional
import os
from transformers import HfArgumentParser, TrainingArguments, Trainer
from transformers.modeling_utils import unwrap_model
from utils import (
    create_and_prepare_model,
    create_datasets,
    SaveDeepSpeedPeftModelCallback,
    peft_module_casting_to_bf16,
)

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """

    local_rank: Optional[int] = field(
        default=-1, metadata={"help": "Used for multi-gpu"}
    )

    per_device_train_batch_size: Optional[int] = field(default=4)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=4)
    learning_rate: Optional[float] = field(default=2e-4)
    max_grad_norm: Optional[float] = field(default=0.3)
    weight_decay: Optional[float] = field(default=0.001)
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64)
    lora_target_modules: Optional[str] = field(
        default=None,
        metadata={
            "help": "comma separated list of target modules to apply LoRA layers to"
        },
    )
    max_seq_length: Optional[int] = field(default=512)
    model_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )

    dataset_name: Optional[str] = field(
        default="tau/scrolls",
        metadata={"help": "The preference dataset to use."},
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
        default="paged_adamw_32bit",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: str = field(
        default="constant",
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
    eval_steps: int = field(default=10, metadata={"help": "Eval model every X steps."})
    logging_steps: int = field(
        default=10, metadata={"help": "Log every X updates steps."}
    )
    output_dir: str = field(
        default="results", metadata={"help": "Where to store the final model."}
    )
    use_flash_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables Flash attention for training."},
    )
    use_peft_lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables PEFT LoRA for training."},
    )
    use_gradient_checkpointing: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables Gradient Checkpointing."},
    )
    dataset_text_field: str = field(
        default="text", metadata={"help": "Dataset field to use as input text."}
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
    # training arguments
    is_deepspeed_peft_enabled = (
        os.environ.get("ACCELERATE_USE_DEEPSPEED", "False").lower() == "true"
        and args.use_peft_lora
    )
    save_strategy = "steps"
    training_arguments = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim=args.optim,
        learning_rate=args.learning_rate,
        fp16=args.fp16,
        bf16=args.bf16,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        num_train_epochs=args.num_train_epochs,
        evaluation_strategy="steps",
        save_strategy=save_strategy,
        max_steps=args.max_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        push_to_hub=args.push_to_hub,
        gradient_checkpointing=args.use_gradient_checkpointing,
        hub_model_id=args.hub_model_id,
        report_to="tensorboard",
        seed=args.seed,
    )

    # model
    model, peft_config, tokenizer = create_and_prepare_model(args)
    model.config.use_cache = False

    # datasets
    train_dataset, eval_dataset = create_datasets(tokenizer, args)


    # trainer
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.accelerator.print(f"{trainer.model}")
    if args.use_peft_lora:
        trainer.model.print_trainable_parameters()

    if is_deepspeed_peft_enabled:
        trainer.add_callback(
            SaveDeepSpeedPeftModelCallback(trainer, save_steps=args.save_steps)
        )

    if args.use_peft_lora:
        peft_module_casting_to_bf16(trainer.model, args)

    # train
    trainer.train()

    # Save the PEFT adapter on main process
    if trainer.args.process_index == 0:
        if args.push_to_hub:
            print("Push to hub...")
            trainer.push_to_hub()
            if args.use_peft_lora:
                trainer.model.push_to_hub(args.output_dir)
        else:
           print("Save model...")
           unwrap_model(trainer.model).save_pretrained(args.output_dir)


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    main(args)
