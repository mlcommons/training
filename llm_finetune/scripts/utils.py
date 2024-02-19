import random
import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset
from tqdm import tqdm
import warnings
from peft import LoraConfig, get_peft_model,prepare_model_for_kbit_training
import yaml
from deepspeed.accelerator import get_accelerator


from peft.tuners.lora import LoraLayer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl,
)
from transformers.utils import (
    is_sagemaker_mp_enabled,
    is_apex_available,
    #get_accelerator,
)
if is_apex_available():
    from apex import amp

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from .trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

from itertools import chain
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

class SaveDeepSpeedPeftModelCallback(TrainerCallback):
    def __init__(self, trainer, save_steps=500):
        self.trainer = trainer
        self.save_steps = save_steps

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if (state.global_step + 1) % self.save_steps == 0:
            self.trainer.accelerator.wait_for_everyone()
            state_dict = self.trainer.accelerator.get_state_dict(self.trainer.deepspeed)
            unwrapped_model = self.trainer.accelerator.unwrap_model(
                self.trainer.deepspeed
            )
            if self.trainer.accelerator.is_main_process:
                unwrapped_model.save_pretrained(args.output_dir, state_dict=state_dict)
            self.trainer.accelerator.wait_for_everyone()
        return control


class ConstantLengthDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
        Args:
            tokenizer (Tokenizer): The processor used for proccessing the data.
            dataset (dataset.Dataset): Dataset with text files.
            infinite (bool): If True the iterator is reset after dataset reaches end else stops.
            seq_length (int): Length of token sequences to return.
            num_of_sequences (int): Number of token sequences to keep in buffer.
            chars_per_token (int): Number of characters per token used to estimate number of tokens in text buffer.
            shuffle (bool): If true, the samples in each buffer are suffled. Default is `True`.
            add_eos_token (bool): If true, each buffer is delimited with eos token. Default is `True`.
    """

    def __init__(
        self,
        tokenizer,
        dataset,
        infinite=False,
        seq_length=1024,
        num_of_sequences=1024,
        chars_per_token=3.6,
        content_field="content",
        shuffle=True,
        add_eos_token=True,
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.eos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.infinite = infinite
        self.current_size = 0
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
        self.content_field = content_field
        self.shuffle = shuffle
        self.add_eos_token = add_eos_token

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    buffer.append(next(iterator)[self.content_field])
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                    else:
                        more_examples = False
                        break

            tokenized_inputs = self.tokenizer(buffer, truncation=False)["input_ids"]
            all_token_ids = []
            for tokenized_input in tokenized_inputs:
                if self.add_eos_token:
                    tokenized_input = tokenized_input + [self.concat_token_id]
                all_token_ids.extend(tokenized_input)
            examples = []
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    examples.append(input_ids)
            if self.shuffle:
                random.shuffle(examples)
            for example in examples:
                self.current_size += 1
                yield {
                    "input_ids": torch.LongTensor(example),
                    "labels": torch.LongTensor(example),
                }


def chars_token_ratio(dataset, tokenizer, data_column, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        total_characters += len(example[data_column])
        total_tokens += len(tokenizer(example[data_column]).tokens())

    return total_characters / total_tokens

def group_texts(examples, block_size):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    if 'labels' not in result:
        result["labels"] = result["input_ids"].copy()
    return result


def create_datasets(tokenizer, args):
    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config_name,
        use_auth_token=True,
        num_proc=args.num_workers,
    )
    train_dataset = dataset["train"]
    valid_dataset = dataset["validation"]
    column_names = train_dataset.features

    def tokenize_function(example,eval=False):
        output_texts = []
        mask_labels_sizes=[]
        for i in range(len(example["input"])):
            if 'gov_report' in args.dataset_config_name:
                output_texts.append(
                    f"### Summarize the following text:\n {example['input'][i]}\n ### Summary:\n {example['output'][i]}{tokenizer.eos_token}"
                )
                if eval:
                    mask_labels_sizes.append(f"### Summarize the following text:\n {example['input'][i]}\n ### Summary:\n")
            else:
                output_texts.append(
                    f"### {example['input'][i]}\n ### The answer is:\n {example['output'][i]}{tokenizer.eos_token}"
               )
            
        input_ids = tokenizer(output_texts).input_ids

        if eval:
            labels_ids = tokenizer(mask_labels_sizes).input_ids
            masked_labels=[]
            for out,lb in zip(input_ids,labels_ids):
                ml=out.copy()
                ml[:len(lb)]=[-100]*len(lb)
                ml[-1]=-100
                masked_labels.append(ml)
            return {"input_ids": input_ids,"labels": masked_labels}
        else:
            return {"input_ids": input_ids}

    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=8,
        remove_columns=column_names,
    )
    valid_dataset = valid_dataset.map(
        partial(tokenize_function,eval=True),
        batched=True,
        num_proc=2,
        remove_columns=column_names,
    )

    def filter_function(example):
        to_keep = []
        for i in range(len(example["input_ids"])):
            if len(example["input_ids"][i]) > args.max_seq_length:
                to_keep.append(False)
            else:
                to_keep.append(True)
        return to_keep

    train_dataset = train_dataset.filter(
        filter_function,
        batched=True,
        # with_indices=True,
        num_proc=8,
        # remove_columns=column_names,
    )
    valid_dataset = valid_dataset.filter(
        filter_function,
        batched=True,
        # with_indices=True,
        num_proc=2,
        # remove_columns=column_names,
    )
    print(
        f"Before packing, Size of the train set: {len(train_dataset)}. Size of the validation set: {len(valid_dataset)}"
    )
    
    packing_method = partial(group_texts, block_size=args.max_seq_length)
    # Packing
    train_dataset = train_dataset.map(
        packing_method,
        batched=True,
        num_proc=8,
    )
    valid_dataset = valid_dataset.map(
        packing_method,
        batched=True,
        num_proc=2,
    )

    print(
        f"Size of the train set: {len(train_dataset)}. Size of the validation set: {len(valid_dataset)}"
    )

    return train_dataset, valid_dataset

def world_size_from_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    return data['num_machines']*data['num_processes']

def create_and_prepare_model(args):
    device_map = None

    model = AutoModelForCausalLM.from_pretrained(
        'regisss/llama2-70b-fused-qkv-mlperf',
        device_map=device_map,
        use_cache=not args.use_gradient_checkpointing,
        trust_remote_code=True,
        use_flash_attention_2=True if args.use_flash_attn else False,
        torch_dtype=torch.bfloat16,
    )

    peft_config = None
    if args.use_peft_lora:
        peft_config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=None
            if args.lora_target_modules is None
            else args.lora_target_modules.split(","),
        )
        if args.use_gradient_checkpointing:
            model.gradient_checkpointing_enable()
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    return model, peft_config, tokenizer

def training_step(self, model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
    """
    Perform a training step on a batch of inputs.
    Subclass and override to inject custom behavior.
    Args:
        model (`nn.Module`):
            The model to train.
        inputs (`Dict[str, Union[torch.Tensor, Any]]`):
            The inputs and targets of the model.
            The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
            argument `labels`. Check your model's documentation for all accepted arguments.
    Return:
        `torch.Tensor`: The tensor with training loss on this batch.
    """
    model.train()
    inputs = self._prepare_inputs(inputs)
    if is_sagemaker_mp_enabled():
        loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
        return loss_mb.reduce_mean().detach().to(self.args.device)
    with self.compute_loss_context_manager():
        loss = self.compute_loss(model, inputs)
    if self.args.n_gpu > 1:
        loss = loss.mean()  # mean() to average on multi-gpu parallel training
    if self.use_apex:
        with amp.scale_loss(loss, self.optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        self.accelerator.backward(loss)
    get_accelerator().empty_cache()
    return loss.detach() / self.args.gradient_accumulation_steps
    

def create_and_prepare_model_unfuse(args):
    device_map = None

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map=device_map,
        use_cache=not args.use_gradient_checkpointing,
        trust_remote_code=True,
        use_flash_attention_2=True if args.use_flash_attn else False,
        torch_dtype=torch.bfloat16,
    )

    peft_config = None
    if args.use_peft_lora:
        peft_config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=None
            if args.lora_target_modules is None
            else args.lora_target_modules.split(","),
        )
        if args.use_gradient_checkpointing:
            model.gradient_checkpointing_enable()
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    return model, peft_config, tokenizer

def peft_module_casting_to_bf16(model, args):
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if args.bf16:
                module = module.to(torch.bfloat16)
        if "norm" in name:
            module = module.to(torch.float32)
        if any(x in name for x in ["lm_head", "embed_tokens", "wte", "wpe"]):
            if hasattr(module, "weight"):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
