import argparse
import torch
from dataclasses import dataclass
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM
from peft.config import PeftConfigMixin
from datasets import load_dataset
import evaluate
import nltk
import numpy as np
from tqdm import tqdm
from typing import Any, Dict, List, Union, Optional
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

nltk.download("punkt")

# Arguments management
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name",
    default=None,
    type=str,
    help="Path to pre-trained model (on the HF Hub or locally).",
)
parser.add_argument(
    "--peft_model_name",
    default=None,
    type=str,
    help="Path to PEFT model (on the HF Hub or locally).",
)
parser.add_argument(
    "--max_new_tokens", type=int, default=300, help="Number of tokens to generate."
)
parser.add_argument("--seq_length", type=int, default=8192, help="Sequence length.")
parser.add_argument("--do_sample", action="store_true", help="Wheter to generate doing multinomial sampling.")
parser.add_argument("--dataset_name", type=str, default="tau/scrolls", help= "The preference dataset to use.")
parser.add_argument("--dataset_config_name", type=str, default="gov_report", help= "The preference dataset config to use.")
args = parser.parse_args()

# Instantiate model
if args.peft_model_name is not None:
    model = (
        AutoPeftModelForCausalLM.from_pretrained(
            args.peft_model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        .merge_and_unload()
        .eval()
    )
    base_model_name = PeftConfigMixin.from_pretrained(
        args.peft_model_name
    ).base_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
else:
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

model.generation_config.pad_token_id = model.generation_config.eos_token_id
tokenizer.pad_token = tokenizer.eos_token

# Load dataset
dataset = load_dataset(
        args.dataset_name,
        args.dataset_config_name,
        use_auth_token=True,
        num_proc=4,
        split="validation"
    )
column_names = dataset.features

def tokenize_function(examples):
    output_texts = []
    for i in range(len(examples["input"])):
        output_texts.append(
            f"### Summarize the following text:\n {examples['input'][i]}\n ### Summary:\n "
        )
    input_ids = tokenizer(output_texts).input_ids

    return {"input_ids": input_ids, "ground_truth": examples["output"]}


test_dataset = dataset.map(
    tokenize_function,
    batched=True,
    num_proc=2,
    remove_columns=column_names,
)


def filter_function(examples):
    to_keep = []
    for i in range(len(examples["input_ids"])):
        if len(examples["input_ids"][i]) > args.seq_length - args.max_new_tokens:
            to_keep.append(False)
        else:
            to_keep.append(True)
    return to_keep


test_dataset = test_dataset.filter(
    filter_function,
    batched=True,
    num_proc=2,
)
print(f"Size of the test set: {len(test_dataset)}.")


@dataclass
class CustomDataCollator:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids = [{"input_ids": sample["input_ids"]} for sample in features]
        batch = self.tokenizer.pad(
            input_ids,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch["ground_truth"] = [sample["ground_truth"] for sample in features]
        return batch


dataloader = DataLoader(
    test_dataset,
    batch_size=1,
    collate_fn=CustomDataCollator(tokenizer),
)


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


metric = evaluate.load("rouge")


def compute_metrics(generated, ground_truth):
    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(generated, ground_truth)
    result = metric.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [
        np.count_nonzero(gen != tokenizer.pad_token_id) for gen in generated
    ]
    result["gen_len"] = np.mean(prediction_lens)
    return result


generated_sequences = []
ground_truths = []
for batch in tqdm(dataloader):
    outputs = model.generate(
        inputs=batch["input_ids"].to("cuda"),do_sample=args.do_sample , max_new_tokens=args.max_new_tokens
    )
    outputs = [
        output.split("### Summary:\n ")[-1]
        for output in tokenizer.batch_decode(outputs, skip_special_tokens=True)
    ]

    print("Batch outputs:", outputs)
    print("Batch ground truths:", batch["ground_truth"])
    generated_sequences += outputs
    ground_truths += batch["ground_truth"]
    print("Current results:", compute_metrics(generated_sequences, ground_truths))

res = compute_metrics(generated_sequences, ground_truths)
print("Final results:", res)
