# LoRA benchmark

LoRA benchmark on GPU (Nvidia A100 80GB). Inspired by [this blog post](https://medium.com/@sourabmangrulkar/falcon-180b-finetuning-using-peft-and-deepspeed-b92643091d99) and [this script](https://github.com/pacman100/DHS-LLM-Workshop/blob/main/chat_assistant/training/train.py).


## Setup

Run the following:
```bash
sudo ./run_docker.sh
cd lora
pip install -r requirements.txt
```

> The Docker run command contains `-v /home/regis_huggingface_co/workspace:/root/workspace --workdir /root/workspace`. Feel free to change these flags at your own convenience.

You will also need to run the following to install flash attention:
```
pip install flash-attn==2.1.0 --no-build-isolation
```

> For flash attention, make sure that the following command returns 0:
> ```
> ninja --version >/dev/null && echo $?
> ```
> If not, run
> ```
> pip uninstall -y ninja && pip install ninja
> ```
> and install `flash-attn` again.
> More information [here](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features).

Make sure to have requested permission for donwloading Llama2 weights on the Hugging Face Hub: https://huggingface.co/meta-llama/Llama-2-7b-hf
Then, you will need to be connected to your Hugging Face account with a read token running:
```
huggingface-cli login
```
Finally please install mlperf logger:
```
git clone https://github.com/mlperf/logging.git mlperf-logging
pip install -e mlperf-logging
```
## Download Data and Model
data can be downloaded from:
[mlperf drive - train data](https://drive.google.com/file/d/1-JgY1mEafcJ7qhggt6UR3OEKAciIPd5s/view?usp=sharing)
[mlperf drive - validation data](https://drive.google.com/file/d/1jrm6Lacrq49AYv0uB_Qy22xRmfPixQvs/view?usp=sharing)
[mlperf drive - llama-v2 model](https://drive.google.com/drive/folders/1sTeuxkPhwkNPKIPFnOLIYCcK53oB3Ypc?usp=sharing)
As defaults the scripts assume the model is under at ```./llama-v2-fused-qkv``` and the both train and validation are under ```dataset``` folder.

## Llama2-70B on 8 devices

Run:
```bash
accelerate launch --config_file configs/default_config.yaml scripts/train.py \
--dataset_path "./dataset" \
--model_path "/software/users/ihubara/lora_clean/llama-v2-fused-qkv" \
--max_seq_len 8192 \
--bf16 True \
--logging_steps 24 \
--eval_steps 48 \
--output_dir "./results/llama-70b_scrolls_gov_report_r16_$1" \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 1 \
--lr_scheduler_type "cosine" \
--learning_rate 4e-4 \
--weight_decay 0.0001 \
--warmup_ratio 0 \
--max_grad_norm 0.3 \
--use_gradient_checkpointing True \
--target_eval_loss 0.925 \
--use_peft_lora True \
--lora_r 16 \
--lora_alpha 32 \
--lora_dropout 0.1 \
--max_steps 1024 \
--use_flash_attn \
--seed 1234 \
--lora_target_modules "qkv_proj,o_proj"
```
where the Accelerate config file is [this one](https://github.com/regisss/lora/blob/main/configs/default_config.yaml).

> Using flash attention with `--use_flash_attn` is necessary for training on 8k-token sequences.

Learning curves of such a run can be found here: https://huggingface.co/regisss/test_5/tensorboard


## Evaluation

To run evaluation for summarizing texts, you can run:
- Without LoRA adapter weights:
   ```
   python scripts/eval.py --model_name meta-llama/Llama-2-70b-hf --max_new_tokens 900 --seq_length 8192 --do_sample --dataset_name "tau/scrolls" --dataset_config_name "gov_report"
   ```
- With LoRA adapter weights:
   ```
   python scripts/eval.py --peft_model_name path_to_my_lora_model --max_new_tokens 900 --seq_length 8192 --do_sample --dataset_name "tau/scrolls" --dataset_config_name "gov_report"
   ```
## expected outcome

A clean output (train and eval loss) of a singel run with 440 steps can be found under 
```
   convergence_example.txt
```