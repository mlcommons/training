# 1. Problem
This benchmark use [PaxML](http://github.com/google/paxml) to 
re-implement the GPT-3 model with the best efforts to match
available details from the [paper](https://arxiv.org/abs/2005.14165):

  Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan,
  et. al. Language Models are Few-Shot Learners. 	arXiv:2005.14165 

# 2. Dataset
This benchmark uses the
[C4](https://www.tensorflow.org/datasets/catalog/c4)/en/3.0.1
dataset from TensorFlow Dataset. A version is 
[available](https://huggingface.co/datasets/allenai/c4/tree/main/en) 
from Hugging Face.

The benchmark uses a different split than the original C4/en/3.0.1:

| Split | What's in it | What it is used for | number of samples |
| - | - | - | - |
| train1 | first 768 of 1024 files of C4/en/3.0.1:train | training the initial checkpoint | 274,651,678 |
| train2 | last 256 of 1024 files of C4/en/3.0.1:train | training dataset of the benchmark | 91,217,223 |
| validation\_24567exp | 1/20th of C4/en/3.0.1:validation | validation datset of the benchmark | 24,567 |

The resplit dataset uses 3.0.4 as its version to differenciate from the original
3.0.1 version, and it's available on
[GCS](https://console.cloud.google.com/storage/browser/mlperf-llm-public2/c4/en/3.0.4)

# 3. Model
The model largely follows the GPT-3 paper, with key model architecture configs
listed below:

| Config | Value |
| - | - |
| Number of layers | 96 |
| Number of heads | 96 |
| Model dimension | 12288 |
| Hidden dimension | 12288 * 4 |
| Vocab size | 50257 |
| Input sequence length | 2048 |
 
Some components are different from GPT-3, listed below:

| Component | GPT-3 | PaxML implementation | Why |
| - | - | - | - |
| Tokenizer | BPE | [SentencePiece](https://github.com/google/sentencepiece) with BPE | To handle languages that don't use space to seperate words. |
| Alternating sparse attention layers | Yes | No | Implementation details unavailable |

## Optimizer
Adam

## Initial checkpoint
The benchmark starts from an initial checkpoint trained with the C4/en/3.0.4:train1
split for 4000 steps of global batch size 1536.

The PaxML initial checkpint is available for different pipeline configurations:

| Pipeline | checkpoint |
| - | - |
| No pipeline | https://console.cloud.google.com/storage/browser/mlperf-llm-public2/gpt3_spmd1x64x24_tpuv4-3072_v84_20221101/checkpoints/checkpoint_00004000 |
| 4 stages | https://console.cloud.google.com/storage/browser/mlperf-llm-public2/gpt3_spmd1x64x24_tpuv4-3072_v84_20221101/checkpoints/checkpoint_00004000_pipeline_4stages |
| 8 stages | https://console.cloud.google.com/storage/browser/mlperf-llm-public2/gpt3_spmd1x64x24_tpuv4-3072_v84_20221101/checkpoints/checkpoint_00004000_pipeline |

# 4. Quality
## Quality metric
Log Perplexity

## Quality target
2.69

## Evaluation frequency
24 * 1024 * 2048 tokens

# 5. Steps to run the model
## On Google Cloud TPU

To run the benchmark on [Cloud TPUs](https://cloud.google.com/tpu),
follow these steps:

- Create the TPU VM

```
export TPU_NAME=paxml-tpu

gcloud alpha compute tpus tpu-vm create $TPU_NAME \
--accelerator-type=v4-1536 \
--version=v2-nightly-tpuv4 \
--project=<GCP project> \
--zone=<GCP zone>

```

-   Setup docker credential and pull the docker image on each TPU host:

```
gcloud alpha compute tpus tpu-vm ssh $TPU_NAME --worker=all --command="\
sudo usermod -a -G docker ${USER} && \
docker-credential-gcr configure-docker && |
gcloud docker -- pull gcr.io/${PROJECT}/pax-dev:llm-ref "
```

-   Copy the initial checkpoint to a GCS directory, i.e. "log\_dir".

-   Launch the training job on each TPU host

```
gcloud alpha compute tpus tpu-vm ssh $TPU_NAME --worker=all --command="\
docker run --net=host -d --privileged -i --name=$TPU_NAME \
gcr.io/${PROJECT}/pax-dev:llm-ref \
  bazel run paxml/tasks/lm/params:main -- \
    --exp=c4.C4SpmdPipelineGpt3AdamMLPerfHPBS1p5k768Replicas \
    --job_log_dir=<log_dir> 2>&1 | tee -a ~/logs.txt &"
```

There won't be output to STDOUT except the hash of successfully started docker containers. To view the log on a worker, use

```
gcloud alpha compute tpus tpu-vm ssh $TPU_NAME --worker=1 --command="\
docker logs $TPU_NAME"
```

In case errors are encountered, the following command will try to stop
the docker container of the remaining process, so that a new job can
be launch again:

```
gcloud alpha compute tpus tpu-vm ssh $TPU_NAME --worker=all --command="docker stop $TPU_NAME && docker rm $TPU_NAME"
```

Note: The working copy of the model is in the
[PaxML library](https://github.com/google/paxml/tree/main/paxml/tasks/lm/params). The
files in this repo are a copy of that as of 1-Feb-2023. The PaxML team is
working to allow an easier way to define customer model configs. When such task
is done, the files & repo steps in this repo will be updated accordingly.

