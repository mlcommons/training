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

The benchmark follows the split in C4/en/3.0.1, with 365,868,901 examples
in `train`, and 364,608 examples in `validation`.

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

# 4. Quality
## Quality metric
Log Perplexity

## Quality target
TBD

## Evaluation frequency
TBD

# 5. Steps to run the model
## On Google Cloud TPU

To run the benchmark on [Cloud TPUs](https://cloud.google.com/tpu),
follow these steps:

- Create the TPU VM

```
export TPU_NAME=paxml-tpu

gcloud alpha compute tpus tpu-vm create $TPU_NAME \
--accelerator-type=v4-3072 \
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

-   Launch the training job on each TPU host

```
gcloud alpha compute tpus tpu-vm ssh $TPU_NAME --worker=all --command="\
docker run --net=host -d --privileged -i --name=$TPU_NAME \
gcr.io/${PROJECT}/pax-dev:llm-ref \
  bazel run paxml/tasks/lm/params:main -- \
    --exp=c4.C4SpmdGpt3AdamOrgHP1536Replicas \
    --job_log_dir=gs://<GCP bucket>/gpt3/logs/$(date +%s) 2>&1 | tee -a ~/logs.txt &"
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
[PaxML library](https://github.com/google/paxml/paxml/tasks/lm/params). The
files in this repo are a copy of that as of 14-Jun-2022. The PaxML team is
working to allow an easier way to define customer model configs. When such task
is done, the files & repo steps in this repo will be updated accordingly.
