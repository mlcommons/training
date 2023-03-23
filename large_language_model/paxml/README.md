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
export ZONE=<tpu zone>
export VERSION=tpu-vm-v4-base
export PROJECT=<gcp project>
export ACCELERATOR=v4-1536
export TPU_NAME=<tpu name>

gcloud alpha compute tpus tpu-vm create $TPU_NAME \
--accelerator-type=$ACCELERATOR \
--version=$VERSION \
--project=$PROJECT \
--zone=$ZONE
```

Confirm the TPU is created successfully and ssh working:

```
gcloud compute tpus tpu-vm ssh $TPU_NAME --worker=all --command="hostname"
```

- Install required packages:

Prerequests:

```
gcloud compute tpus tpu-vm ssh $TPU_NAME --worker=all --command="\
sudo apt-get update && \
sudo apt-get install -y libcairo2-dev"

gcloud compute tpus tpu-vm ssh $TPU_NAME --worker=all --command="\
python3 -m pip install -U pip && \
python3 -m pip install orbax==0.1.6"
```

Jax:

```
git clone https://github.com/google/jax.git && \
cd jax && \
git checkout bd1f53ed6deace0f05cafc38a2c0d98075b0678f && \
python3 -m pip install .[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html"

gcloud compute tpus tpu-vm ssh $TPU_NAME --worker=all --command="\
python3 -m pip install jaxlib==0.4.7.dev20230322 -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html"

gcloud compute tpus tpu-vm ssh $TPU_NAME --worker=all --command="\
python3 -m pip install libtpu-nightly==0.1.dev20230322 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html"
```

PaxML:

Make `{praxis, paxml}-nightly+20230322-py3-none-any.whl` and 
`{praxis, paxml}_requirements.txt` available under `./`, either:

1. Download them from [gs://mlperf-llm-public2/paxml_wheels](https://console.cloud.google.com/storage/browser/mlperf-llm-public2/paxml_wheels).

2. Build from source using PaxML's [Dockerfile](https://github.com/google/paxml/blob/main/paxml/pip_package/Dockerfile).

Then

```
gcloud compute tpus tpu-vm scp --worker=all ./*_requirements.txt ./*.whl ${TPU_NAME}:

gcloud compute tpus tpu-vm ssh $TPU_NAME --worker=all --command="\
python3 -m pip install -r ./praxis_requirements.txt && \
python3 -m pip install ./praxis-nightly+20230322-py3-none-any.whl"

gcloud compute tpus tpu-vm ssh $TPU_NAME --worker=all --command="\
python3 -m pip install -r ./paxml_requirements.txt && \
python3 -m pip install ./paxml-nightly+20230322-py3-none-any.whl"

gcloud compute tpus tpu-vm ssh $TPU_NAME --worker=all --command="\
python3 -m pip install jaxlib==0.4.7.dev20230322 -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html"
```

MLPerf logging:

```
gcloud compute tpus tpu-vm ssh $TPU_NAME --worker=all --command="\
python3 -m pip install git+https://github.com/mlperf/logging.git"
```

The reference model:

```
gcloud compute tpus tpu-vm ssh $TPU_NAME --worker=all --command="\
git clone -b paxml-llm-draft https://github.com/sgpyc/training.git"
```

-   Copy the initial checkpoint to a GCS directory, i.e. "log\_dir".

-   Launch the training job on each TPU host

```
gcloud compute tpus tpu-vm ssh $TPU_NAME --worker=all --command="\
PYTHONPATH=\$HOME/training/large_language_model/paxml python3 \
\$HOME/.local/lib/python3.8/site-packages/paxml/main.py \
--exp=c4_mllog.C4SpmdPipelineGpt3AdamMLPerfHPBS1p5k768ReplicasMllog \
--job_log_dir=gs://<log_dir> 2>&1 | tee -a ~/logs.txt &"
```

There won't be output to STDOUT except the hash of successfully started docker containers. To view the log on a worker, use

```
gcloud alpha compute tpus tpu-vm ssh $TPU_NAME --worker=0 --command="\
tail ~/logs.txt"
```

To test using a smaller model with TPUv4-16, use the following instead
before creating the TPU.

```
...
export ACCELERATOR=v4-16
...
```

And use the following to launch the job:

```
gcloud compute tpus tpu-vm ssh $TPU_NAME --worker=all --command="\
PYTHONPATH=\$HOME/training/large_language_model/paxml python3 \
\$HOME/.local/lib/python3.8/site-packages/paxml/main.py \
--exp=c4_mllog.C4SpmdPipelineGpt3SmallAdam8ReplicasMllog \
--job_log_dir=gs://<log_dir> 2>&1 | tee -a ~/logs.txt &"
```

# 6. Software versions

Note: The working copy of the {c4, lm_cloud, model_params}.py is in the
[PaxML library](https://github.com/google/paxml/tree/main/paxml/tasks/lm/params). The
files in this repo are a copy of that as of 23-Feb-2023.

| Software | Version |
| -- | -- |
| [Jax](https://github.com/google/jax) | @[bd1f53ed](https://github.com/google/jax/commit/bd1f53ed6deace0f05cafc38a2c0d98075b0678f) |
| jaxlib | 0.4.7.dev20230322 |
| libtpu-nightly | 0.1.dev20230322 |
| [Orbax](https://github.com/google/orbax) | 0.1.6 |
| [Praxis](https://github.com/google/praxis) | praxis-nightly 20230322, @[60dba3d](https://github.com/google/praxis/commit/60dba3d3def8c10d71c11c366a870e7e6a828d7e) |
| [PaxML](https://github.com/google/paxml) | paxml-hibhtly 20230322, @[7079a1d](https://github.com/google/paxml/commit/7079a1dbfe6a14668c699e3aebc26e4445db835c) |
| [MLPerf logging](https://github.com/mlcommons/logging) | @[f6ee121](https://github.com/mlcommons/logging/commit/f6ee121b4eb566b93a8ba16dab775b37468b8ef6) |

