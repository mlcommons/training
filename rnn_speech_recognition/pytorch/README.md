# DISCLAIMER
This codebase is a work in progress. There are known and unknown bugs in the implementation, and has not been optimized in any way.

MLPerf has neither finalized on a decision to add a speech recognition benchmark, nor as this implementationn/architecture as a reference implementation.

# 1. Problem 
Speech recognition accepts raw audio samples and produces a corresponding text transcription.

# 2. Directions

## Steps to configure machine
### From Docker

```
git clone https://github.com/mlcommon/training.git
```
2. Install CUDA and Docker
```
source training/install_cuda_docker.sh
```
3. Build the docker image for the single stage detection task
```
# Build from Dockerfile
cd training/rnn_speech_recognition/pytorch/
bash scripts/docker/build.sh
```

#### Requirements
Currently, the reference uses CUDA-11.0 (see [Dockerfile](Dockerfile#L15)).
Here you can find a table listing compatible drivers: https://docs.nvidia.com/deploy/cuda-compatibility/index.html#binary-compatibility__table-toolkit-driver

## Steps to download data
1. Start an interactive session in the NGC container to run data download/training/inference
```
bash scripts/docker/launch.sh <DATA_DIR> <CHECKPOINT_DIR> <RESULTS_DIR>
```

Within the container, the contents of this repository will be copied to the `/workspace/rnnt` directory. The `/datasets`, `/checkpoints`, `/results` directories are mounted as volumes
and mapped to the corresponding directories `<DATA_DIR>`, `<CHECKPOINT_DIR>`, `<RESULT_DIR>` on the host.

2. Download and preprocess the dataset.

No GPU is required for data download and preprocessing. Therefore, if GPU usage is a limited resource, launch the container for this section on a CPU machine by following prevoius steps.

Note: Downloading and preprocessing the dataset requires 500GB of free disk space and can take several hours to complete.

This repository provides scripts to download, and extract the following datasets:

* LibriSpeech [http://www.openslr.org/12](http://www.openslr.org/12)

LibriSpeech contains 1000 hours of 16kHz read English speech derived from public domain audiobooks from LibriVox project and has been carefully segmented and aligned. For more information, see the [LIBRISPEECH: AN ASR CORPUS BASED ON PUBLIC DOMAIN AUDIO BOOKS](http://www.danielpovey.com/files/2015_icassp_librispeech.pdf) paper.

Inside the container, download and extract the datasets into the required format for later training and inference:
```bash
bash scripts/download_librispeech.sh
```
Once the data download is complete, the following folders should exist:

* `/datasets/LibriSpeech/`
   * `train-clean-100/`
   * `train-clean-360/`
   * `train-other-500/`
   * `dev-clean/`
   * `dev-other/`
   * `test-clean/`
   * `test-other/`

Since `/datasets/` is mounted to `<DATA_DIR>` on the host (see Step 3),  once the dataset is downloaded it will be accessible from outside of the container at `<DATA_DIR>/LibriSpeech`.

Next, convert the data into WAV files:
```bash
bash scripts/preprocess_librispeech.sh
```
Once the data is converted, the following additional files and folders should exist:
* `datasets/LibriSpeech/`
   * `librispeech-train-clean-100-wav.json`
   * `librispeech-train-clean-360-wav.json`
   * `librispeech-train-other-500-wav.json`
   * `librispeech-dev-clean-wav.json`
   * `librispeech-dev-other-wav.json`
   * `librispeech-test-clean-wav.json`
   * `librispeech-test-other-wav.json`
   * `train-clean-100-wav/`
   * `train-clean-360-wav/`
   * `train-other-500-wav/`
   * `dev-clean-wav/`
   * `dev-other-wav/`
   * `test-clean-wav/`
   * `test-other-wav/`

## Steps to run benchmark.

### Steps to launch training

Inside the container, use the following script to start training.
Make sure the downloaded and preprocessed dataset is located at `<DATA_DIR>/LibriSpeech` on the host (see Step 3), which corresponds to `/datasets/LibriSpeech` inside the container.

```bash
NUM_GPUS=<NUM_GPUS> bash scripts/train.sh
```

# 3. Dataset/Environment
### Publication/Attribution
["OpenSLR LibriSpeech Corpus"](http://www.openslr.org/12/) provides over 1000 hours of speech data in the form of raw audio.
### Data preprocessing
What preprocessing is done to the the dataset? 
### Training and test data separation
How is the test set extracted?
### Training data order
In what order is the training data traversed?
### Test data order
In what order is the test data traversed?
### Simulation environment (RL models only)
Describe simulation environment briefly, if applicable. 
# 4. Model
### Publication/Attribution
Cite paper describing model plus any additional attribution requested by code authors 
### List of layers 
Brief summary of structure of model
### Weight and bias initialization
How are weights and biases initialized
### Loss function
Transducer Loss
### Optimizer
TBD, currently Adam
# 5. Quality
### Quality metric
Word Error Rate (WER) across all words in the output text of all samples in the validation set.
### Quality target
What is the numeric quality target
### Evaluation frequency
TBD
### Evaluation thoroughness
TBD
