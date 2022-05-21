# 1. Problem
Speech recognition takes raw audio samples and produces a corresponding text transcription.

# 2. DISCLAIMER: Model Update

Recently, the model size was updated to be more representative of speech recognition networks deployed in industry.
While the model has been modified to reflect these changes, its accuracy is low.
We expect this will take additional hyperparameter tuning (i.e., learning rate, data pre-processing) that is outside the core model description.
Hyperparameters to achieve more realistic accuracy will be updated shortly.

# 3. Directions
### Steps to configure machine
Suggested environment : Ubuntu 16.04, 8 CPUs, one P100, 300GB disk

Assume sufficiently recent NVIDIA driver is installed.

The following instructions modify `reference/install_cuda_docker.sh` script to install cuda 9.0 instead of 9.1 in addition to installing and configuring docker and nvidia-docker.

First, get cuda 9.0:

    wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
    sudo dpkg -i cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
    sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
    sudo apt-get update
    sudo apt-get install cuda-libraries-9-0

Next, install docker:

    sudo apt-get install apt-transport-https ca-certificates curl software-properties-common
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
    sudo apt-key fingerprint 0EBFCD88
    sudo add-apt-repository    "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
       $(lsb_release -cs) \
       stable"
    sudo apt update
    sudo apt install docker-ce -y


Next, nvidia-docker2:

    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID) && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    sudo apt-get update
    sudo apt install nvidia-docker2 -y
    sudo pkill -SIGHUP dockerd

### Steps to download and verify data
The `download_dataset` script will use the python data utils defined in the `data` directory to download and process the full LibriSpeech dataset into `./LibriSpeech_dataset`.  This takes up to 6 hours.
The `verify_dataset` script will build a single tarball of the dataset, checksum it, compare against a reference checksum, and report whether they match.  This takes up to an hour.

Please run the download and verification inside the docker instance (described below).

    sh download_dataset.sh
    sh verify_dataset.sh

This should return `Dataset Checksum Passed.`

NOTE: The dataset itself is over 100GB, and intermediate files during download, processing, and verification require 220GB of free disk space to complete.

### Steps to run and time
For each framework, there is a provided docker file and `run_and_time.sh` script.
To run the benchmark, (1) build the docker image, if you haven't already, (2) launch the docker instance (making path modifications as necessary), and (3) run and time the `run_and_time` script, optionally piping output to a log file.
For example, for the pytorch framework:

    cd pytorch
    cd docker
    sh build-docker.sh
    sh run-dev.sh
    time sh run_and_time.sh | tee speech_ds2.out

NOTE: remember to modify paths in `docker/run-dev.sh` as appropriate (e.g., replace line 3 with the base path for the repo `~/mlperf/reference/speech_recognition` or similar).

The model will run until the specified target accuracy is achieved or 10 full epochs have elapsed, whichever is sooner. The maximum number of epochs, along with other network parameters, can be viewed and modified in `pytorch/params.py`.

# 4. Dataset/Environment
### Publication/Attribution
["OpenSLR LibriSpeech Corpus"](http://www.openslr.org/12/) provides over 1000 hours of speech data in the form of raw audio.
### Data preprocessing
The audio files are sampled at 16kHz.
All inputs are also pre-processed into a spectrogram of the first 13 mel-frequency cepstral coefficients (MFCCs).
### Training and test data separation
After running the `download_dataset` script, the `LibriSpeech_dataset` directory will have subdirectories for training set, validation set, and test set.
Each contains both clean and noisy speech samples.
### Data order
Audio samples are sorted by length.
# 5. Model
### Publication/Attribution
This is an implementation of [DeepSpeech2](https://arxiv.org/pdf/1512.02595.pdf) adapted from [deepspeech.pytorch](https://github.com/SeanNaren/deepspeech.pytorch).
### List of layers
Summary: Sampled Audio Spectrograms -> 2 CNN layers -> 5 Bi-Directional GRU layers -> FC classifier layer -> output text

Details:

  (module): DeepSpeech (

    (conv): Sequential (

      (0): Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2))

      (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True)

      (2): Hardtanh (min_val=0, max_val=20, inplace)

      (3): Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1))

      (4): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True)

      (5): Hardtanh (min_val=0, max_val=20, inplace)

    )
    (rnns): Sequential (

      (0): BatchRNN (


        (rnn): GRU(672, 2560)
      )

      (1): BatchRNN (

        (rnn): GRU(2560, 2560)

      )

      (2): BatchRNN (

        (rnn): GRU(2560, 2560)

      )

    )

    (fc): Sequential (

      (0): SequenceWise (

      Sequential (

        (0): BatchNorm1d(2560, eps=1e-05, momentum=0.1, affine=True)

        (1): Linear (2560 -> 29)

      ))

    )

    (inference_log_softmax): InferenceBatchLogSoftmax (

    )

  )

)

# 5. Quality
### Quality metric
Word Error Rate (WER) across all words in the output text of all samples in the validation set.
### Quality target
WER of 23.0.
### Evaluation
All training samples are examined per epoch, and at the end of each epoch the model WER is evaluated on the validation set.
