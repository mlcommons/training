# 1. Problem 
Speech recognition takes raw audio samples and produces a corresponding text transcription.
# 2. Directions
### Steps to configure machine
TODO - separate from base install script to setup nvidia-docker?
### Steps to download and verify data
The 'download_dataset' script will use the python data utils defined in the 'data' directory to download and process the full LibriSpeech dataset into './LibriSpeech_dataset'.  This takes up to 6 hours.
The 'verify_dataset' script will build a single tarball of the dataset, checksum it, compare against a reference checksum, and report whether they match.  This takes up to an hour.

Either run the download and verification inside the docker instance (described below) or, optionally, install additional dependencies locally with `sh install_dependencies.sh`.

    sh download_dataset.sh 
    sh verify_dataset.sh

NOTE: The dataset itself is over 100GB, and intermediate files during download, processing, and verification require 220GB of free disk space to complete.
 
### Steps to run and time
For each framework, there is a provided docker file and 'run_and_time' script.
To run the benchmark, (1) build the docker image, if you haven't already, (2) launch the docker instance (making path modifications as necessary), and (3) run the 'run_and_time' script, optionally piping output to a log file.
For example, for the pytorch framework:

    cd pytorch
    cd docker
    sh build-docker.sh
    sh run-dev.sh
    sh run_and_time.sh | tee mlperf_ds2.log
    
# 3. Dataset/Environment
### Publication/Attribution
["OpenSLR LibriSpeech Corpus"](http://www.openslr.org/12/) provides over 1000 hours of speech data in the form of raw audio.
### Data preprocessing
The audio files are sampled at 16kHz.
All inputs are also pre-processed into a spectrogram of the first 13 mel-frequency cepstral coefficients (MFCCs). 
### Training and test data separation
After running the 'download_dataset' script, the 'LibriSpeech_dataset' directory will have subdirectories for training set, validation set, and test set.
### Training data order
Audio samples are sorted by length.
### Test data order
Audio samples are sorted by length.
# 4. Model
(TODO)
### Publication/Attribution
DeepSpeech2 Paper
Original DS2 Pytorch Repo
### List of layers 
Sampled Audio Spectrograms -> 2 CNN layers -> 5 GRU layers -> FC classifier layer -> output text
### Weight and bias initialization
How are weights and biases intialized
### Loss function
Name/description of loss function used
### Optimizer
Name of optimizer
# 5. Quality
### Quality metric
Word Error Rate (WER) across all words of all samples in the test set.
### Quality target
WER of 23.0.
### Evaluation frequency
All test samples are examined per epoch, then tested on the validation set.
### Evaluation thoroughness
All test samples are examined per epoch, then tested on the validation set.
