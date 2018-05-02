# MLPerf Reference Implementations

This is a repository of reference implementations for the MLPerf benchmark. These implementations are valid for producing benchmark entries but are not intended to be efficient for performant. This repository is intended as a starting point for understand or re-implementing the MLPerf benchmarks. 


# Preliminary release (v0.5)

This release of these implementaitons, and numbers reported, should be considered as preliminary and a "pre-alpha" release. The primary intention of this release is to demonstrate an end-to-end working benchmark. The benchmark is still being developed and refined, see the Suggestions section below to learn how to contribute. 

In particular, you should expect the following is possible:
* Modifications to the reference implementations to fix bugs or add additional functionality.
* Changes to the target quality levels.
* Additional benchmarks to be added.
* Additional implementations for existing benchmarks (in new frameworks) to be added, over time.


# Contents

We provide reference implementations for each of the 7 benchmarks in the MLPerf suite. 

* image_classification - Resnet classifying Imagenet.
* object_detection - Object detection and segmentation using COCO. 
* recommendation - Neural Collaborative Filtering on MovieLens 20 Million (ml-20m).
* reinforcement - Learning Go using methods similar to AlphaGo.
* sentiment_analysis - Positive or negative sentiment on the IMDB dataset.
* speech_recognition - Speech to text using DeepSpeech2.
* transformer - Natural language translation.

Each benchmark implementation provides the following:
 
* A Dockerfile which can be used to run the benchmark in a container.
* A script which downloads the appropriate dataset.
* Documentaiton on the dataset, model and machine setup.

# Running Benchmarks

These benchmarks have been tested on a minimal machine configuration:

* 16 CPUs, one Nvidia P100.
* Ubuntu 16.04, including docker with nvidia support.
* Up to 600GB of disk (though many benchmarks will require less disk).

Generally, a benchmark can be run with the following steps:

1. Setup docker & dependencies. There is a shared script (install_cuda_docker.sh) to do this. Some benchmarks will have additional setup, mentioned in their READMEs.
2. Download the dataset using `./download_dataset.sh`. This should be run outside of docker, on your host machine. This should be run from the directory it is in (it may make assumptions about CWD).
3. Optionally, run `verify_dataset.sh` to ensure the was successfully downloaded.
4. Build and run the docker image, the command to do this is included with each Benchmark. 

Each benchmark will run until the target quality is reached and then stop, printing timing results. 

Some these benchmarks are rather slow or take a long time to run on the reference hardware (i.e. 16 CPUs and one P100). We expect to see significant performance improvements with more hardware and optimized implementations. 

# Suggestions

We are still very much in the early stages of developing MLPerf and we are looking for areas to improve, partners and contributors. If you have recommendations for new benchmarks, or otherwise would like to be involved in the process, please reach out to `info@mlperf.org`. For technical bugs or support, email `support@mlperf.org`.
