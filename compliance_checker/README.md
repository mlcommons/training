# MLPerf Reference Implementations

This is a repository of reference implementations for the MLPerf benchmark. These implementations are valid as starting points for benchmark implementations but are not fully optimized and are not intended to be used for "real" performance measurements of software frameworks or hardware. 

# Preliminary release (v0.5)

This release is very much an "alpha" release -- it could be improved in many ways. The benchmark suite is still being developed and refined, see the Suggestions section below to learn how to contribute. 

We anticipate a significant round of updates at the end of May based on input from users.

# Contents

We provide reference implementations for each of the 7 benchmarks in the MLPerf suite. 

* image_classification - Resnet-50 v1 applied to Imagenet.
* object_detection - Mask R-CNN applied to COCO. 
* single_stage_detector - SSD applied to COCO 2017.
* speech_recognition - DeepSpeech2 applied to Librispeech.
* translation - Transformer applied to WMT English-German.
* recommendation - Neural Collaborative Filtering applied to MovieLens 20 Million (ml-20m).
* sentiment_analysis - Seq-CNN applied to IMDB dataset.
* reinforcement - Mini-go applied to predicting pro game moves.

Each reference implementation provides the following:
 
* Code that implements the model in at least one framework.
* A Dockerfile which can be used to run the benchmark in a container.
* A script which downloads the appropriate dataset.
* A script which runs and times training the model.
* Documentation on the dataset, model, and machine setup.

# Running Benchmarks

These benchmarks have been tested on the following machine configuration:

* 16 CPUs, one Nvidia P100.
* Ubuntu 16.04, including docker with nvidia support.
* 600GB of disk (though many benchmarks do require less disk).
* Either CPython 2 or CPython 3, depending on benchmark (see Dockerfiles for details).

Generally, a benchmark can be run with the following steps:

1. Setup docker & dependencies. There is a shared script (install_cuda_docker.sh) to do this. Some benchmarks will have additional setup, mentioned in their READMEs.
2. Download the dataset using `./download_dataset.sh`. This should be run outside of docker, on your host machine. This should be run from the directory it is in (it may make assumptions about CWD).
3. Optionally, run `verify_dataset.sh` to ensure the was successfully downloaded.
4. Build and run the docker image, the command to do this is included with each Benchmark. 

Each benchmark will run until the target quality is reached and then stop, printing timing results. 

Some these benchmarks are rather slow or take a long time to run on the reference hardware (i.e. 16 CPUs and one P100). We expect to see significant performance improvements with more hardware and optimized implementations. 

# Suggestions

We are still in the early stages of developing MLPerf and we are looking for areas to improve, partners, and contributors. If you have recommendations for new benchmarks, or otherwise would like to be involved in the process, please reach out to `info@mlperf.org`. For technical bugs or support, email `support@mlperf.org`.
