# MLPerf Reference Implementations

This is a repository of reference implementations for the MLPerf benchmarks. These implementations are valid as starting points for benchmark implementations but are not fully optimized and are not intended to be used for "real" performance measurements of software frameworks or hardware. 

These reference implementations are still very much "alpha" or "beta" quality. They could be improved in many ways. Please file issues or pull requests to help us improve quality.

# Contents

We provide reference implementations for benchmarks in the MLPerf suite, as well as several benchmarks under development. 

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


