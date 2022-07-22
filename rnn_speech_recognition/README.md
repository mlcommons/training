# Benchmark execution with MLCube

We'll be updating this section as we merge MLCube PRs and make new MLCube releases.

## Current implementation

We'll be updating this section as we merge MLCube PRs and make new MLCube releases.

### Project setup
```Python
# Create Python environment and install MLCube Docker runner 
virtualenv -p python3 ./env && source ./env/bin/activate && pip install mlcube-docker

# Fetch the RNN speech recognition workload
git clone https://github.com/mlcommons/training && cd ./training
git fetch origin pull/491/head:feature/rnnt_mlcube && git checkout feature/rnnt_mlcube
cd ./rnn_speech_recognition/mlcube
```

### Dataset

The [Librispeech](https://www.openslr.org/12) dataset will be downloaded, extracted, and processed. Sizes of the dataset in each step:

| Dataset Step                   | MLCube Task       | Format     | Size    |
|--------------------------------|-------------------|------------|---------|
| Download (Compressed dataset)  | download_data     | Tar files  | ~62 GB  |
| Extract (Uncompressed dataset) | download_data     | Flac files | ~64 GB  |
| Preprocess (Processed dataset) | preprocess_data   | Wav files  | ~114 GB |
| Total                          | (After all tasks) | All        | ~240 GB |

### Tasks execution
```
# Download Librispeech dataset. Default path = /workspace/data
# To override it, use data_dir=DATA_DIR
mlcube run --task download_data

# Preprocess Librispeech dataset, this will convert .flac audios to .wav format
# It will use the DATA_DIR path defined in the previous step
mlcube run --task preprocess_data

# Run benchmark. Default paths = ./workspace/data
# Parameters to override: data_dir=DATA_DIR, output_dir=OUTPUT_DIR, parameters_file=PATH_TO_TRAINING_PARAMS
mlcube run --task train
```

We are targeting pull-type installation, so MLCube images should be available on docker hub. If not, try this:

```
mlcube run ... -Pdocker.build_strategy=always
```