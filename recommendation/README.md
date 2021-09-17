# Benchmark execution with MLCube

## Project setup

```Bash
# Create Python environment 
virtualenv -p python3 ./env && source ./env/bin/activate

# Install MLCube and MLCube docker runner from GitHub repository (normally, users will just run `pip install mlcube mlcube_docker`)
git clone https://github.com/mlcommons/mlcube && cd mlcube/mlcube
python setup.py bdist_wheel  && pip install --force-reinstall ./dist/mlcube-* && cd ..
cd ./runners/mlcube_docker && python setup.py bdist_wheel  && pip install --force-reinstall --no-deps ./dist/mlcube_docker-* && cd ../../..

# Fetch the image segmentation workload
git clone https://github.com/mlcommons/training && cd ./training
git fetch origin pull/xxx/head:feature/mlcube_recommendation && git checkout feature/mlcube_recommendation
cd ./recommendation/mlcube
```

### Dataset

The [MovieLens](https://grouplens.org/datasets/movielens/) dataset will be downloaded and processed. Sizes of the dataset in each step:

| Dataset Step                   | MLCube Task       | Format      | Size    |
|--------------------------------|-------------------|-------------|---------|
| Download (raw dataset)         | download_data     | .tar        | ~3.1 GB |
| Extract (extracted dataset)    | download_data     | *.npz       | ~3.1 GB  |
| Total                          | (After all tasks) | All         | ~6.2 GB  |

## Tasks execution

```bash
# Download KiTS19 dataset. Default path = mlcube/workspace/data
# To override it, use data_dir=DATA_DIR
mlcube run --task download_data

# Preprocess KiTS19 dataset
# It will use a subdirectory from the DATA_DIR path defined in the previous step
mlcube run --task preprocess_data

# Run benchmark. Default paths input_dir = mlcube/workspace/processed_data
# Parameters to override: input_dir=DATA_DIR, output_dir=OUTPUT_DIR, parameters_file=PATH_TO_TRAINING_PARAMS
mlcube run --task train
```
