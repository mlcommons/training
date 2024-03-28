# Benchmark execution with MLCube

## Project setup

```Bash
# Create Python environment and install MLCube Docker runner 
virtualenv -p python3 ./env && source ./env/bin/activate && pip install mlcube-docker

# Fetch the image segmentation workload
git clone https://github.com/mlcommons/training && cd ./training
git fetch origin pull/494/head:feature/mlcube_image_segmentation && git checkout feature/mlcube_image_segmentation
cd ./image_segmentation/mlcube
```

### Dataset

The [KiTS19](https://kits19.grand-challenge.org/data/) dataset will be downloaded and processed. Sizes of the dataset in each step:

| Dataset Step                   | MLCube Task       | Format     | Size    |
|--------------------------------|-------------------|------------|---------|
| Download (raw dataset)         | download_data     | nii.gz     | ~29 GB  |
| Preprocess (Processed dataset) | preprocess_data   | npy        | ~31 GB |
| Total                          | (After all tasks) | All        | ~60 GB |

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

We are targeting pull-type installation, so MLCube images should be available on docker hub. If not, try this:

```bash
mlcube run ... -Pdocker.build_strategy=always
```