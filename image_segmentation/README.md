## Current implementation

We'll be updating this section as we merge MLCube PRs and make new MLCube releases.

### Project setup
```Python
# Create Python environment 
virtualenv -p python3 ./env && source ./env/bin/activate

# Install MLCube and MLCube docker runner from GitHub repository (normally, users will just run `pip install mlcube mlcube_docker`)
git clone https://github.com/sergey-serebryakov/mlbox.git && cd mlbox && git checkout feature/configV2
cd ./runners/mlcube_docker && export PYTHONPATH=$(pwd)
cd ../../ && pip install -r mlcube/requirements.txt && pip install omegaconf && cd ../

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

### Tasks execution
```
# Download KiTS19 dataset. Default path = mlcube/workspace/data
# To override it, use --data_dir=DATA_DIR
python mlcube_cli.py run --task download_data --platform docker

# Preprocess KiTS19 dataset
# It will use a subdirectory from the DATA_DIR path defined in the previous step
python mlcube_cli.py run --task preprocess_data --platform docker

# Run benchmark. Default paths input_dir = mlcube/workspace/processed_data
# Parameters to override: --input_dir=DATA_DIR, --output_dir=OUTPUT_DIR, --parameters_file=PATH_TO_TRAINING_PARAMS
python mlcube_cli.py run --task train --platform docker
```