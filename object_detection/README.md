# Benchmark execution with MLCube

## Project setup

```bash
# Create Python environment 
virtualenv -p python3 ./env && source ./env/bin/activate

# Install MLCube and MLCube docker runner from GitHub repository (normally, users will just run `pip install mlcube mlcube_docker`)
git clone https://github.com/mlcommons/mlcube && cd mlcube/mlcube
python setup.py bdist_wheel  && pip install --force-reinstall ./dist/mlcube-* && cd ..
cd ./runners/mlcube_docker && python setup.py bdist_wheel  && pip install --force-reinstall --no-deps ./dist/mlcube_docker-* && cd ../../..

# Fetch the Object Detection workload
git clone https://github.com/mlcommons/training && cd ./training
git fetch origin pull/501/head:feature/object_detection && git checkout feature/object_detection
cd ./object_detection/mlcube
```

## Dataset

The COCO dataset will be downloaded and extracted. Sizes of the dataset in each step:

| Dataset Step                   | MLCube Task       | Format         | Size     |
|--------------------------------|-------------------|----------------|----------|
| Download (Compressed dataset)  | download_data     | Tar/Zip files  | ~20.5 GB |
| Extract (Uncompressed dataset) | download_data     | Jpg/Json files | ~21.2 GB |
| Total                          | (After all tasks) | All            | ~41.7 GB |

## Tasks execution

Parameters are defined at these files:

* MLCube user parameters: mlcube/workspace/parameters.yaml
* Project user parameters: pytorch/configs/e2e_mask_rcnn_R_50_FPN_1x.yaml
* Project default parameters: pytorch/maskrcnn_benchmark/config/defaults.py

```bash
# Download COCO dataset. Default path = /workspace/data
mlcube run --task download_data

# Run benchmark. Default paths = ./workspace/data
mlcube run --task train
```

By default MLCube images use pull-type installation, so they should be available on docker hub. If not, try this:

```bash
mlcube run ... -Pdocker.build_strategy=auto
```
