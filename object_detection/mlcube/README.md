# Benchmark execution with MLCube

### Project setup
```Python
# Create Python environment and install MLCube Docker runner 
virtualenv -p python3 ./env && source ./env/bin/activate && pip install mlcube-docker

# Fetch the Object Detection workload
git clone https://github.com/mlcommons/training && cd ./training
git fetch origin pull/501/head:feature/object_detection && git checkout feature/object_detection
cd ./object_detection/mlcube
```

### Dataset


The COCO dataset will be downloaded and extracted. Sizes of the dataset in each step:

| Dataset Step                   | MLCube Task       | Format         | Size     |
|--------------------------------|-------------------|----------------|----------|
| Download (Compressed dataset)  | download_data     | Tar/Zip files  | ~20.5 GB |
| Extract (Uncompressed dataset) | download_data     | Jpg/Json files | ~21.2 GB |
| Total                          | (After all tasks) | All            | ~41.7 GB |

### Tasks execution

Parameters are defined at these files:

* MLCube user parameters: mlcube/workspace/parameters.yaml
* Project user parameters: pytorch/configs/e2e_mask_rcnn_R_50_FPN_1x.yaml
* Project default parameters: pytorch/maskrcnn_benchmark/config/defaults.py

```bash
# Download COCO dataset. Default path = /workspace/data
mlcube run --task=download_data -Pdocker.build_strategy=always

# Run benchmark. Default paths = ./workspace/data
mlcube run --task=train -Pdocker.build_strategy=always
```

### Demo execution

These tasks will use a demo dataset (39M) to execute a faster training workload for a quick demo (~12 min):

```bash
# Download subsampled dataset. Default path = /workspace/demo
mlcube run --task=download_data -Pdocker.build_strategy=always

# Run benchmark. Default paths = ./workspace/demo and ./workspace/demo_output
mlcube run --task=demo -Pdocker.build_strategy=always
```

It's also possible to execute the two tasks in one single instruction:

```bash
mlcube run --task=download_demo,demo -Pdocker.build_strategy=always
```

### Aditonal options

Parameters defined at **mculbe/mlcube.yaml** could be overridden using: `--param=input`

We are targeting pull-type installation, so MLCube images should be available on docker hub. If not, try this:

```bash
mlcube run ... -Pdocker.build_strategy=always
```
