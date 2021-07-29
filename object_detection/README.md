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
```
# Download COCO dataset. Default path = /workspace/data
python mlcube_cli.py run --task download_data --platform docker

# Run benchmark. Default paths = ./workspace/data
python mlcube_cli.py run --task train --platform docker
```