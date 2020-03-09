# 1. Problem
Object detection.

# 2. Directions

### Steps to configure machine
From Source

Standard script.

From Docker
1. Checkout the MLPerf repository
```
git clone https://github.com/mlperf/reference.git
```
2. Install CUDA and Docker
```
source reference/install_cuda_docker.sh
```
3. Build the docker image for the single stage detection task
```
# Build from Dockerfile
cd reference/single_stage_detector/
sudo docker build -t mlperf/single_stage_detector .
```

### Steps to download data
```
cd reference/single_stage_detector/
source download_dataset.sh
```

## Running with Popper

[Popper](https://github.com/systemslab/popper) is a tool for defining and executing container-native workflows either locally or on CI services. Some workflows in this repository contain a `wf.yml` file that defines a Popper workflow for automatically downloading and verifying data, running the benchmark and generating a report. The execution and report generation both comply with the [MLPerf training rules](https://github.com/mlperf/training_policies/blob/master/training_rules.adoc). More details about Popper can be found [here](https://popper.readthedocs.io/).


### Instructions:

1. Clone the repository.
```
git clone https://github.com/mlperf/training
```

2. Install docker, cuda-runtime and nvidia-docker on the machine.
```
./training/install_cuda_docker.sh
```

3. Install the `popper` tool.
```
pip install popper
```
We recommend to use a [virtualenv](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment) for installing Popper.

4. Run the workflow.
```
cd single_stage_detector/
popper run -f wf.yml -c settings.py
```
Here, the `settings.py` file contains necessary configuration that needs to be passed to the container engine in order to use the nvidia drivers. For more information about customizing container engine parameters, see [here](https://popper.readthedocs.io/en/latest/sections/cli_features.html#customizing-container-engine-behavior).

### NVIDIA DGX-1 (single GPU)
Launch configuration and system-specific hyperparameters for the NVIDIA DGX-1
old reference are in the `config_DGX1_32.sh` script.

Steps required to launch old reference training on NVIDIA DGX-1:

```
docker build . -t mlperf-nvidia:single_stage_detector
DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> DGXSYSTEM=DGX1_32 ./run.sub
```

### NVIDIA DGX-1 (single node)
Launch configuration and system-specific hyperparameters for the NVIDIA DGX-1
single node reference are in the `config_DGX1_singlenode.sh` script.

Steps required to launch single node training on NVIDIA DGX-1:

```
docker build . -t mlperf-nvidia:single_stage_detector
DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> DGXSYSTEM=DGX1_singlenode ./run.sub
```

### NVIDIA DGX-1 (multi node)
Launch configuration and system-specific hyperparameters for the NVIDIA DGX-1
multi node reference are in the `config_DGX1_multinode.sh` script.

Steps required to launch multi node training on NVIDIA DGX-1:

```
docker build . -t mlperf-nvidia:single_stage_detector
DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> DGXSYSTEM=DGX1_multinode sbatch -N $DGXNNODES -t $WALLTIME run.sub
```

### Hyperparameter settings

Hyperparameters are recorded in the `config_*.sh` files for each configuration and in `run_and_time.sh`.

# 3. Dataset/Environment
### Publiction/Attribution.
Microsoft COCO: COmmon Objects in Context. 2017.

### Training and test data separation
Train on 2017 COCO train data set, compute mAP on 2017 COCO val data set.

# 4. Model.
### Publication/Attribution
Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg. SSD: Single Shot MultiBox Detector. In the Proceedings of the European Conference on Computer Vision (ECCV), 2016.

Backbone is ResNet34 pretrained on ILSVRC 2012 (from torchvision). Modifications to the backbone networks: remove conv_5x residual blocks, change the first 3x3 convolution of the conv_4x block from stride 2 to stride1 (this increases the resolution of the feature map to which detector heads are attached), attach all 6 detector heads to the output of the last conv_4x residual block. Thus detections are attached to 38x38, 19x19, 10x10, 5x5, 3x3, and 1x1 feature maps.

# 5. Quality.
### Quality metric
Metric is COCO box mAP (averaged over IoU of 0.5:0.95), computed over 2017 COCO val data.

### Quality target
mAP of 0.23

### Evaluation frequency

### Evaluation thoroughness
All the images in COCO 2017 val data set.
