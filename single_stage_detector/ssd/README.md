# 1. Summary
Single Shot MultiBox Detector (SSD) is an object detection network. For an input image, the network outputs a set of bounding boxes around the detected objects, along with their classes. For example:

![](https://upload.wikimedia.org/wikipedia/commons/3/38/Detected-with-YOLO--Schreibtisch-mit-Objekten.jpg)

SSD is a one-stage detector, both localization and classification are done in a single pass of the network. This allows for a faster inference than region proposal network (RPN) based networks, making it more suited for real time applications like automotive and low power devices like mobile phones. This is also sometimes referred to as being a "single shot" detector for inference.

# 2. Directions

## Steps to configure machine
### From Source
tbd
### Standard script
tbd
### From Docker
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

## Steps to download data
```
cd reference/single_stage_detector/
source download_dataset.sh
```

## ResNet-34 pretrained weights
The ResNet-34 backbone is initialized with weights from PyTorch hub:
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

By default, the code will automatically download the weights to
`$TORCH_HOME/models` (default is `~/.torch/models/`) and save them for later use.

Alternatively, you can manually download the weights with:
```
cd reference/single_stage_detector/
./download_resnet34_backbone.sh
```

Then use the downloaded file with `--pretrained-backbone <PATH TO WEIGHTS>` .

## Steps to run benchmark
tbd
## Steps to launch training
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
## Publiction/Attribution
Microsoft COCO: COmmon Objects in Context. 2017.

### Training and test data separation
Train on 2017 COCO train data set, compute mAP on 2017 COCO val data set.

# 4. Model
Given an input 300x300 image from [Coco 2017](https://cocodataset.org/) with 80 categories, the output of this network is a set of category and bounding boxes.  Other detectors make these two predictions in multiple stages, by first proposing a region of interest, then iterating over the regions of interest to try to categorize each object.  SSD does both of these in one stage, making inference faster.

## Backbone
The backbone is loosely based on adapting a ResNet as described in [this paper](https://arxiv.org/pdf/1611.10012.pdf) by Google Research.  It is similar to a ResNet-34 backbone, modified by removing the strides from the convolution at the beginning of the res4* layers (so that the res4* layers work with 38x38 images instead of 19x19 images), and removing everything after the res4* layers (the res5* layers and the fully connected stuff).  This is similar to the modifications to ResNet-101 described in Section 3.1.3 of the Google Research paper (our network has an effective stride of 8, not 16, and has no atrous convolutions).  Most other implementations of SSD are based on ResNet-50 or ResNet-101 or Mobilenet backbones, the ResNet-34 backbone we used here is unique to this implementation.

Input images are 300x300 RGB. They are fed to a 7x7 stride 2 convolution with 64 output channels, then through a 3x3 stride 2 max-pool layer, resulting in a 75x75x64 (HWC) tensor.  The rest of the backbone is built from "building blocks": pairs of 3x3 convolutions with a "short-cut" residual connection around the pair.  All convolutions in the backbone are followed by batch-norm and relu.

![](https://miro.medium.com/max/570/1*D0F3UitQ2l5Q0Ak-tjEdJg.png)

The building blocks are organized into 3 groups:

The conv2 layers consist of 3 building blocks, (so 6 3x3 convolutions), each of which inputs and outputs 75x75x64 tensors.

The conv3 layers start with a special downsizing building block that changes the activation size to 38x38x128, followed by 3 normal building blocks.  The downsizing building block replaces the identity connection with a 1x1 stride 2 convolution, and the first 3x3 convolution is made stride 2 and doubles the number of output channels to 128.

The conv4 layers consist of 6 building blocks.  The very first convolution in the conv4 layers doubles the number of channels to 256, so these convolutions all output 38x38x256 activations.  The identity connection on this initial building block is a 1x1 unstrided convolution.  (This is an artifact of the way the backbone was derived from the original ResNet-34 architecture).

The backbone is initialized with the pretrained weights from the [Torchvision model zoo](https://download.pytorch.org/models/resnet34-333f7ec4.pth), described in detail [here](https://pytorch.org/docs/stable/torchvision/models.html) (This is a ResNet-34 network trained on Imagenet to achieve a Top-1 error rate of 26.7 and a Top-5 error rate of 8.58.)

The resulting network looks somewhat similar to this picture chopped from the ResNet paper:

![](https://miro.medium.com/max/700/1*1CSCPAEhvtBPcjNA9bll0A.png)

The purple layers are "conv2", the green layers are "conv3" and the pink layers are "conv4".  The picture is accurate except for the first pink layer which is stride 1 in the SSD reference backbone, rather than stride 2.

## Head network
The 38x38 output of res4f gets fed into the following "head" network:

TODO: find public image of the head network from Matt's picture.

In the head network, the convolutions coming down vertically from the center to the right hand side are simply downconverting.  The downconversion network is described in the following table.  All of these are followed by bias/relu, but not batch-norm

TODO: copy Matt's table to markdown format

## Detection heads and anchors
The last layers of the network are the detector heads.  These consist of a total of 8732 _anchors_, each with an implicit default center and bounding box size (some papers call the implicit defaults a _prior_).  Each anchor has 85 channels associated with it.  The Coco dataset has 80 categories, so each anchor has 80 channels for categorization of what's "in" that anchor, plus an 81st channel indicating "nothing here", and then 4 channels to indicate adjustments to the bounding box.  The adjustment channels are xywh where xy are centered at the default center, and in the scale of the default bounding box (so a value of 0 in this channel indicates "at the default center", while a value of 1 in this channel indicates "a very large deviation from center."  The wh channels are given in natural log of a multiplicative factor to the implicit default bounding box width and height.  (So a multiplicative factor of 1 (log(1)=0) indicates no adjustment, a multiplicative factor of 1.1 (log(1.1) = .095) indicates a larger bounding box, and multiplicative factor of 0.9 (log(0.9) = -0.105) indicates a smaller bounding box.  Each of the 8732 pixels in the image pyramid has either 4 or 6 anchors associated with it.  4 anchors for the 38x38, 5x5, 3x3, 1x1 levels, 6 anchors for the 19x19, and 10x10.  When there are 4 anchors they have aspect ratios 1:1, 1:1, 1:2, and 2:1.  When there are 6 anchors they have aspect ratios 1:1, 1:1, 1:2, 2:1, 1:3, 3:1.  The first 1:1 box is at the default scale for the image pyramid layer, while the second 1:1 box is at a scale halfway between the scale of this image pyramid layer and the next.  The scales for 38, 19, 10, 5, 3, 1 with respect to 300x300 are 21, 45, 99, 153, 207, 261, 315.


## Ground truth and loss function

In the deployed inference network non-maximum suppression is used to select
only the most confident of any set of overlapping bounding boxes detecting an
object of class x.  Correctness is evaluated by comparing the Jaccard overlap
of the selected bounding box with the ground truth.  The training script, on
the other hand, must associate ground truth with anchor boxes:

1. Encode the ground truth tensor from the list of ground truth bounding boxes (this is implemented in function box_encoder in `csrc/box_encoder_cuda.cu`):

    1. Calculate the Jaccard overlap of each anchor's default box with each ground truth bounding box.

    2. For each ground-truth bounding box: assign a "positive" to the single anchor box with the highest Jaccard overlap for that ground truth.

    3. For each remaining unassigned anchor box: assign a "positive" for the ground truth bounding box with the highest Jaccard overlap > 0.5 (if any).

    4. For each "positive" anchor identified in steps b and c, calculate the 4 offset channels as the difference between the ground truth bounding-box and the defaults for that anchor.

2. _Hard negative mining_ in the loss function.  (implemented with multiple
kernels in `opt_loss.py`).  The ground-truth tells you which anchors are
assigned as "positives", but most anchors will be negatives and so would
overwhelm the loss calculation, so we need to choose a subset to train against.

    1. Count the number of positive anchors, P, identified in steps 1b and 1c.

    2. Of the remaining unassigned anchors, choose the 3P of them that are most
    strongly predicting a category other than "background", and assign them to
    the "background" category.

3. For each assigned anchor: The loss over categories is the softmax loss over
class confidences.  The loss over offsets is the Smooth L1 loss between the
predicted box and the ground truth box.  (These losses are implemented just
after hard negative mining in `opt_loss.py`)

## Input augmentations
The input images are assumed to be sRGB with values in range 0.0 through 1.0.  The input pipeline does the following.

1. Normalize the colors to a mean of (0.485, 0.456, 0.406) and standard deviation (0.229, 0.224, 0.225).

2. To both the image and its ground-truth bounding boxes:

    1. Random crop with equal probability choose between (1) original input,
    (2-7) minimum overlap crop of 0, 0.1, 0.3, 0.5, 0.7, 0.9, with the
    additional constraints that the width and height are (uniformly) chosen
    between 30% and 100% of the original image, and the aspect ratio is less
    than 2:1 (or 1:2).

    2. Random horizontal flip.

    3. Scale to 300x300.

3. Color is jittered by adjusting brightness by a multiplicative factor chosen
uniformly between (.875, 1.125), adjusting contrast by a multiplicative factor
chosen uniformly between 0.5, 1.5, adjusting saturation by a multiplicative
factor chose uniformly from 0.5 to 1.5, and adjusting hue by an additive factor
chosen uniformly from -18 to +18 degrees.  This is done with the call to
`torchvision.transforms.ColorJitter(brightness=0.125, contrast=0.5,
saturation=0.5, hue=0.05)` in `utils.py`.

## Publication/Attribution

_Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed,
Cheng-Yang Fu, Alexander C. Berg. SSD: Single Shot MultiBox Detector. In the
Proceedings of the European Conference on Computer Vision (ECCV), 2016_

Backbone is ResNet-34 pretrained on ILSVRC 2012 (from
torchvision). Modifications to the backbone networks: remove conv_5x residual
blocks, change the first 3x3 convolution of the conv_4x block from stride 2 to
stride1 (this increases the resolution of the feature map to which detector
heads are attached), attach all 6 detector heads to the output of the last
conv_4x residual block. Thus detections are attached to 38x38, 19x19, 10x10,
5x5, 3x3, and 1x1 feature maps.

# 5. Quality
## Quality metric
Metric is COCO box mAP (averaged over IoU of 0.5:0.95), computed over 2017 COCO val data.

## Quality target
mAP of 0.23

## Evaluation frequency
Every 5th epoch, starting with epoch 5.

## Evaluation thoroughness
All the images in COCO 2017 val data set.
