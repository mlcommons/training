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

The backbone is based on adapting a ResNet-34 as described in Section 3.1.3 of
[this paper](https://arxiv.org/abs/1611.10012) by Google Research.  Using the
same notation as Table 1 of the [original ResNet
paper](https://arxiv.org/abs/1512.03385) the backbone looks like:

| layer name | output size | ssd-backbone |
| :--------: | :---------: | :----------: |
| conv1      | 150x150     | 7x7, 64, stride 2 |
|            | 75x75       | 3x3 max pool, stride 2 |
| conv2_x    | 75x75       | pair-of[3x3, 64] x 3 |
| conv3_x    | 38x38       | pair-of[3x3, 128] x 4 |
| conv4_x    | 38x38       | pair-of[3x3, 256] x 6 |

The original ResNet-34 network is adapted by removing the conv5_x layers and
the fully-connected layer at the end, and by only downsampling in conv3_1,
_not_ in conv4_1.  Using the terminology of Section 3.1.3 of the Google
Research paper, our network has an effective stride of 8, and no atrous
convolution.

Input images are 300x300 RGB. They are fed to a 7x7 stride 2 convolution with
64 output channels, then through a 3x3 stride 2 max-pool layer, resulting in a
75x75x64 (HWC) tensor.  The rest of the backbone is built from "building
blocks": pairs of 3x3 convolutions with a "short-cut" residual connection
around the pair.  All convolutions in the backbone are followed by batch-norm
and relu.

![](https://miro.medium.com/max/570/1*D0F3UitQ2l5Q0Ak-tjEdJg.png)

The conv3_1 layer is stride 2 in the first convolution, while also increasing
the number of channels from 64 to 128, and has a 1x1 stride 2 convolution in
its residual shortcut path to increase the number of channels to 128.  The
conv4_1 layer is _not_ strided, but does increase the number of channels from
128 to 256, and so also has a 1x1 convolution in the residual shortcut path to
increase the number of channels to 256.

The backbone is initialized with the pretrained weights from the corresponding
layers of the ResNet-34 implementation from the [Torchvision model
zoo](https://download.pytorch.org/models/resnet34-333f7ec4.pth), described in
detail [here](https://pytorch.org/docs/stable/torchvision/models.html).  It is
a ResNet-34 network trained on 224x224 Imagenet to achieve a Top-1 error rate
of 26.7 and a Top-5 error rate of 8.58.

## Head network

The 38x38, 256 channel output of the conv4 layer gets fed into a downsizing
network with the following structure:

| layer name | input channels | input size | output channels | filter size | padding | stride | output size |
| :------: | :-: | :---: | :-: | :-: | :-: | :-: | :---: | 
| conv7_1  | 256 | 38x38 | 256 | 1x1 | 0 | 1 | 38x38 |
| conv7_2  | 256 | 38x38 | 512 | 3x3 | 1 | 2 | 19x19 |
| conv8_1  | 512 | 19x19 | 256 | 1x1 | 0 | 1 | 19x19 |
| conv8_2  | 256 | 19x19 | 512 | 3x3 | 1 | 2 | 10x10 |
| conv9_1  | 512 | 10x10 | 128 | 1x1 | 0 | 1 | 10x10 |
| conv9_2  | 128 | 10x10 | 256 | 3x3 | 1 | 2 | 5x5 |
| conv10_1 | 256 |   5x5 | 128 | 1x1 | 0 | 1 | 5x5 |
| conv10_2 | 128 |   5x5 | 256 | 3x3 | 0 | 1 | 3x3 |
| conv11_1 | 256 |   3x3 | 128 | 1x1 | 0 | 1 | 3x3 |
| conv11_2 | 128 |   3x3 | 256 | 3x3 | 0 | 1 | 1x1 |

As in the original SSD paper, Each convolution in the downsizing network is
followed by bias/relu, but not batch-norm.

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

2. _Hard negative mining_ in the loss function (implemented in `opt_loss.py`).
The ground-truth tells you which anchors are assigned as "positives", but most
anchors will be negatives and so would overwhelm the loss calculation, so we
need to choose a subset to train against.

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

Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed,
Cheng-Yang Fu, Alexander C. Berg.  [SSD: Single Shot MultiBox
Detector](https://arxiv.org/abs/1512.02325). In the _Proceedings of the
European Conference on Computer Vision_, (ECCV-14):21-37, 2016.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.  [Deep Residual Learning for
Image Recognition](https://arxiv.org/abs/1512.03385).  In the _Proceedings of
the Conference on Computer Vision and Pattern Recognition_, (CVPR):770-778, 2016.

Jonathan Huang, Vivek Rathod, Chen Sun, Menglong Zhu, Anoop Korattikara,
Alireza Fathi, Ian Fischer, Zbigniew Wojna, Yang Song, Sergio Guadarrama, Kevin
Murphy. [Speed/accuracy trade-offs for modern convolutional object
detectors](https://arxiv.org/abs/1611.10012).  In the _Proceedings of the
Conference on Computer Vision and Pattern Recognition_, (CVPR):3296-3305, 2017.

# 5. Quality
## Quality metric
Metric is COCO box mAP (averaged over IoU of 0.5:0.95), computed over 2017 COCO val data.

## Quality target
mAP of 0.23

## Evaluation frequency
Every 5th epoch, starting with epoch 5.

## Evaluation thoroughness
All the images in COCO 2017 val data set.
