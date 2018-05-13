# 1. Problem
Object detection and segmentation. Metrics are mask and box mAP.

# 2. Directions
### Steps to configure machine
Standard script.

### Steps to download and verify data
init and update the submodules in this directory.

Run the provided shell scripts *in this directory*.

### Steps to run and time
Build the docker container.

```
sudo docker build -t detectron .
```

Run the docker container and mount the data appropriately

```
sudo nvidia-docker run
-v /mnt/disks/data/coco/:/packages/detectron/lib/datasets/data/coco
-it detectron /bin/bash
```

(replace /mnt/disks/data/coco/ with the data directory)

Run the command:
```
time stdbuf -o 0 \
  python tools/train_net.py --cfg configs/12_2017_baselines/e2e_mask_rcnn_R-50-FPN_1x.yaml \
    --box_min_ap 0.377 --mask_min_ap 0.339 \
    --seed 3 | tee run.log
 ```

# 3. Dataset/Environment
### Publication/Attribution
Microsoft COCO: Common Objects in Context

### Data preprocessing
Only horizontal flips are allowed.

### Training and test data separation
As provided by MS-COCO (2017 version).

### Training data order
Randomly.

### Test data order
Any order.

# 4. Model
### Publication/Attribution
He, Kaiming, et al. "Mask r-cnn." Computer Vision (ICCV), 2017 IEEE International Conference on.
IEEE, 2017.

We use a version of Mask R-CNN with a ResNet50 backbone.

### List of layers 
Running the timing script will display a list of layers.

### Weight and bias initialization
The ResNet50 base must be loaded from the provided weights. They may be quantized.

### Loss function
Multi-task loss (classification, box, mask). Described in the Mask R-CNN paper.

Classification: Smooth L1 loss

Box: Log loss for true class.

Mask: per-pixel sigmoid, average binary cross-entropy loss.

### Optimizer
Momentum SGD. Weight decay of 0.0001, momentum of 0.9.

# 5. Quality
### Quality metric
As Mask R-CNN can provide both boxes and masks, we evaluate on both box and mask mAP.

### Quality target
Box mAP of 0.377, mask mAP of 0.339

### Evaluation frequency
Once per epoch, 118k.

### Evaluation thoroughness
Evaluate over the entire validation set. Use the official COCO API to compute mAP.
