# 1. Problem
Object detection and segmentation. Metrics are mask and box mAP.

# 2. Directions
### Steps to configure machine
Standard script.

### Steps to download and verify data
Run the provided shell scripts.

### Steps to run and time


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
Mask R-CNN

### List of layers 

### Weight and bias initialization
The ResNet50 base must be loaded from the provided weights. They may be quantized.

### Loss function

### Optimizer

# 5. Quality
### Quality metric
Ask Mask R-CNN can provide both boxes and masks, we evalute on both box and mask mAP.

### Quality target
Box mAP of 0.377, mask mAP of 0.339

### Evaluation frequency
Once per epoch, 118k.

### Evaluation thoroughness
Evaluate over the entire validation set. Use the official COCO API to compute mAP.
