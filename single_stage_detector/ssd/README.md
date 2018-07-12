# 1. Problem
Object detection.

# 2. Directions

### Steps to configure machine
Standard script.

### Steps to download data
```
cd reference/single_stage_detector/
source download_dstaset.sh
```

### Run benchmark.
```
cd reference/single_stage_detector/ssd
source run_and_time.sh SEED TARGET
```
Where SEED is the random seed for a run, TARGET is the quality target from Section 5 below.

# 3. Dataset/Environment
### Publiction/Attribution.
Microsoft COCO: COmmon Objects in Context. 2017.

### Training and test data separation
Train on 2017 COCO train data set, compute mAP on 2017 COCO val data set.

# 4. Model.
### Publication/Attribution
Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg. SSD: Single Shot MultiBox Detector. In the Proceedings of the European Conference on Computer Vision (ECCV), 2016.

Backbone is VGG-D pretrained on ILSVRC 2012 (from torchvision).

# 5. Quality.
### Quality metric
Metric is COCO box mAP (averaged over IoU of 0.5:0.95), computed over 2017 COCO val data.

### Quality target
mAP of 0.212

### Evaluation frequency

### Evaluation thoroughness
All the images in COCO 2017 val data set.
