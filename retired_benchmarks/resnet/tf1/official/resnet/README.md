# ResNet in TensorFlow

Deep residual networks, or ResNets for short, provided the breakthrough idea of identity mappings in order to enable training of very deep convolutional neural networks. This folder contains an implementation of ResNet for the ImageNet dataset written in TensorFlow.

See the following papers for more background:

[1] [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf) by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.

[2] [Identity Mappings in Deep Residual Networks](https://arxiv.org/pdf/1603.05027.pdf) by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.

In code, v1 refers to the ResNet defined in [1] but where a stride 2 is used on
the 3x3 conv rather than the first 1x1 in the bottleneck. This change results
in higher and more stable accuracy with less epochs than the original v1 and has
shown to scale to higher batch sizes with minimal degradation in accuracy.
There is no originating paper and the first mention we are aware of was in the
[torch version of ResNetv1](https://github.com/facebook/fb.resnet.torch). Most
popular v1 implementations are this implementation which we call ResNetv1.5. In
testing we found v1.5 requires ~12% more compute to train and has 6% reduced
throughput for inference compared to ResNetv1. Comparing the v1 model to the
v1.5 model, which has happened in blog posts, is an apples-to-oranges
comparison especially in regards to hardware or platform performance. CIFAR-10
ResNet does not use the bottleneck and is not impacted by these nuances.

v2 refers to [2]. The principle difference between the two versions is that v1
applies batch normalization and activation after convolution, while v2 applies
batch normalization, then activation, and finally convolution. A schematic
comparison is presented in Figure 1 (left) of [2].


## ImageNet

### Setup
To begin, you will need to download the ImageNet dataset and convert it to TFRecord format. Follow along with the [Inception guide](https://github.com/tensorflow/models/tree/master/research/inception#getting-started) in order to prepare the dataset.

Once your dataset is ready, you can begin training the model as follows:

```
python imagenet_main.py --data_dir=/path/to/imagenet
```

The model will begin training and will automatically evaluate itself on the validation data roughly once per epoch.

Note that there are a number of other options you can specify, including `--model_dir` to choose where to store the model and `--resnet_size` to choose the model size (options include ResNet-18 through ResNet-200). See [`resnet.py`](resnet.py) for the full list of options.

### Pre-trained model
You can download 190 MB pre-trained versions of ResNet-50 achieving 76.3% and 75.3% (respectively) top-1 single-crop accuracy here: [resnetv2_imagenet_checkpoint.tar.gz](http://download.tensorflow.org/models/official/resnetv2_imagenet_checkpoint.tar.gz), [resnetv1_imagenet_checkpoint.tar.gz](http://download.tensorflow.org/models/official/resnetv1_imagenet_checkpoint.tar.gz). Simply download and uncompress the file, and point the model to the extracted directory using the `--model_dir` flag.

Other versions and formats:

* [ResNet-v2-ImageNet Checkpoint](http://download.tensorflow.org/models/official/resnet_v2_imagenet_checkpoint.tar.gz)
* [ResNet-v2-ImageNet SavedModel](http://download.tensorflow.org/models/official/resnet_v2_imagenet_savedmodel.tar.gz)
* [ResNet-v1-ImageNet Checkpoint](http://download.tensorflow.org/models/official/resnet_v1_imagenet_checkpoint.tar.gz)
* [ResNet-v1-ImageNet SavedModel](http://download.tensorflow.org/models/official/resnet_v1_imagenet_savedmodel.tar.gz)

## Compute Devices
Training is accomplished using the DistributionStrategies API. (https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/distribute/README.md)

The appropriate distribution strategy is chosen based on the `--num_gpus` flag. By default this flag is one if TensorFlow is compiled with CUDA, and zero otherwise.

num_gpus:
+ 0:  Use OneDeviceStrategy and train on CPU.
+ 1:  Use OneDeviceStrategy and train on GPU.
+ 2+: Use MirroredStrategy (data parallelism) to distribute a batch between devices.
