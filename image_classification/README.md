# 1. Problem
This benchmark uses resnet v1.5 to classify images with a fork from
https://github.com/tensorflow/models/tree/master/official/resnet.

# 2. Directions
### Steps to configure machine
To setup the environment on Ubuntu 16.04 (16 CPUs, one P100, 100 GB disk), you
can use these commands. This may vary on a different operating system or
graphics card.

```bash
# Install docker
sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo apt-key fingerprint 0EBFCD88
sudo add-apt-repository    "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
sudo apt update
# sudo apt install docker-ce -y
sudo apt install docker-ce=18.03.0~ce-0~ubuntu -y --allow-downgrades

# Install nvidia-docker2
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey |   sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/ubuntu16.04/nvidia-docker.list |   sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt install nvidia-docker2 -y


sudo tee /etc/docker/daemon.json <<EOF
{
    "runtimes": {
        "nvidia": {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
EOF
sudo pkill -SIGHUP dockerd

sudo apt install -y bridge-utils
sudo service docker stop
sleep 1;
sudo iptables -t nat -F
sleep 1;
sudo ifconfig docker0 down
sleep 1;
sudo brctl delbr docker0
sleep 1;
sudo service docker start

ssh-keyscan github.com >> ~/.ssh/known_hosts
git clone git@github.com:mlperf/reference.git

```


### Steps to download and prepare data
The following [script](https://github.com/tensorflow/tpu/blob/master/tools/datasets/imagenet_to_gcs.py)
was used to create TFRecords from ImageNet data using instructions in the
[README](https://github.com/tensorflow/tpu/tree/master/tools/datasets#imagenet_to_gcspy).
TFRecords can be created directly from [ImageNet](http://image-net.org) or from
the .tar files downloaded from image-net.org.


### Steps to run and time
We assume that imagenet pre-processed has already been mounted at `/imn`.

```bash
    cd ~/reference/image_classification/tensorflow/
    IMAGE=`sudo docker build . | tail -n 1 | awk '{print $3}'`
    SEED=2
    NOW=`date "+%F-%T"`
    sudo docker run -v /imn:/imn --runtime=nvidia -t -i $IMAGE "./run_and_time.sh" $SEED | tee benchmark-$NOW.log

# For reference,

    $ ls /imn
    imagenet  lost+found
```


# 3. Dataset/Environment
### Publication/Attribution
We use Imagenet (http://image-net.org/):

    O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma,
    Z. Huang, A. Karpathy, A. Khosla, M. Bernstein, et al. Imagenet
    large scale visual recognition challenge. arXiv:1409.0575, 2014.


### Data preprocessing

There are two stages to the data processing. 1) Download and package images
for training that takes place once for a given dataset. 2) Processing as part
of training often called the input pipeline.

**Stage 1**

In the first stage, the images are not manipulated other than converting pngs
to jpegs and a few jpegs encoded with cmyk to rgb. In both instances the quality
saved is 100. The purpose is to get the images into a format that is faster
for reading, e.g. TFRecords or LMDB. Some frameworks suggest resizing images as
part of this phase to reduce I/O. Check the rules to see if resizing or other
manipulations are allowed and if this stage is on the clock.

**Stage 2**

The second stage takes place as part of training and includes cropping, apply
bounding boxes, and some basic color augmentation. The [reference model](https://github.com/mlperf/training/blob/master/image_classification/tensorflow/official/resnet/imagenet_preprocessing.py)
is to be followed.


### Training and test data separation
This is provided by the Imagenet dataset and original authors.

### Training data order
Each epoch goes over all the training data, shuffled every epoch.

### Test data order
We use all the data for evaluation. We don't provide an order for of data
traversal for evaluation.

# 4. Model
### Publication/Attribution
See the following papers for more background:

[1] [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.

[2] [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027) by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.


### Structure & Loss
In brief, this is a 50 layer v1 RNN. Refer to
[Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)
for the layer structure and loss function.


### Weight and bias initialization
Weight initialization is done as described here in
[Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852).


### Optimizer
We use a SGD Momentum based optimizer. The momentum and learning rate are scaled
based on the batch size.


# 5. Quality
### Quality metric
Percent of correct classifications on the Image Net test dataset.

### Quality target
We run to 0.749 accuracy (74.9% correct classifications).

### Evaluation frequency
We evaluate after every epoch.

### Evaluation thoroughness
Every test example is used each time.
