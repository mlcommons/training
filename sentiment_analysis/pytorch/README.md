# Architecture.

Achitecture mimics the original one impelented in PaddlePaddle:
https://github.com/mlperf/reference/blob/master/sentiment_analysis/paddle/train.py

## Brief overview:
1. Embedding layer 
2. Two independent conv + relu + max pool layers with 3x3 and 4x4 filters
3. Fully-connected + softmax

### 1. Embedding layer

Embedding layer is nn.Embedding layer with 
self.vocab_size and self.embedding_size
self.vocab_size is the value computed by IMDB_dataset() function.
Initial setup:
self.embedding_size = 1024


### 2. Two independent conv + relu + max pool layers with 3x3 and 4x4 filters.

### 3. Fully-connected layer takes two concatinated vectors (outputs of conv3x3 and conv4x4)
We apply log-softmax to the output of fully-connected layer and 
get the array of size 2: probabilities of positive and negative classes.



# Dataset loader

Implemented in IMDB_dataset() function - a Pytorch generator for IMDB dataset.




# Installation guide:

## 1. Install conda:

You have to install conda, 
pip doesn't work correctly with torch on Linux right now.

```
wget "https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh"
bash Anaconda-latest-Linux-x86_64.sh
```

## 2. Create and activate virtual environment using conda:
```
conda create -n torchenv_conda python=3.6
source activate torchenv_conda
```

## 3. Install torch, torchvision and torchtext

Without CUDA (CPU only):
```
conda install pytorch-cpu torchvision-cpu -c pytorch
pip install cython
pip install msgpack
pip install torchtext
```
With CUDA 9.0:
```
conda create -n torchenv_conda_gpu_cuda_9 python=3.6
source activate torchenv_conda_gpu_cuda_9
conda install pytorch torchvision cuda90 -c pytorch
pip install cython
pip install msgpack
pip install torchtext
```

If you need to install pytorch for other version of CUDA:

"Get started" section on the main page of https://pytorch.org/

## 4. Run the script:

Remainder: you have to be loggined into the virtual enviroment you created.

Run: 
```
python train.py
```
It will work just on fine on CPU only.

While trying to run the script on GPU, you will get an error "long tensor expected but got cuda.longTensor"

It happens because nn.Embedding layer in pytorch doesn't accept long tensors:
https://pytorch.org/docs/master/nn.html#embedding
"Input: LongTensor of arbitrary shape containing the indices to extract"

Similar problem:
https://github.com/pytorch/pytorch/issues/7236

So, you can just manually disable CUDA in main function (set flag use_cuda to False)


Ways to go over that problem:
1) Write your own embedding layer
2) Get rid of the embedding layer

Getting rid of the embedding layer is a questionary move because our goal is to mimic the PaddlePaddle implementation already presented in the repo.
We can ask to change the configuration in both Pytorch and PaddlePaddle configurations close to the paper it was supposed to be close:
https://arxiv.org/abs/1412.1058

The full explanation can be found in the official repo: https://github.com/mlperf/reference/tree/master/sentiment_analysis


