# 1. Problem 
Sentiment Analysis is a binary classification task. It predicts positive or negative sentiment using raw user text. The IMDB dataset is used for this benchmark.
# 2. Directions
### Steps to configure machine
TODO - Need to update this.
### Steps to set up the Paddle environment
A docker image with PaddlePaddle pre-installed can be pulled using:
```
docker pull sidgoyal78/paddle:benchmark12042018
```

To run the docker container, use:
```
nvidia-docker run -it -v `pwd`:/paddle sidgoyal78/paddle:benchmark12042018 /bin/bash
```

Inside the container, use `cd paddle` to go to the correct directory.

### Steps to download and verify data
It is necessary to run the download and verify scripts inside the docker container.

The `download_dataset.py` script downloads the IMDB dataset into `/root/.cache/paddle` directory.
The `verify_dataset.py` script ensures that the MD5 hash of the downloaded dataset matches the reference hash.

The python scripts are invoked using the following shell scripts for convenience.

```
./download_dataset.sh
./verify_dataset.sh
```

### Steps to run and time

#### Run and Time
Run the bash script `run_and_time.sh` with a seed argument to train a convolution based model, using:
```
./run_and_time.sh <seed>
```

Training stops when the model reaches the pre-defined target quality.

# 3. Dataset/Environment
### Publication/Attribution
[IMDB Dataset](http://ai.stanford.edu/~amaas/data/sentiment/) provides 50000 movie reviews for sentiment analysis.

Maas, A. L.; Daly, R. E.; Pham, P. T.; Huang, D.; Ng, A. Y. & Potts, C. (2011), Learning Word Vectors for Sentiment Analysis, in 'Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies', Association for Computational Linguistics, Portland, Oregon, USA, pp. 142--150.
### Data preprocessing
The dataset isn't preprocessed in any way.
### Training and test data separation
The entire dataset is split into training and test sets. 25000 reviews are used for training and 25000 are used for validation.
This split is pre-determined and cannot be modified.
### Training data order
Training data is traversed in a randomized order.
### Test data order
Test data is evaluated in a fixed order.
# 4. Model
### Publication/Attribution
Johnson, R. and Zhang, T. (2014), Effective use of word order for text categorization with convolutional neural networks, CoRR abs/1412.1058. 
### List of layers
The model consists of an embedding layer followed by two sequence convolution layers and cross entropy cost layer.
### Weight and bias initialization
TODO: Siddharth can you confirm the initialization?
### Loss function
Cross entropy loss function is used for computing the loss.
### Optimizer
Adam is used for optimization.
# 5. Quality
### Quality metric
Average accuracy for all samples in the test set.
### Quality target
Average accuracy of 90.6%
### Evaluation frequency
All test samples are evaluated once per epoch.
### Evaluation thoroughness
All test samples are evaluated once per epoch.
