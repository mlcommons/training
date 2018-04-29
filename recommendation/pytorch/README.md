# 1. Problem
This task benchmarks recommendation with implicit feedback on the [MovieLens 20 Million (ml-20m) dataset](https://grouplens.org/datasets/movielens/20m/) with a [Neural Collaborative Filtering](http://dl.acm.org/citation.cfm?id=3052569) model.
The model trains on binary information about whether or not a user interacted with a specific item.

# 2. Directions
### Steps to configure machine

#### From Source

1. Install [PyTorch v0.4.0](https://github.com/pytorch/pytorch/tree/v0.4.0)
2. Install `unzip` and `curl`

```bash
sudo apt-get install unzip curl
```
3. Checkout the MLPerf repo
```bash
git clone https://github.com/mlperf/reference.git
```

4. Install other python packages

```bash
cd reference/recommendation/pytorch
pip install -r requirements.txt
```

#### From Docker

1. Checkout the MLPerf repo

```bash
git clone https://github.com/mlperf/reference.git
```

2. Get the docker image for the recommendation task

```bash
# Pull from Docker Hub
docker pull ???
```

or

```bash
# Build from Dockerfile
cd reference/recommendation/pytorch
sudo docker build -t ??? .
```

### Steps to download and verify data

You can download and verify the dataset by running the `download_dataset.sh` and `verify_dataset.sh` scripts in the parent directory:

```bash
# Creates ml-20.zip
source ../download_dataset.sh
# Confirms the MD5 checksum of ml-20.zip
source ../verify_dataset.sh
```

### Steps to run and time

#### From Source

Run the `run_and_time.sh` script with an integer seed value between 1 and 5

```bash
source run_and_time.sh SEED
```

#### Docker Image

```bash
sudo nvidia-docker run -i -t --rm --ipc=host \
    --mount "type=bind,source=$(pwd),destination=/mlperf/experiment" \
    ??? SEED
```

# 3. Dataset/Environment
### Publication/Attribution
Cite paper describing dataset plus any additional attribution requested by dataset authors

### Data preprocessing
What preprocessing is done to the the dataset?

### Training and test data separation
Positive training examples are all but the last item each user rated.
Negative training examples are randomly selected from the unrated items for each user.

The last item each user rated is used as a positive example in the test set.
A fixed set of 100 unrated items are also selected to calculate hit rate at 10 for predicting the test item.

### Training data order
Data is traversed randomly with 4 negative examples selected on average for every positive example.


# 4. Model
### Publication/Attribution
Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu and Tat-Seng Chua (2017). [Neural Collaborative Filtering](http://dl.acm.org/citation.cfm?id=3052569). In Proceedings of WWW '17, Perth, Australia, April 03-07, 2017.

The author's original code is available at [hexiangnan/neural_collaborative_filtering](https://github.com/hexiangnan/neural_collaborative_filtering).
### List of layers
Brief summary of structure of model
### Weight and bias initialization
How are weights and biases intialized
### Loss function
Name/description of loss function used
### Optimizer
[Adam](https://arxiv.org/abs/1412.6980) with a learning rate of 0.0005 and batch size of 16,384 examples for at most 20 epochs.

# 5. Quality
### Quality metric
Hit rate at 10 (HR@10) with 100 negative items.

### Quality target
HR@10: 0.9562

### Evaluation frequency
After every epoch through the training data.

### Evaluation thoroughness

Every users last item rated, i.e. all held out positive examples.
