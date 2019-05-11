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
git clone https://github.com/mlperf/training.git
```

4. Install other python packages

```bash
cd training/recommendation/pytorch
pip install -r requirements.txt
```

#### From Docker

1. Checkout the MLPerf repo

```bash
git clone https://github.com/mlperf/training.git
```
2. Install CUDA and Docker

```bash
source training/install_cuda_docker.sh
```

3. Build the docker image for the recommendation task

```bash
# Build from Dockerfile
cd training/recommendation/pytorch
sudo docker build -t mlperf/recommendation:v0.6 .
```

### Steps to run and time

#### Getting the expanded dataset

The original ML-20M dataset is expanded to 16x more users and 32x more items using the code from the `data_generation` directory in the `mlperf/training` repo.
To obtain the expanded dataset, follow the instructions from section 
`Running instructions for the recommendation benchmark` from the README file in the 
`data_generation/fractal_graph_expansions` directory.

#### Run the Docker container

```bash
nvidia-docker run --rm -it --ipc=host --network=host -v /my_data_dir:/data/cache mlperf/recommendation:v0.6 /bin/bash
```

#### Generating the negative test samples

Assuming the expanded dataset is visible in the container under `/data/cache/ml-20mx16x32` 
directory, run inside the container:

```
python convert.py /data/cache/ml-20mx16x32 --seed 0
```

#### Running the training

Assuming the expanded dataset together with the generated test negative samples files are 
visible in the container under `/data/cache/ml-20mx16x32` directory, run inside the container:

```
./run_and_time.sh <SEED>
```

Seed 0 has been shown to converge deterministically.

**Note** The current data generation pipeline is run on CPU and is currently
*very* memory-intensive. It is recommended to run using a host VM with at least
400 GB of memory. This is because the entire dataset is read into memory and
manipulated, in order to generate negative samples and perform global shuffling.

Work is planned to alleviate these requirements. Pull requests are welcome.


# 3. Dataset/Environment
### Publication/Attribution
Harper, F. M. & Konstan, J. A. (2015), 'The MovieLens Datasets: History and Context', ACM Trans. Interact. Intell. Syst. 5(4), 19:1--19:19.

### Data preprocessing

1. Unzip
2. Remove users with less than 20 reviews
3. Create training and test data separation described below

### Training and test data separation
Positive training examples are all but the last item each user rated.
Negative training examples are randomly selected from the unrated items for each user.

The last item each user rated is used as a positive example in the test set. For
the large synthetic dataset, there is no notion of time, since the data points
are randomly generated. For each user, one item is chosen to be used for the
test set.

A fixed set of 999 unrated items are also selected to calculate hit rate at 10 for predicting the test item.

### Training data order
Data is traversed randomly with 4 negative examples selected on average for every positive example.


# 4. Model
### Publication/Attribution
Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu and Tat-Seng Chua (2017). [Neural Collaborative Filtering](http://dl.acm.org/citation.cfm?id=3052569). In Proceedings of WWW '17, Perth, Australia, April 03-07, 2017.

The author's original code is available at [hexiangnan/neural_collaborative_filtering](https://github.com/hexiangnan/neural_collaborative_filtering).

# 5. Quality
### Quality metric
Hit rate at 10 (HR@10) with 999 negative items.

### Quality target
HR@10: 0.51  (ml-1b)

### Evaluation frequency
After every epoch through the training data.

### Evaluation thoroughness

Every users last item rated, i.e. all held out positive examples.
