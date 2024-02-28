# 1. Problem 
This benchmark represents a multi-class node classification task in a heterogenous graph using  the [IGB Heterogeneous Dataset](https://github.com/IllinoisGraphBenchmark/IGB-Datasets) named IGBH-Full. The task is carried out using a [GAT](https://arxiv.org/abs/1710.10903) model based on the [Relational Graph Attention Networks](https://arxiv.org/abs/1904.05811) paper.

The reference implementation is based on [graphlearn-for-pytorch (GLT)](https://github.com/alibaba/graphlearn-for-pytorch).

# 2. Directions
### Steps to configure machine

#### 1. Clone the repository:
```bash
git clone https://github.com/alibaba/graphlearn-for-pytorch
```

or
```bash
git clone https://github.com/mlcommons/training.git
```
once `GNN node classification` is merged into `mlcommons/training`.

#### 2. Build the docker image:

If you cloned the `graphlearn-for-pytorch` repository:
```bash
cd graphlearn-for-pytorch/examples/igbh/
docker build -f Dockerfile -t training_gnn:latest .
```

If you cloned the `mlcommons/training` repository:
```bash
cd training/gnn_node_classification/
docker build -f Dockerfile -t training_gnn:latest .
```


### Steps to download and verify data
Download the dataset:
```bash

bash download_igbh_full.sh
```

Before training, generate the seeds for training and validation:
```bash
python split_seeds.py --dataset_size='full'
```

The size of the `IGBH-Full` dataset is 2.2 TB. If you want to test with 
the `tiny`, `small` or `medium` datasets, the download procedure is included 
in the training script.

### Steps to run and time

#### Single-node Training

The original graph is in the `COO` format and the feature is in the FP32 format. The training script will transform the graph from `COO` to `CSC` and convert the feature to FP16, which could be time consuming due to the graph scale. We provide a script to convert the graph layout from `COO` to `CSC` and persist the feature in FP16 format:

```bash
python compress_graph.py --dataset_size='full' --layout='CSC' --use_fp16
```

To train the model using multiple GPUs:
```bash
CUDA_VISIBLE_DEVICES=0,1 python train_rgnn_multi_gpu.py --model='rgat' --dataset_size='full' --layout='CSC' --use_fp16
```
The number of training processes is equal to the number of GPUS. Option `--pin_feature` decides if the feature data will be pinned in host memory, which enables zero-copy feature access from GPU, but will incur extra memory costs.


#### Distributed Training

##### 1. Data Partitioning
To partition the dataset (including both the topology and feature):
```bash
python partition.py --dataset_size='full' --num_partitions=2 --use_fp16 --layout='CSC'
```
The above script will partition the dataset into two parts, convert the feature into
the FP16 format, and transform the graph layout from `COO` to `CSC`.

We suggest using a distributed file system to store the partitioned data, such as HDFS or NFS, suhc that partitioned data can be accessed by all training nodes.

##### 2. Two-stage Data Partitioning
To speed up the partitioning process, GLT also supports two-stage partitioning, which splits the process of topology partitioning and feature partitioning. After the topology partitioning is executed in a single node, the feature partitioning process can be conducted in each training node in parallel to speedup the partitioning process.

The topology partitioning is conducted by executing:
```bash
python partition.py --dataset_size='full' --num_partitions=2 --layout='CSC' --with_feature=0 
```

The feature partitioning in conducted in each training node:
```bash
# node 0 which holds partition 0:
python build_partition_feature.py --dataset_size='full' --use_fp16 --in_memory=0 --partition_idx=0

# node 1 which holds partition 1:
python build_partition_feature.py --dataset_size='full' --use_fp16 --in_memory=0 --partition_idx=1
```

##### 2. Model Training
The number of partitions and number of training nodes must be the same. In each training node, the model can be trained using the following command:

```bash
# node 0:
CUDA_VISIBLE_DEVICES=0,1 python dist_train_rgnn.py --num_nodes=2 --node_rank=0 --num_training_procs=2 --master_addr=master_address_ip --model='rgat' --dataset_size='full' --layout='CSC'

# node 1:
CUDA_VISIBLE_DEVICES=2,3 python dist_train_rgnn.py --num_nodes=2 --node_rank=1 --num_training_procs=2 --master_addr=master_address_ip --model='rgat' --dataset_size='full' --layout='CSC'
```
The above script assumes that the training nodes are equipped with 2 GPUs and the number of training processes is equal to the number of GPUs. Each training process has a corresponding
sampling process using the same GPU. 

The `master_address_ip` should be replaced with the actual IP address of the master node. The `--pin_feature` option decides if the feature data will be pinned in host memory, which enables zero-copy feature access from GPU but will incur extra memory costs.


 We recommend separating the sampling and training processes to different GPUs to achieve better performance. To seperate the GPU used by sampling and training processes, please add `--split_training_sampling` and set `--num_training_procs` as half of the number of devices:

```bash
# node 0:
CUDA_VISIBLE_DEVICES=0,1 python dist_train_rgnn.py --num_nodes=2 --node_rank=0 --num_training_procs=1 --master_addr=localhost --model='rgat' --dataset_size='full' --layout='CSC' --split_training_sampling

# node 1:
CUDA_VISIBLE_DEVICES=2,3 python dist_train_rgnn.py --num_nodes=2 --node_rank=1 --num_training_procs=1 --master_addr=localhost --model='rgat' --dataset_size='full' --layout='CSC' --split_training_sampling
```
The above script uses one GPU for training and another for sampling in each node.



# 3. Dataset/Environment
### Publication/Attribution
Arpandeep Khatua, Vikram Sharma Mailthody, Bhagyashree Taleka, Tengfei Ma, Xiang Song, and Wen-mei Hwu. 2023. IGB: Addressing The Gaps In Labeling, Features, Heterogeneity, and Size of Public Graph Datasets for Deep Learning Research. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '23). Association for Computing Machinery, New York, NY, USA, 4284–4295. https://doi.org/10.1145/3580305.3599843

### Data preprocessing
The original graph is in the `COO` format and the feature is in FP32 format. It is allowed to transform the graph from `COO` to `CSC` and convert the feature to FP16 (supported by the training script).

### Training and test data separation
The training and validation data are selected from the labeled ``paper`` nodes from the dataset and are generated by `split_seeds.py`. Differnet random seeds will result in different training and test data.

### Training data order
Randomly.

### Test data order
Randomly.

# 4. Model
### Publication/Attribution
Dan Busbridge and Dane Sherburn and Pietro Cavallo and Nils Y. Hammerla, Relational Graph Attention Networks, 2019, https://arxiv.org/abs/1904.05811

### List of layers 
Three-layer RGAT model

### Loss function
CrossEntropyLoss

### Optimizer
Adam

# 5. Quality
### Quality metric
The validation accuracy is the target quality metric.
### Quality target
0.72
### Evaluation frequency
4,730,280 training seeds (5% of the entire training seeds, evaluated every 0.05 epoch)
### Evaluation thoroughness
788,380 validation seeds

# 6. Contributors
This benchmark is a collaborative effort with contributions from Alibaba, Intel, and Nvidia:

- Alibaba: Li Su, Baole Ai, Wenting Shen, Shuxian Hu, Wenyuan Yu, Yong Li
- Nvidia: Yunzhou (David) Liu, Kyle Kranen, Shriya Palasamudram
- Intel: Kaixuan Liu, Hesham Mostafa, Sasikanth Avancha, Keith Achorn, Radha Giduthuri, Deepak Canchi