
# 1. Problem 


# 2. Directions
### Steps to configure machine
# Install docker
sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common

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



### Steps to download and verify data

Download the data:
   
    python3 data_download.py --raw_dir raw_data
    


### Steps to run and time

Run the docker:

    cd ~/reference/transformer/tensorflow
    IMAGE=`sudo docker build . | tail -n 1 | awk '{print $3}'`
    SEED=1
    NOW=`date "+%F-%T"`
    sudo docker run -v $HOME/reference/transformer/raw_data:/raw_data --runtime=nvidia -t -i $IMAGE "./run_and_time.sh" $SEED | tee benchmark-$NOW.log


# 3. Dataset/Environment
### Publication/Attribution
Cite paper describing dataset plus any additional attribution requested by dataset authors
### Data preprocessing
What preprocessing is done to the the dataset? 
### Training and test data separation
How is the test set extracted?
### Training data order
In what order is the training data traversed?
### Test data order
In what order is the test data traversed?
### Simulation environment (RL models only)
Describe simulation environment briefly, if applicable. 
# 4. Model
### Publication/Attribution

This is an implementation of the Transformer translation model as described in the [Attention is All You Need](https://arxiv.org/abs/1706.03762) paper. Based on the code provided by the authors: [Transformer code](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py) from [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor).

Transformer is a neural network architecture that solves sequence to sequence problems using attention mechanisms. Unlike traditional neural seq2seq models, Transformer does not involve recurrent connections. The attention mechanism learns dependencies between tokens in two sequences. Since attention weights apply to all tokens in the sequences, the Tranformer model is able to easily capture long-distance depedencies.

Transformer's overall structure follows the standard encoder-decoder pattern. The encoder uses self-attention to compute a representation of the input sequence. The decoder generates the output sequence one token at a time, taking the encoder output and previous decoder-outputted tokens as inputs.

The model also applies embeddings on the input and output tokens, and adds a constant positional encoding. The positional encoding adds information about the position of each token.

### List of layers 
Brief summary of structure of model
### Weight and bias initialization
How are weights and biases intialized
### Loss function
Name/description of loss function used
### Optimizer
Name of optimzier used
# 5. Quality
### Quality metric
What is the target quality metric
### Quality target
What is the numeric quality target
### Evaluation frequency
How many training items between quality evaluations (typically all, evaluated every epoch)
### Evaluation thoroughness
How many test items per quality evaluation (typically all)
