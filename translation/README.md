
# 1. Problem 

This problem uses Attention mechanisms to do language translation.

## Disclaimer

This benchmark can be higher variance than expected. This implementation and results are still preliminary, modifications may be made in the near future. 


# 2. Directions
### Steps to configure machine

To setup the environment on Ubuntu 16.04 (16 CPUs, one P100, 100 GB disk), you can use these commands. This may vary on a different operating system or graphics card.


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

    # Clone and change work directory to the mlperf training repository
    ssh-keyscan github.com >> ~/.ssh/known_hosts
    git clone git@github.com:mlperf/training.git
    cd training


### Steps to download and verify data

Download the data using the following command. Note: this will require a recent version of tensorflow installed.
   
    bash download_data.sh
    


### Steps to run and time

Run the docker container, assuming you are at the root directory of mlperf/trainiing repository. 

    cd translation/tensorflow
    IMAGE=`sudo docker build . | tail -n 1 | awk '{print $3}'`
    SEED=1
    NOW=`date "+%F-%T"`
    sudo docker run \
        --runtime=nvidia \
        -v $(pwd)/translation/raw_data:/raw_data \
        -v $(pwd)/compliance:/mlperf/training/compliance \
        -e "MLPERF_COMPLIANCE_PKG=/mlperf/training/compliance" \
        -t -i $IMAGE "./run_and_time.sh" $SEED | tee benchmark-$NOW.log


# 3. Dataset/Environment
### Publication/Attribution
We use WMT17 ende training for tranding, and we evaluate using the WMT 2014 English-to-German translation task. See http://statmt.org/wmt17/translation-task.html for more information. 


### Data preprocessing
We combine all the files together and subtokenize the data into a vocabulary.  

### Training and test data separation
We use the train and evaluation sets provided explicitly by the authors.

### Training data order
We split the data into 100 blocks, and we shuffle internally in the blocks. 


# 4. Model
### Publication/Attribution

This is an implementation of the Transformer translation model as described in the [Attention is All You Need](https://arxiv.org/abs/1706.03762) paper. Based on the code provided by the authors: [Transformer code](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py) from [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor).

### Structure 

Transformer is a neural network architecture that solves sequence to sequence problems using attention mechanisms. Unlike traditional neural seq2seq models, Transformer does not involve recurrent connections. The attention mechanism learns dependencies between tokens in two sequences. Since attention weights apply to all tokens in the sequences, the Tranformer model is able to easily capture long-distance dependencies.

Transformer's overall structure follows the standard encoder-decoder pattern. The encoder uses self-attention to compute a representation of the input sequence. The decoder generates the output sequence one token at a time, taking the encoder output and previous decoder-outputted tokens as inputs.

The model also applies embeddings on the input and output tokens, and adds a constant positional encoding. The positional encoding adds information about the position of each token.


### Weight and bias initialization

We have two sets of weights to initialize: embeddings and the transformer network. 

The transformer network is initialized using the standard tensorflow variance initalizer. The embedding are initialized using the tensorflow random uniform initializer. 

### Loss function
Cross entropy loss while taking the padding into consideration, padding is not considered part of loss.

### Optimizer
We use the same optimizer as the original authors, which is the Adam Optimizer. We batch for a single P100 GPU of 4096. 

# 5. Quality

### Quality metric
We use the BLEU scores with data from [Attention is All You Need](https://arxiv.org/abs/1706.03762). 


    https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.en
    https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.de


### Quality target
We currently run to a BLEU score (uncased) of 25. This was picked as a cut-off point based on time. 


### Evaluation frequency
Evaluation of BLEU score is done after every epoch.


### Evaluation thoroughness
Evaluation uses all of `newstest2014.en`.
