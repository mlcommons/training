# Sentiment Analysis
Sentiment analysis for MLPerf benchmark

## Dataset
IMDB Large Movie Reviews dataset 

## Model
* Conv model with Sequence convolution
* Stacked BiDirectional LSTM model

## Running the code
The steps below are based on Docker.

### Setting up Paddle environment
A docker image with PaddlePaddle pre-installed can be pulled using:
```
docker pull sidgoyal78/paddle:benchmark12042018
```

To run the docker container use:
```
docker run -it -v `pwd`:/paddle sidgoyal78/paddle:benchmark12042018 /bin/bash
```

Inside the container, use `cd paddle` to go to the correct directory.

### Downloading and verifying the dataset

Run the bash script `download_and_verify.sh` to download the IMDB dataset and verify the MD5 checksum.

### Training a model 

Run the bash script `run_and_time.sh` with a seed value to train a conv-based model, using:
```
./run_and_time.sh 100
```

The model will train until the validation accuracy reaches 90.6 (default).
