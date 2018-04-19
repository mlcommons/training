# Transformer Translation Model
This is an implementation of the Transformer translation model as described in the [Attention is All You Need](https://arxiv.org/abs/1706.03762) paper. Based on the code provided by the authors: [Transformer code](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py) from [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor).

Transformer is a neural network architecture that solves sequence to sequence problems using attention mechanisms. Unlike traditional neural seq2seq models, Transformer does not involve recurrent connections. The attention mechanism learns dependencies between tokens in two sequences. Since attention weights apply to all tokens in the sequences, the Tranformer model is able to easily capture long-distance depedencies (an issue that appears in recurrent models).

Transformer's overall structure follows the standard encoder-decoder pattern. The encoder uses self-attention to compute a representation of the input sequence. The decoder generates the output sequence one token at a time, taking the encoder output and previous decoder-outputted tokens as inputs.

The model also applies embeddings on the input and output tokens, and adds a constant positional encoding. The positional encoding adds information about the position of each token.

## Contents
* [Walkthrough](#walkthrough)
* [Benchmarks](#benchmarks)
  * [Training times](#training-times)
  * [Evaluation results](#evaluation-results)
* [Detailed instructions](#detailed-instructions)
* [Implementation overview](#implementation-overview)
  * [Model Definition](#model-definition)
  * [Model Estimator](#model-estimator)
  * [Other scripts](#other-scripts)
* [Term definitions](#term-definitions)


## Walkthrough

Below are the commands for running the Transformer model. See the [Detailed instrutions](#detailed-instructions) for more details on running the model.

```
PARAMS=big
DATA_DIR=$HOME/transformer/data
MODEL_DIR=$HOME/transformer/model_$PARAMS

# Download dataset for computing BLEU score reported in the paper
wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.en
wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.de

# Download training/evaluation datasets
python data_download.py --data_dir=$DATA_DIR

# Train the model for 10 epochs, and evaluate after every epoch.
python transformer_main.py --data_dir=$DATA_DIR --model_dir=$MODEL_DIR \
   --params=$PARAMS  # TODO: add BLEU flags

# Run during training in a separate process to get continuous updates,
# or after training is complete.
tensorboard --logdir=$MODEL_DIR
```

## Benchmarks
### Training times

Currently, both big and base params run on a single GPU. The measurements below
are reported from running the model on a P100 GPU.

Params | batches/sec | batches per epoch | time per epoch
--- | --- | --- | ---
base | 4.8 | 83244 | 4 hr
big | 1.1 | 41365 | 10 hr

### Evaluation results
Below are the case-insensitive BLEU scores after 10 epochs.

Params | Score
--- | --- |
base | 20.5
big | 26.1



## Detailed instructions


0. ### Export variables (optional)

   Export the following variables, or modify the values in each of the snippets below:
   ```
   PARAMS=big
   DATA_DIR=$HOME/transformer/data
   MODEL_DIR=$HOME/transformer/model_$PARAMS
   ```

1. ### Download and preprocess datasets for training and evaluation

   [`data_download.py`](data_download.py) downloads and preprocesses the training and evaluation WMT datasets. After the data is downloaded and extracted, the training data is used to generate a vocabulary of subtokens. The evaluation and training strings are tokenized, and the resulting data is sharded, shuffled, and saved as TFRecords.

   1.75GB of compressed data will be downloaded. In total, the raw files (compressed, extracted, and combined files) take up 8.4GB of disk space. The resulting TFRecord and vocabulary files are 722MB. The script takes around 40 minutes to run, with the bulk of the time spent downloading and ~15 minutes spent on preprocessing.

   Command to run:
   ```
   python data_download.py --data_dir=$DATA_DIR
   ```

   Arguments:
   * `--data_dir`: Path where the preprocessed TFRecord data, and vocab file will be saved.
   * Use the `--help` or `-h` flag to get a full list of possible arguments.

2. ### Model training and evaluation

   [`transformer_main.py`](transformer_main.py) creates a Transformer model graph using Tensorflow Estimator, and trains it.

   Command to run:
   ```
   python transformer_main.py --data_dir=$DATA_DIR --model_dir=$MODEL_DIR --params=$PARAMS
   ```

   Arguments:
   * `--data_dir`: This should be set to the same directory given to the `data_download`'s `data_dir` argument.
   * `--model_dir`: Directory to save Transformer model training checkpoints.
   * `--params`: Parameter set to use when creating and training the model. Options are `base` and `big` (default).
   * Use the `--help` or `-h` flag to get a full list of possible arguments.

   #### Training Schedule

   By default, the model will train for 10 epochs, and evaluate after every epoch. The training schedule may be defined through the flags:
   * Training with epochs (default):
     * `--train_epochs`: The total number of complete passes to make through the dataset
     * `--epochs_between_eval`: The number of epochs to train between evaluations.
   * Training with steps:
     * `--train_steps`: sets the total number of training steps to run.
     * `--steps_between_eval`: Number of training steps to run between evaluations.

   Only one of `train_epochs` or `train_steps` may be set. Since the default option is to evaluate the model after training for an epoch, it may take 4 or more hours between model evaluations. To get more frequent evaluations, use the flags `--train_steps=250000 --steps_between_eval=1000`.

   Note: At the beginning of each training session, the training dataset is reloaded and shuffled. Stopping the training before completing an epoch may result in worse model quality, due to the chance that some examples may be seen more than others. Therefore, it is recommended to use epochs when the model quality is important.

   #### Compute official BLEU score during model evaluation

   (TODO)

   #### Tensorboard
   Training and evaluation metrics (loss, accuracy, approximate BLEU score, etc.) are logged, and can be displayed in the browser using Tensorboard.
   ```
   tensorboard --logdir=$MODEL_DIR
   ```
   The values are displayed at [localhost:6006].

3. ### Translate using the model
   (TODO)

4. ### Compute official BLEU score
   (TODO)

## Implementation overview

A brief look at each component in the code:

### Model Definition
The [model](model) subdirectory contains the implementation of the Transformer model.
* [transformer.py](model/transformer.py): Defines the transformer model and its encoder/decoder layer stacks.
* [embedding_layer.py](model/embedding_layer.py): Contains the layer that calculates the embeddings. The embedding weights are also used to calculate the pre-softmax probabilities from the decoder output.
* [attention_layer.py](model/attention_layer.py): Defines the multi-headed and self attention layers that are used in the encoder/decoder stacks.
* [ffn_layer.py](model/ffn_layer.py): Defines the feedforward network that is used in the encoder/decoder stacks. The network is composed of 2 fully connected layers.

Aside from the model and layers code, [model_params.py](model/model_params.py) contains the parameters used for the big and base models. [model_utils.py](model/model_utils.py) defines some helper functions used in the model (calculating padding, bias, etc.).


### Model Estimator
[`transformer_main.py`](model/transformer.py) creates an `Estimator` to train and evaluate the model.

Helper functions:
* [`utils/dataset.py`](utils/dataset.py): contains functions for creating a `dataset` that is passed to the `Estimator`.
* [`utils/metrics.py`](utils/metrics.py): defines metrics functions used by the `Estimator` to evaluate the

### Other scripts

Aside from the main file to train the Transformer model, we provide other scripts for using the model or downloading the data:

#### Data download and preprocessing

[`data_download.py`](data_download.py) downloads and extracts data, then uses `Subtokenizer` to tokenize strings into arrays of int IDs. The int arrays are converted to `tf.Examples` and saved in the `tf.RecordDataset` format.

 The data is downloaded from the Workshop of Machine Transtion (WMT) [news translation task](http://www.statmt.org/wmt17/translation-task.html). The following datasets are used:

 * Europarl v7
 * Common Crawl corpus
 * News Commentary v12

 See the [download section](http://www.statmt.org/wmt17/translation-task.html#download) to explore the raw datasets. The parameters in this model are tuned to fit the English-German translation data, so the EN-DE texts are extracted from the downloaded compressed files.

The text is transformed into arrays of integer IDs using the `Subtokenizer` defined in [`utils/tokenizer.py`](util/tokenizer.py). During initialization of the `Subtokenizer`, the raw training data is used to generate a vocabulary list containing common subtokens.

The target vocabulary size of the WMT dataset is 32,768. The set of subtokens is found through binary search on the minimum number of times a subtoken appears in the data. The actual vocabulary size is 33,708, and is stored in a 324kB file.

#### Translation
Translation is defined in [`translate.py`](translate.py). First, `Subtokenizer` tokenizes the input. The vocabulary file is the same used to tokenize the training/eval files. Next, beam search is used to find the combination of tokens that maximizes the probability outputted by the model decoder. The tokens are then converted back to strings with `Subtokenizer`.

#### BLEU computation
[`compute_bleu.py`](compute_bleu.py): (TODO)

## Term definitions

**Steps / Epochs**:
* Step: unit for processing a single batch of data
* Epoch: a complete run through the dataset

Example: Consider a training a dataset with 100 examples that is divided into 20 batches with 5 examples per batch. A single training step trains the model on one batch. After 20 training steps, the model will have trained on every batch in the dataset, or one epoch.

**Subtoken**: Words are referred as tokens, and parts of words are referred as 'subtokens'. For example, the word 'inclined' may be split into `['incline', 'd_']`. The '\_' indicates the end of the token. The subtoken vocabulary list is guaranteed to contain the alphabet (including numbers and special characters), so all words can be tokenized.
