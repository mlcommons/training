# V1.0 Dataset and Training

# Location of the input files 

This [Google Drive location](https://drive.google.com/drive/folders/1oQF4diVHNPCclykwdvQJw8n_VIWwV0PT) contains the following.
* tf1_ckpt folder: contains checkpoint files 
  - model.ckpt-28252.data-00000-of-00001
  - model.ckpt-28252.index
  - model.ckpt-28252.meta

* tf2_ckpt folder: contains TF2 checkpoint files
  - model.ckpt-28252.data-00000-of-00001
  - model.ckpt-28252.index

* bert_config.json: Config file which specifies the hyperparameters of the model
* enwiki-20200101-pages-articles-multistream.xml.bz2 : Compressed file containing wiki data
* enwiki-20200101-pages-articles-multistream.xml.bz2.md5sum: md5sum hash for the `enwiki-20200101-pages-articles-multistream.xml.bz2` file
* License.txt
* vocab.txt: Contains WordPiece to id mapping

Alternatively, TF2 checkpoint can also be generated using [tf2_encoder_checkpoint_converter.py](https://github.com/tensorflow/models/blob/master/official/nlp/bert/tf2_encoder_checkpoint_converter.py) and TF1 checkpoint

```shell
python3 tf2_encoder_checkpoint_converter.py \
  --bert_config_file=<path to bert_config.json> \
  --checkpoint_to_convert=<path to tf1 model.ckpt-28252> \
  --converted_checkpoint_path=<path to output tf2 model checkpoint>

```
Note that the checkpoint converter removes optimizer slot variables, so the resulting TF2 checkpoint is only about 1/3 size of the TF1 checkpoint.


# Download and preprocess datasets

The dataset was prepared using Python 3.7.6, nltk 3.4.5 and the [tensorflow/tensorflow:1.15.2-gpu](https://hub.docker.com/layers/tensorflow/tensorflow/1.15.2-gpu/images/sha256-da7b6c8a63bdafa77864e7e874664acfe939fdc140cb99940610c34b8c461cd0?context=explore) docker image.

Files after the download, uncompress, extract, clean up and dataset seperation steps are providedat a [Google Drive location](https://drive.google.com/corp/drive/u/0/folders/1cywmDnAsrP5-2vsr8GDc6QUc7VWe-M3v). The main reason is that, WikiExtractor.py replaces some of the tags present in XML such as {CURRENTDAY}, {CURRENTMONTHNAMEGEN} with the current values obtained from time.strftime ([code](https://github.com/attardi/wikiextractor/blob/e4abb4cbd019b0257824ee47c23dd163919b731b/WikiExtractor.py#L632)). Hence, one might see slighly different preprocessed files after the WikiExtractor.py file is invoked. This means the md5sum hashes of these files will also be different each time WikiExtractor is called.

### Files in <bert>/results directory:

| File                | Size (bytes) | MD5                              |
|---------------------|  ----------: |----------------------------------|
| eval.md5            | 330000 | 71a58382a68947e93e88aa0d42431b6c |
| eval.txt            | 32851144 | 2a220f790517261547b1b45ed3ada07a |
| part-00000-of-00500 | 27150902 | a64a7c31eff5cd38ae6d94f7a6229dab |
| part-00001-of-00500 | 27198569 | 549a9ed4f805257245bec936563abfd0 |
| part-00002-of-00500 | 27395616 | 1a1366ddfc03aef9d41ce552ee247abf |
| ... | | |
| part-00497-of-00500 | 24775043 | 66835aa75d4855f2e678e8f3d73812e9 |
| part-00498-of-00500 | 24575505 | e6d68a7632e9f4aa1a94128cce556dc9 |
| part-00499-of-00500 | 21873644 | b3b087ad24e3770d879a351664cebc5a |


Each of `part-00xxx-of-00500` and eval.txt contains one sentence of an article in one line and different articles separated by blank line.

The details of how these files were prepared around Feb. 10, 2020 can be found in [dataset.md](./dataset.md).

## Generate the TFRecords for Wiki dataset

The [create_pretraining_data.py](./create_pretraining_data.py) script tokenizes the words in a sentence using [tokenization.py](./tokenization.py) and `vocab.txt` file. Then, random tokens are masked using the strategy where 80% of time, the selected random tokens are replaced by `[MASK]` tokens, 10% by a random word and the remaining 10% left as is. This process is repeated for `dupe_factor` number of times, where an example with `dupe_factor` number of different masks are generated and written to TFRecords.

```shell
# Generate one TFRecord for each part-00XXX-of-00500 file. The following command is for generating one corresponding TFRecord

python3 create_pretraining_data.py \
   --input_file=<path to ./results of previous step>/part-00XXX-of-00500 \
   --output_file=<tfrecord dir>/part-00XXX-of-00500 \
   --vocab_file=<path to downloaded vocab.txt> \
   --do_lower_case=True \
   --max_seq_length=512 \
   --max_predictions_per_seq=76 \
   --masked_lm_prob=0.15 \
   --random_seed=12345 \
   --dupe_factor=10
```
where

- `dupe_factor`:  Number of times to duplicate the dataset and write to TFrecords. Each of the duplicate example has a different random mask
- `max_sequence_length`: Maximum number of tokens to be present in a single example
-`max_predictions_per_seq`: Maximum number of tokens that can be masked per example
- `masked_lm_prob`: Masked LM Probability
- `do_lower_case`: Whether the tokens are to be converted to lower case or not

After the above command is called 500 times, once per `part-00XXX-of-00500` file, there would be 500 TFrecord files totalling to ~365GB.

**Note: It is extremely critical to set the value of `random_seed` to `12345` so that th examples on which the training is evaluated is consistent among users.**

Use the following steps for the eval set:

```shell
python3 create_pretraining_data.py \
  --input_file=<path to ./results>/eval.txt \
  --output_file=<output path for eval_intermediate> \
  --vocab_file=<path to vocab.txt> \
  --do_lower_case=True \
  --max_seq_length=512 \
  --max_predictions_per_seq=76 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=10

python3 pick_eval_samples.py \
  --input_tfrecord=<path to eval_intermediate from the previous command> \
  --output_tfrecord=<output path for eval_10k> \
  --num_examples_to_pick=10000
```

### TFRecord Features

The examples in the TFRecords have the following key/values in its features dictionary:

| Key                  | Type    | Value |
|----------------------|  -----: |-------|
| input_ids            | int64   | Input token ids, padded with 0's to max_sequence_length. |
| input_mask           | int32   | Mask for padded positions. Has 0's on padded positions and 1's elsewhere in TFRecords. |
| segment_ids          | int32   | Segment ids. Has 0's on the positions corresponding to the first segment and 1's on positions corresponding to the second segment. The padded positions correspond to 0. |
| masked_lm_ids        | int32   | Ids of masked tokens, padded with 0's to max_predictions_per_seq to accommodate a variable number of masked tokens per sample. |
| masked_lm_positions  | int32   | Positions of masked tokens in the input_ids tensor, padded with 0's to max_predictions_per_seq. |
| masked_lm_weights    | float32 | Mask for masked_lm_ids and masked_lm_positions. Has values 1.0 on the positions corresponding to actually masked tokens in the given sample and 0.0 elsewhere. |
| next_sentence_labels | int32   | Carries the next sentence labels. |

### Some stats of the generated tfrecords:

| File                |    Size (bytes) |
|---------------------|  -------------: |
| eval_intermediate   |     843,343,183 |
| eval_10k            |      25,382,591 |
| part-00000-of-00500 |     514,241,279 |
| part-00499-of-00500 |     898,392,312 |
| part-00XXX-of-00500 | 391,434,110,129 | 


# Stopping criteria
A valid submission will evaluate a masked lm accuracy >= 0.720. 

The evaluation will be on the 10,000 samples in the evaluation set. The evalution frequency in terms of number of samples trained is determined by the following formular based on the global batch size, starting from 0 examples. The evaluation can be either offline or online for v1.0. More details please refer to the training policy.

```
eval_frequency = floor(0.05 * (230.23 * batch_size + 3,000,000) / 25,000) * 25,000
```

The purpose of this formular is to make the eval interval 1) not too large to make the results within 5% of the actual place in training that cross the target accuracy; and 2) not too small to make evaluation time significant comparing to the end-to-end training time.

### Example evaluation frequency

| Batch size | Eval frequency |
| ---------: | -------------: |
|  256 | 150,000 |
| 1024 | 150,000 |
| 1536 | 150,000 |
| 2048 | 150,000 |
| 3072 | 175,000 |
| 4096 | 175,000 |
| 8192 | 175,000 |

The generation of the evaluation set shard should follow the exact command shown above, using create_pretraining_data.py. **_In particular the seed (12345) must be set to ensure everyone evaluates on the same data._**

# Running the model

To run this model, use the following command.

```shell

TF_XLA_FLAGS='--tf_xla_auto_jit=2' \
python run_pretraining.py \
  --bert_config_file=<path to bert_config.json> \
  --output_dir=/tmp/output/ \
  --input_file="<tfrecord dir>/part*" \
  --nodo_eval \
  --do_train \
  --eval_batch_size=8 \
  --learning_rate=4e-05 \
  --init_checkpoint=./checkpoint/model.ckpt-28252 \
  --iterations_per_loop=1000 \
  --max_predictions_per_seq=76 \
  --max_seq_length=512 \
  --num_train_steps=682666666 \
  --num_warmup_steps=1562 \
  --optimizer=lamb \
  --save_checkpoints_steps=20833 \
  --start_warmup_step=0 \
  --num_gpus=8 \
  --train_batch_size=24

```

The above parameters are for a machine with 8 V100 GPUs with 16GB memory each; the hyper parameters (learning rate, warm up steps, etc.) are for testing only. The training script won’t print out the masked_lm_accuracy; in order to get masked_lm_accuracy, a separately invocation of run_pretraining.py with the following command with a V100 GPU with 16 GB memory:

```shell

TF_XLA_FLAGS='--tf_xla_auto_jit=2' \
python3 run_pretraining.py \
  --bert_config_file=<path to bert_config.json> \
  --output_dir=/tmp/output/ \
  --input_file="<tfrecord dir>/eval_10k" \
  --do_eval \
  --nodo_train \
  --eval_batch_size=8 \
  --init_checkpoint=./checkpoint/model.ckpt-28252 \
  --iterations_per_loop=1000 \
  --learning_rate=4e-05 \
  --max_eval_steps=1250 \
  --max_predictions_per_seq=76 \
  --max_seq_length=512 \
  --num_gpus=1 \
  --num_train_steps=682666666 \
  --num_warmup_steps=1562 \
  --optimizer=lamb \
  --save_checkpoints_steps=20833 \
  --start_warmup_step=0 \
  --train_batch_size=24 \
  --nouse_tpu
   
```

The model has been tested using the following stack:
- Debian GNU/Linux 10 GNU/Linux 4.19.0-12-amd64 x86_64
- NVIDIA Driver 450.51.06
- NVIDIA Docker 2.5.0-1 + Docker 19.03.13
- docker image tensorflow/tensorflow:2.4.0-gpu

## Gradient Accumulation

The GradientAggregationOptimizer can accumulate gradients across multiple steps, on each accelerators, before actually applying the gradients. To use this feature, please note the following:

- Because an additional set of non-trainable weights are used to store the accumulated gradients, the memory footprint of the model doubles. It is highly recommended to use accelerators with larger memory to overcome the memory limitation, such as A100 GPUs.

- The initial checkpoint needs to be converted using checkpoint_add_gradacc.py. This script adds the extra set of weights to the checkpoint to store accumulated gradients. The converted checkpoint size is roughtly doubled.

```shell

python3 checkpoint_add_gradacc.py --old=<path to the oritinal initial checkpoint> --new=<path to the converted checkpoint>

```

- Adjust the hyper-parameters, assuming the batch size is bs, and gradients are accumulated across n steps:
    - `--train_batch_size=bs`.
    - `--steps_per_update=n`.
    - use the learning rate for batch size bs * n, because that's the effective batch size to the optimizer.
    - use `num_train_steps`, `num_warmup_steps`, `save_checkpoints_steps` and `start_warmup_steps` for batch size bs * n, but scale them up n times.
    - note that the step numbers reported by the training script is based on batch size bs.

- Although Gradient Accumulation is a good technique to simulate training with large batch sizes on small hardware systems, there are places that can introduce slightly different behaviors, thus may bring small variances to the achieved accuracies:
    - when intended to simulate n accelerators each has a sub-batch of size bs, on a single accelerator, the moving mean and variance compuation of LayerNorm layers is performed in serial order, instead of independently on each acclerator;
    - there is a clip_by_globalnorm op just before calling the optimizer; the clipping maybe different for different per-accelerator batch size;
    - the accumulation order of gradients is serial under gradient accumulation, which may be different from the accumulation order of cross-device gradient sumations (i.e. allReduce, or cross-replica-sum).

