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

## Download and uncompress

The files at the time of v0.7 were available at https://dumps.wikimedia.org/enwiki/20200101/enwiki-20200101-pages-articles-multistream.xml.bz2

The files have since been removed from this link. Instead, they are now available at this [Google Drive location](https://drive.google.com/drive/folders/1oQF4diVHNPCclykwdvQJw8n_VIWwV0PT). The file `enwiki-20200101-pages-articles-multistream.xml.bz2` containing wikipedia dump should be downloaded and uncompressed.

```shell
# TODO: Change this to correct commit when patch is merged
# Clone the training repo

git clone https://github.com/sgpyc/training
git checkout bert_fix

# TODO: Add HEAD Commit SHA

cd language_model/tensorflow/bert/cleanup_scripts

# Download and uncompress files from the Google Drive location
source download_and_umcompress.sh

```
After downloading and uncompressing files, confirm if the md5sums match the expected values.

### MD5sums of provided files:

| File                                               |   Size (bytes) | MD5                              |
|----------------------------------------------------|  ------------: |----------------------------------|
| bert_config.json                                   |            314 | 7f59165e21b7d566db610ff6756c926b |
| vocab.txt                                          |        231,508 | 64800d5d8528ce344256daf115d4965e |
| model.ckpt-28252.index (tf1)                       |         17,371 | f97de3ae180eb8d479555c939d50d048 |
| model.ckpt-28252.meta (tf1)                        |     24,740,228 | dbd16c731e8a8113bc08eeed0326b8e7 |
| model.ckpt-28252.data-00000-of-00001 (tf1)         |  4,034,713,312 | 50797acd537880bfb5a7ade80d976129 |
| model.ckpt-28252.index (tf2)                       |          6,420 | fc34dd7a54afc07f2d8e9d64471dc672 |
| model.ckpt-28252.data-00000-of-00001 (tf2)         |  1,344,982,997 | 77d642b721cf590c740c762c7f476e04 | 
| enwiki-20200101-pages-articles-multistream.xml.bz2 | 17,751,214,669 | 00d47075e0f583fb7c0791fac1c57cb3 |
| enwiki-20200101-pages-articles-multistream.xml     | 75,163,254,305 | 1021bd606cba24ffc4b93239f5a09c02 |

## Extract
Run [WikiExtractor.py](https://github.com/attardi/wikiextractor) to extract the wiki pages from the XML
The generated wiki pages file will be stored as <data dir>/LL/wiki_nn; for example <data dir>/AA/wiki_00. Each file is ~1MB, and each sub directory has 100 files from wiki_00 to wiki_99, except the last sub directory. For the 20200101 dump, the last file is FE/wiki_17.

Next, clone the [WikiExtractor](https://github.com/attardi/wikiextractor/tree/3162bb6c3c9ebd2d15be507aa11d6fa818a454ac) repo, and extract data from XML.
```shell
git clone https://github.com/attardi/wikiextractor.git

cd wikiextractor

git checkout 3162bb6c3c9ebd2d15be507aa11d6fa818a454ac

# Back to <bert>/cleanup_scripts
cd .. 

# Run `WikiExtractor.py` to extract data from XML.
python wikiextractor/WikiExtractor.py wiki/enwiki-20200101-pages-articles-multistream.xml
```

The generated wiki pages file will be stored in `<bert>/cleanup_scripts/text/<XX>/wiki_<nn>` where `<XX>` are folders from `AA` to `FE` and `<nn>` ranges from `00` to `99`.

For example :`<bert>/cleanup_scripts/text/BD/wiki_37`.

 Each file is ~1MB, and each sub directory has 100 files from `wiki_00` to `wiki_99`, except the last sub directory `FE`. For the 20200101 dump, the last file is `FE/wiki_17`.


### Files in <bert>/cleanup_scripts/text/FE/:

| File    | Size (bytes) | MD5                              |
|---------|  ----------: |----------------------------------|
| wiki_00 | 1,048,175    | d8ad2f6311e3692e9b5ec9d38bfe8707 |
| wiki_01 | 1,047,515    | f098c976543d39e9aa99f91d278686f8 |
| wiki_02 | 1,047,954    | fab7f42b8df1e3d8dd6db7d672e05cc3 |
| wiki_03 | 1,048,205    | c27cf920d8954f6b76576363d14945ba |
| wiki_04 | 1,047,729    | 0d5ccc12742c2123330b2205ab7bae99 |
| wiki_05 | 1,045,417    | 991f06e6fe50c99e6b50e6f778dc9181 |
| wiki_06 | 1,048,289    | d160d3edcd847b896b988c261d7b3951 |
| wiki_07 | 1,045,378    | 5e8a262f80575aad0f1b3f337fd0a2f9 |
| wiki_08 | 1,047,758    | bbeadd3b9045eb1468d5f546b5013b41 |
| wiki_09 | 1,048,314    | d9d6bf4d61259d7a7760f52da8ca03be |
| wiki_10 | 1,048,422    | a139da62c0cf443401162093a3c8018a |
| wiki_11 | 1,048,255    | 100bd5153de234e4769a6e9baf103d43 |
| wiki_12 | 1,048,548    | 3bda2c6eeea74ef37314e5e3f9d8dbff |
| wiki_13 | 1,046,253    | 9b8084d36640b536458345f6a6400d70 |
| wiki_14 | 1,036,170    | 7d5ca15dab637fc3d36124fd404e037a |
| wiki_15 | 1,048,378    | 9b6dea989a5ca2d46e6f0a0eb730197c |
| wiki_16 | 1,046,493    | ee7870f5dbd4de278825e9d32ee1fa78 |
| wiki_17 |   398,182    | fce4a6b8886e2796409a8588f3e88b75 |

**Note**: WikiExtractor.py replaces some of the tags present in XML such as {CURRENTDAY}, {CURRENTMONTHNAMEGEN} with the current values obtained from time.strftime ([code](https://github.com/attardi/wikiextractor/blob/e4abb4cbd019b0257824ee47c23dd163919b731b/WikiExtractor.py#L632)). Hence, one might see slighly different preprocessed files after the WikiExtractor.py file is invoked. This means the md5sum hashes of these files will also be different each time WikiExtractor is called.

## Clean up and dataset seperation

The scripts are located in [cleanup_scripts](./cleanup_scripts). Specifically, files [clean.sh](./cleanup_scripts/clean.sh), [cleanup_file.py](./cleanup_scripts/cleanup_file.py), [do_gather.py](./cleanup_scripts/do_gather.py), [seperate_test_set.py](./seperate_test_set.py) and [do_sentence_segmentation.py](./cleanup_scripts/do_sentence_segmentation.py) are used for further preprocessing. A wrapper shell script [process_wiki.sh](./cleanup_scripts/process_wiki.sh) calls these cleanup scripts and does end-to-end preprocessing of files

```shell
./process_wiki.sh './text/*/wiki_??'
```

After running the process_wiki.sh script, 

* For every file `<bert>/cleanup_scripts/text/<XX>/wiki_<nn>`, there are four additional files generated as below
  - `<bert>/cleanup_scripts/text/<XX>/wiki_<nn>.1`
  - `<bert>/cleanup_scripts/text/<XX>/wiki_<nn>.2`
  - `<bert>/cleanup_scripts/text/<XX>/wiki_<nn>.3`
  - `<bert>/cleanup_scripts/text/<XX>/wiki_<nn>.4`

* For the 20200101 wiki dump, there will be 502 files, named part-(00000 to 00499)-of-00500, eval.md5 and eval.txt in the `<bert>/results` directory.

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
  --input_tfrecord=<path to eval_intermedia from the previous command> \
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

