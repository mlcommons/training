# V0.7 Dataset and Training

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

```
python3 tf2_encoder_checkpoint_converter.py \
  --bert_config_file=<path to bert_config.json> \
  --checkpoint_to_convert=<path to tf1 model.ckpt-28252> \
  --converted_checkpoint_path=<path to output tf2 model checkpoint>

```
Note that the checkpoint converter removes optimizer slot variables, so the resulting TF2 checkpoint is only about 1/3 size of the TF1 checkpoint.


# Download and preprocess datasets

### **Download**

The files at the time of v0.7 were available at https://dumps.wikimedia.org/enwiki/20200101/enwiki-20200101-pages-articles-multistream.xml.bz2

The files have since been removed from this link. Instead, they are now available at this [Google Drive location](https://drive.google.com/drive/folders/1oQF4diVHNPCclykwdvQJw8n_VIWwV0PT). The file `enwiki-20200101-pages-articles-multistream.xml.bz2` containing wikipedia dump should be downloaded and extracted.

```
# TODO: Change this to correct commit when patch is merged
# Clone the training repo

git clone https://github.com/sgpyc/training
git checkout bert_fix

# TODO: Add HEAD Commit SHA

cd language_model/tensorflow/bert/cleanup_scripts

# Download and extract files from the Google Drive location
source download_and_extract.sh

```
After downloading and extracting files, confirm if the md5sums match the expected values. 

#### _**Expected md5sum hashes**_


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
||||

### **Extract data from XML**

Next, clone the [WikiExtractor](https://github.com/attardi/wikiextractor/tree/e4abb4cbd019b0257824ee47c23dd163919b731b) repo
```
git clone https://github.com/attardi/wikiextractor.git

cd wikiextractor

git checkout e4abb4cbd019b0257824ee47c23dd163919b731b

# Back to <bert>/cleanup_scripts
cd .. 
```

Run `WikiExtractor.py` to extract data from XML.

```
python wikiextractor/WikiExtractor.py wiki/enwiki-20200101-pages-articles-multistream.xml
```


The generated wiki pages file will be stored in `<bert>/cleanup_scripts/text/<XX>/wiki_<nn>` where `<XX>` are folders from `AA` to `FE` and `<nn>` ranges from `00` to `99`.

For example :`<bert>/cleanup_scripts/text/BD/wiki_37`.

 Each file is ~1MB, and each sub directory has 100 files from `wiki_00` to `wiki_99`, except the last sub directory `FE`. For the 20200101 dump, the last file is `FE/wiki_17`.

```
# Contents of <bert>/cleanup_scripts/text/FE/

-rw-r--r--. 1    1048175 Feb  9 17:41 wiki_00
-rw-r--r--. 1    1047515 Feb  9 17:41 wiki_01
-rw-r--r--. 1    1047954 Feb  9 17:41 wiki_02
-rw-r--r--. 1    1048205 Feb  9 17:41 wiki_03
-rw-r--r--. 1    1047729 Feb  9 17:41 wiki_04
-rw-r--r--. 1    1045417 Feb  9 17:41 wiki_05
-rw-r--r--. 1    1048289 Feb  9 17:41 wiki_06
-rw-r--r--. 1    1045378 Feb  9 17:41 wiki_07
-rw-r--r--. 1    1047758 Feb  9 17:41 wiki_08
-rw-r--r--. 1    1048314 Feb  9 17:41 wiki_09
-rw-r--r--. 1    1048422 Feb  9 17:41 wiki_10
-rw-r--r--. 1    1048255 Feb  9 17:41 wiki_11
-rw-r--r--. 1    1048548 Feb  9 17:41 wiki_12
-rw-r--r--. 1    1046253 Feb  9 17:41 wiki_13
-rw-r--r--. 1    1036170 Feb  9 17:41 wiki_14
-rw-r--r--. 1    1048378 Feb  9 17:41 wiki_15
-rw-r--r--. 1    1046493 Feb  9 17:41 wiki_16
-rw-r--r--. 1     398182 Feb  9 17:41 wiki_17
```

**Note**: WikiExtractor.py replaces some of the tags present in XML such as {CURRENTDAY}, {CURRENTMONTHNAMEGEN} with the current values obtained from time.strftime ([code](https://github.com/attardi/wikiextractor/blob/e4abb4cbd019b0257824ee47c23dd163919b731b/WikiExtractor.py#L632)). Hence, one might see slighly different preprocessed files after the WikiExtractor.py file is invoked. This means the md5sum hashes of these files will also be different each time WikiExtractor is called.

### **Cleanup**

The scripts are located in [cleanup_scripts](./cleanup_scripts). Specifically, files [clean.sh](./cleanup_scripts/clean.sh), [cleanup_file.py](./cleanup_scripts/cleanup_file.py), [do_gather.py](./cleanup_scripts/do_gather.py) and [do_sentence_segmentation.py](./cleanup_scripts/do_sentence_segmentation.py) are used for further preprocessing. A wrapper shell script [process_wiki.sh](./cleanup_scripts/process_wiki.sh) calls these cleanup scripts and does end-to-end preprocessing of files

```
./process_wiki.sh './text/*/wiki_??'
```

After running the process_wiki.sh script, 

* For every file `<bert>/cleanup_scripts/text/<XX>/wiki_<nn>`, there are three additional files generated as below
  - `<bert>/cleanup_scripts/text/<XX>/wiki_<nn>.1`
  - `<bert>/cleanup_scripts/text/<XX>/wiki_<nn>.2`
  - `<bert>/cleanup_scripts/text/<XX>/wiki_<nn>.3`

* For the 20200101 wiki dump, there will be 500 files, named part-00xxx-of-00500 in the `<bert>/results` directory.

```
# File sizes of the last few files in <bert>/results
.
.
.
-rw-r--r--. 1    24307483 Feb 10 11:04 part-00496-of-00500
-rw-r--r--. 1    24790598 Feb 10 11:04 part-00497-of-00500
-rw-r--r--. 1    24593596 Feb 10 11:04 part-00498-of-00500
-rw-r--r--. 1    21886959 Feb 10 11:04 part-00499-of-00500
```

`part-00xxx-of-00500` contains one sentence of an article in one line and different articles separated by blank line. 

### **Generate the TFRecords for Wiki dataset**

The [create_pretraining_data.py](./create_pretraining_data.py) script tokenizes the words in a sentence using [tokenization.py](./tokenization.py) and `vocab.txt` file. Then, random tokens are masked using the strategy where 80% of time, the selected random tokens are replaced by `[MASK]` tokens, 10% by a random word and the remaining 10% left as is. This process is repeated for `dupe_factor` number of times, where an example with `dupe_factor` number of different masks are generated and written to TFRecords.

```
#### Generate one TFRecord for each part-00XXX-of-00500 file. The following command is for generating one corresponding TFRecord

python3 create_pretraining_data.py \
   --input_file=<path to ./results of previous step>/part-00XXX-of-00500 \
   --output_file=<path to tfrecord dir>/part-00XXX-of-00500.tfrecord \
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

### **TFRecord Features**

The examples in the TFRecords have the following key/values in itsfeatures dictionary:

`input_ids`:
Input token ids, padded with 0's to max_sequence_length. <br/>
Type: int64

`input_mask`:
Mask for padded positions. Has 0's on padded positions and 1's elsewhere in TFRecords. <br/>
Type: int32

`segment_ids`:
Segment ids. Has 0's on the positions corresponding to the first segment and 1's on positions corresponding to the second segment. The padded positions correspond to 0. <br/>
Type: int32

`masked_lm_ids`:
Ids of masked tokens, padded with 0's to max_predictions_per_seq to accommodate a variable number of masked tokens per sample. <br/>
Type: int32

`masked_lm_positions`:
Positions of masked tokens in the input_ids tensor, padded with 0's to max_predictions_per_seq. <br/>
Type: int32

`masked_lm_weights`:
Mask for masked_lm_ids and masked_lm_positions. Has values 1.0 on the positions corresponding to actually masked tokens in the given sample and 0.0 elsewhere. <br/>
Type: float32

`next_sentence_labels`: Carries the next sentence labels. <br/>
Type: int32

The dataset was generated using Python 3.7.6 and tensorflow-gpu 1.15.2.

# Stopping criteria
The training should occur over a minimum of 3,000,000 samples. A valid submission will evaluate a masked lm accuracy >= 0.712. 

The evaluation will be on the first 10,000 consecutive samples of the training set. The evalution frequency is every 500,000 samples, starting from 3,000,000 samples. The evaluation can be either offline or online for v0.7. More details please refer to the training policy.

The generation of the evaluation set shard should follow the exact command shown above, using create_pretraining_data.py. **_In particular the seed (12345) must be set to ensure everyone evaluates on the same data._**

# Running the model

To run this model, use the following command.

```shell

TF_XLA_FLAGS='--tf_xla_auto_jit=2' \
python run_pretraining.py \
  --bert_config_file=<path to bert_config.json> \
  --output_dir=/tmp/output/ \
  --input_file="<path to tfrecord dir>/part*.tfrecord*" \
  --nodo_eval \
  --do_train \
  --eval_batch_size=8 \
  --learning_rate=4e-05 \
  --init_checkpoint=./checkpoint/model.ckpt-28252 \
  --iterations_per_loop=1000 \
  --max_predictions_per_seq=76 \
  --max_seq_length=512 \
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
  --input_file="<path to tfrecord dir>/part-00000-of-00500.tfrecord*" \
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
- Debian GNU/Linux 9.12 GNU/Linux 4.9.0-12-amd64 x86_64
- NVIDIA Driver 440.64.00
- NVIDIA Docker 2.2.2-1 + Docker 19.03.8
- docker image tensorflow/tensorflow:2.2.0rc0-gpu-py3

