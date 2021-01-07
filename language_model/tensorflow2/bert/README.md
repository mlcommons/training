# Location of the input files 

This [GCS location](https://console.cloud.google.com/storage/browser/pkanwar-bert) contains the following.
* TensorFlow1 checkpoint (bert_model.ckpt) containing the pre-trained weights (which is actually 3 files).
* Vocab file (vocab.txt) to map WordPiece to word id.
* Config file (bert_config.json) which specifies the hyperparameters of the model.
* TensorFlow2 checkpoint should be convertable from the Tensorflow1 checkpoint using the checkpoint_converter:

```shell
python3 tf2_encoder_checkpoint_converter.py \
  --bert_config_file=<path to bert_config.json> \
  --checkpoint_to_convert=<path to tf1 model.ckpt-28252> \
  --converted_checkpoint_path=<path to output tf2 model checkpoint>
```

Alternatively, the tf2 checkpint is also available [here](https://pantheon.corp.google.com/storage/browser/nnigania_perf_profiles/bert_mlperf_data). Note that the checkpoint converter removes optimizer slot variables, so the resulted tf2 checkpoint is only about 1/3 size as the tf1 checkpoint.

# Download and preprocess datasets

Download the [wikipedia dump](https://dumps.wikimedia.org/enwiki/20200101/enwiki-20200101-pages-articles-multistream.xml.bz2) and extract the pages
The wikipedia dump can be downloaded from this link in this directory, and should contain the following file:
enwiki-20200101-pages-articles-multistream.xml.bz2

Run [WikiExtractor.py](https://github.com/attardi/wikiextractor) to extract the wiki pages from the XML
The generated wiki pages file will be stored as <data dir>/LL/wiki_nn; for example <data dir>/AA/wiki_00. Each file is ~1MB, and each sub directory has 100 files from wiki_00 to wiki_99, except the last sub directory. For the 20200101 dump, the last file is FE/wiki_17.

Clean up
The clean up scripts (some references here) are in the scripts directory.
The following command will run the clean up steps, and put the results in ./results
./process_wiki.sh '<data dir>/*/wiki_??'

After running the process_wiki.sh script, for the 20200101 wiki dump, there will be 500 files, named part-00xxx-of-00500 in the ./results directory.

Exact steps (starting in the bert path)  

```shell
cd cleanup_scripts  
mkdir -p wiki  
cd wiki  
wget https://dumps.wikimedia.org/enwiki/20200101/enwiki-20200101-pages-articles-multistream.xml.bz2    # Optionally use curl instead  
bzip2 -d enwiki-20200101-pages-articles-multistream.xml.bz2  
cd ..    # back to bert/cleanup_scripts  
git clone https://github.com/attardi/wikiextractor.git  
python3 wikiextractor/WikiExtractor.py wiki/enwiki-20200101-pages-articles-multistream.xml    # Results are placed in bert/cleanup_scripts/text  
./process_wiki.sh '<text/*/wiki_??'  
python3 extract_test_set_articles.py  
```
  
MD5sums:
7f59165e21b7d566db610ff6756c926b - bert_config.json  
00d47075e0f583fb7c0791fac1c57cb3 - enwiki-20200101-pages-articles-multistream.xml.bz2   
50797acd537880bfb5a7ade80d976129  model.ckpt-28252.data-00000-of-00001
f97de3ae180eb8d479555c939d50d048  model.ckpt-28252.index
dbd16c731e8a8113bc08eeed0326b8e7  model.ckpt-28252.meta
64800d5d8528ce344256daf115d4965e - vocab.txt  

# Generate the BERT input dataset

The create_pretraining_data.py script duplicates the input plain text, replaces different sets of words with masks for each duplication, and serializes the output into the TFRecord file format. 

```shell
python3 create_pretraining_data.py \
   --input_file=<path to ./results of previous step>/part-XX-of-00500 \
   --output_file=<tfrecord dir>/part-XX-of-00500 \
   --vocab_file=<path to vocab.txt> \
   --do_lower_case=True \
   --max_seq_length=512 \
   --max_predictions_per_seq=76 \
   --masked_lm_prob=0.15 \
   --random_seed=12345 \
   --dupe_factor=10
```

The generated tfrecord has 500 parts, totalling to ~365GB.
The dataset was generated using Python 3.7.6 and tensorflow-gpu 1.15.2.

# Stopping criteria
The training should occur over a minimum of 3,000,000 samples. A valid submission will evaluate a masked lm accuracy >= 0.712. 

The evaluation will be on the first 10,000 consecutive samples of the training set. The evalution frequency is every 500,000 samples, starting from 3,000,000 samples. The evaluation can be either offline or online for v0.7. More details please refer to the training policy.

The generation of the evaluation set shard should follow the exact command shown above, using create_pretraining_data.py. In particular the seed (12345) must be set to ensure everyone evaluates on the same data.

# Running the model

To run this model, use the following command.

```shell

TF_GPU_THREAD_MODE=gpu_private python3 run_pretraining.py \
  --all_reduce_alg=nccl \
  --bert_config_file=<path to bert_config.json> \
  --beta_1=0.91063 \
  --beta_2=0.96497 \
  --device_warmup \
  --do_eval \
  --dtype=fp32 \
  --eval_batch_size=48 \
  --init_checkpoint=<path to model.ckpt-28252> \
  '--train_files=<tf_record dir>/part-*' \
  '--eval_files=<tf_record dir>/part-*' \
  --learning_rate=0.00035221 \
  --loss_scale=dynamic \
  --max_predictions_per_seq=76 \
  --max_seq_length=512 \
  --model_dir=<output model dir> \
  --num_accumulation_steps=24 \
  --num_gpus=8 \
  --num_steps_per_epoch=8000 \
  --num_train_epochs=1 \
  --optimizer_type=lamb \
  --scale_loss \
  --steps_before_eval_start=3948 \
  --steps_between_eval=658 \
  --steps_per_loop=658 \
  --stop_steps=8000 \
  --train_batch_size=768 \
  --verbosity=0 \
  --warmup_steps=420

```

The above parameters are for a machine with 8 V100 GPUs with 16GB memory each; the hyper parameters (learning rate, warm up steps, etc.) are from Google's [TF2 BERT submission](https://github.com/mlperf/training_results_v0.7/tree/master/Google/benchmarks/bert/gpu-v100-8-TF2.0) to v0.7, and should converge after about 3.5 to 4.0 million samples.

The model has been tested using the following stack:
- Debian GNU/Linux 10 GNU/Linux 4.19.0-12-amd64 x86_64
- NVIDIA Driver 450.51.06
- NVIDIA Docker 2.5.0-1 + Docker 19.03.13
- docker image tensorflow/tensorflow:2.4.0-gpu

