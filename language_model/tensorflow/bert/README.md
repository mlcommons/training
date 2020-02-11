# Location of the input files 

This [GCS location](https://console.cloud.google.com/storage/browser/pkanwar-bert) contains the following.
* TensorFlow checkpoint (bert_model.ckpt) containing the pre-trained weights (which is actually 3 files).
* Vocab file (vocab.txt) to map WordPiece to word id.
* Config file (bert_config.json) which specifies the hyperparameters of the model.

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

# Running the model

To run this model, use the following command.

```shell

python run_pretraining.py \
  --bert_config_file=./bert_config.json \
  --output_dir=/tmp/output/ \
  --input_file="./uncased_seq_512/wikipedia.tfrecord*" \
  --nodo_eval \
  --do_train \
  --eval_batch_size=8 \
  --learning_rate=4e-05 \
  --init_checkpoint=./checkpoint/model.ckpt-7037 \
  --iterations_per_loop=1000 \
  --max_predictions_per_seq=76 \
  --max_seq_length=512 \
  --num_train_steps=1365333333 \
  --num_warmup_steps=0 \
  --optimizer=lamb \
  --save_checkpoints_steps=1000 \
  --start_warmup_step=0 \
   --num_gpus=8 \
  --train_batch_size=24/

```

The above parameters are for a machine with 8 V100 GPUs with 16GB memory each; the hyper parameters (learning rate, warm up steps, etc.) are for testing only. The training script won’t print out the masked_lm_accuracy; in order to get masked_lm_accuracy, a separately invocation of run_pretraining.py with the following command with a V100 GPU with 16 GB memory:

```shell

python3 run_pretraining.py \
  --bert_config_file=./bert_config.json \
  --output_dir=/tmp/output/ \
  --input_file="<tfrecord dir>/part-*" \
  --do_eval \
  --nodo_train \
  --eval_batch_size=8 \
  --init_checkpoint=./checkpoint/model.ckpt-7037 \
  --iterations_per_loop=1000 \
  --learning_rate=4e-05 \
  --max_eval_steps=1250 \
  --max_predictions_per_seq=76 \
  --max_seq_length=512 \
  --num_gpus=1 \
  --num_train_steps=1365333333 \
  --num_warmup_steps=3125 \
  --optimizer=lamb \
  --save_checkpoints_steps=1000 \
  --start_warmup_step=0 \
  --train_batch_size=24 \
  --nouse_tpu
   
```

