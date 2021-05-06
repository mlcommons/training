# Location of the input files 

This [MLCommons members Google Drive location](https://drive.google.com/drive/u/0/folders/1oQF4diVHNPCclykwdvQJw8n_VIWwV0PT) contains the following.
* TensorFlow checkpoint (bert_model.ckpt) containing the pre-trained weights (which is actually 3 files).
* Vocab file (vocab.txt) to map WordPiece to word id.
* Config file (bert_config.json) which specifies the hyperparameters of the model.

# Checkpoint conversion
python convert_tf_checkpoint.py --tf_checkpoint /cks/model.ckpt-28252 --bert_config_path /cks/bert_config.json --output_checkpoint model.ckpt-28252.pt

# Download the preprocessed text dataset

From the [MLCommons BERT Processed dataset
directory](https://drive.google.com/drive/folders/1cywmDnAsrP5-2vsr8GDc6QUc7VWe-M3v?usp=sharing)
download `results_text.tar.gz`, and `bert_reference_results_text_md5.txt`.  Then perform the following steps:

* tar xf results_text.tar.gz
* cd results4
* md5sum --check ../bert_reference_results_text_md5.txt
* cd ..

# Generate the BERT input dataset

The create_pretraining_data.py script duplicates the input plain text, replaces
different sets of words with masks for each duplication, and serializes the
output into the HDF5 file format.

## Training data

The following shows how create_pretraining_data.py is called by a parallelized
script that can be called as shown below.  The script reads the text data from
the `results4/` subdirectory and outputs the resulting 500 hdf5 files to a
subdirectory named "hdf5".

```shell
./parallel_create_hdf5.sh
```

Next we need to shard the data into 2048 chunks.  This is done by calling the
chop_hdf5_files.py script.  This script reads the 500 hdf5 files from
subdirectory `hdf5/` and creates 2048 hdf5 files in subdirectory
`2048_shards_uncompressed`.

```shell
mkdir -p 2048_shards_uncompressed
python3 ./chop_hdf5_files.py
```

##  Evaluation data

Use the following steps for the eval set:

```shell
mkdir eval_set_uncompressed

python3 create_pretraining_data.py \
  --input_file=results4/eval.txt \
  --output_file=eval_all \
  --vocab_file=<path to vocab.txt> \
  --do_lower_case=True \
  --max_seq_length=512 \
  --max_predictions_per_seq=76 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=10

python3 pick_eval_samples.py \
  --input_hdf5_file=eval_all.hdf5 \
  --output_hdf5_file=eval_set_uncompressed/part_eval_10k.hdf5 \
  --num_examples_to_pick=10000
```

# Running the model

Building the Docker container
```shell
docker build --pull -t <docker/registry>/mlperf-nvidia:language_model .
docker push <docker/registry>/mlperf-nvidia:language_model
```

To run this model, use the following command. Replace the paths in the config file(config_DGXA100_common.sh) to match paths on your system.

```shell
source <CONFIG>
sbatch --nodes ${DGXNNODES} --ntasks-per-node=${DGXNGPU} --time=${WALLTIME} run.sub
```

To run RCP specific configs:

GBS256=TPU-v4-16
```shell
source config_DGXA100_2x8x16x1.sh;
sbatch --nodes 2 --ntasks-per-node 8 --time 01:00:00 run.sub
```

GBS448=TPU-v4-16
```shell
source config_DGXA100_2x8x28x1.sh;
sbatch --nodes 2 --ntasks-per-node 8 --time 01:00:00 run.sub
```

GBS768=TPU-v4-16
```shell
source config_DGXA100_2x8x48x1.sh;
sbatch --nodes 2 --ntasks-per-node 8 --time 01:00:00 run.sub
```

GBS3072=TPU-v4-128
```shell
source config_DGXA100_16x8x24x1.sh;
sbatch --nodes 16 --ntasks-per-node 8 --time 01:00:00 run.sub
```

GBS8192=TPU-v4-128
```shell
source config_DGXA100_16x8x64x1.sh;
sbatch --nodes 16 --ntasks-per-node 8 --time 01:00:00 run.sub
```

Note: Make sure a current run doesn't use stale parameters sourced from a previous run.

For multi-node training, we use Slurm for scheduling and Pyxis to run our container.

The code comes with an option to either clip and accumulate gradients (`--clip_and_accumulate`), or accumulate gradients and then clip(which is the default mode).
The benchmark implements gradient clipping before allreduce. To be identical to a 16x8x64x1 run with gradient accumulation for example, one can run either:
1) 8x8x64x2 with `--clip_and_accumulate`
2) 16x8x32x2 without `--clip_and_accumulate`

## Configuration File Naming Convention

All configuration files follow the format `config_<SYSTEM_NAME>_<NODES>x<GPUS/NODE>x<BATCH/GPU>x<GRADIENT_ACCUMULATION_STEPS>.sh`.

### Example 1
A DGX A100 system with 1 node, 8 GPUs per node, batch size of 6 per GPU, and 6 gradient accumulation steps would use `config_DGXA100_1x8x6x6.sh`.

### Example 2
A DGX A100 system with 32 nodes, 8 GPUs per node, batch size of 20 per GPU, and no gradient accumulation would use `config_DGXA100_32x8x20x1.sh`



# Description of how the `results_text.tar.gz` file was prepared

1. First download the [wikipedia
   dump](https://drive.google.com/file/d/18K1rrNJ_0lSR9bsLaoP3PkQeSFO-9LE7/view?usp=sharing)
   and extract the pages The wikipedia dump can be downloaded from [this google
   drive](https://drive.google.com/drive/u/0/folders/1oQF4diVHNPCclykwdvQJw8n_VIWwV0PT),
   and should contain `enwiki-20200101-pages-articles-multistream.xml.bz2` as
   well as the md5sum.

2. Run [WikiExtractor.py](https://github.com/attardi/wikiextractor), version
   e4abb4cb from March 29, 2020, to extract the wiki pages from the XML The
   generated wiki pages file will be stored as <data dir>/LL/wiki_nn; for
   example <data dir>/AA/wiki_00. Each file is ~1MB, and each sub directory has
   100 files from wiki_00 to wiki_99, except the last sub directory. For the
   20200101 dump, the last file is FE/wiki_17.

3. Clean up and dataset seperation.  The clean up scripts (some references
   here) are in the scripts directory.  The following command will run the
   clean up steps, and put the resulted trainingg and eval data in ./results
   ./process_wiki.sh 'text/*/wiki_??'

4. After running the process_wiki.sh script, for the 20200101 wiki dump, there will be 500 files, named part-00xxx-of-00500 in the ./results directory, together with eval.md5 and eval.txt.

5. Exact steps (starting in the bert path)  

```shell
cd input_preprocessing
mkdir -p wiki  
cd wiki
# download enwiki-20200101-pages-articles-multistream.xml.bz2 from Google drive and check md5sum
bzip2 -d enwiki-20200101-pages-articles-multistream.xml.bz2  
cd ..    # back to bert/input_preprocessing  
git clone https://github.com/attardi/wikiextractor.git
cd wikiextractor
git checkout e4abb4cbd
python3 wikiextractor/WikiExtractor.py wiki/enwiki-20200101-pages-articles-multistream.xml    # Results are placed in bert/input_preprocessing/text  
./process_wiki.sh './text/*/wiki_??'  
```

After running the process_wiki.sh script, for the 20200101 wiki dump, there will be 500 files, named part-00xxx-of-00500 in the ./results directory.

 
MD5sums:

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

