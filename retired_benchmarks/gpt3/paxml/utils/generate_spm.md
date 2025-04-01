This page describes the steps used to generate the SentencePiece tokernizer
model.

# Select examples

5,000,000 examples from the train split of the C4/en/3.0.1 dataset,
using select_text.py with the following command. The tfrecord files are
placed in `/data/c4/en/3.0.1` to get the TensorFLow Dataset loader working.

```
python3 ./select_text.py \
--data_dir="/data" \
--num_examples=5000000 \
--output_text_file=./c4_en_301_5Mexp2.txt
```

The resulted file has the following statistics:

| Stats | Value |
| - | - |
| Num. examples | 5,000,000 |
| Max. length of example | 224,443 bytes |
| Avg. length of example | 2166 bytes |
| Num. lines (mostly sentences) | 42,627,550 |
| Max. length of line | 110,478 bytes |
| Avg. length of line | 253 bytes |

# Train the SentenPiece model

A [SentencePiece](https://github.com/google/sentencepiece) package
was built and installed from its source. Then a SentencePiece model
was trained using the following command:

```
spm_train \
--pad_id=0 \
--eos_id=1 \
--bos_id=-1 \
--unk_id=2 \
--model_type=BPE \
--vocab_size=50257 \
--num_threads=32 \
--split_digits=true \
--train_extremely_large_corpus=true \
--max_sentence_length=10000 \
--input=c4_en_301_5Mexp2.txt \
--model_prefix=/data/c4_en_301_5Mexp2_spm \
--byte_fallback=true \
--character_coverage=0.9995
```

# Package versions

The package versions when the SPM was generated were:

| Package | Version |
| - | - |
| TensorFlow | 2.9.1 |
| TensorFlow-Datasets | 4.6.0 |
| SentencePiece | 91809e5c |

