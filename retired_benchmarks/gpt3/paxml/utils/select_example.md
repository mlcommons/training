# Script to select a subset of examples from dataset

This is a script to select a given number of examples in a 
dataset for the LLM benchmark.

## Requirement

[TensorFlow](https://github.com/tensorflow/tensorflow)
[TensorFlow DataSet](https://github.com/tensorflow/datasets)

## Usage

Assuming the c4/en/3.0.1 dataset has been downloaded to `/data/c4/en/3.0.1`,
the following command will choose 24567 examples from the validation set,
and put the selected examples in `/data/c4_en_3.0.1_validation_24567exp.tfrecord`
with the corresponding hash of the text feature in each example in
`/data/c4_en_3.0.1_validation_24567exp.hash`.

```
python3 ./select_example.py \
--data_dir=/data \
--split=validation \
--num_examples_to_pick=24567 \
--output_filepath=/data/c4_en_3.0.1_validation_24567exp
```

An example output would be

```
Input:
  num_examples =  364608   min_length =  14   avg_length =  2162.6717186677197   max_length =  250537
  num_lines =  3094023   min_length =  9   avg_length =  253.97290033073446   max_length =  98734
Selected:
  num_examples =  24567   min_length =  22   avg_length =  2142.668335572109   max_length =  95736
  num_lines =  209784   min_length =  11   avg_length =  250.03678068870838   max_length =  16736
```

The script has been tested using Python 3.7.3, TensorFlow gpu 2.9.1 and TensorFlow DataSet 4.6.0.
