# Example checkpoint loading script

This is an example script to load layer(s) of the saved checkpoint into NumPy
array(s) for the LLM benchmark.

## Requirement

[TensorStore](https://google.github.io/tensorstore/)

## Usage

Assuming the checkpoint has been downloaded to `/data`, the following command 
will load the weights of `self_attention.combined_qkv.w` for all 96 GPT-3 blocks.

```
python3 ./load_ts_ckpt.py \
--input_path /data/gpt3_spmd1x192x8_tpuv4-3072_20220511/checkpoints/checkpoint_00000200/mdl_vars.params.lm.transformer.repeat.sub.x_layers_0.self_attention.combined_qkv.w/
```

Please use a machine with sufficient memory; the largest single directory is 
more than 200GB in size. It may take a few minutes to load a directory,
depending on the machine specs and the size of the layer(s). Below is an
output from running the command above:

```
path =  /data/gpt3_spmd1x192x8_tpuv4-3072_20220511/checkpoints/checkpoint_00000200/mdl_vars.params.lm.transformer.repeat.sub.x_layers_0.self_attention.combined_qkv.w/ , type =  <class 'numpy.ndarray'> , shape =  (96, 3, 12288, 96, 128)
```

The script has been tested using Python 3.7.8 and TensorStore 0.1.20.
