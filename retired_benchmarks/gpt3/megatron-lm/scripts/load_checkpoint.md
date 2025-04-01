# Load checkpoint

This is an example script to load the checkpoint using PyTorch for LLM benchmark.

## Requirement

Megatron
PyTorch

## Usage

Assuming that the checkpoint has been downloaded to `/data`, the following command 
will load the state_dict for all model parallel units.

```
python3 scripts/load_checkpoint.py \
    --input_path /data/iter_0000300 \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 8
```

Each pickle file is ~37GB and the data is loaded into a list of state_dicts for each model parallel unit.

The script has been tested using Python 3.8.12 and PyTorch 1.11.0
