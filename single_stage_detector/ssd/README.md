# Training


## Dataset Download

Install fiftyone. It's recommended to install the package on your host machine, and not inside docker

```bash
pip install fiftyone
```

Download the MLPerf subset:
```bash
./download_openimages_mlperf.sh -d <DATASET_PATH>
```


## Train model

```bash
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py \
    --dataset openimages \
    --batch-size 32 \
    --lr 0.0001 \
    --lr-steps 16 22 \
    --output-dir=/results
```
