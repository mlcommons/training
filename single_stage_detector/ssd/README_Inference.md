# Inference

## Training

The model was trained on coco2017 dataset using 8xV100 GPUs and FP32 with the following command:

```bash
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py \
    --backbone resnext50_32x4d \
    --batch-size 2 \
    --epochs 26 \
    --lr-steps 16 22 \
    --lr 0.02 \
    --output-dir /results"
```

The training checkpoints are available [here](https://drive.google.com/drive/u/0/folders/1c26Vwew5hrEpkPn1nnCOCuDhUgeJ6y1s),
and the training log is available [here](https://drive.google.com/file/d/1-Swfi6S3HlMGKMazvaLy26wRrWUfoNzW/view?usp=sharing).


## Validation and Accuracy

Use the following command to do a validation run on one of the saved checkpoints:

```bash
python train.py --test-only --resume <checkpoint_file.pth>
```

The model reached a maximum accuracy of 36.781% on the 19th epoch. The pretrained weights can be obtained from
[here](https://drive.google.com/file/d/1hczwQzUg9QIMUwaBgeahNtrwiBmYWl7-/view?usp=sharing),
and the expected validation output is available
[here](https://drive.google.com/file/d/1hlFdiJpSh5MWd48toobGtRW-Ws6pxEcL/view?usp=sharing)


## Convert to onnx

The repo includes a script to convert the pretrained pytorch weights (pth format) to an [onnx](https://onnx.ai/) file.

```bash
python pth_to_onnx.py --input <checkpoint_file.pth> --image-size 800 800
```

Unlike the pth file which includes only the pretrained weights, the onnx file is self contained and includes the full
computational graph. The converted onnx file for the 19th epoch is
available [here](https://drive.google.com/file/d/1NXaSIrte_DPyl-_Bizahai-z6ISmRWso/view?usp=sharing)

