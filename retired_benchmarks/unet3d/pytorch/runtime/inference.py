import numpy as np
from scipy import signal
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

from runtime.distributed_utils import reduce_tensor, get_world_size, get_rank


def evaluate(flags, model, loader, loss_fn, score_fn, device, epoch=0, is_distributed=False):
    rank = get_rank()
    world_size = get_world_size()
    model.to(device)
    if flags.load_ckpt_path:
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        checkpoint = torch.load(flags.load_ckpt_path, map_location=map_location)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['best_model_state_dict'])
        if is_distributed:
            model = torch.nn.parallel.DistributedDataParallel(model,
                                                              device_ids=[flags.local_rank],
                                                              output_device=flags.local_rank)

    model.eval()

    eval_loss = []
    scores = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, disable=(rank != 0) or not flags.verbose)):
            image, label = batch
            image, label = image.to(device), label.to(device)
            if image.numel() == 0:
                continue
            with autocast(enabled=flags.amp):
                output, label = sliding_window_inference(
                    inputs=image,
                    labels=label,
                    roi_shape=flags.val_input_shape,
                    model=model,
                    overlap=flags.overlap,
                    mode="gaussian",
                    padding_val=-2.2
                )
                eval_loss_value = loss_fn(output, label)
                scores.append(score_fn(output, label))
            eval_loss.append(eval_loss_value)
            del output
            del label

    scores = reduce_tensor(torch.mean(torch.stack(scores, dim=0), dim=0), world_size)
    eval_loss = reduce_tensor(torch.mean(torch.stack(eval_loss, dim=0), dim=0), world_size)
    # scores = torch.mean(torch.stack(scores, dim=0), dim=0)
    # eval_loss = torch.mean(torch.stack(eval_loss, dim=0), dim=0)

    scores, eval_loss = scores.cpu().numpy(), float(eval_loss.cpu().numpy())
    eval_metrics = {"epoch": epoch,
                    "L1 dice": scores[-2],
                    "L2 dice": scores[-1],
                    "mean_dice": (scores[-1] + scores[-2]) / 2,
                    "eval_loss": eval_loss}

    return eval_metrics


def pad_input(volume, roi_shape, strides, padding_mode, padding_val, dim=3):
    """
    mode: constant, reflect, replicate, circular
    """
    bounds = [(strides[i] - volume.shape[2:][i] % strides[i]) % strides[i] for i in range(dim)]
    bounds = [bounds[i] if (volume.shape[2:][i] + bounds[i]) >= roi_shape[i] else bounds[i] + strides[i]
              for i in range(dim)]
    paddings = [bounds[2] // 2, bounds[2] - bounds[2] // 2,
                bounds[1] // 2, bounds[1] - bounds[1] // 2,
                bounds[0] // 2, bounds[0] - bounds[0] // 2,
                0, 0,
                0, 0]

    return F.pad(volume, paddings, mode=padding_mode, value=padding_val), paddings


def gaussian_kernel(n, std):
    gaussian1D = signal.gaussian(n, std)
    gaussian2D = np.outer(gaussian1D, gaussian1D)
    gaussian3D = np.outer(gaussian2D, gaussian1D)
    gaussian3D = gaussian3D.reshape(n, n, n)
    gaussian3D = np.cbrt(gaussian3D)
    gaussian3D /= gaussian3D.max()
    return torch.from_numpy(gaussian3D)


def sliding_window_inference(inputs, labels, roi_shape, model, overlap=0.5, mode="gaussian",
                             padding_mode="constant", padding_val=0.0, **kwargs):
    image_shape = list(inputs.shape[2:])
    dim = len(image_shape)
    strides = [int(roi_shape[i] * (1 - overlap)) for i in range(dim)]

    bounds = [image_shape[i] % strides[i] for i in range(dim)]
    bounds = [bounds[i] if bounds[i] < strides[i] // 2 else 0 for i in range(dim)]
    inputs = inputs[...,
             bounds[0] // 2: image_shape[0] - (bounds[0] - bounds[0] // 2),
             bounds[1] // 2: image_shape[1] - (bounds[1] - bounds[1] // 2),
             bounds[2] // 2: image_shape[2] - (bounds[2] - bounds[2] // 2)]
    labels = labels[...,
             bounds[0] // 2: image_shape[0] - (bounds[0] - bounds[0] // 2),
             bounds[1] // 2: image_shape[1] - (bounds[1] - bounds[1] // 2),
             bounds[2] // 2: image_shape[2] - (bounds[2] - bounds[2] // 2)]

    inputs, paddings = pad_input(inputs, roi_shape, strides, padding_mode, padding_val)

    padded_shape = inputs.shape[2:]
    size = [(inputs.shape[2:][i] - roi_shape[i]) // strides[i] + 1 for i in range(dim)]
    result = torch.zeros(size=(1, 3, *padded_shape), dtype=inputs.dtype, device=inputs.device)
    norm_map = torch.zeros_like(result)
    if mode == "constant":
        norm_patch = torch.ones(size=roi_shape, dtype=norm_map.dtype, device=norm_map.device)
    elif mode == "gaussian":
        norm_patch = gaussian_kernel(roi_shape[0], 0.125*roi_shape[0]).type(norm_map.dtype).to(norm_map.device)

    else:
        raise ValueError("Unknown mode. Available modes are {constant, gaussian}.")

    for i in range(0, strides[0] * size[0], strides[0]):
        for j in range(0, strides[1] * size[1], strides[1]):
            for k in range(0, strides[2] * size[2], strides[2]):
                result[
                ...,
                i:(roi_shape[0] + i),
                j:(roi_shape[1] + j),
                k:(roi_shape[2] + k)] += model(inputs[
                                               ...,
                                               i:(roi_shape[0] + i),
                                               j:(roi_shape[1] + j),
                                               k:(roi_shape[2] + k)
                                               ]) * norm_patch
                norm_map[
                ...,
                i:(roi_shape[0] + i),
                j:(roi_shape[1] + j),
                k:(roi_shape[2] + k)] += norm_patch


    # account for any overlapping sections
    # norm_map[norm_map == 0] = norm_map[norm_map > 0].min()
    result /= norm_map

    return result[
           ...,
           paddings[4]: image_shape[0] + paddings[4],
           paddings[2]: image_shape[1] + paddings[2],
           paddings[0]: image_shape[2] + paddings[0]
           ], labels
