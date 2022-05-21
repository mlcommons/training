import os
import glob

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from data_loading.pytorch_loader import PytVal, PytTrain
from runtime.logging import mllog_event


def list_files_with_pattern(path, files_pattern):
    data = sorted(glob.glob(os.path.join(path, files_pattern)))
    assert len(data) > 0, f"Found no data at {path}"
    return data


def load_data(path, files_pattern):
    data = sorted(glob.glob(os.path.join(path, files_pattern)))
    assert len(data) > 0, f"Found no data at {path}"
    return data


def get_split(data, train_idx, val_idx):
    train = list(np.array(data)[train_idx])
    val = list(np.array(data)[val_idx])
    return train, val


def split_eval_data(x_val, y_val, num_shards, shard_id):
    x = [a.tolist() for a in np.array_split(x_val, num_shards)]
    y = [a.tolist() for a in np.array_split(y_val, num_shards)]
    return x[shard_id], y[shard_id]


def get_data_split(path: str, num_shards: int, shard_id: int):
    with open("evaluation_cases.txt", "r") as f:
        val_cases_list = f.readlines()
    val_cases_list = [case.rstrip("\n") for case in val_cases_list]
    imgs = load_data(path, "*_x.npy")
    lbls = load_data(path, "*_y.npy")
    assert len(imgs) == len(lbls), f"Found {len(imgs)} volumes but {len(lbls)} corresponding masks"
    imgs_train, lbls_train, imgs_val, lbls_val = [], [], [], []
    for (case_img, case_lbl) in zip(imgs, lbls):
        if case_img.split("_")[-2] in val_cases_list:
            imgs_val.append(case_img)
            lbls_val.append(case_lbl)
        else:
            imgs_train.append(case_img)
            lbls_train.append(case_lbl)
    mllog_event(key='train_samples', value=len(imgs_train), sync=False)
    mllog_event(key='eval_samples', value=len(imgs_val), sync=False)
    imgs_val, lbls_val = split_eval_data(imgs_val, lbls_val, num_shards, shard_id)
    return imgs_train, imgs_val, lbls_train, lbls_val


class SyntheticDataset(Dataset):
    def __init__(self, channels_in=1, channels_out=3, shape=(128, 128, 128),
                 device="cpu", layout="NCDHW", scalar=False):
        shape = tuple(shape)
        x_shape = (channels_in,) + shape if layout == "NCDHW" else shape + (channels_in,)
        self.x = torch.rand((32, *x_shape), dtype=torch.float32, device=device, requires_grad=False)
        if scalar:
            self.y = torch.randint(low=0, high=channels_out - 1, size=(32, *shape), dtype=torch.int32,
                                   device=device, requires_grad=False)
            self.y = torch.unsqueeze(self.y, dim=1 if layout == "NCDHW" else -1)
        else:
            y_shape = (channels_out,) + shape if layout == "NCDHW" else shape + (channels_out,)
            self.y = torch.rand((32, *y_shape), dtype=torch.float32, device=device, requires_grad=False)

    def __len__(self):
        return 64

    def __getitem__(self, idx):
        return self.x[idx % 32], self.y[idx % 32]


def get_data_loaders(flags, num_shards, global_rank):
    if flags.loader == "synthetic":
        train_dataset = SyntheticDataset(scalar=True, shape=flags.input_shape, layout=flags.layout)
        val_dataset = SyntheticDataset(scalar=True, shape=flags.val_input_shape, layout=flags.layout)

    elif flags.loader == "pytorch":
        x_train, x_val, y_train, y_val = get_data_split(flags.data_dir, num_shards, shard_id=global_rank)
        train_data_kwargs = {"patch_size": flags.input_shape, "oversampling": flags.oversampling, "seed": flags.seed}
        train_dataset = PytTrain(x_train, y_train, **train_data_kwargs)
        val_dataset = PytVal(x_val, y_val)
    else:
        raise ValueError(f"Loader {flags.loader} unknown. Valid loaders are: synthetic, pytorch")

    train_sampler = DistributedSampler(train_dataset, seed=flags.seed, drop_last=True) if num_shards > 1 else None
    val_sampler = None

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=flags.batch_size,
                                  shuffle=not flags.benchmark and train_sampler is None,
                                  sampler=train_sampler,
                                  num_workers=flags.num_workers,
                                  pin_memory=True,
                                  drop_last=True)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=not flags.benchmark and val_sampler is None,
                                sampler=val_sampler,
                                num_workers=flags.num_workers,
                                pin_memory=True,
                                drop_last=False)

    return train_dataloader, val_dataloader
