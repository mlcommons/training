# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

""" Strategies using Zarr as an underlying format. """

from functools import partial
from pathlib import Path
from typing import List

import numpy as np
import torch
import zarr

from ..core import CheckpointingException
from ..mapping import ShardedTensor, ShardedStateDict
from ..dict_utils import dict_list_map_inplace
from .base import default_strategies, StrategyAction, LoadShardedStrategy, \
    SaveShardedStrategy

numpy_to_torch_dtype_dict = {
    np.bool_      : torch.bool,
    np.uint8      : torch.uint8,
    np.int8       : torch.int8,
    np.int16      : torch.int16,
    np.int32      : torch.int32,
    np.int64      : torch.int64,
    np.float16    : torch.float16,
    np.float32    : torch.float32,
    np.float64    : torch.float64,
    np.complex64  : torch.complex64,
    np.complex128 : torch.complex128
}

torch_to_numpy_dtype_dict = {v: k for k, v in numpy_to_torch_dtype_dict.items()}



try:
    import tensorstore
    HAS_BFLOAT16 = True
    numpy_to_torch_dtype_dict[np.dtype('bfloat16')] = torch.bfloat16
    torch_to_numpy_dtype_dict[torch.bfloat16] = np.dtype('bfloat16')
except ImportError:
    HAS_BFLOAT16 = False

_import_trigger = None


class ZarrSaveShardedStrategy(SaveShardedStrategy):
    def save(self, sharded_tensors: List[ShardedTensor], checkpoint_dir: Path):
        arrays = _create_or_open_zarr_arrays(sharded_tensors, checkpoint_dir)
        for ten, arr in zip(sharded_tensors, arrays):
            _save_to_existing_array(ten, arr)
        torch.distributed.barrier()


def _create_or_open_zarr_arrays(sharded_tensors: List[ShardedTensor], checkpoint_dir: Path) -> List[zarr.Array]:
    arrays = []
    for ten in sharded_tensors:
        if ten.replica_id == 0 and set(ten.global_offset) == {0}:
            _create_zarr_array(ten, checkpoint_dir)
            # TODO: maybe reuse the opened arrays

    torch.distributed.barrier()
    for ten in sharded_tensors:
        # if ten.replica_id == 0 and set(ten.global_offset) == {0}:
        #     continue
        arr = zarr.open(checkpoint_dir / ten.key, 'r+')
        arrays.append(arr)
    return arrays


def _save_to_existing_array(sharded_tensor: ShardedTensor, arr: zarr.Array):
    if sharded_tensor.replica_id > 0:
        return
    x = sharded_tensor.data
    x = x.detach().cpu()
    torch.cuda.synchronize()
    if x.dtype == torch.bfloat16:
        x = x.float()
        x = x.numpy()
        x = x.astype('bfloat16')
    else:
        x = x.numpy()
    arr[sharded_tensor.global_slice()] = x

def _create_zarr_array(sharded_tensor: ShardedTensor, checkpoint_dir: Path):
    # TODO: check for array existence?
    np_dtype = torch_to_numpy_dtype_dict[sharded_tensor.dtype]
    arr = zarr.create(sharded_tensor.global_shape, dtype=np_dtype,
                      store=checkpoint_dir / sharded_tensor.key, chunks=sharded_tensor.max_allowed_chunks(),
                      compressor=None, fill_value=None, write_empty_chunks=False)
    if HAS_BFLOAT16 and np_dtype == np.dtype('bfloat16'):
        arr._dtype = np_dtype
        zarray = arr.store['.zarray']
        arr.store['.zarray'] = zarray.replace(b'<V2', b'bfloat16')
    return arr


class ZarrLoadShardedStrategy(LoadShardedStrategy):
    def load(self, sharded_state_dict: ShardedStateDict, checkpoint_dir: Path):
        dict_list_map_inplace(partial(_load_from_array, checkpoint_dir=checkpoint_dir), sharded_state_dict)
        torch.distributed.barrier()
        return sharded_state_dict

    def check_backend_compatibility(self, loaded_version):
        pass  # TODO

    def check_version_compatibility(self, loaded_version):
        pass  # TODO


def _load_from_array(sharded_tensor: ShardedTensor, checkpoint_dir: Path):
    assert isinstance(sharded_tensor, ShardedTensor), type(sharded_tensor)
    try:
        arr = zarr.open(checkpoint_dir / sharded_tensor.key, 'r')
    except zarr.errors.PathNotFoundError as e:
        raise CheckpointingException(f'Array {checkpoint_dir / sharded_tensor.key} not found') from e

    if (not sharded_tensor.allow_shape_mismatch
        and sharded_tensor.global_shape != arr.shape):
            _msg = f'Global shape mismatch for loaded ({arr.shape})' \
                   f' and expected ({sharded_tensor.global_shape}) tensor' \
                   f' for key {sharded_tensor.key}'
            raise CheckpointingException(_msg)

    x = arr[sharded_tensor.global_slice()]
    if HAS_BFLOAT16 and x.dtype == np.dtype('bfloat16'):
        x = x.astype(np.dtype('float32'))
        x = torch.from_numpy(x)
        x = x.bfloat16()
    else:
        x = torch.from_numpy(x)
    # TODO: consider some other consistency checks
    if x.shape != sharded_tensor.local_shape:
        if sharded_tensor.allow_shape_mismatch:
            x = pad_to_expected_shape(x, sharded_tensor)
        else:
            _msg = f'Local shape mismatch for loaded ({x.shape})' \
                   f' and expected ({sharded_tensor.local_shape}) tensor' \
                   f' for key {sharded_tensor.key}'
            raise CheckpointingException(_msg)

    # TODO: consider cuda() tensors support
    return x


def pad_to_expected_shape(x: torch.Tensor, expected_sharded_ten: ShardedTensor):
    pad_args = []
    assert len(x.shape) == len(expected_sharded_ten.local_shape)
    # Reversed iteration order because F.pad expects so
    for x_sh, exp_sh, axis_fragm in reversed(list(zip(x.shape, expected_sharded_ten.local_shape,
                                                      expected_sharded_ten.axis_fragmentations))):
        if x_sh == exp_sh:
            pad_args.extend((0, 0))
        elif x_sh > exp_sh:
            assert 'Expected shape smaller than actual'
        else:
            assert axis_fragm > 1, 'Encountered shape mismatch on non-fragmented axis'
            pad_args.extend((0, exp_sh - x_sh))
    return torch.nn.functional.pad(x, pad_args)


default_strategies[StrategyAction.LOAD_SHARDED.value][('zarr', 1)] = ZarrLoadShardedStrategy()
default_strategies[StrategyAction.SAVE_SHARDED.value][('zarr', 1)] = ZarrSaveShardedStrategy('zarr', 1)
