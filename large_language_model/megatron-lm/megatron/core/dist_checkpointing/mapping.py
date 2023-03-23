# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

""" Core library classes. """

from itertools import chain

import numpy as np
from dataclasses import dataclass, replace
from typing import Dict, Any, Optional, Tuple, Union

import torch

from .core import CheckpointingException

# These type definitions are just hints to differentiate a plain model state
#  dict (StateDict) from a state dict with tensors replaced with ShardedTensors
#  (ShardedStateDict).
StateDict = Dict[str, Any]
ShardedStateDict = Dict[str, Any]


@dataclass
class ShardedTensor:
    """Represents a mapping between a local tensor and a global tensor.

    Global tensor is assumed to consist of many local tensors distributed
    between different processes.

    Attributes:
        key: unique identifier of a global tensor
        data: local tensor data. Can be None only for consistency validation
        dtype: tensor dtype
        local_shape: local tensor shape
        global_shape: global tensor shape
        global_offset: offset of a local tensor in a global tensor, specified
            in number of tensor elements
        axis_fragmentations: global tensor fragmentation of each axis
        replica_id: indicates given local tensor's replication wrt. local
            tensors in different processes
        prepend_axis_num: number of axes prepended to the local tensor
            to reflect global tensor shape.
            The behavior is similar to unsqueezing the local tensor.
        allow_shape_mismatch: if True, during loading, the global shape of a
            stored tensor does not have to match the expected global shape.
            Useful for representing tensors with flexible shape, e.g. padded.
    """
    key: str
    data: Optional[torch.Tensor]
    dtype: torch.dtype
    local_shape: Tuple[int, ...]
    global_shape: Tuple[int, ...]
    global_offset: Tuple[int, ...]
    axis_fragmentations: Optional[Tuple[int, ...]]
    replica_id: int = 0
    prepend_axis_num: int = 0
    allow_shape_mismatch: bool = False

    def global_slice(self) -> Tuple[Union[int, slice], ...]:
        assert len(self.global_offset) == len(self.local_shape) + self.prepend_axis_num
        return tuple(chain(
            (off for off in self.global_offset[:self.prepend_axis_num]),
            (slice(off, off + sh) for off, sh in zip(self.global_offset[self.prepend_axis_num:], self.local_shape))
        ))

    def max_allowed_chunks(self) -> Tuple[int, ...]:
        chunks = []
        for axis_sh, axis_fragm in zip(self.global_shape, self.axis_fragmentations):
            if not self.allow_shape_mismatch and axis_sh % axis_fragm != 0:
                raise CheckpointingException(f'Axis shape ({axis_sh}) not divisible'
                                             f' by axis fragmentation ({axis_fragm}')
            axis_chunk_size = axis_sh // axis_fragm
            chunks.append(axis_chunk_size)
        return tuple(chunks)

    def without_data(self):
        return replace(self, data=None)

    @classmethod
    def from_rank_offsets(cls, key: str, data: torch.Tensor, *rank_offsets: Tuple[int, int, int],
                          replica_id: int = 0, prepend_axis_num: int = 0, allow_shape_mismatch: bool = False):
        """Allows to construct the ShardedTensor given offset specified in process ranks.
        Arguments:
            key: unique key
            data: local tensor data
            rank_offsets: each tuple (axis, axis_rank_offset, axis_fragm)
                says that if global tensor is divided into `axis_fragm`
                 fragment along `axis` axis, then local tensor data
                 corresponds to the `axis_rank_offset` chunk.
            replica_id: see ShardedTensor
            prepend_axis_num: see ShardedTensor
            allow_shape_mismatch: see ShardedTensor
        """
        global_offset = [0] * (data.ndim + prepend_axis_num)
        global_shape = ([1] * prepend_axis_num) + list(data.shape)
        axis_fragmentations = [1] * (data.ndim + prepend_axis_num)
        _seen_axis = set()
        for axis, axis_rank_offset, axis_fragm in rank_offsets:
            assert axis >= 0 and axis_rank_offset >= 0 and axis_fragm >= 0, (axis, axis_rank_offset, axis_fragm)
            assert axis_rank_offset < axis_fragm, 'Rank offset must be lower than axis fragmentation'
            if axis in _seen_axis:
                raise CheckpointingException('Duplicated axis specified')
            _seen_axis.add(axis)

            local_axis_shape = 1 if axis < prepend_axis_num else data.shape[axis - prepend_axis_num]
            global_shape[axis] = axis_fragm * local_axis_shape
            global_offset[axis] = axis_rank_offset * local_axis_shape
            axis_fragmentations[axis] = axis_fragm

        return cls(key, data, data.dtype, tuple(data.shape),
                   tuple(global_shape), tuple(global_offset), tuple(axis_fragmentations),
                   replica_id, prepend_axis_num, allow_shape_mismatch)


class LocalNonpersitentObject:
    """Object that should not be stored in a checkpoint, but restored locally.

    Wrapping any object inside the state dict with LocalNonpersitentObject
    will result in:
    - during saving, this object will *not* be stored in the checkpoint
    - during loading, a local version of this object will be placed in a state dict
    """
    def __init__(self, obj):
        self.obj = obj

    def unwrap(self):
        return self.obj

