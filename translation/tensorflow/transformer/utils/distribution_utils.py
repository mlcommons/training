# Copyright 2019 MLBenchmark Group. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Helper functions for distributed setting."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def _mirrored_cross_device_ops(all_reduce_alg, num_packs):
  """Return a CrossDeviceOps based on all_reduce_alg and num_packs.
  Args:
    all_reduce_alg: a string specifying which cross device op to pick, or None.
    num_packs: an integer specifying number of packs for the cross device op.
  Returns:
    tf.distribute.CrossDeviceOps object or None.
  Raises:
    ValueError: if `all_reduce_alg` not in [None, 'nccl', 'hierarchical_copy'].
  """
  if all_reduce_alg is None:
    return None
  mirrored_all_reduce_options = {
      "nccl": tf.distribute.NcclAllReduce,
      "hierarchical_copy": tf.distribute.HierarchicalCopyAllReduce
  }
  if all_reduce_alg not in mirrored_all_reduce_options:
    raise ValueError(
        "When used with `mirrored`, valid values for all_reduce_alg are "
        "['nccl', 'hierarchical_copy'].  Supplied value: {}".format(
            all_reduce_alg))
  cross_device_ops_class = mirrored_all_reduce_options[all_reduce_alg]
  return cross_device_ops_class(num_packs=num_packs)


def get_num_gpus(num_gpus):
  """Validate and return number of GPUs.
  Args:
    num_gpus: The number of GPUs requested
  Returns:
    num_gpus if num_gpus is non-negative int.
    0 or 1, depending on GPU availability, if num_gpus is None.
  Raises:
    ValueError: if num_gpu is not None or non-negative int.
  """
  if num_gpus is not None and (not isinstance(num_gpus, int) or num_gpus < 0):
    raise ValueError("`num_gpus` must be a non-negative int or None.")
  if num_gpus is None:
    if tf.test.is_gpu_available():
      return 1
    return 0
  return num_gpus


def get_distribution_strategy(distribution_strategy=None,
                              num_gpus=0,
                              all_reduce_alg=None,
                              num_packs=1):
  """Return a DistributionStrategy for running the model.
  Args:
    distribution_strategy: Specifies which distribution strategy to use.
      Accepted values are 'one_device', 'mirrored', and 'parameter_server'.
    num_gpus: Number of GPUs to run this model.
    all_reduce_alg: Optional. Specifies which algorithm to use when performing
      all-reduce. For `MirroredStrategy`, valid values are "nccl" and
      "hierarchical_copy".
    num_packs: Optional. Sets the `num_packs` in `tf.distribute.NcclAllReduce`
      or `tf.distribute.HierarchicalCopyAllReduce` for `MirroredStrategy`.
  Returns:
    tf.distribute.DistibutionStrategy object.
  Raises:
    ValueError: if `distribution_strategy` is None or 'one_device' and
      `num_gpus` is larger than 1.
  """
  if num_gpus < 0:
    raise ValueError("`num_gpus` can not be negative.")

  if distribution_strategy is None:
    if num_gpus > 1:
      raise ValueError(
          "When {} GPUs are specified, distribution_strategy "
          "flag cannot be None.".format(num_gpus))
    return None
  
  distribution_strategy = distribution_strategy.lower()

  if distribution_strategy == "one_device":
    if num_gpus == 0:
      return tf.distribute.OneDeviceStrategy("device:CPU:0")
    else:
      if num_gpus > 1:
        raise ValueError("`OneDeviceStrategy` can not be used for more than "
                         "one device.")
      return tf.distribute.OneDeviceStrategy("device:GPU:0")

  if distribution_strategy == "mirrored":
    if num_gpus == 0:
      devices = ["device:CPU:0"]
    else:
      devices = ["device:GPU:%d" % i for i in range(num_gpus)]
    return tf.distribute.MirroredStrategy(
        devices=devices,
        cross_device_ops=_mirrored_cross_device_ops(all_reduce_alg, num_packs))

  if distribution_strategy == "parameter_server":
    return tf.distribute.experimental.ParameterServerStrategy()

  raise ValueError(
      "Unrecognized Distribution Strategy: %r" % distribution_strategy)
