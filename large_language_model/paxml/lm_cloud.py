# coding=utf-8
# Copyright 2022 Google LLC.
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

"""Decoder-only language model configurations."""

from typing import List

import jax
from jax import numpy as jnp
from paxml import base_experiment
from paxml import base_task
from paxml import experiment_registry
from paxml.tasks.lm import input_generator
from paxml.tasks.lm import model_params
from praxis import base_input
from praxis import layers


class SyntheticDataset(base_experiment.BaseExperiment):
  """Synthetic LM dataset."""
  PERCORE_BATCH_SIZE = 16
  MAX_SEQ_LEN = 1024

  def _dataset_common(self, is_training) -> base_input.BaseInput.HParams:
    num_local_devices = jax.local_device_count()
    batch_size = self.PERCORE_BATCH_SIZE * num_local_devices
    input_p = input_generator.SyntheticLmData.HParams()
    if is_training:
      input_p.batch_size = batch_size
    else:
      # TODO(zhangqiaorjc): Is this batch size too big for test?
      input_p.batch_size = batch_size
    input_p.seq_len = self.MAX_SEQ_LEN
    p = base_input.LingvoInputAdaptor.HParams(
        input=input_p, is_training=is_training)
    return p

  def datasets(self) -> List[base_input.BaseInput.HParams]:
    """Returns a list of dataset parameters."""
    return [
        self._dataset_common(is_training=True),
        self._dataset_common(is_training=False)
    ]

